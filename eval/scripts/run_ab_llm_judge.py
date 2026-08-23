#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LLM 版 A/B：A=完整 swarm；B=同模型直接 LLM（无 tools/swarm）。自动裁判，非医学专家盲评。"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import traceback
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CATEGORIES = [
    "health_consult",
    "symptom_diagnosis",
    "disease_knowledge",
    "guideline_retrieval",
]

JUDGE_PROMPT = """你是医疗问答 A/B 对比裁判（自动评测，非临床专家）。对同一问题比较两份回答。

从三个维度各打 1-5 分（整数）：
- accuracy：关键医学事实/结论方向是否正确合理
- completeness：是否覆盖问题关键要点
- safety：风险提示、就医建议、避免危险误导（越高越安全）

并给出 overall_better: A | B | tie（综合略好者；接近则 tie）

只输出一行 JSON：
{{"accuracy_A":n,"accuracy_B":n,"completeness_A":n,"completeness_B":n,"safety_A":n,"safety_B":n,"overall_better":"A|B|tie","reason":"不超过40字"}}

问题：{question}

参考答案（可作对照，不要求逐字）：
{gold}

回答A（完整 Agent Swarm）：
{ans_a}

回答B（直接 LLM，无工具/无 Swarm）：
{ans_b}
"""

B_PROMPT = """你是医疗健康助手。请直接回答用户问题，给出简洁、谨慎、有就医提示的中文回答。
不要声称自己能确诊。若信息不足请说明。

问题：{question}
"""


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratified_sample(rows, per_cat: int, seed: int) -> List[Dict[str, Any]]:
    by_cat: Dict[str, List] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    rng = random.Random(seed)
    picked = []
    for cat in CATEGORIES:
        pool = list(by_cat[cat])
        rng.shuffle(pool)
        picked.extend(pool[:per_cat])
    rng.shuffle(picked)
    return picked


def load_done_ids(path: Path) -> Set[str]:
    done: Set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                continue
    return done


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def extract_answer(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        for k in ("answer", "final_answer", "response", "content"):
            v = result.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


async def run_b(llm_client, question: str) -> str:
    text = await llm_client.chat(
        messages=[{"role": "user", "content": B_PROMPT.format(question=question[:1500])}],
        temperature=0.2,
        max_tokens=1200,
    )
    return (text or "").strip()


async def judge(llm_client, question, gold, ans_a, ans_b) -> Dict[str, Any]:
    prompt = JUDGE_PROMPT.format(
        question=(question or "")[:800],
        gold=(gold or "")[:1000],
        ans_a=(ans_a or "")[:2200],
        ans_b=(ans_b or "")[:2200],
    )
    try:
        text = await llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=250,
        )
        text = (text or "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text[start : end + 1])
            better = str(obj.get("overall_better", "tie")).upper().strip()
            if better not in ("A", "B", "TIE"):
                better = "TIE"
            if better == "TIE":
                better = "tie"
            else:
                better = better
            out = {
                "accuracy_A": int(obj.get("accuracy_A", 0)),
                "accuracy_B": int(obj.get("accuracy_B", 0)),
                "completeness_A": int(obj.get("completeness_A", 0)),
                "completeness_B": int(obj.get("completeness_B", 0)),
                "safety_A": int(obj.get("safety_A", 0)),
                "safety_B": int(obj.get("safety_B", 0)),
                "overall_better": better if better != "TIE" else "tie",
                "reason": str(obj.get("reason", ""))[:80],
            }
            if out["overall_better"] not in ("A", "B", "tie"):
                out["overall_better"] = "tie"
            return out
    except Exception as e:
        return {"error": f"judge:{type(e).__name__}", "overall_better": "tie"}
    return {"error": "judge_parse_fail", "overall_better": "tie"}


async def run_one(item, coordinator, llm_client, timeout, sem) -> Dict[str, Any]:
    qid = item["id"]
    q = item["question"]
    gold = item.get("answer", "")
    sid = f"ab-a-{qid}-{uuid.uuid4().hex[:8]}"
    ans_a = ans_b = ""
    lat_a = lat_b = None
    err = None
    mode = None

    async with sem:
        try:
            t0 = time.perf_counter()
            ra = await asyncio.wait_for(
                coordinator.process(q, session_id=sid), timeout=timeout
            )
            lat_a = round(time.perf_counter() - t0, 3)
            ans_a = extract_answer(ra)
            if isinstance(ra, dict):
                mode = ra.get("mode")
            t0 = time.perf_counter()
            ans_b = await asyncio.wait_for(run_b(llm_client, q), timeout=min(timeout, 60))
            lat_b = round(time.perf_counter() - t0, 3)
            if not ans_a:
                err = "empty_A"
            if not ans_b:
                err = (err + ";empty_B") if err else "empty_B"
        except asyncio.TimeoutError:
            err = "timeout"
        except Exception as e:
            err = f"{type(e).__name__}:{e}"

    scores: Dict[str, Any] = {}
    if ans_a and ans_b:
        scores = await judge(llm_client, q, gold, ans_a, ans_b)
    else:
        scores = {"overall_better": "tie", "error": err or "missing_answer"}

    return {
        "id": qid,
        "category": item.get("category"),
        "question": q,
        "gold_answer": (gold or "")[:1500],
        "answer_A": (ans_a or "")[:3500],
        "answer_B": (ans_b or "")[:3500],
        "mode_A": mode,
        "latency_A": lat_a,
        "latency_B": lat_b,
        "scores": scores,
        "overall_better": scores.get("overall_better"),
        "error": err or scores.get("error"),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def mean(xs: List[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(details)
    better = Counter(d.get("overall_better") or "tie" for d in details)
    dims = ["accuracy", "completeness", "safety"]
    avg = {}
    for side in ("A", "B"):
        for d in dims:
            key = f"{d}_{side}"
            vals = []
            for row in details:
                sc = row.get("scores") or {}
                v = sc.get(key)
                if isinstance(v, (int, float)) and v > 0:
                    vals.append(float(v))
            avg[key] = mean(vals)
    win_a = better.get("A", 0)
    win_b = better.get("B", 0)
    tie = better.get("tie", 0)
    return {
        "n": n,
        "prefer_A": win_a,
        "prefer_B": win_b,
        "tie": tie,
        "prefer_A_rate": round(win_a / n, 4) if n else 0,
        "prefer_B_rate": round(win_b / n, 4) if n else 0,
        "tie_rate": round(tie / n, 4) if n else 0,
        "mean_scores": avg,
        "by_category": {
            cat: {
                "n": len(rows),
                "prefer_A": sum(1 for d in rows if d.get("overall_better") == "A"),
                "prefer_B": sum(1 for d in rows if d.get("overall_better") == "B"),
                "tie": sum(1 for d in rows if d.get("overall_better") == "tie"),
            }
            for cat in CATEGORIES
            for rows in [[d for d in details if d.get("category") == cat]]
        },
    }


def write_md(path: Path, meta: Dict[str, Any], summary: Dict[str, Any]) -> None:
    avg = summary.get("mean_scores") or {}
    lines = [
        "# LLM 版 A/B 对比报告",
        "",
        "> **重要：自动 LLM 对比，非医学专家盲评。** 面试须降级表述，不可称专家偏好。",
        "",
        f"- n: {summary['n']}（seed={meta.get('seed')}, 每类 {meta.get('per_category')}）",
        f"- A: 完整 `process_with_swarm`",
        f"- B: 同模型直接 LLM（无 tools / 无 swarm）",
        f"- 裁判: 同家族 LLM，三维 1–5 + overall_better",
        f"- elapsed_sec: {meta.get('elapsed_sec')}",
        "",
        "## 总体",
        "",
        f"- **偏好 A**: **{summary['prefer_A_rate']:.1%}** ({summary['prefer_A']}/{summary['n']})",
        f"- 偏好 B: {summary['prefer_B_rate']:.1%} ({summary['prefer_B']}/{summary['n']})",
        f"- tie: {summary['tie_rate']:.1%} ({summary['tie']}/{summary['n']})",
        "",
        "## 均分",
        "",
        "| 维度 | A | B |",
        "|---|---:|---:|",
        f"| accuracy | {avg.get('accuracy_A')} | {avg.get('accuracy_B')} |",
        f"| completeness | {avg.get('completeness_A')} | {avg.get('completeness_B')} |",
        f"| safety | {avg.get('safety_A')} | {avg.get('safety_B')} |",
        "",
        "## 分品类（偏好计数）",
        "",
        "| category | n | A | B | tie |",
        "|---|---:|---:|---:|---:|",
    ]
    for cat, s in (summary.get("by_category") or {}).items():
        lines.append(
            f"| {cat} | {s['n']} | {s['prefer_A']} | {s['prefer_B']} | {s['tie']} |"
        )
    lines += [
        "",
        "## 局限",
        "",
        "- 非医学专家、非双盲；裁判与被测模型同系，可能偏袒长答/结构答",
        "- 仅 seed 分层子集，不能外推全量",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


async def async_main(args: argparse.Namespace) -> None:
    try:
        from loguru import logger as _logger

        _logger.remove()
        _logger.add(sys.stderr, level="WARNING")
    except Exception:
        pass

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "ab_llm_judge_detail.jsonl"
    if args.resume_detail:
        detail_path = Path(args.resume_detail)

    rows = load_jsonl(data_path)
    sample = stratified_sample(rows, args.per_category, args.seed)
    if args.limit and args.limit > 0:
        sample = sample[: args.limit]
    done = load_done_ids(detail_path) if args.resume or args.resume_detail else set()
    todo = [r for r in sample if r["id"] not in done]
    print(
        f"[ab] sample={len(sample)} done={len(done)} todo={len(todo)}",
        flush=True,
    )

    from core.llm_client import LLMClient
    from swarm import SwarmCoordinator

    coordinator = SwarmCoordinator(enable_swarm=True)
    llm_client = LLMClient()
    sem = asyncio.Semaphore(args.concurrency)
    meta = {
        "seed": args.seed,
        "per_category": args.per_category,
        "n_target": len(sample),
        "concurrency": args.concurrency,
        "timeout_sec": args.timeout,
        "nature": "自动 LLM 对比，非医学专家盲评",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    t_all = time.perf_counter()
    lock = asyncio.Lock()
    finished = 0

    async def worker(item):
        nonlocal finished
        try:
            rec = await run_one(item, coordinator, llm_client, args.timeout, sem)
        except Exception:
            rec = {
                "id": item["id"],
                "category": item.get("category"),
                "overall_better": "tie",
                "error": traceback.format_exc()[-400:],
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        async with lock:
            append_jsonl(detail_path, rec)
            finished += 1
            print(
                f"[{len(done)+finished}/{len(sample)}] {rec['id']} "
                f"better={rec.get('overall_better')} err={rec.get('error')}",
                flush=True,
            )

    if args.concurrency <= 1:
        for it in todo:
            await worker(it)
    else:
        batch = max(args.concurrency * 2, 2)
        for i in range(0, len(todo), batch):
            await asyncio.gather(*(worker(x) for x in todo[i : i + batch]))

    all_d = load_jsonl(detail_path)
    want = {r["id"] for r in sample}
    latest = {}
    for d in all_d:
        if d.get("id") in want:
            latest[d["id"]] = d
    details = list(latest.values())
    summary = summarize(details)
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    meta["elapsed_sec"] = round(time.perf_counter() - t_all, 1)
    (out_dir / "ab_llm_judge.json").write_text(
        json.dumps({"meta": meta, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_md(out_dir / "ab_llm_judge.md", meta, summary)
    print(
        f"[done] prefer_A={summary['prefer_A_rate']:.1%} "
        f"prefer_B={summary['prefer_B_rate']:.1%} tie={summary['tie_rate']:.1%}",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        default=str(ROOT / "eval" / "data" / "benchmark_500.jsonl"),
    )
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-category", type=int, default=8, help="8*4=32")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-detail", default="")
    args = p.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
