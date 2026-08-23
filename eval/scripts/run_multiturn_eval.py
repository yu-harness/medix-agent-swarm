#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""多轮指代评测：同 session 两轮 swarm，LLM 判第2轮是否关联 dependency。"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

JUDGE_PROMPT = """你是医疗多轮对话评测裁判。判断「第2轮系统回答」是否正确关联了第1轮的关键依赖实体。

判定：
- correct：明显围绕 dependency / 第1轮话题作答，满足 ok_criteria
- incorrect：忽略指代、答成无关话题、或未体现 dependency

只输出一行 JSON：
{{"verdict":"correct|incorrect","reason":"不超过40字"}}

dependency：{dependency}
ok_criteria：{ok_criteria}
turn1：{turn1}
turn2：{turn2}
第2轮回答：
{pred}
"""


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_done_ids(detail_path: Path) -> Set[str]:
    done: Set[str] = set()
    if not detail_path.exists():
        return done
    with detail_path.open("r", encoding="utf-8") as f:
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


async def llm_judge(client, item: Dict[str, Any], pred: str) -> tuple:
    prompt = JUDGE_PROMPT.format(
        dependency=item.get("dependency", "")[:200],
        ok_criteria=item.get("ok_criteria", "")[:300],
        turn1=(item.get("turn1") or "")[:400],
        turn2=(item.get("turn2") or "")[:400],
        pred=(pred or "")[:2000],
    )
    try:
        text = await client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=120,
        )
        text = (text or "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text[start : end + 1])
            v = str(obj.get("verdict", "incorrect")).lower().strip()
            if v not in ("correct", "incorrect"):
                v = "incorrect"
            return v, str(obj.get("reason", ""))[:80]
    except Exception as e:
        return "incorrect", f"judge_error:{type(e).__name__}"
    return "incorrect", "judge_parse_fail"


async def run_one(item, coordinator, llm_client, timeout, sem) -> Dict[str, Any]:
    sid = f"mt-eval-{item['id']}-{uuid.uuid4().hex[:8]}"
    t1 = item["turn1"]
    t2 = item["turn2"]
    a1 = a2 = ""
    err = None
    lat1 = lat2 = None

    async with sem:
        try:
            t0 = time.perf_counter()
            r1 = await asyncio.wait_for(
                coordinator.process(t1, session_id=sid), timeout=timeout
            )
            lat1 = round(time.perf_counter() - t0, 3)
            a1 = extract_answer(r1)
            if not a1:
                err = "empty_turn1"
            t0 = time.perf_counter()
            r2 = await asyncio.wait_for(
                coordinator.process(t2, session_id=sid), timeout=timeout
            )
            lat2 = round(time.perf_counter() - t0, 3)
            a2 = extract_answer(r2)
            if not a2 and not err:
                err = "empty_turn2"
        except asyncio.TimeoutError:
            err = "timeout"
        except Exception as e:
            err = f"{type(e).__name__}:{e}"

    verdict = "incorrect"
    reason = err or ""
    if a2 and not err:
        verdict, reason = await llm_judge(llm_client, item, a2)
    elif a2 and err in ("empty_turn1",):
        # still judge turn2 if somehow present
        verdict, reason = await llm_judge(llm_client, item, a2)

    return {
        "id": item["id"],
        "category": item.get("category"),
        "turn1": t1,
        "turn2": t2,
        "dependency": item.get("dependency"),
        "ok_criteria": item.get("ok_criteria"),
        "answer1": (a1 or "")[:2000],
        "answer2": (a2 or "")[:3000],
        "session_id": sid,
        "latency_turn1": lat1,
        "latency_turn2": lat2,
        "verdict": verdict,
        "coref_correct": verdict == "correct",
        "judge_reason": reason,
        "error": err,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def summarize(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(details)
    ok = sum(1 for d in details if d.get("coref_correct"))
    by_cat = {}
    for cat in sorted({d.get("category") or "?" for d in details}):
        rows = [d for d in details if d.get("category") == cat]
        c = sum(1 for d in rows if d.get("coref_correct"))
        by_cat[cat] = {
            "n": len(rows),
            "correct": c,
            "acc": round(c / len(rows), 4) if rows else 0,
        }
    return {
        "n": n,
        "coref_correct": ok,
        "coref_accuracy": round(ok / n, 4) if n else 0,
        "by_category": by_cat,
        "verdict_counts": dict(Counter(d.get("verdict") for d in details)),
        "error_counts": dict(Counter(d.get("error") for d in details if d.get("error"))),
    }


def write_md(path: Path, meta: Dict[str, Any], summary: Dict[str, Any]) -> None:
    lines = [
        "# 多轮指代评测报告",
        "",
        "> 协议：同一 session 连续两次 `process_with_swarm`；LLM 判第2轮是否关联 dependency。",
        "",
        f"- n: {summary['n']}",
        f"- **指代正确率**: **{summary['coref_accuracy']:.1%}** ({summary['coref_correct']}/{summary['n']})",
        f"- concurrency: {meta.get('concurrency')}",
        f"- timeout_sec: {meta.get('timeout_sec')}",
        f"- elapsed_sec: {meta.get('elapsed_sec')}",
        f"- data: `{meta.get('data')}`",
        "",
        "## 分品类",
        "",
        "| category | n | coref_acc |",
        "|---|---:|---:|",
    ]
    for cat, s in (summary.get("by_category") or {}).items():
        lines.append(f"| {cat} | {s['n']} | {s['acc']:.1%} ({s['correct']}/{s['n']}) |")
    lines += [
        "",
        f"- verdict_counts: `{summary.get('verdict_counts')}`",
        f"- error_counts: `{summary.get('error_counts')}`",
        "",
        "## 局限",
        "",
        "- 自建 50 条两轮指代集，非公开多轮金标",
        "- 裁判为自动 LLM，非人工盲评",
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
    detail_path = out_dir / "multiturn_eval_detail.jsonl"
    if args.resume_detail:
        detail_path = Path(args.resume_detail)

    rows = load_jsonl(data_path)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    done = load_done_ids(detail_path) if args.resume or args.resume_detail else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"[multiturn] n={len(rows)} done={len(done)} todo={len(todo)}", flush=True)

    from core.llm_client import LLMClient
    from swarm import SwarmCoordinator

    coordinator = SwarmCoordinator(enable_swarm=True)
    llm_client = LLMClient()
    sem = asyncio.Semaphore(args.concurrency)
    meta = {
        "data": str(data_path),
        "n_target": len(rows),
        "concurrency": args.concurrency,
        "timeout_sec": args.timeout,
        "path": "process_with_swarm / SwarmCoordinator.process 同 session 两轮",
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
                "coref_correct": False,
                "verdict": "incorrect",
                "error": traceback.format_exc()[-400:],
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        async with lock:
            append_jsonl(detail_path, rec)
            finished += 1
            print(
                f"[{len(done)+finished}/{len(rows)}] {rec['id']} "
                f"coref={'Y' if rec.get('coref_correct') else 'N'} "
                f"err={rec.get('error')}",
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
    want = {r["id"] for r in rows}
    latest = {}
    for d in all_d:
        if d.get("id") in want:
            latest[d["id"]] = d
    details = list(latest.values())
    summary = summarize(details)
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    meta["elapsed_sec"] = round(time.perf_counter() - t_all, 1)
    (out_dir / "multiturn_eval.json").write_text(
        json.dumps({"meta": meta, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_md(out_dir / "multiturn_eval.md", meta, summary)
    print(
        f"[done] coref_acc={summary['coref_accuracy']:.1%} n={summary['n']}",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "eval" / "data" / "multiturn_50.jsonl"))
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-detail", default="")
    args = p.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
