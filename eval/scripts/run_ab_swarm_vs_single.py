#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A/B 评测：Swarm（多 Agent 协作） vs Single Agent（单 Agent），LLM 盲评裁判。

与 run_ab_llm_judge.py 的区别：
- run_ab_llm_judge.py 对比 "Swarm vs 裸 LLM（无工具）"，证明的是 RAG + 工具的价值
- 本脚本对比 "Swarm vs 单 Agent"，两者都走同一套知识与工具，
  差异只有"是否多 Agent 协作"——证明的是 Swarm 本身的增量价值

盲评设计：每题随机决定把 Swarm 答案呈现为 "回答A" 还是 "回答B"，
裁判不知道哪份是 Swarm，避免立场偏差。结果按映射还原。

用法：
    python eval/scripts/run_ab_swarm_vs_single.py --per-category 6 --concurrency 2
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

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

回答A：
{ans_a}

回答B：
{ans_b}
"""


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def stratified_sample(rows: List[Dict], per_cat: int, seed: int) -> List[Dict]:
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
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                continue
    return done


def extract_answer(result: Any) -> str:
    if isinstance(result, dict):
        return (result.get("answer") or "").strip()
    return str(result or "").strip()


def _extract_json(text: str) -> Dict[str, Any]:
    """从模型输出里抠出第一个 JSON 对象。"""
    if not text:
        return {}
    # 优先直接解析
    try:
        return json.loads(text)
    except Exception:
        pass
    # 去掉 ```json 围栏
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # 抓第一个 {...}
    m = re.search(r"\{[^{}]*\}", cleaned, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 3)


async def run_arm(coordinator, question: str, session_id: str, timeout: float):
    """跑一个臂，返回 (answer, latency, meta)"""
    t0 = time.perf_counter()
    try:
        res = await asyncio.wait_for(
            coordinator.process(question, session_id=session_id), timeout=timeout
        )
    except asyncio.TimeoutError:
        return "", round(time.perf_counter() - t0, 3), {"error": "timeout"}
    except Exception as e:
        return "", round(time.perf_counter() - t0, 3), {"error": f"{type(e).__name__}: {e}"}

    lat = round(time.perf_counter() - t0, 3)
    meta: Dict[str, Any] = {}
    if isinstance(res, dict):
        meta = {
            "used_swarm": "swarm_metadata" in res,
            "timeout_occurred": bool(res.get("timeout_occurred")),
            "agents_involved": res.get("agents_involved") or [],
            "total_time": res.get("total_time"),
        }
    return extract_answer(res), lat, meta


async def judge(llm_client, question: str, gold: str, ans_a: str, ans_b: str) -> Dict[str, Any]:
    """盲评：ans_a / ans_b 是随机呈现顺序，调用方负责还原。"""
    prompt = JUDGE_PROMPT.format(
        question=question[:1500],
        gold=(gold or "")[:800],
        ans_a=(ans_a or "")[:2000],
        ans_b=(ans_b or "")[:2000],
    )
    try:
        raw = await llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
        )
    except Exception as e:
        return {"error": f"judge:{type(e).__name__}", "overall_better": "tie"}

    data = _extract_json(raw or "")
    out: Dict[str, Any] = {}
    for k in (
        "accuracy_A", "accuracy_B", "completeness_A", "completeness_B",
        "safety_A", "safety_B",
    ):
        v = data.get(k)
        try:
            out[k] = max(1, min(5, int(v)))
        except Exception:
            out[k] = None
    better = str(data.get("overall_better", "")).strip().upper()
    out["overall_better"] = better if better in ("A", "B", "TIE") else "tie"
    out["reason"] = str(data.get("reason", ""))[:120]
    if not data:
        out["error"] = "judge_parse_fail"
    return out


async def run_one(item, coord_swarm, coord_single, llm_client, timeout, sem, rng):
    qid = item["id"]
    q = item["question"]
    gold = item.get("answer", "")

    # 两个臂用不同 session_id，避免短期记忆互相污染
    sid_swarm = f"absw-{qid}-{uuid.uuid4().hex[:8]}"
    sid_single = f"absg-{qid}-{uuid.uuid4().hex[:8]}"

    async with sem:
        ans_swarm, lat_swarm, meta_swarm = await run_arm(coord_swarm, q, sid_swarm, timeout)
        ans_single, lat_single, meta_single = await run_arm(coord_single, q, sid_single, timeout)

        if not ans_swarm or not ans_single:
            return {
                "id": qid,
                "category": item["category"],
                "error": f"empty_answer(swarm={bool(ans_swarm)},single={bool(ans_single)})",
                "lat_swarm": lat_swarm,
                "lat_single": lat_single,
                "meta_swarm": meta_swarm,
                "meta_single": meta_single,
            }

        # 盲评：随机决定呈现顺序，swarm_is_A 记录映射
        swarm_is_a = rng.random() < 0.5
        if swarm_is_a:
            ja, jb = ans_swarm, ans_single
        else:
            ja, jb = ans_single, ans_swarm

        verdict = await judge(llm_client, q, gold, ja, jb)

        # 还原：把裁判对 A/B 的评价映射回 swarm / single
        def _map(ka, kb):
            if verdict.get(ka) is None or verdict.get(kb) is None:
                return None, None
            return (verdict[ka], verdict[kb]) if swarm_is_a else (verdict[kb], verdict[ka])

        acc_s, acc_g = _map("accuracy_A", "accuracy_B")
        cmp_s, cmp_g = _map("completeness_A", "completeness_B")
        saf_s, saf_g = _map("safety_A", "safety_B")

        better = verdict.get("overall_better", "tie")
        if better == "A":
            winner = "swarm" if swarm_is_a else "single"
        elif better == "B":
            winner = "single" if swarm_is_a else "swarm"
        else:
            winner = "tie"

    return {
        "id": qid,
        "category": item["category"],
        "question": q,
        "gold": gold,
        "answer_swarm": ans_swarm,
        "answer_single": ans_single,
        "swarm_is_A": swarm_is_a,
        "lat_swarm": lat_swarm,
        "lat_single": lat_single,
        "meta_swarm": meta_swarm,
        "meta_single": meta_single,
        "accuracy_swarm": acc_s,
        "accuracy_single": acc_g,
        "completeness_swarm": cmp_s,
        "completeness_single": cmp_g,
        "safety_swarm": saf_s,
        "safety_single": saf_g,
        "winner": winner,
        "reason": verdict.get("reason", ""),
        "judge_error": verdict.get("error", ""),
    }


def aggregate(rows: List[Dict]) -> Dict[str, Any]:
    def mean(vals):
        vals = [v for v in vals if isinstance(v, (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else None

    ok = [r for r in rows if not r.get("error")]

    lat_s = [r["lat_swarm"] for r in ok]
    lat_g = [r["lat_single"] for r in ok]

    winners = [r["winner"] for r in ok]
    by_cat: Dict[str, Any] = {}
    for cat in CATEGORIES:
        sub = [r for r in ok if r["category"] == cat]
        if not sub:
            continue
        by_cat[cat] = {
            "n": len(sub),
            "swarm_win": sum(1 for r in sub if r["winner"] == "swarm"),
            "single_win": sum(1 for r in sub if r["winner"] == "single"),
            "tie": sum(1 for r in sub if r["winner"] == "tie"),
            "lat_swarm_p50": percentile([r["lat_swarm"] for r in sub], 0.5),
            "lat_single_p50": percentile([r["lat_single"] for r in sub], 0.5),
        }

    return {
        "n_requested": len(rows),
        "n_valid": len(ok),
        "n_error": len(rows) - len(ok),
        "overall": {
            "swarm_win": sum(1 for w in winners if w == "swarm"),
            "single_win": sum(1 for w in winners if w == "single"),
            "tie": sum(1 for w in winners if w == "tie"),
            "swarm_win_rate": round(
                sum(1 for w in winners if w == "swarm") / max(len(winners), 1), 3
            ),
        },
        "scores_mean": {
            "accuracy_swarm": mean([r["accuracy_swarm"] for r in ok]),
            "accuracy_single": mean([r["accuracy_single"] for r in ok]),
            "completeness_swarm": mean([r["completeness_swarm"] for r in ok]),
            "completeness_single": mean([r["completeness_single"] for r in ok]),
            "safety_swarm": mean([r["safety_swarm"] for r in ok]),
            "safety_single": mean([r["safety_single"] for r in ok]),
        },
        "latency_sec": {
            "swarm_p50": percentile(lat_s, 0.5),
            "swarm_p95": percentile(lat_s, 0.95),
            "single_p50": percentile(lat_g, 0.5),
            "single_p95": percentile(lat_g, 0.95),
            "slowdown_p50": round(
                percentile(lat_s, 0.5) / max(percentile(lat_g, 0.5), 1e-9), 2
            ),
        },
        "routing": {
            # enable_swarm=True 时真正走 Swarm 的比例：若很低，说明 A/B 两臂差异被稀释
            "swarm_arm_actually_used_swarm": sum(
                1 for r in ok if (r.get("meta_swarm") or {}).get("used_swarm")
            ),
            "swarm_arm_timeout": sum(
                1 for r in ok if (r.get("meta_swarm") or {}).get("timeout_occurred")
            ),
        },
        # 按"Swarm 是否真的激活"拆分：只有激活的样本才反映 Swarm 的真实增量价值
        "routing_split": {
            "activated": _win_split([r for r in ok if (r.get("meta_swarm") or {}).get("used_swarm")]),
            "not_activated": _win_split(
                [r for r in ok if not (r.get("meta_swarm") or {}).get("used_swarm")]
            ),
        },
        "by_category": by_cat,
    }


def _win_split(rows: List[Dict]) -> Dict[str, Any]:
    """给定子集，统计胜负与延迟。"""
    if not rows:
        return {"n": 0}
    w = [r["winner"] for r in rows]
    return {
        "n": len(rows),
        "swarm_win": sum(1 for x in w if x == "swarm"),
        "single_win": sum(1 for x in w if x == "single"),
        "tie": sum(1 for x in w if x == "tie"),
        "lat_swarm_p50": percentile([r["lat_swarm"] for r in rows], 0.5),
        "lat_single_p50": percentile([r["lat_single"] for r in rows], 0.5),
    }


def write_report(path: Path, summary: Dict[str, Any], meta: Dict[str, Any], rows: List[Dict]):
    ov = summary["overall"]
    sc = summary["scores_mean"]
    lt = summary["latency_sec"]
    rt = summary["routing"]
    rs = summary["routing_split"]
    n = summary["n_valid"]

    lines = [
        "# A/B 评测：Swarm vs Single Agent",
        "",
        "> **自动 LLM 盲评，非医学专家盲评。面试须降级表述。**",
        "",
        f"- 样本 n={n}（seed={meta.get('seed')}，每类 {meta.get('per_category')}，共 {meta.get('n_requested')}）",
        f"- A 臂：`SwarmCoordinator(enable_swarm=True)`",
        f"- B 臂：`SwarmCoordinator(enable_swarm=False)`",
        f"- 两者使用同一知识库（{meta.get('kb_chunks', '?')} 条 chunk）与同一套工具，差异仅在于是否启用多 Agent 协作",
        f"- 盲评：每题随机决定呈现顺序，裁判不知哪个是 Swarm",
        f"- 耗时 {meta.get('elapsed_sec')}s",
        "",
        "## 总体结果",
        "",
        f"| 结果 | 数量 | 占比 |",
        f"|---|---|---|",
        f"| Swarm 胜 | {ov['swarm_win']} | {ov['swarm_win'] / max(n, 1):.1%} |",
        f"| Single 胜 | {ov['single_win']} | {ov['single_win'] / max(n, 1):.1%} |",
        f"| 平局 | {ov['tie']} | {ov['tie'] / max(n, 1):.1%} |",
        "",
        "## 三维均分（1-5）",
        "",
        "| 维度 | Swarm | Single | 差值 |",
        "|---|---|---|---|",
    ]
    for name, ks, kg in (
        ("accuracy", "accuracy_swarm", "accuracy_single"),
        ("completeness", "completeness_swarm", "completeness_single"),
        ("safety", "safety_swarm", "safety_single"),
    ):
        a, b = sc.get(ks), sc.get(kg)
        diff = round(a - b, 3) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "-"
        lines.append(f"| {name} | {a} | {b} | {diff} |")

    lines += [
        "",
        "## 延迟（秒）",
        "",
        "| 指标 | Swarm | Single |",
        "|---|---|---|",
        f"| p50 | {lt['swarm_p50']} | {lt['single_p50']} |",
        f"| p95 | {lt['swarm_p95']} | {lt['single_p95']} |",
        f"| p50 倍数 | {lt['slowdown_p50']}x | 1x |",
        "",
        "## 路由与超时",
        "",
        f"- A 臂（enable_swarm=True）中真正走 Swarm 的：{rt['swarm_arm_actually_used_swarm']}/{n}",
        f"- A 臂内部超时（55s）：{rt['swarm_arm_timeout']}/{n}",
        "",
        "### 按 Swarm 是否激活拆分（关键）",
        "",
        "| 子集 | n | Swarm胜 | Single胜 | 平 | Swarm p50 | Single p50 |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, key in (("Swarm 已激活", "activated"), ("未激活（回落单 Agent）", "not_activated")):
        d = rs.get(key) or {"n": 0}
        if not d.get("n"):
            lines.append(f"| {label} | 0 | - | - | - | - | - |")
            continue
        lines.append(
            f"| {label} | {d['n']} | {d['swarm_win']} | {d['single_win']} | {d['tie']} "
            f"| {d['lat_swarm_p50']} | {d['lat_single_p50']} |"
        )

    lines += [
        "",
        "## 分类明细",
        "",
        "| 类别 | n | Swarm胜 | Single胜 | 平 | Swarm p50 | Single p50 |",
        "|---|---|---|---|---|---|---|",
    ]
    for cat, d in summary["by_category"].items():
        lines.append(
            f"| {cat} | {d['n']} | {d['swarm_win']} | {d['single_win']} | {d['tie']} "
            f"| {d['lat_swarm_p50']} | {d['lat_single_p50']} |"
        )

    lines += [
        "",
        "## 结论口径（可直接用于面试）",
        "",
        f"- Swarm 相对单 Agent 的胜率：{ov['swarm_win_rate']:.1%}；"
        f"三维均分差值 accuracy {sc.get('accuracy_swarm')} vs {sc.get('accuracy_single')}、"
        f"completeness {sc.get('completeness_swarm')} vs {sc.get('completeness_single')}",
        f"- 代价：p50 延迟 {lt['swarm_p50']}s vs {lt['single_p50']}s（{lt['slowdown_p50']}x）",
        f"- 样本量 {n}，自动裁判，结论为方向性参考，不做统计显著性断言",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


async def async_main(args):
    from loguru import logger

    from core.llm_client import LLMClient
    from swarm import SwarmCoordinator

    rows = load_jsonl(Path(args.data))
    picked = stratified_sample(rows, args.per_category, args.seed)
    if args.limit:
        picked = picked[: args.limit]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"ab_swarm_vs_single_{ts}.jsonl"

    done = load_done_ids(Path(args.resume_detail)) if args.resume else set()
    if done:
        picked = [r for r in picked if r["id"] not in done]
        logger.info(f"resume: skip {len(done)} done, {len(picked)} remaining")

    logger.info(f"SwarmCoordinator(enable_swarm=True)  ...")
    coord_swarm = SwarmCoordinator(enable_swarm=True)
    logger.info(f"SwarmCoordinator(enable_swarm=False) ...")
    coord_single = SwarmCoordinator(enable_swarm=False)
    llm_client = LLMClient()

    # 记录评测时的知识库规模，便于结论可复现
    kb_chunks = None
    try:
        from knowledge.milvus_kb import MedicalKnowledgeBase

        kb_chunks = len(MedicalKnowledgeBase()._fetch_all_chunks())
    except Exception:
        pass

    # 预热：首次调用会加载 embedding 模型 + reranker（可能数十秒），
    # 不预热会把这两项成本算进前几题的延迟，污染 p50/p95
    logger.info("warming up (loading embedding model + reranker) ...")
    try:
        await asyncio.wait_for(
            coord_swarm.process("你好", session_id="warmup-swarm"), timeout=240
        )
        await asyncio.wait_for(
            coord_single.process("你好", session_id="warmup-single"), timeout=240
        )
        logger.info("warmup done")
    except Exception as e:
        logger.warning(f"warmup failed (ignored): {type(e).__name__}: {e}")

    sem = asyncio.Semaphore(args.concurrency)
    rng = random.Random(args.seed + 1)

    t_start = time.perf_counter()
    # 并发执行（受 --concurrency 限制），串行跑 24 题需 20+ 分钟
    tasks = [
        asyncio.create_task(
            run_one(item, coord_swarm, coord_single, llm_client, args.timeout, sem, rng)
        )
        for item in picked
    ]

    results: List[Dict] = []
    finished = 0
    for coro in asyncio.as_completed(tasks):
        r = await coro
        finished += 1
        results.append(r)
        with detail_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        w = r.get("winner", "ERR")
        logger.info(
            f"[{finished}/{len(picked)}] {r.get('id')} ({r.get('category')}) -> {w} "
            f"| lat swarm={r.get('lat_swarm')}s single={r.get('lat_single')}s"
        )

    elapsed = round(time.perf_counter() - t_start, 1)
    summary = aggregate(results)
    meta = {
        "seed": args.seed,
        "per_category": args.per_category,
        "n_requested": len(picked),
        "concurrency": args.concurrency,
        "timeout": args.timeout,
        "elapsed_sec": elapsed,
        "kb_chunks": kb_chunks,
        "generated_at": ts,
    }

    json_path = out_dir / f"ab_swarm_vs_single_{ts}.json"
    json_path.write_text(
        json.dumps({"meta": meta, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path = out_dir / f"ab_swarm_vs_single_{ts}.md"
    write_report(md_path, summary, meta, results)

    print("\n" + "=" * 60)
    print(f"n_valid={summary['n_valid']}  n_error={summary['n_error']}")
    print(f"win: swarm={summary['overall']['swarm_win']} "
          f"single={summary['overall']['single_win']} tie={summary['overall']['tie']}")
    print(f"latency p50: swarm={summary['latency_sec']['swarm_p50']}s "
          f"single={summary['latency_sec']['single_p50']}s "
          f"({summary['latency_sec']['slowdown_p50']}x)")
    act = summary["routing_split"]["activated"]
    print(f"routing: swarm_arm_used_swarm={summary['routing']['swarm_arm_actually_used_swarm']}/{summary['n_valid']} "
          f"timeout={summary['routing']['swarm_arm_timeout']}")
    if act.get("n"):
        print(f"  [swarm activated n={act['n']}] "
              f"swarm_win={act['swarm_win']} single_win={act['single_win']} tie={act['tie']} "
              f"| lat p50 swarm={act['lat_swarm_p50']}s single={act['lat_single_p50']}s")
    print(f"report: {md_path}")
    print("=" * 60)


def merge_main(args) -> int:
    """合并模式：把多轮评测的明细 jsonl 合并聚合，统一出报告。"""
    import glob

    paths = sorted(glob.glob(args.merge))
    if not paths:
        print(f"no files match: {args.merge}")
        return 1

    rows: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

    summary = aggregate(rows)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = args.merge_name or "merged"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "files": [Path(p).name for p in paths],
        "merged_at": ts,
        "n_total": len(rows),
        "kb_chunks": "见各轮明细",
    }
    json_path = out_dir / f"ab_swarm_vs_single_{name}.json"
    json_path.write_text(
        json.dumps({"meta": meta, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path = out_dir / f"ab_swarm_vs_single_{name}.md"
    write_report(md_path, summary, meta, rows)

    act = summary["routing_split"]["activated"]
    print("=" * 60)
    print(f"merged {len(rows)} rows from {len(paths)} files")
    print(f"n_valid={summary['n_valid']}  n_error={summary['n_error']}")
    print(f"win: swarm={summary['overall']['swarm_win']} "
          f"single={summary['overall']['single_win']} tie={summary['overall']['tie']}")
    print(f"activated: n={act.get('n')} swarm_win={act.get('swarm_win')} "
          f"single_win={act.get('single_win')} tie={act.get('tie')}")
    print(f"report: {md_path}")
    print("=" * 60)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "eval" / "data" / "benchmark_500.jsonl"))
    p.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-category", type=int, default=6, help="6*4=24")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-detail", default="")
    p.add_argument("--merge", default="", help="合并模式：glob 匹配多轮明细 jsonl")
    p.add_argument("--merge-name", default="")
    args = p.parse_args()
    if args.merge:
        sys.exit(merge_main(args))
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
