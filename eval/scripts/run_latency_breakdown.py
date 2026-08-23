#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分阶段耗时：小样本插桩 Coordinator（路由/主路径/检索/汇总）。"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CATEGORIES = [
    "health_consult",
    "symptom_diagnosis",
    "disease_knowledge",
    "guideline_retrieval",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratified_sample(rows: List[Dict[str, Any]], per_cat: int, seed: int) -> List[Dict[str, Any]]:
    import random

    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    rng = random.Random(seed)
    picked: List[Dict[str, Any]] = []
    for cat in CATEGORIES:
        pool = list(by_cat[cat])
        if len(pool) < per_cat:
            raise ValueError(f"{cat}: need {per_cat}, have {len(pool)}")
        rng.shuffle(pool)
        picked.extend(pool[:per_cat])
    rng.shuffle(picked)
    return picked


def percentile(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    k = (len(ys) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return round(ys[f], 3)
    return round(ys[f] + (ys[c] - ys[f]) * (k - f), 3)


def stats(xs: List[float]) -> Dict[str, Any]:
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None}
    return {
        "n": len(xs),
        "mean": round(sum(xs) / len(xs), 3),
        "p50": percentile(xs, 50),
        "p95": percentile(xs, 95),
    }


def install_kb_timer(bucket: Dict[str, float], kb=None):
    from knowledge.milvus_kb import MedicalKnowledgeBase

    if kb is None:
        kb = MedicalKnowledgeBase._instance
    if kb is None:
        # 尚未初始化：稍后再挂；不在此处新建以免抢锁
        return None
    if getattr(kb, "_latency_timer_installed", False):
        orig = kb._latency_orig_search
    else:
        orig = kb.search
        kb._latency_orig_search = orig
        kb._latency_timer_installed = True

    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            bucket["retrieval"] = bucket.get("retrieval", 0.0) + (time.perf_counter() - t0)

    kb.search = wrapped  # type: ignore
    return kb


def restore_kb_search():
    from knowledge.milvus_kb import MedicalKnowledgeBase

    kb = MedicalKnowledgeBase._instance
    if kb is None:
        return
    orig = getattr(kb, "_latency_orig_search", None)
    if orig is not None:
        kb.search = orig  # type: ignore
    kb._latency_timer_installed = False


async def run_one(coordinator, item: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    timings: Dict[str, float] = {}
    q = item["question"]
    sid = f"latbd-{item['id']}-{uuid.uuid4().hex[:6]}"

    # wrap Lead decompose
    lead = coordinator.lead_agent
    orig_decomp = lead.assess_and_decompose

    async def decomp_wrap(*a, **k):
        t0 = time.perf_counter()
        try:
            return await orig_decomp(*a, **k)
        finally:
            timings["route_decompose"] = time.perf_counter() - t0

    lead.assess_and_decompose = decomp_wrap  # type: ignore

    orig_synth = lead.synthesize_results

    async def synth_wrap(*a, **k):
        t0 = time.perf_counter()
        try:
            return await orig_synth(*a, **k)
        finally:
            timings["synthesize"] = timings.get("synthesize", 0.0) + (time.perf_counter() - t0)

    lead.synthesize_results = synth_wrap  # type: ignore

    # wrap worker agents process as main_llm_path (含工具/LLM，检索另计)
    wrapped_agents = []
    for agent in [
        coordinator.consultation_agent,
        getattr(coordinator, "diagnostic_agent", None),
        getattr(coordinator, "research_agent", None),
    ]:
        if agent is None:
            continue
        orig_p = agent.process

        async def process_wrap(*a, _op=orig_p, **k):
            t0 = time.perf_counter()
            try:
                return await _op(*a, **k)
            finally:
                timings["agent_process"] = timings.get("agent_process", 0.0) + (
                    time.perf_counter() - t0
                )

        agent.process = process_wrap  # type: ignore
        wrapped_agents.append((agent, orig_p))

    install_kb_timer(timings)

    t0 = time.perf_counter()
    err = None
    mode = None
    answer_len = 0
    try:
        result = await asyncio.wait_for(coordinator.process(q, session_id=sid), timeout=timeout)
        mode = result.get("mode") or (
            "swarm" if result.get("swarm_enabled") else "single"
        )
        ans = result.get("answer") or ""
        answer_len = len(ans)
    except Exception as e:
        err = f"{type(e).__name__}:{e}"
    total = time.perf_counter() - t0

    # restore
    lead.assess_and_decompose = orig_decomp  # type: ignore
    lead.synthesize_results = orig_synth  # type: ignore
    for agent, op in wrapped_agents:
        agent.process = op  # type: ignore
    restore_kb_search()

    # agent_process 含检索；主 LLM 近似 = agent_process - retrieval（同一次请求内）
    retrieval = timings.get("retrieval", 0.0)
    agent_p = timings.get("agent_process", 0.0)
    main_llm_approx = max(0.0, agent_p - retrieval)

    accounted = (
        timings.get("route_decompose", 0.0)
        + agent_p
        + timings.get("synthesize", 0.0)
    )
    other = max(0.0, total - accounted)

    return {
        "id": item["id"],
        "category": item["category"],
        "mode": mode,
        "error": err,
        "total_sec": round(total, 3),
        "route_decompose_sec": round(timings.get("route_decompose", 0.0), 3),
        "agent_process_sec": round(agent_p, 3),
        "retrieval_sec": round(retrieval, 3),
        "main_llm_approx_sec": round(main_llm_approx, 3),
        "synthesize_sec": round(timings.get("synthesize", 0.0), 3),
        "other_sec": round(other, 3),
        "answer_len": answer_len,
    }


def write_md(path: Path, summary: Dict[str, Any], details: List[Dict[str, Any]]) -> None:
    s = summary
    lines = [
        "# 分阶段耗时评测",
        "",
        f"- 时间: {s['finished_at']}",
        f"- n={s['n']}（seed={s['seed']}，每类 {s['per_category']}）",
        f"- 超时: {s['timeout_sec']}s",
        "",
        "## 方法",
        "",
        "- 对 `SwarmCoordinator` 插桩：`assess_and_decompose`（路由）、各 Agent `process`、`synthesize_results`、`MedicalKnowledgeBase.search`",
        "- `main_llm_approx` = `agent_process - retrieval`（同请求内；含非检索工具与多轮 LLM，非纯 chat）",
        "- `other` ≈ 记忆检索/开销（未单独拆 Mem0）",
        "",
        "## 局限",
        "",
        "- 小样本；DeepSeek 限流会影响绝对值",
        "- Swarm 并行时 agent_process 为各 worker 累加墙钟近似，可能略大于 wall",
        "- 检索计时挂在 KB 单例上，并发评测勿并行跑本脚本",
        "",
        "## 汇总（秒）",
        "",
        "| 阶段 | mean | p50 | p95 |",
        "|---|---:|---:|---:|",
    ]
    for key in [
        "total",
        "route_decompose",
        "agent_process",
        "retrieval",
        "main_llm_approx",
        "synthesize",
        "other",
    ]:
        st = s["stages"][key]
        lines.append(f"| {key} | {st['mean']} | {st['p50']} | {st['p95']} |")

    lines.extend(["", "## 分品类 total mean", "", "| category | n | mean_total |", "|---|---:|---:|"])
    for cat, st in s["by_category"].items():
        lines.append(f"| {cat} | {st['n']} | {st['mean_total']} |")

    ok = [d for d in details if not d.get("error")]
    lines.extend(["", f"成功 {len(ok)}/{s['n']}；失败见 detail jsonl", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


async def async_main(args: argparse.Namespace) -> None:
    try:
        from loguru import logger as _logger

        _logger.remove()
        _logger.add(sys.stderr, level="WARNING")
    except Exception:
        pass

    rows = load_jsonl(Path(args.data))
    sample = stratified_sample(rows, args.per_category, args.seed)
    if args.limit and args.limit > 0:
        sample = sample[: args.limit]

    from swarm import SwarmCoordinator

    coordinator = SwarmCoordinator(enable_swarm=True)
    details: List[Dict[str, Any]] = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.run_id
    detail_path = out_dir / f"latency_breakdown_detail_{stamp}.jsonl"
    if detail_path.exists():
        detail_path.unlink()

    print(f"[latency_bd] n={len(sample)}")
    for i, item in enumerate(sample, 1):
        rec = await run_one(coordinator, item, args.timeout)
        details.append(rec)
        with detail_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"  [{i}/{len(sample)}] {item['id']} total={rec['total_sec']}s "
            f"route={rec['route_decompose_sec']} agent={rec['agent_process_sec']} "
            f"retr={rec['retrieval_sec']} err={rec.get('error')}"
        )
        await asyncio.sleep(args.pace)

    def col(name: str) -> List[float]:
        key = f"{name}_sec" if name != "total" else "total_sec"
        return [float(d[key]) for d in details if not d.get("error") and isinstance(d.get(key), (int, float))]

    by_cat = {}
    for cat in CATEGORIES:
        rows_c = [d for d in details if d.get("category") == cat and not d.get("error")]
        by_cat[cat] = {
            "n": len(rows_c),
            "mean_total": round(sum(d["total_sec"] for d in rows_c) / len(rows_c), 3) if rows_c else None,
        }

    summary = {
        "seed": args.seed,
        "per_category": args.per_category,
        "n": len(details),
        "n_ok": sum(1 for d in details if not d.get("error")),
        "timeout_sec": args.timeout,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "stages": {
            "total": stats(col("total")),
            "route_decompose": stats(col("route_decompose")),
            "agent_process": stats(col("agent_process")),
            "retrieval": stats(col("retrieval")),
            "main_llm_approx": stats(col("main_llm_approx")),
            "synthesize": stats(col("synthesize")),
            "other": stats(col("other")),
        },
        "by_category": by_cat,
    }
    md_path = out_dir / f"latency_breakdown_{stamp}.md"
    json_path = out_dir / f"latency_breakdown_{stamp}.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(md_path, summary, details)
    print(f"[done] -> {md_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "eval" / "data" / "benchmark_500.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-category", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--pace", type=float, default=0.5, help="请求间隔秒，缓解限流")
    ap.add_argument("--run-id", default="n24")
    args = ap.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
