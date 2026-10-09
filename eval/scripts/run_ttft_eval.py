#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""TTFT 实测：同批问题下对比「用户感知的首 token 时间」与「总时长」。

不变量（与 run_latency_breakdown.py 保持一致，便于与既往往总数对照）：
- 同一数据集 eval/data/benchmark_500.jsonl
- 同样分层抽样（四类各 per-category 条，同一 seed）
- 串行执行（concurrency=1），避免排队与限流干扰绝对值

两个 TTFT：
- client_ttft_ms：从发出请求到第一个答案片段到达的时间，即用户感知的响应时间
- answer_ttft_ms：模型侧首个 token 的时间（由 agent loop / Lead 汇总打点）
两者之差是检索与编排等前置开销。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from swarm import process_with_swarm  # noqa: E402

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
        return round(ys[f], 1)
    return round(ys[f] + (ys[c] - ys[f]) * (k - f), 1)


def stats(xs: List[float]) -> Dict[str, Any]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "n": len(xs),
        "mean": round(statistics.mean(xs), 1),
        "p50": percentile(xs, 50),
        "p95": percentile(xs, 95),
        "max": round(max(xs), 1),
    }


async def run_one(item: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """跑一条问题，记录 client TTFT / 模型 TTFT / 总时长。"""
    trace_id = uuid.uuid4().hex[:12]
    first_delta_ms: Optional[float] = None
    delta_chars = 0
    t0 = time.perf_counter()

    async def on_delta(text: str) -> None:
        nonlocal first_delta_ms, delta_chars
        if first_delta_ms is None:
            first_delta_ms = round((time.perf_counter() - t0) * 1000, 1)
        delta_chars += len(text)

    record: Dict[str, Any] = {
        "id": item.get("id"),
        "category": item.get("category"),
        "question": item.get("question"),
        "trace_id": trace_id,
    }
    try:
        result = await asyncio.wait_for(
            process_with_swarm(
                item["question"],
                context={"trace_id": trace_id},
                trace_id=trace_id,
                on_delta=on_delta,
                stream=True,
            ),
            timeout=timeout,
        )
        record.update({
            "ok": True,
            "mode": "swarm" if result.get("swarm_enabled") else "single_agent",
            "client_ttft_ms": first_delta_ms,
            "answer_ttft_ms": result.get("answer_ttft_ms"),
            "total_ms": round((time.perf_counter() - t0) * 1000, 1),
            "llm_calls": result.get("llm_calls"),
            "delta_chars": delta_chars,
            "agents_involved": result.get("agents_involved"),
            "answer_chars": len(result.get("answer") or ""),
        })
    except Exception as e:
        record.update({
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "total_ms": round((time.perf_counter() - t0) * 1000, 1),
        })
    return record


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in records if r.get("ok")]
    summary: Dict[str, Any] = {
        "n_total": len(records),
        "n_ok": len(ok),
        "n_failed": len(records) - len(ok),
        "overall": {
            "client_ttft_ms": stats([r.get("client_ttft_ms") for r in ok]),
            "answer_ttft_ms": stats([r.get("answer_ttft_ms") for r in ok]),
            "total_ms": stats([r.get("total_ms") for r in ok]),
        },
        "by_mode": {},
    }
    for mode in ("single_agent", "swarm"):
        group = [r for r in ok if r.get("mode") == mode]
        if group:
            summary["by_mode"][mode] = {
                "client_ttft_ms": stats([r.get("client_ttft_ms") for r in group]),
                "answer_ttft_ms": stats([r.get("answer_ttft_ms") for r in group]),
                "total_ms": stats([r.get("total_ms") for r in group]),
            }
    return summary


def render_markdown(summary: Dict[str, Any], args: argparse.Namespace, stamp: str) -> str:
    lines = [
        "# TTFT 实测（流式）",
        "",
        f"- 时间：{stamp}",
        f"- 样本：n={summary['n_ok']}/{summary['n_total']}（seed={args.seed}，每类 {args.per_category}）",
        "- 执行方式：串行（concurrency=1），问题与 run_latency_breakdown.py 同源同抽样",
        "",
        "## 总体",
        "",
        "| 指标 | n | mean | p50 | p95 | max |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    labels = {
        "client_ttft_ms": "用户感知 TTFT（ms）",
        "answer_ttft_ms": "模型侧 TTFT（ms）",
        "total_ms": "总时长（ms）",
    }
    for key, label in labels.items():
        s = summary["overall"][key]
        lines.append(f"| {label} | {s['n']} | {s['mean']} | {s['p50']} | {s['p95']} | {s['max']} |")

    for mode, group in summary["by_mode"].items():
        lines += ["", f"## {mode}", "", "| 指标 | n | mean | p50 | p95 | max |", "| --- | --- | --- | --- | --- | --- |"]
        for key, label in labels.items():
            s = group[key]
            lines.append(f"| {label} | {s['n']} | {s['mean']} | {s['p50']} | {s['p95']} | {s['max']} |")

    lines += [
        "",
        "## 读法与边界",
        "",
        "- 用户感知的是 client_ttft_ms：流式下它取代总时长成为「响应快不快」的指标",
        "- client_ttft_ms 减 answer_ttft_ms 是检索与编排等前置开销，不含生成",
        "- Swarm 路由推的是 Lead 汇总那一层，它的 client_ttft_ms 天然包含各 worker 的执行时间",
        "- 本机单机、串行执行；模型限流仍会推高绝对值，跨机器比较需谨慎",
        "",
    ]
    return "\n".join(lines)


async def async_main(args: argparse.Namespace) -> None:
    rows = load_jsonl(Path(args.data))
    picked = stratified_sample(rows, args.per_category, args.seed)
    print(f"样本 {len(picked)} 条（seed={args.seed}，每类 {args.per_category}），串行执行")

    records: List[Dict[str, Any]] = []
    for index, item in enumerate(picked, 1):
        record = await run_one(item, args.timeout)
        records.append(record)
        flag = "OK " if record.get("ok") else "ERR"
        print(
            f"[{index}/{len(picked)}] {flag} {record.get('id')} "
            f"client_ttft={record.get('client_ttft_ms')}ms "
            f"answer_ttft={record.get('answer_ttft_ms')}ms "
            f"total={record.get('total_ms')}ms mode={record.get('mode')}"
        )

    summary = summarize(records)
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"ttft_eval_{stamp}.json"
    md_path = out_dir / f"ttft_eval_{stamp}.md"
    detail_path = out_dir / f"ttft_eval_detail_{stamp}.jsonl"

    json_path.write_text(
        json.dumps({"args": vars(args), "summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(summary, args, stamp), encoding="utf-8")
    with detail_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n=== 汇总 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n结果文件：{md_path.name} / {json_path.name} / {detail_path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "eval" / "data" / "benchmark_500.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-category", type=int, default=5, help="每类取几条；默认 5，合计 20 条")
    ap.add_argument("--timeout", type=float, default=180.0, help="单条超时（秒）")
    args = ap.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
