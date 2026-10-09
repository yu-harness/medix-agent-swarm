#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 2 实测：Per-Request 成本归因 + 请求耗时瀑布流。

跑两条典型问题：
1. 单 Agent 路由（慢病生活方式类，预期只拆出 1 个子任务）；
2. Swarm 多 Agent 路由（多症状 + 共病 + 高危血压，预期拆出多个子任务）。

每条问题输出：最终回答、本次运行实际命中的中文来源头（带页码）、
按阶段拆分的 Token/费用账单、ASCII 耗时瀑布流；最后给出两者的倍数对比。

用法：
    python scripts/eval_cost_and_waterfall.py
    python scripts/eval_cost_and_waterfall.py --answer-chars 800 --json-out
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.observability import format_span_waterfall  # noqa: E402
from swarm import process_with_swarm  # noqa: E402

CASES: List[Dict[str, str]] = [
    {
        "label": "单 Agent 路由",
        "expect": "single_agent",
        "question": "高血压患者日常如何进行低盐饮食管理？",
    },
    {
        "label": "Swarm 多 Agent 路由",
        "expect": "swarm",
        "question": "头晕胸闷、血压 165/105，同时有糖尿病病史，该怎么处理？",
    },
]


def _sources_from_spans(spans: List[Dict[str, Any]], limit: int = 6) -> List[str]:
    """从 Span 里取本次运行实际命中的来源头（Skill 执行时挂上去的，不是事后复算）。"""
    found: List[str] = []
    for s in spans:
        for src in (s.get("meta") or {}).get("sources") or []:
            if src not in found:
                found.append(src)
    return found[:limit]


def _print_bill(usage: Dict[str, Any]) -> None:
    print("  💰 成本账单（DeepSeek 计价：输入 ¥%.2f/1M，输出 ¥%.2f/1M）" % (
        usage["price"]["prompt_cny_per_1m"], usage["price"]["completion_cny_per_1m"]))
    print("     %-30s %8s %8s %11s %11s %5s" % ("阶段", "tokens", "prompt", "completion", "费用(¥)", "调用"))
    for stage, row in usage.get("breakdown", {}).items():
        print("     %-30s %8d %8d %11d %11.5f %5d" % (
            stage, row["tokens"], row["prompt_tokens"], row["completion_tokens"],
            row["cost_cny"], row["calls"]))
    print("     %s" % ("-" * 78))
    print("     %-30s %8d %8d %11d %11.5f" % (
        "合计", usage["total_tokens"], usage["prompt_tokens"],
        usage["completion_tokens"], usage["estimated_cost_cny"]))
    if usage.get("unpriced_calls"):
        print("     ⚠️ 有 %d 次调用未拿到 usage（已标为未计价，未用字符数估算）" % usage["unpriced_calls"])


def warmup() -> float:
    """预热检索链路：加载向量模型、CrossEncoder 重排器并建好 BM25 索引。

    不做这一步，**第一条问题会把 20~30 秒的冷启动算进自己的耗时**，
    "单 Agent vs Swarm 的耗时倍数"就会失真（实测第一次跑出来是 0.68x，
    看起来 Swarm 更快，其实只是第一个用例替第二个用例付了模型加载的钱）。
    预热是纯本地计算，不调 LLM、不花钱。
    """
    from knowledge.milvus_kb import MedicalKnowledgeBase

    t0 = time.perf_counter()
    MedicalKnowledgeBase().search("预热：高血压与低盐饮食", top_k=2)
    return round((time.perf_counter() - t0) * 1000, 1)


async def run_case(case: Dict[str, str], answer_chars: int) -> Dict[str, Any]:
    print()
    print("=" * 78)
    print("▶ %s" % case["label"])
    print("  问题：%s" % case["question"])
    print("=" * 78)

    t0 = time.perf_counter()
    result = await process_with_swarm(case["question"])
    wall_ms = round((time.perf_counter() - t0) * 1000, 1)

    route = "swarm" if result.get("swarm_enabled") else "single_agent"
    matched = "✅ 与预期一致" if route == case["expect"] else "⚠️ 实际路由与预期不同（如实记录）"
    print("\n🧭 路由：%s（%s）｜trace_id=%s" % (route, matched, result.get("trace_id")))
    if result.get("route_reason"):
        print("   路由原因：%s" % result["route_reason"])
    if result.get("agents_involved"):
        print("   参与 Agent：%s" % "、".join(result["agents_involved"]))
    print("   端到端耗时：%.1f ms（脚本侧 wall=%.1f ms）" % (result.get("total_ms") or 0, wall_ms))

    print("\n📄 最终回答（前 %d 字）：" % answer_chars)
    answer = (result.get("answer") or "").strip()
    print("   " + answer[:answer_chars].replace("\n", "\n   "))
    if len(answer) > answer_chars:
        print("   …（共 %d 字）" % len(answer))

    spans = result.get("spans") or []
    sources = _sources_from_spans(spans)
    print("\n🔖 本次运行命中的来源头（来自检索 Skill 的 Span，共 %d 条，展示前 %d）："
          % (len(_sources_from_spans(spans, 999)), len(sources)))
    if sources:
        for s in sources:
            print("   %s" % s)
    else:
        print("   （本次没有触发检索）")

    usage = result.get("usage_and_cost") or {}
    print()
    if usage:
        _print_bill(usage)
    else:
        print("  ⚠️ 未拿到成本账单")

    print()
    print(format_span_waterfall(spans, title="⏱️ 耗时瀑布流（%s）" % case["label"]))

    return {
        "label": case["label"],
        "question": case["question"],
        "route": route,
        "route_expected": case["expect"],
        "route_matched": route == case["expect"],
        "answer": answer,
        "sources": sources,
        "total_ms": result.get("total_ms"),
        "wall_ms": wall_ms,
        "usage_and_cost": usage,
        "spans": spans,
        "timings": result.get("timings"),
        "trace_id": result.get("trace_id"),
        "agents_involved": result.get("agents_involved") or [],
    }


def _ratio(a: float, b: float) -> str:
    if not b:
        return "—"
    return "%.2fx" % (a / b)


def main() -> int:
    ap = argparse.ArgumentParser(description="成本归因 + 耗时瀑布流实测")
    ap.add_argument("--answer-chars", type=int, default=600, help="回答截断长度（默认 600）")
    ap.add_argument("--json-out", action="store_true", help="把原始结果落到 eval/results/")
    ap.add_argument("--no-warmup", action="store_true", help="跳过检索链路预热（不推荐）")
    args = ap.parse_args()

    if not args.no_warmup:
        ms = warmup()
        print("🔥 预热完成：向量模型 + 重排器 + BM25 索引已就绪（%.1f ms，纯本地、零 API 成本）" % ms)

    records = [asyncio.run(run_case(c, args.answer_chars)) for c in CASES]

    # ---- 对比 ----
    single = next((r for r in records if r["route"] == "single_agent"), None)
    swarm = next((r for r in records if r["route"] == "swarm"), None)
    print()
    print("=" * 78)
    print("📊 单 Agent vs Swarm：耗时与成本倍数对比")
    print("=" * 78)
    if not single or not swarm:
        print("  两次运行没有同时覆盖两条路由，跳过倍数对比（见上面的实际路由）")
    else:
        s_usage, w_usage = single["usage_and_cost"], swarm["usage_and_cost"]
        rows = [
            ("端到端耗时", "%.1f ms" % (single["total_ms"] or 0), "%.1f ms" % (swarm["total_ms"] or 0),
             _ratio(swarm["total_ms"] or 0, single["total_ms"] or 0)),
            ("Token 总量", str(s_usage.get("total_tokens", 0)), str(w_usage.get("total_tokens", 0)),
             _ratio(w_usage.get("total_tokens", 0), s_usage.get("total_tokens", 0))),
            ("费用", "¥%.5f" % s_usage.get("estimated_cost_cny", 0),
             "¥%.5f" % w_usage.get("estimated_cost_cny", 0),
             _ratio(w_usage.get("estimated_cost_cny", 0), s_usage.get("estimated_cost_cny", 0))),
            ("参与 Agent 数", str(len(single.get("agents_involved") or []) or 1),
             str(len(swarm.get("agents_involved") or [])), "—"),
        ]
        print("  %-14s %-16s %-16s %-8s" % ("指标", "单 Agent", "Swarm", "Swarm/单 Agent"))
        for name, a, b, r in rows:
            print("  %-14s %-16s %-16s %-8s" % (name, a, b, r))
        print()
        print("  口径：两条问题在同一进程内串行执行、检索链路已预热，模型冷启动不计入；"
              "每次只有 1 条样本，倍数只看量级。")
        print("  结论：Swarm 用 %s 的耗时与 %s 的成本，换来多 Agent 分工覆盖（含高风险筛查与生活方式建议）。"
              % (_ratio(swarm["total_ms"] or 0, single["total_ms"] or 0),
                 _ratio(w_usage.get("estimated_cost_cny", 0), s_usage.get("estimated_cost_cny", 0))))

    if args.json_out:
        out_dir = ROOT / "eval" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"day2_cost_waterfall_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(
            json.dumps({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": records},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("\n原始结果已落盘：%s" % path.relative_to(ROOT))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
