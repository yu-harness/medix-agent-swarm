#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""小规模并发吞吐：成功率 / QPS / 平均延迟。"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# 短路径健康问句，降低单请求复杂度
QUESTIONS = [
    "如何预防感冒？",
    "多喝水有什么好处？",
    "每天走多少步比较合适？",
    "睡眠不足会有什么影响？",
    "低盐饮食怎么做？",
]


async def run_one(coordinator, idx: int, timeout: float, sem: asyncio.Semaphore) -> Dict[str, Any]:
    q = QUESTIONS[idx % len(QUESTIONS)]
    sid = f"tput-{idx}-{uuid.uuid4().hex[:6]}"
    async with sem:
        t0 = time.perf_counter()
        err = None
        ok = False
        answer_len = 0
        try:
            result = await asyncio.wait_for(coordinator.process(q, session_id=sid), timeout=timeout)
            ans = (result or {}).get("answer") or ""
            answer_len = len(ans)
            ok = bool(ans.strip())
            if not ok:
                err = "empty_answer"
        except Exception as e:
            err = f"{type(e).__name__}:{e}"
        lat = time.perf_counter() - t0
        return {
            "idx": idx,
            "question": q,
            "ok": ok,
            "latency_sec": round(lat, 3),
            "answer_len": answer_len,
            "error": err,
            "ts": datetime.now(timezone.utc).isoformat(),
        }


def write_md(path: Path, summary: Dict[str, Any]) -> None:
    s = summary
    lines = [
        "# 并发吞吐评测",
        "",
        f"- 时间: {s['finished_at']}",
        f"- n_requests={s['n_requests']}，concurrency={s['concurrency']}，timeout={s['timeout_sec']}s",
        f"- 问句池: 简单健康问题（短路径倾向）",
        "",
        "## 注意",
        "",
        "- **受 DeepSeek 限流影响**；失败已记录，勿外推为大盘容量",
        "- 单机本地 Coordinator 复用；非压测集群",
        "",
        "## 结果",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| 成功 | {s['success_n']}/{s['n_requests']}（{s['success_rate_pct']}%） |",
        f"| 失败 | {s['fail_n']} |",
        f"| 墙钟 elapsed | {s['wall_sec']}s |",
        f"| QPS（成功/墙钟） | {s['qps_success']} |",
        f"| QPS（全部完成/墙钟） | {s['qps_all']} |",
        f"| 平均延迟（成功） | {s['mean_latency_ok_sec']}s |",
        f"| 平均延迟（全部） | {s['mean_latency_all_sec']}s |",
        f"| p50 / p95 延迟（成功） | {s['p50_ok']} / {s['p95_ok']} |",
        "",
        "## 失败摘要",
        "",
    ]
    fails = s.get("failures") or []
    if not fails:
        lines.append("- （无）")
    else:
        for f in fails[:20]:
            lines.append(f"- idx={f['idx']}: {f.get('error')}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def percentile(xs: List[float], p: float):
    if not xs:
        return None
    ys = sorted(xs)
    k = (len(ys) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return round(ys[f], 3)
    return round(ys[f] + (ys[c] - ys[f]) * (k - f), 3)


async def async_main(args: argparse.Namespace) -> None:
    try:
        from loguru import logger as _logger

        _logger.remove()
        _logger.add(sys.stderr, level="WARNING")
    except Exception:
        pass

    from swarm import SwarmCoordinator

    coordinator = SwarmCoordinator(enable_swarm=True)
    sem = asyncio.Semaphore(args.concurrency)
    print(f"[tput] n={args.n} concurrency={args.concurrency}")

    t_wall0 = time.perf_counter()
    tasks = [run_one(coordinator, i, args.timeout, sem) for i in range(args.n)]
    details = await asyncio.gather(*tasks)
    wall = time.perf_counter() - t_wall0

    ok_rows = [d for d in details if d.get("ok")]
    fail_rows = [d for d in details if not d.get("ok")]
    lat_ok = [d["latency_sec"] for d in ok_rows]
    lat_all = [d["latency_sec"] for d in details]

    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "n_requests": args.n,
        "concurrency": args.concurrency,
        "timeout_sec": args.timeout,
        "wall_sec": round(wall, 2),
        "success_n": len(ok_rows),
        "fail_n": len(fail_rows),
        "success_rate_pct": round(100.0 * len(ok_rows) / args.n, 1) if args.n else 0,
        "qps_success": round(len(ok_rows) / wall, 3) if wall > 0 else 0,
        "qps_all": round(args.n / wall, 3) if wall > 0 else 0,
        "mean_latency_ok_sec": round(sum(lat_ok) / len(lat_ok), 3) if lat_ok else None,
        "mean_latency_all_sec": round(sum(lat_all) / len(lat_all), 3) if lat_all else None,
        "p50_ok": percentile(lat_ok, 50),
        "p95_ok": percentile(lat_ok, 95),
        "failures": [
            {"idx": d["idx"], "error": d.get("error"), "latency_sec": d.get("latency_sec")}
            for d in fail_rows
        ],
        "note": "DeepSeek rate limits may dominate; small-n only",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / "throughput_eval.md"
    js = out_dir / "throughput_eval.json"
    detail = out_dir / "throughput_eval_detail.jsonl"
    with detail.open("w", encoding="utf-8") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    js.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(md, summary)
    print(
        f"[done] ok={summary['success_n']}/{args.n} qps={summary['qps_success']} "
        f"mean_lat={summary['mean_latency_ok_sec']} -> {md}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    args = ap.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
