#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""智能路由本地评测：只跑 Lead 分解 + collapse，不跑完整 Swarm 答题。"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

AGENT_SHORT = {
    "consultation_agent": "consultation",
    "diagnostic_agent": "diagnostic",
    "research_agent": "research",
    "consultation": "consultation",
    "diagnostic": "diagnostic",
    "research": "research",
}
AGENT_ORDER = {"consultation": 0, "diagnostic": 1, "research": 2}


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


def normalize_agents(raw: List[Any]) -> List[str]:
    out: List[str] = []
    for a in raw or []:
        if not a:
            continue
        s = AGENT_SHORT.get(str(a).strip(), None)
        if s and s not in out:
            out.append(s)
    out.sort(key=lambda x: AGENT_ORDER.get(x, 99))
    return out


def mode_from_subtasks(n: int) -> str:
    if n >= 2:
        return "swarm"
    return "single"


def summarize(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(details)
    mode_ok = sum(1 for d in details if d.get("mode_match"))
    combo_ok = sum(1 for d in details if d.get("combo_match"))
    by_cat: Dict[str, Dict[str, Any]] = {}
    for cat in sorted({d.get("category") or "?" for d in details}):
        rows = [d for d in details if d.get("category") == cat]
        cn = len(rows)
        by_cat[cat] = {
            "n": cn,
            "mode_acc": round(sum(1 for d in rows if d.get("mode_match")) / cn, 4) if cn else 0,
            "combo_acc": round(sum(1 for d in rows if d.get("combo_match")) / cn, 4) if cn else 0,
            "mode_ok": sum(1 for d in rows if d.get("mode_match")),
            "combo_ok": sum(1 for d in rows if d.get("combo_match")),
        }
    conf = Counter()
    for d in details:
        if d.get("error"):
            conf["error"] += 1
        elif not d.get("mode_match") and not d.get("combo_match"):
            conf["both_wrong"] += 1
        elif not d.get("mode_match"):
            conf["mode_only"] += 1
        elif not d.get("combo_match"):
            conf["combo_only"] += 1
        else:
            conf["both_ok"] += 1
    pred_mode = Counter(d.get("predicted_mode") for d in details)
    exp_mode = Counter(d.get("expected_mode") for d in details)
    return {
        "n": n,
        "mode_accuracy": round(mode_ok / n, 4) if n else 0,
        "combo_exact_match": round(combo_ok / n, 4) if n else 0,
        "mode_ok": mode_ok,
        "combo_ok": combo_ok,
        "by_category": by_cat,
        "confusion": dict(conf),
        "pred_mode_dist": dict(pred_mode),
        "expected_mode_dist": dict(exp_mode),
    }


def write_md(path: Path, meta: Dict[str, Any], summary: Dict[str, Any]) -> None:
    nature = meta.get("label_nature") or "规则预标，非人工终审"
    lines = [
        f"# 路由评测报告 — {meta.get('run_id', '')}",
        "",
        f"> **金标性质：{nature}**",
        "",
        "## Meta",
        "",
        f"- n: {summary.get('n')}",
        f"- labels: `{meta.get('labels')}`",
        f"- concurrency: {meta.get('concurrency')}",
        f"- timeout_sec: {meta.get('timeout_sec')}",
        f"- elapsed_sec: {meta.get('elapsed_sec')}",
        f"- started_at: {meta.get('started_at')}",
        f"- finished_at: {meta.get('finished_at')}",
        "",
        "## 指标",
        "",
        f"- **模式准确率** (predicted_mode vs expected_mode): "
        f"**{summary['mode_accuracy']:.1%}** ({summary['mode_ok']}/{summary['n']})",
        f"- **组合完全匹配率** (agents 集合): "
        f"**{summary['combo_exact_match']:.1%}** ({summary['combo_ok']}/{summary['n']})",
        "",
        "## 分品类",
        "",
        "| category | n | mode_acc | combo_acc |",
        "|---|---:|---:|---:|",
    ]
    for cat, s in (summary.get("by_category") or {}).items():
        lines.append(
            f"| {cat} | {s['n']} | {s['mode_acc']:.1%} ({s['mode_ok']}/{s['n']}) "
            f"| {s['combo_acc']:.1%} ({s['combo_ok']}/{s['n']}) |"
        )
    lines += [
        "",
        "## 分布",
        "",
        f"- expected_mode: `{summary.get('expected_mode_dist')}`",
        f"- predicted_mode: `{summary.get('pred_mode_dist')}`",
        f"- confusion: `{summary.get('confusion')}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


async def route_one(
    item: Dict[str, Any],
    coordinator: Any,
    timeout: float,
    sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    q = item.get("question") or ""
    expected_mode = item.get("expected_mode")
    expected_agents = normalize_agents(item.get("expected_agents") or [])
    t0 = time.perf_counter()
    err = None
    predicted_mode = None
    predicted_agents: List[str] = []
    raw_subtasks: List[Dict[str, Any]] = []
    collapsed: List[Dict[str, Any]] = []
    reason = ""

    async with sem:
        try:
            assessment = await asyncio.wait_for(
                coordinator.lead_agent.assess_and_decompose(q, None),
                timeout=timeout,
            )
            raw_subtasks = assessment.get("subtasks") or []
            collapsed = coordinator._collapse_subtasks(list(raw_subtasks), q)
            reason = assessment.get("reason") or ""
            predicted_agents = normalize_agents(
                [t.get("assigned_agent") for t in collapsed]
            )
            predicted_mode = mode_from_subtasks(len(collapsed))
            if not collapsed:
                predicted_mode = "single"
                predicted_agents = ["consultation"]
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            predicted_mode = "single"
            predicted_agents = []

    lat = round(time.perf_counter() - t0, 3)
    mode_match = (predicted_mode == expected_mode) and not err
    combo_match = (set(predicted_agents) == set(expected_agents)) and not err
    return {
        "id": item["id"],
        "category": item.get("category"),
        "question": q,
        "expected_mode": expected_mode,
        "expected_agents": expected_agents,
        "predicted_mode": predicted_mode,
        "predicted_agents": predicted_agents,
        "mode_match": mode_match,
        "combo_match": combo_match,
        "raw_n_subtasks": len(raw_subtasks),
        "collapsed_n_subtasks": len(collapsed),
        "raw_agents": [t.get("assigned_agent") for t in raw_subtasks],
        "collapsed_agents": [t.get("assigned_agent") for t in collapsed],
        "prelabel_rule": item.get("prelabel_rule"),
        "review_status": item.get("review_status"),
        "reason": reason,
        "latency_sec": lat,
        "error": err,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


async def async_main(args: argparse.Namespace) -> None:
    try:
        from loguru import logger as _logger

        _logger.remove()
        _logger.add(sys.stderr, level="WARNING")
    except Exception:
        pass

    labels_path = Path(args.labels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or "baseline"
    detail_path = out_dir / f"route_eval_detail_{run_id}.jsonl"
    if args.resume_detail:
        detail_path = Path(args.resume_detail)

    if args.subset:
        labels_path = Path(args.subset)
    rows = load_jsonl(labels_path)
    if args.ids:
        allow = {x.strip() for x in args.ids.split(",") if x.strip()}
        rows = [r for r in rows if r["id"] in allow]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    done = load_done_ids(detail_path) if args.resume or args.resume_detail else set()
    todo = [r for r in rows if r["id"] not in done]
    print(
        f"[route_eval] n={len(rows)} done={len(done)} todo={len(todo)} "
        f"run_id={run_id} detail={detail_path}",
        flush=True,
    )

    from swarm import SwarmCoordinator

    coordinator = SwarmCoordinator(enable_swarm=True)
    sem = asyncio.Semaphore(args.concurrency)

    meta = {
        "run_id": run_id,
        "labels": str(labels_path),
        "label_nature": args.label_nature
        or ("按用户复核规则完成的抽审子集" if args.subset else "规则预标，非人工终审"),
        "n_target": len(rows),
        "concurrency": args.concurrency,
        "timeout_sec": args.timeout,
        "detail_path": str(detail_path),
        "route_only": True,
        "subset": args.subset or "",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    finished = 0
    t_all = time.perf_counter()
    lock = asyncio.Lock()

    async def worker(item: Dict[str, Any]) -> None:
        nonlocal finished
        try:
            rec = await route_one(item, coordinator, args.timeout, sem)
        except Exception:
            rec = {
                "id": item["id"],
                "category": item.get("category"),
                "question": item.get("question"),
                "expected_mode": item.get("expected_mode"),
                "expected_agents": normalize_agents(item.get("expected_agents") or []),
                "predicted_mode": None,
                "predicted_agents": [],
                "mode_match": False,
                "combo_match": False,
                "error": traceback.format_exc()[-500:],
                "latency_sec": None,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        async with lock:
            append_jsonl(detail_path, rec)
            finished += 1
            i = len(done) + finished
            print(
                f"[{i}/{len(rows)}] {rec['id']} mode={rec.get('predicted_mode')}"
                f"/{rec.get('expected_mode')} "
                f"agents={rec.get('predicted_agents')}/{rec.get('expected_agents')} "
                f"m={'Y' if rec.get('mode_match') else 'N'} "
                f"c={'Y' if rec.get('combo_match') else 'N'} "
                f"lat={rec.get('latency_sec')} err={rec.get('error')}",
                flush=True,
            )

    if args.concurrency <= 1:
        for it in todo:
            await worker(it)
    else:
        batch = max(args.concurrency * 2, 4)
        for i in range(0, len(todo), batch):
            chunk = todo[i : i + batch]
            await asyncio.gather(*(worker(it) for it in chunk))

    all_details = load_jsonl(detail_path)
    sample_ids = {r["id"] for r in rows}
    latest: Dict[str, Dict[str, Any]] = {}
    for d in all_details:
        if d.get("id") in sample_ids:
            latest[d["id"]] = d
    details = list(latest.values())

    summary = summarize(details)
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    meta["elapsed_sec"] = round(time.perf_counter() - t_all, 1)
    payload = {"meta": meta, "summary": summary}

    summary_json = out_dir / f"route_eval_{run_id}.json"
    summary_md = out_dir / f"route_eval_{run_id}.md"
    summary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_md(summary_md, meta, summary)

    print(
        f"[done] mode_acc={summary['mode_accuracy']:.1%} "
        f"combo={summary['combo_exact_match']:.1%} "
        f"n={summary['n']}",
        flush=True,
    )
    print(f"[files] {detail_path}", flush=True)
    print(f"[files] {summary_json}", flush=True)
    print(f"[files] {summary_md}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--labels",
        default=str(ROOT / "eval" / "routing" / "route_labels_200.jsonl"),
    )
    p.add_argument(
        "--out-dir",
        default=str(ROOT / "eval" / "results"),
    )
    p.add_argument("--run-id", default="baseline")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=float, default=45.0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--subset", default="", help="jsonl of ids/rows to evaluate")
    p.add_argument("--ids", default="", help="comma-separated ids")
    p.add_argument("--label-nature", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-detail", default="")
    return p


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
