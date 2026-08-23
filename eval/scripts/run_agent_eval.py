#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""完整 Agent Swarm 评测：分层抽样 + embedding + LLM 裁判。"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
import traceback
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CATEGORIES = [
    "health_consult",
    "symptom_diagnosis",
    "disease_knowledge",
    "guideline_retrieval",
]
EMB_HIGH = 0.75
EMB_LOW = 0.40  # 优化后：略降门禁，减少合理长答被误杀


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratified_sample(
    rows: List[Dict[str, Any]],
    per_cat: int,
    seed: int,
) -> List[Dict[str, Any]]:
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


def load_done_ids(detail_path: Path) -> set:
    done = set()
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


def resolve_bge_local() -> Optional[str]:
    import os

    roots = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub" / "models--BAAI--bge-small-zh-v1.5" / "snapshots")
    roots.append(
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--BAAI--bge-small-zh-v1.5"
        / "snapshots"
    )
    for local in roots:
        if not local.exists():
            continue
        snaps = sorted(
            [p for p in local.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if snaps:
            return str(snaps[0])
    return None


class EmbeddingScorer:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import numpy as np

        self.np = np
        name = resolve_bge_local() or "BAAI/bge-small-zh-v1.5"
        print(f"[emb] loading {name}")
        self.model = SentenceTransformer(name, device="cpu")

    def cosine(self, a: str, b: str) -> float:
        va, vb = self.model.encode([a or "", b or ""], normalize_embeddings=True)
        return float(self.np.dot(va, vb))


JUDGE_PROMPT = """你是医疗问答评测裁判。判断「系统回答」是否覆盖「标准答案」的关键医疗要点。

判定原则（轻度校准，不要放水到全对）：
- correct：覆盖标准答案中的核心医疗要点（结论方向、主要处理思路、关键风险/就医提示等）即可；语义一致即可，措辞可不同
- 不要求药名、中成药商品名、偏方细节、剂量或检查清单与金标逐字一致；写出同类措施/原则即算覆盖
- 系统回答比金标更严谨、更全面、或补充风险提示 → 不因此判 partial
- 金标极短而系统答得更细 → 只要核心方向一致判 correct
- partial：仅当遗漏金标的核心结论方向（如该分型用药却完全未提分型；该检查路径却完全未提检查）
- incorrect：答非所问、空泛敷衍、关键医学事实严重错误，或与金标核心结论明显矛盾

只输出一行 JSON，不要其它文字：
{{"verdict":"correct|partial|incorrect","reason":"不超过40字"}}

问题：{question}

标准答案：
{gold}

系统回答：
{pred}
"""


async def llm_judge(
    client,
    question: str,
    gold: str,
    pred: str,
) -> Tuple[str, str]:
    prompt = JUDGE_PROMPT.format(
        question=question[:800],
        gold=(gold or "")[:1200],
        pred=(pred or "")[:2000],
    )
    try:
        text = await client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200,
        )
        text = (text or "").strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            obj = json.loads(text[start : end + 1])
            v = str(obj.get("verdict", "incorrect")).lower().strip()
            if v not in ("correct", "partial", "incorrect"):
                v = "incorrect"
            return v, str(obj.get("reason", ""))[:80]
    except Exception as e:
        return "incorrect", f"judge_error:{type(e).__name__}"
    return "incorrect", "judge_parse_fail"


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


async def run_one(
    item: Dict[str, Any],
    emb: EmbeddingScorer,
    llm_client,
    timeout_sec: float,
    sem: asyncio.Semaphore,
    coordinator=None,
) -> Dict[str, Any]:
    qid = item["id"]
    question = item["question"]
    gold = item.get("answer", "")
    category = item["category"]
    session_id = f"eval-{qid}-{uuid.uuid4().hex[:8]}"

    pred = ""
    err = None
    latency = None
    mode = None
    attempt = 0

    async def _call():
        if coordinator is not None:
            return await coordinator.process(question, session_id=session_id)
        from swarm import process_with_swarm

        return await process_with_swarm(question, session_id=session_id)

    async with sem:
        for attempt in (1, 2):
            t0 = time.perf_counter()
            try:
                result = await asyncio.wait_for(_call(), timeout=timeout_sec)
                latency = time.perf_counter() - t0
                pred = extract_answer(result)
                if isinstance(result, dict):
                    mode = result.get("mode") or (
                        "swarm" if result.get("swarm_enabled") else "single_agent"
                    )
                if pred:
                    err = None
                    break
                err = "empty_answer"
            except asyncio.TimeoutError:
                latency = time.perf_counter() - t0
                err = "timeout"
                pred = ""
            except Exception as e:
                latency = time.perf_counter() - t0
                err = f"{type(e).__name__}:{e}"
                pred = ""
            if attempt == 1:
                await asyncio.sleep(1.0)

    emb_score = None
    llm_verdict = None
    llm_reason = ""
    rule = ""

    if err in ("timeout",) or (err and err.startswith("empty")) or not pred:
        if err is None:
            err = "empty_answer"
        verdict = "incorrect"
        rule = "empty_or_error"
        is_correct = False
    else:
        try:
            emb_score = emb.cosine(pred, gold)
        except Exception as e:
            emb_score = 0.0
            err = f"emb_error:{type(e).__name__}"

        if emb_score < EMB_LOW:
            verdict = "incorrect"
            rule = f"emb_low<{EMB_LOW}"
            is_correct = False
            llm_verdict = None
            llm_reason = "skipped_due_to_low_embedding"
        else:
            llm_verdict, llm_reason = await llm_judge(llm_client, question, gold, pred)
            verdict = llm_verdict
            rule = f"emb>={EMB_LOW}_then_llm"
            is_correct = llm_verdict == "correct"

    return {
        "id": qid,
        "category": category,
        "question": question,
        "gold_answer": gold,
        "pred_answer": pred[:4000],
        "embedding_score": emb_score,
        "llm_verdict": llm_verdict,
        "llm_reason": llm_reason,
        "verdict": verdict,
        "is_correct": is_correct,
        "is_partial": verdict == "partial",
        "judge_rule": rule,
        "latency_sec": round(latency, 3) if latency is not None else None,
        "mode": mode,
        "error": err,
        "attempts": attempt,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def percentile(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] * (c - k) + ys[c] * (k - f)


def summarize(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(details)
    correct = sum(1 for d in details if d.get("is_correct"))
    partial = sum(1 for d in details if d.get("is_partial"))
    latencies = [d["latency_sec"] for d in details if isinstance(d.get("latency_sec"), (int, float))]
    by_cat: Dict[str, Any] = {}
    for cat in CATEGORIES:
        sub = [d for d in details if d.get("category") == cat]
        c = sum(1 for d in sub if d.get("is_correct"))
        p = sum(1 for d in sub if d.get("is_partial"))
        by_cat[cat] = {
            "n": len(sub),
            "correct": c,
            "partial": p,
            "exact_accuracy": round(c / len(sub), 4) if sub else None,
            "partial_rate": round(p / len(sub), 4) if sub else None,
        }

    err_counter = Counter()
    for d in details:
        if d.get("is_correct"):
            continue
        reason = d.get("llm_reason") or d.get("judge_rule") or d.get("error") or "unknown"
        err_counter[reason] += 1

    return {
        "n": n,
        "correct": correct,
        "partial": partial,
        "exact_accuracy": round(correct / n, 4) if n else None,
        "partial_rate": round(partial / n, 4) if n else None,
        "latency": {
            "mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
            "p50": round(percentile(latencies, 50), 3) if latencies else None,
            "p95": round(percentile(latencies, 95), 3) if latencies else None,
            "n": len(latencies),
        },
        "by_category": by_cat,
        "error_top": err_counter.most_common(10),
        "verdict_counts": dict(Counter(d.get("verdict") for d in details)),
        "mode_counts": dict(Counter(d.get("mode") or "unknown" for d in details)),
    }


def write_summary_md(path: Path, meta: Dict[str, Any], summary: Dict[str, Any]) -> None:
    lines = [
        "# Agent Swarm 评测报告",
        "",
        f"- 生成时间: {meta.get('finished_at')}",
        f"- 样本: 从 benchmark_500.jsonl **分层抽样 {meta.get('n_target')} 条**（四类各 {meta.get('per_category')}），seed={meta.get('seed')}",
        f"- 系统路径: 真实调用 `process_with_swarm`（完整 Agent）",
        f"- 判定: embedding（BAAI/bge-small-zh-v1.5）+ DeepSeek LLM 终判",
        f"- 并发 semaphore={meta.get('semaphore')}, 单条超时 {meta.get('timeout_sec')}s, 失败重试 1 次",
        "",
        "## 判定细则",
        "",
        f"- embedding < {EMB_LOW} → **直接判错**（不再交 LLM）",
        f"- embedding ≥ {EMB_LOW} → 交 DeepSeek 输出 correct / partial / incorrect",
        f"- 主准确率（exact）**仅计 correct**；partial 单独汇报",
        "- 空答、异常、超时 → 计错",
        "- **裁判校准（相对基线）**：correct=覆盖金标关键医疗要点即可，不要求药名/偏方逐字；emb门禁由0.45降至0.40",
        "",
        "## 总体结果",
        "",
        f"- 完成条数: {summary['n']}",
        f"- exact 正确率: **{summary['exact_accuracy']}** ({summary['correct']}/{summary['n']})",
        f"- partial 比率: {summary['partial_rate']} ({summary['partial']}/{summary['n']})",
        f"- 延迟 mean / p50 / p95 (秒): "
        f"{summary['latency']['mean']} / {summary['latency']['p50']} / {summary['latency']['p95']}",
        "",
        "## 分类型 exact 正确率",
        "",
        "| category | n | correct | exact_acc | partial | partial_rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cat in CATEGORIES:
        c = summary["by_category"][cat]
        lines.append(
            f"| {cat} | {c['n']} | {c['correct']} | {c['exact_accuracy']} | {c['partial']} | {c['partial_rate']} |"
        )
    lines.extend(
        [
            "",
            "## 路由/模式分布（观测，非准确率评测）",
            "",
            f"- mode_counts: `{summary.get('mode_counts')}`",
            "",
            "## 典型错因 Top",
            "",
        ]
    )
    for reason, cnt in summary.get("error_top", [])[:8]:
        lines.append(f"- ({cnt}) {reason}")
    lines.extend(
        [
            "",
            "## 未评指标",
            "",
            "- **智能路由准确率**：本次未评（无路由金标）",
            "- **多轮对话准确率**：本次未评（无多轮标注）",
            "",
            "## 局限",
            "",
            "- 子集 200 条，非全量 500",
            "- LLM 裁判存在偏差；指南类标准答案偏文档片段，系统长答易因表述差异被判 partial/incorrect",
            "- embedding 低阈值直接判错，可能漏掉「表述差异大但语义正确」的答案",
            "",
        ]
    )
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
    stamp = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = out_dir / f"agent_eval_detail_{stamp}.jsonl"
    if args.resume_detail:
        detail_path = Path(args.resume_detail)

    rows = load_jsonl(data_path)
    sample = stratified_sample(rows, args.per_category, args.seed)
    if args.categories:
        allow = {c.strip() for c in args.categories.split(",") if c.strip()}
        sample = [r for r in sample if r.get("category") in allow]
    if args.limit and args.limit > 0:
        sample = sample[: args.limit]

    done = load_done_ids(detail_path) if args.resume or args.resume_detail else set()
    todo = [r for r in sample if r["id"] not in done]
    print(
        f"[eval] sample={len(sample)} done={len(done)} todo={len(todo)} "
        f"detail={detail_path}"
    )

    from core.llm_client import LLMClient
    from swarm import SwarmCoordinator

    emb = EmbeddingScorer()
    llm_client = LLMClient()
    # 复用单一 Coordinator（逻辑等同 process_with_swarm，避免每条重建三 Agent/Mem0）
    coordinator = SwarmCoordinator(enable_swarm=True)
    sem = asyncio.Semaphore(args.semaphore)

    meta = {
        "seed": args.seed,
        "per_category": args.per_category,
        "n_target": len(sample),
        "semaphore": args.semaphore,
        "timeout_sec": args.timeout,
        "data": str(data_path),
        "detail_path": str(detail_path),
        "emb_high": EMB_HIGH,
        "emb_low": EMB_LOW,
        "reuse_coordinator": True,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    finished = 0
    total = len(todo)
    t_all = time.perf_counter()

    async def worker(item: Dict[str, Any]) -> None:
        nonlocal finished
        try:
            rec = await run_one(
                item, emb, llm_client, args.timeout, sem, coordinator=coordinator
            )
        except Exception:
            rec = {
                "id": item["id"],
                "category": item["category"],
                "question": item.get("question"),
                "gold_answer": item.get("answer"),
                "pred_answer": "",
                "embedding_score": None,
                "llm_verdict": None,
                "llm_reason": "",
                "verdict": "incorrect",
                "is_correct": False,
                "is_partial": False,
                "judge_rule": "fatal_error",
                "latency_sec": None,
                "mode": None,
                "error": traceback.format_exc()[-500:],
                "attempts": 0,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        append_jsonl(detail_path, rec)
        finished += 1
        ok = "Y" if rec.get("is_correct") else "N"
        print(
            f"[{len(done) + finished}/{len(sample)}] {rec['id']} cat={rec['category']} "
            f"ok={ok} verdict={rec.get('verdict')} "
            f"emb={rec.get('embedding_score')} lat={rec.get('latency_sec')} "
            f"err={rec.get('error')}",
            flush=True,
        )
        if finished % 10 == 0:
            import gc

            gc.collect()

    # 串行或小并发；默认按 semaphore 控制
    if args.semaphore <= 1:
        for it in todo:
            await worker(it)
    else:
        batch_size = max(args.semaphore, 2)
        for i in range(0, len(todo), batch_size):
            batch = todo[i : i + batch_size]
            await asyncio.gather(*(worker(it) for it in batch))
            import gc

            gc.collect()

    all_details = load_jsonl(detail_path)
    # 只汇总本次 sample 内的 id
    sample_ids = {r["id"] for r in sample}
    details = [d for d in all_details if d.get("id") in sample_ids]
    # 同 id 保留最后一条
    latest: Dict[str, Dict[str, Any]] = {}
    for d in details:
        latest[d["id"]] = d
    details = list(latest.values())

    summary = summarize(details)
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    meta["elapsed_sec"] = round(time.perf_counter() - t_all, 1)
    payload = {"meta": meta, "summary": summary}

    summary_json = out_dir / f"agent_eval_summary_{stamp}.json"
    summary_md = out_dir / f"agent_eval_summary_{stamp}.md"
    summary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary_md(summary_md, meta, summary)
    # 固定最新指针
    (out_dir / "latest_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_summary_md(out_dir / "latest_summary.md", meta, summary)
    (out_dir / "latest_detail.path").write_text(str(detail_path), encoding="utf-8")

    print("[done]", json.dumps(summary, ensure_ascii=False))
    print(f"[files] {detail_path}")
    print(f"[files] {summary_json}")
    print(f"[files] {summary_md}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        default=str(ROOT / "eval" / "data" / "benchmark_500.jsonl"),
    )
    p.add_argument(
        "--out-dir",
        default=str(ROOT / "eval" / "results"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-category", type=int, default=50)
    p.add_argument("--limit", type=int, default=0, help="冒烟：只跑前 N 条")
    p.add_argument("--semaphore", type=int, default=2)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--run-id", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-detail", default="", help="续跑指定 detail jsonl")
    p.add_argument(
        "--categories",
        default="",
        help="逗号分隔类别过滤，如 health_consult,symptom_diagnosis",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
