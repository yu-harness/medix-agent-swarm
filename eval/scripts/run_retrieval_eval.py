#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""纯检索准确率：只调 MedicalKnowledgeBase，不跑 Agent。"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
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

# 与质量评测同模型；命中门禁与 post_opt emb_low 对齐
HIT_THRESH = 0.40
TOP_K = 8


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


def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1) if d else 0.0


def summarize(details: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    n = len(details)
    hit1 = sum(1 for d in details if d.get("hit_at_1"))
    hit3 = sum(1 for d in details if d.get("hit_at_3"))
    hitk = sum(1 for d in details if d.get("hit_at_k"))
    empty = sum(1 for d in details if d.get("n_hits", 0) == 0)
    max_sims = [d["max_sim"] for d in details if isinstance(d.get("max_sim"), (int, float))]
    top1_sims = [d["top1_sim"] for d in details if isinstance(d.get("top1_sim"), (int, float))]

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    by_cat = {}
    for cat in CATEGORIES:
        rows = [d for d in details if d.get("category") == cat]
        cn = len(rows)
        by_cat[cat] = {
            "n": cn,
            "hit_at_1": pct(sum(1 for d in rows if d.get("hit_at_1")), cn),
            "hit_at_3": pct(sum(1 for d in rows if d.get("hit_at_3")), cn),
            "hit_at_k": pct(sum(1 for d in rows if d.get("hit_at_k")), cn),
            "mean_max_sim": mean([d["max_sim"] for d in rows if isinstance(d.get("max_sim"), (int, float))]),
        }
    return {
        "meta": meta,
        "n": n,
        "hit_at_1_pct": pct(hit1, n),
        "hit_at_3_pct": pct(hit3, n),
        "hit_at_k_pct": pct(hitk, n),
        "hit_at_1": hit1,
        "hit_at_3": hit3,
        "hit_at_k": hitk,
        "empty_hits": empty,
        "mean_max_sim": mean(max_sims),
        "mean_top1_sim": mean(top1_sims),
        "by_category": by_cat,
        "thresh": HIT_THRESH,
        "top_k": TOP_K,
    }


def write_md(path: Path, summary: Dict[str, Any]) -> None:
    m = summary["meta"]
    lines = [
        "# 纯检索准确率评测",
        "",
        f"- 时间: {m.get('finished_at')}",
        f"- n={summary['n']}（seed={m.get('seed')}，per_category={m.get('per_category')}，与质量评测同批分层）",
        f"- 路径: MedicalKnowledgeBase.search（不跑 Agent）",
        f"- top_k={summary['top_k']}，命中门禁 emb≥{summary['thresh']}",
        "",
        "## 协议",
        "",
        "1. 对每条问题调用 `MedicalKnowledgeBase.search(question, top_k=8)`",
        "2. 用 `BAAI/bge-small-zh-v1.5` 计算每条 hit.content 与标准答案的 cosine",
        "3. `max_sim` = top_k 内最大相似度；`top1_sim` = 第 1 条",
        "4. hit@1 / hit@3 / hit@k：对应范围内 max_sim ≥ 0.40（对齐 post_opt emb_low）",
        "5. 本轮**未**用 LLM 裁判（省 API；协议以 embedding 为准）",
        "",
        "## 结果",
        "",
        f"| 指标 | 值 |",
        f"|---|---:|",
        f"| hit@1 | {summary['hit_at_1_pct']}%（{summary['hit_at_1']}/{summary['n']}） |",
        f"| hit@3 | {summary['hit_at_3_pct']}%（{summary['hit_at_3']}/{summary['n']}） |",
        f"| hit@{summary['top_k']} | {summary['hit_at_k_pct']}%（{summary['hit_at_k']}/{summary['n']}） |",
        f"| mean max_sim | {summary['mean_max_sim']} |",
        f"| mean top1_sim | {summary['mean_top1_sim']} |",
        f"| 空检索 | {summary['empty_hits']} |",
        "",
        "## 分品类 hit@k",
        "",
        "| category | n | hit@1 | hit@3 | hit@k | mean_max_sim |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cat, row in summary["by_category"].items():
        lines.append(
            f"| {cat} | {row['n']} | {row['hit_at_1']}% | {row['hit_at_3']}% | {row['hit_at_k']}% | {row['mean_max_sim']} |"
        )
    lines.extend(
        [
            "",
            "## 局限",
            "",
            "- 金标答案未必在知识库中；检索相关≠端到端答对",
            "- embedding 门禁与质量裁判同模型，非人工相关度终审",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "eval" / "data" / "benchmark_500.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "results"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-category", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--run-id", default="seed42")
    args = ap.parse_args()

    top_k = args.top_k

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.run_id
    detail_path = out_dir / f"retrieval_eval_detail_{stamp}.jsonl"
    summary_json = out_dir / f"retrieval_eval_{stamp}.json"
    summary_md = out_dir / f"retrieval_eval_{stamp}.md"

    rows = load_jsonl(Path(args.data))
    sample = stratified_sample(rows, args.per_category, args.seed)
    print(f"[retrieval] n={len(sample)} seed={args.seed}")

    from knowledge.milvus_kb import MedicalKnowledgeBase

    kb = MedicalKnowledgeBase()
    emb = EmbeddingScorer()

    started = datetime.now(timezone.utc).isoformat()
    t_all = time.perf_counter()
    details: List[Dict[str, Any]] = []
    if detail_path.exists():
        detail_path.unlink()

    for i, item in enumerate(sample, 1):
        q = item["question"]
        gold = item.get("answer") or ""
        t0 = time.perf_counter()
        try:
            hits = kb.search(q, top_k=top_k)
        except Exception as e:
            hits = []
            err = f"{type(e).__name__}:{e}"
        else:
            err = None
        lat = time.perf_counter() - t0

        sims = []
        for h in hits:
            content = (h.get("content") or "")[:2000]
            sim = emb.cosine(content, gold[:2000]) if gold else 0.0
            sims.append(
                {
                    "sim": round(sim, 4),
                    "milvus_score": round(float(h.get("score") or 0), 4),
                    "content_preview": content[:120],
                }
            )
        max_sim = max((s["sim"] for s in sims), default=0.0)
        top1_sim = sims[0]["sim"] if sims else 0.0
        max3 = max((s["sim"] for s in sims[:3]), default=0.0)

        rec = {
            "id": item["id"],
            "category": item["category"],
            "n_hits": len(hits),
            "top1_sim": round(top1_sim, 4),
            "max_sim": round(max_sim, 4),
            "max_sim_at_3": round(max3, 4),
            "hit_at_1": top1_sim >= HIT_THRESH,
            "hit_at_3": max3 >= HIT_THRESH,
            "hit_at_k": max_sim >= HIT_THRESH,
            "latency_sec": round(lat, 3),
            "error": err,
            "hits": sims[:5],
        }
        details.append(rec)
        with detail_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i % 20 == 0 or i == len(sample):
            print(f"  [{i}/{len(sample)}] hit@k_so_far={pct(sum(1 for d in details if d['hit_at_k']), len(details))}%")

    meta = {
        "seed": args.seed,
        "per_category": args.per_category,
        "data": str(args.data),
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.perf_counter() - t_all, 1),
        "protocol": "emb_cosine_hit_content_vs_gold",
        "hit_thresh": HIT_THRESH,
        "top_k": top_k,
        "llm_judge": False,
    }
    summary = summarize(details, meta)
    summary["top_k"] = top_k
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(summary_md, summary)
    print(
        f"[done] hit@1={summary['hit_at_1_pct']}% hit@3={summary['hit_at_3_pct']}% "
        f"hit@k={summary['hit_at_k_pct']}% -> {summary_md}"
    )


if __name__ == "__main__":
    main()
