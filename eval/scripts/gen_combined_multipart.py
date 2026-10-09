#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成"双主题复合问题"评测样本：从 benchmark_500 两两拼接。

用途：Swarm 只在子任务 >=2 时激活（实测单一提问仅 8% 激活）。
复合问题（两个独立子问）能稳定触发多 Agent 分解，用于把 A/B 激活样本
从个位数扩充到 20+。

拼接方式：q1 + "另外，" + q2（尽量跨类别），gold 也对应拼接。
种子固定，结果可复现。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = [
    "health_consult",
    "symptom_diagnosis",
    "disease_knowledge",
    "guideline_retrieval",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--data", default=str(ROOT / "eval/data/benchmark_500.jsonl"))
    p.add_argument("--out", default=str(ROOT / "eval/data/multipart_combined_30.jsonl"))
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    by_cat = {c: [r for r in rows if r["category"] == c] for c in CATEGORIES}
    rng = random.Random(args.seed)

    items = []
    used = set()
    guard = 0
    while len(items) < args.n and guard < args.n * 20:
        guard += 1
        c1, c2 = rng.sample(CATEGORIES, 2)
        pool1 = [r for r in by_cat[c1] if r["id"] not in used]
        pool2 = [r for r in by_cat[c2] if r["id"] not in used]
        if not pool1 or not pool2:
            continue
        r1, r2 = rng.choice(pool1), rng.choice(pool2)
        used.update([r1["id"], r2["id"]])
        items.append({
            "id": f"cb-{len(items):03d}",
            "category": r1["category"],
            "source": "combined_multipart",
            "question": f"{r1['question']}\n另外，{r2['question']}",
            "answer": f"{r1['answer']}\n\n【第二问】{r2['answer']}",
        })

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"generated {len(items)} combined multipart items -> {out.name}")
    print("example:", items[0]["question"][:120].replace("\n", " | "))


if __name__ == "__main__":
    main()
