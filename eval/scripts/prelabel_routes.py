#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""确定性规则预标路由金标（200 条分层抽样，与 agent_eval seed=42 对齐）。"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval" / "scripts"))

from run_agent_eval import CATEGORIES, load_jsonl, stratified_sample  # noqa: E402

AGENT_ORDER = ("consultation", "diagnostic", "research")

EMERGENCY_KW = (
    "胸痛",
    "胸口闷",
    "胸闷",
    "呼吸困难",
    "喘不过气",
    "冷汗",
    "冒冷汗",
    "大汗",
    "晕厥",
    "昏迷",
    "抽搐",
    "咯血",
    "呕血",
    "黑便",
    "急性",
    "急救",
    "120",
    "心梗",
    "脑梗",
    "中风",
    "偏瘫",
    "喉癌晚期",
    "肝硬化严重",
)

MULTI_SYMPTOM_SEP = re.compile(r"[，,、；;＋+/]|同时|还有|伴随|并且|以及|而且|一边.+一边")

# 长词优先，避免「疼痛」同时命中「疼」「痛」
SYMPTOM_TOKENS = (
    "头痛",
    "胸闷",
    "气短",
    "发烧",
    "发热",
    "恶心",
    "呕吐",
    "头晕",
    "乏力",
    "出汗",
    "麻木",
    "肿胀",
    "红肿",
    "口苦",
    "口臭",
    "腹泻",
    "便秘",
    "失眠",
    "心悸",
    "视力模糊",
    "出血",
    "咳嗽",
    "咳",
    "疼痛",
    "痛",
    "疼",
    "痒",
)

LIFESTYLE_KW = (
    "饮食",
    "吃",
    "喝",
    "运动",
    "锻炼",
    "睡眠",
    "睡觉",
    "生活",
    "护理",
    "注意什么",
    "注意事项",
    "能吃",
    "可以吃",
    "日常",
)

GUIDE_LOOKUP_KW = (
    "指南",
    "知识库",
    "文档",
    "权威资料",
    "检索文档",
    "引用文档",
    "规定或建议",
    "知识要点",
)

RESEARCH_KW = (
    "最新研究",
    "研究进展",
    "循证",
    "文献",
    "临床指南",
    "诊疗指南",
    "专家共识",
)

DEFINITION_KW = (
    "是什么",
    "什么是",
    "定义",
    "会传染",
    "传染吗",
    "传染么",
    "属于什么",
    "成分",
    "用法用量",
    "药理作用",
)

CODING_KW = (
    "icd",
    "ICD",
    "编码",
    "疾病分类",
    "诊断编码",
)

DIAG_REASON_KW = (
    "是怎么回事",
    "什么原因",
    "啥原因",
    "是不是",
    "会不会是",
    "鉴别",
    "可能是什么病",
    "怎么诊断",
    "如何诊断",
    "症状是",
    "有哪些症状",
    "的症状",
    "的病因",
    "的并发症",
)


def sort_agents(agents: List[str]) -> List[str]:
    return sorted(set(agents), key=lambda a: AGENT_ORDER.index(a))


def has_any(text: str, kws: Tuple[str, ...]) -> bool:
    return any(k in text for k in kws)


def count_symptom_hits(text: str) -> int:
    """按最长匹配计数，避免子串重复计。"""
    covered = [False] * len(text)
    hits = 0
    for t in SYMPTOM_TOKENS:
        start = 0
        while True:
            i = text.find(t, start)
            if i < 0:
                break
            if not any(covered[i : i + len(t)]):
                for j in range(i, i + len(t)):
                    covered[j] = True
                hits += 1
            start = i + 1
    return hits


def is_multi_complaint(text: str) -> bool:
    hits = count_symptom_hits(text)
    if hits >= 3:
        return True
    if hits >= 2 and MULTI_SYMPTOM_SEP.search(text):
        return True
    if hits >= 2 and ("还有" in text or "并且" in text or "同时" in text or "而且" in text):
        return True
    return False


def prelabel(question: str, category: str) -> Tuple[str, List[str], str, str]:
    """返回 expected_mode, expected_agents, prelabel_rule, notes。"""
    q = question or ""
    cat = category or ""

    # ---- health_consult ----
    if cat == "health_consult":
        if has_any(q, EMERGENCY_KW):
            return (
                "swarm",
                sort_agents(["consultation", "diagnostic"]),
                "R_HC_EMERGENCY",
                "含急救/急危重症线索，需诊断评估+咨询建议",
            )
        if is_multi_complaint(q) or ("合并" in q and has_any(q, ("高血压", "糖尿病", "脑梗", "心血管"))):
            return (
                "swarm",
                sort_agents(["consultation", "diagnostic"]),
                "R_HC_MULTI",
                "多症状或明显共病，单咨询不足",
            )
        if has_any(q, ("如何治疗", "怎样治疗", "怎么治", "如何救治", "治疗方案")) and has_any(
            q, ("高血压", "糖尿病", "癫痫", "肝硬化", "癌")
        ):
            return (
                "swarm",
                sort_agents(["consultation", "research"]),
                "R_HC_TREAT_GUIDE",
                "疾病治疗方案，宜指南+生活建议",
            )
        return (
            "single",
            ["consultation"],
            "R_HC_DEFAULT",
            "常规健康咨询/生活建议，单 Consultation",
        )

    # ---- symptom_diagnosis ----
    if cat == "symptom_diagnosis":
        if has_any(q, EMERGENCY_KW) or ("胸口" in q and ("疼" in q or "闷" in q)):
            return (
                "swarm",
                sort_agents(["consultation", "diagnostic"]),
                "R_SD_EMERGENCY",
                "胸痛气短等急症线索，至少含 diagnostic，并给处置建议",
            )
        if is_multi_complaint(q):
            return (
                "swarm",
                sort_agents(["consultation", "diagnostic"]),
                "R_SD_MULTI",
                "复杂多主诉，鉴别+处置建议",
            )
        if has_any(q, DIAG_REASON_KW) or (
            ("原因" in q or "是不是" in q or "会不会" in q) and count_symptom_hits(q) >= 1
        ):
            return (
                "single",
                ["diagnostic"],
                "R_SD_DIAG_ONLY",
                "侧重病因/鉴别/是否某病，单 Diagnostic",
            )
        if has_any(q, LIFESTYLE_KW) and not has_any(q, DIAG_REASON_KW):
            return (
                "single",
                ["consultation"],
                "R_SD_AS_CONSULT",
                "题面偏生活/用药咨询，按单咨询",
            )
        return (
            "single",
            ["consultation"],
            "R_SD_DEFAULT",
            "单一症状常规咨询，默认 Consultation",
        )

    # ---- disease_knowledge ----
    if cat == "disease_knowledge":
        if has_any(q, CODING_KW):
            return (
                "single",
                ["diagnostic"],
                "R_DK_CODING",
                "疾病编码/分类，Diagnostic",
            )
        if has_any(q, RESEARCH_KW) or "最新" in q:
            return (
                "single",
                ["research"],
                "R_DK_RESEARCH",
                "研究进展/最新证据，Research",
            )
        if has_any(q, ("如何治疗", "怎么治疗", "怎样治疗", "怎么治", "治疗方案")):
            if has_any(q, LIFESTYLE_KW):
                return (
                    "swarm",
                    sort_agents(["consultation", "research"]),
                    "R_DK_TREAT_LIFE",
                    "治疗+生活管理，Research+Consultation",
                )
            return (
                "single",
                ["consultation"],
                "R_DK_TREAT",
                "一般治疗科普，Consultation",
            )
        if has_any(q, DEFINITION_KW) or q.strip().endswith("?") or q.strip().endswith("？"):
            # 「XX是什么」类
            if re.search(r"(是什么|什么是|属于|传染|药理|用法|成分|解析)", q):
                return (
                    "single",
                    ["consultation"],
                    "R_DK_DEFINE",
                    "定义/科普/是否传染，Consultation",
                )
        if has_any(q, ("的症状", "的病因", "的并发症", "病因是什么", "症状")):
            return (
                "single",
                ["diagnostic"],
                "R_DK_CLINICAL",
                "症状/病因/并发症偏临床结构化，Diagnostic",
            )
        return (
            "single",
            ["consultation"],
            "R_DK_DEFAULT",
            "疾病科普默认 Consultation",
        )

    # ---- guideline_retrieval ----
    if cat == "guideline_retrieval":
        pure_guide = has_any(q, GUIDE_LOOKUP_KW) or cat == "guideline_retrieval"
        wants_life = has_any(
            q,
            (
                "生活建议",
                "日常管理",
                "患者饮食",
                "我该怎么吃",
                "给我建议",
                "个人方案",
            ),
        )
        # 指南章节本身含运动/膳食，但题面是「检索文档」→ 仍算纯查指南
        if wants_life and pure_guide:
            return (
                "swarm",
                sort_agents(["consultation", "research"]),
                "R_GR_GUIDE_LIFE",
                "既要指南又要个人生活建议",
            )
        return (
            "single",
            ["research"],
            "R_GR_PURE",
            "纯指南/知识库检索，单 Research",
        )

    return (
        "single",
        ["consultation"],
        "R_FALLBACK",
        "未知 category，回退 Consultation",
    )


def build_records(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        mode, agents, rule, notes = prelabel(r["question"], r["category"])
        out.append(
            {
                "id": r["id"],
                "question": r["question"],
                "category": r["category"],
                "expected_mode": mode,
                "expected_agents": agents,
                "prelabel_rule": rule,
                "notes": notes,
                "review_status": "pending",
            }
        )
    return out


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    mode_c = Counter(r["expected_mode"] for r in records)
    combo_c = Counter(",".join(r["expected_agents"]) for r in records)
    rule_c = Counter(r["prelabel_rule"] for r in records)
    by_cat = {}
    for cat in CATEGORIES:
        sub = [r for r in records if r["category"] == cat]
        by_cat[cat] = {
            "n": len(sub),
            "mode": dict(Counter(r["expected_mode"] for r in sub)),
            "combo": dict(Counter(",".join(r["expected_agents"]) for r in sub)),
        }
    return {
        "n": len(records),
        "mode": dict(mode_c),
        "combo": dict(combo_c),
        "rules": dict(rule_c),
        "by_category": by_cat,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data",
        default=str(ROOT / "eval" / "data" / "benchmark_500.jsonl"),
    )
    p.add_argument(
        "--align-detail",
        default=str(ROOT / "eval" / "results" / "agent_eval_detail_full200.jsonl"),
        help="若存在则按其中 id 对齐同一批 200",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--per-category", type=int, default=50)
    p.add_argument(
        "--out",
        default=str(ROOT / "eval" / "routing" / "route_labels_200.jsonl"),
    )
    p.add_argument(
        "--stats-out",
        default=str(ROOT / "eval" / "routing" / "prelabel_stats.json"),
    )
    args = p.parse_args()

    rows = load_jsonl(Path(args.data))
    sample = stratified_sample(rows, args.per_category, args.seed)

    align_path = Path(args.align_detail)
    if align_path.exists():
        detail_ids = []
        with align_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    detail_ids.append(json.loads(line)["id"])
        by_id = {r["id"]: r for r in rows}
        missing = [i for i in detail_ids if i not in by_id]
        if missing:
            raise SystemExit(f"align detail ids missing in benchmark: {missing[:5]}")
        sample_ids = {r["id"] for r in sample}
        if set(detail_ids) != sample_ids:
            raise SystemExit(
                "align detail ids != stratified_sample(seed=%s); abort" % args.seed
            )
        # 保持与 detail 文件相同顺序，便于对照
        sample = [by_id[i] for i in detail_ids]

    records = build_records(sample)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = summarize(records)
    stats["seed"] = args.seed
    stats["per_category"] = args.per_category
    stats["aligned_detail"] = str(align_path) if align_path.exists() else None
    Path(args.stats_out).write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[wrote] {out_path}")
    print(f"[wrote] {args.stats_out}")


if __name__ == "__main__":
    main()
