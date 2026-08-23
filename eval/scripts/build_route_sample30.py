#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""分层抽 30 条并按用户复核规则再审，写出 sample30 + 回写 200。"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "eval" / "routing" / "route_labels_200.jsonl"
OUT = ROOT / "eval" / "routing" / "route_labels_sample30.jsonl"

CATEGORIES = [
    "health_consult",
    "symptom_diagnosis",
    "disease_knowledge",
    "guideline_retrieval",
]
AGENT_ORDER = {"consultation": 0, "diagnostic": 1, "research": 2}

EMERGENCY_RE = re.compile(
    r"胸痛|胸闷|胸口闷|气短|呼吸困难|冷汗|晕厥|休克|猝死|急救|"
    r"脑梗|中风|偏瘫|喉癌|溃烂|大出血|昏迷",
)
CHRONIC_RE = re.compile(r"一年多|几年|后遗症|康复|慢性|得了\d+年")
PHARM_RE = re.compile(r"药理|是什么\??$|定义|传染|成分|作用机制")
ETIOLOGY_RE = re.compile(r"病因|并发症|鉴别|病理生理")
NONMED_RE = re.compile(r"^(金鱼|猫咪|狗狗|天气|股票)")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sort_agents(agents: List[str]) -> List[str]:
    uniq = []
    for a in agents:
        if a and a not in uniq:
            uniq.append(a)
    return sorted(uniq, key=lambda x: AGENT_ORDER.get(x, 99))


def dedupe_question(q: str) -> str:
    q = (q or "").strip()
    if not q:
        return q
    # half-half exact duplicate
    n = len(q)
    if n >= 8 and n % 2 == 0:
        a, b = q[: n // 2], q[n // 2 :]
        if a == b:
            return a.strip()
    # sentence repeated after ？/。
    m = re.match(r"^(.+[？?。！!])\1+$", q)
    if m:
        return m.group(1)
    # trailing echo without punctuation
    for sep in ("？", "?", "：", ":"):
        if sep in q:
            parts = q.split(sep)
            if len(parts) >= 3 and parts[0] and parts[0] == parts[1]:
                return parts[0] + sep + "".join(parts[2:])
    # substring echo: ABCABC
    for i in range(len(q) // 2, 3, -1):
        if q[:i] == q[i : 2 * i] and len(q) >= 2 * i:
            rest = q[2 * i :]
            if not rest or rest.startswith(q[: min(4, i)]):
                return (q[:i] + rest).strip()
    return q


def priority_score(r: Dict[str, Any]) -> Tuple[int, int, str]:
    score = 0
    if r.get("review_status") == "revised":
        score += 100
    if r.get("expected_mode") == "swarm":
        score += 50
    rule = r.get("prelabel_rule") or ""
    if "EMERGENCY" in rule or "MULTI" in rule:
        score += 30
    q = r.get("question") or ""
    if EMERGENCY_RE.search(q):
        score += 20
    # symptom boundary: single short symptom vs multi
    if r.get("category") == "symptom_diagnosis":
        score += 15
        if len(q) < 40:
            score += 10
    return (-score, 0 if r.get("review_status") == "revised" else 1, r["id"])


def pick_sample(rows: List[Dict[str, Any]], per_cat: int = 8, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    picked: List[Dict[str, Any]] = []
    # disease has 51 due to category remaps; symptom 49 — still ~7-8 each
    quotas = {
        "health_consult": 8,
        "symptom_diagnosis": 8,
        "disease_knowledge": 7,
        "guideline_retrieval": 7,
    }
    for cat in CATEGORIES:
        pool = list(by_cat.get(cat, []))
        pool.sort(key=priority_score)
        # keep top priority, then shuffle within bands for diversity
        top = pool[: max(quotas[cat] * 2, quotas[cat])]
        rng.shuffle(top)
        # force include revised+swarm first
        must = [r for r in pool if r.get("review_status") == "revised" or r.get("expected_mode") == "swarm"]
        chosen_ids = set()
        chosen: List[Dict[str, Any]] = []
        for r in must:
            if len(chosen) >= quotas[cat]:
                break
            chosen.append(r)
            chosen_ids.add(r["id"])
        for r in top + pool:
            if len(chosen) >= quotas[cat]:
                break
            if r["id"] in chosen_ids:
                continue
            chosen.append(r)
            chosen_ids.add(r["id"])
        picked.extend(chosen[: quotas[cat]])
    return picked


def review_one(r: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(r)
    q0 = out.get("question") or ""
    q = dedupe_question(q0)
    changed = False
    notes_extra: List[str] = []

    if q != q0:
        out["question"] = q
        changed = True
        notes_extra.append("题面去重")

    # 用户已人工 revised：只做题面去重，不改 expected_*
    if (r.get("review_status") or "") == "revised":
        out["review_status"] = "revised"
        n = out.get("notes") or ""
        if changed and "抽审：" not in n:
            out["notes"] = (n + "；抽审：题面去重").strip("；") if n else "抽审：题面去重"
        elif "抽审确认" not in n and "抽审：" not in n:
            out["notes"] = (n + "；抽审确认保留原修订").strip("；") if n else "抽审确认保留原修订"
        out["sample30"] = True
        return out

    cat = out.get("category")
    mode = out.get("expected_mode")
    agents = list(out.get("expected_agents") or [])
    rule = out.get("prelabel_rule") or ""

    # non-medical replace (already done for 042, keep guard)
    if NONMED_RE.search(q.strip()):
        out["question"] = "金银花是什么？"
        q = out["question"]
        out["category"] = "disease_knowledge"
        cat = "disease_knowledge"
        mode = "single"
        agents = ["consultation"]
        rule = "R_DK_DEFINE"
        changed = True
        notes_extra.append("非医疗替换")

    # pharmacological misfiled as symptom
    if cat == "symptom_diagnosis" and re.search(r"药理|是什么药|药物作用", q):
        out["category"] = "disease_knowledge"
        cat = "disease_knowledge"
        mode = "single"
        agents = ["consultation"]
        rule = "R_DK_DEFINE"
        changed = True
        notes_extra.append("分类冲突：药理归 disease_knowledge")

    # etiology → diagnostic
    if cat == "disease_knowledge" and ETIOLOGY_RE.search(q) and "consultation" in agents and "diagnostic" not in agents:
        agents = ["diagnostic"]
        rule = "R_DK_CLINICAL"
        mode = "single"
        changed = True
        notes_extra.append("病因/并发症走 diagnostic")

    # chronic rehab mislabeled emergency
    if EMERGENCY_RE.search(q) and CHRONIC_RE.search(q) and "EMERGENCY" in rule:
        mode = "swarm"
        agents = ["consultation", "research"]
        rule = "R_HC_CHRONIC_REHAB"
        changed = True
        notes_extra.append("后遗症/慢病非急症")

    # drug safety with 晕厥词 ≠ current emergency
    if re.search(r"可以吃|能不能吃|会致使|会不会", q) and re.search(r"晕厥|血压", q):
        if "EMERGENCY" in rule or mode == "swarm":
            mode = "single"
            agents = ["consultation"]
            rule = "R_HC_DRUG_SAFETY"
            changed = True
            notes_extra.append("用药咨询非急救")

    # cerebrovascular + neuro signs → swarm consult+diag
    if re.search(r"脑|血管阻塞|毛细血管阻塞", q) and re.search(r"嘴角|抽|麻|跳", q):
        if mode == "single" and agents == ["consultation"]:
            mode = "swarm"
            agents = ["consultation", "diagnostic"]
            changed = True
            notes_extra.append("脑血管+神经体征需诊断协作")

    # single symptom shouldn't be swarm (user rule on 029/045 pattern)
    if cat == "symptom_diagnosis" and mode == "swarm":
        # count symptom-ish chunks
        chunks = re.split(r"[，,、；;和及以及]", q)
        chunks = [c for c in chunks if c.strip()]
        has_emergency = bool(EMERGENCY_RE.search(q))
        multi_systems = len(chunks) >= 3 or (
            len(chunks) >= 2 and bool(re.search(r"而且|还|同时|伴", q))
        )
        if not has_emergency and not multi_systems:
            mode = "single"
            agents = ["diagnostic"]
            if rule == "R_SD_MULTI":
                rule = "R_SD_DIAG_ONLY"
            changed = True
            notes_extra.append("单一症状不必 swarm")
        elif not has_emergency and multi_systems and len(q) < 25 and "口苦" in q and "口臭" in q:
            # mild dual local oral symptoms — still can stay swarm or single; user kept 030 as swarm
            pass

    # pure guideline stay research single
    if cat == "guideline_retrieval" and re.search(r"指南|权威资料|知识库|文档|检索", q):
        if mode != "single" or agents != ["research"]:
            # only if clearly pure retrieval
            if not re.search(r"我该怎么|个人|饮食建议给我", q):
                mode = "single"
                agents = ["research"]
                rule = "R_GR_PURE"
                # don't mark changed if already correct
                if out.get("expected_mode") != mode or out.get("expected_agents") != agents:
                    changed = True
                    notes_extra.append("纯指南检索")

    # false emergency: 肝硬化严重 with labs but not acute collapse — keep swarm if severe disease ask treatment
    if "肝硬化严重" in q and "EMERGENCY" in rule:
        # chronic severe disease management: consult+diag ok as swarm but not "急救"
        rule = "R_HC_MULTI"
        mode = "swarm"
        agents = ["consultation", "diagnostic"]
        changed = True
        notes_extra.append("重症慢病非超急症词误用，改 MULTI")

    # 脑梗糖尿病高血压 — multi comorbidity, swarm ok
    if re.search(r"脑梗.*糖尿病|糖尿病.*高血压", q) and "怎么办" in q:
        mode = "swarm"
        agents = ["consultation", "diagnostic"]
        if "EMERGENCY" in rule:
            rule = "R_HC_MULTI"
            changed = True
            notes_extra.append("共病管理非纯急救")

    agents = sort_agents(agents)
    out["expected_mode"] = mode
    out["expected_agents"] = agents
    out["prelabel_rule"] = rule

    # already revised by user: keep labels unless we found more issues; status stays revised/confirmed
    prev = r.get("review_status") or "pending"
    if changed:
        out["review_status"] = "revised"
        base = (out.get("notes") or "").rstrip("；;")
        extra = "；".join(notes_extra)
        out["notes"] = (base + "；抽审：" + extra) if base else ("抽审：" + extra)
    else:
        out["review_status"] = "confirmed" if prev in ("pending", "confirmed") else prev
        if prev == "revised":
            out["review_status"] = "revised"
        n = out.get("notes") or ""
        if "抽审确认" not in n:
            out["notes"] = (n + "；抽审确认无改").strip("；") if n else "抽审确认无改"

    out["sample30"] = True
    return out


def main() -> None:
    rows = load_jsonl(LABELS)
    sample = pick_sample(rows, seed=42)
    reviewed = [review_one(r) for r in sample]
    write_jsonl(OUT, reviewed)

    # sync back into 200
    by_id = {r["id"]: r for r in reviewed}
    updated = []
    for r in rows:
        if r["id"] in by_id:
            u = deepcopy(by_id[r["id"]])
            u.pop("sample30", None)
            updated.append(u)
        else:
            updated.append(r)
    write_jsonl(LABELS, updated)

    from collections import Counter

    print("n", len(reviewed))
    print("by_cat", Counter(r["category"] for r in reviewed))
    print("status", Counter(r["review_status"] for r in reviewed))
    print("mode", Counter(r["expected_mode"] for r in reviewed))
    print("ids", [r["id"] for r in reviewed])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
