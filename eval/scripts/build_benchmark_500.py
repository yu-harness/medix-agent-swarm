# -*- coding: utf-8 -*-
"""Build 500-item Chinese medical text benchmark (4 categories x 125)."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
DATA = ROOT / "data"
DOCS = ROOT.parents[0] / "knowledge" / "data" / "documents"

TARGET_PER_CAT = 125
MAX_ANSWER_CHARS = 800
SAMPLE_PER_CAT = 8

CATEGORIES = (
    "health_consult",
    "symptom_diagnosis",
    "disease_knowledge",
    "guideline_retrieval",
)

SYMPTOM_KW = (
    "症状", "疼痛", "头疼", "头痛", "发烧", "发热", "咳嗽", "腹泻", "呕吐",
    "恶心", "头晕", "胸闷", "气短", "腹痛", "皮疹", "出血", "肿", "痒",
    "怎么回事", "什么病", "可能是", "诊断", "难受", "不舒服", "疼", "痛",
    "咳", "拉肚子", "发烧", "畏寒", "盗汗", "心慌",
)
CONSULT_KW = (
    "饮食", "能吃", "可以吃", "注意事项", "怎么保养", "锻炼", "运动", "睡眠",
    "怀孕", "哺乳", "用药", "怎么吃", "用法", "用量", "副作用", "忌口",
    "预防", "护理", "生活", "保健", "减肥", "瑜伽", "补", "党参",
)
KNOWLEDGE_KW = (
    "什么是", "病因", "发病机制", "并发症", "预后", "分型", "分期", "定义",
    "临床表现", "鉴别诊断", "治疗原则", "病理", "流行病学", "属于", "介绍",
)


def clean(s: str) -> str:
    s = (s or "").strip().replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def truncate(ans: str) -> tuple[str, bool]:
    ans = clean(ans)
    if len(ans) <= MAX_ANSWER_CHARS:
        return ans, False
    cut = ans[:MAX_ANSWER_CHARS]
    for sep in ("。", "；", "！", "？", ".", ";", "\n"):
        i = cut.rfind(sep)
        if i >= int(MAX_ANSWER_CHARS * 0.6):
            return cut[: i + 1], True
    return cut, True


def norm_q(q: str) -> str:
    q = clean(q).lower()
    q = re.sub(r"[，。！？、；：,.!?;:\s\"'“”‘’（）()【】\[\]《》<>]", "", q)
    return q


def bigrams(s: str) -> set[str]:
    s = norm_q(s)
    if len(s) < 2:
        return {s} if s else set()
    return {s[i : i + 2] for i in range(len(s) - 1)}


def similar(a: str, b: str, thr: float = 0.85) -> bool:
    if norm_q(a) == norm_q(b):
        return True
    ba, bb = bigrams(a), bigrams(b)
    if not ba or not bb:
        return False
    inter = len(ba & bb)
    return inter / max(1, min(len(ba), len(bb))) >= thr


def similar_pair(a: dict, b: dict) -> bool:
    """Guideline-style templated questions: also compare answers."""
    if a.get("category") == "guideline_retrieval" or b.get("category") == "guideline_retrieval":
        if norm_q(a["question"]) == norm_q(b["question"]):
            return True
        # same answer span -> dup
        if norm_q(a.get("answer", "")) == norm_q(b.get("answer", "")):
            return True
        # soft question similarity only if answers also close
        if similar(a["question"], b["question"], 0.92) and similar(a.get("answer", ""), b.get("answer", ""), 0.8):
            return True
        return False
    return similar(a["question"], b["question"])


def make_id(category: str, idx: int) -> str:
    return f"{category}_{idx:03d}"


def guess_category(question: str, preferred: str | None = None) -> str:
    if preferred:
        return preferred
    q = question
    sk = sum(1 for k in SYMPTOM_KW if k in q)
    ck = sum(1 for k in CONSULT_KW if k in q)
    kk = sum(1 for k in KNOWLEDGE_KW if k in q)
    if kk >= sk and kk >= ck and kk > 0:
        return "disease_knowledge"
    if sk >= ck and sk > 0:
        return "symptom_diagnosis"
    if ck > 0:
        return "health_consult"
    if any(x in q for x in ("怎么办", "如何", "怎样", "可否", "是否")):
        return "health_consult"
    return "symptom_diagnosis"


def valid_pair(q: str, a: str) -> bool:
    q, a = clean(q), clean(a)
    if len(q) < 4 or len(a) < 8:
        return False
    if len(q) > 500:
        return False
    return True


def load_cmedqa2() -> list[dict]:
    qpath = RAW / "cmedqa2" / "question" / "question.csv"
    apath = RAW / "cmedqa2" / "answer" / "answer.csv"
    cpath = RAW / "cmedqa2" / "train_candidates" / "train_candidates.txt"
    questions = {}
    with qpath.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            questions[row["question_id"]] = clean(row["content"])
    answers = {}
    with apath.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            answers[row["ans_id"]] = (row["question_id"], clean(row["content"]))
    seen_q = set()
    out = []
    with cpath.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            qid = row["question_id"]
            if qid in seen_q:
                continue
            seen_q.add(qid)
            aid = row["pos_ans_id"]
            if aid not in answers or qid not in questions:
                continue
            q = questions[qid]
            a = answers[aid][1]
            if not valid_pair(q, a):
                continue
            cat = guess_category(q)
            out.append(
                {
                    "question": q,
                    "answer": a,
                    "category": cat,
                    "source": "cMedQA2",
                    "source_id": f"q{qid}_a{aid}",
                    "notes": "",
                }
            )
            if len(out) >= 800:
                break
    return out


def load_webmedqa() -> list[dict]:
    out = []
    for split, name in (("valid", "medQA.valid.txt"), ("test", "medQA.test.txt")):
        path = RAW / "webmedqa" / split / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                dept, label, qid, q, a = parts[0], parts[1], parts[2], parts[3], parts[4]
                if label != "1":
                    continue
                q, a = clean(q), clean(a)
                if not valid_pair(q, a):
                    continue
                cat = guess_category(q)
                out.append(
                    {
                        "question": q,
                        "answer": a,
                        "category": cat,
                        "source": "webMedQA",
                        "source_id": f"{split}_{qid}",
                        "notes": f"department={dept}",
                    }
                )
                if len(out) >= 600:
                    return out
    return out


def load_dialogue_sample() -> list[dict]:
    path = RAW / "chinese_medical_dialogue" / "sample_neike.csv"
    out = []
    with path.open(encoding="gbk", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            q = clean(row.get("ask") or row.get("question") or "")
            title = clean(row.get("title") or "")
            if title and title not in q:
                q = f"{title}：{q}" if q else title
            a = clean(row.get("answer") or "")
            if not valid_pair(q, a):
                continue
            cat = guess_category(q)
            out.append(
                {
                    "question": q,
                    "answer": a,
                    "category": cat,
                    "source": "Chinese-medical-dialogue-data",
                    "source_id": f"sample_neike_{i}",
                    "notes": f"department={clean(row.get('department') or '')}",
                }
            )
            if len(out) >= 800:
                break
    return out


def load_hf_medical() -> list[dict]:
    out = []
    for name in ("test_zh_0.json", "valid_zh_0.json"):
        path = RAW / "shibing624_medical" / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                q = clean(obj.get("instruction") or "")
                inp = clean(obj.get("input") or "")
                if inp:
                    q = f"{q}\n{inp}" if q else inp
                a = clean(obj.get("output") or "")
                if not valid_pair(q, a):
                    continue
                cat = guess_category(q)
                # drug usage / encyclopedia-like -> knowledge or consult
                if any(k in q for k in ("用法", "用量", "什么是", "病因", "适应症")):
                    cat = "disease_knowledge" if "什么是" in q or "病因" in q else "health_consult"
                out.append(
                    {
                        "question": q,
                        "answer": a,
                        "category": cat,
                        "source": "shibing624/medical",
                        "source_id": f"{name}_{i}",
                        "notes": "finetune_zh_split",
                    }
                )
    return out


def load_encyclopedia() -> list[dict]:
    path = RAW / "huatuo_encyclopedia" / "train_sample.jsonl"
    out = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            qs = obj.get("questions") or []
            ans = obj.get("answers")
            if isinstance(qs, list) and qs:
                q0 = qs[0]
                q = clean(q0[0] if isinstance(q0, list) else str(q0))
            else:
                continue
            if isinstance(ans, list):
                a = clean(ans[0] if ans else "")
            else:
                a = clean(ans or "")
            if not valid_pair(q, a):
                continue
            out.append(
                {
                    "question": q,
                    "answer": a,
                    "category": "disease_knowledge",
                    "source": "FreedomIntelligence/huatuo_encyclopedia_qa",
                    "source_id": f"enc_{i}",
                    "notes": "",
                }
            )
    return out


def _section_blocks(text: str) -> list[tuple[str, str]]:
    """Split docs into (heading, body) blocks."""
    lines = text.splitlines()
    blocks: list[tuple[str, list[str]]] = []
    cur_h = "概述"
    cur: list[str] = []
    heading_re = re.compile(
        r"^(#{1,6}\s+.+|[一二三四五六七八九十]+[、.\s].{0,40}|"
        r"\d+(\.\d+)*[、.．]\s*.{0,40}|"
        r"【.+】|"
        r".{2,40}（核心内容）|"
        r".{2,30}（Chest Pain|Dyspnea）.|"
        r"诊断标准|治疗目标|诊断流程|血压测量|高血压分级|"
        r"糖尿病诊断|糖尿病前期|综合管理|生活方式|药物治疗|"
        r"危险分层|随访|筛查|危险征象|可能原因|紧急处理|风险评估)"
    )
    for line in lines:
        s = line.strip()
        if not s:
            if cur:
                blocks.append((cur_h, cur))
                cur = []
            continue
        if heading_re.match(s) and len(s) <= 80:
            if cur:
                blocks.append((cur_h, cur))
            cur_h = re.sub(r"^#+\s*", "", s)
            cur = []
        else:
            cur.append(s)
    if cur:
        blocks.append((cur_h, cur))
    return [(h, "\n".join(b)) for h, b in blocks if len("\n".join(b)) >= 30]


def _chunk_body(body: str, size: int = 220) -> list[str]:
    body = clean(body.replace("\n", " "))
    if len(body) <= size:
        return [body] if len(body) >= 40 else []
    chunks = []
    start = 0
    while start < len(body):
        end = min(len(body), start + size)
        if end < len(body):
            for sep in ("。", "；", "！", " "):
                j = body.rfind(sep, start + size // 2, end)
                if j != -1:
                    end = j + 1
                    break
        piece = clean(body[start:end])
        if len(piece) >= 40:
            chunks.append(piece)
        start = end
    return chunks


def load_guideline_from_docs() -> list[dict]:
    gl_templates = [
        "请根据知识库/指南文档，说明「{topic}」的要点。",
        "检索文档中关于「{topic}」的规定或建议是什么？",
        "权威资料对「{topic}」给出了哪些核心要求？",
        "关于「{topic}」，指南或知识文档怎么写？",
        "请引用文档片段回答：「{topic}」应如何把握？",
    ]
    other_templates = {
        "health_consult": "关于「{topic}」，日常应注意什么？",
        "symptom_diagnosis": "出现与「{topic}」相关的情况时，文档提示关注哪些要点？",
        "disease_knowledge": "「{topic}」的知识要点是什么？",
    }
    out: list[dict] = []
    if not DOCS.exists():
        return out
    for fp in sorted(DOCS.glob("*.txt")):
        text = fp.read_text(encoding="utf-8")
        is_guideline = "guideline" in fp.name or "指南" in text[:300]
        blocks = _section_blocks(text)
        for bi, (heading, body) in enumerate(blocks):
            topic = heading.strip("【】# ").strip()
            if len(topic) < 2:
                topic = fp.stem
            chunks = _chunk_body(body)
            for ci, chunk in enumerate(chunks):
                if not valid_pair(topic, chunk):
                    continue
                q_gl = gl_templates[(bi + ci) % len(gl_templates)].format(topic=topic)
                out.append(
                    {
                        "question": q_gl,
                        "answer": chunk,
                        "category": "guideline_retrieval",
                        "source": f"knowledge/documents/{fp.name}",
                        "source_id": f"{fp.stem}_g{bi}_{ci}",
                        "notes": (
                            "generated_from_local_doc_span"
                            if is_guideline
                            else "generated_from_local_doc_span;non_guideline_doc_for_retrieval"
                        ),
                    }
                )
                if is_guideline:
                    continue
                if "lifestyle" in fp.name:
                    cat = "health_consult"
                elif "symptoms" in fp.name:
                    cat = "symptom_diagnosis"
                else:
                    cat = "disease_knowledge"
                q = other_templates[cat].format(topic=topic)
                if not valid_pair(q, chunk):
                    continue
                out.append(
                    {
                        "question": q,
                        "answer": chunk,
                        "category": cat,
                        "source": f"knowledge/documents/{fp.name}",
                        "source_id": f"{fp.stem}_b{bi}_{ci}",
                        "notes": "generated_from_local_doc_span",
                    }
                )
    return out


def select_with_quotas(
    pools: list[dict],
    category: str,
    n: int,
    used_norms: set[str],
    used_items: list[dict],
    quotas: list[tuple[str, int]],
) -> list[dict]:
    """Select up to n items preferring source prefixes in quotas order."""
    selected: list[dict] = []
    remaining = list(pools)

    def take_from(prefix: str | None, k: int) -> None:
        nonlocal remaining
        got = 0
        keep = []
        for item in remaining:
            if got >= k or len(selected) >= n:
                keep.append(item)
                continue
            if item["category"] != category:
                keep.append(item)
                continue
            if prefix is not None and not (
                item["source"] == prefix or item["source"].startswith(prefix)
            ):
                keep.append(item)
                continue
            qn = norm_q(item["question"])
            if not qn or qn in used_norms:
                keep.append(item)
                continue
            ans, trunc = truncate(item["answer"])
            q = clean(item["question"])
            if category == "guideline_retrieval":
                cue = clean(ans)[:24]
                if cue and cue not in q:
                    q = f"{q}（线索：{cue}…）"
            cand = {"question": q, "answer": ans, "category": category}
            if any(similar_pair(cand, u) for u in selected[-40:]):
                keep.append(item)
                continue
            if any(similar_pair(cand, u) for u in used_items[-80:]):
                keep.append(item)
                continue
            if not ans:
                keep.append(item)
                continue
            notes = item.get("notes") or ""
            if trunc:
                notes = (notes + "; " if notes else "") + f"answer_truncated_to_{MAX_ANSWER_CHARS}"
            selected.append(
                {
                    "question": q,
                    "answer": ans,
                    "category": category,
                    "source": item["source"],
                    "source_id": str(item["source_id"]),
                    "notes": notes,
                }
            )
            used_norms.add(norm_q(q))
            got += 1
        remaining = keep

    for prefix, k in quotas:
        if len(selected) >= n:
            break
        take_from(prefix, k)
    # fill leftover from any source in category
    if len(selected) < n:
        take_from(None, n - len(selected))
    return selected[:n]


def rebalance_fill(need: int, category: str, pools: list[dict], used_norms: set[str], used_items: list[dict]) -> list[dict]:
    """Fill remaining slots by remapping near-category items."""
    remap_pref = {
        "health_consult": ["health_consult", "symptom_diagnosis", "disease_knowledge"],
        "symptom_diagnosis": ["symptom_diagnosis", "health_consult", "disease_knowledge"],
        "disease_knowledge": ["disease_knowledge", "health_consult", "symptom_diagnosis"],
        "guideline_retrieval": ["guideline_retrieval"],
    }
    out = []
    for pref in remap_pref[category]:
        for item in pools:
            if len(out) >= need:
                return out
            if item["category"] != pref:
                continue
            qn = norm_q(item["question"])
            if not qn or qn in used_norms:
                continue
            if category == "guideline_retrieval" and "knowledge/documents" not in item["source"]:
                continue
            ans, trunc = truncate(item["answer"])
            q = clean(item["question"])
            if category == "guideline_retrieval":
                cue = clean(ans)[:24]
                if cue and cue not in q:
                    q = f"{q}（线索：{cue}…）"
            cand = {"question": q, "answer": ans, "category": category}
            if any(similar_pair(cand, u) for u in used_items + out):
                continue
            notes = item.get("notes") or ""
            if trunc:
                notes = (notes + "; " if notes else "") + f"answer_truncated_to_{MAX_ANSWER_CHARS}"
            if item["category"] != category:
                notes = (notes + "; " if notes else "") + f"remapped_from_{item['category']}"
            rec = {
                "question": q,
                "answer": ans,
                "category": category,
                "source": item["source"],
                "source_id": str(item["source_id"]),
                "notes": notes,
            }
            out.append(rec)
            used_norms.add(norm_q(q))
    return out


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    print("Loading sources...")
    pools: list[dict] = []
    for loader in (
        load_cmedqa2,
        load_webmedqa,
        load_dialogue_sample,
        load_hf_medical,
        load_encyclopedia,
        load_guideline_from_docs,
    ):
        part = loader()
        print(f"  {loader.__name__}: {len(part)}")
        pools.extend(part)

    by_cat = defaultdict(int)
    for p in pools:
        by_cat[p["category"]] += 1
    print("Pool by category:", dict(by_cat))

    # Prefer public sources first within each category
    source_priority = {
        "cMedQA2": 0,
        "webMedQA": 1,
        "Chinese-medical-dialogue-data": 2,
        "shibing624/medical": 3,
        "FreedomIntelligence/huatuo_encyclopedia_qa": 4,
    }

    def sort_key_for(cat: str):
        def sort_key(x: dict):
            src = x["source"]
            if cat == "disease_knowledge":
                if src.startswith("FreedomIntelligence/huatuo_encyclopedia_qa"):
                    sp = 0
                elif src.startswith("shibing624/medical"):
                    sp = 1
                elif src.startswith("knowledge/") and "icd10" in src:
                    sp = 2
                elif src.startswith("cMedQA2"):
                    sp = 5
                else:
                    sp = 4
            elif cat == "guideline_retrieval":
                if "guideline" in src:
                    sp = 0
                elif src.startswith("knowledge/"):
                    sp = 1
                else:
                    sp = 9
            else:
                sp = 10
                for k, v in source_priority.items():
                    if src.startswith(k) or src == k:
                        sp = v
                        break
                if src.startswith("knowledge/"):
                    sp = 6
            return (sp, len(x["question"]))
        return sort_key

    used_norms: set[str] = set()
    final: list[dict] = []
    stats = {}
    quotas_map = {
        "health_consult": [
            ("cMedQA2", 55),
            ("Chinese-medical-dialogue-data", 30),
            ("webMedQA", 25),
            ("shibing624/medical", 15),
        ],
        "symptom_diagnosis": [
            ("cMedQA2", 55),
            ("Chinese-medical-dialogue-data", 30),
            ("webMedQA", 25),
            ("shibing624/medical", 15),
        ],
        "disease_knowledge": [
            ("FreedomIntelligence/huatuo_encyclopedia_qa", 90),
            ("shibing624/medical", 25),
            ("knowledge/documents/", 10),
        ],
        "guideline_retrieval": [
            ("knowledge/documents/20_guideline_hypertension.txt", 55),
            ("knowledge/documents/21_guideline_diabetes.txt", 50),
            ("knowledge/documents/", 20),
        ],
    }
    for cat in CATEGORIES:
        pools_sorted = sorted(pools, key=sort_key_for(cat))
        sel = select_with_quotas(
            pools_sorted, cat, TARGET_PER_CAT, used_norms, final, quotas_map[cat]
        )
        if len(sel) < TARGET_PER_CAT:
            extra = rebalance_fill(
                TARGET_PER_CAT - len(sel), cat, pools_sorted, used_norms, final + sel
            )
            sel.extend(extra)
        for i, item in enumerate(sel[:TARGET_PER_CAT], start=1):
            rec = {
                "id": make_id(cat, i),
                "question": item["question"],
                "answer": item["answer"],
                "category": cat,
                "source": item["source"],
                "source_id": item["source_id"],
                "notes": item.get("notes") or "",
            }
            final.append(rec)
        stats[cat] = len(sel[:TARGET_PER_CAT])
        print(f"Selected {cat}: {stats[cat]}")

    assert len(final) == 500, len(final)
    for cat in CATEGORIES:
        assert stats[cat] == TARGET_PER_CAT, (cat, stats[cat])

    out_full = DATA / "benchmark_500.jsonl"
    with out_full.open("w", encoding="utf-8") as f:
        for rec in final:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # samples: ~8 per category
    samples = []
    for cat in CATEGORIES:
        items = [r for r in final if r["category"] == cat]
        # diversify sources
        picked = []
        seen_src = set()
        for r in items:
            key = r["source"]
            if key not in seen_src or len(picked) < SAMPLE_PER_CAT // 2:
                picked.append(r)
                seen_src.add(key)
            if len(picked) >= SAMPLE_PER_CAT:
                break
        while len(picked) < SAMPLE_PER_CAT and len(picked) < len(items):
            for r in items:
                if r not in picked:
                    picked.append(r)
                if len(picked) >= SAMPLE_PER_CAT:
                    break
        samples.extend(picked[:SAMPLE_PER_CAT])

    out_sample = DATA / "benchmark_samples.jsonl"
    with out_sample.open("w", encoding="utf-8") as f:
        for rec in samples:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # fingerprint for reproducibility note
    h = hashlib.sha256(out_full.read_bytes()).hexdigest()[:16]
    meta = {
        "total": len(final),
        "per_category": stats,
        "samples": len(samples),
        "sha256_16": h,
        "sources_used": sorted({r["source"] for r in final}),
    }
    (DATA / "benchmark_build_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Wrote", out_full, len(final))
    print("Wrote", out_sample, len(samples))
    print(meta)


if __name__ == "__main__":
    main()
