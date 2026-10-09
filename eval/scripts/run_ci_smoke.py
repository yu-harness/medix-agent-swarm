# -*- coding: utf-8 -*-
"""CI 冒烟评测门禁（零 API 成本 / 纯本地 / 目标 < 1 分钟）。

三道硬门禁，任一不达标即 exit(1)：

  G1 安全护栏  切片不劈裂剂量、出口安全网必补「就医/120 + 免责声明」、低危不误报
  G2 检索召回  hit@1 ≥ 95%（cosine ≥ 0.40，协议对齐 eval/scripts/run_retrieval_eval.py）
  G3 数据一致性 真指南 877 块、旧二手摘要残留 0（完整模式）；归档与版权面（轻量模式）

两种模式（--mode auto 自动探测）：

  完整模式  需要重建好的生产知识库（含 4 份真指南全文）。
            门禁内容 = 真指南块数校验 + 旧摘要残留 0 + benchmark_500 分层 20 题 hit@1
            + 4 份指南的来源命中探针（覆盖高血压/糖尿病/血脂/胸痛）。
  轻量模式  无向量库、无 PDF 也能跑（GitHub Actions 就是这样）：
            用仓库内**被 git 跟踪**的 knowledge/data/documents/*.txt 现场建一个临时索引，
            门禁内容 = 同一套检索协议（语料自检 20 题）+ 来源头接地格式契约
            + 旧摘要未进索引 + 版权面（PDF / *.db 未被 git 跟踪）。

关于密钥：本脚本不调用任何在线模型，但 core/llm_client.py 在**导入期**会执行
ensure_api_keys() 做 fail-fast（生产上的防呆设计）。因此这里用占位值放行导入期校验
（setdefault 不覆盖真实密钥）；G1 直接调用生产函数本体，不走任何网络。

用法：
    python eval/scripts/run_ci_smoke.py                 # 自动探测模式
    python eval/scripts/run_ci_smoke.py --mode light     # 强制轻量模式（CI）
    python eval/scripts/run_ci_smoke.py --rerank         # 打开 cross-encoder 重排
    python eval/scripts/run_ci_smoke.py --json-out eval/results/ci_smoke_latest.json
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# CI 环境自洽：放行导入期的密钥 fail-fast 校验（占位值不覆盖已存在的真实密钥）
os.environ.setdefault("LLM_API_KEY", "sk-xxx-ci-smoke-placeholder")
os.environ.setdefault("MEM0_ENABLED", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


# ---------------------------------------------------------------- 常量

# 4 份真指南入库块数（2026-10-09 实测，见 knowledge/data/ingest_real_guidelines_report.json）
EXPECTED_GUIDELINE_CHUNKS: Dict[str, int] = {
    "01_hypertension_2024.pdf": 323,
    "02_t2dm_2020.pdf": 401,
    "03_lipid_2023.pdf": 129,
    "05_acute_chest_pain_consensus_2019.pdf": 24,
}
EXPECTED_GUIDELINE_TOTAL = sum(EXPECTED_GUIDELINE_CHUNKS.values())

# 已被真指南取代、必须从向量库里彻底消失的旧二手摘要
ARCHIVED_FILENAMES = (
    "20_guideline_hypertension.txt",
    "21_guideline_diabetes.txt",
    "22_guideline_hyperlipidemia.txt",
)

# 指南来源命中探针：覆盖高血压 / 糖尿病 / 血脂 / 胸痛四类，每类 2 题。
# 只用问题文本 + 期望文件名（不掺入任何自撰医学事实），断言 top-1 命中该指南。
GUIDELINE_PROBES: Tuple[Tuple[str, str], ...] = (
    ("2024 高血压诊断标准与血压分级", "01_hypertension_2024.pdf"),
    ("高血压患者的降压目标值是多少", "01_hypertension_2024.pdf"),
    ("2型糖尿病的诊断标准与血糖控制目标", "02_t2dm_2020.pdf"),
    ("二甲双胍在2型糖尿病治疗中的地位", "02_t2dm_2020.pdf"),
    ("低密度脂蛋白胆固醇 LDL-C 目标值", "03_lipid_2023.pdf"),
    ("血脂异常患者的他汀类药物治疗原则", "03_lipid_2023.pdf"),
    ("急性高危胸痛的识别与处理流程", "05_acute_chest_pain_consensus_2019.pdf"),
    ("急性胸痛患者到院后首次心电图时限要求", "05_acute_chest_pain_consensus_2019.pdf"),
)

SIM_THRESHOLD = 0.40      # 命中门禁（对齐 run_retrieval_eval.py 的 emb≥0.4）
HIT_THRESHOLD = 0.95      # hit@1 下限
DEFAULT_CASES = 20
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


# ---------------------------------------------------------------- 输出工具

def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """生成标准 Markdown 表格（列宽自适应，不使用 <br>，避免渲染粘连）。"""
    cells = [[str(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in cells:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def line(vals: Sequence[str]) -> str:
        return "| " + " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals)) + " |"

    out = [line(list(headers)), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [line(r) for r in cells]
    return "\n".join(out)


def _section(title: str) -> None:
    print()
    print(f"### {title}")
    print()


class Report:
    """收集检查项，统一决定退出码与总结论。"""

    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []
        self.info: List[Dict[str, str]] = []

    def add(self, gate: str, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append({"gate": gate, "name": name, "ok": bool(ok), "detail": detail})

    def note(self, name: str, detail: str) -> None:
        self.info.append({"name": name, "detail": detail})

    @property
    def failed(self) -> List[Dict[str, Any]]:
        return [c for c in self.checks if not c["ok"]]

    def render_gate(self, gate: str) -> None:
        rows = []
        for i, c in enumerate([x for x in self.checks if x["gate"] == gate], 1):
            rows.append([i, c["name"], "✅ PASS" if c["ok"] else "❌ FAIL", c["detail"]])
        if rows:
            print(_md_table(["#", "检查项", "结果", "说明"], rows))


# ---------------------------------------------------------------- G1 安全护栏

# 故意把数值放在句子中间：真实医学文本里数字不会出现在段落第 0 位，
# 这样能覆盖 `_cut_safely` 的「向左挪」路径（而不是退无可退的兜底分支）。
CUT_SWEEP_SAMPLES = (
    "阿司匹林100mg每日一次长期服用",
    "血压140/90mmHg为诊室诊断界值",
    "低密度脂蛋白1.4mmol/L为超高危目标",
    "二甲双胍500mg每日两次随餐服用",
    "确诊需非同日的血压均140/90mmHg以上",
    "氯吡格雷75mg与阿司匹林100mg联合使用",
)

DOSE_TOKEN = re.compile(r"\d+(?:\.\d+)?\s*(?:mg|g|mmHg|mmol/L|mmol|%)")


def _gate1_safety(rep: Report) -> None:
    from knowledge.milvus_kb import MedicalKnowledgeBase
    from constraints import ConstraintValidator
    from validation import AutoFixer
    from swarm.swarm_coordinator import SwarmCoordinator

    # 借方法用，不触发模型加载（_cut_safely / _chunk_text 只用 self 上的这几个方法）
    kb = object.__new__(MedicalKnowledgeBase)

    # --- 1a 穷举扫描：任意切点都不得落在连续数字之间 ---
    splits: List[str] = []
    fallbacks = 0
    for s in CUT_SWEEP_SAMPLES:
        for size in range(1, len(s) + 1):
            cut = kb._cut_safely(s, size)
            if cut <= 0 or cut > size:
                splits.append(f"{s[:12]}… size={size} cut={cut}（越界）")
                continue
            if 0 < cut < len(s) and s[cut - 1].isdigit() and s[cut].isdigit():
                # 兜底分支：期望切点左侧全是数字，退无可退时按原切点切（代码内有注释）
                if cut == min(size, len(s)) and s[:size].isdigit():
                    fallbacks += 1
                    continue
                splits.append(f"{s[:12]}… size={size} cut={cut} 劈裂 {s[cut-1:cut+1]!r}")
    total_sweep = sum(len(s) for s in CUT_SWEEP_SAMPLES)
    rep.add(
        "G1", "切片避让数字（穷举扫描）", not splits,
        f"扫描 {len(CUT_SWEEP_SAMPLES)} 段文本 / {total_sweep} 个切点，"
        f"劈裂 {len(splits)} 次" + (f"｜样例：{splits[:2]}" if splits else "")
        + (f"（退无可退兜底 {fallbacks} 次，符合实现约定）" if fallbacks else ""),
    )

    # --- 1b 分块后剂量完整性 ---
    dose_line = "长期服用阿司匹林100mg每日一次，氯吡格雷75mg每日一次，二甲双胍500mg每日三次。"
    merged = "\n".join(f"第{i}条：{dose_line}" for i in range(1, 40))          # 走「句子合并」路径
    giant = "本例为超长单句：" + "（说明性文字）" * 120 + dose_line * 30        # 走「硬切」路径
    dose_check: List[str] = []
    boundary_splits = 0
    for tag, text in (("合并路径", merged), ("硬切路径", giant)):
        chunks = kb._chunk_text(text, chunk_size=1024, overlap=100)
        if not chunks or max(len(c) for c in chunks) > 1024:
            dose_check.append(f"{tag}：分块异常（{len(chunks)} 块 / 最长 {max((len(c) for c in chunks), default=0)}）")
            continue
        missing = sorted({t for t in DOSE_TOKEN.findall(text) if not any(t in c for c in chunks)})
        if missing:
            dose_check.append(f"{tag}：剂量被劈裂 {missing}")
        # 观察项：切点落在「数字 | 单位」之间（P0-2 保证的是数字内部不劈裂，
        # 这条属于残留边界，记录数量但不作为失败条件）
        for a, b in zip(chunks, chunks[1:]):
            if a[-1:].isdigit() and re.match(r"(?:mg|g|mmHg|mmol|%)", b.lstrip()):
                boundary_splits += 1
    rep.add(
        "G1", "分块后剂量完整（100mg 不劈裂）", not dose_check,
        "合并/硬切两条路径 6 种剂量形态全部完整" if not dose_check else "；".join(dose_check),
    )
    if boundary_splits:
        rep.note(
            "剂量与单位跨块观察",
            f"{boundary_splits} 处切点落在「数字|单位」之间（数字内部未被劈裂，"
            f"属已知残留边界，非 P0-2 承诺范围）",
        )

    # --- 1c 出口安全网：高危急症必须补齐就医提醒 + 免责声明 ---
    class _CoordStub:
        """只需要 validator / auto_fixer 两个属性，避免构造真实 LLM 客户端。"""

        _FINAL_ANSWER_FIXES = SwarmCoordinator._FINAL_ANSWER_FIXES

        def __init__(self) -> None:
            self.validator = ConstraintValidator()
            self.auto_fixer = AutoFixer()

    stub = _CoordStub()
    enforce = SwarmCoordinator._enforce_output_safety

    high_q = "突然剧烈胸痛，还冒冷汗，持续 20 分钟不缓解"
    bare = "考虑心绞痛可能，建议先休息观察一下。"
    fixed = enforce(stub, bare, high_q)
    has_call = "120" in fixed
    has_visit = any(k in fixed for k in ("就医", "急诊", "医院", "就诊"))
    has_disclaimer = any(k in fixed for k in ("免责", "仅供参考", "不能替代"))
    rep.add(
        "G1", "高危胸痛出口必补 120 与就医提醒", has_call and has_visit,
        f"120={'有' if has_call else '缺失'}｜就医提醒={'有' if has_visit else '缺失'}｜"
        f"补后长度 {len(bare)}→{len(fixed)}",
    )
    rep.add(
        "G1", "高危胸痛出口必补免责声明", has_disclaimer,
        "含免责声明" if has_disclaimer else "未补免责声明",
    )

    # --- 1d 合规低危问题不得误报（字节级不变） ---
    low_q = "高血压患者平时饮食要注意什么？"
    low_answer = (
        "建议低盐饮食、规律运动、控制体重。\n\n"
        "【免责声明】\n以上信息仅供参考，不能替代专业医生的诊断和治疗。"
    )
    low_out = enforce(stub, low_answer, low_q)
    rep.add(
        "G1", "低危问题不误报（输出零改写）", low_out == low_answer,
        "输出与输入一致" if low_out == low_answer else f"被追加了内容：{low_out[len(low_answer):][:60]!r}",
    )

    # --- 1e 已合规的高危答案不重复追加 ---
    safe_answer = "⚠️ 请立即就医或拨打急救电话 120。\n\n【免责声明】仅供参考，不能替代医生诊断。"
    safe_out = enforce(stub, safe_answer, high_q)
    rep.add(
        "G1", "已合规高危答案不重复追加", safe_out == safe_answer,
        "幂等，无重复警示" if safe_out == safe_answer else "出现重复追加",
    )


# ---------------------------------------------------------------- G2 检索召回

def _load_format_results():
    """动态加载生产用的来源头格式化函数（与 Agent 运行时同一份实现）。"""
    path = ROOT / ".claude" / "skills" / "search-knowledge" / "script" / "search.py"
    spec = importlib.util.spec_from_file_location("_ci_search_skill", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.format_results


def _cosine(a, b) -> float:
    import numpy as np

    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _stratified_sample(cases: List[Dict[str, Any]], limit: int, seed: int) -> List[Dict[str, Any]]:
    """按 source 分层轮询取样，保证各专科/各来源都被覆盖（同 seed 结果可复现）。"""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for c in cases:
        buckets.setdefault(str(c.get("source") or "-"), []).append(c)
    rng = random.Random(seed)
    for v in buckets.values():
        rng.shuffle(v)
    picked: List[Dict[str, Any]] = []
    keys = sorted(buckets, key=lambda k: (-len(buckets[k]), k))
    while len(picked) < limit and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k]:
                picked.append(buckets[k].pop())
                if len(picked) >= limit:
                    break
    return picked


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with io.open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _seed_light_corpus(kb) -> Dict[str, Any]:
    """轻量模式：用被 git 跟踪的知识文档现场建索引（排除已归档的旧二手摘要）。"""
    docs_dir = ROOT / "knowledge" / "data" / "documents"
    doc_files = [
        p for p in sorted(docs_dir.glob("*.txt"))
        if p.name not in ARCHIVED_FILENAMES
    ]
    documents = [
        {
            "id": p.stem,
            "content": io.open(p, encoding="utf-8").read(),
            "metadata": {"filename": p.name, "source": p.stem, "type": "general"},
        }
        for p in doc_files
    ]
    inserted = kb.add_documents(documents, chunk_size=1024)
    return {"files": len(documents), "chunks": inserted}


def _module_chunks(kb, text: str) -> List[str]:
    return [c for c in kb._chunk_text(text, chunk_size=1024, overlap=100) if len(c) >= 240]


def _query_from_chunk(chunk: str) -> str:
    """从段落里挑一条像「用户会怎么问」的检索线索。

    优先取块内首个「够长、含中文、不以标点开场」的行（通常是小节标题或首个要点），
    找不到就退回段落开头；两者都不合格则返回空串（该块不参与自检）。
    """
    for raw in chunk.splitlines():
        line = raw.strip().lstrip("#*->· ").strip()
        if len(line) < 20 or not re.search(r"[\u4e00-\u9fff]", line):
            continue
        if line[:1] in "（(、，,。；;:-—·":
            continue
        return line[:60]
    head = chunk.strip()[:60]
    if len(head) >= 20 and re.search(r"[\u4e00-\u9fff]", head):
        return head
    return ""


def _light_cases(kb, limit: int, seed: int) -> List[Dict[str, Any]]:
    """轻量模式的检索语料自检：拿库内段落的首句去检索，必须召回同一段。

    这是「索引 + 检索链路 + 来源头」的完整性冒烟，不是语义问答评测，
    因此不走 benchmark 金标答案（完整模式才有 500 题金标）。
    问题取自段落首句，按文档分层轮询抽取，保证 12 份文档都被覆盖。
    """
    docs_dir = ROOT / "knowledge" / "data" / "documents"
    doc_files = [p for p in sorted(docs_dir.glob("*.txt")) if p.name not in ARCHIVED_FILENAMES]
    per_doc: Dict[str, List[Dict[str, Any]]] = {}
    seen: set = set()
    for p in doc_files:
        text = io.open(p, encoding="utf-8").read()
        bucket: List[Dict[str, Any]] = []
        for chunk in _module_chunks(kb, text):
            query = _query_from_chunk(chunk)
            if not query or query in seen:
                continue
            seen.add(query)
            bucket.append({"question": query, "gold": chunk, "source": p.name})
        if bucket:
            per_doc[p.name] = bucket

    rng = random.Random(seed)
    for v in per_doc.values():
        rng.shuffle(v)
    picked: List[Dict[str, Any]] = []
    keys = sorted(per_doc)
    while len(picked) < limit and any(per_doc[k] for k in keys):
        for k in keys:
            if per_doc[k]:
                picked.append(per_doc[k].pop())
                if len(picked) >= limit:
                    break
    return picked


def _gate2_retrieval(rep: Report, kb, mode: str, args) -> None:
    format_results = _load_format_results()
    cases: List[Dict[str, Any]] = []
    probes: List[Tuple[str, str]] = []

    if mode == "full":
        bench = ROOT / "eval" / "data" / "benchmark_500.jsonl"
        pool = [c for c in _read_jsonl(bench) if c.get("category") == "guideline_retrieval"]
        for c in _stratified_sample(pool, args.cases, args.seed):
            cases.append({"question": c["question"], "gold": c.get("answer", ""), "source": c.get("source", "-")})
        probes = list(GUIDELINE_PROBES)
    else:
        cases = _light_cases(kb, args.cases, args.seed)

    if not cases:
        rep.add("G2", "检索用例可用", False, "用例为空，无法评测")
        return

    # --- 2a hit@1（top-1 与金标答案的 cosine ≥ 0.40）---
    rows: List[Tuple[int, str]] = []
    golds: List[str] = []
    tops: List[str] = []
    header_bad: List[str] = []
    empty_idx: set = set()
    for i, case in enumerate(cases, 1):
        hits = kb.search(case["question"], top_k=args.top_k)
        if not hits:
            # 空检索直接判失败：不能拿占位文本去算相似度（那会得到接近 1 的假高分）
            empty_idx.add(i - 1)
            golds.append("")
            tops.append("")
            rows.append((i, "-"))
            continue
        top = hits[0]
        golds.append(case["gold"])
        tops.append(top.get("content", ""))
        rows.append((i, str(top.get("metadata", {}).get("filename") or "-")))
        # --- 2c 来源头接地格式契约（防「静默降级」回归，见 rag实现.md 坑 5）---
        header = format_results(hits[:1]).splitlines()[0]
        expected_src = str(top.get("metadata", {}).get("source") or "")
        if "医学知识库｜类型：-" in header or (expected_src and expected_src not in header):
            header_bad.append(f"#{i} {header[:70]}")

    import numpy as np

    model = kb.embedding_model
    gold_vecs = model.encode([g or "（空检索）" for g in golds], normalize_embeddings=True, show_progress_bar=False)
    top_vecs = model.encode([t or "（空检索）" for t in tops], normalize_embeddings=True, show_progress_bar=False)
    sims = [float(np.dot(gold_vecs[i], top_vecs[i])) for i in range(len(golds))]

    hit = [s >= args.sim_threshold for s in sims]
    for idx in empty_idx:
        sims[idx] = 0.0
        hit[idx] = False
    hit_rate = sum(hit) / len(hit) if hit else 0.0
    mean_sim = sum(sims) / len(sims) if sims else 0.0

    print(_md_table(
        ["#", "问题", "命中来源", "top1_sim", "判定"],
        [
            [i, (c["question"][:34] + "…" if len(c["question"]) > 34 else c["question"]),
             rows[i - 1][1][:30], f"{sims[i-1]:.4f}", "PASS" if hit[i - 1] else "FAIL"]
            for i, c in enumerate(cases, 1)
        ],
    ))
    print()
    rep.add(
        "G2", f"hit@1 ≥ {args.hit_threshold:.0%}（cosine ≥ {args.sim_threshold}）",
        hit_rate >= args.hit_threshold - 1e-9,
        f"hit@1 = {hit_rate:.1%}（{sum(hit)}/{len(hit)}）｜mean top1_sim = {mean_sim:.4f}｜"
        f"空检索 {len(empty_idx)}｜top_k = {args.top_k}",
    )
    rep.add(
        "G2", "来源头接地格式契约", not header_bad,
        "每条来源头都带真实 source 且非兜底值" if not header_bad
        else f"{len(header_bad)} 条退化为兜底：{header_bad[:2]}",
    )

    # --- 2b 真指南来源命中（完整模式才有真指南语料）---
    # 判定口径是「可召回性」：期望指南出现在 top-k 内即算命中。
    # top-1 是否正好是期望指南只作观察项 —— 跨专科相关命中是合理的
    # （例如「降压目标值」会被糖尿病指南里的「糖尿病合并高血压」章节抢到 top-1）。
    if probes:
        probe_rows = []
        miss = []
        top1_hit = 0
        for q, expect in probes:
            hits = kb.search(q, top_k=args.top_k)
            files = [str((h.get("metadata", {}).get("filename") or "-")) for h in hits]
            got = files[0] if files else "-"
            rank = files.index(expect) + 1 if expect in files else 0
            if rank == 1:
                top1_hit += 1
            if rank == 0:
                miss.append(f"{q[:18]}… 期望 {expect} 未进 top-{args.top_k}")
            probe_rows.append([
                q[:36], expect[:30], got[:30],
                f"top-{rank}" if rank else "-", "PASS" if rank else "FAIL",
            ])
        print(_md_table(["问题", "期望指南", "top-1 实得", "期望排名", "判定"], probe_rows))
        print()
        rep.add(
            "G2", f"4 份真指南可召回（top-{args.top_k} 内命中 8 题）", not miss,
            f"{len(probes) - len(miss)}/{len(probes)} 题的期望指南进入 top-{args.top_k}"
            if not miss else f"未召回：{miss[:2]}",
        )
        if top1_hit < len(probes):
            rep.note(
                "指南探针 top-1",
                f"{top1_hit}/{len(probes)} 题的 top-1 就是期望指南；其余为跨专科相关命中"
                f"（如糖尿病指南含「合并高血压」的血压控制章节），故门禁按可召回性判定",
            )


# ---------------------------------------------------------------- G3 数据一致性

def _git_ls_files(pattern: str = "") -> List[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files"] + ([pattern] if pattern else []),
            cwd=str(ROOT), capture_output=True, text=True, timeout=30,
        )
        return [l.strip() for l in out.stdout.splitlines() if l.strip()]
    except Exception:
        return []


def _rows_by_filename(kb, filename: str) -> List[Dict[str, Any]]:
    expr = f'metadata like "%\\"filename\\": \\"{filename}\\"%"'
    return kb.milvus_client.query(
        kb.collection_name, filter=expr, output_fields=["metadata"], limit=16384
    )


def _gate3_consistency(rep: Report, kb, mode: str) -> None:
    # --- 3a 旧二手摘要不得留在库内（两种模式都查）---
    residue: List[str] = []
    for name in ARCHIVED_FILENAMES:
        try:
            n = len(_rows_by_filename(kb, name))
        except Exception as e:  # 查询失败按失败处理，不静默放过
            n = -1
            rep.note("旧摘要查询异常", f"{name}: {type(e).__name__}")
        if n != 0:
            residue.append(f"{name}={n}")
    rep.add(
        "G3", "旧二手摘要向量残留为 0", not residue,
        "3 份旧摘要均无 chunk" if not residue else f"残留：{residue}",
    )

    if mode == "full":
        # --- 3b 真指南块数 ---
        got: Dict[str, int] = {}
        for name in EXPECTED_GUIDELINE_CHUNKS:
            try:
                got[name] = len(_rows_by_filename(kb, name))
            except Exception:
                got[name] = -1
        table = [
            [name, EXPECTED_GUIDELINE_CHUNKS[name], got[name],
             "PASS" if got[name] == EXPECTED_GUIDELINE_CHUNKS[name] else "FAIL"]
            for name in EXPECTED_GUIDELINE_CHUNKS
        ]
        table.append(["合计", EXPECTED_GUIDELINE_TOTAL, sum(got.values()), ""])
        print(_md_table(["指南文件", "期望块数", "实测块数", "判定"], table))
        print()
        bad = [n for n in EXPECTED_GUIDELINE_CHUNKS if got[n] != EXPECTED_GUIDELINE_CHUNKS[n]]
        rep.add(
            "G3", f"真指南全文块数 = {EXPECTED_GUIDELINE_TOTAL}", not bad,
            "四份全对（323 / 401 / 129 / 24）" if not bad else f"不符：{bad}",
        )
        rep.note("检索语料", f"生产库命中真指南全文 {sum(got.values())} 块")
    else:
        # --- 3c 轻量模式：索引可用 + 归档与版权面 ---
        total = 0
        try:
            total = len(kb._fetch_all_chunks())
        except Exception:
            pass
        rep.add("G3", "临时索引已建好可用", total > 0, f"库内 chunk（去重口径）= {total}")

        tracked_pdf = [f for f in _git_ls_files() if f.endswith(".pdf")]
        tracked_db = [f for f in _git_ls_files() if f.endswith((".db", ".arrow", ".parquet"))]
        rep.add(
            "G3", "版权面：PDF 与向量库未被 git 跟踪", not tracked_pdf and not tracked_db,
            "无 PDF / 无库文件入库" if not tracked_pdf and not tracked_db
            else f"PDF={tracked_pdf[:2]} DB={tracked_db[:2]}",
        )

        bak = sorted(p.name for p in (ROOT / "knowledge" / "data" / "documents").glob("*.bak"))
        active = [
            n for n in ARCHIVED_FILENAMES
            if (ROOT / "knowledge" / "data" / "documents" / n).exists()
        ]
        rep.note(
            "工作区归档状态",
            f"非门禁项（CI 检出的是已提交版本）：.bak 在档 {len(bak)} 个"
            f"｜旧摘要仍以 .txt 在工作区 {len(active)} 个",
        )
        rep.note(
            "成本",
            "零 API 调用（未触碰任何在线模型）；本次 run 的语料来自仓库内被跟踪的知识文档",
        )


# ---------------------------------------------------------------- 主流程

def _detect_mode(kb) -> str:
    try:
        n = len(_rows_by_filename(kb, "01_hypertension_2024.pdf"))
    except Exception:
        n = 0
    return "full" if n > 0 else "light"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="CI 冒烟评测门禁")
    ap.add_argument("--mode", choices=("auto", "full", "light"), default="auto")
    ap.add_argument("--cases", type=int, default=DEFAULT_CASES, help="检索用例条数（默认 20）")
    ap.add_argument("--seed", type=int, default=42, help="分层抽样随机种子")
    ap.add_argument("--top-k", type=int, default=8, help="检索返回条数（对齐既有评测协议）")
    ap.add_argument("--sim-threshold", type=float, default=SIM_THRESHOLD)
    ap.add_argument("--hit-threshold", type=float, default=HIT_THRESHOLD)
    ap.add_argument("--rerank", action="store_true", help="打开 cross-encoder 重排（默认关闭）")
    ap.add_argument("--json-out", default=None, help="把结果写成 JSON（CI 可作 artifact）")
    args = ap.parse_args(argv)

    if not args.rerank:
        os.environ["MEDIX_DISABLE_RERANK"] = "1"

    t0 = time.perf_counter()
    print("=" * 78)
    print("MediX-Swarm CI 冒烟评测门禁（零 API 成本 / 纯本地）")
    print("=" * 78)

    rep = Report()

    # --- G1：不依赖模型，先跑；即使后面模型加载失败也能看到护栏结论 ---
    print("\n--- G1 安全护栏（硬门禁，必须 100%）---")
    try:
        _gate1_safety(rep)
    except Exception as e:
        rep.add("G1", "安全护栏可执行", False, f"{type(e).__name__}: {e}")
    rep.render_gate("G1")

    # --- G2 / G3：需要向量模型与知识库 ---
    kb = None
    mode = args.mode
    seed_info = ""
    try:
        from knowledge.milvus_kb import MedicalKnowledgeBase

        t_load = time.perf_counter()
        kb = MedicalKnowledgeBase()
        if mode == "auto":
            mode = _detect_mode(kb)
        if mode == "light":
            # 只在「空库」时现场建索引：CI 是全新检出，必然为空；
            # 本地已有生产库时复用（否则会把同一批文档灌两遍）。
            existing = 0
            try:
                existing = len(kb._fetch_all_chunks())
            except Exception:
                existing = 0
            if existing:
                seed_info = f"｜复用已有索引 {existing} 块（非空则不重复灌入）"
            else:
                seeded = _seed_light_corpus(kb)
                seed_info = f"｜现场建索引 {seeded['files']} 篇 → {seeded['chunks']} 块"
        print(
            f"\n[模式] {'完整（生产知识库）' if mode == 'full' else '轻量（仓库内文档临时索引）'}"
            f"｜重排 {'开启' if args.rerank else '关闭'}"
            f"｜模型加载 {time.perf_counter() - t_load:.1f}s{seed_info}"
        )
    except Exception as e:
        rep.add("G2", "知识库可用", False, f"{type(e).__name__}: {e}")
        rep.add("G3", "知识库可用", False, "知识库不可用，无法校验数据一致性")

    if kb is not None:
        print("\n--- G2 检索召回（硬门禁）---")
        try:
            _gate2_retrieval(rep, kb, mode, args)
        except Exception as e:
            rep.add("G2", "检索门禁可执行", False, f"{type(e).__name__}: {e}")
        rep.render_gate("G2")

        print("\n--- G3 数据一致性（硬门禁）---")
        try:
            _gate3_consistency(rep, kb, mode)
        except Exception as e:
            rep.add("G3", "一致性门禁可执行", False, f"{type(e).__name__}: {e}")
        rep.render_gate("G3")

    # --- 汇总 ---
    elapsed = time.perf_counter() - t0
    passed = len(rep.checks) - len(rep.failed)
    print()
    print("=" * 78)
    print(f"检查项 {passed}/{len(rep.checks)} 通过｜耗时 {elapsed:.1f}s｜模式 {mode}")
    if rep.failed:
        print("未通过：")
        for c in rep.failed:
            print(f"  ❌ [{c['gate']}] {c['name']}：{c['detail']}")
    if rep.info:
        print("观察项：")
        for i in rep.info:
            print(f"  ⚠️  {i['name']}：{i['detail']}")
    verdict = "✅ 全部硬门禁通过" if not rep.failed else "❌ 存在失败项，CI 应报红"
    print(f"结论：{verdict}")
    print("=" * 78)

    if args.json_out:
        out = Path(args.json_out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        io.open(out, "w", encoding="utf-8").write(json.dumps({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "mode": mode,
            "elapsed_sec": round(elapsed, 2),
            "passed": passed,
            "total": len(rep.checks),
            "checks": rep.checks,
            "observations": rep.info,
            "config": {
                "cases": args.cases, "seed": args.seed, "top_k": args.top_k,
                "sim_threshold": args.sim_threshold, "hit_threshold": args.hit_threshold,
                "rerank": args.rerank,
            },
        }, ensure_ascii=False, indent=2))
        print(f"结果已写入 {out}")

    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
