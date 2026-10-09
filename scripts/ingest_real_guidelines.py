#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实权威指南入库与平滑替换。

四件事：
1. 抽取与清洗：用 pypdf 逐页抽文本，去掉纯页码、跨页重复的页眉页脚，把硬换行拼回段落
2. 分块与元数据：复用知识库修复后的 _chunk_text（句级拆分 + _cut_safely 数字回退），
   为每个 chunk 绑定 filename / source / chunk_id / total_chunks / type / pages
3. 写入与替换：先确认新数据写入成功，再淘汰 3 份旧二手摘要
4. 探针验证：3 个典型 query 跑召回，打印 Top-1 与来源头

关于「平滑替换」的关键点：
    只把旧摘要文件改名为 .bak 是不够的 —— 它的向量早已在 Milvus 里，
    检索照样会命中旧口径。所以必须同时按 metadata.filename 把旧 chunk 从集合里删掉，
    本脚本两者都做（文件归档 + 向量删除），且严格放在新数据写入成功之后。

用法：
    python scripts/ingest_real_guidelines.py --dry-run     # 只抽取与分块，不碰数据库
    python scripts/ingest_real_guidelines.py               # 入库 + 淘汰旧摘要 + 探针
    python scripts/ingest_real_guidelines.py --force       # 已入库过时，先删旧全文再灌
    python scripts/ingest_real_guidelines.py --keep-old    # 只入库，不淘汰旧摘要
    python scripts/ingest_real_guidelines.py --only 01 05   # 只处理部分 PDF（按文件名前缀）
    python scripts/ingest_real_guidelines.py --probe-only   # 只看召回探针，不碰数据
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import re
import sys
import time
import warnings
from bisect import bisect_right
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "knowledge" / "data" / "raw_guidelines"
DOC_DIR = PROJECT_ROOT / "knowledge" / "data" / "documents"

# 关闭 pypdf 对「PDF 字典重复键」的刷屏警告（这几份期刊 PDF 普遍存在，属历史排版问题）
warnings.filterwarnings("ignore")
logging.getLogger("pypdf").setLevel(logging.CRITICAL)
logging.getLogger("pypdf._reader").setLevel(logging.CRITICAL)

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    colorize=False,
    format="{time:HH:mm:ss} | {level:<7} | {message}",
)

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
INSERT_BATCH = 128

# 收录的四份原文（source 与 scripts/download_medical_guidelines.py 的官方标题保持一致）
SOURCES: List[Dict[str, Any]] = [
    {
        "prefix": "01",
        "filename": "01_hypertension_2024.pdf",
        "guideline_id": "hypertension_2024",
        "source": "中国高血压防治指南（2024年修订版）",
        "publisher": "中国高血压防治指南修订委员会 / 高血压联盟（中国）",
        "year": 2024,
    },
    {
        "prefix": "02",
        "filename": "02_t2dm_2020.pdf",
        "guideline_id": "t2dm_2020",
        "source": "中国2型糖尿病防治指南（2020年版）",
        "publisher": "中华医学会糖尿病学分会 / 中华糖尿病杂志",
        "year": 2020,
    },
    {
        "prefix": "03",
        "filename": "03_lipid_2023.pdf",
        "guideline_id": "lipid_2023",
        "source": "中国血脂管理指南（2023年）",
        "publisher": "中国血脂管理指南修订联合专家委员会 / 中华心血管病杂志",
        "year": 2023,
    },
    {
        "prefix": "05",
        "filename": "05_acute_chest_pain_consensus_2019.pdf",
        "guideline_id": "acute_chest_pain_2019",
        "source": "急性胸痛急诊诊疗专家共识（2019）",
        "publisher": "中华医学会急诊医学分会 / 中国医促会胸痛分会",
        "year": 2019,
    },
]

# 待淘汰的二手摘要（已被上面的真全文取代）；23/24/25 无对应真指南，保留
LEGACY = [
    {
        "filename": "20_guideline_hypertension.txt",
        "replaced_by": "01_hypertension_2024.pdf",
    },
    {
        "filename": "21_guideline_diabetes.txt",
        "replaced_by": "02_t2dm_2020.pdf",
    },
    {
        "filename": "22_guideline_hyperlipidemia.txt",
        "replaced_by": "03_lipid_2023.pdf",
    },
]

# 必须保留的过渡文档（删除后这些主题覆盖会空）
MUST_KEEP = [
    "23_guideline_asthma_copd.txt",
    "24_guideline_gastritis_hp.txt",
    "25_guideline_gout.txt",
]

PROBES = [
    "2024 高血压诊断标准与血压分级",
    "低密度脂蛋白胆固醇 LDL-C 目标值",
    "急性高危胸痛的识别与处理流程",
]

# 只看每页前 3 行与后 6 行非空行：期刊刊头/页脚只出现在页边，
# 正文里重复出现的小节标签（如「要点提示：」）不在这个范围内，因此不会被误删
PAGE_HEAD_LINES = 3
PAGE_TAIL_LINES = 6

# 纯页码形态： 12 / - 12 - / ·415· / 7
_PAGE_NO = re.compile(r"^[\s\-–—_·•\.\|\u3000]*\d{1,8}[\s\-–—_·•\.\|\u3000]*$")
_EXPLICIT_PAGE = re.compile(r"^[\s\-–—_·•\.\|\u3000]*第\s*\d+\s*页[\s\-–—_·•\.\|\u3000]*$")
_DIGITS = re.compile(r"\d+")
# 归一化后整行只剩符号与数字：页脚的「No.」不够纯，另有短行高频规则兜底
_NOISE_CHARS = re.compile(r"^[\s\-–—_·•\.\|\u3000\[\]\(\)（）\d#:：,，;；/]+$")
_CJK = re.compile(r"[\u4e00-\u9fff]")

# 期刊栏头形态（归一化后）：只认卷期号与 Vol/Serial No。
# 这里刻意不含 DOI 与期刊英文缩写：参考文献条目恰好都带 DOI，
# 一旦纳入就会把「中华糖尿病杂志, 2021, 13(4): 320-327. DOI: 10.3760/…」
# 这类文献条目当成栏头整批删掉（实测会误删 245 行）
_MASTHEAD_SHAPE = re.compile(r"(第\s*#\s*[卷期]|总第\s*#\s*期|Vol\s*\.|Serial\s*No)", re.I)
# 文献条目特征：命中者一律不按栏头处理
_REFERENCE_LIKE = re.compile(r"(DOI|\[J\]|\[M\]|\[C\]|et\s+al|//)", re.I)
# 全文档范围内判定栏头的最低重复次数
MASTHEAD_MIN_REPEAT = 3


# --------------------------------------------------------------------------- #
# 一、抽取与清洗
# --------------------------------------------------------------------------- #
def _extract_pages(pdf_path: Path) -> List[Tuple[int, List[str]]]:
    """逐页抽取文本，保留页码，供后续跨页页眉检测与页码归因。"""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages: List[Tuple[int, List[str]]] = []
    for pno, page in enumerate(reader.pages, 1):
        try:
            raw = page.extract_text() or ""
        except Exception as e:  # 单页解析失败不该毁掉整份指南
            logger.warning(f"  {pdf_path.name} 第 {pno} 页解析失败：{type(e).__name__}: {e}")
            raw = ""
        pages.append((pno, raw.splitlines()))
    return pages


def _norm(line: str) -> str:
    """把数字统一成 # 再比较：栏头里的期号/页码逐页不同，不归一化就聚不成同一行。"""
    return _DIGITS.sub("#", line).strip()


def _looks_like_masthead(norm: str) -> bool:
    """刊头/页脚特征：带数字（期号、页码、DOI），或整行只剩符号与数字。"""
    if "#" in norm and len(norm) <= 120:
        return True
    return bool(_NOISE_CHARS.match(norm))


def _edge_index(lines: Sequence[str]) -> set:
    """每页前 3 行 / 后 6 行非空行的下标 —— 刊头页脚只会出现在这里。"""
    idx = [i for i, l in enumerate(lines) if l.strip()]
    return set(idx[:PAGE_HEAD_LINES]) | set(idx[-PAGE_TAIL_LINES:])


def _detect_running_lines(
    pages: Sequence[Tuple[int, List[str]]]
) -> Tuple[set, set, Dict[str, int]]:
    """识别页眉页脚，返回 (页边集, 栏头集, 阈值信息)。

    两套规则，因为栏头不总是乖乖待在页边：
    - 页边集：只在每页前 3 / 后 6 行统计，出现在 ≥50% 页面且带数字，或极短行
      （≤12 字）出现在 ≥80% 页面。这一套抓「·#·」「#,」「No.」这类碎片。
    - 栏头集：全文档统计，形如期刊栏头（含卷期号/Vol/No/DOI）且归一化后重复 ≥3 次。
      这一套抓「·276· 中华高血压杂志(中英文) 2024年7月第32卷第7期 …」这类
      前面挂着页码、落在页边窗口之外的整行栏头。
    """
    total = max(1, len(pages))
    thr_half = max(3, math.ceil(total * 0.5))
    thr_most = max(3, math.ceil(total * 0.8))

    edge_freq: Counter = Counter()
    all_freq: Counter = Counter()
    for _, lines in pages:
        for i in _edge_index(lines):
            edge_freq[_norm(lines[i].strip())] += 1
        for l in lines:
            if l.strip():
                all_freq[_norm(l.strip())] += 1

    edge_set = set()
    for norm, n in edge_freq.items():
        if n < thr_half:
            continue
        if _looks_like_masthead(norm):
            edge_set.add(norm)
        elif len(norm) <= 12 and n >= thr_most:
            edge_set.add(norm)

    furniture = {
        norm
        for norm, n in all_freq.items()
        if n >= MASTHEAD_MIN_REPEAT
        and _MASTHEAD_SHAPE.search(norm)
        and not _REFERENCE_LIKE.search(norm)
        # 栏头是中文刊物抬头，或退化成很短的 Vol./No. 碎片；
        # 长的纯英文数字行更可能是文献条目残片，不碰
        and (bool(_CJK.search(norm)) or len(norm) <= 20)
    }
    return edge_set, furniture, {"pages": total, "half": thr_half, "most": thr_most}


def _clean_page(
    lines: Sequence[str], edge_set: set, furniture: set
) -> Tuple[List[str], int]:
    """清洗单页：去掉页码与页眉页脚，保留空行作为段落信号。返回 (保留行, 剔除行数)。"""
    edge = _edge_index(lines)
    out: List[str] = []
    dropped = 0
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            out.append("")
            continue
        if _EXPLICIT_PAGE.match(s):
            dropped += 1
            continue
        if _norm(s) in furniture:
            dropped += 1
            continue
        if i in edge:
            # 页码只在页边剔除，且不含中文 —— 避免误删正文里独立成行的数字
            if _PAGE_NO.match(s) and not _CJK.search(s):
                dropped += 1
                continue
            if _norm(s) in edge_set:
                dropped += 1
                continue
        out.append(s)
    return out, dropped


def _repair_split_units(
    paras: List[str], pages: List[int]
) -> Tuple[List[str], List[int]]:
    """把被抽取器拆散的「数字 + 单位」接回来。

    pypdf 受表格列宽与文字块影响，会把一个数值拆到下一段，例如：
        「海拔超过 500」+「m 的高原地区」      → 500m
        「生活在 2」+「000」+「m 以上」        → 2000m
    这类拆分正是切分环节要防的「数字与单位分离」，只是发生在抽取阶段，必须在这里修。

    只处理无歧义的两种形态：
    1. 本段以数字结尾，且下一段以拉丁单位/百分号/万/亿开头 → 一定是同一个数值
    2. 本段以数字结尾，下一段是纯数字且再下一段以单位开头  → 同一个数值被拆成两截

    绝不动「两个相邻的纯数字段」这类表格形态：那里 739 与 204 是不同列，拼起来会凭空造数。
    """
    unit_head = re.compile(r"^(?:[A-Za-z%]|万|亿)")
    out_p: List[str] = []
    out_pg: List[int] = []

    for i, (p, pg) in enumerate(zip(paras, pages)):
        if not out_p:
            out_p.append(p)
            out_pg.append(pg)
            continue
        nxt = paras[i + 1] if i + 1 < len(paras) else ""
        prev_digit_tail = bool(re.search(r"\d$", out_p[-1]))
        if prev_digit_tail and unit_head.match(p):
            out_p[-1] += p
        elif prev_digit_tail and re.fullmatch(r"\d{1,3}", p) and unit_head.match(nxt):
            out_p[-1] += p
        else:
            out_p.append(p)
            out_pg.append(pg)
    return out_p, out_pg


def _build_paragraphs(lines: Sequence[str]) -> List[str]:
    """把 PDF 的硬换行拼回段落。

    PDF 每行都是硬换行，直接拿去切分会让句子在任意位置断开。
    规则：上一行已以句末标点结束 → 另起一段；否则视为续行拼接，
    英文连字符断词（Collabora- / tion）去掉连字符后接上。
    """
    paras: List[str] = []
    buf = ""
    for s in lines:
        if not s:
            if buf:
                paras.append(buf)
                buf = ""
            continue
        if not buf:
            buf = s
            continue
        if buf[-1] in "。！？；":
            paras.append(buf)
            buf = s
        elif buf.endswith("-") and re.match(r"^[A-Za-z0-9]", s):
            buf = buf[:-1] + s
        else:
            buf += s
    if buf:
        paras.append(buf)
    return paras


def _assemble(
    pages: Sequence[Tuple[int, List[str]]], edge_set: set, furniture: set
) -> Tuple[str, List[Tuple[int, int, int]], int, int]:
    """拼出扁平文本与「段落偏移 → 页码」映射表，供 chunk 归因页码。

    返回 (扁平文本, 段落跨度表, 剔除行数, 接回的数字单位数)。
    """
    paras: List[str] = []
    para_pages: List[int] = []
    dropped = 0
    for pno, lines in pages:
        cleaned, n_drop = _clean_page(lines, edge_set, furniture)
        dropped += n_drop
        for p in _build_paragraphs(cleaned):
            paras.append(p)
            para_pages.append(pno)

    before = len(paras)
    paras, para_pages = _repair_split_units(paras, para_pages)
    repaired = before - len(paras)

    text = ""
    spans: List[Tuple[int, int, int]] = []
    for i, p in enumerate(paras):
        if i:
            text += "\n"
        start = len(text)
        text += p
        spans.append((start, start + len(p), para_pages[i]))
    return text, spans, dropped, repaired


def _page_of(spans: Sequence[Tuple[int, int, int]], pos: int) -> int:
    """按字符偏移查所在页码。"""
    if not spans:
        return 0
    starts = [s[0] for s in spans]
    idx = max(0, bisect_right(starts, pos) - 1)
    return spans[idx][2]


def _chunk_pages(flat: str, chunks: Sequence[str], spans: Sequence[Tuple[int, int, int]]) -> List[str]:
    """为每个 chunk 标注页码区间。

    chunk 之间带 overlap，并非源文本的连续切片，所以按 chunk 开头在原文里定位；
    定位不到（理论上只在 strip 改变首字符时发生）就沿用上一块的页码。
    """
    out: List[str] = []
    prev_page = spans[0][2] if spans else 0
    cursor = 0
    for c in chunks:
        probe = c[:60]
        pos = flat.find(probe, max(0, cursor - CHUNK_SIZE))
        if pos < 0:
            pos = flat.find(probe)
        if pos < 0:
            out.append(str(prev_page))
            continue
        cursor = pos
        start_page = _page_of(spans, pos)
        end_page = _page_of(spans, pos + max(1, len(c)) - 1)
        prev_page = start_page
        out.append(str(start_page) if start_page == end_page else f"{start_page}-{end_page}")
    return out


def build_chunks(pdf_path: Path, kb) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """抽取 → 清洗 → 分块 → 绑定元数据。"""
    t0 = time.perf_counter()
    pages = _extract_pages(pdf_path)
    edge_set, furniture, thresholds = _detect_running_lines(pages)
    flat, spans, dropped_lines, repaired_units = _assemble(pages, edge_set, furniture)

    raw_chars = sum(len(l) for _, lines in pages for l in lines)
    raw_lines = sum(1 for _, lines in pages for l in lines if l.strip())
    chunks = kb._chunk_text(flat, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    page_tags = _chunk_pages(flat, chunks, spans)

    stat = {
        "file": pdf_path.name,
        "pages": len(pages),
        "raw_chars": raw_chars,
        "clean_chars": len(flat),
        "removed_ratio": round(1 - len(flat) / raw_chars, 4) if raw_chars else 0.0,
        "raw_lines": raw_lines,
        "dropped_lines": dropped_lines,
        "repaired_units": repaired_units,
        "running_lines": sorted(furniture),
        "edge_lines": sorted(edge_set),
        "running_thresholds": thresholds,
        "chunks": len(chunks),
        "avg_chunk": round(sum(len(c) for c in chunks) / len(chunks), 1) if chunks else 0,
        "max_chunk": max((len(c) for c in chunks), default=0),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    return [{"content": c, "pages": page_tags[i]} for i, c in enumerate(chunks)], stat


# --------------------------------------------------------------------------- #
# 二、数据库读写
# --------------------------------------------------------------------------- #
def _all_chunks(kb) -> List[Dict[str, Any]]:
    """全量 chunk（按内容去重）。只用于看集合总量与兜底统计。"""
    return kb._fetch_all_chunks()


def _filename_expr(filename: str) -> str:
    return f'metadata like "%\\"filename\\": \\"{filename}\\"%"'


def _rows_by_filename(kb, filename: str, fields: Sequence[str] = ("metadata",)) -> List[Dict[str, Any]]:
    """按 metadata.filename 查原始行。

    注意：不能用 _fetch_all_chunks 做数量校验 —— 它按内容去重，
    文件里若有两块文本完全相同的 chunk，就会少数几块，看起来像写入失败。
    这里必须查原始行。
    """
    return kb.milvus_client.query(
        kb.collection_name, filter=_filename_expr(filename), output_fields=list(fields), limit=16384
    )


def _count_by_filename(kb, filename: str) -> int:
    try:
        return len(_rows_by_filename(kb, filename))
    except Exception as e:
        logger.warning(f"  查询 {filename} 行数失败，回退到去重统计：{e}")
        return sum(
            1 for c in _all_chunks(kb)
            if (c.get("metadata") or {}).get("filename") == filename
        )


def _ids_by_filename(kb, filename: str) -> List[Any]:
    """按 metadata.filename 查出主键，用于精确删除旧摘要的向量。"""
    for fields in (("id", "metadata"), ("metadata",)):
        try:
            rows = _rows_by_filename(kb, filename, fields=fields)
        except Exception as e:
            logger.warning(f"  查询 {filename} 主键失败（output_fields={fields}）：{e}")
            continue
        ids = [r.get("id") for r in rows if r.get("id") is not None]
        if ids:
            return ids
        if rows:  # 查到了行却拿不到 id，换个字段组合再试
            continue
        return []
    return []


def _insert(kb, rows: Sequence[Dict[str, Any]]) -> int:
    """分批向量化并写入。"""
    total = 0
    for i in range(0, len(rows), INSERT_BATCH):
        part = rows[i:i + INSERT_BATCH]
        vectors = kb.embedding_model.encode(
            [r["content"] for r in part], show_progress_bar=False
        )
        data = [
            {
                "vector": v.tolist(),
                "content": r["content"],
                "metadata": json.dumps(r["metadata"], ensure_ascii=False),
            }
            for r, v in zip(part, vectors)
        ]
        kb.milvus_client.insert(kb.collection_name, data)
        total += len(data)
        logger.info(f"  已写入 {total}/{len(rows)} 块")
    # 语料变了，BM25 缓存必须失效，否则混合检索会命中过期语料
    kb.invalidate_bm25_cache()
    return total


def _delete_by_filename(kb, filename: str) -> int:
    """删除某个修订版的全部旧 chunk，返回删除条数。"""
    ids = _ids_by_filename(kb, filename)
    if not ids:
        return 0
    kb.milvus_client.delete(kb.collection_name, ids=ids)
    kb.invalidate_bm25_cache()
    return len(ids)


# --------------------------------------------------------------------------- #
# 三、探针检索（复用生产用的来源头格式化，验证的是真实链路）
# --------------------------------------------------------------------------- #
def _load_formatter():
    path = PROJECT_ROOT / ".claude" / "skills" / "search-knowledge" / "script" / "search.py"
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("search_skill_probe", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return getattr(mod, "format_results", None)
    except Exception as e:
        logger.warning(f"加载来源头格式化函数失败：{e}")
        return None


def _print_probes(probes: Sequence[Dict[str, Any]]) -> None:
    """打印探针结果。

    来源头直接调生产用的 format_results（所以改了 search.py 这里自动同步），
    命中行再单独把中文全称与页码摆出来，便于一眼核对替换是否生效。
    """
    for item in probes:
        print(f"\n  Q: {item['query']}")
        if item.get("header"):
            print(f"     来源头: {item['header']}")
        for h in item["hits"]:
            kind = "真全文" if h.get("doc_kind") == "full_text_pdf" else "库内其它文档"
            label = str(h.get("source") or h.get("filename") or "-")
            pages = str(h.get("pages") or "").strip()
            page_label = f"p.{pages}" if pages and pages != "-" else "-"
            print(
                "     Top%-2d score=%.4f  %-32s %-8s [%s]"
                % (h["rank"], h["score"], label[:32], page_label, kind)
            )
        if item["hits"]:
            print(f"     摘要: {item['hits'][0]['preview']}")


def run_probes(kb, top_k: int = 3) -> List[Dict[str, Any]]:
    fmt = _load_formatter()
    report: List[Dict[str, Any]] = []
    for q in PROBES:
        res = kb.search(q, top_k=top_k)
        item = {"query": q, "hits": []}
        for i, d in enumerate(res, 1):
            m = d.get("metadata") or {}
            item["hits"].append(
                {
                    "rank": i,
                    "score": d.get("score", 0),
                    "filename": m.get("filename"),
                    "source": m.get("source"),
                    "type": m.get("type"),
                    "pages": m.get("pages"),
                    "doc_kind": m.get("doc_kind"),
                    "preview": " ".join((d.get("content") or "").split())[:160],
                }
            )
        if res and fmt:
            item["header"] = fmt(res[:1]).splitlines()[0]
        report.append(item)
    return report


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="真实权威指南入库与平滑替换")
    ap.add_argument("--dry-run", action="store_true", help="只抽取与分块，不写数据库、不动旧文件")
    ap.add_argument("--force", action="store_true", help="同名全文已入库时，先删旧全文再灌")
    ap.add_argument("--keep-old", action="store_true", help="只入库，不淘汰旧摘要")
    ap.add_argument("--only", nargs="*", default=None, help="只处理指定前缀，如 --only 01 05")
    ap.add_argument("--top-k", type=int, default=3, help="探针检索返回条数（默认 3）")
    ap.add_argument("--probe-only", action="store_true", help="只跑探针检索，不抽取、不写库、不动旧文件")
    args = ap.parse_args()

    # 只跑探针：连 PDF 都不需要，所以放在最前面，跳过存在性检查
    if args.probe_only:
        print("=" * 78)
        print("指南召回探针（只读，不改动任何数据）")
        print("=" * 78)
        from knowledge.milvus_kb import MedicalKnowledgeBase

        _print_probes(run_probes(MedicalKnowledgeBase(), top_k=args.top_k))
        return 0

    selected = [s for s in SOURCES if not args.only or s["prefix"] in args.only]
    if not selected:
        logger.error(f"--only {args.only} 没有匹配到任何 PDF")
        return 2

    missing = [s["filename"] for s in selected if not (RAW_DIR / s["filename"]).exists()]
    if missing:
        logger.error(f"缺少 PDF：{missing}；先跑 scripts/download_medical_guidelines.py")
        return 2

    print("=" * 78)
    print("真实权威指南入库与平滑替换")
    print("=" * 78)
    print(f"PDF 目录   : {RAW_DIR}")
    print(f"文本目录   : {DOC_DIR}")
    print(f"分块参数   : chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"本次处理   : {[s['filename'] for s in selected]}")
    print(f"模式       : {'DRY-RUN（不写库）' if args.dry_run else ('入库 + 不淘汰旧摘要' if args.keep_old else '入库 + 淘汰旧摘要')}")
    print()

    # 延迟导入：dry-run 不需要连库，但要用库里的 _chunk_text，所以仍然实例化
    from knowledge.milvus_kb import MedicalKnowledgeBase

    kb = MedicalKnowledgeBase()
    baseline_total = len(_all_chunks(kb))
    print(f"入库前集合内 chunk 数（按内容去重）：{baseline_total}")
    print()

    # ---- 阶段 1/2：抽取 + 分块 ----
    print("--- 阶段 1/2：抽取、清洗、分块 ---")
    stats: List[Dict[str, Any]] = []
    per_file_rows: Dict[str, List[Dict[str, Any]]] = {}
    for src in selected:
        pdf = RAW_DIR / src["filename"]
        rows, stat = build_chunks(pdf, kb)
        # 绑定标准化元数据
        for i, r in enumerate(rows):
            r["metadata"] = {
                "type": "clinical_guideline",
                "doc_kind": "full_text_pdf",
                "guideline_id": src["guideline_id"],
                "source": src["source"],
                "publisher": src["publisher"],
                "year": src["year"],
                "filename": src["filename"],
                "doc_id": f"clinical_guideline_{src['guideline_id']}",
                "chunk_id": i,
                "total_chunks": len(rows),
                "pages": r["pages"],
                "char_count": len(r["content"]),
            }
        per_file_rows[src["filename"]] = rows
        stats.append(stat)
        print(
            "  %-42s 页=%3d  原始=%7d 字 → 清洗后=%7d 字（去 %.1f%%）  %4d 块（均 %.0f 字，最长 %d）  %.1fs"
            % (
                stat["file"], stat["pages"], stat["raw_chars"], stat["clean_chars"],
                stat["removed_ratio"] * 100, stat["chunks"], stat["avg_chunk"],
                stat["max_chunk"], stat["seconds"],
            )
        )
        shown = [r[:56] for r in stat["running_lines"][:4]]
        tail = " …" if len(stat["running_lines"]) > 4 else ""
        print(
            f"      剔除 {stat['dropped_lines']} 行页码/栏头；接回被抽取器拆散的数字单位 "
            f"{stat['repaired_units']} 处"
        )
        if shown:
            print(f"      栏头样本：{shown}{tail}")
    total_new = sum(len(v) for v in per_file_rows.values())
    print(f"  合计待写入 {total_new} 块")
    print()

    if args.dry_run:
        print("DRY-RUN 结束：未写入数据库，未改动任何文件。")
        return 0

    # ---- 阶段 3：写入（先确认写成功，再动旧数据）----
    print("--- 阶段 3：写入 Milvus ---")
    for fname, rows in per_file_rows.items():
        existing = _count_by_filename(kb, fname)
        if existing and not args.force:
            logger.error(
                f"{fname} 已有 {existing} 块全文在库；确认要重灌请加 --force（会先删这些旧块）"
            )
            return 3
        if existing and args.force:
            removed = _delete_by_filename(kb, fname)
            logger.info(f"  --force：先删除 {fname} 的旧全文 {removed} 块")
    written = 0
    for fname, rows in per_file_rows.items():
        logger.info(f"  写入 {fname}（{len(rows)} 块）")
        written += _insert(kb, rows)
    print(f"  写入完成：{written} 块")

    # ---- 阶段 4：写入校验（不通过就不动旧摘要，避免检索空窗）----
    print()
    print("--- 阶段 4：写入校验 ---")
    verify_ok = True
    for fname, rows in per_file_rows.items():
        got = _count_by_filename(kb, fname)
        ok = got == len(rows)
        verify_ok &= ok
        print(f"  {fname:44s} 期望 {len(rows):4d} 块  实到 {got:4d} 块  {'OK' if ok else '不一致'}")
    if not verify_ok:
        logger.error("写入校验未通过，已中止：不淘汰旧摘要，检索不会出现空窗。")
        return 4
    print(f"  集合内 chunk 数：{baseline_total} → {len(_all_chunks(kb))}")
    print()

    # ---- 阶段 5：淘汰旧摘要（文件归档 + 向量删除）----
    if args.keep_old:
        print("--- 阶段 5：跳过淘汰旧摘要（--keep-old）---")
        print()
    else:
        print("--- 阶段 5：淘汰旧二手摘要 ---")
        keep_before = {k: _count_by_filename(kb, k) for k in MUST_KEEP}
        for e in LEGACY:
            fname = e["filename"]
            src = DOC_DIR / fname
            bak = DOC_DIR / (fname + ".bak")
            vec = _count_by_filename(kb, fname)
            # 1) 文件归档
            if src.exists():
                if bak.exists():
                    print(f"  {fname}: 已存在 .bak，跳过改名（保留现场）")
                else:
                    src.rename(bak)
                    print(f"  {fname}: 文件归档为 .bak")
            else:
                print(f"  {fname}: 文件已不在（可能此前归档过）")
            # 2) 向量删除 —— 只改文件名不会让旧向量消失
            deleted = _delete_by_filename(kb, fname)
            left = _count_by_filename(kb, fname)
            print(
                f"  {fname}: 删除旧向量 {deleted} 块（删前 {vec} 块，删后残留 {left} 块）"
                f"  已被 {e['replaced_by']} 取代"
            )
        print("  保留过渡文档（无对应真指南）：")
        for k in MUST_KEEP:
            print(f"    {k}: {keep_before[k]} 块（删除后 {_count_by_filename(kb, k)} 块）")
        print()

    # ---- 阶段 6：探针检索 ----
    print("--- 阶段 6：探针检索（验证来源与接地）---")
    probes = run_probes(kb, top_k=args.top_k)
    _print_probes(probes)

    # ---- 结果落盘，便于复盘 ----
    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunk_size": CHUNK_SIZE,
        "overlap": CHUNK_OVERLAP,
        "files": stats,
        "written_chunks": written,
        "collection_before": baseline_total,
        "collection_after": len(_all_chunks(kb)),
        "probes": probes,
    }
    out_path = PROJECT_ROOT / "knowledge" / "data" / "ingest_real_guidelines_report.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    print(f"报告已落盘：{out_path.relative_to(PROJECT_ROOT)}")
    print()
    print("=" * 78)
    print(f"完成：新增 {written} 块全文；集合 {baseline_total} → {out['collection_after']} 块")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
