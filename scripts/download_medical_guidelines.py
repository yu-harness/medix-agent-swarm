#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下载 / 收录权威医学指南，供医学知识库离线使用。

目标（4 份）：
    1. 中国高血压防治指南（2024年修订版）
    2. 中国2型糖尿病防治指南（2020年版）
    3. 中国血脂管理指南（2023年）
    4. 胸痛中心急危重症患者院前医疗急救转运专家共识

用法：
    python scripts/download_medical_guidelines.py              # 处理全部
    python scripts/download_medical_guidelines.py --only 1 3    # 只处理第 1、3 份
    python scripts/download_medical_guidelines.py --list        # 只列目标与官方地址
    python scripts/download_medical_guidelines.py --force       # 覆盖已存在的文件

行为约定：
    1. 只从「公开页面 / 开放镜像」取 PDF 或已转录文本，不碰需要账号的库；
    2. 直链取不到时，会解析候选页面里暴露的 .pdf 链接再试一次；
    3. 403、跳转登录页、验证码、返回 HTML 冒充 PDF，一律按失败处理，不落脏文件；
    4. 失败时写 <id>.placeholder，内容含标题、官方浏览地址、失败原因与手动下载指引；
    5. 已存在的文件默认不覆盖（手动放进去的文件不会被冲掉）；
    6. 结束后打印就绪清单与待补清单。

版权提示：指南原文受版权保护。请勿把下载到的 PDF 提交到公开仓库，
本仓库 .gitignore 已屏蔽 knowledge/data/raw_guidelines/ 下的 pdf/文本。
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "knowledge" / "data" / "raw_guidelines"

# 浏览器 UA：多数国内站点对非浏览器 UA 直接 403
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

TIMEOUT = 30
MIN_PDF_BYTES = 50 * 1024        # 小于 50KB 的「PDF」基本是错误页
LOGIN_HINTS = ("登录", "验证码", "请先登录", "会员", "sign in", "log in", "captcha", "403 forbidden")

TARGETS = [
    {
        "id": "01_hypertension_2024",
        "title": "中国高血压防治指南（2024年修订版）",
        "official": [
            "中国高血压联盟 http://www.chl.org.cn/",
            "中华心血管病杂志（指南与共识栏目）",
            "国家心血管病中心 https://www.nccd.org.cn/",
        ],
        "candidates": [
            ("page", "https://www.yantai.gov.cn/art/2025/7/1/art_66786_3274344.html"),
            ("page", "https://www.hnysfww.com/mobile/article.php?id=4031"),
        ],
        "note": "政府网站与药事服务网属公开转载，正式引用以中国高血压联盟/杂志社版本为准",
    },
    {
        "id": "02_t2dm_2020",
        "title": "中国2型糖尿病防治指南（2020年版）",
        "official": [
            "中华医学会糖尿病学分会 http://www.diabetes.org.cn/",
            "中华糖尿病杂志（指南与共识栏目）",
        ],
        "candidates": [
            ("direct", "https://www.hnysfww.com/data/article/1619222032934605040.pdf"),
            ("page", "https://www.hnysfww.com/article.php?id=2633"),
            ("page", "https://www.hncdc.com.cn/sitesources/sjk/page_pc/jkjy/jkzt/mxbyjyfkz/jsbz/article1fafd1cad8c7407a8fe3c869a886832c.html"),
        ],
        "note": "河南省疾控、湖南药事服务网为公开转载页",
    },
    {
        "id": "03_lipid_2023",
        "title": "中国血脂管理指南（2023年）",
        "official": [
            "中华心血管病杂志（2023年第51卷第3期）",
            "中国血脂管理指南修订联合专家委员会",
        ],
        "candidates": [
            ("page", "https://www.hnysfww.com/mobile/article.php?id=3598"),
            ("page", "https://www.yixiuqixie.com/doc/7946.html"),
        ],
        "note": "该指南以期刊论文形式发布，公开镜像多为转载",
    },
    {
        "id": "04_chest_pain_transfer_consensus",
        "title": "胸痛中心急危重症患者院前医疗急救转运专家共识",
        "official": [
            "中华急诊医学杂志 https://www.cem.org.cn/",
            "中国胸痛中心总部 https://www.chinacpc.org/",
        ],
        "candidates": [
            ("page", "http://www.cem.org.cn/"),
            ("page", "http://www.cem.org.cn/zine/content/id/8797"),
            ("page", "https://www.cem.org.cn/"),
        ],
        "note": (
            "相邻文献（内容相关但不是同一份，需你判断是否采用）：\n"
            "    急性胸痛急诊诊疗专家共识（2019，中华医学会急诊医学分会）\n"
            "    https://csem.cma.org.cn/attach/0/5f104921b18243f98543fc5769ba58cb.pdf\n"
            "    危重症患者院际转运专家共识\n"
            "    https://icuguideline.com/wp-content/uploads/2022/06/危重症患者院际转运专家共识.pdf\n"
            "    拿不到原文时的替代：见第 5 条《急性胸痛急诊诊疗专家共识》（已落本地）"
        ),
    },
    {
        "id": "05_acute_chest_pain_consensus_2019",
        "title": "急性胸痛急诊诊疗专家共识（2019）",
        "official": [
            "中华急诊网文章页 http://www.cem.org.cn/zine/content/id/8797",
            "中华急诊医学杂志 2019年4月（中华医学会急诊医学分会 + 中国医促会胸痛分会）",
        ],
        "candidates": [
            # 该站的规律：文章页 content/id/<ID> 与 PDF 下载口 down/id/<ID>/flag/<ID> 共用同一个 id；
            # 注意用 http，https 会 TLS 握手失败
            ("direct", "http://www.cem.org.cn/zine/down/id/8797/flag/8797"),
        ],
        "note": "第 4 条（胸痛中心急危重症患者院前医疗急救转运专家共识）的替代文献：原文未找到公开版本，这份讲的是胸痛急诊诊疗全流程，含院前急救与分流转运。",
    },
]


def _looks_like_login(text: str) -> bool:
    low = text.lower()
    return any(hint.lower() in low for hint in LOGIN_HINTS)


def _find_pdf_links(html: str, base_url: str) -> list[str]:
    """从页面里挑出可能的 PDF 直链。"""
    links = re.findall(r'''(?:href|src)\s*=\s*["']([^"']+\.pdf[^"']*)["']''', html, re.I)
    seen, out = set(), []
    for link in links:
        absolute = urljoin(base_url, link)
        if absolute not in seen:
            seen.add(absolute)
            out.append(absolute)
    return out[:5]


def _download_pdf(url: str, referer: str | None = None) -> tuple[bytes | None, str]:
    """下载并校验是否为真实 PDF。返回 (内容, 失败原因)。"""
    headers = dict(HEADERS)
    if referer:
        headers["Referer"] = referer
    try:
        resp = requests.get(url, headers=headers, timeout=TIMEOUT, allow_redirects=True, stream=True)
    except requests.exceptions.SSLError:
        return None, "TLS 握手失败（站点证书或网络阻断）"
    except requests.exceptions.Timeout:
        return None, f"请求超时（>{TIMEOUT}s）"
    except requests.exceptions.RequestException as e:
        return None, f"{type(e).__name__}: {e}"

    if resp.status_code == 403:
        return None, "403 拒绝访问（站点反爬）"
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}"
    if "login" in resp.url.lower() or "passport" in resp.url.lower():
        return None, "被重定向到登录页"

    body = resp.content
    if not body.startswith(b"%PDF"):
        head = body[:2000].decode("utf-8", errors="ignore")
        if _looks_like_login(head):
            return None, "返回的是登录/验证页，不是 PDF"
        return None, f"返回内容不是 PDF（Content-Type={resp.headers.get('Content-Type', '?')}）"
    if len(body) < MIN_PDF_BYTES:
        return None, f"PDF 体积异常（{len(body) // 1024}KB），疑似错误页"
    return body, ""


def _save(path: Path, body: bytes, force: bool) -> str:
    if path.exists() and not force:
        return "已存在，跳过"
    path.write_bytes(body)
    return f"已保存 {len(body) / 1024:.0f}KB"


def _clear_placeholder(target: dict) -> None:
    """下载成功或本地已有文件时，清掉同 id 的占位文件，避免残留误导。"""
    placeholder = OUT_DIR / f"{target['id']}.placeholder"
    if placeholder.exists():
        placeholder.unlink()
        logger.info(f"[{target['id']}] 已清理占位文件")


def _write_placeholder(target: dict, reasons: list[str]) -> Path:
    path = OUT_DIR / f"{target['id']}.placeholder"
    lines = [
        f"# 待手动补充：{target['title']}",
        "",
        "自动下载未成功，请手动获取后放到本目录（保持文件名以该 id 开头即可被收录）：",
        f"    目标文件名：{target['id']}.pdf 或 {target['id']}.md",
        "",
        "## 官方浏览地址",
    ]
    lines += [f"    - {u}" for u in target["official"]]
    lines += ["", "## 各来源失败原因"]
    lines += [f"    - {r}" for r in reasons] if reasons else ["    - 未找到可用的公开来源"]
    if target.get("note"):
        lines += ["", "## 备注", f"    {target['note']}"]
    lines += [
        "",
        "## 手动下载提示",
        "    - 从官方站点或机构公众号获取最稳妥；第三方文库常需登录或付费",
        "    - 拿到 PDF 后直接放进本目录，再跑一次 knowledge/scripts/import_hardcoded_data.py 入库",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def process(target: dict, force: bool, sleep_sec: float) -> tuple[bool, list[str]]:
    """处理一份指南。返回 (是否就绪, 失败原因列表)。"""
    pdf_path = OUT_DIR / f"{target['id']}.pdf"
    md_path = OUT_DIR / f"{target['id']}.md"
    if (pdf_path.exists() or md_path.exists()) and not force:
        logger.info(f"[{target['id']}] 本地已有文件，跳过下载")
        _clear_placeholder(target)
        return True, []

    reasons: list[str] = []
    for kind, url in target["candidates"]:
        logger.info(f"[{target['id']}] 尝试 {kind}: {url}")
        if kind == "direct":
            body, reason = _download_pdf(url)
            if body:
                logger.info(f"[{target['id']}] {_save(pdf_path, body, force)}")
                _clear_placeholder(target)
                return True, []
            reasons.append(f"{url} → {reason}")
            time.sleep(sleep_sec)
            continue

        # kind == "page"：先看页面，再试页面里暴露的 pdf 直链
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        except requests.exceptions.SSLError:
            reasons.append(f"{url} → TLS 握手失败")
            time.sleep(sleep_sec)
            continue
        except requests.exceptions.RequestException as e:
            reasons.append(f"{url} → {type(e).__name__}")
            time.sleep(sleep_sec)
            continue

        if resp.status_code == 403:
            reasons.append(f"{url} → 403 拒绝访问（站点反爬）")
            time.sleep(sleep_sec)
            continue
        if resp.status_code != 200:
            reasons.append(f"{url} → HTTP {resp.status_code}")
            time.sleep(sleep_sec)
            continue

        html = resp.text
        if _looks_like_login(html[:4000]):
            reasons.append(f"{url} → 页面含登录/验证提示")
            time.sleep(sleep_sec)
            continue

        pdf_links = _find_pdf_links(html, resp.url)
        if not pdf_links:
            reasons.append(f"{url} → 页面可访问，但未找到 PDF 直链")
            time.sleep(sleep_sec)
            continue

        for pdf_url in pdf_links:
            body, reason = _download_pdf(pdf_url, referer=resp.url)
            if body:
                logger.info(f"[{target['id']}] {_save(pdf_path, body, force)}  ← {pdf_url}")
                _clear_placeholder(target)
                return True, []
            reasons.append(f"{pdf_url} → {reason}")
            time.sleep(sleep_sec)

    return False, reasons


def main() -> int:
    ap = argparse.ArgumentParser(description="下载/收录权威医学指南")
    ap.add_argument("--only", nargs="*", type=int, help="只处理指定序号（1-4）")
    ap.add_argument("--list", action="store_true", help="只打印目标与官方地址")
    ap.add_argument("--force", action="store_true", help="覆盖已存在文件")
    ap.add_argument("--sleep", type=float, default=1.5, help="每次尝试之间的间隔秒数")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        for i, t in enumerate(TARGETS, 1):
            print(f"{i}. {t['title']}")
            for u in t["official"]:
                print(f"   官方: {u}")
        return 0

    picked = TARGETS if not args.only else [TARGETS[i - 1] for i in args.only if 1 <= i <= len(TARGETS)]
    print(f"目标目录: {OUT_DIR}")
    print(f"处理 {len(picked)} 份指南\n")

    ready, pending = [], []
    for target in picked:
        ok, reasons = process(target, args.force, args.sleep)
        if ok:
            ready.append(target)
        else:
            path = _write_placeholder(target, reasons)
            pending.append((target, reasons, path))
            print(f"  ✗ {target['title']} —— 已写占位文件 {path.name}")

    print("\n=== 就绪文件 ===")
    if ready:
        for t in ready:
            for suffix in (".pdf", ".md"):
                p = OUT_DIR / f"{t['id']}{suffix}"
                if p.exists():
                    print(f"  ✓ {p.name}  {p.stat().st_size / 1024:.0f}KB  {t['title']}")
    else:
        print("  （无）")

    print("\n=== 需你手动补充 ===")
    if pending:
        for t, reasons, path in pending:
            print(f"  - {t['title']}")
            for u in t["official"]:
                print(f"      官方地址: {u}")
            for r in reasons[:3]:
                print(f"      失败原因: {r}")
            print(f"      占位说明: {path.name}")
    else:
        print("  （无）")

    print(f"\n目录内文件共 {len(list(OUT_DIR.iterdir()))} 个：{OUT_DIR}")
    return 0 if not pending else 1


if __name__ == "__main__":
    sys.exit(main())
