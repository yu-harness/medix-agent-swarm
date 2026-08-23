"""P0 multiturn smoke: same session T1→T2."""
import asyncio
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from swarm import process_with_swarm


def _cli_extra_disclaimer(answer: str, disclaimer: str | None) -> bool:
    """Mirrors main.py: print disclaimer only if answer lacks one."""
    answer_has = (
        "【免责声明】" in answer
        or "仅供参考" in answer
        or "不能替代" in answer
    )
    return bool(disclaimer) and not answer_has


def _cli_extra_suggestions(answer: str, suggestions) -> bool:
    return bool(suggestions) and "【核心建议】" not in answer


FAKE_RECALL = re.compile(
    r"(您之前问过|上一轮已给出|刚才已经.*建议|之前已经.*饮食)"
)


async def main():
    session_id = f"p0-mt-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    t1 = "我有高血压"
    t2 = "那饮食方面要注意什么？"

    r1 = await process_with_swarm(t1, session_id=session_id)
    r2 = await process_with_swarm(t2, session_id=session_id)

    a1 = r1.get("answer") or ""
    a2 = r2.get("answer") or ""

    checks = {
        "t2_no_fake_recall": not bool(FAKE_RECALL.search(a2)),
        "t2_mentions_diet": any(k in a2 for k in ("饮食", "盐", "低钠", "少盐", "钾")),
        "t2_has_disclaimer_once": a2.count("【免责声明】") <= 1 and (
            "免责" in a2 or "仅供参考" in a2 or "不能替代" in a2
        ),
        "cli_no_double_disclaimer_t2": not _cli_extra_disclaimer(a2, r2.get("disclaimer")),
        "cli_no_double_suggestions_t2": not _cli_extra_suggestions(a2, r2.get("suggestions")),
        "t1_fullish_or_ok": len(a1) > 40,
    }

    print("session:", session_id)
    print("T1 len:", len(a1), "T2 len:", len(a2))
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    print("\n--- T2 excerpt (first 800 chars) ---")
    print(a2[:800])

    out = ROOT / "eval" / "docs" / "multiturn_p0_fix.md"
    lines = [
        "# Multiturn P0 验收",
        "",
        f"- session: `{session_id}`",
        f"- T1: {t1}",
        f"- T2: {t2}",
        "",
        "## Checks",
        "",
    ]
    for k, v in checks.items():
        lines.append(f"- {'PASS' if v else 'FAIL'}: `{k}`")
    lines += [
        "",
        "## T2 excerpt",
        "",
        "```",
        a2[:1200],
        "```",
        "",
        "## 手动复测",
        "",
        "```bash",
        "python main.py",
        "# 同会话输入：我有高血压 → 那饮食方面要注意什么？",
        "# 或：python eval/scripts/run_multiturn_p0_smoke.py",
        "```",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\nWrote", out)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    asyncio.run(main())
