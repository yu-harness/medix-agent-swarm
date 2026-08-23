"""儿科多轮上下文锚点验收：同 session T1→T2。"""
import asyncio
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from swarm import process_with_swarm

MISSING_SYMPTOM = re.compile(
    r"(尚未提供(具体)?症状|请补充(具体)?症状|未提供.*症状|缺少.*症状描述)"
)
ADULT_GENERIC = re.compile(
    r"(高血压|限盐|低钠饮食|成人原发性|收缩压|舒张压)"
)


async def main():
    session_id = f"ctx-mt-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    t1 = "孩子反复咳嗽两周了"
    t2 = "需要去医院吗？有什么护理建议？"

    r1 = await process_with_swarm(t1, session_id=session_id)
    r2 = await process_with_swarm(t2, session_id=session_id)

    a1 = r1.get("answer") or ""
    a2 = r2.get("answer") or ""

    child_ok = any(k in a2 for k in ("儿童", "孩子", "小儿", "宝宝", "幼儿"))
    cough_ok = "咳嗽" in a2
    no_missing = not bool(MISSING_SYMPTOM.search(a2))
    # 允许偶尔提到鉴别，但不得以成人高血压式通用饮食护理为主
    adult_hits = len(ADULT_GENERIC.findall(a2))
    care_ok = any(k in a2 for k in ("护理", "就医", "医院", "儿科", "观察", "休息", "湿度", "雾化"))
    not_adult_main = adult_hits <= 1 or (child_ok and cough_ok and care_ok and adult_hits < 4)

    checks = {
        "t1_ok": len(a1) > 40,
        "t2_mentions_child": child_ok,
        "t2_mentions_cough": cough_ok,
        "t2_no_missing_symptom": no_missing,
        "t2_care_or_hospital": care_ok,
        "t2_not_adult_htn_main": not_adult_main,
    }

    print("session:", session_id)
    print("T1 len:", len(a1), "T2 len:", len(a2))
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    print("\n--- T2 excerpt (first 900 chars) ---")
    print(a2[:900])

    out = ROOT / "eval" / "docs" / "multiturn_context_fix.md"
    lines = [
        "# Multiturn 上下文锚点验收",
        "",
        "## 改动要点",
        "",
        "1. `short_term.extract_session_anchor`：从用户消息提取人群/主诉锚点",
        "2. `swarm_coordinator`：注入 `session_anchor`，子任务强制「承接：…」",
        "3. `lead_agent`：分解/汇总继承锚点；禁止「尚未提供症状」",
        "4. `base_agent.process_subtask`：Swarm Worker 带上 session_id + context",
        "5. consultation/diagnostic：format_user_input 写入已知信息与强制规则",
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
        "## 复测",
        "",
        "```bash",
        "cd medix-agent-swarm",
        "python eval/scripts/run_multiturn_context_smoke.py",
        "```",
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\nWrote", out)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    asyncio.run(main())
