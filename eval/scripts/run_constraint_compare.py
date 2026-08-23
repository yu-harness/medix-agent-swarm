#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CONSTRAINT_ENFORCE=0 vs 1：越权执行 vs 拦截对比（构造场景，不打爆 API）。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# consultation 白名单外；diagnostic 白名单外
FORCE_CASES: List[Tuple[str, str]] = [
    ("consultation_agent", "deep_research"),
    ("consultation_agent", "clinical_guideline"),
    ("consultation_agent", "analyze_symptoms"),
    ("consultation_agent", "disease_code"),
    ("diagnostic_agent", "recommend_lifestyle"),
    ("diagnostic_agent", "deep_research"),
    ("research_agent", "assess_risk"),
    ("research_agent", "analyze_symptoms"),
]


async def run_forced_loop(agent_id: str, tool_name: str, enforce: bool) -> Dict[str, Any]:
    from core.agent_loop import AgentLoop
    from core.llm_client import LLMResponse, ToolCall

    old = os.environ.get("CONSTRAINT_ENFORCE")
    os.environ["CONSTRAINT_ENFORCE"] = "1" if enforce else "0"
    try:
        async def fake_chat_with_tools(messages, tools=None, tool_choice="auto", temperature=0.7):
            has_tool_result = any(m.get("role") == "tool" for m in messages)
            if has_tool_result:
                return LLMResponse(
                    content="【回答】建议休息。\n【免责声明】仅供参考。",
                    tool_calls=[],
                    finish_reason="stop",
                )
            return LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name=tool_name, arguments={"query": "test"})],
                finish_reason="tool_calls",
            )

        agent = MagicMock()
        agent.agent_id = agent_id
        agent.config = {"temperature": 0.7}
        agent.get_system_prompt.return_value = "你是医疗助手"
        agent.format_user_input.return_value = "测试问题"
        agent.get_tools_for_llm.return_value = []
        agent.post_process_result = AsyncMock(side_effect=lambda result, final: result)
        agent.llm_client = MagicMock()
        agent.llm_client.chat_with_tools = AsyncMock(side_effect=fake_chat_with_tools)
        agent.llm_client.create_tool_message = MagicMock(
            side_effect=lambda tool_call_id, tool_name, result: {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": str(result),
            }
        )
        agent.execute_tool = AsyncMock(return_value={"success": True, "mock": True})

        loop = AgentLoop(max_iterations=5, max_tool_calls=2)
        await loop.run(agent, {"question": "测试"}, session_id=None)

        executed = agent.execute_tool.called
        blocked = False
        for c in agent.llm_client.create_tool_message.call_args_list:
            res = c.kwargs.get("result")
            if res is None and c.args:
                res = c.args[2] if len(c.args) >= 3 else None
            if isinstance(res, dict) and res.get("blocked"):
                blocked = True
        return {
            "agent_id": agent_id,
            "tool": tool_name,
            "enforce": enforce,
            "executed": executed,
            "blocked": blocked,
            "outcome": "blocked" if blocked else ("executed_overreach" if executed else "unknown"),
        }
    finally:
        if old is None:
            os.environ.pop("CONSTRAINT_ENFORCE", None)
        else:
            os.environ["CONSTRAINT_ENFORCE"] = old


def validator_matrix() -> List[Dict[str, Any]]:
    from constraints.validator import ConstraintValidator, is_constraint_enforce_enabled

    v = ConstraintValidator()
    rows = []
    for enforce in (False, True):
        old = os.environ.get("CONSTRAINT_ENFORCE")
        os.environ["CONSTRAINT_ENFORCE"] = "1" if enforce else "0"
        try:
            for agent_id, tool in FORCE_CASES:
                r = v.validate_tool_call(agent_id, tool)
                rows.append(
                    {
                        "agent_id": agent_id,
                        "tool": tool,
                        "enforce": enforce,
                        "valid": r.get("valid"),
                        "severity": r.get("severity"),
                        "env_enforce": is_constraint_enforce_enabled(),
                    }
                )
        finally:
            if old is None:
                os.environ.pop("CONSTRAINT_ENFORCE", None)
            else:
                os.environ["CONSTRAINT_ENFORCE"] = old
    return rows


def write_md(path: Path, summary: Dict[str, Any]) -> None:
    w = summary["warn"]
    e = summary["enforce"]
    lines = [
        "# Skill 越权：warn vs enforce",
        "",
        f"- 时间: {summary['finished_at']}",
        f"- 样本: 构造越权 n={summary['n_cases']}（每模式各跑一轮 AgentLoop mock，**不调真实 LLM/API**）",
        "",
        "## 协议",
        "",
        "1. 取 YAML 白名单外 (agent, tool) 对，强制 LLM mock 返回该 tool_call",
        "2. `CONSTRAINT_ENFORCE=0`：期望 severity=warning 且仍 `execute_tool`",
        "3. `CONSTRAINT_ENFORCE=1`：期望 severity=block 且不 `execute_tool`，messages 含 blocked",
        "4. 另跑 `ConstraintValidator.validate_tool_call` 矩阵核对 severity",
        "",
        "## AgentLoop 结果",
        "",
        "| 模式 | 越权仍执行 | 被拦截 | 其它 |",
        "|---|---:|---:|---:|",
        f"| warn (ENFORCE=0) | {w['executed_overreach']} | {w['blocked']} | {w['unknown']} |",
        f"| enforce (ENFORCE=1) | {e['executed_overreach']} | {e['blocked']} | {e['unknown']} |",
        "",
        "## Validator severity",
        "",
        f"- warn 模式 warning 次数: {summary['validator']['warn_warning']}",
        f"- enforce 模式 block 次数: {summary['validator']['enforce_block']}",
        "",
        "## 结论",
        "",
        f"- warn：越权仍执行 **{w['executed_overreach']}/{summary['n_cases']}**",
        f"- enforce：被拦截 **{e['blocked']}/{summary['n_cases']}**",
        "",
        "## 局限",
        "",
        "- 为构造场景，非线上自然越权率；自然触发需另做日志统计",
        "- 未跑真实 DeepSeek，避免打爆 API",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


async def async_main() -> None:
    try:
        from loguru import logger as _logger

        _logger.remove()
        _logger.add(sys.stderr, level="WARNING")
    except Exception:
        pass

    details_warn = []
    details_enforce = []
    for agent_id, tool in FORCE_CASES:
        details_warn.append(await run_forced_loop(agent_id, tool, enforce=False))
        details_enforce.append(await run_forced_loop(agent_id, tool, enforce=True))

    def count(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        return {
            "executed_overreach": sum(1 for r in rows if r["outcome"] == "executed_overreach"),
            "blocked": sum(1 for r in rows if r["outcome"] == "blocked"),
            "unknown": sum(1 for r in rows if r["outcome"] == "unknown"),
        }

    vrows = validator_matrix()
    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "n_cases": len(FORCE_CASES),
        "cases": [{"agent_id": a, "tool": t} for a, t in FORCE_CASES],
        "warn": count(details_warn),
        "enforce": count(details_enforce),
        "details_warn": details_warn,
        "details_enforce": details_enforce,
        "validator": {
            "warn_warning": sum(
                1 for r in vrows if (not r["enforce"]) and r.get("severity") == "warning"
            ),
            "enforce_block": sum(
                1 for r in vrows if r["enforce"] and r.get("severity") == "block"
            ),
            "rows": vrows,
        },
    }

    out = Path(ROOT) / "eval" / "results" / "constraint_enforce_compare.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_md(out, summary)
    js = out.with_suffix(".json")
    js.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[done] warn_exec={summary['warn']['executed_overreach']} "
        f"enforce_block={summary['enforce']['blocked']} -> {out}"
    )


def main() -> None:
    argparse.ArgumentParser().parse_args()
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
