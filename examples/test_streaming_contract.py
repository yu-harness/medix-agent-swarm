#!/usr/bin/env python3
"""流式链路契约测试（不调用真实 LLM，不花钱）。

为什么单独有这个文件：流式参数要从 API 一路穿到模型调用，中间经过
process_with_swarm → SwarmCoordinator.process →（单 Agent 路由 / Swarm 路由的
_process_with_swarm）→ BaseAgent.process → AgentLoop.run → LLMClient。

新增一层 hop 却忘了给它加参数时，只会在「走到那条路由时」抛 NameError ——
平时的测试全绿，真实请求却间歇性失败。2026-10-09 就踩过一次：
Swarm 路由漏传，20 条实测里 4 条报 `NameError: name 'on_delta' is not defined`。
本文件把这条契约固定下来：任何一层不再接受这些参数，这里立刻红。
"""
import asyncio
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.base_agent import BaseAgent  # noqa: E402
from core.agent_loop import AgentLoop  # noqa: E402
from core.llm_client import LLMClient, StreamStats  # noqa: E402
from swarm import process_with_swarm  # noqa: E402
from swarm.lead_agent import LeadAgent  # noqa: E402
from swarm.swarm_coordinator import SwarmCoordinator  # noqa: E402

STREAM_PARAMS = ("on_delta", "stream")
ROOT_PARAMS = ("on_delta", "stream")


def _accepts(func, *names):
    params = inspect.signature(func).parameters
    return [name for name in names if name not in params]


# ============================================================================
# 契约一：每一层都能接住流式参数
# ============================================================================

def test_every_hop_accepts_stream_params():
    """失败即说明某一层漏了参数——真实请求会在该路由上抛 NameError。"""
    hops = [
        ("process_with_swarm", process_with_swarm, ROOT_PARAMS),
        ("SwarmCoordinator.process", SwarmCoordinator.process, STREAM_PARAMS),
        ("SwarmCoordinator._process_with_swarm",
         SwarmCoordinator._process_with_swarm, STREAM_PARAMS),
        ("BaseAgent.process", BaseAgent.process, STREAM_PARAMS),
        ("BaseAgent.run_loop", BaseAgent.run_loop, STREAM_PARAMS),
        ("AgentLoop.run", AgentLoop.run, STREAM_PARAMS),
        ("LeadAgent.synthesize_results", LeadAgent.synthesize_results, STREAM_PARAMS),
    ]
    problems = []
    for label, func, names in hops:
        missing = _accepts(func, *names)
        if missing:
            problems.append(f"{label} 缺少参数 {missing}")
    assert not problems, "流式参数未穿透到：" + "；".join(problems)


def test_llm_client_stream_entrypoints():
    """Lead 汇总走 chat_stream，Agent 循环走 chat_with_tools_stream。"""
    assert callable(getattr(LLMClient, "chat_stream", None)), "缺少 chat_stream"
    assert callable(getattr(LLMClient, "chat_with_tools_stream", None)), \
        "缺少 chat_with_tools_stream"
    assert "on_delta" in inspect.signature(LLMClient.chat_stream).parameters
    assert "on_delta" in inspect.signature(LLMClient.chat_with_tools_stream).parameters


# ============================================================================
# 契约二：default 参数必须是「不流式」，避免改造悄悄改变老调用方的行为
# ============================================================================

def test_stream_defaults_are_off():
    for label, func in [
        ("process_with_swarm", process_with_swarm),
        ("SwarmCoordinator.process", SwarmCoordinator.process),
        ("BaseAgent.process", BaseAgent.process),
        ("AgentLoop.run", AgentLoop.run),
        ("LeadAgent.synthesize_results", LeadAgent.synthesize_results),
    ]:
        params = inspect.signature(func).parameters
        assert params["on_delta"].default is None, f"{label}.on_delta 默认值应为 None"
        assert params["stream"].default is False, f"{label}.stream 默认值应为 False"


# ============================================================================
# 契约三：chat_stream 真的按片段回调，且 TTFT 有打点
# ============================================================================

def _fake_client(chunks):
    class FakeCompletions:
        async def create(self, **kwargs):
            async def gen():
                for chunk in chunks:
                    await asyncio.sleep(0)
                    yield chunk
            return gen()

    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))


def _text_chunk(text):
    delta = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=None)])


def test_chat_stream_forwards_deltas_and_times_first_token():
    async def scenario():
        llm = LLMClient()
        llm.client = _fake_client([_text_chunk("一"), _text_chunk("二"), _text_chunk("三")])
        seen = []

        async def on_delta(text):
            seen.append(text)

        stats = StreamStats()
        content = await llm.chat_stream(
            messages=[{"role": "user", "content": "hi"}],
            on_delta=on_delta,
            stats=stats,
        )
        return content, seen, stats

    content, seen, stats = asyncio.run(scenario())
    assert content == "一二三", f"拼接结果不对: {content!r}"
    assert seen == ["一", "二", "三"], f"增量回调不对: {seen!r}"
    assert stats.ttft_ms is not None and stats.total_ms is not None, "TTFT 未打点"


if __name__ == "__main__":
    test_every_hop_accepts_stream_params()
    test_llm_client_stream_entrypoints()
    test_stream_defaults_are_off()
    test_chat_stream_forwards_deltas_and_times_first_token()
    print("流式链路契约测试全部通过")
