#!/usr/bin/env python3
"""MediX Agent Swarm 可运行测试套件（本地优先，LLM 可 mock）"""
import asyncio
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.skill_registry import SkillRegistry, SkillParameter
from core.agent_loop import AgentLoop
from core.llm_client import LLMResponse, ToolCall
from constraints import ConstraintValidator
from constraints.validator import is_constraint_enforce_enabled
from validation import AutoFixer
from memory import ShortTermMemory, MemoryEntropyManager
from swarm import SharedContext, EventType
from swarm.events import Event
from swarm.swarm_coordinator import SwarmCoordinator


def check_api_key() -> bool:
    try:
        from config import LLM_CONFIG
        from openai import OpenAI
        key = LLM_CONFIG.get("api_key", "")
        if not key or key.startswith("sk-xxx") or "替换" in key:
            return False
        client = OpenAI(api_key=key, base_url=LLM_CONFIG["base_url"])
        client.chat.completions.create(
            model=LLM_CONFIG["model_name"],
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        return True
    except Exception:
        return False


API_KEY_VALID = False


class TestSkillRegistry(unittest.TestCase):
    def test_register_and_openai_format(self):
        reg = SkillRegistry()

        def echo(query: str) -> Dict[str, Any]:
            return {"success": True, "query": query}

        reg.register(
            name="echo_skill",
            function=echo,
            description="回显查询",
            parameters=[
                SkillParameter(name="query", type="string", description="查询文本", required=True)
            ],
        )
        self.assertIn("echo_skill", reg.get_all())
        tools = reg.to_openai_format()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["name"], "echo_skill")
        self.assertIn("query", tools[0]["function"]["parameters"]["properties"])
        self.assertEqual(tools[0]["function"]["parameters"]["required"], ["query"])

    def test_execute_sync_skill(self):
        reg = SkillRegistry()

        def add(a: int, b: int) -> Dict[str, Any]:
            return {"success": True, "sum": a + b}

        reg.register(
            name="add",
            function=add,
            description="加法",
            parameters=[
                SkillParameter(name="a", type="number", description="a", required=True),
                SkillParameter(name="b", type="number", description="b", required=True),
            ],
        )

        async def _run():
            return await reg.execute("add", a=2, b=3)

        result = asyncio.run(_run())
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("sum"), 5)

    def test_agent_skills_registered(self):
        """注册应按 YAML 白名单裁剪（而非全量注册 + 运行时警告）。"""
        from agents import ConsultationAgent
        from constraints.validator import get_allowed_tools

        agent = ConsultationAgent()
        skills = agent.skill_registry.get_all()
        allowed = set(get_allowed_tools("consultation_agent"))

        # 注册的 Skill 应恰好等于白名单：不多（越权工具不该出现在工具列表里）
        self.assertEqual(set(skills), allowed)
        self.assertNotIn("analyze_symptoms", skills)
        self.assertNotIn("deep_research", skills)

        # 工具列表与注册表一致
        tools = agent.get_tools_for_llm()
        self.assertEqual(len(tools), len(skills))

    def test_agent_prompt_matches_registry(self):
        """system prompt 里的 Skills 必须由注册表渲染，不能另抄一份清单。"""
        from agents import ConsultationAgent

        agent = ConsultationAgent()
        rendered = agent.render_available_skills()
        for name in agent.skill_registry.get_all():
            self.assertIn(name, rendered)
        # 未注册的 Skill 不应出现在提示词中
        self.assertNotIn("deep_research", rendered)
        # 提示词确实被渲染进了 system prompt
        self.assertIn(rendered.splitlines()[0], agent.get_system_prompt())


class TestKnowledgeBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from knowledge.milvus_kb import MedicalKnowledgeBase
        MedicalKnowledgeBase._instance = None
        cls.kb = MedicalKnowledgeBase()
        count = cls.kb.count_documents()
        results = cls.kb.search("高血压", top_k=3)
        if not results:
            docs_dir = project_root / "knowledge" / "data" / "documents"
            if docs_dir.exists():
                from knowledge.scripts.import_hardcoded_data import load_documents_from_directory
                docs = load_documents_from_directory(docs_dir)
                if docs:
                    MedicalKnowledgeBase._instance = None
                    cls.kb = MedicalKnowledgeBase()
                    if cls.kb.milvus_client.has_collection(cls.kb.collection_name):
                        cls.kb.milvus_client.drop_collection(cls.kb.collection_name)
                        cls.kb.milvus_client.create_collection(
                            collection_name=cls.kb.collection_name,
                            dimension=cls.kb.embedding_dim,
                            metric_type="COSINE",
                            auto_id=True,
                        )
                    cls.kb.add_documents(docs)
                    cls.kb.milvus_client.load_collection(cls.kb.collection_name)
        cls.count = cls.kb.count_documents()

    def test_search_hypertension(self):
        results = self.kb.search("高血压", top_k=5)
        self.assertGreater(len(results), 0, "高血压检索应有结果")
        text = " ".join(r.get("content", "") for r in results)
        self.assertTrue(any(k in text for k in ("高血压", "血压", "低盐", "钠")))

    def test_search_lifestyle(self):
        results = self.kb.search("生活方式 饮食 运动", top_k=5)
        self.assertGreater(len(results), 0)
        text = " ".join(r.get("content", "") for r in results)
        self.assertTrue(any(k in text for k in ("饮食", "运动", "生活", "睡眠")))

    def test_singleton(self):
        from knowledge.milvus_kb import MedicalKnowledgeBase
        kb2 = MedicalKnowledgeBase()
        self.assertIs(self.kb, kb2)


class TestConstraintsAndAutoFixer(unittest.TestCase):
    def setUp(self):
        self.validator = ConstraintValidator()
        self.fixer = AutoFixer()

    def test_tool_call_allowed(self):
        result = self.validator.validate_tool_call("consultation_agent", "search_knowledge")
        self.assertTrue(result.get("valid"))

    def test_tool_call_disallowed_blocks_by_default(self):
        """默认应硬拦：只警告不拦截的约束等于没有约束。"""
        import os
        old = os.environ.pop("CONSTRAINT_ENFORCE", None)
        try:
            result = self.validator.validate_tool_call("consultation_agent", "deep_research")
            self.assertFalse(result.get("valid"))
            self.assertEqual(result.get("severity"), "block")
            self.assertIn("请改用", result.get("reason", ""))
            self.assertTrue(is_constraint_enforce_enabled())
        finally:
            if old is not None:
                os.environ["CONSTRAINT_ENFORCE"] = old

    def test_tool_call_disallowed_warn_opt_out(self):
        """显式 CONSTRAINT_ENFORCE=0 时退回 warn 模式（用于评估影响面）。"""
        import os
        old = os.environ.get("CONSTRAINT_ENFORCE")
        os.environ["CONSTRAINT_ENFORCE"] = "0"
        try:
            result = self.validator.validate_tool_call("consultation_agent", "deep_research")
            self.assertFalse(result.get("valid"))
            self.assertEqual(result.get("severity"), "warning")
            self.assertFalse(is_constraint_enforce_enabled())
        finally:
            if old is None:
                os.environ.pop("CONSTRAINT_ENFORCE", None)
            else:
                os.environ["CONSTRAINT_ENFORCE"] = old

    def test_tool_call_disallowed_enforce(self):
        import os
        old = os.environ.get("CONSTRAINT_ENFORCE")
        os.environ["CONSTRAINT_ENFORCE"] = "1"
        try:
            result = self.validator.validate_tool_call("consultation_agent", "deep_research")
            self.assertFalse(result.get("valid"))
            self.assertEqual(result.get("severity"), "block")
            self.assertTrue(is_constraint_enforce_enabled())
        finally:
            if old is None:
                os.environ.pop("CONSTRAINT_ENFORCE", None)
            else:
                os.environ["CONSTRAINT_ENFORCE"] = old

    def test_missing_disclaimer(self):
        result = self.validator.validate_output("consultation_agent", "高血压需要低盐饮食。")
        self.assertFalse(result.get("valid"))
        self.assertIn("缺少免责声明", result.get("violations", []))

    # ===== 医疗安全：risk_level 一等公民 =====

    def test_high_risk_forces_emergency_guidance_even_without_keywords(self):
        """核心用例：risk_level=high 时，即使回答里一个高危关键词都没有，也必须补就医提示。

        这是"扫输出"方案的致命漏洞——用户说胸痛、模型没复述，旧逻辑就漏判了。
        新逻辑依据结构化 risk_level，不受模型是否复述影响。
        """
        answer = "建议您多休息，清淡饮食，观察一下情况。"
        result = self.validator.validate_output(
            "diagnostic_agent", answer, risk_level="high"
        )
        self.assertFalse(result.get("valid"))
        self.assertIn("高危情况未建议就医", result.get("violations", []))
        self.assertIn("add_emergency_warning", result.get("auto_fixable", []))

        fixed = self.fixer.fix_output(
            answer, result.get("auto_fixable", []), risk_level="high"
        )
        self.assertIn("120", fixed)
        self.assertIn("就医", fixed)

    def test_emergency_level_also_forces_guidance(self):
        """emergency 等级同样必须补提示。"""
        answer = "请保持安静。"
        result = self.validator.validate_output(
            "diagnostic_agent", answer, risk_level="emergency"
        )
        self.assertIn("高危情况未建议就医", result.get("violations", []))

    def test_low_risk_without_keywords_no_false_alarm(self):
        """低风险且无高危关键词时不应误报（避免告警疲劳）。"""
        answer = "建议多饮水、注意休息。如症状加重请及时就医。"
        result = self.validator.validate_output(
            "diagnostic_agent", answer, risk_level="low"
        )
        self.assertNotIn("高危情况未建议就医", result.get("violations", []))

    def test_high_risk_already_has_guidance_not_flagged(self):
        """回答里已有就医引导时不应重复告警。"""
        answer = "您的情况需要尽快到医院就诊，必要时拨打120。"
        result = self.validator.validate_output(
            "diagnostic_agent", answer, risk_level="high"
        )
        self.assertNotIn("高危情况未建议就医", result.get("violations", []))

    def test_risk_level_helpers_ordering(self):
        """风险等级取最大值，且无法识别的等级降级为 low。"""
        from constraints.validator import max_risk_level, normalize_risk_level

        self.assertEqual(max_risk_level("low", "high", "medium"), "high")
        self.assertEqual(max_risk_level("emergency", "low"), "emergency")
        self.assertEqual(max_risk_level("low", "medium"), "medium")
        self.assertEqual(normalize_risk_level("HIGH"), "high")
        self.assertEqual(normalize_risk_level("unknown"), "low")
        self.assertEqual(normalize_risk_level(None), "low")

    def test_autofix_disclaimer(self):
        fixed = self.fixer.fix_missing_disclaimer("高血压需要低盐饮食。")
        self.assertTrue("免责声明" in fixed or "仅供参考" in fixed)

    def test_autofix_high_risk(self):
        fixed = self.fixer.fix_high_risk_warning("您的胸痛可能是心绞痛。")
        self.assertTrue("就医" in fixed or "120" in fixed)

    # ===== Swarm 路由硬约束（高危必须含 diagnostic_agent）=====

    def test_required_agents_rules(self):
        """agent_selection_rules 是纯规则，不依赖 LLM 分解是否自觉。"""
        from constraints.validator import get_required_agents

        self.assertIn("diagnostic_agent", get_required_agents("突然胸痛和呼吸困难"))
        self.assertIn("diagnostic_agent", get_required_agents("最近心悸、容易昏厥"))
        self.assertIn("consultation_agent", get_required_agents("高血压饮食注意什么"))
        self.assertIn("research_agent", get_required_agents("高血压最新诊疗指南"))

    def test_enforce_required_agents_swarm_mode_adds_subtask(self):
        """Swarm 模式下，高危问题即使只被 LLM 分给 consultation，也必须补上 diagnostic。"""
        from swarm.swarm_coordinator import SwarmCoordinator

        coord = SwarmCoordinator(enable_swarm=True)
        subtasks = [
            {"type": "consultation_agent_task", "description": "回答用户问题",
             "assigned_agent": "consultation_agent"}
        ]
        out = coord._enforce_required_agents(subtasks, "突然胸痛和呼吸困难")
        agents = {t["assigned_agent"] for t in out}
        self.assertIn("diagnostic_agent", agents)
        diag = [t for t in out if t["assigned_agent"] == "diagnostic_agent"][0]
        self.assertIn("安全规则", diag["description"])

    def test_enforce_required_agents_single_mode_reassigns(self):
        """单 Agent 模式下，高危问题应把任务改派给 diagnostic_agent。"""
        from swarm.swarm_coordinator import SwarmCoordinator

        coord = SwarmCoordinator(enable_swarm=False)
        subtasks = [
            {"type": "consultation_agent_task", "description": "回答用户问题",
             "assigned_agent": "consultation_agent"}
        ]
        out = coord._enforce_required_agents(subtasks, "突然胸痛和呼吸困难")
        self.assertEqual(out[0]["assigned_agent"], "diagnostic_agent")

    def test_enforce_required_agents_no_force_when_present(self):
        """diagnostic 已在子任务里时，不应重复强制加入。"""
        from swarm.swarm_coordinator import SwarmCoordinator

        coord = SwarmCoordinator(enable_swarm=True)
        subtasks = [
            {"type": "diagnostic_agent_task", "description": "风险评估",
             "assigned_agent": "diagnostic_agent"}
        ]
        out = coord._enforce_required_agents(subtasks, "突然胸痛和呼吸困难")
        self.assertEqual(len(out), 1)

class TestShortTermMemoryAndEntropy(unittest.TestCase):
    def test_short_term_memory(self):
        ShortTermMemory._instance = None
        stm = ShortTermMemory(storage_type="memory")
        sid = "suite-stm-001"
        stm.create_session(sid, metadata={"test": True})
        stm.add_message(sid, "user", "我头痛")
        stm.add_message(sid, "assistant", "建议休息")
        msgs = stm.get_recent_messages(sid, limit=10)
        self.assertEqual(len(msgs), 2)
        stm.clear_session(sid)

    def test_session_ttl_expiry(self):
        """空闲超过 TTL 的会话应被清除（sessions 原本只增不删，会 OOM）。"""
        import time

        ShortTermMemory._instance = None
        stm = ShortTermMemory(storage_type="memory", session_ttl_seconds=1, max_sessions=100)
        sid = "suite-ttl-001"
        stm.create_session(sid)
        stm.add_message(sid, "user", "我头痛")
        self.assertIsNotNone(stm.get_session(sid))

        time.sleep(1.2)
        self.assertIsNone(stm.get_session(sid), "过期会话应返回 None 并被移除")

    def test_session_max_sessions_lru(self):
        """会话数超过上限时按最久未用淘汰，且稳态恰好等于上限（不多不少）。"""
        ShortTermMemory._instance = None
        stm = ShortTermMemory(storage_type="memory", session_ttl_seconds=0, max_sessions=5)

        for i in range(12):
            stm.create_session(f"suite-lru-{i}")

        self.assertEqual(len(stm.sessions), 5, "应恰好保留 5 个（上限），不能是 6 个")
        # 保留的应是最新的 5 个
        self.assertIn("suite-lru-11", stm.sessions)
        self.assertNotIn("suite-lru-0", stm.sessions)

    def test_entropy_dedup_compress(self):
        manager = MemoryEntropyManager()
        messages = [
            {"role": "user", "content": "高血压怎么办？"},
            {"role": "assistant", "content": "建议低盐饮食。"},
            {"role": "user", "content": "高血压怎么办？"},
            {"role": "assistant", "content": "建议低盐饮食。"},
        ]
        deduped = manager.deduplicate_messages(messages)
        self.assertEqual(len(deduped), 2)

        long_msgs = []
        for i in range(30):
            long_msgs.append({"role": "user", "content": f"问题 {i}"})
            long_msgs.append({"role": "assistant", "content": f"回答 {i}"})
        compressed = manager.compress_session_history(long_msgs, max_messages=10)
        self.assertLess(len(compressed), len(long_msgs))
        self.assertEqual(compressed[-1]["content"], "回答 29")

        entropy = manager.estimate_entropy(long_msgs)
        self.assertIn(entropy["entropy_level"], ("low", "medium", "high"))

    def test_entropy_sessions_and_cleanup(self):
        manager = MemoryEntropyManager()
        sessions = [
            {"memory_id": "1", "content": "问题：高血压怎么办？\n回答：低盐", "timestamp": datetime.now()},
            {"memory_id": "2", "content": "问题：高血压怎么办？\n回答：低盐", "timestamp": datetime.now()},
            {"memory_id": "3", "content": "问题：感冒了怎么办？\n回答：多喝水", "timestamp": datetime.now()},
        ]
        deduped = manager.deduplicate_sessions(sessions)
        self.assertEqual(len(deduped), 2)

        now = datetime.now()
        old_memories = [
            {"memory_id": "m1", "content": "旧1", "timestamp": now - timedelta(days=200)},
            {"memory_id": "m2", "content": "旧2", "timestamp": now - timedelta(days=120)},
            {"memory_id": "m3", "content": "新", "timestamp": now - timedelta(days=10)},
        ]
        cleaned = manager.cleanup_old_memories(old_memories, max_age_days=90)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned[0]["memory_id"], "m3")


class TestAgentLoopMaxToolCalls(unittest.IsolatedAsyncioTestCase):
    async def test_max_tool_calls_enforced(self):
        call_log: List[str] = []

        # **kwargs 兜底 agent_loop 传入的 fallback 等参数，避免调用签名漂移导致测试假失败
        async def fake_chat_with_tools(messages, tools=None, tool_choice="auto", temperature=0.7, **kwargs):
            n_user_force = sum(
                1 for m in messages
                if m.get("role") == "user" and "信息检索" in str(m.get("content", ""))
            )
            if n_user_force > 0 or len(call_log) >= 2:
                return LLMResponse(
                    content="【回答】建议就医休息。\n【免责声明】仅供参考。",
                    tool_calls=[],
                    finish_reason="stop",
                )
            call_log.append("tool")
            return LLMResponse(
                content=None,
                tool_calls=[ToolCall(id=f"c{len(call_log)}", name="assess_risk", arguments={"symptoms": "头痛"})],
                finish_reason="tool_calls",
            )

        agent = MagicMock()
        agent.agent_id = "consultation_agent"
        agent.config = {"temperature": 0.7}
        agent.get_system_prompt.return_value = "你是医疗助手"
        agent.format_user_input.return_value = "我头痛"
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
        agent.execute_tool = AsyncMock(return_value={"success": True, "risk": "low"})

        loop = AgentLoop(max_iterations=10, max_tool_calls=2)
        result = await loop.run(agent, {"question": "我头痛"}, session_id=None)

        self.assertIn("answer", result)
        # 工具调用次数由本次 run() 的结果返回，不再挂在共享的 loop 实例上
        # （挂在 self 上会被并发请求互相清零/累加，导致 max_tool_calls 限额失效）
        self.assertIn("tool_calls", result)
        self.assertLessEqual(result["tool_calls"], 2)
        self.assertGreaterEqual(result["tool_calls"], 1)
        self.assertTrue(result["answer"])


class TestAgentLoopConstraintEnforce(unittest.IsolatedAsyncioTestCase):
    async def test_disallowed_tool_blocked_when_enforce(self):
        import os
        old = os.environ.get("CONSTRAINT_ENFORCE")
        os.environ["CONSTRAINT_ENFORCE"] = "1"
        try:
            async def fake_chat_with_tools(messages, tools=None, tool_choice="auto", temperature=0.7, **kwargs):
                has_tool_result = any(m.get("role") == "tool" for m in messages)
                if has_tool_result:
                    return LLMResponse(
                        content="【回答】建议休息。\n【免责声明】仅供参考。",
                        tool_calls=[],
                        finish_reason="stop",
                    )
                return LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="c1", name="deep_research", arguments={"query": "高血压"})],
                    finish_reason="tool_calls",
                )

            agent = MagicMock()
            agent.agent_id = "consultation_agent"
            agent.config = {"temperature": 0.7}
            agent.get_system_prompt.return_value = "你是医疗助手"
            agent.format_user_input.return_value = "高血压怎么办"
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
            agent.execute_tool = AsyncMock(return_value={"success": True})

            loop = AgentLoop(max_iterations=5, max_tool_calls=2)
            result = await loop.run(agent, {"question": "高血压怎么办"}, session_id=None)

            agent.execute_tool.assert_not_called()
            self.assertIn("answer", result)
            tool_msgs = [
                c for c in agent.llm_client.create_tool_message.call_args_list
            ]
            self.assertTrue(tool_msgs)
            blocked_result = tool_msgs[0].kwargs.get("result") or tool_msgs[0][1].get("result")
            if blocked_result is None:
                blocked_result = tool_msgs[0].args[2] if len(tool_msgs[0].args) >= 3 else tool_msgs[0].kwargs["result"]
            self.assertTrue(blocked_result.get("blocked"))
            self.assertIn("请改用", blocked_result.get("error", ""))
        finally:
            if old is None:
                os.environ.pop("CONSTRAINT_ENFORCE", None)
            else:
                os.environ["CONSTRAINT_ENFORCE"] = old


class TestSwarmBasics(unittest.TestCase):
    def test_shared_context(self):
        ctx = SharedContext(session_id="suite-ctx-001")
        ctx.data["symptoms"] = ["头痛"]
        self.assertEqual(ctx.data["symptoms"], ["头痛"])
        ctx.publish_event(Event(
            type=EventType.CONTEXT_UPDATED,
            source_agent="test",
            data={"key": "symptoms"},
        ))
        events = ctx.get_events(event_type=EventType.CONTEXT_UPDATED)
        self.assertGreater(len(events), 0)

    def test_coordinator_init(self):
        coord = SwarmCoordinator(enable_swarm=True)
        self.assertTrue(coord.enable_swarm)
        self.assertEqual(len(coord.worker_pool), 3)
        self.assertIsNotNone(coord.consultation_agent)
        self.assertIsNotNone(coord.lead_agent)

    def test_agent_capabilities(self):
        from agents import DiagnosticAgent, ResearchAgent
        diag = DiagnosticAgent()
        research = ResearchAgent()
        self.assertTrue(len(diag.get_capabilities()) > 0)
        self.assertTrue(len(research.get_capabilities()) > 0)


@unittest.skipUnless(False, "placeholder; toggled at runtime")
class TestE2ESmoke(unittest.IsolatedAsyncioTestCase):
    async def test_simple_consult(self):
        from agents import ConsultationAgent
        agent = ConsultationAgent()
        result = await agent.process({"question": "多喝水对健康有什么好处？"})
        self.assertIn("answer", result)
        self.assertTrue(result["answer"])


def _build_suite(api_valid: bool) -> unittest.TestSuite:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestSkillRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestKnowledgeBase))
    suite.addTests(loader.loadTestsFromTestCase(TestConstraintsAndAutoFixer))
    suite.addTests(loader.loadTestsFromTestCase(TestShortTermMemoryAndEntropy))
    suite.addTests(loader.loadTestsFromTestCase(TestAgentLoopMaxToolCalls))
    suite.addTests(loader.loadTestsFromTestCase(TestAgentLoopConstraintEnforce))
    suite.addTests(loader.loadTestsFromTestCase(TestSwarmBasics))
    if api_valid:
        class TestE2ELive(unittest.IsolatedAsyncioTestCase):
            async def test_simple_consult(self):
                from agents import ConsultationAgent
                agent = ConsultationAgent()
                result = await agent.process({"question": "多喝水对健康有什么好处？"})
                self.assertIn("answer", result)
                self.assertTrue(len(result["answer"]) > 10)

            async def test_swarm_route_simple(self):
                from swarm import process_with_swarm
                result = await process_with_swarm("如何预防感冒？")
                self.assertIn("answer", result)
                self.assertIn("swarm_enabled", result)

        suite.addTests(loader.loadTestsFromTestCase(TestE2ELive))
    return suite


def main():
    global API_KEY_VALID
    print("=" * 60)
    print("MediX Agent Swarm 测试套件")
    print("=" * 60)
    print("检查 API Key ...")
    API_KEY_VALID = check_api_key()
    print(f"API Key: {'有效' if API_KEY_VALID else '无效（E2E 跳过，LLM 相关用 mock）'}")
    print()

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(_build_suite(API_KEY_VALID))

    print()
    print("=" * 60)
    print(f"通过: {result.testsRun - len(result.failures) - len(result.errors)}/{result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 60)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
