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
        from agents import ConsultationAgent
        agent = ConsultationAgent()
        skills = agent.skill_registry.get_all()
        self.assertGreaterEqual(len(skills), 7)
        tools = agent.get_tools_for_llm()
        self.assertEqual(len(tools), len(skills))
        names = {t["function"]["name"] for t in tools}
        for expected in ("search_knowledge", "assess_risk", "analyze_symptoms"):
            self.assertIn(expected, names)


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

    def test_missing_disclaimer(self):
        result = self.validator.validate_output("consultation_agent", "高血压需要低盐饮食。")
        self.assertFalse(result.get("valid"))
        self.assertIn("缺少免责声明", result.get("violations", []))

    def test_autofix_disclaimer(self):
        fixed = self.fixer.fix_missing_disclaimer("高血压需要低盐饮食。")
        self.assertTrue("免责声明" in fixed or "仅供参考" in fixed)

    def test_autofix_high_risk(self):
        fixed = self.fixer.fix_high_risk_warning("您的胸痛可能是心绞痛。")
        self.assertTrue("就医" in fixed or "120" in fixed)

    def test_required_agents_high_risk(self):
        agents = self.validator.get_required_agents("突然胸痛和呼吸困难")
        self.assertIsInstance(agents, list)


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

        async def fake_chat_with_tools(messages, tools=None, tool_choice="auto", temperature=0.7):
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
        self.assertLessEqual(loop.tool_call_count, 2)
        self.assertGreaterEqual(loop.tool_call_count, 1)
        self.assertTrue(result["answer"])


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
