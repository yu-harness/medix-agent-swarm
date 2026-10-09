"""
Demo：验证"新增 Worker Agent 自动被 LeadAgent 识别"（2026-08-24 改造）

验证内容：
1. register_worker() 后，新 Worker 自动进入 worker_pool
2. _get_agent_by_id() 动态解析到新 Worker（无需改硬编码映射）
3. LeadAgent 提示词「可用的 Worker Agents」段自动包含新 Worker 及其中文画像
4. 无 yaml role 的 Worker 也能被识别（回退用 capabilities 兜底）
5. （可选 --online）真实 LLM 分解时 LeadAgent 可把任务分配给新 Worker

用法：
    python examples/demo_new_worker.py            # 离线验证（推荐，无需网络）
    python examples/demo_new_worker.py --online   # 在线验证真实分配
"""
import asyncio
import sys
from pathlib import Path

# 让脚本无论从仓库根还是 examples/ 下运行都能正确导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import loguru
from loguru import logger

# 压掉框架自身的 INFO/DEBUG 日志，只保留 demo 的 print 输出
loguru.logger.remove()
loguru.logger.add(sys.stderr, level="WARNING")

from agents.base_agent import BaseAgent
from agents.skill_registry_mixin import SkillRegistryMixin
from swarm.swarm_coordinator import SwarmCoordinator


# ---------------------------------------------------------------
# 1. 定义一个"新" Worker：营养与饮食专家（模仿现有 Agent 的写法）
# ---------------------------------------------------------------
class NutritionAgent(BaseAgent, SkillRegistryMixin):
    """营养与饮食专家（Demo 新增 Worker，已配置 agent_constraints.yaml role 画像）"""

    def __init__(self, config=None):
        default_config = {
            "model": "openai_compatible",
            "max_iterations": 5,
            "temperature": 0.7,
            "description": "营养与饮食专家，负责膳食方案与营养素建议",
        }
        super().__init__(
            agent_id="nutrition_agent",
            config=config or default_config,
        )
        # 能力标签（Swarm 协作用）
        self.set_capabilities(["nutrition_advice", "diet_planning", "food_safety"])

    def get_system_prompt(self) -> str:
        return """你是一位注册营养师。专注于膳食方案设计与营养素补充建议。

可用 Skills：search_knowledge / recommend_lifestyle / search_history / search_similar_cases。

**Skills 原则**：
- 涉及营养/饮食/补剂的问题：先 search_knowledge，再给具体可执行建议
- 最多 2-3 个 Skills，然后必须给出最终答案

**回答模式**：给出问题理解 → 具体膳食建议（含食物举例与忌口）→ 营养补充剂说明 → 就医/咨询指征 → 免责声明。

**内容策略**：
- 不做疾病诊断、不替代医生；急危重症必须建议立即就医
- 补剂建议说明证据等级，避免绝对化

输出格式：
【回答】
...
【核心建议】
1. ...
【免责声明】
以上信息仅供参考，不能替代专业医生的诊断和治疗。如有疑虑，请及时就医。
"""

    def register_tools(self):
        """按 YAML 白名单注册本 Agent 的 Skills（共享实现，来自 SkillRegistryMixin）"""
        self.register_all_skills()


class FallbackWorker(BaseAgent, SkillRegistryMixin):
    """
    未配置 yaml role 的"裸" Worker：
    用于验证"即使没写 role，也能靠 capabilities 兜底被 LeadAgent 识别"。
    """

    def __init__(self, config=None):
        default_config = {
            "model": "openai_compatible",
            "max_iterations": 5,
            "temperature": 0.7,
            "description": "示例扩展 Worker",
        }
        super().__init__(
            agent_id="demo_fallback_worker",
            config=config or default_config,
        )
        self.set_capabilities(["mental_health_support", "stress_management"])

    def get_system_prompt(self) -> str:
        return "你是一位心理健康支持助手。提供情绪疏导与压力管理建议。"

    def register_tools(self):
        self.register_all_skills()


# ---------------------------------------------------------------
# 2. 验证逻辑
# ---------------------------------------------------------------
def verify_registered(coordinator, agent_id: str, expect_ok: bool = True):
    instance = coordinator._get_agent_by_id(agent_id)
    ok = instance is not None
    status = "PASS" if ok == expect_ok else "FAIL"
    print(f"  [{status}] _get_agent_by_id('{agent_id}') → {instance.__class__.__name__ if instance else None}")
    return ok == expect_ok


def verify_lead_prompt(coordinator, agent_id: str, expect_present: bool = True):
    prompt = coordinator.lead_agent._get_system_prompt()
    present = agent_id in prompt
    status = "PASS" if present == expect_present else "FAIL"
    print(f"  [{status}] LeadAgent 提示词包含 '{agent_id}' → {present}")
    return present == expect_present


def show_worker_section(coordinator):
    print("\n---- LeadAgent「可用的 Worker Agents」段（节选） ----")
    prompt = coordinator.lead_agent._get_system_prompt()
    start = prompt.find("## 可用的 Worker Agents")
    end = prompt.find("## 任务分配策略")
    section = prompt[start:end].strip()
    # 只打印每个 Agent 块的标题行（### N. ...）和 Agent ID
    for line in section.splitlines():
        if line.strip().startswith("###") or "**Agent ID**" in line:
            print("   " + line.strip())
    print("-----------------------------------------------")


async def online_check(coordinator):
    """真实 LLM 分解：看 LeadAgent 是否会把任务分配给新 Worker。"""
    question = "我最近有点贫血，想了解饮食上应该怎么补铁，有什么忌口？"
    print(f"\n[--online] 真实 LLM 分解问题：{question}")
    try:
        result = await coordinator.lead_agent.assess_and_decompose(question)
        subtasks = result.get("subtasks", [])
        print(f"  LeadAgent 输出 {len(subtasks)} 个子任务：")
        for t in subtasks:
            mark = "  [新Worker被分配]" if t.get("assigned_agent") == "nutrition_agent" else ""
            print(f"    - assigned_agent={t.get('assigned_agent')}{mark} | {t.get('description', '')[:40]}")
        assigned_to_new = any(t.get("assigned_agent") == "nutrition_agent" for t in subtasks)
        print(f"  结论：{'[OK] 新 Worker 被成功分配任务' if assigned_to_new else '[INFO] 未分到新 Worker（LLM 决策，提示词已可见；可加强角色画像或明确提问匹配）'}")
    except Exception as e:
        print(f"  [WARN] 在线调用失败（可能无网络/无 API key）：{type(e).__name__}: {e}")


async def main():
    online = "--online" in sys.argv

    print("=" * 62)
    print("Demo：新增 Worker 自动被 LeadAgent 识别")
    print("=" * 62)

    # 3. 构建 SwarmCoordinator（初始只有 3 个内置 Worker）
    coordinator = SwarmCoordinator()
    print(f"\n初始化 worker_pool：{[w.agent_id for w in coordinator.worker_pool]}")
    print(f"LeadAgent 已同步画像：{coordinator.lead_agent._worker_profiles and [p['agent_id'] for p in coordinator.lead_agent._worker_profiles]}")

    # 4. 注册新 Worker（一行搞定：入池 + 注记忆 + 重同步 LeadAgent 提示词）
    print("\n>>> register_worker(NutritionAgent())")
    coordinator.register_worker(NutritionAgent())

    print("\n[验证 1] 注册后 worker_pool 成员：")
    print(f"  {[w.agent_id for w in coordinator.worker_pool]}")

    print("\n[验证 2] 动态路由解析：")
    all_ok = True
    for aid in ["consultation_agent", "diagnostic_agent", "research_agent", "nutrition_agent"]:
        all_ok &= verify_registered(coordinator, aid)

    print("\n[验证 3] LeadAgent 提示词自动包含新 Worker：")
    for aid in ["consultation_agent", "diagnostic_agent", "research_agent", "nutrition_agent"]:
        all_ok &= verify_lead_prompt(coordinator, aid)

    show_worker_section(coordinator)

    # 5. 未配置 yaml role 的 Worker 也能被识别（capabilities 兜底）
    print("\n>>> register_worker(FallbackWorker())   # 无 yaml role 的裸 Worker")
    coordinator.register_worker(FallbackWorker())
    print("\n[验证 4] 无 role 画像也能被识别（兜底）：")
    all_ok &= verify_registered(coordinator, "demo_fallback_worker")
    all_ok &= verify_lead_prompt(coordinator, "demo_fallback_worker")
    show_worker_section(coordinator)

    # 6. 在线验证（可选）
    if online:
        await online_check(coordinator)

    print("\n" + "=" * 62)
    print("验证结论：" + ("[OK] 全部通过：新增 Worker 可被 LeadAgent 自动识别并可分配任务" if all_ok else "[FAIL] 存在失败项，请检查"))
    print("=" * 62)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
