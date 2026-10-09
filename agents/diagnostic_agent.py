"""
DiagnosticAgent：症状诊断推理 Agent

这是第一个 WorkerAgent 实现，展示如何：
1. 参与 Swarm 协作
2. 自主认领任务
3. 调用医疗工具
4. 将结果写入 SharedContext
"""
from typing import Dict, Any, Optional
from loguru import logger

from .base_agent import BaseAgent
from .skill_registry_mixin import SkillRegistryMixin
from core import LLMClient


class DiagnosticAgent(BaseAgent, SkillRegistryMixin):
    """
    诊断 Agent

    职责：
    - 复杂症状的鉴别诊断
    - 多系统关联分析
    - 诊断思路推理（类似医生的临床思维）

    能力标签：
    - symptom_analysis
    - differential_diagnosis
    - clinical_reasoning
    """

    def __init__(
        self,
        agent_id: str = "diagnostic_agent",
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[LLMClient] = None
    ):
        config = config or {}
        config.setdefault('max_iterations', 5)

        super().__init__(agent_id, config, llm_client)

        # 设置能力标签（Swarm 协作用）
        self.set_capabilities([
            "symptom_analysis",
            "differential_diagnosis",
            "clinical_reasoning",
            "multi_system_analysis"
        ])

    def register_tools(self):
        """按 YAML 白名单注册本 Agent 的 Skills（共享实现，来自 SkillRegistryMixin）"""
        self.register_all_skills()


    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        # 可用 Skills 由注册表动态渲染，与 YAML 白名单同源（单一来源）
        order = self.render_skill_refs("assess_risk", "analyze_symptoms", sep=" → ")
        order_block = ""
        if order:
            order_block = f"**调用顺序（症状题必须）**：\n{order}"
            if self.has_skill("search_knowledge"):
                order_block += "；信息不足再 search_knowledge"
            order_block += "\n\n"
        return f"""你是专业的诊断 Agent（DiagnosticAgent）。职责：症状模式分析、鉴别诊断思路、风险分层。永远不做确诊。

**原则**：常见病优先，不漏危险疾病；明确建议检查；可执行的下一步。

**Skills**（{len(self.skill_registry.get_all())} 个，由注册表自动渲染，与 YAML 白名单同源）：
{self.render_available_skills()}

{order_block}最多 2-3 次 Skill，然后输出最终思路

**输出模式**：
A. 首轮/无历史 → 完整结构：
【问题理解】【风险评估】【症状分析】【鉴别诊断】【可执行建议】【建议检查】【就医指征】【推理过程】【免责声明】
B. 有历史且本轮细化追问 → 增量：一句承接（本轮诉求+已知人群/主诉）→ 可执行增量（就医/护理等）→ 一句就医红线 → 免责至多一次。禁止假回忆、禁止再造整套分段。有【本会话已知信息】时禁止「尚未提供症状」，必须继承锚点。
"""

    def format_user_input(self, input_data: Dict[str, Any]) -> str:
        question = input_data.get('question', input_data.get('query', ''))
        context = input_data.get('context') or {}
        parts = []

        anchor = context.get('session_anchor') or ""
        if anchor:
            parts.append(f"【本会话已知信息】{anchor}")
            parts.append(
                "【强制规则】已知信息不空时：禁止「尚未提供症状/请补充症状」；"
                "追问必须继承上述人群与主诉；就医/护理建议须针对该主诉。"
            )

        historical_cases = context.get('historical_cases')
        if historical_cases:
            case_lines = []
            for i, case in enumerate(historical_cases[:3], 1):
                summary = case.get('summary', case) if isinstance(case, dict) else case
                case_lines.append(f"{i}. {summary}")
            parts.append(
                "参考案例（仅供启发，勿当作本会话用户事实）：\n"
                + "\n".join(case_lines)
            )

        skip_keys = {'recent_history', 'historical_cases', 'is_followup', 'session_anchor'}
        other = {k: v for k, v in context.items() if k not in skip_keys and v is not None}
        if other:
            parts.append("背景信息：\n" + "\n".join(f"{k}: {v}" for k, v in other.items()))

        if context.get('is_followup') or context.get('recent_history') or anchor:
            parts.append(
                "【本轮模式：增量回答】细化追问（含就医/护理等）：一句承接 → 可执行增量 → 就医红线 → 免责至多一次。"
                "禁止假回忆、整套复述、声称尚未提供症状。"
            )

        parts.append(f"用户问题：{question}")
        return "\n".join(parts)

    async def post_process_result(
        self,
        result: Dict[str, Any],
        final_response: str
    ) -> Dict[str, Any]:
        """
        结果后处理：提取结构化诊断信息

        这里可以添加更复杂的解析逻辑
        """
        # 尝试提取风险等级
        risk_level = "unknown"
        if "风险等级" in final_response:
            if "高" in final_response or "HIGH" in final_response:
                risk_level = "high"
            elif "中" in final_response or "MEDIUM" in final_response:
                risk_level = "medium"
            elif "低" in final_response or "LOW" in final_response:
                risk_level = "low"

        result.update({
            "risk_level": risk_level,
            "diagnosis_provided": True
        })

        return result


# 便捷函数
async def diagnose(question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    便捷函数：快速使用 DiagnosticAgent

    Args:
        question: 症状描述
        context: 额外上下文（年龄、既往史等）

    Returns:
        诊断结果
    """
    agent = DiagnosticAgent()
    input_data = {'question': question}
    if context:
        input_data['context'] = context

    return await agent.process(input_data)
