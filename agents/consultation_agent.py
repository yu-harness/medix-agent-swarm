"""
健康咨询Agent
支持 Skills 调用
"""
from typing import Dict, Any
from loguru import logger
import re

from .base_agent import BaseAgent
from .skill_registry_mixin import SkillRegistryMixin


class ConsultationAgent(BaseAgent, SkillRegistryMixin):
    """
    健康咨询Agent
    通过 Skills 调用底层工具
    """

    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            "model": "openai_compatible",
            "max_iterations": 5,
            "temperature": 0.8,
            "description": "健康咨询Agent，提供通用医疗咨询和健康建议"
        }

        config = config or default_config
        super().__init__(
            agent_id="consultation_agent",
            config=config
        )

        # 设置能力标签（Swarm 协作用）
        self.set_capabilities([
            "general_health_advice",
            "risk_assessment",
            "symptom_triage"
        ])

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一位专业的医疗健康咨询顾问。提供准确、可执行的健康建议。

可用 Skills：
1. search_knowledge 2. recommend_lifestyle 3. assess_risk 4. analyze_symptoms
5. disease_code 6. clinical_guideline 7. deep_research 8. search_history 9. search_similar_cases

**Skills 原则**：
- 涉及疾病/用药/饮食/护理的问题：必须先 search_knowledge（可再加 recommend_lifestyle 或 assess_risk）
- 最多 2-3 个 Skills，然后必须给出最终答案
- 极简单寒暄可直接答

**回答模式（二选一）**：

A. **首轮 / 无对话历史 → 完整六段（缺一不可）**：
1. 问题理解：用1-2句复述用户诉求
2. 相关知识：病因分型或机制要点（能区分病毒/细菌等时要写清）
3. 风险分层：低/中/高/紧急 + 依据
4. 可执行建议：居家措施 + 常见处理方向（生活方式；社区常见外用药/口服药类别或通用名可写，强调遵医嘱；忌口与护理要具体）
5. 就医指征：何时门诊/急诊、建议科室
6. 免责声明

B. **有对话历史且本轮是细化追问 → 增量模式（禁止再造整套六段）**：
1. 一句承接：只锚定用户本轮诉求 + 已知人群/主诉（勿回顾「您之前问过XX」「上一轮已给出完整建议」）
2. 可执行增量：直接给本轮细化点（饮食/用药/运动/就医判断/居家护理等实操），勿无必要复述机制或全套风险分层
3. 一句就医红线
4. 免责声明至多一次
禁止倒打一耙式假回忆；禁止双免责。
**有【本会话已知信息】时强制**：禁止「尚未提供症状/请补充症状」；追问必须继承锚点人群与主诉；护理勿写成无关成人通用病建议。

**内容策略**：
- 优先给可落地要点，避免空泛科普长文
- 民间偏方：写“民间认为可能有一定辅助作用，证据有限，不能替代规范治疗”，勿绝对否定或绝对承诺
- 不做确诊、不替代医生；急危重症必须建议立即就医

输出格式：
【回答】
（按当前模式组织）

【核心建议】
1. ...
2. ...

【免责声明】
以上信息仅供参考，不能替代专业医生的诊断和治疗。如有疑虑，请及时就医。
"""

    def register_tools(self):
        """注册所有 9 个 Skills（共享实现，来自 SkillRegistryMixin）"""
        self.register_all_skills()

    def format_user_input(self, input_data: Dict[str, Any]) -> str:
        """格式化用户输入（注入会话锚点；不 dump 整段 recent_history）"""
        question = input_data.get('question', '')
        session_id = input_data.get('session_id', '')
        context = input_data.get('context') or {}

        parts = []

        if session_id:
            parts.append(f"[系统信息] 当前会话ID: {session_id}")

        anchor = context.get('session_anchor') or ""
        if anchor:
            parts.append(f"【本会话已知信息】{anchor}")
            parts.append(
                "【强制规则】已知信息不空时：禁止「尚未提供症状/请补充症状」；"
                "追问必须继承上述人群与主诉；就医/护理建议须针对该主诉，勿换成无关成人通用病建议。"
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
            context_str = "\n".join([f"{k}: {v}" for k, v in other.items()])
            parts.append(f"背景信息：\n{context_str}")

        is_followup = bool(context.get('is_followup') or context.get('recent_history') or anchor)
        if is_followup:
            parts.append(
                "【本轮模式：增量回答】有对话历史且本轮为细化追问（含就医/护理/饮食等）。"
                "一句承接（锚定本轮诉求+已知人群/主诉）→ 可执行增量 → 一句就医红线 → 免责至多一次。"
                "禁止假回忆；禁止复述整套六段；禁止声称尚未提供症状。"
            )

        parts.append(f"用户问题：{question}")

        return "\n".join(parts)

    async def post_process_result(
        self,
        result: Dict[str, Any],
        final_response: str
    ) -> Dict[str, Any]:
        """
        后处理：从最终响应中提取结构化信息
        """
        # 提取核心建议
        suggestions = []
        suggestion_pattern = r'【核心建议】\s*\n((?:\d+\.\s*.+\n?)+)'
        match = re.search(suggestion_pattern, final_response)

        if match:
            suggestion_text = match.group(1)
            suggestion_lines = re.findall(r'\d+\.\s*(.+)', suggestion_text)
            suggestions = [s.strip() for s in suggestion_lines if s.strip()]

        # 提取免责声明
        disclaimer_pattern = r'【免责声明】\s*\n(.+)'
        disclaimer_match = re.search(disclaimer_pattern, final_response)
        disclaimer = disclaimer_match.group(1) if disclaimer_match else \
            "⚠️ 以上信息仅供参考，不能替代专业医生的诊断和治疗。如有疑虑，请及时就医。"

        result.update({
            'suggestions': suggestions[:5],  # 最多5条
            'disclaimer': disclaimer
        })

        return result


# 便捷函数
async def consult(question: str, **kwargs) -> Dict[str, Any]:
    """快捷咨询函数"""
    agent = ConsultationAgent()
    return await agent.process({'question': question, **kwargs})
