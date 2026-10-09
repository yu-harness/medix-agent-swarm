"""
约束验证器
运行时检查 Agent 行为是否违反约束

基于 Harness Engineering 原则：
- 显式化约束
- 运行时验证
- 自动修复（可选）
"""
import os
from typing import Dict, Any, List, Optional
import yaml
from pathlib import Path
from loguru import logger


_AGENT_CONSTRAINTS_CACHE: Optional[Dict[str, Any]] = None


def load_agent_constraints() -> Dict[str, Any]:
    """加载 agent_constraints.yaml（带缓存），作为 Agent 角色/能力的单一来源。"""
    global _AGENT_CONSTRAINTS_CACHE
    if _AGENT_CONSTRAINTS_CACHE is None:
        path = Path(__file__).parent / "agent_constraints.yaml"
        with open(path, 'r', encoding='utf-8') as f:
            _AGENT_CONSTRAINTS_CACHE = yaml.safe_load(f) or {}
    return _AGENT_CONSTRAINTS_CACHE


def get_agent_role(agent_id: str) -> Dict[str, Any]:
    """获取某个 Agent 的角色画像（display/specialties/scenarios），供 LeadAgent 动态生成提示词。"""
    agents = load_agent_constraints().get('agents', {}) or {}
    return dict(agents.get(agent_id, {}).get('role', {}) or {})


def get_allowed_tools(agent_id: str) -> List[str]:
    """模块级便捷函数：读取某 Agent 的 Skill 白名单。

    与 ConstraintValidator.get_allowed_tools 共用同一份缓存，保证单一来源。
    返回空列表表示该 Agent 未配置白名单（语义：不限制）。
    """
    agents = load_agent_constraints().get('agents', {}) or {}
    return list((agents.get(agent_id, {}) or {}).get('allowed_tools', []) or [])


# ===================== Swarm 路由硬约束 =====================
_SWARM_CONSTRAINTS_CACHE: Optional[Dict[str, Any]] = None


def load_swarm_constraints() -> Dict[str, Any]:
    """加载 swarm_constraints.yaml（带缓存）。"""
    global _SWARM_CONSTRAINTS_CACHE
    if _SWARM_CONSTRAINTS_CACHE is None:
        swarm_path = Path(__file__).parent / "swarm_constraints.yaml"
        with open(swarm_path, "r", encoding="utf-8") as f:
            _SWARM_CONSTRAINTS_CACHE = yaml.safe_load(f)
    return _SWARM_CONSTRAINTS_CACHE


def get_required_agents(question: str) -> List[str]:
    """根据 agent_selection_rules 判断问题必须包含哪些 Agent（模块级，带缓存）。

    这是 Harness 层的硬约束，**不依赖 LLM 分解是否自觉**——
    高危症状（胸痛/呼吸困难…）必须让 diagnostic_agent 参与风险评估，
    指南/最新进展类必须让 research_agent 参与。
    """
    try:
        swarm = load_swarm_constraints().get("swarm", {}) or {}
        rules = swarm.get("agent_selection_rules", []) or []
    except Exception as e:
        logger.error(f"Failed to load agent_selection_rules: {e}")
        return []

    required = []
    for rule in rules:
        must_include = rule.get("must_include", []) or []
        if any(s in question for s in (rule.get("if_symptoms") or [])):
            required.extend(must_include)
            logger.info(
                f"高危信号命中，必须包含: {must_include}（{rule.get('reason', '')}）"
            )
        if any(k in question for k in (rule.get("if_keywords") or [])):
            required.extend(must_include)
            logger.info(
                f"关键词命中，必须包含: {must_include}（{rule.get('reason', '')}）"
            )
    return list(set(required))


# ===================== 风险等级（一等公民） =====================
# 医疗安全不能依赖"模型自己写的输出里有没有出现某个关键词"来判断。
# 反例：用户说"我胸痛得厉害"，模型在回答里没复述"胸痛"二字，
# 关键词扫描就不触发 -> 高危就医提示被漏掉（假阴性，这是最危险的一类）。
# 因此把 risk_level 提升为结构化字段：由「工具返回的 risk_level」+「用户输入」共同决定，
# 代码据此强制插入就医提示，而不是让模型自己决定要不要提醒。

RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "emergency": 3}

HIGH_RISK_SIGNALS = (
    "胸痛", "呼吸困难", "昏厥", "剧烈头痛", "心悸", "突然视力模糊",
    "意识模糊", "严重出血", "持续呕吐", "高热不退", "剧烈腹痛", "面部下垂",
)

_VISIT_HINTS = ("就医", "急诊", "医院", "120")


def normalize_risk_level(level: Any) -> str:
    """把任意来源的风险等级归一到 low/medium/high/emergency；无法识别按 low 处理。"""
    if not isinstance(level, str):
        return "low"
    v = level.strip().lower()
    return v if v in RISK_ORDER else "low"


def max_risk_level(*levels: Any) -> str:
    """取多个风险等级中最高的一级（保守取最大值）。"""
    best = "low"
    for lv in levels:
        cand = normalize_risk_level(lv)
        if RISK_ORDER[cand] > RISK_ORDER[best]:
            best = cand
    return best


def is_high_risk(level: Any) -> bool:
    """是否达到需要强制就医提示的等级（high / emergency）。"""
    return RISK_ORDER[normalize_risk_level(level)] >= RISK_ORDER["high"]


def detect_high_risk_signals(text: Any) -> bool:
    """文本中是否出现高危信号。用于扫描「用户输入」，而不是扫描模型输出。"""
    if not isinstance(text, str) or not text:
        return False
    return any(sig in text for sig in HIGH_RISK_SIGNALS)


def needs_emergency_guidance(answer: Any) -> bool:
    """回答中是否缺少就医/急诊引导（缺失则需要补）。"""
    if not isinstance(answer, str) or not answer.strip():
        return True
    return not any(h in answer for h in _VISIT_HINTS)


def is_constraint_enforce_enabled() -> bool:
    """CONSTRAINT_ENFORCE 环境变量优先，其次 config.CONSTRAINT_ENFORCE，默认 True（硬拦）。

    默认开启的理由：约束若默认只警告不拦截，等于没有约束——模型仍可调用白名单外的 Skill，
    "约束系统"沦为日志装饰。要评估影响面时可显式设 CONSTRAINT_ENFORCE=0 退回 warn 模式。
    """
    env = os.getenv("CONSTRAINT_ENFORCE", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    try:
        from config import CONSTRAINT_ENFORCE  # type: ignore
        return bool(CONSTRAINT_ENFORCE)
    except Exception:
        return True


class ConstraintValidator:
    """约束验证器"""

    def __init__(
        self,
        agent_constraints_file: str = "constraints/agent_constraints.yaml",
        swarm_constraints_file: str = "constraints/swarm_constraints.yaml"
    ):
        """
        初始化约束验证器

        Args:
            agent_constraints_file: Agent约束定义文件
            swarm_constraints_file: Swarm约束定义文件
        """
        # 加载 Agent 约束（复用带缓存的单一来源）
        self.agent_constraints = load_agent_constraints()

        # 加载 Swarm 约束
        swarm_path = Path(__file__).parent / "swarm_constraints.yaml"
        with open(swarm_path, 'r', encoding='utf-8') as f:
            self.swarm_constraints = yaml.safe_load(f)

        logger.info("✅ ConstraintValidator initialized")

    def get_allowed_tools(self, agent_id: str) -> List[str]:
        # 委托给模块级函数，避免两处各读一次 yaml 造成语义漂移
        return get_allowed_tools(agent_id)

    def validate_tool_call(self, agent_id: str, tool_name: str) -> Dict[str, Any]:
        """
        验证工具调用是否允许

        Args:
            agent_id: Agent ID
            tool_name: 工具名称

        Returns:
            {
                "valid": bool,
                "reason": str (如果不允许),
                "allowed_tools": List[str],
                "severity": "warning" | "block"
            }
        """
        allowed_tools = self.get_allowed_tools(agent_id)

        # 如果 allowed_tools 为空，表示没有限制
        if not allowed_tools:
            return {"valid": True, "allowed_tools": []}

        # 检查工具是否在允许列表中
        if tool_name not in allowed_tools:
            enforce = is_constraint_enforce_enabled()
            severity = "block" if enforce else "warning"
            reason = (
                f"该 Skill「{tool_name}」不被当前 Agent「{agent_id}」允许，"
                f"请改用：{', '.join(allowed_tools)}"
            )
            return {
                "valid": False,
                "reason": reason,
                "allowed_tools": allowed_tools,
                "severity": severity,
            }

        return {"valid": True, "allowed_tools": allowed_tools}

    def validate_output(
        self,
        agent_id: str,
        output: str,
        risk_level: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        验证输出是否符合约束

        Args:
            agent_id: Agent ID
            output: Agent 的输出文本
            risk_level: 结构化风险等级（low/medium/high/emergency），
                来自 assess_risk 工具返回 + 用户输入的高危信号。
                这是判断"是否必须提示就医"的**首要依据**，
                文本关键词匹配只作为兜底（详见本模块顶部的说明）。

        Returns:
            {
                "valid": bool,
                "violations": List[str],
                "auto_fixable": List[str]  # 可以自动修复的违规
            }
        """
        agent_constraints = self.agent_constraints['agents'].get(agent_id, {})
        output_constraints = agent_constraints.get('output_constraints', [])
        common_constraints = self.agent_constraints.get('common', {}).get('output_constraints', [])
        # 结构化风险等级；缺失时退化为 None，由下面的关键词兜底
        effective_risk = normalize_risk_level(risk_level)

        # 合并约束
        all_constraints = output_constraints + common_constraints

        violations = []
        auto_fixable = []

        # 检查免责声明
        if 'must_include_disclaimer' in all_constraints:
            if '免责声明' not in output and 'disclaimer' not in output.lower() and '仅供参考' not in output:
                violations.append("缺少免责声明")
                auto_fixable.append("add_disclaimer")

        # 检查长度限制
        max_length_constraint = next(
            (c for c in all_constraints if isinstance(c, dict) and 'max_response_length' in c),
            None
        )
        if max_length_constraint:
            max_length = max_length_constraint.get('max_response_length')
            if len(output) > max_length:
                violations.append(f"回答过长（{len(output)} > {max_length}字）")

        # 检查高危情况必须建议就医
        # 首要依据是结构化 risk_level（来自 assess_risk 工具 + 用户输入的高危信号），
        # 它不看模型有没有在回答里复述症状，因此不会出现
        # "用户说胸痛、模型恰好没写胸痛"导致的漏判（假阴性）。
        # 文本关键词匹配仅作为兜底。
        # 这条对所有 Agent 生效：common.safety_rules 里有 never_delay_emergency_care，
        # 属于通用安全底线，不应受单个 Agent 的 output_constraints 是否声明而影响。
        structural_high_risk = is_high_risk(effective_risk)
        keyword_high_risk = any(kw in output for kw in HIGH_RISK_SIGNALS)
        if (structural_high_risk or keyword_high_risk) and needs_emergency_guidance(output):
            violations.append("高危情况未建议就医")
            auto_fixable.append("add_emergency_warning")

        # 检查是否引用来源（仅 ResearchAgent）
        if 'must_cite_sources' in all_constraints:
            if "指南" not in output and "文献" not in output and "研究" not in output:
                violations.append("未引用来源或证据")

        # 检查禁止行为
        forbidden_actions = agent_constraints.get('forbidden_actions', [])
        if 'diagnose_disease' in forbidden_actions:
            if any(phrase in output for phrase in ["您患有", "确诊为", "肯定是", "就是"]):
                violations.append("包含明确诊断（越界行为）")

        if 'prescribe_medication' in forbidden_actions:
            # 更精细的药物处方检测（避免误报）
            # 只检测明确的药物处方模式
            import re

            # 模式1: 具体药物剂量（如：硝苯地平20mg）
            if re.search(r'(药物|药品|药).{0,10}(\d+\s*(mg|g|毫克|克))', output):
                violations.append("包含具体药物处方（越界行为）")

            # 模式2: 用药频率和剂量（如：每日3次，每次10mg）
            elif re.search(r'每(日|天|次).{0,5}\d+\s*次.{0,10}(\d+\s*(mg|g|毫克|克))', output):
                violations.append("包含具体药物处方（越界行为）")

            # 模式3: 明确的处方建议（如：建议服用XX 20mg）
            elif re.search(r'(建议|推荐)(服用|使用).{0,15}\d+\s*(mg|g|毫克|克)', output):
                violations.append("包含具体药物处方（越界行为）")

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "auto_fixable": auto_fixable
        }

    def get_required_agents(self, question: str) -> List[str]:
        """根据约束规则推荐必须包含的 Agent（委托模块级函数，共用带缓存的 yaml 加载）。"""
        return get_required_agents(question)

