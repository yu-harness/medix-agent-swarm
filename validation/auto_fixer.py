"""
自动修复器
根据约束违规自动修复输出

基于 Harness Engineering 原则：
- 自动检测问题
- 自动修复（在可能的情况下）
- 保持 Agent 输出质量
"""
from typing import Dict, Any, List, Optional
from loguru import logger

# 风险等级判定复用约束模块的单一实现，避免两处各写一份关键词表
try:
    from constraints.validator import (
        HIGH_RISK_SIGNALS,
        RISK_ORDER,
        needs_emergency_guidance,
        normalize_risk_level,
    )
except Exception:  # 约束模块不可用时退化为本地兜底，保证自动修复仍可用
    RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "emergency": 3}
    HIGH_RISK_SIGNALS = (
        "胸痛", "呼吸困难", "昏厥", "剧烈头痛", "心悸", "突然视力模糊",
        "意识模糊", "严重出血", "持续呕吐", "高热不退", "剧烈腹痛", "面部下垂",
    )

    def normalize_risk_level(level):  # type: ignore
        if not isinstance(level, str):
            return "low"
        v = level.strip().lower()
        return v if v in RISK_ORDER else "low"

    def needs_emergency_guidance(answer):  # type: ignore
        if not isinstance(answer, str) or not answer.strip():
            return True
        return not any(h in answer for h in ("就医", "急诊", "医院", "120"))


class AutoFixer:
    """自动修复器"""

    def fix_output(
        self,
        output: str,
        auto_fixable: List[str],
        risk_level: Optional[str] = None
    ) -> str:
        """
        自动修复输出

        Args:
            output: 原始输出
            auto_fixable: 可修复的违规列表
            risk_level: 结构化风险等级（low/medium/high/emergency），用于高危提醒判定

        Returns:
            修复后的输出
        """
        fixed_output = output

        for fix_type in auto_fixable:
            if fix_type == "add_disclaimer":
                fixed_output = self.fix_missing_disclaimer(fixed_output)
            elif fix_type == "add_emergency_warning":
                fixed_output = self.fix_high_risk_warning(fixed_output, risk_level=risk_level)

        if fixed_output != output:
            logger.info("🔧 输出已自动修复")

        return fixed_output

    def fix_missing_disclaimer(self, output: str) -> str:
        """
        自动添加免责声明

        Args:
            output: 原始输出

        Returns:
            添加免责声明后的输出
        """
        if "免责" in output or "仅供参考" in output or "不能替代" in output:
            return output
        disclaimer = "\n\n【免责声明】\n以上信息仅供参考，不能替代专业医生的诊断和治疗。如有疑虑，请及时就医。"
        logger.debug("+ 自动添加免责声明")
        return output + disclaimer

    def fix_high_risk_warning(self, output: str, risk_level: Optional[str] = None) -> str:
        """
        自动添加高危情况警告

        Args:
            output: 原始输出
            risk_level: 结构化风险等级。达到 high/emergency 时**无条件**补警告，
                不再依赖"模型输出里是否恰好写了胸痛"这种关键词匹配——
                后者会因为模型没复述症状而漏掉真正的高危场景。

        Returns:
            添加警告后的输出
        """
        level = normalize_risk_level(risk_level)
        structural_high_risk = RISK_ORDER.get(level, 0) >= RISK_ORDER["high"]
        keyword_high_risk = any(kw in output for kw in HIGH_RISK_SIGNALS)

        if (structural_high_risk or keyword_high_risk) and needs_emergency_guidance(output):
            warning = "⚠️ **重要提醒**：您描述的情况可能提示严重问题，建议立即就医或拨打急救电话120，不要延误。\n\n"
            logger.debug(f"+ 自动添加高危情况警告 (risk_level={level})")
            return warning + output

        return output

    def fix_excessive_length(self, output: str, max_length: int) -> str:
        """
        截断过长的输出

        Args:
            output: 原始输出
            max_length: 最大长度

        Returns:
            截断后的输出
        """
        if len(output) > max_length:
            logger.warning(f"输出过长（{len(output)} > {max_length}），自动截断")
            truncated = output[:max_length - 50]  # 保留50字空间添加提示
            truncated += "\n\n[回答内容较长，已截断。如需完整信息，请咨询专业医生]"
            return truncated

        return output

    def remove_diagnosis_statements(self, output: str) -> str:
        """
        移除明确的诊断语句（高级功能，需要 LLM 辅助）

        Args:
            output: 原始输出

        Returns:
            移除诊断语句后的输出
        """
        # 简单替换（实际应该使用 LLM 进行更智能的重写）
        output = output.replace("您患有", "可能存在")
        output = output.replace("确诊为", "建议检查")
        output = output.replace("肯定是", "很可能是")

        return output
