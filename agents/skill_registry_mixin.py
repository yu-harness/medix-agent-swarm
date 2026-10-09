"""
统一的 Skills 注册（自动发现）
所有 Worker Agents 共享
"""
from typing import List, Optional

from core.skill_loader import discover_skills
from core.skill_registry import SkillParameter
from pathlib import Path
from loguru import logger
import inspect


class SkillRegistryMixin:
    """
    统一注册 Skills（自动发现 + 白名单裁剪）

    所有 Worker Agents (ConsultationAgent, DiagnosticAgent, ResearchAgent)
    都继承这个 Mixin，避免重复代码

    设计要点：
    - 发现：从 .claude/skills/ 自动扫描，无需手写清单
    - 裁剪：**注册阶段**就按 YAML 白名单剔除不允许的 Skill，
      而不是"全量注册 + 运行时警告"——后者在 warn 模式下等于没有约束，
      而且会让 LLM 在工具列表里看到它本不该调用的工具。
    """

    def register_all_skills(self, allowed: Optional[List[str]] = None):
        """
        自动扫描并注册本 Agent 允许的 Skills

        Args:
            allowed: 白名单（Skill 函数名列表）。为 None 时自动从
                constraints/agent_constraints.yaml 读取该 Agent 的 allowed_tools。
                白名单为空列表表示不限制（未配置该 Agent 时保持旧行为）。
        """
        project_root = Path(__file__).parent.parent
        discovered = discover_skills(project_root)
        agent_id = getattr(self, "agent_id", "?")

        if allowed is None:
            allowed = self.get_allowed_tools()
        allowed_set = set(allowed or [])

        registered = 0
        for skill_info in discovered:
            function_name = skill_info["function_name"]
            metadata = skill_info["metadata"]
            func = skill_info["function"]

            # 白名单裁剪：不允许的直接不注册，而不是注册后再靠运行时约束
            if allowed_set and function_name not in allowed_set:
                logger.debug(
                    f"⏭️  Skipped skill (not in allowlist for {agent_id}): {function_name}"
                )
                continue

            # 从 metadata 获取描述
            description = metadata.get("description", f"Skill: {skill_info['name']}")

            # 根据函数签名自动推断参数
            parameters = self._infer_skill_parameters(skill_info)

            # 注册到 SkillRegistry
            self.skill_registry.register(
                name=function_name,
                function=func,
                description=description,
                parameters=parameters
            )
            registered += 1
            logger.info(f"✅ Registered skill: {function_name}")

        logger.info(
            f"Total {registered}/{len(discovered)} skills registered for {agent_id} "
            f"(allowlist={'ALL' if not allowed_set else len(allowed_set)})"
        )

    def get_allowed_tools(self) -> List[str]:
        """读取本 Agent 的 Skill 白名单（单一来源：constraints/agent_constraints.yaml）。"""
        try:
            from constraints.validator import get_allowed_tools
            return get_allowed_tools(getattr(self, "agent_id", ""))
        except Exception as e:
            logger.warning(f"Failed to load allowlist for {getattr(self, 'agent_id', '?')}: {e}")
            return []

    def render_available_skills(self) -> str:
        """把「已注册」的 Skill 渲染成提示词片段。

        单一来源：直接读注册表，而不是在 system prompt 里手抄一份清单。
        手抄会与 YAML 白名单形成两份各自维护的真相，改一边忘另一边是必然的。
        """
        skills = self.skill_registry.get_all()
        if not skills:
            return "（无可用 Skill）"
        lines = []
        for i, (name, meta) in enumerate(skills.items(), 1):
            desc = (meta.get("description") or "").strip().split("\n")[0][:80]
            lines.append(f"{i}. {name}" + (f"：{desc}" if desc else ""))
        return "\n".join(lines)

    def has_skill(self, name: str) -> bool:
        """该 Skill 是否在当前白名单（=已注册）内。"""
        return name in self.skill_registry.get_all()

    def render_skill_refs(self, *names: str, sep: str = " 或 ") -> str:
        """把提示词里点名的 Skill 按当前白名单过滤后拼接（名单外的自动消失）。

        策略句里点名 Skill 时用它，避免"YAML 白名单改了、提示词还在要求调用
        已不存在的 Skill"这种两份真相。
        """
        available = self.skill_registry.get_all()
        return sep.join(n for n in names if n in available)

    def _infer_skill_parameters(self, skill_info: dict) -> list:
        """
        从 skill 信息推断参数

        Args:
            skill_info: skill 信息字典

        Returns:
            [SkillParameter(...), ...]
        """
        func = skill_info["function"]

        # 获取函数签名
        sig = inspect.signature(func)
        parameters = []

        for param_name, param in sig.parameters.items():
            # 跳过 self 和特殊参数
            if param_name in ["self", "args", "kwargs"]:
                continue

            # 判断是否必需
            required = param.default == inspect.Parameter.empty

            # 推断类型（简单规则）
            param_type = "string"
            if "count" in param_name or "limit" in param_name or "max" in param_name or "iterations" in param_name:
                param_type = "number"

            # 生成描述
            param_desc = param_name.replace('_', ' ').title()

            parameters.append(SkillParameter(
                param_name,
                param_type,
                param_desc,
                required
            ))

        return parameters
