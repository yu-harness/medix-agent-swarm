"""
SessionSummary：会话总结和经验提取

每次 Swarm 协作后自动生成会话总结，记录：
- 问题和背景
- 参与的 Agent
- 协作过程
- 关键发现
- 经验教训
- 性能指标

这是群体智能"持续学习"的关键机制
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from loguru import logger


@dataclass
class AgentParticipation:
    """Agent 参与记录"""
    agent_id: str
    role: str  # lead/worker
    subtasks_handled: List[str]
    tool_calls: int
    execution_time: float  # 秒
    contribution_quality: float = 1.0  # 0-1


@dataclass
class KeyFinding:
    """关键发现"""
    category: str  # diagnosis/risk/evidence/treatment
    finding: str
    source_agent: str
    confidence: float = 1.0


@dataclass
class Lesson:
    """经验教训"""
    agent_id: str
    lesson_type: str  # success/failure/improvement
    description: str
    actionable: str  # 可执行的改进措施


@dataclass
class PerformanceMetrics:
    """性能指标

    原则：**只填能从过程数据里真实算出来的字段，算不出来的保持 None。**
    绝不用"看起来合理"的常数占位——假指标比没指标更危险，
    它会让人基于错误数字做判断（例如拿 0.8 的并行效率去论证 Swarm 有效）。
    """
    total_time: float  # 总耗时（秒）
    agent_count: int  # 参与 Agent 数量
    parallel_efficiency: float  # 平均每个 Agent 的时间利用率（0-1），由子任务时间戳计算
    # 以下三项暂无可信计算方式，保持 None 表示"未测量"
    information_coverage: Optional[float] = None  # 需要 ground truth 才能算
    redundancy: Optional[float] = None  # 需要重复度判定标准才能算
    speedup_vs_single: Optional[float] = None  # 需要单 Agent 基线对照才能算


@dataclass
class SessionSummary:
    """
    会话总结数据类

    记录一次完整的 Swarm 协作过程
    """
    session_id: str
    question: str
    context: Dict[str, Any]
    timestamp: datetime

    # 参与者
    agents_participated: List[AgentParticipation]

    # 过程
    subtasks_created: int
    subtasks_completed: int
    events_count: int

    # 结果
    final_answer: str
    key_findings: List[KeyFinding]

    # 学习
    lessons_learned: List[Lesson]

    # 性能
    performance: PerformanceMetrics

    # 元数据
    swarm_enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _fmt_pct(value: Optional[float]) -> str:
        """百分比格式化；未测量（None）时明确显示"未测量"，而不是显示 0%。"""
        if value is None:
            return "未测量"
        return f"{value:.1%}"

    @staticmethod
    def _fmt_ratio(value: Optional[float]) -> str:
        """倍数格式化；未测量（None）时明确显示"未测量"，而不是显示 1.00x（等于谎称无加速）。"""
        if value is None:
            return "未测量"
        return f"{value:.2f}x"

    def to_markdown(self) -> str:
        """转换为 Markdown 格式"""
        date_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"# Session Summary: {self.session_id}",
            "",
            f"**时间**: {date_str}",
            "",
            "## 问题",
            self.question,
            ""
        ]

        if self.context:
            lines.extend([
                "## 背景",
                "```json",
                json.dumps(self.context, ensure_ascii=False, indent=2),
                "```",
                ""
            ])

        lines.extend([
            "## 参与 Agent",
            ""
        ])

        for agent in self.agents_participated:
            lines.append(f"### {agent.agent_id} ({agent.role})")
            lines.append(f"- 处理子任务：{len(agent.subtasks_handled)} 个")
            lines.append(f"- 工具调用：{agent.tool_calls} 次")
            lines.append(f"- 执行时间：{agent.execution_time:.2f} 秒")
            lines.append("")

        lines.extend([
            "## 协作过程",
            "",
            f"- 创建子任务：{self.subtasks_created} 个",
            f"- 完成子任务：{self.subtasks_completed} 个",
            f"- 发布事件：{self.events_count} 个",
            ""
        ])

        if self.key_findings:
            lines.extend([
                "## 关键发现",
                ""
            ])

            for finding in self.key_findings:
                lines.append(f"### {finding.category.upper()}")
                lines.append(f"**来源**: {finding.source_agent}")
                lines.append(f"**发现**: {finding.finding}")
                lines.append(f"**置信度**: {finding.confidence:.1%}")
                lines.append("")

        lines.extend([
            "## 最终答案",
            "",
            self.final_answer[:500] + ("..." if len(self.final_answer) > 500 else ""),
            ""
        ])

        if self.lessons_learned:
            lines.extend([
                "## 经验教训",
                ""
            ])

            for lesson in self.lessons_learned:
                emoji = "✅" if lesson.lesson_type == "success" else "⚠️" if lesson.lesson_type == "failure" else "💡"
                lines.append(f"### {emoji} {lesson.agent_id}")
                lines.append(f"**{lesson.lesson_type.upper()}**: {lesson.description}")
                if lesson.actionable:
                    lines.append(f"**改进措施**: {lesson.actionable}")
                lines.append("")

        lines.extend([
            "## 性能指标",
            "",
            f"- 总耗时：{self.performance.total_time:.2f} 秒",
            f"- 参与 Agent：{self.performance.agent_count} 个",
            f"- 并行效率：{self._fmt_pct(self.performance.parallel_efficiency)}",
            f"- 信息覆盖度：{self._fmt_pct(self.performance.information_coverage)}",
            f"- 信息冗余度：{self._fmt_pct(self.performance.redundancy)}",
            f"- 加速比：{self._fmt_ratio(self.performance.speedup_vs_single)}",
            ""
        ])

        return "\n".join(lines)

    @classmethod
    def from_shared_context(
        cls,
        session_id: str,
        question: str,
        shared_context: Any,
        final_answer: str,
        start_time: datetime,
        end_time: datetime
    ) -> "SessionSummary":
        """从 SharedContext 构建 SessionSummary"""

        # 计算性能指标
        total_time = (end_time - start_time).total_seconds()

        # 每个 Agent 的真实忙碌时长：把它负责的子任务耗时相加。
        # 原来写的是 total_time / agent_count —— 那是把墙钟时间平摊，
        # 不是执行时间，而且会让"1 个 Agent 跑满"和"3 个 Agent 并行"看起来一样。
        agent_busy: Dict[str, float] = {}
        for subtask in getattr(shared_context, "task_decomposition", {}).values():
            started = getattr(subtask, "started_at", None)
            completed = getattr(subtask, "completed_at", None)
            if started and completed:
                delta = (completed - started).total_seconds()
                owner = getattr(subtask, "assigned_agent", None) or "unknown"
                agent_busy[owner] = agent_busy.get(owner, 0.0) + max(0.0, delta)

        # 提取 Agent 参与信息
        agents_participated = []
        for agent_id, contributions in shared_context.agent_contributions.items():
            tool_calls = sum(
                1 for c in contributions
                if c.result.get('success', True)
            )
            agents_participated.append(AgentParticipation(
                agent_id=agent_id,
                role="worker",
                subtasks_handled=[c.subtask_id for c in contributions],
                tool_calls=tool_calls,
                execution_time=round(agent_busy.get(agent_id, 0.0), 3)
            ))

        # 提取关键发现
        key_findings = []
        for contrib in shared_context.get_contributions():
            if "risk_level" in contrib.result:
                key_findings.append(KeyFinding(
                    category="risk",
                    finding=f"风险等级：{contrib.result['risk_level']}",
                    source_agent=contrib.agent_id,
                    confidence=contrib.confidence
                ))

        # 性能指标
        # 并行效率 = 各 Agent 忙碌时长之和 / (墙钟总耗时 × Agent 数)
        # 语义是"平均每个 Agent 的时间利用率"：
        # 全员全程并行 → 接近 1.0；N 个 Agent 完全串行 → 接近 1/N。
        # 分母为 0 时给 0.0（而不是编一个看起来合理的常数）。
        agent_count = len(shared_context.agent_contributions)
        total_busy = sum(agent_busy.values())
        if total_time > 0 and agent_count > 0:
            parallel_efficiency = max(0.0, min(1.0, total_busy / (total_time * agent_count)))
        else:
            parallel_efficiency = 0.0

        # information_coverage / redundancy / speedup_vs_single 需要 ground truth
        # 或单 Agent 基线对照才能算，当前没有，因此传 None 表示"未测量"
        performance = PerformanceMetrics(
            total_time=total_time,
            agent_count=agent_count,
            parallel_efficiency=parallel_efficiency,
        )

        return cls(
            session_id=session_id,
            question=question,
            context={},
            timestamp=start_time,
            agents_participated=agents_participated,
            subtasks_created=len(shared_context.task_decomposition),
            subtasks_completed=len(shared_context.get_all_completed_subtasks()),
            events_count=len(shared_context.events),
            final_answer=final_answer,
            key_findings=key_findings,
            # 经验教训需要单独的结构化抽取逻辑，当前未实现，因此为空列表（不是占位假数据）
            lessons_learned=[],
            performance=performance
        )


class SessionSummaryManager:
    """
    会话总结管理器

    负责保存和检索会话总结
    """

    def __init__(self, base_dir: str = "memory/swarm/session_summaries"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_summary_path(self, session_id: str) -> Path:
        """获取会话总结文件路径"""
        # 按日期组织
        date_str = session_id.split("-")[0] if "-" in session_id else "unknown"
        date_dir = self.base_dir / date_str
        date_dir.mkdir(parents=True, exist_ok=True)
        return date_dir / f"{session_id}.md"

    def save_summary(self, summary: SessionSummary):
        """保存会话总结"""
        summary_path = self._get_summary_path(summary.session_id)

        try:
            content = summary.to_markdown()
            summary_path.write_text(content, encoding="utf-8")
            logger.info(f"Saved session summary: {summary.session_id}")
        except Exception as e:
            logger.error(f"Error saving session summary: {e}")

    def search_similar_sessions(
        self,
        query: str,
        limit: int = 5
    ) -> List[Path]:
        """
        搜索相似的会话（简化实现）

        未来可以使用向量相似度搜索
        """
        # 简单实现：返回最近的会话
        all_summaries = list(self.base_dir.rglob("*.md"))
        all_summaries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return all_summaries[:limit]
