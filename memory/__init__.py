"""
记忆系统：Agent 的持久化学习和记忆管理

包含：
- ShortTermMemory：会话级对话历史（内存/Redis）
- LongTermMemory：跨会话记忆（Mem0）
- MemoryEntropyManager：熵管理器（去重和压缩）
"""

# 短期和长期记忆
from .short_term import (
    ShortTermMemory,
    ConversationHistory,
    fit_history_to_budget,
    estimate_tokens,
)
from .long_term import (
    LongTermMemory
)

# 运行摘要（短期记忆 L2：触发式语义摘要）
from .running_summary import (
    RunningSummaryManager,
    split_history_by_budget,
    generate_summary,
)

# Harness Engineering: 熵管理
from .entropy_manager import (
    MemoryEntropyManager
)

from .session_summary import (
    SessionSummary,
    SessionSummaryManager,
    AgentParticipation,
    KeyFinding,
    Lesson,
    PerformanceMetrics
)

__all__ = [
    # 短期和长期记忆
    'ShortTermMemory',
    'ConversationHistory',
    'LongTermMemory',
    # token 预算（L1）
    'fit_history_to_budget',
    'estimate_tokens',
    # 运行摘要（L2）
    'RunningSummaryManager',
    'split_history_by_budget',
    'generate_summary',
    # Harness Engineering: 熵管理
    'MemoryEntropyManager',
    # 会话总结（本地 Markdown 持久化）
    'SessionSummary',
    'SessionSummaryManager',
    'AgentParticipation',
    'KeyFinding',
    'Lesson',
    'PerformanceMetrics',
]
