"""
运行摘要（RunningSummary）：短期记忆 · L2 触发式语义摘要层

对应设计（L1 原文 / L2 运行摘要 / L3 关键事实 三层）：
- L1 原文：最近 N 轮全量（token 预算倒推）——保留细节
- L2 运行摘要：预算超限时，把"超出预算的最老消息"压缩为语义摘要——保语义不保细节
- L3 关键事实：患者档案（独立注入通道，天然跳过压缩）

关键设计点：
1. 触发式：只有历史 token 超预算才生成，不预先生成（省钱）
2. 白名单保护：主诉锚点（user_turns）、患者档案（facts）走独立注入通道，
   不经过本层压缩——"关键事件跳过压缩"由架构保证
3. 增量合并：生成时以旧摘要为底稿 + 新增超限消息，避免旧信息丢失
4. 降级：LLM 失败 → 沿用纯截断（摘要为空），不阻塞主路径
"""
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .short_term import estimate_tokens

# 摘要生成结果的最大长度（字符）
SUMMARY_MAX_CHARS = 240

_SUMMARY_SYSTEM_PROMPT = """你是医疗对话背景摘要器。把更早轮次的对话压缩成背景摘要，
供后续轮次引用，避免上下文超限时丢失早期信息。

要求：
1. 只保留：用户身份信息、主诉与关键症状、已确认的医疗事实、已给出的结论与建议
2. 去掉：客套话、工具调用细节、重复表述、与医疗无关的内容
3. 若提供了"已有摘要"，在其基础上合并新增对话，输出完整摘要（不是只输出新增部分）
4. 只输出摘要正文，不要任何前缀或解释"""


def split_history_by_budget(
    history: List[Dict[str, str]],
    budget_tokens: int,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """把历史按 token 预算拆成 (超出部分, 保留部分)。

    - 从尾部（最新）累计 token，单条消息不可再拆分
    - 保留部分 = 最新、能塞进预算的消息；超出部分 = 更早、将被压缩为摘要的消息
    """
    kept: List[Dict[str, str]] = []
    total = 0
    for m in reversed(history):
        cost = estimate_tokens(str(m.get("content") or ""))
        if kept and total + cost > budget_tokens:
            break
        kept.append(m)
        total += cost
    kept.reverse()
    excess = history[: len(history) - len(kept)]
    return excess, kept


class RunningSummaryManager:
    """运行摘要的存取与节流（会话级；内存后端，可扩展 Redis）。"""

    def __init__(self):
        # session_id -> {"text": str, "updated_at": str, "user_msg_count": int}
        self._summaries: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> str:
        if not session_id:
            return ""
        with self._lock:
            entry = self._summaries.get(session_id)
            return (entry or {}).get("text") or ""

    def set(self, session_id: str, text: str, user_msg_count: int) -> None:
        if not session_id or not text:
            return
        with self._lock:
            self._summaries[session_id] = {
                "text": text.strip(),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "user_msg_count": user_msg_count,
            }

    def should_regenerate(self, session_id: str, user_msg_count: int) -> bool:
        """节流：只有会话出现新的 user 消息才重新生成，避免每轮重复调 LLM。"""
        if not session_id:
            return False
        with self._lock:
            entry = self._summaries.get(session_id)
            if not entry:
                return True
            return user_msg_count > entry.get("user_msg_count", 0)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._summaries.pop(session_id, None)


async def generate_summary(
    llm_client: Any,
    old_summary: str,
    excess_messages: List[Dict[str, str]],
    max_chars: int = SUMMARY_MAX_CHARS,
) -> str:
    """基于旧摘要 + 新增超限消息，生成（合并后的）运行摘要。

    LLM 不可用/失败 → 返回旧摘要（若存在），保证信息不丢、不阻塞。
    """
    if not llm_client or not excess_messages:
        return old_summary or ""

    # 格式化超限消息为对话文本（去掉 system 注入痕迹，只留用户/助手）
    lines = []
    for m in excess_messages:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        label = "用户" if role == "user" else "助手"
        lines.append(f"{label}：{content}")
    if not lines:
        return old_summary or ""

    try:
        import asyncio

        user_prompt = ""
        if old_summary:
            user_prompt += f"已有摘要：\n{old_summary}\n\n"
        user_prompt += "新增对话（更早轮次）：\n" + "\n".join(lines)
        user_prompt += (
            f"\n\n请输出合并后的完整背景摘要（不超过 {max_chars} 字）。"
        )

        result = await llm_client.chat(
            [
                {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        text = (result or "").strip()
        if not text:
            return old_summary or ""
        return text[: max_chars * 2]  # 上限放宽一倍，防止 LLM 超长失控
    except Exception:
        return old_summary or ""
