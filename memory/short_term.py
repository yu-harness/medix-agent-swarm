"""
短期记忆：会话级对话历史管理

功能：
- 管理会话级的对话历史（messages）
- 支持两种存储后端：内存（默认）和 Redis（可选）
- 自动过期机制（Redis 1小时）
- 熵管理：自动去重和压缩（Harness Engineering）
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import os
import threading
from loguru import logger

# Harness Engineering: 熵管理
try:
    from .entropy_manager import MemoryEntropyManager
    ENTROPY_ENABLED = True
except ImportError:
    logger.warning("EntropyManager not found, running without entropy management")
    ENTROPY_ENABLED = False


def estimate_tokens(text: str) -> int:
    """粗估文本 token 数（无需引入分词依赖）。

    中文/日文约 1 汉字 ≈ 1 token，英文约 4 字符 ≈ 1 token。
    折中按 1.5 字符/token 估算；宁可高估（多留预算）不可低估。
    """
    if not text:
        return 0
    return max(1, int(len(text) / 1.5))


def fit_history_to_budget(
    messages: List[Dict[str, Any]],
    max_tokens: int = 8000
) -> List[Dict[str, Any]]:
    """从消息尾部往前累加 token，返回预算内的消息（保最新、丢最旧）。

    记忆的物理约束是 token（上下文窗口），不是消息条数：
    50 条中文长消息可能占 3 万 token，而 5 条短消息可能只有 2000。
    用 token 预算倒推，让"能装下多少轮"由预算决定，而不是拍脑袋定条数。
    """
    budget = max(200, int(max_tokens or 8000))
    total = 0
    keep: List[Dict[str, Any]] = []
    for msg in reversed(messages):
        total += estimate_tokens(str(msg.get("content") or ""))
        if total > budget:
            break
        keep.append(msg)
    keep.reverse()
    return keep


@dataclass
class ConversationHistory:
    """对话历史数据类"""
    session_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str):
        """添加消息"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self.last_updated = datetime.now()

    def get_recent_messages(self, limit: int = 50) -> List[Dict[str, str]]:
        """获取最近的消息"""
        return self.messages[-limit:]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于 Redis 存储）"""
        return {
            "session_id": self.session_id,
            "messages": self.messages,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationHistory":
        """从字典创建（从 Redis 加载）"""
        return cls(
            session_id=data["session_id"],
            messages=data["messages"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            metadata=data.get("metadata", {})
        )


class ShortTermMemory:
    """
    短期记忆管理器（单例模式）

    支持两种存储后端：
    1. memory：纯内存存储（默认，快速但不持久）
    2. redis：Redis 存储（可选，持久但需要 Redis 服务）

    使用场景：
    - 管理单次会话的对话历史
    - Agent Loop 中的消息记录
    - 会话结束后转换为长期记忆
    """

    _instance = None  # 单例实例
    # 单例创建与初始化必须加锁：Skill 现由线程池并发执行，
    # 并发首次调用若不保护会重复初始化，把已建立的 sessions 清空
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """单例模式（线程安全，双重检查锁定）"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        storage_type: str = "memory",
        redis_config: Optional[Dict[str, Any]] = None,
        session_ttl_seconds: Optional[int] = None,
        max_sessions: Optional[int] = None
    ):
        """
        初始化短期记忆管理器

        Args:
            storage_type: 存储类型，"memory" 或 "redis"
            redis_config: Redis 配置（storage_type="redis" 时需要）
            session_ttl_seconds: 内存后端会话空闲过期时间（秒），0 表示不过期
            max_sessions: 内存后端最多保留的会话数，超出按最久未用淘汰
        """
        # 防止重复初始化：锁覆盖整个初始化过程，
        # 保证并发首次调用只有一个线程真正执行 _initialize()
        with self._instance_lock:
            if getattr(self, '_initialized', False):
                return
            self._initialize(
                storage_type, redis_config, session_ttl_seconds, max_sessions
            )
            self._initialized = True

    @staticmethod
    def _env_int(name: str, default: int, override: Optional[int]) -> int:
        """读取整数配置：显式入参 > 环境变量 > 默认值。"""
        if override is not None:
            return int(override)
        try:
            return int(os.getenv(name, str(default)))
        except Exception:
            return default

    def _initialize(
        self,
        storage_type: str,
        redis_config: Optional[Dict[str, Any]],
        session_ttl_seconds: Optional[int],
        max_sessions: Optional[int],
    ):
        """真正的初始化逻辑（由 __init__ 在锁内调用，请勿直接调用）"""
        self.storage_type = storage_type
        self.sessions: Dict[str, ConversationHistory] = {}
        self.redis_client = None

        # 内存后端的容量与过期策略。
        # 这两个上限是必需的：sessions 原本只增不删，生产跑一天就会 OOM。
        # Redis 后端不需要手动淘汰——由 _save_to_redis 的 setex TTL 负责。
        self.session_ttl_seconds = self._env_int(
            "MEDIX_SESSION_TTL", 3600, session_ttl_seconds
        )
        self.max_sessions = self._env_int("MEDIX_MAX_SESSIONS", 1000, max_sessions)
        # Skill 现在由线程池并发执行（search_history 会在工作线程读本对象），
        # 而事件循环线程在写，因此读写 sessions 必须加锁。
        self._lock = threading.RLock()

        # Harness Engineering: 熵管理器
        self.entropy_manager = MemoryEntropyManager() if ENTROPY_ENABLED else None
        if ENTROPY_ENABLED:
            logger.debug("✅ Entropy management enabled for short-term memory")

        if storage_type == "redis":
            try:
                import redis
                config = redis_config or {}
                self.redis_client = redis.Redis(
                    host=config.get("host", "localhost"),
                    port=config.get("port", 6379),
                    db=config.get("db", 0),
                    password=config.get("password"),
                    decode_responses=True
                )
                # 测试连接
                self.redis_client.ping()
                logger.info("ShortTermMemory initialized with Redis")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}. Falling back to memory storage.")
                self.storage_type = "memory"
                self.redis_client = None
        else:
            logger.info("ShortTermMemory initialized with in-memory storage")

    # ===== 内存后端的过期与容量控制 =====
    # sessions 原本只增不删，会话数会随请求量单调增长，最终 OOM。
    # 这里补两层保护：空闲过期（TTL）+ 容量上限（按最久未用淘汰）。

    def _is_expired(self, history: ConversationHistory, now: datetime) -> bool:
        """会话是否已空闲超过 TTL。ttl<=0 表示不过期。"""
        if self.session_ttl_seconds <= 0:
            return False
        return (now - history.last_updated).total_seconds() > self.session_ttl_seconds

    def _evict_if_needed(self, reserve: int = 0) -> None:
        """淘汰过期会话；仍超上限时按最久未用（LRU）淘汰。仅内存后端需要。

        Args:
            reserve: 预留的位置数。create_session 在写入前调用，需要预留 1 个，
                否则淘汰发生在插入之前，稳态会变成 max_sessions + 1。
        """
        if self.storage_type != "memory" or self.max_sessions <= 0:
            return

        now = datetime.now()
        expired = [
            sid for sid, h in self.sessions.items() if self._is_expired(h, now)
        ]
        for sid in expired:
            self.sessions.pop(sid, None)
        if expired:
            logger.debug(
                f"Evicted {len(expired)} expired sessions (ttl={self.session_ttl_seconds}s)"
            )

        overflow = len(self.sessions) + reserve - self.max_sessions
        if overflow > 0:
            oldest = sorted(
                self.sessions.items(), key=lambda kv: kv[1].last_updated
            )[:overflow]
            for sid, _ in oldest:
                self.sessions.pop(sid, None)
            logger.warning(
                f"Session limit reached ({self.max_sessions}); "
                f"evicted {len(oldest)} least-recently-used sessions"
            )

    def create_session(
        self,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationHistory:
        """
        创建新会话

        Args:
            session_id: 会话ID
            metadata: 会话元数据

        Returns:
            ConversationHistory 对象
        """
        history = ConversationHistory(
            session_id=session_id,
            metadata=metadata or {}
        )

        if self.storage_type == "memory":
            with self._lock:
                # reserve=1：给即将写入的会话预留位置
                self._evict_if_needed(reserve=1)
                self.sessions[session_id] = history
        elif self.storage_type == "redis" and self.redis_client:
            self._save_to_redis(history)

        logger.debug(f"Created session: {session_id}")
        return history

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str
    ):
        """
        添加消息到会话历史

        Args:
            session_id: 会话ID
            role: 消息角色（user/assistant/tool）
            content: 消息内容
        """
        history = self.get_session(session_id)

        if history is None:
            history = self.create_session(session_id)

        history.add_message(role, content)

        # 保存到存储
        if self.storage_type == "redis" and self.redis_client:
            self._save_to_redis(history)

        logger.debug(f"Added {role} message to session {session_id}")

    def get_session(self, session_id: str) -> Optional[ConversationHistory]:
        """
        获取会话历史

        Args:
            session_id: 会话ID

        Returns:
            ConversationHistory 对象，如果不存在或已过期返回 None
        """
        if self.storage_type == "memory":
            with self._lock:
                history = self.sessions.get(session_id)
                if history is None:
                    return None
                # 惰性过期：读到才发现过期就顺手清掉，避免过期会话占着内存
                if self._is_expired(history, datetime.now()):
                    self.sessions.pop(session_id, None)
                    logger.debug(f"Session expired and removed: {session_id}")
                    return None
                return history
        elif self.storage_type == "redis" and self.redis_client:
            return self._load_from_redis(session_id)
        return None

    def get_recent_messages(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[Dict[str, str]]:
        """
        获取最近的消息（自动熵管理）

        Args:
            session_id: 会话ID
            limit: 最大消息数

        Returns:
            消息列表（去重和压缩后）
        """
        history = self.get_session(session_id)
        if history:
            messages = history.get_recent_messages(limit)

            # Harness Engineering: 统一熵管理
            if self.entropy_manager and len(messages) > 0:
                # 估算熵（用于监控系统健康）
                if len(messages) >= 10:
                    entropy_info = self.entropy_manager.estimate_entropy(messages)
                    if entropy_info["entropy_level"] == "high":
                        logger.warning(
                            f"📊 会话 {session_id} 熵等级: {entropy_info['entropy_level']} "
                            f"(消息数: {entropy_info['total_messages']}, "
                            f"重复率: {entropy_info['duplicate_rate']:.1%})"
                        )

                # 统一使用 auto_clean: 自动去重+压缩
                messages = self.entropy_manager.auto_clean(
                    messages,
                    enable_deduplication=True,
                    enable_compression=True,
                    max_messages=limit
                )

            return messages
        return []

    def get_history(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, str]]:
        """
        获取历史对话（OpenAI 格式，用于 Agent Loop）

        Args:
            session_id: 会话ID
            limit: 最大轮数（一轮 = user + assistant）

        Returns:
            消息列表（OpenAI 格式: [{"role": "user", "content": "..."}, ...]）
        """
        # get_recent_messages 已处理熵管理，这里只做格式转换
        messages = self.get_recent_messages(session_id, limit * 2)  # 每轮2条消息

        # 转换为 OpenAI 格式（只保留 user 和 assistant 消息）
        openai_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
            if msg["role"] in ["user", "assistant"]
        ]

        return openai_messages

    @staticmethod
    def _strip_user_content(content: str) -> str:
        text = (content or "").strip()
        if "用户问题：" in text:
            text = text.split("用户问题：")[-1].strip()
        if "用户本轮问题：" in text:
            text = text.split("用户本轮问题：")[-1].split("\n任务：")[0].strip()
        for marker in ("【本轮模式", "【本会话已知信息】", "【强制规则】", "背景信息：", "[系统信息]"):
            if marker in text:
                text = text.split(marker)[0].strip()
        return text.strip()

    def _anchor_from_complaints(self, complaints: List[str]) -> str:
        if not complaints:
            return ""
        blob = "；".join(complaints[-3:])
        tags: List[str] = []
        pop_rules = [
            (("孩子", "儿童", "宝宝", "小儿", "婴儿", "幼儿"), "儿童"),
            (("孕妇", "怀孕", "妊娠"), "孕妇"),
            (("老人", "老年", "高龄"), "老年人"),
        ]
        for keys, label in pop_rules:
            if any(k in blob for k in keys):
                tags.append(label)
                break

        symptom_keys = (
            "咳嗽", "发烧", "发热", "头痛", "胸闷", "腹泻", "呕吐", "皮疹",
            "高血压", "糖尿病", "哮喘", "鼻塞", "咽痛", "腹痛", "气喘",
        )
        for k in symptom_keys:
            if k in blob and k not in tags:
                tags.append(k)

        if tags:
            return "、".join(tags) + f"（用户表述：{complaints[-1]}）"
        return complaints[-1]

    def extract_session_anchor(
        self,
        session_id: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        limit: int = 50
    ) -> str:
        """从用户主诉轮次（优先）或短期记忆用户消息提取本会话锚点。"""
        turns = self.get_user_turns(session_id)
        if turns:
            return self._anchor_from_complaints(turns)

        msgs = messages if messages is not None else self.get_recent_messages(session_id, limit)
        complaints: List[str] = []
        for m in msgs:
            if m.get("role") != "user":
                continue
            text = self._strip_user_content(m.get("content") or "")
            if not text or len(text) > 300:
                continue
            if text.startswith(("评估", "提供", "检索", "回答用户", "承接：", "用户本轮问题")):
                continue
            if text not in complaints:
                complaints.append(text)

        return self._anchor_from_complaints(complaints)

    def record_user_question(self, session_id: str, question: str):
        """记录原始用户主诉（不受 Swarm Worker 消息污染）。"""
        q = (question or "").strip()
        if not q:
            return
        history = self.get_session(session_id)
        if history is None:
            history = self.create_session(session_id)
        turns = history.metadata.setdefault("user_turns", [])
        if not turns or turns[-1] != q:
            turns.append(q)
            history.last_updated = datetime.now()
            if self.storage_type == "redis" and self.redis_client:
                self._save_to_redis(history)

    def get_user_turns(self, session_id: str) -> List[str]:
        history = self.get_session(session_id)
        if not history:
            return []
        return list(history.metadata.get("user_turns") or [])

    def clear_session(self, session_id: str):
        """
        清空会话

        Args:
            session_id: 会话ID
        """
        if self.storage_type == "memory":
            with self._lock:
                self.sessions.pop(session_id, None)
        elif self.storage_type == "redis" and self.redis_client:
            key = f"session:{session_id}"
            self.redis_client.delete(key)

        logger.debug(f"Cleared session: {session_id}")

    def _save_to_redis(self, history: ConversationHistory):
        """保存到 Redis（内部方法）"""
        if not self.redis_client:
            return

        try:
            key = f"session:{history.session_id}"
            value = json.dumps(history.to_dict())
            # 设置过期时间：1小时（3600秒）
            self.redis_client.setex(key, 3600, value)
        except Exception as e:
            logger.error(f"Failed to save to Redis: {e}")

    def _load_from_redis(self, session_id: str) -> Optional[ConversationHistory]:
        """从 Redis 加载（内部方法）"""
        if not self.redis_client:
            return None

        try:
            key = f"session:{session_id}"
            value = self.redis_client.get(key)

            if value:
                data = json.loads(value)
                return ConversationHistory.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load from Redis: {e}")

        return None
