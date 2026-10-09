"""
可观测性基座：Per-Request Token 账本 + Span 耗时瀑布流。

设计要点（为什么是「ContextVar 里放一个可变对象」）：

`asyncio.create_task` 会复制当前 context；线程池里我们统一走 `asyncio.to_thread`
（它内部就是 `copy_context().run(...)`，**不依赖** `run_in_executor` 的版本行为）。
在复制出来的 context 里给 ContextVar 重新赋值，**不会**回传到父 context。
但只要多个 context 持有**同一个对象**，对对象内部状态的写入是互相可见的。

所以这里 ContextVar 里放的是 TokenLedger / SpanRecorder 实例，而不是数字：
- 并发的 Worker 任务、线程池里的 Skill，都能把用量与耗时写进同一个账本；
- 请求结束时在主 context 上取快照即可，不需要任何回传机制。

账单结构（对外契约，见 `TokenLedger.snapshot()`）：

    {
      "total_tokens": 3120,
      "prompt_tokens": 1980,
      "completion_tokens": 1140,
      "estimated_cost_cny": 0.00426,
      "price": {"prompt_cny_per_1m": 1.0, "completion_cny_per_1m": 2.0},
      "breakdown": {
        "lead_decompose": {"tokens": ..., "prompt_tokens": ..., "completion_tokens": ...,
                           "cost_cny": ..., "calls": 1},
        ...
      }
    }
"""
from __future__ import annotations

import contextvars
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from loguru import logger

# --------------------------------------------------------------------------- #
# 计价模型（默认按 DeepSeek 官方定价：输入 ¥1 / 1M tokens，输出 ¥2 / 1M tokens）
# 可用环境变量覆盖，避免把价格写死在代码里：
#   MEDIX_PRICE_PROMPT_CNY_PER_M / MEDIX_PRICE_COMPLETION_CNY_PER_M
# --------------------------------------------------------------------------- #
def _price_per_token_per_m(env_key: str, default_per_m: float) -> float:
    raw = os.getenv(env_key)
    if raw:
        try:
            return float(raw) / 1_000_000.0
        except ValueError:
            logger.warning(f"环境变量 {env_key}={raw!r} 不是数字，回退默认 ¥{default_per_m}/1M")
    return default_per_m / 1_000_000.0


PRICE_PROMPT_CNY_PER_TOKEN = _price_per_token_per_m("MEDIX_PRICE_PROMPT_CNY_PER_M", 1.0)
PRICE_COMPLETION_CNY_PER_TOKEN = _price_per_token_per_m("MEDIX_PRICE_COMPLETION_CNY_PER_M", 2.0)

# 没打阶段标签的调用归到这里（例如脚本直连 LLM、或未来新增的调用点忘了传 stage）
DEFAULT_STAGE = "unlabeled"

# 瀑布流的层级命名：Coordinator 与 AgentLoop 共用同一组常量，避免两边写岔导致挂错父节点
ROOT_SPAN = "Request_Root"
ROUTE_SPAN = "Route_Decompose"
WORKER_POOL_SPAN = "Worker_Pool_Execution"
SYNTH_SPAN = "Lead_Synthesize"

# 成本保留到小数点后 5 位：单次请求量级在 0.001~0.01 元，再少就没意义了
COST_DECIMALS = 5


def _cost_cny(prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens * PRICE_PROMPT_CNY_PER_TOKEN
        + completion_tokens * PRICE_COMPLETION_CNY_PER_TOKEN
    )


class TokenLedger:
    """一次请求的 Token 账本（按阶段累加）。

    线程安全：Worker 在 asyncio 任务里跑，Skill 在线程池里跑，
    理论上都可能触发 LLM 调用，所以计数必须加锁。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: Dict[str, Dict[str, int]] = {}
        self._unpriced_calls = 0  # 没拿到 usage 的调用次数（流式未开 include_usage 时会出现）

    def add(self, stage: str, prompt_tokens: int, completion_tokens: int,
            total_tokens: Optional[int] = None) -> None:
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        if not prompt_tokens and not completion_tokens:
            with self._lock:
                self._unpriced_calls += 1
            return
        row = self._rows.get(stage)
        with self._lock:
            if row is None:
                row = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
                self._rows[stage] = row
            row["prompt_tokens"] += prompt_tokens
            row["completion_tokens"] += completion_tokens
            row["calls"] += 1

    def snapshot(self) -> Dict[str, Any]:
        """导出结构化账单（按阶段明细 + 合计 + 估算费用）。"""
        with self._lock:
            rows = {k: dict(v) for k, v in self._rows.items()}
            unpriced = self._unpriced_calls

        breakdown: Dict[str, Dict[str, Any]] = {}
        total_prompt = total_completion = 0
        for stage, row in sorted(rows.items(), key=lambda kv: -kv[1]["prompt_tokens"] - kv[1]["completion_tokens"]):
            pt, ct = row["prompt_tokens"], row["completion_tokens"]
            total_prompt += pt
            total_completion += ct
            breakdown[stage] = {
                "tokens": pt + ct,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "cost_cny": round(_cost_cny(pt, ct), COST_DECIMALS),
                "calls": row["calls"],
            }

        return {
            "total_tokens": total_prompt + total_completion,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "estimated_cost_cny": round(_cost_cny(total_prompt, total_completion), COST_DECIMALS),
            "price": {
                "prompt_cny_per_1m": round(PRICE_PROMPT_CNY_PER_TOKEN * 1_000_000, 4),
                "completion_cny_per_1m": round(PRICE_COMPLETION_CNY_PER_TOKEN * 1_000_000, 4),
            },
            "breakdown": breakdown,
            "unpriced_calls": unpriced,
        }


# --------------------------------------------------------------------------- #
# Span 记录器：把散落的耗时打点升级成层级结构
# --------------------------------------------------------------------------- #
class SpanRecorder:
    """记录层级化的耗时 Span。

    父节点用**显式名字**指定，而不是靠「当前栈」推断：
    Worker 是并发跑的，隐式栈会把并发任务的父子关系串错。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spans: List[Dict[str, Any]] = []
        self._t0 = time.perf_counter()

    @contextmanager
    def span(self, name: str, parent: Optional[str] = None, **meta: Any):
        """记录一个 Span；yield 一块可变 meta，调用方可在块内补充信息。"""
        record: Dict[str, Any] = {
            "name": name,
            "parent": parent,
            "start_ms": round((time.perf_counter() - self._t0) * 1000, 1),
            "duration_ms": 0.0,
            "meta": dict(meta),
        }
        started = time.perf_counter()
        try:
            yield record["meta"]
        finally:
            record["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
            with self._lock:
                self._spans.append(record)

    def mark(self, name: str, parent: Optional[str] = None,
             duration_ms: float = 0.0, **meta: Any) -> None:
        """直接登记一个已经测好的 Span（用于复用代码里已有的打点）。"""
        with self._lock:
            self._spans.append({
                "name": name,
                "parent": parent,
                "start_ms": round((time.perf_counter() - self._t0) * 1000, 1),
                "duration_ms": round(float(duration_ms), 1),
                "meta": dict(meta),
            })

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(s, meta=dict(s.get("meta") or {})) for s in self._spans]


# --------------------------------------------------------------------------- #
# 请求级上下文（ContextVar 里放可变对象，见模块 docstring）
# --------------------------------------------------------------------------- #
_ledger_var: contextvars.ContextVar[Optional[TokenLedger]] = contextvars.ContextVar(
    "medix_token_ledger", default=None
)
_spans_var: contextvars.ContextVar[Optional[SpanRecorder]] = contextvars.ContextVar(
    "medix_span_recorder", default=None
)


def begin_observability() -> tuple[TokenLedger, SpanRecorder]:
    """在请求入口调用：为本请求启用账本与 Span 记录。

    幂等：同一请求上下文里若已建过账本，就复用同一对对象并返回。
    这样入口层（如 API 的 SSE 端点）可以先建账本、再把同一个对象交给下游，
    从而在流式响应过程中**实时读到真实的阶段 Span**（前端进度展示要用），
    而不是等整条请求结束才能对账。
    """
    ledger = _ledger_var.get()
    spans = _spans_var.get()
    if ledger is not None and spans is not None:
        return ledger, spans

    ledger = TokenLedger()
    spans = SpanRecorder()
    _ledger_var.set(ledger)
    _spans_var.set(spans)
    return ledger, spans


def current_ledger() -> Optional[TokenLedger]:
    return _ledger_var.get()


def current_spans() -> Optional[SpanRecorder]:
    return _spans_var.get()


class _NoopRecorder(SpanRecorder):
    """不在请求上下文里时用的空记录器。

    有了它，调用点可以统一写 `spans = current_spans() or NOOP_SPANS`，
    不必每个打点处都判断 None，少一层分支就少一处将来会写错的地方。
    """

    @contextmanager
    def span(self, name: str, parent: Optional[str] = None, **meta: Any):
        yield {}

    def mark(self, name: str, parent: Optional[str] = None,
             duration_ms: float = 0.0, **meta: Any) -> None:
        return None

    def snapshot(self) -> List[Dict[str, Any]]:
        return []


NOOP_SPANS = _NoopRecorder()


# --------------------------------------------------------------------------- #
# ASCII 瀑布流格式化
# --------------------------------------------------------------------------- #
_BAR_WIDTH = 28


def format_span_waterfall(spans: List[Dict[str, Any]], title: str = "耗时瀑布流") -> str:
    """把 Span 列表渲染成缩进树 + 柱状条的 ASCII 瀑布流。

    Args:
        spans: SpanRecorder.snapshot() 的结果，每项含 name / parent / duration_ms /
               start_ms / meta
        title: 顶部标题

    Returns:
        可直接 print 的多行字符串
    """
    if not spans:
        return f"{title}\n（无 Span 记录）"

    root = next((s for s in spans if s.get("parent") in (None, "")), None)
    # 柱状条以根 Span 为 100% 基准；没有根节点时退化为最长 Span
    root_ms = (root.get("duration_ms") if root else max((s.get("duration_ms") or 0.0) for s in spans)) or 1.0

    lines = [title, "─" * 78]

    def bar(duration_ms: float) -> str:
        filled = max(1, round(duration_ms / root_ms * _BAR_WIDTH))
        return "█" * min(filled, _BAR_WIDTH)

    def walk(span: Dict[str, Any], prefix: str, is_last: bool, depth: int) -> None:
        duration = span.get("duration_ms") or 0.0
        pct = duration / root_ms * 100
        label = span["name"]
        meta = span.get("meta") or {}
        extra = ""
        if meta.get("detail"):
            extra = f"  · {meta['detail']}"
        connector = "" if depth == 0 else ("└─ " if is_last else "├─ ")
        lines.append(
            f"{prefix}{connector}{label:<34} {bar(duration):<{_BAR_WIDTH}} "
            f"{duration:>8.1f}ms  {pct:>5.1f}%{extra}"
        )
        children = [c for c in spans if c.get("parent") == span["name"]]
        children.sort(key=lambda c: (c.get("start_ms") or 0.0))
        child_prefix = prefix + ("" if depth == 0 else ("   " if is_last else "│  "))
        for i, child in enumerate(children):
            walk(child, child_prefix, i == len(children) - 1, depth + 1)

    roots = [s for s in spans if s.get("parent") in (None, "")]
    roots.sort(key=lambda s: (s.get("start_ms") or 0.0))
    for r in roots:
        walk(r, "", True, 0)

    lines.append("─" * 78)
    return "\n".join(lines)
