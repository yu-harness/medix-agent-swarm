"""
LLM客户端
支持调用 OpenAI 兼容的 API（如字节跳动豆包、OpenAI、Deepseek 等）
支持 function calling
"""
import sys
import time
import asyncio
import inspect
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
    BadRequestError,
    AuthenticationError,
)
from loguru import logger

# 使用项目根目录下的 config.py
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from config import LLM_CONFIG, ensure_api_keys

# 启动 fail-fast：缺少密钥立即给出指引，避免带空 key 静默运行
ensure_api_keys()


@dataclass
class ToolCall:
    """Function call 数据结构"""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """LLM 响应数据结构（支持 function calling）"""
    content: Optional[str]
    tool_calls: List[ToolCall]
    finish_reason: str  # "stop", "tool_calls", "length", "content_filter"

    def has_tool_calls(self) -> bool:
        """是否包含 function calls"""
        return len(self.tool_calls) > 0


@dataclass
class StreamStats:
    """流式调用的计时结果。

    TTFT（time to first token）才是用户感知到的「响应时间」：
    没有流式时用户要等整段生成完，TTFT 与总耗时相同；有流式后两者分离。
    """

    first_chunk_ms: Optional[float] = None   # 第一个 chunk 到达（含空 chunk）
    ttft_ms: Optional[float] = None          # 第一个正文增量到达
    total_ms: Optional[float] = None         # 流结束
    chunks: int = 0
    content_chars: int = 0


# ---- LLM 调用可靠性配置 ----
LLM_CALL_TIMEOUT = 30.0        # 单次调用超时（秒）：快速失败以便进入重试
LLM_MAX_RETRIES = 3            # 可重试异常的最大重试次数（连续失败达到即熔断）
LLM_RETRY_MAX_BACKOFF = 10.0   # 指数退避上限（秒）

# 仅这些异常值得重试：连接错误 / 超时 / 限流(429) / 服务端错误(5xx)
RETRYABLE_EXCEPTIONS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)
# 这些异常不可重试：请求本身有问题（参数/鉴权），重试无意义
NON_RETRYABLE_EXCEPTIONS = (
    BadRequestError,
    AuthenticationError,
)

# 医疗场景安全降级话术（连续重试失败后的兜底）
LLM_SAFE_FALLBACK = (
    "抱歉，我暂时无法处理您的请求。如遇紧急或严重症状，请立即就医或拨打急救电话；"
    "其他情况建议咨询专业医生。"
)


def llm_safe_fallback_response() -> "LLMResponse":
    """构造一个安全降级的 LLMResponse（无工具调用，finish_reason=error）。"""
    return LLMResponse(content=LLM_SAFE_FALLBACK, tool_calls=[], finish_reason="error")


async def _call_maybe_async(callback: Callable[[str], Any], value: str) -> Any:
    """增量回调既可能是普通函数也可能是协程函数，统一处理。"""
    result = callback(value)
    if inspect.isawaitable(result):
        return await result
    return result


class LLMClient:
    """统一的LLM客户端，支持多种模型"""

    def __init__(self, model_type: str = "openai_compatible"):
        """
        初始化LLM客户端

        Args:
            model_type: 模型类型，默认 "openai_compatible"（支持 OpenAI 兼容的 API）
        """
        self.model_type = model_type

        if model_type == "openai_compatible":
            # 使用 OpenAI 兼容的 API（通过 config.py 配置）
            self.config = LLM_CONFIG
            self.client = AsyncOpenAI(
                api_key=self.config["api_key"],
                base_url=self.config["base_url"],
                # 显式超时 + 关掉 SDK 内置重试，统一由本类的 _with_retry 接管（便于观测）
                timeout=LLM_CALL_TIMEOUT,
                max_retries=0,
            )
            self.model_name = self.config["model_name"]
            self.temperature = self.config.get("temperature", 0.7)
            self.max_tokens = self.config.get("max_tokens", 8192)
            self.retry_count = 0  # 累计重试次数（结构化日志用）
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    async def _with_retry(self, coro_factory, fallback=None) -> Any:
        """统一重试包装（内层：单 call 超时后的快速重试）。

        - 只对可重试异常（连接/超时/限流/5xx）重试，指数退避封顶 LLM_RETRY_MAX_BACKOFF；
        - 工具调用 JSON 残缺也视为可重试（模型偶发抽风）；
        - 不可重试异常（400/401 等）立即抛出，避免无意义重试；
        - 连续 LLM_MAX_RETRIES 次失败即熔断：若调用方传入 fallback 则返回之，否则抛出；
        - 每次重试、最终失败原因均结构化写入日志。
        """
        last_exc = None
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                return await coro_factory()
            except RETRYABLE_EXCEPTIONS as e:
                last_exc = e
                self.retry_count += 1
                logger.warning(
                    f"[LLM retry] attempt {attempt}/{LLM_MAX_RETRIES} failed: "
                    f"{type(e).__name__}: {e}"
                )
            except json.JSONDecodeError as e:
                last_exc = e
                self.retry_count += 1
                logger.warning(
                    f"[LLM retry] tool-args JSON decode failed "
                    f"attempt {attempt}/{LLM_MAX_RETRIES}: {e}"
                )
            except NON_RETRYABLE_EXCEPTIONS as e:
                logger.error(
                    f"[LLM] non-retryable error, aborting: {type(e).__name__}: {e}"
                )
                raise
            except Exception as e:
                logger.error(
                    f"[LLM] unexpected error (no retry): {type(e).__name__}: {e}"
                )
                raise

            # 退避（最后一次失败不再 sleep）
            if attempt < LLM_MAX_RETRIES:
                backoff = min(2 ** (attempt - 1), LLM_RETRY_MAX_BACKOFF)
                await asyncio.sleep(backoff)

        # 重试耗尽：结构化记录最终失败（便于日志采集 / 监控告警）
        logger.error(json.dumps({
            "event": "llm_call_exhausted",
            "model": self.model_name,
            "attempts": LLM_MAX_RETRIES,
            "final_error_type": type(last_exc).__name__,
            "final_error": str(last_exc),
            "total_retries": self.retry_count,
        }, ensure_ascii=False))

        # 熔断后降级：调用方传入 fallback 则返回之，否则抛出
        if fallback is not None:
            return fallback
        raise last_exc

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        fallback: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        异步聊天接口（带超时 + 分类重试兜底）

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            temperature: 温度参数（可选）
            max_tokens: 最大token数（可选）
            fallback: 连续重试耗尽后的安全降级文本；为 None 则抛出

        Returns:
            模型返回的文本
        """
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens

        logger.debug(f"Calling LLM ({self.model_type}) with {len(messages)} messages")

        async def _do() -> str:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content

        content = await self._with_retry(_do, fallback=fallback)
        logger.debug(f"LLM response length: {len(content)} chars")
        return content

    def create_message(self, role: str, content: str) -> Dict[str, str]:
        """
        创建消息对象

        Args:
            role: 角色，"user" 或 "assistant" 或 "system"
            content: 消息内容

        Returns:
            消息字典
        """
        return {"role": role, "content": content}

    async def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        fallback: Optional["LLMResponse"] = None,
        **kwargs
    ) -> LLMResponse:
        """
        带工具支持的聊天接口（带超时 + 分类重试 + 工具 JSON 容错兜底）

        Args:
            messages: 消息列表
            tools: 工具定义列表（OpenAI format）
            tool_choice: 工具选择策略 ("auto"/"required"/"none")
            temperature: 温度参数
            max_tokens: 最大token数
            fallback: 连续重试耗尽后的安全降级 LLMResponse；为 None 则抛出

        Returns:
            LLMResponse 对象
        """
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens

        # 准备请求参数
        request_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        if tools:
            request_params["tools"] = tools
            if tool_choice != "auto":
                request_params["tool_choice"] = tool_choice

        logger.debug(f"Calling LLM with {len(tools) if tools else 0} tools")

        async def _do() -> LLMResponse:
            response = await self.client.chat.completions.create(**request_params)
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # 提取工具调用（模型偶发残缺 JSON → 单独捕获以便触发重试）
            tool_calls = []
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        logger.warning("LLM returned malformed tool-call JSON, will retry")
                        raise
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    ))
                logger.debug(f"LLM requested {len(tool_calls)} tool calls")

            return LLMResponse(
                content=message.content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )

        return await self._with_retry(_do, fallback=fallback)

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        on_delta: Optional[Callable[[str], Any]] = None,
        stats: Optional[StreamStats] = None,
        fallback: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        流式版 chat：逐段把正文交给 on_delta，返回完整文本。

        用于「最终答案由一次纯生成完成」的场景（例如 Lead Agent 的汇总），
        这类调用不需要工具，但耗时最长，是用户感知延迟的主要来源。
        """
        fallback_response = (
            LLMResponse(content=fallback, tool_calls=[], finish_reason="chat_fallback")
            if fallback is not None else None
        )
        response = await self.chat_with_tools_stream(
            messages=messages,
            tools=None,
            temperature=temperature,
            max_tokens=max_tokens,
            on_delta=on_delta,
            stats=stats,
            fallback=fallback_response,
            **kwargs,
        )
        if response.content:
            return response.content
        return fallback or ""

    async def chat_with_tools_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        on_delta: Optional[Callable[[str], Any]] = None,
        stats: Optional[StreamStats] = None,
        fallback: Optional["LLMResponse"] = None,
        **kwargs
    ) -> LLMResponse:
        """
        流式版 chat_with_tools（带 TTFT 打点），返回结构与非流式版一致。

        与非流式版的差别：
        - 请求带 stream=True，逐段拿增量；
        - 每个正文增量立刻交给 on_delta（同步或协程函数都支持），用于把 token 实时推给前端；
        - 流式下 tool_calls 是分片返回的，这里按 index 累积成完整调用；
        - 计时写进 stats：ttft_ms 是首个正文增量到达的时刻，total_ms 是流结束。

        重试策略：只有「还没吐出任何增量」时才重试（此时重放安全）；一旦已经吐字，
        重试会让前端收到重复内容，因此改为返回半截结果并标 finish_reason="interrupted"。

        Args:
            on_delta: 增量回调，收到正文片段时调用
            stats: 传入 StreamStats 实例即可取回 TTFT 与总耗时
        """
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens

        request_params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kwargs,
        }
        if tools:
            request_params["tools"] = tools
            if tool_choice != "auto":
                request_params["tool_choice"] = tool_choice

        logger.debug(f"Calling LLM (stream) with {len(tools) if tools else 0} tools")

        async def _do() -> LLMResponse:
            started = time.perf_counter()
            content_parts: List[str] = []
            tool_acc: Dict[int, Dict[str, str]] = {}
            finish_reason = "stop"
            emitted = False

            try:
                stream = await self.client.chat.completions.create(**request_params)
                async for chunk in stream:
                    if stats is not None:
                        stats.chunks += 1
                        if stats.first_chunk_ms is None:
                            stats.first_chunk_ms = round(
                                (time.perf_counter() - started) * 1000, 1
                            )

                    choices = getattr(chunk, "choices", None) or []
                    if not choices:
                        continue
                    choice = choices[0]
                    if getattr(choice, "finish_reason", None):
                        finish_reason = choice.finish_reason

                    delta = getattr(choice, "delta", None)
                    if delta is None:
                        continue

                    text = getattr(delta, "content", None)
                    if text:
                        if stats is not None:
                            stats.content_chars += len(text)
                            if stats.ttft_ms is None:
                                stats.ttft_ms = round(
                                    (time.perf_counter() - started) * 1000, 1
                                )
                        content_parts.append(text)
                        emitted = True
                        if on_delta is not None:
                            await _call_maybe_async(on_delta, text)

                    for tc in getattr(delta, "tool_calls", None) or []:
                        index = getattr(tc, "index", 0) or 0
                        slot = tool_acc.setdefault(
                            index, {"id": "", "name": "", "arguments": ""}
                        )
                        if getattr(tc, "id", None):
                            slot["id"] = tc.id
                        function = getattr(tc, "function", None)
                        if function is not None:
                            if getattr(function, "name", None):
                                slot["name"] += function.name
                            if getattr(function, "arguments", None):
                                slot["arguments"] += function.arguments
                        emitted = True
            except RETRYABLE_EXCEPTIONS as e:
                if not emitted:
                    raise
                logger.warning(
                    f"[LLM stream] 已吐字后中断，返回半截结果: {type(e).__name__}: {e}"
                )
                finish_reason = "interrupted"

            if stats is not None:
                stats.total_ms = round((time.perf_counter() - started) * 1000, 1)

            tool_calls: List[ToolCall] = []
            for index in sorted(tool_acc):
                slot = tool_acc[index]
                if not slot["name"]:
                    continue
                try:
                    arguments = json.loads(slot["arguments"] or "{}")
                except json.JSONDecodeError:
                    logger.warning("LLM streamed malformed tool-call JSON, will retry")
                    raise
                tool_calls.append(ToolCall(
                    id=slot["id"] or f"call_{index}",
                    name=slot["name"],
                    arguments=arguments,
                ))

            return LLMResponse(
                content="".join(content_parts) or None,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )

        return await self._with_retry(_do, fallback=fallback)

    def create_tool_message(
        self,
        tool_call_id: str,
        tool_name: str,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        创建工具执行结果消息

        Args:
            tool_call_id: 工具调用ID
            tool_name: 工具名称
            result: 工具执行结果

        Returns:
            工具消息字典
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": json.dumps(result, ensure_ascii=False)
        }
