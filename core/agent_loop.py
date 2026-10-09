"""
Agent循环引擎
实现 LLM 驱动的 Skill 调用循环
支持短期记忆集成
支持约束验证（Harness Engineering）
"""
import uuid
import json
import re
import time
from typing import Dict, Any, List, Optional
from loguru import logger

from .state_manager import StateManager, TaskStatus
from .llm_client import LLMResponse, StreamStats, llm_safe_fallback_response

# Harness Engineering: 约束验证和自动修复
try:
    from constraints import ConstraintValidator
    from constraints.validator import (
        detect_high_risk_signals,
        is_constraint_enforce_enabled,
        max_risk_level,
    )
    from validation import AutoFixer
    CONSTRAINTS_ENABLED = True
except ImportError:
    logger.warning("Constraints module not found, running without constraint validation")
    CONSTRAINTS_ENABLED = False

    def is_constraint_enforce_enabled() -> bool:  # type: ignore
        return False

    def max_risk_level(*levels):  # type: ignore
        return "low"

    def detect_high_risk_signals(text):  # type: ignore
        return False


def _sanitize_final_answer(text: str) -> str:
    """
    净化最终答案，剔除工具调用残留。

    背景：deep_research 等 Skill 返回结构化 dict，经 create_tool_message 注入对话后，
    LLM 有时会把其中的 <tool_calls> 标记或工具 JSON 原样带进最终回答，导致答非所问。
    这里做三层清理：
      1. 去掉 <tool_calls> / <tool> 等 XML 残留片段
      2. 去掉 OpenAI 风格 function_call 的 JSON 参数块
      3. 若清理后为空（纯残留），触发调用方走强制重生成兜底
    """
    if not text:
        return text

    # 1) XML 风格工具调用残留：删除 <tool_calls>...</tool_calls> 等整个块（含内部内容）
    text = re.sub(
        r"<\s*/?\s*(tool_calls|tool_call|tool|function_call|invoke)\b[^>]*>.*?</\s*\1\s*>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # 若只有开标签没有闭标签，也把从标签到结尾的残留清掉
    text = re.sub(
        r"<\s*(tool_calls|tool_call|tool|function_call|invoke)\b[^>]*>[^<]*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # 孤立标签本身
    text = re.sub(r"<\s*/\s*(tool_calls|tool_call|tool|function_call|invoke)\s*>", "", text, flags=re.IGNORECASE)

    # 2) OpenAI function calling 残留：{"name": "...", "arguments": "..."} 整段
    text = re.sub(
        r"\s*\{\s*\"(?:name|function)\"\s*:\s*\"[^\"]*\"[^\}]*\}",
        "",
        text,
    )
    text = re.sub(r"\b(function_call|tool_calls)\b[^\n]*", "", text, flags=re.IGNORECASE)
    # 清理上述正则可能遗留的残缺 JSON 尾缀（如仅剩的 "}）以及开头的 {"name": 半截
    text = re.sub(r"\s*\"\s*\}\s*$", "", text)
    text = re.sub(r"^\s*\{\s*\"(?:name|function)\"\s*:\s*\"[^\"]*\"[^\n]*$", "", text)

    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned


def _format_tool_result_for_context(result) -> str:
    """
    将 Skill 执行结果格式化为紧凑字符串注入上下文，避免把整份工具返回塞进对话。

    - dict/list：只保留关键短字段的 JSON 摘要，截断到 600 字符
    - 其他：截断到 600 字符
    """
    limit = 600
    try:
        if isinstance(result, dict):
            keys = list(result.keys())
            if keys == ["success"] and result.get("success") is False:
                return f"skill_error: {str(result.get('error', ''))[:limit]}"
            return json.dumps(result, ensure_ascii=False)[:limit]
        if isinstance(result, (list, tuple)):
            return json.dumps(result, ensure_ascii=False)[:limit]
    except Exception:
        pass
    return str(result)[:limit]


class AgentLoop:
    """
    Agent循环引擎
    LLM 自主决策 Skill 调用，循环直到任务完成

    功能：
    - 支持短期记忆（ShortTermMemory）
    - 自动记录每轮的 user/assistant 消息
    """

    def __init__(self, max_iterations: int = 10, short_term_memory: Optional[Any] = None, max_tool_calls: int = 2):
        """
        初始化Agent循环引擎

        Args:
            max_iterations: 最大迭代次数（防止无限循环）
            short_term_memory: 短期记忆管理器（可选）
            max_tool_calls: 最大 Skill 调用次数（硬性限制，默认2次）
        """
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.state_manager = StateManager()
        self.short_term_memory = short_term_memory
        # 注意：单次运行的计数（tool_call_count）不挂在 self 上。
        # AgentLoop 是 Worker 级共享实例（Coordinator 走进程级缓存），若计数挂在 self，
        # 并发请求会在 await 让出后互相清零/累加，导致 max_tool_calls 限额失效
        # （表现为：误杀——只用 1 次就被强制收尾；或漏拦——突破上限拖慢响应）。
        # 因此 self 上只放只读配置，随任务变化的状态一律下沉到 run() 的局部变量。

        # Harness Engineering: 约束验证器和自动修复器（默认硬拦；CONSTRAINT_ENFORCE=0 退回 warn）
        self.validator = ConstraintValidator() if CONSTRAINTS_ENABLED else None
        self.auto_fixer = AutoFixer() if CONSTRAINTS_ENABLED else None
        if CONSTRAINTS_ENABLED:
            mode = "enforce" if is_constraint_enforce_enabled() else "warn"
            logger.debug(f"✅ Constraint validation enabled (mode={mode})")

    async def run(
        self,
        agent,
        input_data: Dict[str, Any],
        session_id: Optional[str] = None,
        on_delta: Optional[Any] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        执行Agent循环

        Args:
            agent: Agent实例
            input_data: 输入数据
            session_id: 会话 ID
            on_delta: 正文增量回调（同步或协程函数均可）；传入即启用流式，
                用于把 token 实时推给前端
            stream: 为 True 时也走流式（只取 TTFT 指标、不对外吐字）

        Returns:
            最终结果；流式开启时额外带 llm_ttft_ms 与 answer_ttft_ms
        """
        task_id = str(uuid.uuid4())
        state = self.state_manager.create_state(
            task_id=task_id,
            agent_id=agent.agent_id,
            input_data=input_data,
            max_iterations=self.max_iterations
        )

        # 单次运行的计数：局部变量，天然与其他并发 run 隔离
        tool_call_count = 0
        # 本轮观察到的最高风险等级（来自 Skill 返回的结构化 risk_level）。
        # 这是医疗安全判断的首要依据，不用"模型输出里有没有写胸痛"来决定。
        observed_risk = "low"

        # 观测：本次运行的耗时打点（配合 trace_id 排查每次 LLM/Skill 调用的开销）
        llm_calls_ms: List[float] = []
        skill_calls_ms: List[float] = []
        # 流式：每一轮的首 token 时间（TTFT），以及最终答案那一轮的值
        llm_ttft_ms: List[float] = []
        answer_ttft_ms: Optional[float] = None
        stream_enabled = bool(stream or on_delta is not None)

        logger.info(f"Starting Agent Loop for {agent.agent_id}, task_id={task_id}")

        try:
            state.status = TaskStatus.IN_PROGRESS

            record_memory = bool(input_data.get("record_memory", True))
            load_history = bool(input_data.get("load_history", True))

            # 初始化消息历史（包含历史对话）
            messages = self._initialize_messages(
                agent, input_data, session_id if load_history else None
            )

            # 记录用户消息到短期记忆（Swarm 子任务可关闭，避免污染主诉）
            if record_memory and self.short_term_memory and session_id:
                user_message = messages[-1]["content"] if messages else str(input_data)
                self.short_term_memory.add_message(
                    session_id=session_id,
                    role="user",
                    content=user_message
                )
                logger.debug(f"Recorded user message to short-term memory (session={session_id})")

            # 获取 Agent 的 Skills (OpenAI format)
            tools_openai_format = agent.get_tools_for_llm()

            logger.debug(f"Agent has {len(tools_openai_format) if tools_openai_format else 0} skills available")

            # 主循环：LLM → Skill Calls → Results → LLM
            while state.should_continue():
                state.iteration += 1
                logger.debug(f"=== Iteration {state.iteration}/{state.max_iterations} ===")

                try:
                    # 调用 LLM（可能返回 tool_calls）
                    t_llm = time.perf_counter()
                    llm_args = {
                        "messages": messages,
                        "tools": tools_openai_format,
                        "tool_choice": "auto",
                        "temperature": agent.config.get('temperature', 0.7),
                        "fallback": llm_safe_fallback_response(),
                    }
                    if stream_enabled:
                        # 流式：正文增量实时交给 on_delta，并取回本轮 TTFT
                        stream_stats = StreamStats()
                        llm_response: LLMResponse = await agent.llm_client.chat_with_tools_stream(
                            on_delta=on_delta,
                            stats=stream_stats,
                            **llm_args,
                        )
                        llm_ttft_ms.append(
                            stream_stats.ttft_ms
                            if stream_stats.ttft_ms is not None
                            else stream_stats.first_chunk_ms
                        )
                    else:
                        llm_response = await agent.llm_client.chat_with_tools(**llm_args)
                    llm_calls_ms.append(
                        round((time.perf_counter() - t_llm) * 1000, 1)
                    )

                    # 记录中间结果
                    state.add_intermediate_result({
                        'iteration': state.iteration,
                        'llm_response': {
                            'content': llm_response.content,
                            'tool_calls': [
                                {'name': tc.name, 'arguments': tc.arguments}
                                for tc in llm_response.tool_calls
                            ],
                            'finish_reason': llm_response.finish_reason
                        }
                    })

                    # 情况1: LLM 返回 tool_calls，执行 Skills
                    if llm_response.has_tool_calls():
                        # 硬性限制：检查是否已达到最大调用次数
                        if tool_call_count >= self.max_tool_calls:
                            logger.warning(f"⚠️ 已达到最大 Skill 调用次数限制 ({self.max_tool_calls})，强制生成最终答案")
                            # 强制要求 LLM 提供最终答案
                            messages.append({
                                'role': 'user',
                                'content': f'已完成 {self.max_tool_calls} 次信息检索。请基于已获取的信息提供最终答复。'
                            })
                            continue

                        logger.info(f"LLM requested {len(llm_response.tool_calls)} tool calls (当前已调用 {tool_call_count}/{self.max_tool_calls})")

                        # 添加 assistant 消息（包含 tool_calls）
                        messages.append(self._create_assistant_message_with_tools(llm_response))

                        # 记录 assistant 消息到短期记忆
                        if record_memory and self.short_term_memory and session_id:
                            tool_names = [tc.name for tc in llm_response.tool_calls]
                            self.short_term_memory.add_message(
                                session_id=session_id,
                                role="assistant",
                                content=f"调用工具：{', '.join(tool_names)}"
                            )

                        # 执行每个 Skill 调用
                        for tool_call in llm_response.tool_calls:
                            # 增加计数
                            tool_call_count += 1
                            logger.debug(f"Executing: {tool_call.name}({tool_call.arguments}) - 第 {tool_call_count} 次调用")

                            # Harness Engineering: 验证调用（warn 只记日志；enforce 不执行并回写拒绝原因）
                            blocked = False
                            if self.validator:
                                validation_result = self.validator.validate_tool_call(
                                    agent.agent_id,
                                    tool_call.name
                                )
                                if not validation_result.get("valid"):
                                    reason = validation_result.get("reason") or "Skill 不被允许"
                                    if is_constraint_enforce_enabled():
                                        blocked = True
                                        logger.warning(f"🚫 约束拦截(blocked): {reason}")
                                        tool_result = {
                                            "success": False,
                                            "blocked": True,
                                            "error": reason,
                                            "allowed_tools": validation_result.get("allowed_tools", []),
                                        }
                                        messages.append(
                                            agent.llm_client.create_tool_message(
                                                tool_call_id=tool_call.id,
                                                tool_name=tool_call.name,
                                                result=tool_result,
                                            )
                                        )
                                        if record_memory and self.short_term_memory and session_id:
                                            self.short_term_memory.add_message(
                                                session_id=session_id,
                                                role="tool",
                                                content=f"{tool_call.name}: blocked — {reason}",
                                            )
                                    else:
                                        logger.info(f"⚠️ 约束警告(warned): {reason}")

                            if blocked:
                                continue

                            t_skill = time.perf_counter()
                            tool_result = await agent.execute_tool(
                                tool_name=tool_call.name,
                                arguments=tool_call.arguments
                            )
                            skill_calls_ms.append(
                                round((time.perf_counter() - t_skill) * 1000, 1)
                            )

                            # 采集结构化风险等级：assess_risk 等 Skill 会返回 risk_level 字段。
                            # 取本轮观察到的最高等级（保守取最大值）。
                            if isinstance(tool_result, dict) and tool_result.get("risk_level"):
                                new_risk = max_risk_level(
                                    observed_risk, tool_result.get("risk_level")
                                )
                                if new_risk != observed_risk:
                                    logger.info(
                                        f"Risk level updated: {observed_risk} -> {new_risk} "
                                        f"(from {tool_call.name})"
                                    )
                                observed_risk = new_risk

                            # 添加结果消息（紧凑格式化，避免把整份工具返回塞进上下文）
                            messages.append(
                                agent.llm_client.create_tool_message(
                                    tool_call_id=tool_call.id,
                                    tool_name=tool_call.name,
                                    result=tool_result
                                )
                            )

                            # 记录结果到短期记忆（同样用紧凑格式化）
                            if record_memory and self.short_term_memory and session_id:
                                result_summary = _format_tool_result_for_context(tool_result)
                                self.short_term_memory.add_message(
                                    session_id=session_id,
                                    role="tool",
                                    content=f"{tool_call.name}: {result_summary}"
                                )

                        # 继续下一轮循环
                        continue

                    # 情况2: LLM 返回文本响应，任务完成
                    else:
                        logger.info(f"LLM provided final response (no tool calls)")

                        # 流式：最终答案这一轮的 TTFT，才是用户感知到的响应时间
                        answer_ttft_ms = llm_ttft_ms[-1] if llm_ttft_ms else None

                        # Harness Engineering: 验证和修复输出
                        final_answer = llm_response.content

                        # 输出净化：剔除工具调用残留（<tool_calls> 等），避免答非所问
                        if final_answer:
                            sanitized = _sanitize_final_answer(final_answer)
                            if sanitized != final_answer:
                                logger.info("🔧 最终答案已净化（剔除工具残留）")
                                final_answer = sanitized

                        # 若净化后为空（纯残留），强制要求 LLM 重生成，而不是返回空串
                        if not final_answer:
                            logger.warning("⚠️ 最终答案净化后为空，强制要求 LLM 重新生成")
                            messages.append({
                                'role': 'user',
                                'content': '上一条回复包含工具调用残留，请忽略工具内容，直接用文字回答用户问题，不要输出任何 XML 或 JSON 格式。'
                            })
                            continue

                        # 最终判定：把「用户输入」的高危信号也纳入。
                        # 用户说"胸痛"但全程没调 assess_risk 时，工具侧 risk_level 仍是 low，
                        # 只依赖工具侧会漏判——所以输入端也要扫一遍（扫输入，不扫模型输出）。
                        input_text = input_data.get('question') or ""
                        effective_risk = max_risk_level(
                            observed_risk,
                            "high" if detect_high_risk_signals(input_text) else "low",
                        )

                        if self.validator and final_answer:
                            validation_result = self.validator.validate_output(
                                agent.agent_id,
                                final_answer,
                                risk_level=effective_risk,
                            )

                            if not validation_result.get("valid"):
                                logger.warning(
                                    f"⚠️ 输出约束违规: {validation_result.get('violations')}"
                                )

                                # 自动修复
                                if self.auto_fixer and validation_result.get("auto_fixable"):
                                    fixed_answer = self.auto_fixer.fix_output(
                                        final_answer,
                                        validation_result.get("auto_fixable", []),
                                        risk_level=effective_risk,
                                    )
                                    if fixed_answer != final_answer:
                                        logger.info("🔧 输出已自动修复")
                                        final_answer = fixed_answer

                        # 记录最终回答到短期记忆
                        if record_memory and self.short_term_memory and session_id:
                            self.short_term_memory.add_message(
                                session_id=session_id,
                                role="assistant",
                                content=final_answer or "(empty response)"
                            )
                            logger.debug(f"Recorded final answer to short-term memory (session={session_id})")

                        result = {
                            'answer': final_answer,
                            'iterations': state.iteration,
                            'agent_id': agent.agent_id,
                            'tool_calls': tool_call_count,
                            # 结构化风险等级：供上层（日志/告警/审计）使用
                            'risk_level': effective_risk,
                            # 观测：本次运行的耗时打点
                            'llm_calls': len(llm_calls_ms),
                            'llm_total_ms': round(sum(llm_calls_ms), 1),
                            'skill_calls': len(skill_calls_ms),
                            'skill_total_ms': round(sum(skill_calls_ms), 1),
                            # 流式指标：streamed=False 时 TTFT 不可得（用户要等整段生成完）
                            'streamed': stream_enabled,
                            'llm_ttft_ms': llm_ttft_ms,
                            'answer_ttft_ms': answer_ttft_ms,
                        }

                        # 让 Agent 进行结果后处理（如提取建议等）
                        if hasattr(agent, 'post_process_result'):
                            result = await agent.post_process_result(result, final_answer)

                        state.mark_completed(result)
                        break

                except Exception as e:
                    logger.error(f"Error in iteration {state.iteration}: {e}")
                    if state.iteration >= state.max_iterations:
                        state.mark_failed(str(e))
                        break
                    # 否则继续尝试

            # 如果达到最大迭代次数但没有完成
            if not state.is_completed():
                logger.warning(f"Max iterations reached without completion")

                # 强制调用 LLM 生成最终总结
                try:
                    logger.info("Forcing LLM to provide final answer")

                    # 添加强制总结的提示
                    messages.append({
                        'role': 'user',
                        'content': '请基于以上信息，提供最终的答复。'
                    })

                    # 调用 LLM（禁用 function calling）
                    t_final = time.perf_counter()
                    final_response = await agent.llm_client.chat_with_tools(
                        messages=messages,
                        tools=None,
                        temperature=0.7,
                        fallback=llm_safe_fallback_response(),
                    )
                    llm_calls_ms.append(
                        round((time.perf_counter() - t_final) * 1000, 1)
                    )

                    result = {
                        'answer': final_response.content or '抱歉，未能完成任务',
                        'iterations': state.iteration,
                        'tool_calls': tool_call_count,
                        'llm_calls': len(llm_calls_ms),
                        'llm_total_ms': round(sum(llm_calls_ms), 1),
                        'skill_calls': len(skill_calls_ms),
                        'skill_total_ms': round(sum(skill_calls_ms), 1),
                        'warning': 'max_iterations_reached'
                    }

                    # 记录最终回答到短期记忆
                    if record_memory and self.short_term_memory and session_id:
                        self.short_term_memory.add_message(
                            session_id=session_id,
                            role="assistant",
                            content=result['answer']
                        )

                    state.mark_completed(result)
                    logger.info("Generated fallback answer after max iterations")

                except Exception as e:
                    logger.error(f"Failed to generate fallback answer: {e}")
                    # 降级到简单提取
                    result = {
                        'answer': '抱歉，系统在处理您的问题时遇到了问题。建议您简化问题或稍后重试。',
                        'iterations': state.iteration,
                        'tool_calls': tool_call_count,
                        'llm_calls': len(llm_calls_ms),
                        'llm_total_ms': round(sum(llm_calls_ms), 1),
                        'skill_calls': len(skill_calls_ms),
                        'skill_total_ms': round(sum(skill_calls_ms), 1),
                        'warning': 'max_iterations_reached',
                        'error': str(e)
                    }
                    state.mark_completed(result)

            logger.info(f"Agent Loop finished: status={state.status.value}, iterations={state.iteration}")
            return state.final_result or {}

        except Exception as e:
            logger.error(f"Agent Loop failed: {e}")
            state.mark_failed(str(e))
            raise
        finally:
            # StateManager 同样是共享实例：任务结束即回收，
            # 否则 states 字典随请求数单调增长（内存泄漏）。
            self.state_manager.delete_state(task_id)

    def _initialize_messages(self, agent, input_data: Dict[str, Any], session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """初始化消息列表，包含历史对话上下文"""
        messages = []

        # 系统提示词
        system_prompt = agent.get_system_prompt()
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })

        # 加载历史对话（短期记忆）
        if self.short_term_memory and session_id:
            history = self.short_term_memory.get_history(session_id, limit=5)  # 最近5轮对话
            if history:
                logger.info(f"Loaded {len(history)} historical messages from short-term memory")
                messages.extend(history)

        # 用户输入
        user_message = agent.format_user_input(input_data)
        messages.append({
            'role': 'user',
            'content': user_message
        })

        return messages

    def _create_assistant_message_with_tools(self, llm_response: LLMResponse) -> Dict[str, Any]:
        """创建包含 tool_calls 的 assistant 消息"""
        message = {
            'role': 'assistant',
            'content': llm_response.content or None
        }

        # 添加 tool_calls（OpenAI 格式）
        if llm_response.tool_calls:
            message['tool_calls'] = [
                {
                    'id': tc.id,
                    'type': 'function',
                    'function': {
                        'name': tc.name,
                        'arguments': json.dumps(tc.arguments, ensure_ascii=False)
                    }
                }
                for tc in llm_response.tool_calls
            ]

        return message
