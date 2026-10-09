"""
SwarmCoordinator：Swarm 入口和智能路由

注意：这不是编排器！
- 只负责路由决策：简单问题 → 单 Agent，复杂问题 → Swarm
- 不控制 Agent 执行
- 不编排任务顺序

类比：交通信号灯，决定车辆走哪条路，但不控制车辆如何行驶
"""
import asyncio
import os
import re
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from loguru import logger

from core import LLMClient
from core.llm_client import StreamStats
# 可观测性：本请求的 Token 账本 / Span 记录器（ContextVar 里放可变对象，见 observability.py）
from core.observability import (
    NOOP_SPANS,
    ROOT_SPAN,
    ROUTE_SPAN,
    SYNTH_SPAN,
    WORKER_POOL_SPAN,
    begin_observability,
    current_spans,
    format_span_waterfall,
)
from .shared_context import SharedContext
from .lead_agent import LeadAgent
from .events import Event, EventType
from agents import ConsultationAgent, DiagnosticAgent, ResearchAgent
from memory import SessionSummaryManager, SessionSummary, ShortTermMemory, LongTermMemory
from memory.patient_profile import PatientProfile
from memory.short_term import fit_history_to_budget, estimate_tokens
from memory.running_summary import (
    RunningSummaryManager,
    split_history_by_budget,
    generate_summary,
)
from constraints import ConstraintValidator
from constraints.validator import detect_high_risk_signals, get_agent_role
from validation import AutoFixer


# --------------------------------------------------------------------------- #
# 单 Agent 直出的「编排头」清洗
#
# ResearchAgent 的 system prompt 规定了三段内部结构
# （【文献检索结果】/【证据摘要】/【综合评估】），那是**给 LeadAgent 消费的输入契约**，
# 不能动；但单 Agent 路由下它会被直接端给用户，其中「关键词：… 找到相关文献：…」
# 属于检索元数据，对患者无意义且带调试感。
# 因此只在**出口**剥掉这一段元数据，正文（证据摘要 / 综合评估）完整保留。
# --------------------------------------------------------------------------- #
_META_SECTION = "【文献检索结果】"
# 判定「这段确实是我们认识的那种检索元数据头」的线索词
_META_HINTS = ("关键词", "找到相关文献", "检索词", "检索结果", "相关文献")
# 下一节标记（【证据摘要】【综合评估】等），元数据头到此为止
_NEXT_SECTION = re.compile(r"【[^】\n]{2,16}】")
# 孤立的分隔线：整行只有 --- / *** / ___ （markdown 表格分隔行含 | ，不会被匹配）
_ISOLATED_RULE = re.compile(r"^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$\n?", re.M)
# 模型偶尔在节标记前多写一个 markdown 标题号：# 【综合评估】 → 【综合评估】
_STRAY_HASH = re.compile(r"^([ \t]*)#{1,6}[ \t]*(?=【)", re.M)
# 元数据头最多吃掉的字符数，防止在没有下一节标记时误吞正文
_META_MAX_CHARS = 600


class SwarmCoordinator:
    """
    Swarm 协调器

    职责：
    1. 智能路由（简单 → 单 Agent，复杂 → Swarm）
    2. 初始化 SharedContext
    3. 启动和监控 Swarm
    4. 生成 SessionSummary

    不做：
    - 不编排 Worker 执行顺序
    - 不直接调用 Worker
    - 不控制任务分配
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        enable_swarm: bool = True
    ):
        self.llm_client = llm_client or LLMClient()
        self.enable_swarm = enable_swarm

        # 初始化 Agent
        self.lead_agent = LeadAgent(llm_client=self.llm_client)
        self.consultation_agent = ConsultationAgent()
        self.diagnostic_agent = DiagnosticAgent()
        self.research_agent = ResearchAgent()

        # Worker 池
        self.worker_pool: List[Any] = [
            self.consultation_agent,
            self.diagnostic_agent,
            self.research_agent
        ]

        # 最终答案的安全网：Lead 汇总后仍要过一遍输出校验与自动修复
        self.validator = ConstraintValidator()
        self.auto_fixer = AutoFixer()

        # 记忆管理器
        self.session_manager = SessionSummaryManager()
        self.short_term_memory = ShortTermMemory(storage_type="memory")  # 或 "redis"
        self.long_term_memory = LongTermMemory()
        # 患者档案：跨会话确定性事实（长期侧 L3）
        self.patient_profile = PatientProfile(llm_client=self.llm_client)
        # 运行摘要：短期侧 L2（触发式语义摘要，超预算时压缩最老消息）
        self.running_summary = RunningSummaryManager()
        # 短期记忆注入的 token 预算（物理约束是 token，不是条数）
        self.history_max_tokens = int(os.getenv("MEDIX_HISTORY_MAX_TOKENS", "8000"))
        # 运行摘要可占的 token 配额（给 L1 原文留出剩余空间）
        self.summary_token_quota = min(2000, int(self.history_max_tokens * 0.25))
        # 患者档案提取节流：每患者记录已提取的 user 消息数，避免每轮重复 LLM 提取
        self._last_fact_user_msgs: Dict[str, int] = {}

        # 将短期记忆注入到所有 Worker Agent 的 Loop
        # 注意：LeadAgent 不继承 BaseAgent，没有 loop 属性，不需要注入
        for worker in self.worker_pool:
            if hasattr(worker, 'loop'):
                worker.loop.short_term_memory = self.short_term_memory

        # 让 LeadAgent 的"可用 Worker"提示词随 worker_pool 自动生成（新增 Worker 自动被识别）
        self._sync_lead_worker_profiles()

        logger.info(f"SwarmCoordinator initialized with {len(self.worker_pool)} workers")
        logger.info(f"Memory system: short_term={self.short_term_memory.storage_type}, long_term={'enabled' if self.long_term_memory.enabled else 'disabled'}")

    def _collapse_subtasks(self, subtasks: List[Dict[str, Any]], question: str) -> List[Dict[str, Any]]:
        """减少不必要 Swarm：同 Agent 合并；指南类只留 research。"""
        if not subtasks or len(subtasks) <= 1:
            return subtasks or []
        agent_ids = [t.get("assigned_agent") for t in subtasks]
        if len(set(agent_ids)) == 1:
            return [subtasks[0]]
        q = question or ""
        guide_kw = ("指南", "共识", "诊疗规范", "专家共识", "推荐意见")
        if any(k in q for k in guide_kw) and "research_agent" in agent_ids:
            for t in subtasks:
                if t.get("assigned_agent") == "research_agent":
                    return [t]
        return subtasks

    def _enforce_required_agents(
        self,
        subtasks: List[Dict[str, Any]],
        question: str
    ) -> List[Dict[str, Any]]:
        """Harness 硬约束：问题命中 agent_selection_rules 时，强制确保对应 Agent 参与。

        这是纯规则、零 LLM 依赖：即使 LeadAgent 只把高危问题分给了
        consultation_agent，也必须在路由前补上 diagnostic_agent 的风险评估——
        不能指望模型"自觉"把危险症状交给正确的 Agent。

        - 开启 Swarm 时：补一个必需 Agent 的子任务（自然触发多 Agent 协作）
        - 单 Agent 模式：把第一个子任务改派给必需 Agent（高危 → diagnostic）
        """
        try:
            from constraints.validator import get_required_agents
            required = get_required_agents(question)
        except Exception as e:
            logger.warning(f"get_required_agents failed (skip enforcement): {e}")
            return subtasks
        if not required:
            return subtasks

        assigned = {t.get("assigned_agent") for t in subtasks}
        missing = [a for a in required if a not in assigned]
        if not missing:
            return subtasks

        if self.enable_swarm:
            # 多 Agent 可用：补必需 Agent 的子任务 → 该问题必然走 Swarm
            for agent_id in missing:
                subtasks.append({
                    "type": f"{agent_id}_task",
                    "description": self._forced_subtask_desc(agent_id, question),
                    "assigned_agent": agent_id,
                })
                logger.warning(
                    f"安全规则强制加入 Agent: {agent_id}（问题命中高危/关键词约束）"
                )
        else:
            # 单 Agent 模式：把已有任务改派给必需的 Agent
            if subtasks:
                first = subtasks[0]
                required_agent = missing[0]
                first["assigned_agent"] = required_agent
                first["description"] = self._forced_subtask_desc(
                    required_agent, question
                )
                logger.warning(
                    f"单 Agent 模式：高危/关键词约束将任务改派给 {required_agent}"
                )
            else:
                required_agent = missing[0]
                subtasks.append({
                    "type": f"{required_agent}_task",
                    "description": self._forced_subtask_desc(
                        required_agent, question
                    ),
                    "assigned_agent": required_agent,
                })
        return subtasks

    @staticmethod
    def _forced_subtask_desc(agent_id: str, question: str) -> str:
        """为强制加入的子任务生成描述。"""
        head = (question or "")[:50]
        if agent_id == "diagnostic_agent":
            return (
                f"对「{head}」进行症状风险等级评估与模式分析，"
                "明确严重程度与是否需要立即就医"
                "（本子任务由安全规则强制加入，不允许删除）"
            )
        if agent_id == "research_agent":
            return (
                f"检索「{head}」相关的权威指南或最新循证证据"
                "（本子任务由安全规则强制加入，不允许删除）"
            )
        return f"处理问题「{head}」（本子任务由安全规则强制加入，不允许删除）"

    def _ensure_anchor_in_subtasks(
        self,
        subtasks: List[Dict[str, Any]],
        session_anchor: str
    ) -> List[Dict[str, Any]]:
        """follow-up 子任务 description 强制带上会话锚点。"""
        if not session_anchor or not subtasks:
            return subtasks or []
        prefix = f"承接：{session_anchor}"
        for t in subtasks:
            desc = (t.get("description") or "").strip()
            if prefix in desc or session_anchor in desc:
                continue
            t["description"] = f"{prefix}。{desc}" if desc else prefix
        return subtasks

    def _sync_lead_worker_profiles(self):
        """
        把当前 worker_pool 的角色画像同步给 LeadAgent，
        使其「可用的 Worker Agents」提示词随 worker_pool 自动生成。
        画像来源：constraints/agent_constraints.yaml 的 role 块（单一来源），
        缺失时用 capabilities / agent_id 兜底，保证新 Worker 至少可被识别。
        """
        profiles = []
        for worker in self.worker_pool:
            agent_id = getattr(worker, 'agent_id', None)
            if not agent_id:
                continue
            role = get_agent_role(agent_id)
            capabilities = []
            if hasattr(worker, 'get_capabilities'):
                try:
                    capabilities = worker.get_capabilities() or []
                except Exception:
                    capabilities = []
            profiles.append({
                "agent_id": agent_id,
                "display": role.get("display") or agent_id,
                "specialties": role.get("specialties") or capabilities,
                "scenarios": role.get("scenarios") or [],
            })
        self.lead_agent.set_worker_profiles(profiles)
        logger.info(f"LeadAgent worker profiles synced: {[p['agent_id'] for p in profiles]}")

    def register_worker(self, worker: Any):
        """
        注册一个新的 Worker Agent（运行时/扩展点）。

        只需把 Worker 实例传进来即可：
        - 自动加入 worker_pool（参与并行认领）
        - 自动注入短期记忆
        - 自动同步 LeadAgent 的「可用 Worker」提示词（新 Worker 立即可被分配任务）
        - _get_agent_by_id 动态遍历 worker_pool，无需再改映射

        注意：Worker 的角色画像（display/specialties/scenarios）来自
        constraints/agent_constraints.yaml 的 agents.<agent_id>.role；
        若 yaml 未配置，则回退用 capabilities/agent_id，保证至少可被 LeadAgent 识别。
        """
        if hasattr(worker, 'loop'):
            worker.loop.short_term_memory = self.short_term_memory
        self.worker_pool.append(worker)
        self._sync_lead_worker_profiles()
        logger.info(f"Registered new worker: {getattr(worker, 'agent_id', '?')}")

    def _get_agent_by_id(self, agent_id: str):
        """根据 agent_id 返回对应的 Agent 实例（动态遍历 worker_pool，新增 Worker 自动生效）"""
        for worker in self.worker_pool:
            if getattr(worker, 'agent_id', None) == agent_id:
                return worker
        return None

    async def process(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        on_delta: Optional[Any] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        处理用户问题

        Args:
            question: 用户问题
            context: 额外上下文（年龄、既往史等）
            session_id: 会话ID（如果不提供，将自动生成）
            on_delta: 正文增量回调；传入即对流经用户的最终生成启用流式
            stream: 为 True 时也启用流式（只取 TTFT 指标，不对外吐字）

        Returns:
            处理结果；流式时带 answer_ttft_ms
        """
        start_time = datetime.now()
        if session_id is None:
            session_id = f"{start_time.strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"

        logger.info(f"Processing question (session={session_id}): {question[:50]}...")

        # 患者档案键：优先 user_id（api/app.py 已透传），缺失时退化为 session_id
        patient_id = (context or {}).get("user_id") or session_id

        # Mem0 记忆空间键：仅显式 user_id 时按用户隔离；
        # 缺省（demo/eval/旧调用）回落全局共享空间 "medix_user"，保持单用户原行为
        mem_user_id = (context or {}).get("user_id") or "medix_user"

        # ===== 统一的记忆检索（所有模式都使用）=====
        # 1. 检索短期记忆（当前会话历史）+ 用户主诉轮次（抗 Swarm 污染）
        #    先取原始上限 200 条（熵管理已压缩/去重），再按 token 预算倒推截断——
        #    记忆的物理约束是 token 而不是条数。
        recent_history = self.short_term_memory.get_recent_messages(
            session_id=session_id,
            limit=200
        )

        # 1.2 L2 运行摘要：历史超预算时，把"最老、超出预算"的消息压缩为语义摘要。
        #     摘要走独立注入通道，与 L1 原文、L3 患者档案互不冲突（白名单天然跳过压缩）。
        running_summary_block = ""
        if recent_history:
            raw_cost = sum(
                estimate_tokens(str(m.get("content") or ""))
                for m in recent_history
            )
            if raw_cost > self.history_max_tokens:
                excess_history, recent_history = split_history_by_budget(
                    recent_history,
                    self.history_max_tokens - self.summary_token_quota,
                )
                if excess_history:
                    running_summary_block = self.running_summary.get(session_id)
                    # 节流：会话出现新的 user 消息才重新生成（短超时，失败不阻塞主路径）
                    session_hist = self.short_term_memory.get_session(session_id)
                    user_count = (
                        len([m for m in session_hist.messages if m.get("role") == "user"])
                        if session_hist else 0
                    )
                    if self.running_summary.should_regenerate(session_id, user_count):
                        try:
                            running_summary_block = await asyncio.wait_for(
                                generate_summary(
                                    self.llm_client,
                                    running_summary_block,
                                    excess_history,
                                ),
                                timeout=2.5,
                            )
                            self.running_summary.set(
                                session_id, running_summary_block, user_count
                            )
                        except Exception as e:
                            logger.warning(f"running summary skipped: {type(e).__name__}")

        recent_history = fit_history_to_budget(
            recent_history, self.history_max_tokens
        )
        prior_turns = self.short_term_memory.get_user_turns(session_id)

        # 2. 检索长期记忆（相似历史会话）——短超时，避免拖慢主路径
        similar_memories = []
        try:
            similar_memories = await asyncio.wait_for(
                asyncio.to_thread(
                    self.long_term_memory.search_similar_sessions,
                    question,
                    3,
                    user_id=mem_user_id,
                ),
                timeout=2.0,
            )
        except Exception as e:
            logger.warning(f"long-term memory search skipped: {type(e).__name__}")

        # 3. 构建增强上下文（强制会话锚点，避免追问丢主诉）
        enhanced_context = dict(context or {})
        is_followup = bool(prior_turns or recent_history)
        session_anchor = ""
        if is_followup:
            session_anchor = self.short_term_memory.extract_session_anchor(
                session_id=session_id,
                messages=recent_history,
            )
            enhanced_context["is_followup"] = True
            enhanced_context["recent_history"] = True
            if session_anchor:
                enhanced_context["session_anchor"] = session_anchor
            logger.info(
                f"follow-up: prior_turns={len(prior_turns)}, "
                f"msgs={len(recent_history)}, "
                f"session_anchor={session_anchor[:80] if session_anchor else '(empty)'}"
            )

        # 3.5 患者档案注入（长期侧 L3）：白名单强制（过敏/当前用药）+ 其余按相关性筛选
        try:
            patient_facts_block = self.patient_profile.build_injection_block(
                patient_id, question
            )
            if patient_facts_block:
                enhanced_context["patient_facts"] = patient_facts_block
                logger.info(f"patient_facts injected (patient={patient_id}): {patient_facts_block[:100]}")
        except Exception as e:
            logger.warning(f"patient profile injection skipped: {e}")

        # 3.6 运行摘要注入（L2）：仅在本会话历史超预算时携带（更早轮次的背景）
        if running_summary_block:
            enhanced_context["running_summary"] = running_summary_block
            logger.info(
                f"running_summary injected (session={session_id}): {running_summary_block[:60]}"
            )

        # 记录本轮原始用户问题（供后续轮次抽锚点；须在抽锚点之后）
        self.short_term_memory.record_user_question(session_id, question)

        # 添加长期记忆（参考案例，勿当作本会话用户事实）
        if similar_memories:
            enhanced_context["historical_cases"] = [
                {
                    "summary": mem["content"],
                    "score": mem["score"]
                }
                for mem in similar_memories
            ]
            logger.info(f"Found {len(similar_memories)} similar historical cases from long-term memory")

        # 可观测性：本请求的 Span 记录器（不在请求上下文里时退化为空记录器）
        spans = current_spans() or NOOP_SPANS

        # Step 1: LeadAgent 分解任务
        t_lead = time.perf_counter()
        assessment = await self.lead_agent.assess_and_decompose(question, enhanced_context)
        # 观测：各环节耗时（配合 trace_id 定位性能瓶颈）
        timings: Dict[str, Any] = {
            "lead_decompose_ms": round((time.perf_counter() - t_lead) * 1000, 1),
        }
        # 瀑布流第一层：任务拆解（子节点由 LeadAgent / AgentLoop 自己挂到本节点之下）
        spans.mark(
            ROUTE_SPAN,
            parent=ROOT_SPAN,
            duration_ms=timings["lead_decompose_ms"],
            detail="LeadAgent 任务拆解",
        )
        subtasks = self._collapse_subtasks(assessment.get("subtasks", []), question)
        session_anchor = enhanced_context.get("session_anchor") or ""
        if session_anchor:
            subtasks = self._ensure_anchor_in_subtasks(subtasks, session_anchor)
        # Harness 硬约束：高危症状等必须包含指定 Agent（不依赖 LLM 分解自觉）
        subtasks = self._enforce_required_agents(subtasks, question)
        assessment["subtasks"] = subtasks

        logger.info(f"LeadAgent 分解任务：{len(subtasks)} 个")

        # Step 2: 根据任务数量路由
        final_answer = None
        mode = None

        if len(subtasks) == 1:
            # 单任务 → 直接调用对应 Agent
            task = subtasks[0]
            agent_id = task.get("assigned_agent")
            agent = self._get_agent_by_id(agent_id)

            if agent is None:
                # 如果找不到 Agent，降级到 ConsultationAgent
                logger.warning(f"Unknown agent_id: {agent_id}, fallback to ConsultationAgent")
                agent = self.consultation_agent

            logger.info(f"Route: Single Agent ({agent_id})")
            mode = "single_agent"
            t_agent = time.perf_counter()
            result = await agent.process(
                {
                    'question': question,
                    'context': enhanced_context,
                    'session_id': session_id
                },
                on_delta=on_delta,
                stream=stream,
            )
            timings[f"agent_{agent_id}_ms"] = round((time.perf_counter() - t_agent) * 1000, 1)
            # 瀑布流：单 Agent 路由下，这个 Agent 就是唯一的工作节点
            spans.mark(
                f"worker_{agent_id}",
                parent=ROOT_SPAN,
                duration_ms=timings[f"agent_{agent_id}_ms"],
                detail=f"单 Agent 路由 · {len(subtasks)} 个子任务",
            )
            final_answer = result.get('answer', '')

            result.update({
                'swarm_enabled': False,
                'session_id': session_id,
                'route_reason': f'单任务路由到 {agent_id}'
            })

            result['disclaimer'] = self._resolve_disclaimer(
                final_answer, result.get('disclaimer'), timeout_occurred=False
            )
            if 'suggestions' not in result:
                result['suggestions'] = []

        elif len(subtasks) >= 2 and self.enable_swarm:
            # 多任务 → 启动 Swarm
            logger.info(f"Route: Swarm (Multi-Agent Collaboration) - {len(subtasks)} tasks")
            mode = "swarm"
            result = await self._process_with_swarm(
                question=question,
                context=enhanced_context,
                assessment=assessment,
                session_id=session_id,
                start_time=start_time,
                on_delta=on_delta,
                stream=stream
            )
            final_answer = result.get('answer', '')

            # 观测：合并 Lead 分解耗时到 Swarm 内部耗时
            result.setdefault("timings", {})
            result["timings"]["lead_decompose_ms"] = timings["lead_decompose_ms"]
            logger.info(f"trace timings: {result['timings']}")

            # Swarm 模式已经在 _process_with_swarm 中保存了长期记忆，直接返回
            return result

        else:
            # 0 个子任务或 Swarm 未开启 → 单 Agent 处理
            if len(subtasks) == 0:
                logger.warning("No subtasks generated, fallback to ConsultationAgent")
                mode = "fallback"
                agent = self.consultation_agent
            else:
                # Swarm 未开启但有多个子任务：不再无条件 consultation。
                # 优先选安全规则要求的 Agent（高危→diagnostic），其次选第一个子任务指定的
                # Agent——否则 _enforce_required_agents 刚补上的 diagnostic 子任务
                # 会被这里无脑降级回 consultation，形成安全缺口。
                logger.info(
                    f"Swarm disabled ({len(subtasks)} subtasks), routing to single agent"
                )
                mode = "disabled_swarm"
                try:
                    from constraints.validator import get_required_agents
                    required = get_required_agents(question)
                except Exception:
                    required = []
                preferred = (
                    required[0]
                    if required
                    else subtasks[0].get("assigned_agent")
                )
                agent = self._get_agent_by_id(preferred) or self.consultation_agent
                if getattr(agent, "agent_id", None) != preferred:
                    logger.warning(
                        f"Preferred agent {preferred} not found, "
                        f"fallback to {getattr(agent, 'agent_id', '?')}"
                    )

            t_agent = time.perf_counter()
            result = await agent.process(
                {
                    'question': question,
                    'context': enhanced_context,
                    'session_id': session_id
                },
                on_delta=on_delta,
                stream=stream,
            )
            timings[f"agent_{agent.agent_id}_ms"] = round(
                (time.perf_counter() - t_agent) * 1000, 1
            )
            # 瀑布流：Swarm 未开启 / 无子任务时的降级单 Agent 路径
            spans.mark(
                f"worker_{agent.agent_id}",
                parent=ROOT_SPAN,
                duration_ms=timings[f"agent_{agent.agent_id}_ms"],
                detail=f"单 Agent 路由（{mode}）",
            )
            # 出口清洗：单 Agent 直出时剥掉 ResearchAgent 的检索元数据头与装饰性分隔线
            # （只动「编排头」，正文与结论完整保留；同时在 result 里写回，保证
            #  SSE done.answer 与 /v1/chat 的 answer 都是清洗后的文本）
            final_answer = self._sanitize_display_answer(result.get('answer', ''))
            result['answer'] = final_answer
            result.update({
                'swarm_enabled': False,
                'session_id': session_id
            })
            result['disclaimer'] = self._resolve_disclaimer(
                final_answer, result.get('disclaimer'), timeout_occurred=False
            )

        # 观测：非 Swarm 路径输出分环节耗时（Swarm 路径已在分支内合并）
        result["timings"] = timings
        logger.info(f"trace timings: {timings}")

        # ===== 统一的记忆保存（非 Swarm 模式）=====
        end_time = datetime.now()

        # 注意：短期记忆已经在 Agent Loop 中保存了，这里不需要重复保存

        # 保存到长期记忆
        try:
            self.long_term_memory.add_session_summary(
                session_id=session_id,
                question=question,
                answer=final_answer,
                user_id=mem_user_id,
                metadata={
                    "mode": mode,
                    "subtasks_count": len(subtasks),
                    "total_time": (end_time - start_time).total_seconds(),
                }
            )
            logger.info(
                f"Saved to long-term memory (session={session_id}, mode={mode}, user={mem_user_id})"
            )
        except Exception as e:
            logger.error(f"Failed to save to long-term memory: {e}")

        # 提取并更新患者档案（长期确定性事实；短超时，失败不阻塞主路径）
        await self._update_patient_facts(patient_id, session_id)

        return result

    async def _update_patient_facts(self, patient_id: str, session_id: str) -> None:
        """从本会话 user 消息提取并更新患者档案（节流 + 短超时）。"""
        if not patient_id:
            return
        try:
            history = self.short_term_memory.get_session(session_id)
            user_msgs = (
                [m for m in history.messages if m.get("role") == "user"]
                if history else []
            )
            if not user_msgs:
                return
            # 节流：只有本会话出现新的 user 消息才重新提取，避免每轮重复 LLM 调用
            n = len(user_msgs)
            if n <= self._last_fact_user_msgs.get(patient_id, 0):
                return
            added = await asyncio.wait_for(
                self.patient_profile.extract_and_update(patient_id, user_msgs),
                timeout=3.0,
            )
            self._last_fact_user_msgs[patient_id] = n
            if added:
                logger.info(
                    f"Patient profile updated: +{added} facts (patient={patient_id})"
                )
        except Exception as e:
            logger.warning(f"Patient profile extraction skipped: {type(e).__name__}")

    def _sanitize_display_answer(self, answer: str) -> str:
        """剥掉单 Agent 直出时的「检索元数据头」与装饰性分隔线。

        只做无损的两件事：
        1. 答案开头附近（前 80 字内）出现 `【文献检索结果】`，且这段里含
           `关键词`/`找到相关文献` 等线索时，把这段**元数据头**删到下一个 `【…】` 节标记为止
           （没有节标记时只删连续的元数据行，绝不吞正文）；
        2. 删掉孤立成行的 `---` / `***` / `___`（纯装饰；markdown 表格分隔行含 `|`，不受影响）。
        附带一个窄修正：节标记前多写的 markdown 标题号（`# 【综合评估】` → `【综合评估】`）。

        三条设计约束：
        - **正常回答字节级不变**：没命中上述形态时直接原样返回（收尾的空白整理也只在真的删过东西时才做）；
        - **不改 ResearchAgent 的 prompt**：那三段结构是 Swarm 里 LeadAgent 的输入契约；
        - **只在单 Agent 直出路径调用**：Swarm 的答案是 Lead 重写过的正文，
          它可能合法使用 `---` 做分隔，不该在这里被动刀。
        """
        if not isinstance(answer, str) or not answer:
            return answer

        text = answer

        # 1) 检索元数据头
        head_idx = text.find(_META_SECTION)
        if 0 <= head_idx <= 80:
            block = text[head_idx:head_idx + _META_MAX_CHARS]
            if any(hint in block for hint in _META_HINTS):
                nxt = _NEXT_SECTION.search(block, len(_META_SECTION))
                if nxt:
                    end = head_idx + nxt.start()
                else:
                    # 没有下一节标记：只删「开头的连续元数据行」，遇到第一条正经正文就停
                    end = head_idx + len(_META_SECTION)
                    consumed = 0
                    for line in block.splitlines(keepends=True):
                        consumed += len(line)
                        stripped = line.strip()
                        if stripped.startswith(_META_SECTION) or any(h in line for h in _META_HINTS):
                            end = head_idx + consumed
                        else:
                            break
                text = text[:head_idx] + text[end:]

        # 2) 装饰性分隔线 + 节标记前多余的标题号
        text = _STRAY_HASH.sub(r"\1", text)
        text = _ISOLATED_RULE.sub("", text)

        # 3) 只有真的删过东西时，才整理因删除产生的多余空行
        if text != answer:
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    # 最终答案只处理这两类问题：缺免责声明、缺就医提醒。
    # 长度与语气之类的违规留给 worker 层，不在最终答案上改写用户看到的内容。
    _FINAL_ANSWER_FIXES = ("add_disclaimer", "add_emergency_warning")

    def _enforce_output_safety(self, answer: str, question: str) -> str:
        """给 Swarm 的最终答案补一道安全校验。

        worker 内部各自过了校验，但 Lead 会重写答案，可能把就医提醒和免责声明丢掉；
        这里在返回用户之前再过一遍，缺什么补什么。校验异常时原样返回，不影响主流程。
        """
        try:
            risk_level = "high" if detect_high_risk_signals(question) else "low"
            result = self.validator.validate_output(
                "consultation_agent", answer, risk_level=risk_level
            )
            fixable = [
                fix for fix in (result.get("auto_fixable") or [])
                if fix in self._FINAL_ANSWER_FIXES
            ]
            if not fixable:
                if not result.get("valid"):
                    logger.warning(
                        f"⚠️ Swarm 最终答案有非安全类违规（仅记录，不改写）: "
                        f"{result.get('violations')}"
                    )
                return answer
            logger.warning(f"⚠️ Swarm 最终答案缺安全要素，已自动补齐: {fixable}")
            return self.auto_fixer.fix_output(answer, fixable, risk_level=risk_level)
        except Exception as e:
            logger.error(f"Final answer safety check failed, keep original: {e}")
            return answer

    async def _process_with_swarm(
        self,
        question: str,
        context: Optional[Dict[str, Any]],
        assessment: Dict[str, Any],
        session_id: str,
        start_time: datetime,
        on_delta: Optional[Any] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        使用 Swarm 处理复杂问题

        这是群体智能的核心流程

        注意：context 已经包含了长短期记忆（在 process() 中注入）
        """
        # 可观测性：Span 记录器挂在请求上下文里，本方法自己取一次。
        # 注意不能直接引用 process() 里的同名局部变量——那是另一个作用域。
        spans = current_spans() or NOOP_SPANS

        # context 已经包含 recent_history 和 historical_cases
        # 无需重复检索

        # 创建 SharedContext（注入本轮问题与会话锚点，供 Worker 继承）
        shared_context = SharedContext(session_id=session_id)
        shared_context.set_data("user_question", question)
        shared_context.set_data("user_context", context or {})

        # 并发安全说明：不再将 SharedContext 附加到共享 Worker 实例
        # （即不再写入单例 worker 的 self.shared_context 字段）。
        # 改为在调用 worker.process_subtask 时以「依赖注入」方式传入，
        # 使每个请求的黑板随调用栈走，从根上消除多请求并发时的串台。
        # 参见下方 _execute_single_subtask 的传参。

        # 发布 Swarm 启动事件
        shared_context.publish_event(Event(
            type=EventType.SWARM_STARTED,
            source_agent="swarm_coordinator",
            data={
                "question": question,
                "num_subtasks": len(assessment.get("subtasks", []))
            }
        ))

        # Step 1: LeadAgent 分解任务
        subtasks = self.lead_agent.create_subtasks(assessment, shared_context)
        logger.info(f"Created {len(subtasks)} subtasks")

        # Step 2: Worker 执行分配的任务（并行）
        t_pool = time.perf_counter()
        tasks = []
        for worker in self.worker_pool:
            task = asyncio.create_task(
                self._worker_execute_assigned_tasks(worker, shared_context)
            )
            tasks.append(task)

        # 等待所有 Worker 完成（或超时）
        timeout_occurred = False
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=55.0
            )
        except asyncio.TimeoutError:
            timeout_occurred = True
            logger.warning("Swarm execution timeout (55s)")
            # 记录哪些 Agent 已完成，哪些未完成
            completed_agents = list(shared_context.agent_contributions.keys())
            claimed_tasks = [
                (subtask.assigned_to, subtask.type)
                for subtask in shared_context.task_decomposition.values()
                if subtask.status.value == "claimed"
            ]
            logger.info(f"Completed agents: {completed_agents}")
            logger.info(f"Timed out tasks: {claimed_tasks}")

        # 瀑布流第二层：Worker 池整体（子节点 worker_xxx 由下面按各 Worker 耗时补挂）
        # 注意区分两个数：池里有几个 Worker，和这次真的有几个产出了结果
        spans.mark(
            WORKER_POOL_SPAN,
            parent=ROOT_SPAN,
            duration_ms=round((time.perf_counter() - t_pool) * 1000, 1),
            detail=f"池内 {len(tasks)} 个 Worker，{len(shared_context.agent_contributions)} 个产出结果"
            + ("（55s 超时中断）" if timeout_occurred else ""),
        )

        # Step 3: LeadAgent 汇总结果
        # 即使超时，也尝试汇总已完成的部分结果
        t_synth = time.perf_counter()
        # 推给前端的是 Lead 汇总这一层——worker 的中间结果是过程，不是答案
        synth_stats = StreamStats() if (on_delta is not None or stream) else None
        final_answer = await self.lead_agent.synthesize_results(
            question=question,
            shared_context=shared_context,
            timeout_occurred=timeout_occurred,
            context=context,
            on_delta=on_delta,
            stream=stream,
            stream_stats=synth_stats,
        )
        answer_ttft_ms = synth_stats.ttft_ms if synth_stats else None
        # 安全网：Lead 会重写 worker 的结论，可能丢掉就医提醒或免责声明
        final_answer = self._enforce_output_safety(final_answer, question)
        # 瀑布流第三层：Lead 汇总（含出口安检）
        spans.mark(
            SYNTH_SPAN,
            parent=ROOT_SPAN,
            duration_ms=round((time.perf_counter() - t_synth) * 1000, 1),
            detail="Lead 汇总" + ("（超时后汇总部分结果）" if timeout_occurred else ""),
        )

        end_time = datetime.now()

        # Step 4: 生成 SessionSummary
        try:
            summary = SessionSummary.from_shared_context(
                session_id=session_id,
                question=question,
                shared_context=shared_context,
                final_answer=final_answer,
                start_time=start_time,
                end_time=end_time
            )
            self.session_manager.save_summary(summary)
        except Exception as e:
            logger.error(f"Failed to generate session summary: {e}")

        # 注意：短期记忆已经在 Agent Loop 中保存了，这里不需要重复保存
        # Agent Loop 保存了完整的对话历史（user + assistant + tool messages）

        # Swarm Worker 不写短期记忆；此处补记本轮问答，供后续单 Agent 加载历史
        try:
            self.short_term_memory.add_message(session_id, "user", question)
            self.short_term_memory.add_message(
                session_id, "assistant", (final_answer or "")[:3000]
            )
        except Exception as e:
            logger.warning(f"Failed to record swarm turn to short-term: {e}")

        # 保存到 Mem0 长期记忆
        try:
            # 记忆空间键：enhanced_context 已含 user_id；缺省回落全局共享（demo/eval 兼容）
            mem_user_id = (context or {}).get("user_id") or "medix_user"

            # 保存会话总结
            self.long_term_memory.add_session_summary(
                session_id=session_id,
                question=question,
                answer=final_answer,
                user_id=mem_user_id,
                metadata={
                    "mode": "swarm",
                    "agents_count": len(shared_context.agent_contributions),
                    "total_time": (end_time - start_time).total_seconds(),
                    "timeout_occurred": timeout_occurred
                }
            )

            logger.info(
                f"Saved to Mem0 long-term memory (session={session_id}, user={mem_user_id})"
            )

        except Exception as e:
            logger.error(f"Failed to save to Mem0: {e}")

        # 提取并更新患者档案（长期确定性事实）
        await self._update_patient_facts(
            (context or {}).get("user_id") or session_id, session_id
        )

        # 发布 Swarm 完成事件
        shared_context.publish_event(Event(
            type=EventType.SWARM_COMPLETED,
            source_agent="swarm_coordinator",
            data={
                "duration": (end_time - start_time).total_seconds(),
                "agents_count": len(shared_context.agent_contributions)
            }
        ))

        # 返回结果
        completed_agents = list(shared_context.agent_contributions.keys())
        # 观测：各 Worker 执行耗时（来自子任务时间戳）
        worker_timings: Dict[str, Any] = {}
        for subtask in shared_context.task_decomposition.values():
            if subtask.started_at and subtask.completed_at:
                owner = getattr(subtask, "assigned_agent", None) or "unknown"
                ms = (subtask.completed_at - subtask.started_at).total_seconds() * 1000
                worker_timings[f"agent_{owner}_ms"] = max(
                    worker_timings.get(f"agent_{owner}_ms", 0.0), ms
                )
        # 瀑布流：把每个 Worker 挂到 Worker 池之下；它们的 llm_call_* / skill_* 子节点
        # 由 AgentLoop 在运行时按同名父节点挂好（两边都用 worker_<agent_id> 命名）
        for key, ms in worker_timings.items():
            owner = key[len("agent_"):-len("_ms")]
            spans.mark(
                f"worker_{owner}",
                parent=WORKER_POOL_SPAN,
                duration_ms=ms,
                detail=f"Worker {owner}（子任务耗时）",
            )
        result = {
            'answer': final_answer,
            'swarm_enabled': True,
            'session_id': session_id,
            'agents_involved': completed_agents,
            'subtasks_completed': len(shared_context.get_all_completed_subtasks()),
            'total_time': (end_time - start_time).total_seconds(),
            # 流式指标：answer_ttft_ms 是 Lead 汇总这次生成的首 token 时间
            'streamed': bool(on_delta is not None or stream),
            'answer_ttft_ms': answer_ttft_ms,
            'swarm_metadata': shared_context.get_summary(),
            'timeout_occurred': timeout_occurred,
            'timings': {
                **worker_timings,
                "synthesize_ms": round((time.perf_counter() - t_synth) * 1000, 1),
                "swarm_total_ms": round((end_time - start_time).total_seconds() * 1000, 1),
            },
        }

        result['suggestions'] = self._extract_suggestions(final_answer)

        if timeout_occurred and not completed_agents:
            fallback = "由于系统超时，未能提供完整分析。建议简化问题重试，或在紧急情况下立即就医。"
        elif timeout_occurred:
            fallback = f"以上分析基于 {len(completed_agents)} 个 Agent 的部分协作结果（部分分析模块超时未完成），仅供参考，不能替代医生诊断。"
        else:
            fallback = "以上分析基于多个专业 Agent 的协作，仅供参考，不能替代医生诊断。"
        result['disclaimer'] = self._resolve_disclaimer(
            final_answer, fallback, timeout_occurred=timeout_occurred
        )

        return result

    async def _worker_execute_assigned_tasks(
        self,
        worker: Any,
        shared_context: SharedContext
    ):
        """
        Worker 执行分配给它的任务

        简化后的流程：
        - 查找分配给自己的任务
        - 执行任务
        - 记录结果
        """
        try:
            # 获取分配给该 Agent 的任务
            assigned_tasks = shared_context.get_subtasks_for_agent(worker.agent_id)

            if not assigned_tasks:
                logger.debug(f"{worker.agent_id}: No assigned tasks")
                return

            # 并行执行所有分配的任务
            tasks = []
            for subtask in assigned_tasks:
                logger.info(f"{worker.agent_id}: Starting {subtask.type}")
                shared_context.start_subtask(subtask.id)

                task = asyncio.create_task(
                    self._execute_single_subtask(worker, subtask, shared_context)
                )
                tasks.append(task)

            # 等待所有任务完成
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"{worker.agent_id}: Error processing subtask: {e}")

    async def _execute_single_subtask(self, worker, subtask, shared_context):
        """执行单个子任务"""
        try:
            result = await worker.process_subtask(subtask, shared_context)
            shared_context.complete_subtask(subtask.id, worker.agent_id, result)
            logger.info(f"{worker.agent_id}: Completed {subtask.type}")
        except Exception as e:
            logger.error(f"{worker.agent_id}: Error in {subtask.type}: {e}")

    def _extract_disclaimer_from_answer(self, final_answer: str) -> Optional[str]:
        if not final_answer:
            return None
        if "【免责声明】" in final_answer:
            start = final_answer.find("【免责声明】")
            rest = final_answer[start + len("【免责声明】"):].strip()
            end = rest.find("【")
            text = (rest[:end] if end != -1 else rest).strip()
            if text:
                return text
        if "仅供参考" in final_answer or "不能替代" in final_answer:
            return ""
        return None

    def _resolve_disclaimer(
        self,
        final_answer: str,
        fallback: Optional[str],
        timeout_occurred: bool = False
    ) -> str:
        """答案已有免责则提取/复用，不另造第二条（超时无完成除外可用 fallback）"""
        extracted = self._extract_disclaimer_from_answer(final_answer)
        if extracted is not None:
            return extracted
        return fallback or "⚠️ 以上信息仅供参考，不能替代专业医生的诊断和治疗。如有疑虑，请及时就医。"

    def _extract_suggestions(self, final_answer: str) -> List[str]:
        """从最终答案中提取建议（简化实现）"""
        suggestions = []

        # 简单的文本匹配
        if "【核心建议】" in final_answer:
            # 提取核心建议部分
            start_idx = final_answer.find("【核心建议】")
            end_idx = final_answer.find("【", start_idx + 1)
            if end_idx == -1:
                end_idx = len(final_answer)

            suggestions_text = final_answer[start_idx:end_idx]

            # 提取编号列表
            import re
            matches = re.findall(r'\d+\.\s*([^\n]+)', suggestions_text)
            suggestions = matches[:5]  # 最多5条

        return suggestions or ["请遵循医嘱，注意休息和营养"]

_COORDINATOR_CACHE: Dict[bool, "SwarmCoordinator"] = {}


def get_shared_coordinator(enable_swarm: bool = True) -> SwarmCoordinator:
    if enable_swarm not in _COORDINATOR_CACHE:
        _COORDINATOR_CACHE[enable_swarm] = SwarmCoordinator(enable_swarm=enable_swarm)
    return _COORDINATOR_CACHE[enable_swarm]


async def process_with_swarm(
    question: str,
    context: Optional[Dict[str, Any]] = None,
    enable_swarm: bool = True,
    session_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    on_delta: Optional[Any] = None,
    stream: bool = False
) -> Dict[str, Any]:
    """
    便捷函数：使用 Swarm 处理问题

    Args:
        question: 用户问题
        context: 额外上下文
        enable_swarm: 是否启用 Swarm（False 则总是用单 Agent）
        session_id: 会话ID（如果提供，将使用该ID而不是生成新的）
        trace_id: 链路追踪 ID（不传则自动生成，贯穿整条调用链）

    Returns:
        处理结果（含 trace_id 与 timings）
    """
    trace_id = trace_id or uuid.uuid4().hex[:12]
    coordinator = get_shared_coordinator(enable_swarm=enable_swarm)
    # loguru.contextualize：本请求内所有日志自动携带 trace_id。
    # Python 3.12+ 的 asyncio.run_in_executor 会把 contextvars 传播进线程池，
    # 因此连 Skill 线程里的日志也能带上 trace_id，实现端到端贯穿。
    with logger.contextualize(trace_id=trace_id, session_id=session_id or "-"):
        logger.info(
            f"trace start: enable_swarm={enable_swarm} "
            f"question_len={len(question) if question else 0}"
        )
        # 成本归因 / 瀑布流：为本请求启用账本与 Span 记录。
        # 必须在任何 create_task / 线程池派发之前设置——子上下文会复制这个 ContextVar。
        ledger, span_recorder = begin_observability()

        t0 = time.perf_counter()
        # 根 Span：整条请求（含记忆读写、路由、出口安检等所有开销）
        span_recorder.mark(ROOT_SPAN, parent=None, duration_ms=0.0, detail="整条请求")
        result = await coordinator.process(
            question,
            context,
            session_id=session_id,
            on_delta=on_delta,
            stream=stream,
        )
        result["trace_id"] = trace_id
        result["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # 根 Span 的真实耗时 = 整条请求耗时（覆盖记忆读写等未单独打点的部分）
        usage_and_cost = ledger.snapshot()
        spans = span_recorder.snapshot()
        for s in spans:
            if s["name"] == ROOT_SPAN and s.get("parent") in (None, ""):
                s["duration_ms"] = result["total_ms"]
                break

        result["usage_and_cost"] = usage_and_cost
        result["spans"] = spans
        # 需要时把瀑布流直接打进日志（排查线上长尾请求很方便）：
        #   MEDIX_PRINT_WATERFALL=1 uvicorn api.app:app
        if os.getenv("MEDIX_PRINT_WATERFALL") == "1":
            logger.info("\n" + format_span_waterfall(spans, title=f"耗时瀑布流 trace={trace_id}"))
        logger.info(
            f"trace end: total_ms={result.get('total_ms')} "
            f"swarm_enabled={result.get('swarm_enabled')} "
            f"agents={result.get('agents_involved')} "
            f"cost_cny={usage_and_cost.get('estimated_cost_cny')} "
            f"tokens={usage_and_cost.get('total_tokens')} "
            f"timings={result.get('timings')}"
        )
    return result
