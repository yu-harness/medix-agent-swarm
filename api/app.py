import os
import sys
import json
import time
import uuid
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env")

from swarm import get_shared_coordinator, process_with_swarm  # noqa: E402
from core.observability import (  # noqa: E402
    ROOT_SPAN,
    ROUTE_SPAN,
    SYNTH_SPAN,
    WORKER_POOL_SPAN,
    begin_observability,
)

# 单页问诊 Demo 的静态目录（无构建步骤，浏览器直开）
WEB_DIR = project_root / "web"


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "*").strip()
    if raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.coordinator = get_shared_coordinator(enable_swarm=True)
    yield


app = FastAPI(title="MediX Agent Swarm API", version="1.0.0", lifespan=lifespan)
_origins = _cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str = ""
    session_id: Optional[str] = None
    swarm_enabled: Optional[bool] = None
    agents_involved: Optional[list] = None
    suggestions: Optional[list] = None
    disclaimer: Optional[str] = None
    total_time: Optional[float] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


async def verify_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    expected = os.getenv("APP_API_KEY", "").strip()
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def index():
    """问诊 Web Demo 主页面：`uvicorn api.app:app` 后直接打开就能用。"""
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="web/index.html 不存在")
    return FileResponse(page)


# 页面本体是自包含单文件；挂载目录是为了以后加图标/截图等静态资源不用改代码
if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")


@app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat(body: ChatRequest):
    # 链路追踪：一次请求一个 trace_id，贯穿 coordinator → agent → loop → LLM/Skill 日志
    trace_id = uuid.uuid4().hex[:12]
    context = {"trace_id": trace_id}
    if body.user_id:
        context["user_id"] = body.user_id

    with logger.contextualize(trace_id=trace_id, session_id=body.session_id or "-"):
        logger.info(
            f"POST /v1/chat: question_len={len(body.question)} user_id={body.user_id}"
        )
        t0 = time.perf_counter()
        result = await process_with_swarm(
            body.question,
            context=context or None,
            session_id=body.session_id,
            trace_id=trace_id,
        )
        logger.info(
            f"POST /v1/chat done: api_ms={round((time.perf_counter() - t0) * 1000, 1)} "
            f"total_ms={result.get('total_ms')} "
            f"swarm={result.get('swarm_enabled')} "
            f"timings={result.get('timings')}"
        )
    known = {
        "answer",
        "session_id",
        "swarm_enabled",
        "agents_involved",
        "suggestions",
        "disclaimer",
        "total_time",
    }
    return ChatResponse(
        answer=result.get("answer") or "",
        session_id=result.get("session_id"),
        swarm_enabled=result.get("swarm_enabled"),
        agents_involved=result.get("agents_involved"),
        suggestions=result.get("suggestions"),
        disclaimer=result.get("disclaimer"),
        total_time=result.get("total_time"),
        extra={k: v for k, v in result.items() if k not in known},
    )


def _sse(event: str, data: Dict[str, Any]) -> str:
    """把一次事件编码成 SSE 帧。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# 阶段映射：(已完成的 Span, 前端阶段 key, 此刻正在做什么)。
# Span 是「阶段结束时」打点的，所以每个标签描述的是**最后一个已完成阶段之后**的动作；
# 顺序即优先级，取第一个命中的（越是下游的阶段越靠前）。
_STATUS_STAGES = (
    (SYNTH_SPAN, "synthesize", "Lead 综合汇总中"),
    (WORKER_POOL_SPAN, "synthesize_pending", "汇总中"),
    (ROUTE_SPAN, "workers", "多专家并行执行中"),
    (ROOT_SPAN, "route", "检索与任务分解中"),
)


def _status_from_spans(recorder, t0: float) -> Optional[Dict[str, Any]]:
    """把真实的 Span 记录翻译成前端要展示的协作状态（不是编造的进度条）。"""
    try:
        spans = recorder.snapshot()
    except Exception:
        return None

    names = {str(s.get("name") or "") for s in spans}
    phase, label = "received", "请求已接收"
    for span_name, key, text in _STATUS_STAGES:
        if span_name in names:
            phase, label = key, text
            break

    return {
        "phase": phase,
        "label": label,
        "agents": sorted(n.split("_", 1)[1] for n in names if n.startswith("worker_")),
        "skills": sorted(n.split("_", 1)[1] for n in names if n.startswith("skill_")),
        "llm_calls": sum(1 for n in names if n.startswith("llm_call_")),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


@app.post("/v1/chat/stream", dependencies=[Depends(verify_api_key)])
async def chat_stream(body: ChatRequest):
    """
    流式版 /v1/chat：用 SSE 逐段推送答案正文，结束时推一次元信息。

    事件类型：
    - status: 协作阶段（由真实 Span 记录推导，见 _status_from_spans），用于前端状态胶囊
    - delta:  {"text": "..."}  答案片段，边生成边推
    - done:   答案、TTFT、耗时、Token 与费用账单等元信息
    - error:  {"message": "..."}

    client_ttft_ms 是从收到请求到第一个片段到达客户端的时间，即用户感知的响应时间；
    answer_ttft_ms 是模型侧首个 token 的时间。两者之差就是检索与编排等前置开销。

    Swarm 路由下推送的是 Lead 汇总那一层（worker 的中间结果是过程，不是答案），
    因此它的 client_ttft_ms 天然包含各 worker 的执行时间。
    """
    trace_id = uuid.uuid4().hex[:12]
    context = {"trace_id": trace_id}
    if body.user_id:
        context["user_id"] = body.user_id

    # 可观测性：必须在 create_task 之前建账本 —— 子任务会复制这份上下文，
    # 于是下游（coordinator → worker → agent_loop）写进的是同一个 Span 记录器，
    # 这里就能在流式过程中实时读出真实阶段（_status_from_spans）。
    _, spans = begin_observability()

    t0 = time.perf_counter()
    first_delta_ms: Optional[float] = None
    queue: "asyncio.Queue" = asyncio.Queue()

    async def on_delta(text: str) -> None:
        nonlocal first_delta_ms
        if first_delta_ms is None:
            first_delta_ms = round((time.perf_counter() - t0) * 1000, 1)
        await queue.put(("delta", text))

    async def produce() -> None:
        try:
            with logger.contextualize(trace_id=trace_id, session_id=body.session_id or "-"):
                logger.info(f"POST /v1/chat/stream: question_len={len(body.question)}")
                result = await process_with_swarm(
                    body.question,
                    context=context or None,
                    session_id=body.session_id,
                    trace_id=trace_id,
                    on_delta=on_delta,
                    stream=True,
                )
                logger.info(
                    f"POST /v1/chat/stream done: client_ttft_ms={first_delta_ms} "
                    f"answer_ttft_ms={result.get('answer_ttft_ms')} "
                    f"total_ms={result.get('total_ms')} "
                    f"swarm={result.get('swarm_enabled')}"
                )
            await queue.put(("done", result))
        except Exception as e:
            logger.exception(f"POST /v1/chat/stream failed: {e}")
            await queue.put(("error", f"{type(e).__name__}: {e}"))
        finally:
            await queue.put(None)

    producer = asyncio.create_task(produce())

    async def event_source():
        last_phase: Optional[str] = None
        getter: Optional["asyncio.Task"] = None
        try:
            # 先推一帧初始状态，前端一收到请求就能点亮第一颗胶囊
            initial = _status_from_spans(spans, t0)
            if initial:
                last_phase = initial["phase"]
                yield _sse("status", initial)

            while True:
                # 复用同一个 getter task，避免 wait_for 取消时丢掉恰好到达的片段
                if getter is None:
                    getter = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait({getter}, timeout=0.4)
                if not done:
                    # 片段还没来：顺手报一次真实阶段变化
                    st = _status_from_spans(spans, t0)
                    if st and st["phase"] != last_phase:
                        last_phase = st["phase"]
                        yield _sse("status", st)
                    continue

                item = getter.result()
                getter = None
                if item is None:
                    break
                kind, payload = item
                if kind == "delta":
                    yield _sse("delta", {"text": payload})
                elif kind == "error":
                    yield _sse("error", {"message": payload})
                else:
                    yield _sse("done", {
                        "answer": payload.get("answer") or "",
                        "session_id": payload.get("session_id"),
                        "swarm_enabled": payload.get("swarm_enabled"),
                        "agents_involved": payload.get("agents_involved"),
                        "suggestions": payload.get("suggestions"),
                        "disclaimer": payload.get("disclaimer"),
                        "total_time": payload.get("total_time"),
                        "total_ms": payload.get("total_ms"),
                        "trace_id": payload.get("trace_id"),
                        "streamed": payload.get("streamed"),
                        "client_ttft_ms": first_delta_ms,
                        "answer_ttft_ms": payload.get("answer_ttft_ms"),
                        "timings": payload.get("timings"),
                        # 成本归因：本次请求的 Token 与费用账单（按阶段明细）
                        "usage_and_cost": payload.get("usage_and_cost"),
                    })
        finally:
            if not producer.done():
                producer.cancel()

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 关掉反向代理缓冲，否则片段会被攒着一起发，流式就白做了
            "X-Accel-Buffering": "no",
        },
    )
