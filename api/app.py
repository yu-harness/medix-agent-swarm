import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env")

from swarm import get_shared_coordinator, process_with_swarm  # noqa: E402


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


@app.post("/v1/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
async def chat(body: ChatRequest):
    context = {}
    if body.user_id:
        context["user_id"] = body.user_id
    result = await process_with_swarm(
        body.question,
        context=context or None,
        session_id=body.session_id,
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
