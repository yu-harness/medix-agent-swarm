"""
配置文件示例

使用前请复制为 config.py：`cp config.example.py config.py`
密钥优先读环境变量，其次读 .env（本文件会自动加载）。

注意：本示例必须与代码保持「导入兼容」——`core/llm_client.py` 会
`from config import LLM_CONFIG, ensure_api_keys`，`constraints/validator.py` 会
`from config import CONSTRAINT_ENFORCE`。示例缺少任何一个名字，CI 与全新克隆都会直接导入失败。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 自动加载项目根目录下的 .env（已被 .gitignore 忽略，不会提交）
_project_root = Path(__file__).resolve().parent
load_dotenv(_project_root / ".env")

LLM_CONFIG = {
    # 无默认密钥：缺失时由 ensure_api_keys() 在启动处 fail fast，
    # 避免带一个假的占位 key 静默跑到线上
    "api_key": os.getenv("LLM_API_KEY", "").strip(),
    "model_name": os.getenv("LLM_MODEL_NAME") or "deepseek-chat",
    "base_url": os.getenv("LLM_BASE_URL") or "https://api.deepseek.com/v1",
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "8192")),
}

# 约束是否硬拦（True=拦截越权 Skill；False=仅警告）。
# 默认 True：约束只警告不拦截等于没有约束。
# 需要评估影响面时可临时设环境变量 CONSTRAINT_ENFORCE=0 退回 warn 模式。
CONSTRAINT_ENFORCE = os.getenv("CONSTRAINT_ENFORCE", "1").strip().lower() not in (
    "0", "false", "off", "no"
)

MEM0_CONFIG = {
    "enabled": os.getenv("MEM0_ENABLED", "1").strip().lower() not in ("0", "false", "off"),
    "api_key": os.getenv("MEM0_API_KEY", "").strip(),
}


def ensure_api_keys() -> None:
    """启动校验：密钥缺失时给出明确指引（fail fast），避免带空 key 静默运行。"""
    missing = []
    if not LLM_CONFIG["api_key"]:
        missing.append("LLM_API_KEY")
    if MEM0_CONFIG["enabled"] and not MEM0_CONFIG["api_key"]:
        missing.append("MEM0_API_KEY")
    if missing:
        raise RuntimeError(
            "缺少密钥配置: " + ", ".join(missing)
            + "。请复制 .env.example 为 .env 并填入真实密钥（不需要长期记忆可设 MEM0_ENABLED=0）；"
            "禁止把真实密钥写进 config.py（会被 git 跟踪）。"
        )
