"""
配置文件示例

使用前请复制为 config.py。密钥优先读环境变量，其次用下方默认值。
也可直接用 .env（见 .env.example）。
"""
import os

LLM_CONFIG = {
    "api_key": os.getenv("LLM_API_KEY") or "sk-your-api-key-here",
    "model_name": os.getenv("LLM_MODEL_NAME") or "deepseek-chat",
    "base_url": os.getenv("LLM_BASE_URL") or "https://api.deepseek.com/v1",
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "8192")),
}

CONSTRAINT_ENFORCE = False

MEM0_CONFIG = {
    "enabled": os.getenv("MEM0_ENABLED", "1").strip().lower() not in ("0", "false", "off"),
    "api_key": os.getenv("MEM0_API_KEY") or "m0-your-mem0-api-key-here",
}
