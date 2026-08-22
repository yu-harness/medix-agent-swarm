"""
配置文件示例

使用前请复制为 config.py，并填入 LLM_CONFIG 中的 api_key。

支持的 API 提供商：
1. OpenAI: https://api.openai.com/v1
2. 字节跳动火山引擎: https://ark.cn-beijing.volces.com/api/v3
3. DeepSeek: https://api.deepseek.com/v1
4. 阿里通义千问: https://dashscope.aliyuncs.com/compatible-mode/v1
5. 智谱清言: https://open.bigmodel.cn/api/paas/v4
"""

LLM_CONFIG = {
    "api_key": "sk-your-api-key-here",
    "model_name": "deepseek-chat",
    "base_url": "https://api.deepseek.com/v1",
    "temperature": 0.7,
    "max_tokens": 8192,
}

MEM0_CONFIG = {
    "api_key": "m0-your-mem0-api-key-here"
}
