FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# HF 模型与 Milvus Lite 体积大：建议挂 volume
#   - ./knowledge/data -> /app/knowledge/data
#   - HF cache -> $HF_HOME
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
