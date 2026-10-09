FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

# apt 换清华源（deb.debian.org 直连常 502/超时）
RUN sed -i 's|http://deb.debian.org|http://mirrors.tuna.tsinghua.edu.cn|g; s|http://security.debian.org|http://mirrors.tuna.tsinghua.edu.cn|g' \
        /etc/apt/sources.list.d/debian.sources 2>/dev/null || true; \
    sed -i 's|http://deb.debian.org|http://mirrors.tuna.tsinghua.edu.cn|g; s|http://security.debian.org|http://mirrors.tuna.tsinghua.edu.cn|g' \
        /etc/apt/sources.list 2>/dev/null || true; \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# pip：先装 torch CPU 版（避免 Linux 默认拉 CUDA 全家桶），其余走清华 pypi
RUN pip install torch --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cpu && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# HF 模型与 Milvus Lite 体积大：建议挂 volume
#   - ./knowledge/data -> /app/knowledge/data
#   - HF cache -> $HF_HOME
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
