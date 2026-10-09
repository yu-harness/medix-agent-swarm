# MediX Agent Swarm

Skills → Agent Loop →（可选）Agent Swarm 的文本医疗助手。Milvus 知识库 + 短/长期记忆 + 约束（warn→enforce）。

**仅供学习研究，不能替代就医。**

## 快速开始

```bash
conda create -n medix-swarm python=3.12 -y
conda activate medix-swarm
pip install -r requirements.txt
cp config.example.py config.py   # 或 cp .env.example .env 填密钥（env 优先）
python knowledge/scripts/import_hardcoded_data.py
python examples/test_all.py
python main.py
```

本地 API：

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

两个端点（都需要 `X-API-Key`）：

- `POST /v1/chat` —— 一次性返回完整答案
- `POST /v1/chat/stream` —— SSE 流式返回，事件为 `delta`（答案片段）、`done`（元信息）、`error`

```bash
curl -N -X POST http://localhost:8000/v1/chat/stream \
  -H "X-API-Key: $APP_API_KEY" -H "Content-Type: application/json" \
  -d '{"question": "高血压平时要注意什么"}'
```

`done` 事件里有两个首 token 时间：`client_ttft_ms`（从发出请求到第一个片段到达，即用户感知的响应时间）与 `answer_ttft_ms`（模型侧首个 token）。两者之差就是检索与编排等前置开销。

实测（`eval/scripts/run_ttft_eval.py`，20 条分层抽样，与延迟基线同源同抽样）：

| 路由 | 用户感知首字 p50 | 整段总时长 p50 |
|---|---|---|
| 单 Agent（17 条） | 2.7 s | 9.8 s |
| Swarm（3 条） | 9.7 s | 14.9 s |

单 Agent 路由流式后，用户在 2.7 秒就能看到内容，不必等整段 9.8 秒；Swarm 路由推的是 Lead 汇总那一层（worker 的中间结果是过程、不是答案），首字仍要等 worker 跑完——瓶颈在编排而非生成，Lead 汇总的模型侧首 token 只有 0.6 秒。

Docker：

```bash
cp .env.example .env   # 填密钥后改 docker-compose env_file 为 .env
docker compose up --build
```

配置见 `config.example.py` / `.env.example`（`LLM_*`、`MEM0_*`、`APP_API_KEY`）。`config.py` / `.env` 已 gitignore。

## 架构一页

```
用户问题 → SwarmCoordinator
  ├─ 简单 → 单 Agent + Agent Loop（调用 Skills）
  └─ 复杂 → LeadAgent 分解 → 多 Worker 并行 → Lead 汇总
Skills ↔ Milvus / 规则 / 网络检索
记忆：短期会话历史 + Mem0 长期；约束：YAML 白名单 warn→enforce
```

Skills（9）：`search_knowledge`、`recommend_lifestyle`、`assess_risk`、`analyze_symptoms`、`disease_code`、`clinical_guideline`、`deep_research`、`search_history`、`search_similar_cases`  
Agents：Consultation / Diagnostic / Research + LeadAgent / Coordinator

## 文档导航

| 文档 | 内容 |
|------|------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 请求链路、Skills/Agents、记忆与知识库、约束 |
| [docs/ROADMAP.md](docs/ROADMAP.md) | 质量已达线后的延迟/路由后续 |
| [eval/README.md](eval/README.md) | 评测集、跑评测、结果 |
| [eval/docs/quality_optimization_29_to_91.md](eval/docs/quality_optimization_29_to_91.md) | exact 29%→91% 复盘 |
| [eval/routing/README.md](eval/routing/README.md) | 路由金标标注 |
| [constraints/README.md](constraints/README.md) | 约束 warn→enforce |

## 目录（精简）

```
agents/  core/  swarm/  memory/  knowledge/  research/
constraints/  validation/  eval/  api/  .claude/skills/  main.py
```

MIT License
