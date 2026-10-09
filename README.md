# MediX Agent Swarm

[![Evaluation CI](https://github.com/yu-harness/medix-agent-swarm/actions/workflows/eval.yml/badge.svg)](https://github.com/yu-harness/medix-agent-swarm/actions/workflows/eval.yml)

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
- `POST /v1/chat/stream` —— SSE 流式返回，事件为 `status`（协作阶段，由后端真实 Span 推导）、`delta`（答案片段）、`done`（元信息，含 Token 与费用账单）、`error`

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

## Web Demo 快速体验

一条命令起服务，浏览器直接打开就能问诊（页面是单文件、无构建步骤）：

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
# 然后打开 http://127.0.0.1:8000/
```

三个值得点开看的交互：

- **真实 Span 驱动的协作胶囊**：`任务分解中 → 多专家并行执行中 → 综合汇总中` 的胶囊由**后端真实 Span** 推导并通过 SSE `status` 事件推送，不是前端假动画——胶囊的节奏就是后端真实的运行时节奏；
- **高危急症自动报警**：回答里出现「立即就医 / 拨打 120 / 急诊」等安全要素时，气泡顶部自动亮起红色急救横幅（由出口安全网保证这些要素不会被汇总环节吃掉）；
- **精确指南页码的折叠抽屉**：把回答里的 `【资料 i｜来源：中国高血压防治指南（2024年修订版） p.15｜类型：临床指南】` 解析成可展开的来源卡片，点开看原文片段与页码。

另外每个回答右下角有一条**账单小尾巴**（`TTFT 12.9s · 19.7k tokens · ¥0.0229`），点开可按 `lead_decompose` / `worker_*` / `lead_synthesize` 看分阶段明细。

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
constraints/  validation/  eval/  api/  web/  .claude/skills/  main.py
```

MIT License
