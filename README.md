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
constraints/  validation/  eval/  api/  .agents/skills/  main.py
```

MIT License
