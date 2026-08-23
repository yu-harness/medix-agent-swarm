# 架构

## 请求链路

```
用户输入 (main.py / process_with_swarm)
        ↓
SwarmCoordinator
        ↓
LeadAgent 分解子任务（LLM）
        ↓
len(subtasks)==1 或 swarm 关闭 → 单 Worker Agent
len(subtasks)>=2 且 enable_swarm → SharedContext 发布任务
        → Worker 自主认领、并行 Agent Loop
        → LeadAgent 汇总
        ↓
短期记忆写入；会话结束可摘要进 Mem0
```

Coordinator **只按子任务数量**切 `single` / `swarm`，不二次改 Agent 分配。

## Skills 与 Agents

Skills 自 `.agents/skills/`（及 `.claude/skills/` 镜像）加载，经 `SkillRegistry` 转成 OpenAI function calling；Agent Loop：Think → 调 Skill → Observe。

| Agent | 常见用途 |
|-------|----------|
| ConsultationAgent | 科普、生活/饮食建议、初步风险 |
| DiagnosticAgent | 多症状鉴别、ICD、风险分级 |
| ResearchAgent | 指南/循证、深度检索 |

各 Agent 可注册多 Skill；运行时白名单见 `constraints/agent_constraints.yaml`。

## 记忆与知识库

| 层 | 实现 | 作用 |
|----|------|------|
| 短期 | `memory/short_term.py`（内存/可选 Redis） | 多轮上下文 |
| 长期 | Mem0（`MEM0_CONFIG`）；无 key 则降级 | 跨会话相似案例 |
| 熵 | `entropy_manager` | 去重、压缩历史 |
| 知识 | Milvus Lite + `bge-small-zh-v1.5` | 语义检索；数据在 `knowledge/data/documents/` |

导入：`python knowledge/scripts/import_hardcoded_data.py`（追加文档用 `import_append_docs.py`）。

## 约束 warn → enforce

见 [constraints/README.md](../constraints/README.md)。默认 warn；`CONSTRAINT_ENFORCE=1` 硬拦越权 Skill。输出侧：`validation/auto_fixer` 可补免责声明/高危提醒。

## 后续

质量评测与延迟/路由待办见 [ROADMAP.md](ROADMAP.md)。
