# Agent Swarm 评测报告

- 生成时间: 2026-07-27T10:56:51.803482+00:00
- 样本: 从 benchmark_500.jsonl **分层抽样 200 条**（四类各 50），seed=42
- 系统路径: 完整 Agent（`SwarmCoordinator.process`，逻辑等同 `process_with_swarm`；评测中复用单例 Coordinator）
- 判定: embedding（BAAI/bge-small-zh-v1.5）+ DeepSeek LLM 终判
- 并发 semaphore=1, 单条超时 120.0s, 失败重试 1 次

## 判定细则

- embedding < 0.45 → **直接判错**（不再交 LLM）
- embedding ≥ 0.45 → 交 DeepSeek 输出 correct / partial / incorrect
- 主准确率（exact）**仅计 correct**；partial 单独汇报
- 空答、异常、超时 → 计错

## 总体结果

- 完成条数: 200
- exact 正确率: **29.0%** (58/200)
- partial 比率: **55.5%** (111/200)
- correct+partial 覆盖率: **84.5%** (169/200)（非正式主指标，仅供参考）
- 延迟 mean / p50 / p95 (秒): **32.5 / 22.6 / 76.7**

## 分类型 exact 正确率

| category | n | correct | exact_acc | partial | partial_rate |
|---|---:|---:|---:|---:|---:|
| health_consult | 50 | 7 | 14% | 36 | 72% |
| symptom_diagnosis | 50 | 8 | 16% | 27 | 54% |
| disease_knowledge | 50 | 16 | 32% | 27 | 54% |
| guideline_retrieval | 50 | 27 | 54% | 21 | 42% |

## 路由/模式分布（观测，非准确率评测）

- mode_counts: single_agent=131 (65.5%), swarm=69 (34.5%)

## 典型错因

1. **关键点遗漏（主因）**：系统给通用/安全建议，但未覆盖标准答案中的具体药物名、检查项、数值阈值 → 大量 partial
2. **社区 QA 金标过细**：健康咨询/症状类金标常含具体药名或偏方细节，与 Agent 保守表述不对齐
3. **指南数值细节不一致**：如血压目标分层 vs 金标固定 `<130/80`、用药方案 ACEI/ARB+利尿剂等
4. **embedding 低相似直接判错**：5 条 emb&lt;0.45 未交 LLM
5. **表述覆盖但裁判偏严**：emb 较高仍判 partial/incorrect（LLM 裁判偏差）

## 未评指标

- **智能路由准确率**：本次未评（无路由金标）
- **多轮对话准确率**：本次未评（无多轮标注）
- **纯知识库检索准确率**：本次为端到端回答评测，未单独评检索召回

## 局限

- 子集 200 条，非全量 500
- LLM 裁判存在偏差；指南类标准答案偏文档片段，系统长答易因表述差异被判 partial/incorrect
- embedding 低阈值直接判错，可能漏掉「表述差异大但语义正确」的答案
- 主指标 exact 偏严；若业务接受 partial，覆盖率会显著更高
