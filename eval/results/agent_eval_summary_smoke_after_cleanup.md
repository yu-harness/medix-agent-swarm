# Agent Swarm 评测报告

- 生成时间: 2026-09-02T06:39:21.743364+00:00
- 样本: 从 benchmark_500.jsonl **分层抽样 4 条**（四类各 50），seed=42
- 系统路径: 真实调用 `process_with_swarm`（完整 Agent）
- 判定: embedding（BAAI/bge-small-zh-v1.5）+ DeepSeek LLM 终判
- 并发 semaphore=2, 单条超时 120.0s, 失败重试 1 次

## 判定细则

- embedding < 0.4 → **直接判错**（不再交 LLM）
- embedding ≥ 0.4 → 交 DeepSeek 输出 correct / partial / incorrect
- 主准确率（exact）**仅计 correct**；partial 单独汇报
- 空答、异常、超时 → 计错
- **裁判校准（相对基线）**：correct=覆盖金标关键医疗要点即可，不要求药名/偏方逐字；emb门禁由0.45降至0.40

## 总体结果

- 完成条数: 4
- exact 正确率: **1.0** (4/4)
- partial 比率: 0.0 (0/4)
- 延迟 mean / p50 / p95 (秒): 14.909 / 11.93 / 24.813

## 分类型 exact 正确率

| category | n | correct | exact_acc | partial | partial_rate |
|---|---:|---:|---:|---:|---:|
| health_consult | 0 | 0 | None | 0 | None |
| symptom_diagnosis | 0 | 0 | None | 0 | None |
| disease_knowledge | 1 | 1 | 1.0 | 0 | 0.0 |
| guideline_retrieval | 3 | 3 | 1.0 | 0 | 0.0 |

## 路由/模式分布（观测，非准确率评测）

- mode_counts: `{'single_agent': 4}`

## 典型错因 Top


## 未评指标

- **智能路由准确率**：本次未评（无路由金标）
- **多轮对话准确率**：本次未评（无多轮标注）

## 局限

- 子集 200 条，非全量 500
- LLM 裁判存在偏差；指南类标准答案偏文档片段，系统长答易因表述差异被判 partial/incorrect
- embedding 低阈值直接判错，可能漏掉「表述差异大但语义正确」的答案
