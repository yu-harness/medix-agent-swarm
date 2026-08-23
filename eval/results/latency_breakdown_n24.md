# 分阶段耗时评测

- 时间: 2026-08-23T09:18:47.933174+00:00
- n=24（seed=42，每类 6）
- 超时: 120.0s

## 方法

- 对 `SwarmCoordinator` 插桩：`assess_and_decompose`（路由）、各 Agent `process`、`synthesize_results`、`MedicalKnowledgeBase.search`
- `main_llm_approx` = `agent_process - retrieval`（同请求内；含非检索工具与多轮 LLM，非纯 chat）
- `other` ≈ 记忆检索/开销（未单独拆 Mem0）

## 局限

- 小样本；DeepSeek 限流会影响绝对值
- Swarm 并行时 agent_process 为各 worker 累加墙钟近似，可能略大于 wall
- 检索计时挂在 KB 单例上，并发评测勿并行跑本脚本

## 汇总（秒）

| 阶段 | mean | p50 | p95 |
|---|---:|---:|---:|
| total | 20.77 | 14.291 | 41.388 |
| route_decompose | 1.455 | 1.421 | 2.077 |
| agent_process | 12.82 | 10.702 | 37.474 |
| retrieval | 1.139 | 0.593 | 3.231 |
| main_llm_approx | 12.125 | 9.857 | 34.589 |
| synthesize | 1.803 | 0.0 | 11.127 |
| other | 4.692 | 1.319 | 21.398 |

## 分品类 total mean

| category | n | mean_total |
|---|---:|---:|
| health_consult | 6 | 19.453 |
| symptom_diagnosis | 6 | 24.409 |
| disease_knowledge | 6 | 13.468 |
| guideline_retrieval | 6 | 25.752 |

成功 24/24；失败见 detail jsonl
