# LLM 版 A/B 对比报告

> **重要：自动 LLM 对比，非医学专家盲评。** 面试须降级表述，不可称专家偏好。

- n: 32（seed=42, 每类 8）
- A: 完整 `process_with_swarm`
- B: 同模型直接 LLM（无 tools / 无 swarm）
- 裁判: 同家族 LLM，三维 1–5 + overall_better
- elapsed_sec: 80.4

## 总体

- **偏好 A**: **81.2%** (26/32)
- 偏好 B: 12.5% (4/32)
- tie: 6.2% (2/32)

## 均分

| 维度 | A | B |
|---|---:|---:|
| accuracy | 4.594 | 4.344 |
| completeness | 4.75 | 3.625 |
| safety | 4.75 | 4.375 |

## 分品类（偏好计数）

| category | n | A | B | tie |
|---|---:|---:|---:|---:|
| health_consult | 8 | 6 | 1 | 1 |
| symptom_diagnosis | 8 | 6 | 1 | 1 |
| disease_knowledge | 8 | 7 | 1 | 0 |
| guideline_retrieval | 8 | 7 | 1 | 0 |

## 局限

- 非医学专家、非双盲；裁判与被测模型同系，可能偏袒长答/结构答
- 仅 seed 分层子集，不能外推全量
