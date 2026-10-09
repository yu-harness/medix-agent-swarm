# A/B 评测：Swarm vs Single Agent

> **自动 LLM 盲评，非医学专家盲评。面试须降级表述。**

- 样本 n=24（seed=42，每类 6，共 24）
- A 臂：`SwarmCoordinator(enable_swarm=True)`
- B 臂：`SwarmCoordinator(enable_swarm=False)`
- 两者使用同一知识库（64 条 chunk）与同一套工具，差异仅在于是否启用多 Agent 协作
- 盲评：每题随机决定呈现顺序，裁判不知哪个是 Swarm
- 耗时 325.0s

## 总体结果

| 结果 | 数量 | 占比 |
|---|---|---|
| Swarm 胜 | 11 | 45.8% |
| Single 胜 | 6 | 25.0% |
| 平局 | 7 | 29.2% |

## 三维均分（1-5）

| 维度 | Swarm | Single | 差值 |
|---|---|---|---|
| accuracy | 4.792 | 4.542 | 0.25 |
| completeness | 4.625 | 4.25 | 0.375 |
| safety | 4.833 | 4.542 | 0.291 |

## 延迟（秒）

| 指标 | Swarm | Single |
|---|---|---|
| p50 | 13.512 | 15.264 |
| p95 | 41.264 | 36.651 |
| p50 倍数 | 0.89x | 1x |

## 路由与超时

- A 臂（enable_swarm=True）中真正走 Swarm 的：2/24
- A 臂内部超时（55s）：0/24

### 按 Swarm 是否激活拆分（关键）

| 子集 | n | Swarm胜 | Single胜 | 平 | Swarm p50 | Single p50 |
|---|---|---|---|---|---|---|
| Swarm 已激活 | 2 | 2 | 0 | 0 | 33.311 | 15.605 |
| 未激活（回落单 Agent） | 22 | 9 | 6 | 7 | 12.724 | 15.264 |

## 分类明细

| 类别 | n | Swarm胜 | Single胜 | 平 | Swarm p50 | Single p50 |
|---|---|---|---|---|---|---|
| health_consult | 6 | 3 | 2 | 1 | 11.608 | 17.386 |
| symptom_diagnosis | 6 | 2 | 1 | 3 | 14.808 | 13.303 |
| disease_knowledge | 6 | 3 | 1 | 2 | 9.415 | 11.619 |
| guideline_retrieval | 6 | 3 | 2 | 1 | 24.408 | 32.733 |

## 结论口径（可直接用于面试）

- Swarm 相对单 Agent 的胜率：45.8%；三维均分差值 accuracy 4.792 vs 4.542、completeness 4.625 vs 4.25
- 代价：p50 延迟 13.512s vs 15.264s（0.89x）
- 样本量 24，自动裁判，结论为方向性参考，不做统计显著性断言
