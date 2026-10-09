# A/B 评测：Swarm vs Single Agent

> **自动 LLM 盲评，非医学专家盲评。面试须降级表述。**

- 样本 n=12（seed=42，每类 3，共 12）
- A 臂：`SwarmCoordinator(enable_swarm=True)`
- B 臂：`SwarmCoordinator(enable_swarm=False)`
- 两者使用同一知识库（64 条 chunk）与同一套工具，差异仅在于是否启用多 Agent 协作
- 盲评：每题随机决定呈现顺序，裁判不知哪个是 Swarm
- 耗时 187.4s

## 总体结果

| 结果 | 数量 | 占比 |
|---|---|---|
| Swarm 胜 | 6 | 50.0% |
| Single 胜 | 1 | 8.3% |
| 平局 | 5 | 41.7% |

## 三维均分（1-5）

| 维度 | Swarm | Single | 差值 |
|---|---|---|---|
| accuracy | 4.917 | 4.917 | 0.0 |
| completeness | 4.917 | 4.583 | 0.334 |
| safety | 4.917 | 4.917 | 0.0 |

## 延迟（秒）

| 指标 | Swarm | Single |
|---|---|---|
| p50 | 23.762 | 12.602 |
| p95 | 35.161 | 26.252 |
| p50 倍数 | 1.89x | 1x |

## 路由与超时

- A 臂（enable_swarm=True）中真正走 Swarm 的：5/12
- A 臂内部超时（55s）：0/12

### 按 Swarm 是否激活拆分（关键）

| 子集 | n | Swarm胜 | Single胜 | 平 | Swarm p50 | Single p50 |
|---|---|---|---|---|---|---|
| Swarm 已激活 | 5 | 5 | 0 | 0 | 30.145 | 12.726 |
| 未激活（回落单 Agent） | 7 | 1 | 1 | 5 | 16.979 | 12.477 |

## 分类明细

| 类别 | n | Swarm胜 | Single胜 | 平 | Swarm p50 | Single p50 |
|---|---|---|---|---|---|---|
| health_consult | 3 | 2 | 0 | 1 | 30.145 | 12.79 |
| symptom_diagnosis | 3 | 3 | 0 | 0 | 26.589 | 12.726 |
| disease_knowledge | 3 | 1 | 0 | 2 | 11.118 | 12.064 |
| guideline_retrieval | 3 | 0 | 1 | 2 | 23.538 | 12.477 |

## 结论口径（可直接用于面试）

- Swarm 相对单 Agent 的胜率：50.0%；三维均分差值 accuracy 4.917 vs 4.917、completeness 4.917 vs 4.583
- 代价：p50 延迟 23.762s vs 12.602s（1.89x）
- 样本量 12，自动裁判，结论为方向性参考，不做统计显著性断言
