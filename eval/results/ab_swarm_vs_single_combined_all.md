# A/B 评测：Swarm vs Single Agent

> **自动 LLM 盲评，非医学专家盲评。面试须降级表述。**

- 样本 n=65（seed=None，每类 None，共 None）
- A 臂：`SwarmCoordinator(enable_swarm=True)`
- B 臂：`SwarmCoordinator(enable_swarm=False)`
- 两者使用同一知识库（见各轮明细 条 chunk）与同一套工具，差异仅在于是否启用多 Agent 协作
- 盲评：每题随机决定呈现顺序，裁判不知哪个是 Swarm
- 耗时 Nones

## 总体结果

| 结果 | 数量 | 占比 |
|---|---|---|
| Swarm 胜 | 29 | 44.6% |
| Single 胜 | 11 | 16.9% |
| 平局 | 25 | 38.5% |

## 三维均分（1-5）

| 维度 | Swarm | Single | 差值 |
|---|---|---|---|
| accuracy | 4.769 | 4.662 | 0.107 |
| completeness | 4.785 | 4.446 | 0.339 |
| safety | 4.877 | 4.738 | 0.139 |

## 延迟（秒）

| 指标 | Swarm | Single |
|---|---|---|
| p50 | 23.538 | 13.971 |
| p95 | 50.629 | 39.083 |
| p50 倍数 | 1.68x | 1x |

## 路由与超时

- A 臂（enable_swarm=True）中真正走 Swarm 的：21/65
- A 臂内部超时（55s）：1/65

### 按 Swarm 是否激活拆分（关键）

| 子集 | n | Swarm胜 | Single胜 | 平 | Swarm p50 | Single p50 |
|---|---|---|---|---|---|---|
| Swarm 已激活 | 21 | 17 | 2 | 2 | 31.254 | 13.422 |
| 未激活（回落单 Agent） | 44 | 12 | 9 | 23 | 15.969 | 14.674 |

## 分类明细

| 类别 | n | Swarm胜 | Single胜 | 平 | Swarm p50 | Single p50 |
|---|---|---|---|---|---|---|
| health_consult | 17 | 8 | 3 | 6 | 22.356 | 16.09 |
| symptom_diagnosis | 17 | 9 | 1 | 7 | 24.27 | 12.726 |
| disease_knowledge | 14 | 7 | 2 | 5 | 12.724 | 12.726 |
| guideline_retrieval | 17 | 5 | 5 | 7 | 28.901 | 22.742 |

## 结论口径（可直接用于面试）

- Swarm 相对单 Agent 的胜率：44.6%；三维均分差值 accuracy 4.769 vs 4.662、completeness 4.785 vs 4.446
- 代价：p50 延迟 23.538s vs 13.971s（1.68x）
- 样本量 65，自动裁判，结论为方向性参考，不做统计显著性断言
