# A/B 评测：Swarm vs Single Agent

> **自动 LLM 盲评，非医学专家盲评。面试须降级表述。**

- 样本 n=27（seed=42，每类 8，共 27）
- A 臂：`SwarmCoordinator(enable_swarm=True)`
- B 臂：`SwarmCoordinator(enable_swarm=False)`
- 两者使用同一知识库（64 条 chunk）与同一套工具，差异仅在于是否启用多 Agent 协作
- 盲评：每题随机决定呈现顺序，裁判不知哪个是 Swarm
- 耗时 482.6s

## 总体结果

| 结果 | 数量 | 占比 |
|---|---|---|
| Swarm 胜 | 12 | 44.4% |
| Single 胜 | 3 | 11.1% |
| 平局 | 12 | 44.4% |

## 三维均分（1-5）

| 维度 | Swarm | Single | 差值 |
|---|---|---|---|
| accuracy | 4.704 | 4.63 | 0.074 |
| completeness | 4.852 | 4.556 | 0.296 |
| safety | 4.889 | 4.815 | 0.074 |

## 延迟（秒）

| 指标 | Swarm | Single |
|---|---|---|
| p50 | 27.429 | 14.681 |
| p95 | 74.39 | 36.982 |
| p50 倍数 | 1.87x | 1x |

## 路由与超时

- A 臂（enable_swarm=True）中真正走 Swarm 的：13/27
- A 臂内部超时（55s）：1/27

### 按 Swarm 是否激活拆分（关键）

| 子集 | n | Swarm胜 | Single胜 | 平 | Swarm p50 | Single p50 |
|---|---|---|---|---|---|---|
| Swarm 已激活 | 13 | 10 | 2 | 1 | 46.146 | 13.863 |
| 未激活（回落单 Agent） | 14 | 2 | 1 | 11 | 18.262 | 16.098 |

## 分类明细

| 类别 | n | Swarm胜 | Single胜 | 平 | Swarm p50 | Single p50 |
|---|---|---|---|---|---|---|
| health_consult | 8 | 3 | 1 | 4 | 22.668 | 16.505 |
| symptom_diagnosis | 7 | 4 | 0 | 3 | 24.27 | 13.273 |
| disease_knowledge | 4 | 3 | 0 | 1 | 48.055 | 12.726 |
| guideline_retrieval | 8 | 2 | 2 | 4 | 40.208 | 20.637 |

## 结论口径（可直接用于面试）

- Swarm 相对单 Agent 的胜率：44.4%；三维均分差值 accuracy 4.704 vs 4.63、completeness 4.852 vs 4.556
- 代价：p50 延迟 27.429s vs 14.681s（1.87x）
- 样本量 27，自动裁判，结论为方向性参考，不做统计显著性断言
