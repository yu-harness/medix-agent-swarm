# A/B 评测：Swarm vs Single Agent

> **自动 LLM 盲评，非医学专家盲评。面试须降级表述。**

- 样本 n=2（seed=42，每类 1，共 2）
- A 臂：`SwarmCoordinator(enable_swarm=True)`
- B 臂：`SwarmCoordinator(enable_swarm=False)`
- 两者使用同一知识库（64 条 chunk）与同一套工具，差异仅在于是否启用多 Agent 协作
- 盲评：每题随机决定呈现顺序，裁判不知哪个是 Swarm
- 耗时 37.0s

## 总体结果

| 结果 | 数量 | 占比 |
|---|---|---|
| Swarm 胜 | 0 | 0.0% |
| Single 胜 | 1 | 50.0% |
| 平局 | 1 | 50.0% |

## 三维均分（1-5）

| 维度 | Swarm | Single | 差值 |
|---|---|---|---|
| accuracy | 4.5 | 5.0 | -0.5 |
| completeness | 5.0 | 4.5 | 0.5 |
| safety | 5.0 | 5.0 | 0.0 |

## 延迟（秒）

| 指标 | Swarm | Single |
|---|---|---|
| p50 | 21.937 | 12.866 |
| p95 | 24.858 | 15.226 |
| p50 倍数 | 1.71x | 1x |

## 路由与超时

- A 臂（enable_swarm=True）中真正走 Swarm 的：1/2
- A 臂内部超时（55s）：0/2

### 按 Swarm 是否激活拆分（关键）

| 子集 | n | Swarm胜 | Single胜 | 平 | Swarm p50 | Single p50 |
|---|---|---|---|---|---|---|
| Swarm 已激活 | 1 | 0 | 0 | 1 | 25.183 | 10.245 |
| 未激活（回落单 Agent） | 1 | 0 | 1 | 0 | 18.691 | 15.488 |

## 分类明细

| 类别 | n | Swarm胜 | Single胜 | 平 | Swarm p50 | Single p50 |
|---|---|---|---|---|---|---|
| symptom_diagnosis | 1 | 0 | 0 | 1 | 25.183 | 10.245 |
| disease_knowledge | 1 | 0 | 1 | 0 | 18.691 | 15.488 |

## 结论口径（可直接用于面试）

- Swarm 相对单 Agent 的胜率：0.0%；三维均分差值 accuracy 4.5 vs 5.0、completeness 5.0 vs 4.5
- 代价：p50 延迟 21.937s vs 12.866s（1.71x）
- 样本量 2，自动裁判，结论为方向性参考，不做统计显著性断言
