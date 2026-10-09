# A/B 评测：Swarm vs Single Agent

> **自动 LLM 盲评，非医学专家盲评。面试须降级表述。**

- 样本 n=0（seed=42，每类 1，共 2）
- A 臂：`SwarmCoordinator(enable_swarm=True)`
- B 臂：`SwarmCoordinator(enable_swarm=False)`
- 两者使用同一知识库（64 条 chunk）与同一套工具，差异仅在于是否启用多 Agent 协作
- 盲评：每题随机决定呈现顺序，裁判不知哪个是 Swarm
- 耗时 169.9s

## 总体结果

| 结果 | 数量 | 占比 |
|---|---|---|
| Swarm 胜 | 0 | 0.0% |
| Single 胜 | 0 | 0.0% |
| 平局 | 0 | 0.0% |

## 三维均分（1-5）

| 维度 | Swarm | Single | 差值 |
|---|---|---|---|
| accuracy | None | None | - |
| completeness | None | None | - |
| safety | None | None | - |

## 延迟（秒）

| 指标 | Swarm | Single |
|---|---|---|
| p50 | 0.0 | 0.0 |
| p95 | 0.0 | 0.0 |
| p50 倍数 | 0.0x | 1x |

## 路由与超时

- A 臂（enable_swarm=True）中真正走 Swarm 的：0/0
- A 臂内部超时（55s）：0/0

## 分类明细

| 类别 | n | Swarm胜 | Single胜 | 平 | Swarm p50 | Single p50 |
|---|---|---|---|---|---|---|

## 结论口径（可直接用于面试）

- Swarm 相对单 Agent 的胜率：0.0%；三维均分差值 accuracy None vs None、completeness None vs None
- 代价：p50 延迟 0.0s vs 0.0s（0.0x）
- 样本量 0，自动裁判，结论为方向性参考，不做统计显著性断言
