# 路由评测报告 — baseline

> **金标性质：规则预标，非人工终审。** 准确率仅相对预标规则，不可当作人工金标准。

## Meta

- n: 200
- labels: `D:\md\AI实习\project\居丽叶简历项目7：医疗助手\medix-agent-swarm\eval\routing\route_labels_200.jsonl`
- concurrency: 4
- timeout_sec: 45.0
- elapsed_sec: 70.5
- started_at: 2026-08-23T08:53:56.962715+00:00
- finished_at: 2026-08-23T08:55:07.458668+00:00

## 指标

- **模式准确率** (predicted_mode vs expected_mode): **87.5%** (175/200)
- **组合完全匹配率** (agents 集合): **83.0%** (166/200)

## 分品类

| category | n | mode_acc | combo_acc |
|---|---:|---:|---:|
| disease_knowledge | 50 | 98.0% (49/50) | 86.0% (43/50) |
| guideline_retrieval | 50 | 100.0% (50/50) | 96.0% (48/50) |
| health_consult | 50 | 90.0% (45/50) | 88.0% (44/50) |
| symptom_diagnosis | 50 | 62.0% (31/50) | 62.0% (31/50) |

## 分布

- expected_mode: `{'single': 185, 'swarm': 15}`
- predicted_mode: `{'single': 168, 'swarm': 32}`
- confusion: `{'both_ok': 166, 'both_wrong': 25, 'combo_only': 9}`
