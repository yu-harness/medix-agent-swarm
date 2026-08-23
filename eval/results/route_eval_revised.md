# 路由评测报告 — revised

> **金标性质：规则预标，非人工终审。** 准确率仅相对预标规则，不可当作人工金标准。

## Meta

- n: 200
- labels: `D:\md\AI实习\project\居丽叶简历项目7：医疗助手\medix-agent-swarm\eval\routing\route_labels_200.jsonl`
- concurrency: 4
- timeout_sec: 45.0
- elapsed_sec: 65.0
- started_at: 2026-08-23T09:10:57.906173+00:00
- finished_at: 2026-08-23T09:12:02.909947+00:00

## 指标

- **模式准确率** (predicted_mode vs expected_mode): **89.5%** (179/200)
- **组合完全匹配率** (agents 集合): **82.0%** (164/200)

## 分品类

| category | n | mode_acc | combo_acc |
|---|---:|---:|---:|
| disease_knowledge | 51 | 98.0% (50/51) | 82.3% (42/51) |
| guideline_retrieval | 50 | 100.0% (50/50) | 96.0% (48/50) |
| health_consult | 50 | 96.0% (48/50) | 90.0% (45/50) |
| symptom_diagnosis | 49 | 63.3% (31/49) | 59.2% (29/49) |

## 分布

- expected_mode: `{'single': 187, 'swarm': 13}`
- predicted_mode: `{'single': 170, 'swarm': 30}`
- confusion: `{'both_ok': 164, 'combo_only': 15, 'both_wrong': 21}`
