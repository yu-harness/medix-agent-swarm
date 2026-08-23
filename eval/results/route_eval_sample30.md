# 路由评测报告 — sample30

> **金标性质：按用户复核规则完成的抽审子集（人工规则指导下的抽审金标 n=30）。**  
> 非全量人工终审；由 Agent 按用户既定复核规则（急症误标、单症状不 swarm、分类冲突、题面去重、非医疗替换等）抽审落地。

## Meta

- n: 30
- labels: `eval/routing/route_labels_sample30.jsonl`
- concurrency: 3
- timeout_sec: 45.0
- elapsed_sec: 12.5
- started_at: 2026-08-23T09:31:09.375345+00:00
- finished_at: 2026-08-23T09:31:21.865701+00:00

## 指标

- **模式准确率** (predicted_mode vs expected_mode): **90.0%** (27/30)
- **组合完全匹配率** (agents 集合): **73.3%** (22/30)

## 分品类

| category | n | mode_acc | combo_acc |
|---|---:|---:|---:|
| disease_knowledge | 7 | 100.0% (7/7) | 57.1% (4/7) |
| guideline_retrieval | 7 | 100.0% (7/7) | 100.0% (7/7) |
| health_consult | 8 | 100.0% (8/8) | 87.5% (7/8) |
| symptom_diagnosis | 8 | 62.5% (5/8) | 50.0% (4/8) |

## 分布

- expected_mode: `{'single': 23, 'swarm': 7}`
- predicted_mode: `{'single': 24, 'swarm': 6}`
- confusion: `{'both_ok': 22, 'combo_only': 5, 'both_wrong': 3}`
