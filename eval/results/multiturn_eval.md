# 多轮指代评测报告

> 协议：同一 session 连续两次 `process_with_swarm`；LLM 判第2轮是否关联 dependency。

- n: 50
- **指代正确率**: **88.0%** (44/50)
- concurrency: 2
- timeout_sec: 120.0
- elapsed_sec: 1164.5
- data: `D:\md\AI实习\project\居丽叶简历项目7：医疗助手\medix-agent-swarm\eval\data\multiturn_50.jsonl`

## 分品类

| category | n | coref_acc |
|---|---:|---:|
| disease_knowledge | 12 | 100.0% (12/12) |
| guideline_retrieval | 11 | 100.0% (11/11) |
| health_consult | 14 | 85.7% (12/14) |
| symptom_diagnosis | 13 | 69.2% (9/13) |

- verdict_counts: `{'correct': 44, 'incorrect': 6}`
- error_counts: `{}`

## 局限

- 自建 50 条两轮指代集，非公开多轮金标
- 裁判为自动 LLM，非人工盲评
