# 纯检索准确率评测

- 时间: 2026-08-23T09:09:30.474252+00:00
- n=200（seed=42，per_category=50，与质量评测同批分层）
- 路径: MedicalKnowledgeBase.search（不跑 Agent）
- top_k=8，命中门禁 emb≥0.4

## 协议

1. 对每条问题调用 `MedicalKnowledgeBase.search(question, top_k=8)`
2. 用 `BAAI/bge-small-zh-v1.5` 计算每条 hit.content 与标准答案的 cosine
3. `max_sim` = top_k 内最大相似度；`top1_sim` = 第 1 条
4. hit@1 / hit@3 / hit@k：对应范围内 max_sim ≥ 0.40（对齐 post_opt emb_low）
5. 本轮**未**用 LLM 裁判（省 API；协议以 embedding 为准）

## 结果

| 指标 | 值 |
|---|---:|
| hit@1 | 98.5%（197/200） |
| hit@3 | 98.5%（197/200） |
| hit@8 | 98.5%（197/200） |
| mean max_sim | 0.6666 |
| mean top1_sim | 0.6279 |
| 空检索 | 0 |

## 分品类 hit@k

| category | n | hit@1 | hit@3 | hit@k | mean_max_sim |
|---|---:|---:|---:|---:|---:|
| health_consult | 50 | 100.0% | 100.0% | 100.0% | 0.649 |
| symptom_diagnosis | 50 | 98.0% | 98.0% | 98.0% | 0.64 |
| disease_knowledge | 50 | 96.0% | 96.0% | 96.0% | 0.6024 |
| guideline_retrieval | 50 | 100.0% | 100.0% | 100.0% | 0.7749 |

## 局限

- 金标答案未必在知识库中；检索相关≠端到端答对
- embedding 门禁与质量裁判同模型，非人工相关度终审
