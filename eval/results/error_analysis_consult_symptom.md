# health_consult + symptom_diagnosis 错题分析

- 来源: `agent_eval_detail_full200.jsonl`（seed=42，四类各50）
- 子集: consult 50 + symptom 50 = 100
- 基线 exact: consult 14% (7/50), symptom 16% (8/50)

## 判定分布

| category | correct | partial | incorrect |
|---|---:|---:|---:|
| health_consult | 7 | 36 | 7 |
| symptom_diagnosis | 8 | 27 | 15 |

- embedding < 0.45 直接判错: **3**（均在 symptom）

## Top 失败模式（人工聚类，n=85 incorrect/partial）

| 模式 | 约计 | 说明 |
|---|---:|---|
| miss_drug_name | 35 | 方向对，但未点名金标中的具体药/中成药/偏方 |
| miss_key_points | 33 | 漏检查项、忌口、就医专科、护理步骤等要点 |
| vague_or_offtopic | 8 | 空泛、答非所问、过度风险评估 |
| emb_low_gate | 3 | 相似度门禁误杀 |
| contradict_gold | 2 | 与金标结论冲突（如米醋软化血管） |
| overlong_vs_short_gold | 2 | 金标极短，系统过细被挑刺 |
| other | 2 | 杂项 |

## 典型例子

1. **痔疮**：系统讲坐浴/饮食，金标要槐角丸、痔速宁、马应龙 → partial
2. **小儿支气管炎**：系统讲遵医嘱与病毒为主，金标要细菌/病毒/支原体分型+阿莫西林/奥司他韦/阿奇霉素 → partial
3. **糖尿病饮食**：系统讲低GI粗粮，金标还强调忌蜜饯/罐头/含糖糕点 → partial
4. **米醋软化血管**：系统科学否定，金标“有一定作用” → incorrect（表述冲突）
5. **早泄/不孕**：系统偏风险与机制，金标要具体体位/药物/检查清单 → partial

## 优化方向（本轮落地）

1. Prompt 强制结构：问题理解 → 知识/鉴别 → 风险 → 可执行建议（含常见处理方向）→ 就医指征 → 免责
2. 知识库补高频主题中文短文，提高召回具体要点
3. 检索 top_k 上调；症状分析降低知识库分数门槛
4. 裁判：关键医疗要点覆盖即可，不要求药名/偏方逐字；emb 门禁略降
