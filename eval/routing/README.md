# 路由标注（200）

| 文件 | 说明 |
|------|------|
| [labeling_guide.md](labeling_guide.md) | 标注规范与规则 ID |
| `route_labels_200.jsonl` | **规则预标，非人工终审** |
| `prelabel_stats.json` | 预标分布 |

样本：`seed=42`，四类各 50，与 `eval/results/agent_eval_detail_full200.jsonl` 同批 `id`。

## 评测结果（相对预标）

脚本：`eval/scripts/run_route_eval.py`（只跑 Lead `assess_and_decompose` + `_collapse_subtasks`）。

| 指标 | 基线 |
|---|---:|
| 模式准确率 | **87.5%**（175/200） |
| 组合完全匹配率 | **83.0%**（166/200） |

分品类 mode：guideline 100% / disease 98% / health 90% / symptom 62%。

总体 ≥80%，**未改**路由代码。报告：`eval/results/route_eval_baseline.md`。

面试口径：200 是规则预标对齐率；**抽审 n=30** 可报模式 90.0% / 组合 73.3%，须说「按用户复核规则的抽审子集，非全量人工终审」。

## 抽审子集（已完成）

- 金标：`route_labels_sample30.jsonl`（每类 7–8，优先 revised / swarm / symptom 边界）
- 复测：模式 **90.0%**（27/30）；组合 **73.3%**（22/30）
- 报告：`eval/results/route_eval_sample30.md`

```text
python eval/scripts/run_route_eval.py --labels eval/routing/route_labels_sample30.jsonl --run-id sample30 --concurrency 3
```

## 终审

1. 编辑 `route_labels_200.jsonl`：改 `expected_mode`、`expected_agents`、`notes`、`review_status`。
2. `expected_agents` 排序 list，如 `["consultation","diagnostic"]`。
3. 改过 → `revised`；确认无改 → `confirmed`。

```text
python eval/scripts/prelabel_routes.py
```

会覆盖预标文件（终审改动先备份）。

```text
python eval/scripts/run_route_eval.py --run-id baseline --concurrency 4
```

## 人工复核修订摘要

下列 id 已按人工意见改为 `review_status=revised`（约 200 行未删行）：

| id | 修订要点 |
|---|---|
| `disease_knowledge_109` | 题面改为 VNS 治疗癫痫定义；非病因 |
| `disease_knowledge_042` | 金鱼→金钱草医学词条 |
| `health_consult_043` | 后遗症非急症；swarm `[consultation,research]` / `R_HC_CHRONIC_REHAB` |
| `health_consult_074` | 用药咨询非急救；single `[consultation]` / `R_HC_DRUG_SAFETY` |
| `health_consult_103` | 脑血管+神经体征；swarm `[consultation,diagnostic]` |
| `symptom_diagnosis_029` / `045` | 单一症状；single `[diagnostic]` |
| `symptom_diagnosis_120` | 改 `disease_knowledge` / `R_DK_DEFINE` / single `[consultation]` |
| `disease_knowledge_096` / `098` | 病因临床结构化；`R_DK_CLINICAL` / `[diagnostic]` |
| 题面去重 | 全文件截断句尾重复拼接（含 030/093/097/098/099 等） |

审完可重跑：`python eval/scripts/run_route_eval.py --run-id revised --concurrency 4`
