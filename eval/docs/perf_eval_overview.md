# 性能测试总览（面试口径）

> 数字均来自已落盘报告，未外推。路由：全量 200 为规则预标；**抽审 n=30** 为按用户复核规则完成的子集金标（非全量人工终审）。A/B 为自动 LLM 对比，**非医学专家盲评**。

---

## 1. 测试集

| 项    | 说明                                                                                                                          |
| ---- | --------------------------------------------------------------------------------------------------------------------------- |
| 全集   | `eval/data/benchmark_500.jsonl`，**500** 条，四类各 **125**                                                                       |
| 公开样例 | `eval/data/benchmark_samples.jsonl`（每类 8）                                                                                   |
| 构建   | `eval/scripts/build_benchmark_500.py`                                                                                       |
| 字段   | `id`、`question`、`answer`、`category`、`source`、`source_id`、`notes`                                                            |
| 四类   | `health_consult` / `symptom_diagnosis` / `disease_knowledge` / `guideline_retrieval`                                        |
| 来源   | cMedQA2、webMedQA、Chinese-medical-dialogue-data、shibing624/medical、huatuo_encyclopedia_qa、本地 `knowledge/data/documents/`（指南） |
| 公开性  | **完整集私有**（gitignore，勿公开分发）；公开样例可提交。类别为启发式映射；答案非临床终审                                                                         |

---

## 2. 如何测试

| 项 | 说明 |
|---|---|
| 脚本 | `eval/scripts/run_agent_eval.py` |
| 质量子集 | seed=**42**，`--per-category 50` → **200** 条（四类各 50） |
| 延迟子集 | seed=**42**，四类各 20 → **80** 条（`latency80`） |
| 系统路径 | 真实端到端 Agent（`process_with_swarm` / Coordinator） |
| 裁判 | embedding `BAAI/bge-small-zh-v1.5` + DeepSeek LLM 终判；exact 仅计 `correct`；空答/超时/异常计错 |
| 基线门禁 | emb &lt; **0.45** → 直接判错 |
| 优化后门禁 | emb &lt; **0.40** → 直接判错；correct=覆盖关键医疗要点即可，**不要求**药名/偏方逐字 |
| 超时/重试 | 单条 120s，失败重试 1 次 |
| 延迟测法 | 同脚本记录每条端到端耗时，汇报 mean / p50 / p95；延迟优化后用 latency80 验收（门槛 p50≤20s、p95≤45s） |

```powershell
cd medix-agent-swarm
python eval/scripts/run_agent_eval.py --seed 42 --per-category 50 --run-id post_opt
# 延迟子集示例：--per-category 20 --run-id latency80
```

---

## 3. 结果

| 指标 | 基线 full200 | 质量优化 post_opt | 延迟优化 latency80 |
|---|---:|---:|---:|
| 样本 | 200（seed=42） | 200（同 seed） | 80（seed=42，各 20） |
| exact | 29.0%（58/200） | **91.0%（182/200）** | **90%（72/80）** |
| partial | 55.5% | 3.5% | 3.75% |
| health_consult | 14% | 96% | 100% |
| symptom_diagnosis | 16% | 82% | 85% |
| disease_knowledge | 32% | 88% | 80% |
| guideline_retrieval | 54% | 98% | 95% |
| mean / p50 / p95 (s) | 32.5 / 22.6 / 76.7 | 31.0 / 26.7 / 61.6 | **21.5 / 16.3 / 41.9** |

| 路由 | 模式 | 组合 | 报告 |
|---|---:|---:|---|
| 规则预标 n=200 | 87.5%（175/200） | 83.0%（166/200） | `route_eval_baseline.md` |
| 修订后预标 n=200 | 89.5%（179/200） | 82.0%（164/200） | `route_eval_revised.md` |
| **抽审子集 n=30** | **90.0%（27/30）** | **73.3%（22/30）** | `route_eval_sample30.md` |

**裁判校准（必交代）**：29%→91% = 系统侧（Prompt 结构、知识短文、检索 top_k）+ 裁判轻度校准（emb 0.45→0.40、correct 不要求药名/偏方逐字）。**勿把 91% 说成旧严裁判下的纯系统分数**；未做「仅系统 / 仅裁判」消融，不能报精确贡献占比。

延迟：post_opt 未过延迟线；latency80 达 p50≤20s / p95≤45s（子集验收，全 200 延迟复核可选未做）。

---

## 4. 智能路由

| 项 | 说明 |
|---|---|
| 预标金标 | `eval/routing/route_labels_200.jsonl`（规则预标 + 部分 revised） |
| 抽审子集 | `eval/routing/route_labels_sample30.jsonl`，**按用户复核规则完成的抽审子集**（人工规则指导下的抽审金标 n=30） |
| 脚本 | `eval/scripts/run_route_eval.py`（`--labels` / `--subset`；只跑 decompose + collapse） |
| 预标基线 n=200 | 模式 **87.5%**；组合 **83.0%** |
| **抽审 n=30** | 模式 **90.0%**（27/30）；组合 **73.3%**（22/30）；symptom mode 62.5% |
| 面试口径 | 抽审数字可讲；须声明「n=30 规则指导下抽审，非全量人工终审」；组合低于模式因 Agent 集合（尤其 diagnostic vs consultation）分歧 |

报告：`eval/results/route_eval_sample30.md` / `route_eval_baseline.md`

---

## 5. 纯检索准确率

| 项 | 说明 |
|---|---|
| 脚本 | `eval/scripts/run_retrieval_eval.py` |
| 样本 | n=**200**（seed=42，四类各 50，与质量同批分层） |
| 路径 | 仅 `MedicalKnowledgeBase.search`，**不跑 Agent** |
| 协议 | bge-small-zh 对 hit.content↔金标 cosine；hit@k = max_sim≥**0.40**；未用 LLM 裁判 |
| hit@1 / @3 / @8 | **98.5%**（197/200） |
| mean max_sim | 0.6666 |
| 分品类 hit@8 | health 100% / symptom 98% / disease 96% / guideline 100% |

报告：`eval/results/retrieval_eval_seed42.md` / `.json`  
局限：检索相关 ≠ 端到端答对；门禁非人工相关度终审。

---

## 6. 分阶段耗时

| 项 | 说明 |
|---|---|
| 脚本 | `eval/scripts/run_latency_breakdown.py` |
| 样本 | n=**24**（seed=42，四类各 6） |
| 方法 | 插桩 route / agent_process / retrieval / synthesize；`main_llm_approx≈agent−retrieval` |
| total mean/p50/p95 | **20.8 / 14.3 / 41.4** s |
| route_decompose | mean **1.46** s（p50 1.42） |
| main_llm_approx | mean **12.1** s（p50 9.86） |
| retrieval | mean **1.14** s（p50 0.59） |
| synthesize | mean **1.80** s（多数单 Agent 为 0） |
| other（记忆等） | mean **4.69** s |

报告：`eval/results/latency_breakdown_n24.md`  
局限：小样本；Swarm 并行时 agent 累加可能略大于墙钟；受 DeepSeek 限流影响。

---

## 7. Skill 越权 warn vs enforce

| 项 | 说明 |
|---|---|
| 脚本 | `eval/scripts/run_constraint_compare.py` |
| 样本 | 构造越权 n=**8**（AgentLoop mock，**不调真实 API**） |
| warn（ENFORCE=0） | 越权仍执行 **8/8** |
| enforce（ENFORCE=1） | 被拦截 **8/8** |
| Validator | warning×8 / block×8 与上一致 |

报告：`eval/results/constraint_enforce_compare.md`  
局限：构造场景，非线上自然越权率。

---

## 8. 并发吞吐

| 项 | 说明 |
|---|---|
| 脚本 | `eval/scripts/run_throughput_eval.py` |
| 设置 | n=**30**，concurrency=**3**，简单健康问句 |
| 成功率 | **100%**（30/30） |
| QPS（成功/墙钟） | **0.233** |
| 平均延迟（成功） | **12.6** s（p50 11.6 / p95 21.8） |
| 墙钟 | 128.5 s |

报告：`eval/results/throughput_eval.md`  
**受 DeepSeek 限流影响**；小规模，勿外推为大盘容量。

---

## 9. 多轮指代

| 项 | 说明 |
|---|---|
| 数据 | `eval/data/multiturn_50.jsonl`，n=**50**（自建两轮，含 dependency / ok_criteria） |
| 脚本 | `eval/scripts/run_multiturn_eval.py` |
| 路径 | 同 session 连续两次 `process_with_swarm` |
| 裁判 | LLM 判第 2 轮是否关联 dependency |
| **指代正确率** | **88.0%**（44/50） |
| 分品类 | disease/guideline 100%；health 85.7%；symptom **69.2%** |

报告：`eval/results/multiturn_eval.md`  
局限：自建集 + 自动 LLM 裁判，非人工盲评。

---

## 10. LLM 版 A/B（降级表述）

| 项 | 说明 |
|---|---|
| 样本 | n=**32**（seed=42，四类各 8） |
| A | 完整 `process_with_swarm` |
| B | 同模型直接 LLM（无 tools / 无 swarm） |
| 裁判 | 同系 LLM，accuracy/completeness/safety 1–5 + overall_better |
| **偏好 A** | **81.2%**（26/32）；B 12.5%；tie 6.2% |
| 均分亮点 | completeness A **4.75** vs B **3.63**（完整性差距最大） |

报告：`eval/results/ab_llm_judge.md`  
**面试必须说：自动 LLM 对比，非医学专家盲评**；不可称专家偏好或临床优越性。

---

## 11. 未测 / 待办

| 项 | 状态 |
|---|---|
| 路由全量人工终审 | 仅抽审 n=30；200 仍以预标+部分 revised 为主 |
| 全量 500 质量 | **未跑**（正式评测为分层 200） |
| 医学专家 A/B 盲评 | 未做（仅有自动 LLM 版） |

---

## 12. 关键路径

| 文档 | 路径 |
|---|---|
| 本总览 | `eval/docs/perf_eval_overview.md` |
| 评测说明 | `eval/README.md` |
| 质量优化复盘 | `eval/docs/quality_optimization_29_to_91.md` |
| 基线 | `eval/results/agent_eval_summary_full200.md` |
| 质量优化后 | `eval/results/agent_eval_summary_post_opt.md` |
| 延迟子集 | `eval/results/agent_eval_summary_latency80.md` |
| 纯检索 | `eval/results/retrieval_eval_seed42.md` |
| 分阶段耗时 | `eval/results/latency_breakdown_n24.md` |
| 约束对比 | `eval/results/constraint_enforce_compare.md` |
| 吞吐 | `eval/results/throughput_eval.md` |
| 路由基线 / 抽审 | `eval/results/route_eval_baseline.md` / `route_eval_sample30.md` |
| 多轮 | `eval/results/multiturn_eval.md` |
| A/B | `eval/results/ab_llm_judge.md` |
| 评测脚本 | `eval/scripts/run_*.py` |
| 路线图 | `docs/ROADMAP.md`、`eval/PERF_ROADMAP.md` |
| 路由标注 | `eval/routing/README.md` |
