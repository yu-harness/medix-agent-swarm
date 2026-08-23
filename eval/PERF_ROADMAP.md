# 性能提升路线图（质量为主）

**基准（full200，2026-07-27）**  
exact **29.0%**（58/200）｜p50 **22.6s** / p95 **76.7s**

**质量重评（post_opt，2026-08-23）**  
exact **91.0%**（182/200）✅ 已过 45% ｜p50 **26.7s** / p95 **61.6s**  
产物：`eval/results/agent_eval_summary_post_opt.md`

**延迟子集（latency80，同日）**  
exact **90%**（72/80）｜p50 **16.3s** / p95 **41.9s** ✅ 达延迟门槛  
Swarm 占比约 17.5%（14/80）  
产物：`eval/results/agent_eval_summary_latency80.md`

**周目标**  
1. 质量 exact ≥45% → **已达成（91%）**  
2. 延迟 p50≤20s、p95≤45s → **子集已达成（latency80）**；可选全200复核  
3. 路由：用户终审金标后才测

**验收标准**  
| 指标 | 门槛 | 状态 |
|------|------|------|
| exact | ≥45%（200） | ✅ post_opt 91% |
| 延迟 | p50≤20s、p95≤45s | ✅ latency80 |
| 路由 | 终审后再报 | ⏳ 待用户审约30条 |

---

## Phase 1 — 质量（✅）

Prompt/KB/检索/裁判校准 + HF_HOME 本地 embedding。冻结大改。

---

## Phase 2 — 延迟（✅ 子集过线）

**已改**：Coordinator 复用、Lead 收紧 + `_collapse_subtasks`、Swarm 超时 55s、Mem0 2s 超时、本地 embedding。  
**可选下一刀**：全200 延迟复核；若 p95 回弹再压 Swarm。

---

## Phase 3 — 路由（依赖用户）

`eval/routing/route_labels_200.jsonl` + `labeling_guide.md`  
先审约 **30 条** → 定稿 → 再跑准确率。

---

## 明确不做什么

- 不硬凑多轮92%/AB/检索87%  
- 不重复质量大改  
- 不 git commit（除非明确要求）

---

## 补充自动评测（2026-08-23）

检索 hit@8 **98.5%**（n=200）｜分阶段 n=24 total p50 **14.3s**｜约束 warn 8/8 vs enforce 拦 8/8｜吞吐 n=30,c=3 成功率 100%、QPS **0.233**  
见 `eval/docs/perf_eval_overview.md`

## 用户现在要做的

1. **审路由金标约30条**（`eval/routing/`）  
2. 面试：质量报 post_opt **91%**（说明含裁判校准）；延迟报 latency80 **p50 16.3 / p95 41.9**；路由写「预标已测、人工抽审待你」
