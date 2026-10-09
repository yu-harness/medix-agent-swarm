# 评测

全中文文本集 **500** 条，四类各 **125**。完整集默认不入库；公开样例可提交。

性能测试总览：[docs/perf_eval_overview.md](docs/perf_eval_overview.md)  
质量优化（exact 29%→91%）：[docs/quality_optimization_29_to_91.md](docs/quality_optimization_29_to_91.md)  
多轮上下文丢失 Bad Case：[docs/badcase_multiturn_context_loss.md](docs/badcase_multiturn_context_loss.md)  
路由基线（规则预标非人工终审：模式 87.5% / 组合 83.0%；**人工抽审待用户**）：[results/route_eval_baseline.md](results/route_eval_baseline.md)  
纯检索（hit@8 98.5%，n=200）：[results/retrieval_eval_seed42.md](results/retrieval_eval_seed42.md)
CI 冒烟门禁（安全护栏 / 检索 hit@1 / 数据一致性，纯本地零 API）：`scripts/run_ci_smoke.py`，工作流 `.github/workflows/eval.yml`  
分阶段耗时（n=24）：[results/latency_breakdown_n24.md](results/latency_breakdown_n24.md)  
约束 warn vs enforce：[results/constraint_enforce_compare.md](results/constraint_enforce_compare.md)  
并发吞吐（n=30,c=3）：[results/throughput_eval.md](results/throughput_eval.md)  
路由标注：[routing/README.md](routing/README.md)  
后续指标：[../docs/ROADMAP.md](../docs/ROADMAP.md)

## 数据

| 路径 | 说明 |
|------|------|
| `data/benchmark_500.jsonl` | 完整集（gitignore） |
| `data/benchmark_samples.jsonl` | 公开样例（每类 8） |
| `scripts/build_benchmark_500.py` | 复现构建 |
| `results/` | 评测 summary / detail（保留产物，勿当百科改） |

字段：`id`、`question`、`answer`、`category`（`health_consult` / `symptom_diagnosis` / `disease_knowledge` / `guideline_retrieval`）、`source`、`source_id`、`notes`。

主要来源：cMedQA2、webMedQA、Chinese-medical-dialogue-data、shibing624/medical、huatuo_encyclopedia_qa、本地 `knowledge/data/documents/`（指南类）。许可与局限：完整集勿公开分发；类别为启发式映射；答案非临床终审。

```powershell
cd eval
python .\scripts\build_benchmark_500.py
```

## 跑 Agent 评测

需已配置 `config.py`、知识库已导入。

```powershell
cd medix-agent-swarm
# 冒烟
python eval/scripts/run_agent_eval.py --per-category 50 --limit 4 --run-id smoke
# 正式分层 200（四类各 50，seed=42）
python eval/scripts/run_agent_eval.py --seed 42 --per-category 50 --run-id post_opt
# 续跑
python eval/scripts/run_agent_eval.py --resume --resume-detail eval/results/agent_eval_detail_<run>.jsonl
```

常用参数：`--data`、`--out-dir`、`--semaphore`、`--timeout`、`--categories`。

产物：`eval/results/agent_eval_summary_*.md`、`agent_eval_detail_*.jsonl`；最新可看 `latest_summary.md`。对照报告：`agent_eval_summary_full200.md`（基线）、`agent_eval_summary_post_opt.md`（优化后）。

## 其它自动评测

```powershell
# 纯检索（本地 Milvus，不跑 Agent）
python eval/scripts/run_retrieval_eval.py --seed 42 --per-category 50 --run-id seed42
# 分阶段耗时（小样本）
python eval/scripts/run_latency_breakdown.py --seed 42 --per-category 6 --run-id n24
# 约束 warn vs enforce（mock，不打 API）
python eval/scripts/run_constraint_compare.py
# 并发吞吐（注意限流）
python eval/scripts/run_throughput_eval.py --n 30 --concurrency 3
```
