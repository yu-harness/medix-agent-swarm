# Multiturn P0 验收

## 改动文件

- `main.py`：answer 已含免责/核心建议则 CLI 不再另打
- `agents/consultation_agent.py`：首轮六段 / 追问增量；不 dump `recent_history`
- `agents/diagnostic_agent.py`：同上增量规则
- `swarm/lead_agent.py`：synthesis 支持增量汇总
- `swarm/swarm_coordinator.py`：follow-up 只打标；disclaimer 提取复用；historical_cases 标注参考
- `validation/auto_fixer.py`：已有免责不再 append
- `eval/scripts/run_multiturn_p0_smoke.py`：同 session T1→T2 冒烟

## 真对话结果

- session: `p0-mt-20260823-184257-ba75da`
- T1: 我有高血压（len≈1233，完整向）
- T2: 那饮食方面要注意什么？（len≈519，限盐/蛋白/脂肪实操增量）

### Checks

- PASS: `t2_no_fake_recall`
- PASS: `t2_mentions_diet`
- PASS: `t2_has_disclaimer_once`
- PASS: `cli_no_double_disclaimer_t2`
- PASS: `cli_no_double_suggestions_t2`
- PASS: `t1_fullish_or_ok`

### T2 excerpt

```
【回答】 关于高血压的饮食管理，您需要重点把握"**限盐、换蛋白、控脂肪**"三件事：…【就医红线】…以上信息仅供参考，不能替代专业医生的诊断和治疗。
```

## 手动复测

```bash
cd medix-agent-swarm
python eval/scripts/run_multiturn_p0_smoke.py
# 或
python main.py
# 同会话：我有高血压 → 那饮食方面要注意什么？
# 看：无「您之前问过饮食」、T2 偏实操、终端免责不双重
```
