# 约束：默认 enforce（硬拦），可退回 warn

## 问题
注册阶段已按 YAML 白名单裁剪（consultation 5 / diagnostic 7 / research 5），LLM 的 tools 里本就没有白名单外 Skill；但模型仍可能幻觉出未注册调用、或白名单漏配真实合理调用 → validator 作**第二道兜底闸**（否则有合规风险 + 延迟/质量下降）。

## 方法
1. YAML（`agent_constraints.yaml`）定义各 Agent `allowed_tools`
2. **默认 enforce（硬拦）**：越权不 `execute_tool`，向 messages 回写拒绝原因（含可用 Skill 列表）
3. **可选 warn**：`CONSTRAINT_ENFORCE=0` 时只打 `约束警告(warned)` 日志、仍执行，仅用于排查白名单漏配

## 退回 warn（仅排查用）
```bash
# Windows PowerShell
$env:CONSTRAINT_ENFORCE="0"

# 或在 config.py
CONSTRAINT_ENFORCE = False
```
不设或 `1` / `True` → 硬拦（默认）。

## 复现对比（可选）
```bash
# warn：看日志里 warned 次数（工具仍会执行）
$env:CONSTRAINT_ENFORCE="0"
python examples/test_suite.py

# enforce：看 blocked；mock 用例 TestAgentLoopConstraintEnforce 断言未调用 execute_tool
$env:CONSTRAINT_ENFORCE="1"
python -m unittest examples.test_suite.TestAgentLoopConstraintEnforce -v
```
小样本人工对比时可各跑 5～10 条真实问句，统计日志中 `warned` vs `blocked` 次数；未跑则勿写假数字。
