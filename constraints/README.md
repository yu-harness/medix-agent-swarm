# 约束：warn → enforce

## 问题
全量注册 Skills 后，LLM 可能越权调用非本 Agent 白名单工具 → 合规风险 + 延迟/质量下降。

## 方法
1. YAML（`agent_constraints.yaml`）定义各 Agent `allowed_tools`
2. **默认 warn**：违规只打 `约束警告(warned)` 日志，仍执行（防误伤、可观测）
3. **可选 enforce**：`CONSTRAINT_ENFORCE=1` 时硬拦，不 `execute_tool`，向 messages 回写拒绝原因（含可用 Skill 列表）

## 开启 enforce
```bash
# Windows PowerShell
$env:CONSTRAINT_ENFORCE="1"

# 或在 config.py
CONSTRAINT_ENFORCE = True
```
不设或 `0` / `False` → 保持现网 warn。

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
