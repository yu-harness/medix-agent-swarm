# Bad Case：多轮上下文丢失

## 1. 现象

同 session 第二轮追问时，系统当作新对话：否认已有症状，或按成人通用建议回答儿科问题。

## 2. 复现（同 session）

| 轮次 | 原文 |
|------|------|
| T1 | 孩子反复咳嗽两周了 |
| T2 | 需要去医院吗？有什么护理建议？ |

## 3. 修复前错误表现

- 回复含「尚未提供症状 / 请补充症状」等
- 未锚定「孩子 / 咳嗽」
- 套用成人高血压式建议（限盐、低钠等）

## 4. 根因（编排，非纯模型）

1. Swarm 子任务描述未携带 T1 主诉，Worker 只见 T2 短问
2. Worker `record_memory` 污染短期记忆，主诉被淹没
3. Lead 分解/汇总未强制继承人群与症状锚点

## 5. 修复要点

| 机制 | 文件 |
|------|------|
| `user_turns` + `extract_session_anchor` | `memory/short_term.py` |
| 注入 `session_anchor`；子任务前缀「承接：…」；补记问答 | `swarm/swarm_coordinator.py` |
| 分解/汇总继承锚点 | `swarm/lead_agent.py` |
| `process_subtask` 注入 context；`record_memory=False` | `agents/base_agent.py`、`core/agent_loop.py` |
| 已知信息 + 禁止「尚未提供症状」 | `agents/consultation_agent.py`、`diagnostic_agent.py` |

## 6. 修复后验收

T2 须：提儿童/孩子 + 咳嗽；无「尚未提供症状」；含就医/护理；非成人高血压主线。

对照验收：[multiturn_context_fix.md](multiturn_context_fix.md)

## 7. 关联脚本

`eval/scripts/run_multiturn_context_smoke.py`
