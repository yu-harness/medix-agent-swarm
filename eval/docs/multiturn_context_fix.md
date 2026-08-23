# Multiturn 上下文锚点验收

## 改动文件

- `memory/short_term.py`：`user_turns` + `extract_session_anchor`
- `swarm/swarm_coordinator.py`：注入 `session_anchor`；子任务强制「承接：…」；Swarm 补记问答
- `swarm/lead_agent.py`：分解/汇总继承锚点
- `agents/base_agent.py`：`process_subtask` 注入 context；`record_memory=False`
- `core/agent_loop.py`：支持 `record_memory` / `load_history`
- `agents/consultation_agent.py` / `diagnostic_agent.py`：已知信息 + 禁止「尚未提供症状」
- `eval/scripts/run_multiturn_context_smoke.py`

## 真对话结果

- session: `ctx-mt-20260823-205426-c92785`
- T1: 孩子反复咳嗽两周了
- T2: 需要去医院吗？有什么护理建议？

### Checks

- PASS: `t1_ok`
- PASS: `t2_mentions_child`
- PASS: `t2_mentions_cough`
- PASS: `t2_no_missing_symptom`
- PASS: `t2_care_or_hospital`
- PASS: `t2_not_adult_htn_main`

### T2 excerpt

针对孩子反复咳嗽两周：**建议尽快就医**；护理含温水、湿度、禁烟、清淡饮食、拍背等儿科要点；无「尚未提供症状」、非成人高血压式建议。

## 复测

```bash
cd medix-agent-swarm
python eval/scripts/run_multiturn_context_smoke.py
```
