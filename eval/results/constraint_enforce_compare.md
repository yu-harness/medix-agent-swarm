# Skill 越权：warn vs enforce

- 时间: 2026-08-23T09:06:06.285104+00:00
- 样本: 构造越权 n=8（每模式各跑一轮 AgentLoop mock，**不调真实 LLM/API**）

## 协议

1. 取 YAML 白名单外 (agent, tool) 对，强制 LLM mock 返回该 tool_call
2. `CONSTRAINT_ENFORCE=0`：期望 severity=warning 且仍 `execute_tool`
3. `CONSTRAINT_ENFORCE=1`：期望 severity=block 且不 `execute_tool`，messages 含 blocked
4. 另跑 `ConstraintValidator.validate_tool_call` 矩阵核对 severity

## AgentLoop 结果

| 模式 | 越权仍执行 | 被拦截 | 其它 |
|---|---:|---:|---:|
| warn (ENFORCE=0) | 8 | 0 | 0 |
| enforce (ENFORCE=1) | 0 | 8 | 0 |

## Validator severity

- warn 模式 warning 次数: 8
- enforce 模式 block 次数: 8

## 结论

- warn：越权仍执行 **8/8**
- enforce：被拦截 **8/8**

## 局限

- 为构造场景，非线上自然越权率；自然触发需另做日志统计
- 未跑真实 DeepSeek，避免打爆 API
