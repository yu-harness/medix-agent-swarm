# 事故复盘：并行化之后，TraceID 断在 线程边界 + 云端 SDK 拖垮主链路

> 一句话：**并发改造不是"把 async 换成线程池"就结束了，它同时带出两个新问题——观测会断、外部依赖会失控。**
> 前者让你失去定位能力（日志断成两截），后者让一个"锦上添花"的云端记忆服务有了拖垮主链路的权力。

## 现象

这次是**两个问题一起爆**的，都发生在「8 个 Skill 从假异步降级为同步 `def`、改由线程池执行」之后（见 [fake_async_eventloop.md](fake_async_eventloop.md)）：

1. **TraceID 截断丢失**：一次请求的日志在 Agent 层之后就"断"了——Skill 线程里打出的日志不再携带调用方的 `trace_id`，同一条链路在日志里被切成两段，端到端无法串联。做耗时归因时，Skill 段的耗时全部落进"未知"。
2. **Mem0 云端同步 SDK 抖动直接拖垮主链路**：长期记忆检索走 Mem0 的**同步**网络 SDK；某次云端抖动时，首字延迟被它一个人拉长了十几倍。而记忆检索在业务上只是"给答案补充历史上下文"的**可选增强**——不该有权决定用户等多久。

## 根因

**问题 1：`contextvars` 是"每个 context 一份"，跨线程不会自动可见**

- `contextvars` 的值绑定在**当前 context** 上，线程池里的函数运行在**另一个 context**里，发起方 `set` 过的值不会自动出现在那边；
- 更麻烦的是 `loop.run_in_executor` 的上下文行为**与 Python 版本/实现相关**（3.12 起才有向线程池传播上下文的说法），把观测正确性押在版本行为上等于给自己埋雷；
- 所以现象是"日志静默少字段"——不会报错，只会让你事后查不到证据。

**问题 2：同步网络 I/O 卡在 async 路径上，且没有上界**

- Mem0 SDK 是同步的，直接在协程里调用 = **阻塞唯一的事件循环**（和 [fake_async_eventloop.md](fake_async_eventloop.md) 是同一个病根的另一张脸）；
- 没有超时上限 → 外部依赖的抖动会**无上界**地传导到用户首字延迟上。

## 解决

**1）跨线程上下文：统一走 `asyncio.to_thread`**

`asyncio.to_thread()` 内部就是"先 `copy_context()`，再把函数丢进线程池"，因此**与版本无关**。技能注册中心的做法（`core/skill_registry.py`）：

```python
# 必须用 asyncio.to_thread：它会把当前 contextvars（如 trace_id）
# 复制进工作线程，让 Skill 线程里的日志也能携带调用方的 trace_id。
# 直接 run_in_executor 不会复制上下文，观测会被截断在 agent 层。
result = await asyncio.to_thread(skill['function'], **kwargs)
```

配套的关键设计（`core/observability.py`）：**ContextVar 里放的是「可变对象」而不是数字**——

> `asyncio.create_task` 与 `run_in_executor` 都会**复制**当前 context。
> 在复制出来的 context 里给 ContextVar 重新赋值，**不会**回传到父 context。
> 但只要多个 context 持有**同一个对象**，对对象内部状态的写入是互相可见的。

所以 `TokenLedger` / `SpanRecorder` 的实例被放进 ContextVar：

- 并发的 Worker 任务、线程池里的 Skill，**都能把用量与耗时写进同一个账本**；
- 请求结束时在主 context 上取一次快照即可，**不需要任何回传机制**。

这条设计同时解决了"跨线程"和"汇总"两件事，也是 Day 4 前端能用真实 Span 驱动状态胶囊的前提。

**2）外部依赖：`asyncio.to_thread` + 硬超时 + 静默降级**

长期记忆检索（`swarm/swarm_coordinator.py`）：

```python
# 2. 检索长期记忆（相似历史会话）——短超时，避免拖慢主路径
similar_memories = []
try:
    similar_memories = await asyncio.wait_for(
        asyncio.to_thread(
            self.long_term_memory.search_similar_sessions,
            question, 3, user_id=mem_user_id,
        ),
        timeout=2.0,
    )
except Exception as e:
    logger.warning(f"long-term memory search skipped: {type(e).__name__}")
```

三条要点：

- **`to_thread`**：不让同步 SDK 阻塞事件循环；
- **`wait_for(timeout=2.0)`**：给外部依赖设一个不可突破的上界，2s 内没回来就放弃；
- **静默降级但留 warning**：没记忆也能照常回答（记忆是可选增强），但**必须留下 warning 日志**——"静默吞掉"等于把故障藏起来，正好是问题 1 的翻版。

同一手法也用在画像抽取的 LLM 调用上（`memory/patient_profile.py`，同样 2.0s 硬超时），保证任何单个可选增强都不能主导首字延迟。

## 启示（面试口径）

> 「我们把 8 个 Skill 从假异步改成线程池之后，立刻冒出两个新问题——**这类改造真正的难点从来不是'换成并发'，而是并发之后新引入的失败模式。**
> 第一，`contextvars` 是每个 context 一份，线程池里的函数看不到调用方的 `trace_id`，日志静默少字段、链路断成两截。我们的做法是**统一走 `asyncio.to_thread`**（它内部就是 `copy_context().run(...)`，与 Python 版本无关，不押 `run_in_executor` 的版本行为）；同时把账本设计成『ContextVar 里放**可变对象**』——并发的 Worker 和线程池里的 Skill 写进的是**同一个账本对象**，请求结束在主 context 取快照就行，不需要回传机制。
> 第二，Mem0 是同步云端 SDK，一次抖动就把首字延迟拉长十几倍，而记忆只是可选增强。我们给它包了 `to_thread` + **2.0s 硬超时 + 静默降级**——没记忆也能回答，但**必须留一条 warning**：超时可以降级，故障不能藏。」
>
> 一句话总结这条线：**并发改造要同时交付三样东西——让出控制权、别丢观测、给外部依赖设上界。**

## 遗留债（建议清理，未动）

仓库里关于「`run_in_executor` 是否会传播 `contextvars`」目前有**三处互相矛盾的注释**：

| 位置 | 现有说法 |
| --- | --- |
| `core/observability.py`（模块 docstring） | 「`create_task` 与 `run_in_executor` 都会复制当前 context」 |
| `core/skill_registry.py`（调度处） | 「直接 `run_in_executor` 不会复制上下文，观测会被截断在 agent 层」 |
| `swarm/swarm_coordinator.py`（请求入口） | 「Python 3.12+ 的 `asyncio.run_in_executor` 会把 contextvars 传播进线程池」 |

三者各说一半，读者会困惑。建议统一成一条可执行规则：

> **跨线程一律走 `asyncio.to_thread`（显式复制上下文，与版本无关）；不要依赖 `run_in_executor` 的上下文行为。**

（本 RFC 只记录事实，未改动这三处代码注释——避免把"文档口径统一"和"行为变更"混进同一个提交。）
