# 事故复盘：假异步把事件循环堵死，并行彻底退化成串行

> 一句话：**`async def` 是一张"我会让出控制权"的承诺书，而这 8 个 Skill 从头到尾没兑过现。**
> 结果 `asyncio.gather` 只是"给了并发的机会"，任务照样在唯一的 Event Loop 上排队跑到黑。
> 这是全项目里最"伪装得像间歇性故障"的一次：日志正常、异常没有、延迟稳定得可疑（正好等于串行之和）。

## 现象

- 场景：3 个 Agent 用 `asyncio.gather` 并行执行，每个任务阻塞 0.5s。
- 预期：总耗时 ≈ **0.5s**（三者重叠）。
- 实测：总耗时 **1.5s**，正好是 3 × 0.5s —— **完全串行**，`gather` 形同虚设。
- 诊断动作：注入一个 **50ms 的事件循环心跳探针**（在事件循环里每 50ms 打一次点）。
  - 1.5s 内心跳只应答 **1 次**（正常应为约 30 次；去掉 50ms 这个量级看，串行场景下应有的 9 次也没出现）。
  - 结论：**单线程 Event Loop 被彻底占死**，连"跳一下"的机会都没有。

## 根因

`.claude/skills/` 下 9 个 Skill，**8 个声明为 `async def` 但函数体内一个 `await` 都没有**：

- 它们做的是 `search_knowledge` / `assess_risk` / `clinical_guideline` 这类活，内部是 Milvus 检索、向量编码等**同步阻塞调用**；
- 注册中心（`core/skill_registry.py`）只看 `skill['is_async']`：见到 `async` 就直接 `await skill['function'](**kwargs)` 派发给事件循环；
- 于是这 8 个"假异步"占着唯一的 Event Loop 跑到返回，**承诺让出、从未让出** → 全局阻塞；
- 只有 `deep_research` 是真异步（内部有并发网页抓取的 `await`），它才是那个"该被 `await` 的"。

诊断难点：**没有异常、没有超时、日志一切正常**，唯一反常的是"耗时恰好等于串行之和"。这类故障很容易被当成"网络慢/模型慢"糊过去。

## 解决

1. **拒绝一刀切**：8 个无 `await` 的 Skill 从 `async def` **降级为同步 `def`**，注册中心的 `else` 分支把它们丢进**后台线程池**执行：

   ```python
   if skill['is_async']:
       result = await skill['function'](**kwargs)          # 真异步：让它 await
   else:
       result = await asyncio.to_thread(skill['function'], **kwargs)  # 同步：交给线程池
   ```

2. **只保留 `deep_research` 为 `async def`**——它内部有真实的 `await`（并发抓取），降级反而会损失并发。

3. **效果**：耗时 **1.5s → 0.5s**；心跳探针 **1 次 → 9 次**（对外不再卡死，也说明事件循环回到了"随时可响应"的状态）。

## 并发改造自带的坑：单例懒加载竞态

Skill 一旦真的跑在多个线程里，**首次并发调用会同时触发单例懒加载**：

- 现象：向量模型被重复加载（内存翻倍、启动变慢）、SQLite/Milvus 出现锁冲突；
- 根因：`MedicalKnowledgeBase` 是单例，但"检查是否存在 → 创建"这段没有保护；
- 解决：**Double-Checked Locking（双重检查锁定）**——只在**初始化**那段加锁，检索执行路径**不加锁**：

  ```python
  def __new__(cls, *args, **kwargs):
      if cls._instance is None:
          with cls._instance_lock:          # 只锁创建
              if cls._instance is None:     # 双重检查
                  cls._instance = super().__new__(cls)
      return cls._instance
  ```

- 关键细节：**锁的粒度必须只覆盖初始化**。如果连 `search()` 也一起锁，多线程检索会被重新"串行化"，等于把刚修好的并发又按下去了（还会让并发收益为零、且延迟看起来像回到从前）。
- 同一手法也用在 reranker 的懒加载上（`_get_reranker`，模型约 1GB，并发首次检索会重复加载）；`ShortTermMemory` 等单例同理。

## 启示（面试口径）

> 「这次故障最有价值的地方是**它的伪装**：没有异常、没有超时、日志全绿，唯一线索是"总耗时恰好等于几个任务耗时之和"。我用一个 **50ms 的心跳探针**确认事件循环被占死，然后定位到 9 个 Skill 里有 8 个是**假异步**——声明了 `async def`，函数体里一个 `await` 都没有，全是 Milvus 和向量推理这类同步阻塞调用。
> **修法是"拒绝一刀切"**：8 个降级为同步 `def`，交给线程池；只有内部真有并发抓取的 `deep_research` 保留 `async def`。耗时从 1.5s 回到 0.5s，心跳从 1 次恢复到 9 次。
> 而且线程化立刻带出第二个坑：单例懒加载竞态导致模型重复加载，我们用**双重检查锁定**只锁初始化、不锁检索——**锁的粒度错了，等于把并发重新按回串行**。
> 这件事之后我形成了一个判断标准：**看到 `async def` 就先问它内部有没有 `await`；没有，它就不该是 `async`。**」
