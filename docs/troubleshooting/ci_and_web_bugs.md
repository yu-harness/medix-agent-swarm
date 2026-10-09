# CI 与 Web Demo：四个深水区事故复盘

> 范围：Day 3（CI 门禁落地）与 Day 4（问诊 Web Demo）真实发生、且**在本地永远撞不到**的四次故障。
> 共同规律：前三个都是「本地环境已就位」掩盖的真实缺陷——配置漂移、依赖缺失、测试选择规则差异；
> 第四个是架构约束——请求级上下文（ContextVar）在多入口下被下游覆盖。
> 一句话总结：**CI 的价值不在"跑一遍测试"，而在于它是唯一一个"全新环境"的检验场。**

## 速览

| # | 现象一句话 | 本地为什么看不见 | 归属 |
| --- | --- | --- | --- |
| 1 | CI 第一步就 `ImportError` 猝死 | 那个文件被 `.gitignore` 忽略，本地从没经过它 | 配置契约 |
| 2 | `No module named 'milvus_lite'` | 本地早就装过这个包，只是没写进依赖清单 | 依赖声明 |
| 3 | 15 个用例在 CI 红屏 | 本地 pytest 恰好在"警告后跳过"而不是"判失败" | 测试分层 |
| 4 | 前端状态胶囊读不到真实 Span | 单进程手工调用时不会出现双记录器竞争 | 请求级上下文 |

---

## Bug 1：`config.example.py` 配置漂移，CI 导入期猝死

**现象**

- 本地 `python -m pytest examples/ -q` 全绿；push 后 CI 第一步（`Run unit tests`）直接死：

```text
core/llm_client.py:28: in <module>
    from config import LLM_CONFIG, ensure_api_keys
E   ImportError: cannot import name 'ensure_api_keys' from 'config'
    (/home/runner/work/medix-agent-swarm/medix-agent-swarm/config.py)
```

**根因**

`config.py` 含密钥，被 `.gitignore` 忽略，从来不在版本控制里。CI 只能靠 `cp config.example.py config.py` 现造一份。
本地 `config.py` 在演进中新增了 `ensure_api_keys()`（启动 fail-fast）与 `.env` 自动加载，`config.example.py` 没跟着改 —— 于是**公开模板与代码的「导入契约」断裂**。
更麻烦的是：这个漂移在本地**没有任何路径能暴露**，也没有任何 diff/评审会提醒你（被忽略的文件不会出现在 PR 里）。

**解决**

- `config.example.py` 补齐 `ensure_api_keys()` 与 `load_dotenv()`；
- 顺手纠正一处相反的口径：模板里 `CONSTRAINT_ENFORCE = False`（= warn 模式），而项目文档与代码的默认口径是**硬拦 enforce** —— 新克隆按模板跑会得到更弱的约束模式；
- 明确一条规则：**示例配置是"新克隆能否跑起来"的唯一真源，必须与代码导入需求全量对齐**（`LLM_CONFIG` / `ensure_api_keys` / `CONSTRAINT_ENFORCE` / `MEM0_CONFIG`）。

**启示（面试口径）**

> 「被 gitignore 的私有配置，在 CI 里必须有一个**全量对齐**的模板来顶上。我们那次是把 `ensure_api_keys` 加进了私有的 `config.py` 却忘了同步模板——本地永远绿，CI 第一步就死。**CI 是唯一能发现这种漂移的地方。**」

---

## Bug 2：Linux CI 与本地环境的依赖鸿沟（`milvus-lite`）

**现象**

CI 上 `TestKnowledgeBase` 三个用例在 setup 阶段全挂：

```text
pymilvus/client/connection_manager.py:102
E   ModuleNotFoundError: No module named 'milvus_lite'
    message=f"Open local milvus failed, dir: {parent} not exists"
```

**根因**

`pymilvus` 只是客户端；**本地文件模式（Milvus Lite）是另一个独立包 `milvus-lite`**，`pip install pymilvus` 不会替你装上。
本地原本能用，是因为这台机器上恰好装过它 —— 但 `requirements.txt` 里从来没声明。

**解决**

- `requirements.txt` 显式声明 `milvus-lite>=2.4.0`；
- **踩坑中的坑**：我第一版加了平台标记 `; sys_platform != "win32"`（凭"Milvus Lite 只支持 Linux/macOS"的旧印象），随后核实发现它在 PyPI 上是 **`py3-none-any` 纯 Python 轮子**（本地 Windows 跑的就是它），标记是错的，会让 Windows 用户重演同一个 bug —— 已去掉。

**启示（面试口径）**

> 「依赖清单里要区分'我的机器上恰好有'和'声明式依赖'。CI 是唯一的全新环境，它会立刻把这类'隐性前提'打出来。顺带一个反例：我一开始凭旧印象给这个包加了 Windows 排除标记，一查 PyPI 是纯 Python 包——**凭印象写依赖和凭印象写代码一样危险**。」

---

## Bug 3：Pytest 8.4+ 异步判定变更造成的「假性崩塌」

**现象**

- 本地：`37 passed, 26 skipped`（一片祥和）。
- CI：同一命令 → `15 failed`，典型报错：

```text
FAILED examples/test_all.py::test_agent_loop_simple_question
    - Failed: async def functions are not natively supported.
```

**根因（两层）**

1. `examples/test_all.py` 里的用例**不是单测**：它们 `await ConsultationAgent().process(...)`，会真的调外部 LLM、真的等网络返回。CI 没有密钥，注定失败；有密钥则会每次烧钱。
2. **pytest 8.4+ 行为变更**：未打 marker 的裸 `async def` 用例，从「警告后跳过」改成「直接判失败」。本地 pytest 8.3.4 + pytest-asyncio 0.25.0 正好把它们静默跳过 —— 所以之前的"本地全绿"是假象：**那 15 个用例从来没真正执行过**。

**解决**

- 新增 `pytest.ini`：

  ```ini
  [pytest]
  addopts = -m "not live"
  markers =
      live: 需要真实 LLM API 与网络的端到端用例（CI 不跑，本地用 pytest -m live 运行）
  testpaths = examples
  ```

- `examples/test_all.py` 模块级打标：`pytestmark = pytest.mark.live`；
- 工作流**显式**安装 `pytest pytest-asyncio`：不让 CI 行为取决于"某个依赖顺手带上的插件"；
- 新基线：`37 passed, 1 skipped, 25 deselected`（CI 内实测 14.4s）。

**启示（面试口径）**

> 「我们把测试明确分成两层：**离线单测 + 轻量 smoke 门禁**进 CI（无密钥、无网络、一分钟内），**live 端到端**只在发布打 tag 时人工触发。关键在于'被静默跳过的用例等于没写'——那 15 条以前一直是 skip 状态，我们以为自己有覆盖，其实没有；现在它们有了 `live` 标签，想跑就 `pytest -m live`，而且 CI 的行为不再随 pytest 版本漂移。」

---

## Bug 4：前端「真假动画」之争与 Span 记录器幂等性改造

**现象**

要让前端展示「任务分解中 → 多专家并行执行中 → 综合汇总中」的协作胶囊，有两条路：

- 前端 `setTimeout` 假动画：**节奏与被测系统无关**。实测 Swarm 首字 12.9s、单 Agent 3.4s，假动画根本对不上真实阶段；
- 后端推真实阶段：但 API 层先建了 Span 记录器后，`process_with_swarm()` 里的 `begin_observability()` 会**再建一个**并覆盖 ContextVar —— 于是 SSE 生成器（父上下文）读到的永远是那个空记录器，阶段永远停在「请求已接收」。

**根因**

1. `begin_observability()` 不幂等，多入口调用时后建的会覆盖先建的；
2. Span 是用 `span_recorder.mark(...)` 在**阶段结束时**打点的，所以它能给出的语义是「最后一个已完成的阶段 + 此刻正在做的动作」，而不是"进度条"。

**解决**

1. `begin_observability()` 改为**幂等**：同一请求上下文已建过账本则复用同一对对象；
2. API 层在 `asyncio.create_task` **之前**建账本 —— 子任务复制该上下文，下游（coordinator → worker → agent_loop）写进的是同一个记录器；
3. SSE 生成器按 0.4s 轮询 Span 快照，用优先级映射阶段（`Lead_Synthesize` → `Worker_Pool_Execution` → `Route_Decompose` → `Request_Root`），**只在阶段变化时**推 `status` 事件：

   ```python
   # 复用同一个 getter task：wait_for 在超时取消时可能丢掉恰好到达的 delta
   if getter is None:
       getter = asyncio.create_task(queue.get())
   done, _ = await asyncio.wait({getter}, timeout=0.4)
   if not done:
       st = _status_from_spans(spans, t0)
       if st and st["phase"] != last_phase:
           last_phase = st["phase"]
           yield _sse("status", st)
       continue
   ```

4. 前端只消费 `status` 事件点亮胶囊（`synthesize_pending` 与 `synthesize` 同属第 3 阶段）。

**实测**：单 Agent 路由 `route 3.13s → workers 3.46s`；Swarm 双专家 `route 2.87s → workers 12.89s → 汇总 12.89s`，与后端 Span 一一对应。

**启示（面试口径）**

> 「我们**不做前端伪动画**。协作胶囊由后端真实 Span 驱动：API 层和下游复用同一个 Span 记录器（为此把初始化改成幂等），阶段变化时才推自定义 SSE `status` 事件。所以胶囊的节奏就是后端真实运行时的节奏——这个动效可以当证据用，不是装饰。」

---

## 附录（待决策）：`research_agent` 内部标记的出口排版修剪

**问题**：单 Agent 路由下，`research_agent` 会把内部结构直出给用户，例如 `--- # 【综合评估】`、`【文献检索结果】关键词：… 找到相关文献：X 篇`。

**来源**：这些标记不是模型幻觉，是 `agents/research_agent.py` 里 system prompt 的「**输出格式**」明文规定的（`【文献检索结果】` / `【证据摘要】` / `【综合评估】`）。它本质是**给 LeadAgent 消费的证据结构**，在 Swarm 路径下正确；只有单 Agent 直出时才是瑕疵。

**三个选项**

| 方案 | 做法 | 代价 / 风险 |
| --- | --- | --- |
| A（推荐） | **出口层**做白名单式排版修剪：仅当判定为单 Agent 直出时，剥掉「编排头」（如 `【文献检索结果】关键词：… 找到相关文献：N 篇`）与裸 `---`，保留 `【证据摘要】`/`【综合评估】` 的正文（患者能看懂的证据与结论）；并在 CI 门禁加一条断言：构造带标记的答案 → 断言修剪后不再含「找到相关文献」 | 需改出口逻辑 + 补断言；要防止误删正常内容（白名单 + 只在匹配到具体形态时才动） |
| B | 只在前端正则去标记 | 零后端风险，但脏数据仍在 SSE `done.answer` 与日志里，对 API 消费者无效，且属于「用展示层掩盖问题」 |
| C | 改 ResearchAgent 的输出格式说明为面向患者的最终答案 | 最"正确"，但会改变 Swarm 内证据结构，`agent_eval` 的 exact 91% 等指标需整体重跑；风险最大 |

**我的建议**：走 A，但**先不动**——它属于产品/口径调整，应和一次评测复跑一起做（避免"改了文案、指标失去可比性"）。
