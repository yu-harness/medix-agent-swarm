# 智能路由标注规范

> 用途：为「LeadAgent 路由」建立可终审的金标草案。  
> 预标文件：`route_labels_200.jsonl`（规则预标，**≠** 最终金标）。  
> 与系统对齐：`swarm/swarm_coordinator.py` + `swarm/lead_agent.py`。

---

## 1. 标签定义

### 1.1 模式 `expected_mode`

| 值 | 含义 | 与系统对应 |
|---|---|---|
| `single` | 只需 1 个子任务 / 1 个 Worker | `len(subtasks)==1` → `single_agent` |
| `swarm` | 需要 ≥2 个子任务 / 多 Worker 协作 | `len(subtasks)>=2` 且 `enable_swarm` → `swarm` |

系统不按「难度分数」硬切，而是看 **LeadAgent 分解出的子任务个数**。标注时问自己：这个问题是否**值得并行两个以上专业角色**？值得 → `swarm`，否则 → `single`。

### 1.2 Agent 集合 `expected_agents`

集合语义，**顺序无关**；落盘时用排序后的 list（按 `consultation` < `diagnostic` < `research`）。

| 标注名 | 系统 `assigned_agent` | 能力边界（与 Lead 提示词一致） |
|---|---|---|
| `consultation` | `consultation_agent` | 常见病科普、初步风险、生活/饮食/睡眠建议、日常健康管理 |
| `diagnostic` | `diagnostic_agent` | 多症状关联、鉴别推理、ICD/分类、复杂风险分级 |
| `research` | `research_agent` | 临床指南/共识、标准治疗证据、最新研究与循证检索 |

合法示例：

- `["consultation"]`
- `["diagnostic"]`
- `["research"]`
- `["consultation","diagnostic"]`
- `["consultation","research"]`
- `["diagnostic","research"]`
- `["consultation","diagnostic","research"]`（少用；能 2 个解决不要 3 个）

---

## 2. 书面规则（与系统能力对齐）

原则（同 LeadAgent）：**尽量少分配**；常见感冒发烧优先单 `consultation`；复杂症状才上 `diagnostic`；纯指南检索才上 `research`。

### 2.1 `health_consult`

| 规则 ID | 条件 | 预标 |
|---|---|---|
| `R_HC_DEFAULT` | 常规咨询、单一生活建议，无急救/多主诉 | `single` + `[consultation]` |
| `R_HC_EMERGENCY` | 含胸痛/胸闷/呼吸困难/冷汗/晕厥/中风偏瘫/急救等 | `swarm` + `[consultation,diagnostic]` |
| `R_HC_MULTI` | 多症状或明显共病（如脑梗+糖尿病+高血压） | `swarm` + `[consultation,diagnostic]` |
| `R_HC_TREAT_GUIDE` | 慢病「如何治疗/救治」且需规范方案感 | `swarm` + `[consultation,research]` |

**例**

- 「高血压能吃黑米吗？」→ `single` + `[consultation]`（`R_HC_DEFAULT`）
- 「脑梗糖尿病高血压怎么办」→ `swarm` + `[consultation,diagnostic]`（`R_HC_MULTI`）
- 「喉癌晚期喉溃烂怎么办」→ `swarm` + `[consultation,diagnostic]`（`R_HC_EMERGENCY`）

### 2.2 `symptom_diagnosis`

| 规则 ID | 条件 | 预标 |
|---|---|---|
| `R_SD_EMERGENCY` | 胸痛/气短/冷汗等急症线索 | `swarm` + `[consultation,diagnostic]`（**至少含 diagnostic**） |
| `R_SD_MULTI` | 复杂多主诉（≥3 类症状词，或 ≥2 且有并列连接） | `swarm` + `[consultation,diagnostic]` |
| `R_SD_DIAG_ONLY` | 侧重「原因/是不是某病/鉴别」且非急症多主诉 | `single` + `[diagnostic]` |
| `R_SD_AS_CONSULT` | 题面实为饮食/用药咨询 | `single` + `[consultation]` |
| `R_SD_DEFAULT` | 其余单一症状常规问诊 | `single` + `[consultation]` |

**例**

- 「胸口闷疼出汗…」→ `swarm` + `[consultation,diagnostic]`（`R_SD_EMERGENCY`）
- 「感冒咳嗽胸部麻木15天是怎么回事」→ `swarm` 或 `single+[diagnostic]`：多症状+追因 → 预标走 `R_SD_MULTI` / `R_SD_DIAG_ONLY`（以脚本命中为准，终审可改）
- 「糖尿病人能喝茯苓水吗」→ `single` + `[consultation]`（`R_SD_AS_CONSULT`）

### 2.3 `disease_knowledge`

| 规则 ID | 条件 | 预标 |
|---|---|---|
| `R_DK_DEFINE` | 定义/是什么/是否传染/药理/成分 | `single` + `[consultation]` |
| `R_DK_CLINICAL` | 症状/病因/并发症结构化描述 | `single` + `[diagnostic]` |
| `R_DK_CODING` | ICD/编码/分类 | `single` + `[diagnostic]` |
| `R_DK_RESEARCH` | 最新研究/循证/指南进展 | `single` + `[research]` |
| `R_DK_TREAT` | 一般「怎么治」科普 | `single` + `[consultation]` |
| `R_DK_TREAT_LIFE` | 治疗 + 明确生活管理 | `swarm` + `[consultation,research]` |
| `R_DK_DEFAULT` | 其余科普 | `single` + `[consultation]` |

**例**

- 「禽流感是什么？」→ `single` + `[consultation]`
- 「白血病的并发症」→ `single` + `[diagnostic]`
- 「高血压最新诊疗指南」→ `single` + `[research]`

### 2.4 `guideline_retrieval`

| 规则 ID | 条件 | 预标 |
|---|---|---|
| `R_GR_PURE` | 「检索文档/权威资料/指南要点」类纯查 | `single` + `[research]` |
| `R_GR_GUIDE_LIFE` | 既要指南要点，又要个人生活建议 | `swarm` + `[consultation,research]` |

说明：指南章节标题含「运动/膳食」但题面仍是「请根据知识库/指南文档说明…」→ 仍算 **纯查指南**（`R_GR_PURE`），不要因章节名误标 swarm。

---

## 3. 字段说明（`route_labels_200.jsonl`）

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | string | 与 `benchmark_500` / `agent_eval_detail_full200` 对齐 |
| `question` | string | 原问题 |
| `category` | string | 四类之一 |
| `expected_mode` | `single` \| `swarm` | 期望路由模式 |
| `expected_agents` | string[] | 排序后的 Agent 集合 |
| `prelabel_rule` | string | 命中的规则 ID（见上表） |
| `notes` | string | 规则说明（给终审看） |
| `review_status` | string | 初始 `pending`；终审后改为 `confirmed` / `revised` |

终审改标时：改 `expected_mode` / `expected_agents`，把 `review_status` 设为 `revised`，可在 `notes` 追加「终审：…」。

---

## 4. 如何终审

1. **先抽 30 条**：每类约 7–8 条，优先抽 `swarm`、急症词、边界题（症状类题面像咨询）。
2. **只问两个问题**：  
   - 一个 Agent 是否够？够 → `single`。  
   - 需要谁？对照上表能力边界，集合越小越好。
3. **不要按「答得全不全」标 swarm**：金标是「合理路由」，不是「答案要最长」。
4. **与系统观测对照（可选）**：`agent_eval_detail_full200.jsonl` 的 `mode` 是系统实际路由，**不是**金标；分歧处重点人工看。
5. **全量改完**：`review_status` 尽量全部非 `pending`，再进入路由准确率评测（本专项暂不跑）。

---

## 5. 面试怎么表述

- **评测分层**：  
  1）**模式准确率**：`expected_mode` vs 系统 `single`/`swarm`；  
  2）**组合匹配率**：在模式正确前提下，`expected_agents` 集合是否相等（顺序无关）；可另报「部分匹配」（有交集但不等）。
- **预标 ≠ 金标**：本批是 **category + 关键词确定性规则** 预标，供人工终审；面试应强调「先规则冷启动 → 人工校准 → 再评路由」，而不是「LLM 自标自测」。
- **样本**：与全量 Agent 评测同一批 **seed=42、四类各 50、共 200**，可与 `agent_eval_detail_full200.jsonl` 按 `id` 对齐。
- **不夸大**：当前交付是标注资产，**尚未**改路由代码、**尚未**跑路由准确率全量评测。

---

## 6. 与代码行为的边界

- Coordinator：**只**根据 `subtasks` 数量切 single/swarm，不二次改 Agent。
- LeadAgent：LLM 分解；标注规范描述的是「合理期望」，用于衡量 Lead 是否分对，不是复刻某次 LLM 输出。
- Fallback（0 子任务 / swarm 关闭）→ 系统走 consultation；金标仍按「应有合理路由」标，不把 fallback 标成正确。
