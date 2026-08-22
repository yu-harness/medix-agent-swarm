# medix-agent-swarm 文本医疗评测集

全中文、仅文本。共 **500** 条，四类各 **125** 条。私有完整集默认不入库；公开仓库仅保留样例。

## 文件

| 路径 | 说明 | Git |
|------|------|-----|
| `data/benchmark_500.jsonl` | 私有完整评测集（500） | 忽略 |
| `data/benchmark_samples.jsonl` | 公开样例（每类 8，共 32） | 提交 |
| `data/benchmark_build_meta.json` | 构建指纹与条数元数据 | 忽略 |
| `raw/` | 原始下载数据 | 忽略 |
| `scripts/build_benchmark_500.py` | 复现构建脚本 | 提交 |

## 字段

每行一条 JSON：

- `id`：如 `health_consult_001`
- `question` / `answer`：中文问答
- `category`：仅允许下列 slug
  - `health_consult` 日常健康咨询
  - `symptom_diagnosis` 症状诊断分析
  - `disease_knowledge` 疾病知识查询
  - `guideline_retrieval` 权威指南/知识文档检索
- `source`：数据来源名或本地文档路径
- `source_id`：源内标识
- `notes`：截断、文档抽问答等备注

## 采用的公开数据源（已核实可下载）

| 数据集 | URL | 许可/用途 | 适配类别 | 本机下载方式 |
|--------|-----|-----------|----------|--------------|
| **cMedQA2** | https://github.com/zhangsheng93/cMedQA2 | 非商业研究；仓库标 GPL-3.0 | health_consult / symptom_diagnosis | `question.zip` + `answer.zip` + `train_candidates.zip` |
| **webMedQA** | https://github.com/hejunqing/webMedQA | Apache-2.0 | 同上 | `valid.zip` / `test.zip`（取 label=1 正答） |
| **Chinese-medical-dialogue-data** | https://github.com/Toyhom/Chinese-medical-dialogue-data | MIT | 同上 | 样例 CSV `样例_内科5000-6000.csv`（GBK） |
| **shibing624/medical** | https://huggingface.co/datasets/shibing624/medical | Apache-2.0 | 咨询/症状/知识 | `finetune/test_zh_0.json`、`valid_zh_0.json` |
| **huatuo_encyclopedia_qa** | https://huggingface.co/datasets/FreedomIntelligence/huatuo_encyclopedia_qa | 见 HF 卡片（华佗百科 QA） | disease_knowledge | HuggingFace datasets-server 抽样 |
| **本地知识文档** | `../knowledge/data/documents/` | 项目内 | guideline_retrieval（不足时用文档片段补齐） | 直接读 txt |

调研中核对但**未整库纳入最终 500** 的源：

- 丁香/春雨等商业站点对话：许可不清晰，未直接抓取。
- Toyhom 全量科室 CSV：体积大；本构建用公开样例 CSV 即可满足配额。

## 最终条数（按类）

| category | 条数 |
|----------|------|
| health_consult | 125 |
| symptom_diagnosis | 125 |
| disease_knowledge | 125 |
| guideline_retrieval | 125 |
| **合计** | **500** |

指南类：优先 `20_guideline_hypertension.txt`、`21_guideline_diabetes.txt`；不足部分由同目录其他知识文档抽「问原文 / 答片段」补齐（`notes` 含 `generated_from_local_doc_span`）。

## 本地查看 / 抽样

```powershell
cd medix-agent-swarm\eval

# 条数与类别统计
python -c "import json; from collections import Counter; from pathlib import Path; rows=[json.loads(l) for l in Path('data/benchmark_500.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]; print(len(rows), Counter(r['category'] for r in rows))"

# 看公开样例
Get-Content .\data\benchmark_samples.jsonl -TotalCount 5 -Encoding utf8

# 随机抽 3 条完整集
python -c "import json,random; from pathlib import Path; rows=[json.loads(l) for l in Path('data/benchmark_500.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]; random.seed(42); [print(json.dumps(r,ensure_ascii=False)[:300]) for r in random.sample(rows,3)]"
```

## 复现构建

```powershell
cd medix-agent-swarm\eval
# 1) 将公开集下载到 raw\（脚本假设目录结构已存在；可用 raw 内现有文件）
# 2) 构建
python .\scripts\build_benchmark_500.py
```

脚本会：筛选非空问答、按规则映射四类、question 近似去重、答案截断至约 800 字（`notes` 标明）、按来源配额取样、写出 `benchmark_500.jsonl` 与 `benchmark_samples.jsonl`。

## 已知局限

1. **许可**：cMedQA2 明确非商业研究；完整集请勿公开分发，仅样例入库。
2. **指南类**：中文临床指南整库公开可下载且可自由再分发的 QA 很少；本集以项目内指南/知识文档抽问答为主，答案为原文片段，便于核对，但覆盖病种有限（高血压、糖尿病为主）。
3. **类别映射**：公开对话/社区 QA 靠关键词启发式归类，边界题可能存在交叉。
4. **医学正确性**：标准答案来自公开集或文档片段，未做临床专家二次标注；不可当诊疗依据。
5. **无评测脚本 / 无图像**：本目录只提供文本评测数据。
