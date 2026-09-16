# 中医数据集自动审计报告

> 本报告由确定性规则生成。原始数据未被修改；`qa_candidate` 只是候选标记，不代表医学事实已经核验。

## 总览

- 总记录数：7036
- 分层抽样复核记录数：492
- 清洗原则：先分类和标记，再人工复核；不按单一长度阈值批量删除。

| 划分 | 原始记录 | 问答候选 | 硬排除候选 | 需重点复核 |
|---|---:|---:|---:|---:|
| train | 5628 | 4023 | 230 | 1776 |
| test | 1408 | 1033 | 59 | 471 |

## 任务类型

| 类型 | train | test |
|---|---:|---:|
| `extraction_classification` | 442 | 109 |
| `knowledge_qa` | 4236 | 1089 |
| `long_form_generation` | 398 | 82 |
| `rewriting_editing` | 104 | 21 |
| `title_generation` | 448 | 107 |

## 风险标记

| 标记 | train | test |
|---|---:|---:|
| `document_style_output` | 61 | 15 |
| `document_style_task_mismatch` | 23 | 5 |
| `dosage_question` | 8 | 1 |
| `dosage_without_safety_language` | 7 | 0 |
| `duplicate_output` | 12 | 0 |
| `heading_only_output` | 149 | 36 |
| `heading_only_task_mismatch` | 92 | 20 |
| `latin_text_fragment` | 37 | 10 |
| `malformed_instruction` | 8 | 2 |
| `medical_advice` | 993 | 250 |
| `medical_advice_without_safety_language` | 780 | 197 |
| `personal_medical_advice` | 396 | 112 |
| `personal_medical_advice_without_safety_language` | 298 | 87 |
| `placeholder_output` | 1 | 0 |
| `potentially_toxic_herb` | 73 | 23 |
| `serious_condition` | 127 | 43 |
| `serious_condition_without_safety_language` | 91 | 34 |
| `short_output` | 1000 | 228 |
| `title_only_output` | 474 | 111 |
| `title_only_task_mismatch` | 108 | 32 |
| `toxic_herb_without_safety_language` | 67 | 22 |
| `unsafe_absolute_claim` | 4 | 1 |
| `unverified_research_claim` | 776 | 200 |
| `very_short_output` | 60 | 12 |
| `vulnerable_population` | 96 | 31 |
| `vulnerable_population_without_safety_language` | 66 | 19 |

## 文件说明

- `audit_records.jsonl`：全部记录及审计元数据。
- `qa_candidates_train.jsonl` / `qa_candidates_test.jsonl`：保守筛出的知识问答候选，尚未完成医学事实核验。
- `hard_exclusions.jsonl`：格式损坏、占位回答或明显标题错配等候选。
- `review_queue.jsonl`：短回答、医疗建议、特殊人群、有毒药材等重点复核样本。
- `review_sample.jsonl`：按任务类型与风险标记生成的可复现分层样本。
- `review_sample.csv`：同一批分层样本的人工复核表，可填写 `decision` 和 `review_notes`。
- `summary.json`：机器可读统计。

## 解释边界

规则只能识别结构和关键词风险，不能判断中医知识是否正确。正式训练前仍需对高风险样本人工或借助可靠知识源复核。
