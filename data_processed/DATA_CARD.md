# 中医问答训练数据卡

> 生成日期：2026-09-15。该数据包用于课程实验，不构成医疗知识库或诊疗依据。

## 范围

最终数据只保留中医知识问答。标题生成、长文写作、术语改写、抽取分类等混合任务被移出训练范围。
原始回答未被自动改写；规则无法证明医学事实正确，因此保留了来源追踪和抽检表。

## 最终划分

| 划分 | 条数 | 输出最短 | 输出中位 | 输出 P90 | 输出最长 |
|---|---:|---:|---:|---:|---:|
| train | 3061 | 16 | 106 | 179 | 602 |
| valid | 341 | 22 | 109 | 175 | 334 |
| test | 850 | 22 | 106 | 174 | 354 |

- `train` 与 `valid` 均来自原始训练集，并按主题做确定性 9:1 划分。
- `test` 只来自原始测试集，没有进入训练或验证集。
- 三份数据均已做规范化去重；测试集与训练/验证集无重复问题。

## 排除策略

| 原因 | 条数 |
|---|---:|
| `dedup:instruction` | 11 |
| `dedup:output` | 2 |
| `hard:document_style_task_mismatch` | 28 |
| `hard:heading_only_task_mismatch` | 112 |
| `hard:malformed_instruction` | 10 |
| `hard:placeholder_output` | 1 |
| `hard:title_only_task_mismatch` | 140 |
| `leakage:instruction_overlap_with_train` | 3 |
| `out_of_scope_task:extraction_classification` | 551 |
| `out_of_scope_task:long_form_generation` | 480 |
| `out_of_scope_task:rewriting_editing` | 125 |
| `out_of_scope_task:title_generation` | 555 |
| `policy:dosage_without_safety_language` | 7 |
| `policy:latin_text_fragment` | 47 |
| `policy:personal_medical_advice_without_safety_language` | 385 |
| `policy:serious_condition_without_safety_language` | 125 |
| `policy:toxic_herb_without_safety_language` | 89 |
| `policy:unsafe_absolute_claim` | 5 |
| `policy:unverified_research_claim` | 976 |
| `policy:very_short_output` | 72 |
| `policy:vulnerable_population_without_safety_language` | 85 |

安全策略会隔离以下回答：针对个人症状给出用药但缺少就医提示、特殊人群缺少安全提示、涉及潜在毒性药材却无警示、剂量问题缺少安全约束。
短于 40 字不是自动删除条件；只有极短回答、占位回答和标题式错配会被排除。

## 文件

- `mlx/*.jsonl`：MLX-LM 对话格式。
- `alpaca/*.jsonl`：保留原始三字段格式，便于其他训练框架使用。
- `lineage.jsonl`：最终样本与原始行号、主题、审计标记的映射。
- `excluded.jsonl`：全部未入选样本及明确排除原因。
- `safety_quarantine.jsonl`：因医疗安全规则被隔离的样本。
- `research_quarantine.jsonl`：要求现代研究、药理或疗效证据但缺少可核查来源的样本。
- `medical_spotcheck.csv`：分层抽检表，供后续人工或专业人员填写。
- `evaluation/core_benchmark.jsonl`：仅从测试集抽取的固定主题基准题，不进入训练。
- `manifest.json`：统计、规则版本和文件校验值。

## 已知限制

自动处理可以解决格式、任务混杂、重复和显式安全措辞问题，但无法系统判断药性、归经、方剂配伍及现代研究结论是否正确。
因此本数据包应称为“规则清洗后的训练候选集”，不能称为“医学专家审核数据集”。
