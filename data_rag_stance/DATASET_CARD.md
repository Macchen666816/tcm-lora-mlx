# 三立场 RAG 数据集（消融实验用）

- 每立场文档数：**500**，合计 **1500**
- 随机种子：`20260916`（确定性抽样，可复现）
- 生成脚本：`scripts/build_stance_dataset.py`

## aligned（500 条）

| 主题 | 条数 |
|---|---:|
| materia_medica_and_formulas | 296 |
| syndromes_and_theory | 62 |
| clinical_conditions | 56 |
| acupuncture_and_meridians | 48 |
| safety_alignment | 25 |
| general_tcm_knowledge | 11 |
| modern_research | 2 |

| 来源 | 条数 |
|---|---:|
| mlx/test.jsonl | 475 |
| data_safety_alignment/seed_examples.jsonl | 25 |

## ambiguous（500 条）

| 主题 | 条数 |
|---|---:|
| materia_medica_and_formulas | 296 |
| syndromes_and_theory | 62 |
| clinical_conditions | 56 |
| acupuncture_and_meridians | 48 |
| modern_research | 27 |
| general_tcm_knowledge | 11 |

| 来源 | 条数 |
|---|---:|
| data_processed/excluded.jsonl | 500 |

| 剔除原因（立场冲突标签） | 条数 |
|---|---:|
| `out_of_scope_task:extraction_classification` | 165 |
| `out_of_scope_task:long_form_generation` | 144 |
| `out_of_scope_task:title_generation` | 108 |
| `hard:title_only_task_mismatch` | 33 |
| `out_of_scope_task:rewriting_editing` | 30 |
| `hard:heading_only_task_mismatch` | 25 |
| `policy:very_short_output` | 12 |

## opposed（500 条）

| 主题 | 条数 |
|---|---:|
| materia_medica_and_formulas | 316 |
| acupuncture_and_meridians | 56 |
| clinical_conditions | 53 |
| modern_research | 37 |
| syndromes_and_theory | 27 |
| general_tcm_knowledge | 11 |

| 来源 | 条数 |
|---|---:|
| data_processed/safety_quarantine.jsonl | 366 |
| data_processed/research_quarantine.jsonl | 134 |

| 剔除原因（立场冲突标签） | 条数 |
|---|---:|
| `policy:personal_medical_advice_without_safety_language` | 240 |
| `policy:unverified_research_claim` | 134 |
| `policy:serious_condition_without_safety_language` | 69 |
| `policy:vulnerable_population_without_safety_language` | 50 |
| `policy:toxic_herb_without_safety_language` | 21 |
| `policy:dosage_without_safety_language` | 3 |
