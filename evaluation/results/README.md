# 评测结果索引

本目录保留机器可读原始输出和由脚本生成的汇总。报告结论应优先引用原始 JSONL/CSV，再引用 Markdown 摘要。

## 当前最终实验

| 实验 | 原始数据 | 汇总 |
|---|---|---|
| 2×2 新旧 LoRA 人工复核 | `safety_2x2_manual_review.csv` | `../SAFETY_2X2_REVIEW.md` |
| 2×3 三题先导实验 | `rag_2x3_webapp_20260917-142332.jsonl` | `rag_2x3_webapp_20260917-142332.md` |
| 2×3 134 题全量实验 | `rag_2x3_webapp_20260917-144422.jsonl` | `rag_2x3_webapp_20260917-144422.md` |
| 全量 NLP 自动筛查 | `rag_2x3_webapp_20260917-144422_nlp.jsonl`、`*_arms.csv` | `*_nlp_summary.json`、`*_nlp_report.md` |

## 核心问答

- `qwen25_base_core.jsonl`：基座 54 道核心题。
- `qwen25_lora_final_core.jsonl`：正式领域 LoRA。
- `qwen25_lora_safety_legacy_v2_core.jsonl`：旧版安全 LoRA。
- `qwen25_lora_safety_internalized_v2_core.jsonl`：新版安全内化 LoRA。
- `internalized_v2_core_comparison.json`：新旧版核心代理指标对比。

## 安全提示矩阵

- `qwen25_lora_safety_legacy_prompt_matrix_v2.*`：旧版最终六提示矩阵。
- `qwen25_lora_safety_internalized_v2_prompt_matrix.*`：新版最终六提示矩阵。
- `qwen25_lora_safety_2x2.*`、`qwen25_lora_internalized_2x2.*`：2×2 原始输出。
- 不带 `v2` 的 `*_prompt_matrix.*` 是较早阶段产物，保留用于历史追溯。

## 检查点筛查

`internalized*_checkpoint_*_screen.*` 保存不同训练步数的高风险题筛查。它们参与了工程检查点选择，不属于独立测试集。

## Loss 文件的使用限制

`base_test_loss_v2.json`、`legacy_safety_test_loss_v2.json` 和 `internalized_v2_test_loss.json` 不能直接组成严格的三模型横向比较：当前脚本对基座与 adapter 使用了不同 prompt mask，且安全内化数据的 `test` 与 `valid` 相同。详细说明见 [`../../docs/项目全流程复盘与最终报告.md`](../../docs/项目全流程复盘与最终报告.md)。
