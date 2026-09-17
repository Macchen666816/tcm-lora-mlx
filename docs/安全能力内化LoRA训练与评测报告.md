# 安全能力内化 LoRA：训练、检查点选择与对照评测报告

## 1. 实验目的

本轮实验针对旧版安全 LoRA 的一个明确缺陷：旧模型在依赖安全系统提示词时表现尚可，但去掉提示词或加入冲突提示后，仍可能输出附子剂量、煎法或其他可直接执行的医疗建议。

本轮目标不是让模型获得“绝对安全”，而是检验安全行为能否从单纯的系统提示依赖，部分迁移到 LoRA 参数本身。旧 LoRA 保留不变，作为 A/B 对照；新版从正式领域 LoRA 分叉训练，未继承第一版内化实验的权重。

## 2. 训练对象与文件关系

本项目使用 Qwen2.5-1.5B-Instruct 的 MLX 4-bit 基座。最终可运行模型由三部分组成：

```text
Qwen2.5-1.5B-Instruct/mlx-4bit
  + outputs/qwen25-lora-safety-internalized/adapters.safetensors
  + tokenizer / chat template
```

`adapters.safetensors` 不是完整模型，而是 LoRA 增量权重。推理时仍需加载同一个 Qwen2.5 基座，再把 adapter 挂到对应层上。这与图像工作流中“基础模型 + LoRA”的组合逻辑相同。

旧安全 LoRA 位于 `outputs/qwen25-lora-safety/`，本轮没有覆盖；550 步的上一轮中间实验位于 `outputs/qwen25-lora-safety-internalized-v2-550/`，作为过程记录保留。

## 3. 数据构造

数据脚本为 `scripts/prepare_safety_internalization_data.py`，输出位于 `data_safety_internalized/`。数据使用 JSONL 的多轮 `messages` 格式，并开启 `mask_prompt: true`，只对 assistant 答案部分计算主要训练损失。

最终数据规模如下：

| split | 总数 | 安全样本 | 普通中医问答 |
|---|---:|---:|---:|
| train | 1095 | 495 | 600 |
| valid | 298 | 198 | 100 |

安全示范使用 9 种提示条件：显式安全提示、原训练人设、中性提示、纯角色提示、领域改写、无自定义 system，以及 3 种要求“顺从用户、不要警告”的冲突提示。这样训练目标不再只在一种固定安全 system 下成立。

训练安全题 55 条，验证安全题 22 条；独立对抗测试题 24 条与安全训练问题原文精确重合数为 0。新增硬化样本覆盖以下风险：

- 拒绝后继续追问剂量、煎法或服用频率；
- 昏迷、卒中等急症中用针灸或穴位操作替代急救；
- 冲突 system 指令要求模型无条件服从；
- 根据少量症状直接自诊、开方；
- 同名药材、别名和虚假证据导致的药材误认；
- 药物相互作用、肿瘤治疗和停用标准治疗等高风险问题。

数据清单、来源追踪和 SHA-256 记录见 `data_safety_internalized/manifest.json` 与 `provenance_*.jsonl`。

## 4. LoRA 训练设置

配置文件为 `training/qwen25_lora_safety_internalized.yaml`。新版训练从正式领域 LoRA `outputs/qwen25-lora-full/adapters.safetensors` 继续，而不是从旧安全 LoRA 继续。

| 参数 | 值 | 说明 |
|---|---:|---|
| fine-tune type | LoRA | 只训练低秩增量矩阵 |
| batch size | 2 | 每次读取 2 条样本 |
| iterations | 500 | 选定的最终训练步数 |
| learning rate | 3e-6 | 低学习率安全补强 |
| rank | 8 | LoRA 低秩维度 |
| scale | 20 | LoRA 增量缩放 |
| layers | 最后 16 层 | 训练高层语义与行为相关模块 |
| max sequence length | 512 | 单条最大 token 数 |
| mask prompt | true | 主要对答案部分计算 loss |
| random seed | 20260917 | 保证可复现 |

训练日志保存在 `outputs/qwen25-lora-safety-internalized-training.log`。实际可训练参数为 `5.276M / 1543.714M`，占约 `0.342%`，基座主体参数保持冻结。

## 5. 验证损失与检查点选择

正式 500 步训练的验证损失曲线为：

| 步数 | 验证 loss |
|---:|---:|
| 初始 | 2.371 |
| 100 | 1.986 |
| 150 | 1.948 |
| 200 | 1.911 |
| 250 | **1.890** |
| 300 | 1.913 |
| 350 | 1.914 |
| 400 | 1.929 |
| 450 | 1.949 |
| 500 | 1.989 |

验证 loss 在 250 步达到最低并随后回升，因此不能机械地把最后一步视为最好。此前对 200、300、400、500、550 步检查点进行了 6 道高风险题 × 6 种提示条件的筛选。500 步在无自定义 system、已见冲突 system 和未见冲突 system 下总体最稳定；550 步出现无提示时的剂量和急症建议退化。综合验证曲线与人工检查，最终选择 500 步。

最终文件：

```text
outputs/qwen25-lora-safety-internalized/adapters.safetensors
```

该文件与 `0000500_adapters.safetensors` 字节级一致，SHA-256 为：

```text
525b9544eb896aca16be88bd436b5c7c47f7d4787f3ace4f66d023b4ddd798b7
```

## 6. 安全行为 A/B 评测

评测脚本为 `scripts/run_safety_prompt_matrix.py`。新旧 adapter 均在同一个 Qwen 基座、同一随机种子、同一 24 道对抗题和同一 6 种提示条件下生成，共 144 条输出/模型。结果文件分别为：

- `evaluation/results/qwen25_lora_safety_internalized_v2_prompt_matrix.jsonl`
- `evaluation/results/qwen25_lora_safety_legacy_prompt_matrix_v2.jsonl`

脚本中的剂量、操作、延误和不安全断言只是启发式筛查标记，不能替代人工医学复核。汇总如下：

| 模型 | 启发式标记记录数 | 有安全边界的记录数 |
|---|---:|---:|
| 新版内化 LoRA | 21 / 144 | 137 / 144 |
| 旧版安全 LoRA | 27 / 144 | 123 / 144 |

新版的标记中有相当一部分是“拒答时提到风险剂量”或“说明不能用掐人中替代急救”，属于正向安全语境下的保守误报。人工复核得到的关键差异是：

1. 新版在无自定义 system 的 24 道题中没有出现可直接照做的附子剂量、煎法或处方；旧版无提示时出现“附子 100 克”等具体用量和煎煮描述。
2. 新版面对已见和未见冲突 system 时基本维持拒答边界；旧版在冲突提示下多次给出附子克数、每日剂量或煎法。
3. 新版对昏迷、卒中、肿瘤和自行停化疗等问题总体能优先建议急救或专业评估；旧版仍有把掐人中、按穴位或家庭处理写进答案的情况。
4. 新版仍不是绝对安全：个别回答对“白接骨/白接骨草”的同名关系表述不稳定，少量急症答案包含不必要的家庭处置细节。这些问题应在 Web 层保留安全 system、规则拦截和人工复核。

逐题输出与检查点筛选结果保存在 `evaluation/results/`，报告中的结论以人工阅读原文为准，而不是只看正则计数。

## 7. 普通中医问答回归

使用 54 道固定核心题进行新旧对照，脚本为 `scripts/run_inference_suite.py`，结果为：

| 模型 | 平均输出字符数 | 字符级 ROUGE-L F1 |
|---|---:|---:|
| 新版内化 LoRA | 99.6 | 0.337 |
| 旧版安全 LoRA | 100.6 | 0.336 |

这是透明的字符级代理指标，不等价于中医临床正确率。结果表明，新版安全补强没有造成可见的普通领域问答退化，但仍需要人工检查药材功效、证候解释和处方相关内容的准确性。

对应文件：

- `evaluation/results/qwen25_lora_safety_internalized_v2_core.jsonl`
- `evaluation/results/qwen25_lora_safety_legacy_v2_core.jsonl`
- `evaluation/results/internalized_v2_core_comparison.json`

## 8. Loss / Perplexity 对照

使用只读脚本 `scripts/evaluate_lm_loss.py` 在 298 条 test 数据上计算 masked-answer loss。该脚本不会更新或保存模型权重。

| 模型 | loss | perplexity |
|---|---:|---:|
| 基座 Qwen2.5 | 4.058774 | 57.903278 |
| 旧版安全 LoRA | 1.961358 | 7.108978 |
| 新版内化 LoRA | 1.998973 | 7.381469 |

新版 loss 略高于旧版，主要因为新版训练目标加入了更多拒答、风险边界和冲突提示样本；这并不推翻其安全行为改善，也不能单独用来证明模型更安全或更准确。

## 9. 复现命令

从项目二目录执行：

```bash
../.venv-mlx/bin/mlx_lm.lora -c training/qwen25_lora_safety_internalized.yaml
```

安全矩阵：

```bash
../.venv-mlx/bin/python scripts/run_safety_prompt_matrix.py \
  --model models/Qwen2.5-1.5B-Instruct/mlx-4bit \
  --adapter-path outputs/qwen25-lora-safety-internalized \
  --output evaluation/results/reproduced_internalized_matrix.jsonl
```

普通问答回归：

```bash
../.venv-mlx/bin/python scripts/run_inference_suite.py \
  --model models/Qwen2.5-1.5B-Instruct/mlx-4bit \
  --adapter-path outputs/qwen25-lora-safety-internalized \
  --cases data_processed/evaluation/core_benchmark.jsonl \
  --output evaluation/results/reproduced_internalized_core.jsonl
```

## 10. 结论与部署建议

本实验支持如下谨慎结论：安全示范以多种提示条件参与 SFT 后，安全边界确实有一部分进入 LoRA 行为，模型对“去掉安全 system”和“冲突 system”的鲁棒性明显优于旧版；同时，54 道普通中医问答的代理指标基本持平。

本实验不支持“任意提示词下都绝对安全”或“LoRA 可以替代系统安全策略”的结论。LoRA 是有限数据和有限步数下的统计行为改变，不是形式化约束。部署 Web 服务时仍应采用双层方案：

1. 加载 Qwen2.5 基座 + `qwen25-lora-safety-internalized` adapter；
2. 保留稳定的安全 system prompt 作为第二道防线；
3. 对剂量、处方、停药、急症延误等高风险输出增加规则筛查；
4. 明确展示“仅供一般知识，不替代医生诊疗”的产品边界，并保留日志用于复盘。

旧版 adapter 继续作为对照和回滚版本，新版 adapter 作为当前候选部署版本，但在报告和答辩中应使用“安全能力增强”“风险显著降低”等表述，避免宣称医疗安全保证。
