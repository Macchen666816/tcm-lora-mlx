# 基于 LoRA 微调的中医问答模型

一个可复现的课程实训项目：在 Apple Silicon 上使用 MLX-LM 对 `Qwen2.5-1.5B-Instruct` 进行 4-bit LoRA 微调，并通过固定核心题、安全对抗题和本地 Web 应用比较微调前后的表现。

> 本项目仅用于课程实验，不构成医疗建议，也不能用于诊断、开方或剂量决策。数据和模型回答尚未经过中医专业人员逐条审核。

## 当前结果

| 项目 | 结果 |
|---|---:|
| 清洗后 train / valid / test | 3061 / 341 / 850 |
| 正式训练 | 1531 步，约 2 epoch |
| 最终测试 loss / perplexity | 1.369 / 3.931 |
| 54 道核心题字符级 ROUGE-L | 0.208 → 0.336 |
| 核心题平均回答长度 | 332.1 → 100.6 字 |
| 安全测试可执行高危方案 | 0 / 24（带部署安全提示） |

字符级 ROUGE-L 只是开放式问答的辅助指标，不能代表医学正确率。完整限制和逐题结果见 [`evaluation/RESULTS.md`](evaluation/RESULTS.md)。

## 仓库内容

```text
data/                    原始课程数据
data_audit/              全量审计记录
data_processed/          清洗后的训练数据、追溯关系和数据卡
data_safety_alignment/   与测试题隔离的安全补强数据
docs/                    LoRA 原理、复现步骤和答辩讲解
evaluation/              核心题、安全题和逐题模型输出
outputs/                 最终 LoRA 与训练日志
scripts/                 数据、模型准备和评测脚本
training/                正式训练与安全续训配置
webapp/                  基座 / LoRA 并排推理界面
```

基座模型权重和 Python 虚拟环境不纳入 Git。仓库保留约 20 MB 的最终 LoRA：

```text
outputs/qwen25-lora-safety/adapters.safetensors
SHA-256: fc3020144a5ac233df777dc917b4a8bc0a74c851b5ad1b36a14b855ce4e177e3
```

## 环境要求

- Apple Silicon Mac
- Python 3.13（本项目实测 3.13.12）
- 建议至少 16 GB 统一内存
- 约 5 GB 可用磁盘空间用于原始和 4-bit 基座

```bash
python3 -m venv .venv-mlx
source .venv-mlx/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 准备基座模型

脚本从 ModelScope 官方 Qwen 仓库下载原始权重，核验 SHA-256，并转换为 MLX 4-bit：

```bash
.venv-mlx/bin/python scripts/prepare_base_model.py
```

输出目录：

```text
models/Qwen2.5-1.5B-Instruct/hf
models/Qwen2.5-1.5B-Instruct/mlx-4bit
```

## 验证数据

```bash
.venv-mlx/bin/python scripts/validate_processed_data.py
```

预期核心统计：train 3061、valid 341、test 850、核心题 54、安全题 24。

## 复现训练

正式 LoRA：

```bash
.venv-mlx/bin/mlx_lm.lora -c training/qwen25_lora_full.yaml
```

安全数据制备与低学习率续训：

```bash
.venv-mlx/bin/python scripts/prepare_safety_alignment_data.py
.venv-mlx/bin/mlx_lm.lora -c training/qwen25_lora_safety.yaml
```

详细原理、参数解释和答辩表述见 [`docs/LoRA训练与部署讲解.md`](docs/LoRA训练与部署讲解.md)。

## 启动 Web 对比界面

```bash
.venv-mlx/bin/python webapp/server.py --port 8088
```

浏览器访问 [http://127.0.0.1:8088](http://127.0.0.1:8088)。服务真实加载两套运行实例：

- `Qwen2.5-1.5B-Instruct` 4-bit 基座；
- 相同基座 + 最终安全 LoRA。

两边使用相同问题、安全系统提示、温度和最大输出长度，区别只有是否加载 LoRA。第一次调用会加载权重，之后复用内存中的模型。

## 关键文档

- [`项目交接说明.md`](项目交接说明.md)：完整决策记录和当前进度。
- [`data_processed/DATA_CARD.md`](data_processed/DATA_CARD.md)：数据范围、清洗规则和限制。
- [`data_audit/REPORT.md`](data_audit/REPORT.md)：原始数据审计。
- [`evaluation/RESULTS.md`](evaluation/RESULTS.md)：训练与评测结果。
- [`docs/LoRA训练与部署讲解.md`](docs/LoRA训练与部署讲解.md)：原理、命令和答辩准备。

## 已知限制

- 医学事实未经专业人员逐条审核。
- LoRA 依赖完全匹配的 Qwen2.5-1.5B 基座，不能直接套用到 Qwen3 或其他尺寸模型。
- 安全提示和补强训练降低了高危输出，但不能提供数学上的绝对安全保证。
- 原始课程数据的公开再分发许可尚未确认，因此建议保持仓库私有，仅用于组内协作。
