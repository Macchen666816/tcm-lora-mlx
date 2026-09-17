# 基于 LoRA 微调的中医问答模型

一个可复现的课程实训项目：在 Apple Silicon 上使用 MLX-LM 对 `Qwen2.5-1.5B-Instruct` 进行 4-bit LoRA 微调，并通过固定核心题、安全对抗题和本地 Web 应用比较微调前后的表现。

> 本项目仅用于课程实验，不构成医疗建议，也不能用于诊断、开方或剂量决策。数据和模型回答尚未经过中医专业人员逐条审核。

## 当前结果

| 项目 | 结果 |
|---|---:|
| 清洗后 train / valid / test | 3061 / 341 / 850 |
| 正式训练 | 1531 步，约 2 epoch |
| 正式领域 LoRA 测试 loss / perplexity | 1.369 / 3.931 |
| 新版安全内化 LoRA 复用验证集 loss / perplexity | 1.999 / 7.381 |
| 54 道核心题字符级 ROUGE-L | 0.208 → 0.337 |
| 新版 LoRA × 10 字中性提示人工复核 | 21 pass / 3 partial / 0 fail |
| 新版 LoRA × 10 字中性提示高风险失守 | 0 / 24 |

字符级 ROUGE-L 只是开放式问答的辅助指标，不能代表医学正确率。安全内化数据的 `test` 与 `valid`
实际相同，且现有脚本对基座和 adapter 使用了不同的 prompt mask，因此相关 loss 不能作严格的三模型横向比较。
完整限制见 [`docs/项目全流程复盘与最终报告.md`](docs/项目全流程复盘与最终报告.md)，逐题结果见
[`evaluation/RESULTS.md`](evaluation/RESULTS.md)。

## 仓库内容

```text
data/                    原始课程数据
data_audit/              全量审计记录
data_processed/          清洗后的训练数据、追溯关系和数据卡
data_safety_alignment/   与测试题隔离的安全补强数据
docs/                    LoRA 原理、复现步骤和答辩讲解
evaluation/              核心题、安全题和逐题模型输出
outputs/                 最终 LoRA 与训练日志
rag_service/             MySQL + FAISS/BM25/RRF 检索微服务
rag_console/             RAG 管理与检索调试界面
scripts/                 数据、模型准备和评测脚本
tests/                   本地自动评分测试
training/                正式训练与安全续训配置
webapp/                  基座 / LoRA 并排推理界面
```

基座模型权重和 Python 虚拟环境不纳入 Git。当前候选部署权重为新版安全内化 LoRA；旧版
`outputs/qwen25-lora-safety/` 原样保留用于对照和回滚：

```text
outputs/qwen25-lora-safety-internalized/adapters.safetensors
SHA-256: 525b9544eb896aca16be88bd436b5c7c47f7d4787f3ace4f66d023b4ddd798b7
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

安全能力内化版使用 9 种 system 条件的 1095 条训练数据，从正式领域 LoRA 分叉续训 500 步：

```bash
../.venv-mlx/bin/python scripts/prepare_safety_internalization_data.py
../.venv-mlx/bin/mlx_lm.lora -c training/qwen25_lora_safety_internalized.yaml
```

详细原理、参数解释和答辩表述见 [`docs/LoRA训练与部署讲解.md`](docs/LoRA训练与部署讲解.md)。

## 启动 Web 对比界面

> 虚拟环境不纳入 Git，本机把它建在**工作区根目录**（`实训项目选题/.venv-mlx`），
> 比仓库根高一层，因此启动时要用「工作区根目录 + 脚本全路径」的方式。

```bash
cd "/Users/chenyixuan/Desktop/Study/大模型实训/实训项目选题"
.venv-mlx/bin/python "项目2：基于LoRA微调的中医医学专家大模型/webapp/server.py" --port 8088
```

浏览器访问 [http://127.0.0.1:8088](http://127.0.0.1:8088)。

如果虚拟环境就在仓库根目录（例如队友 clone 后自己建的），则在仓库根执行：

```bash
.venv-mlx/bin/python webapp/server.py --port 8088
```

服务真实加载三套运行实例：

- `Qwen2.5-1.5B-Instruct` 4-bit 基座；
- 相同基座 + `outputs/qwen25-lora-safety` 旧版安全 LoRA；
- 相同基座 + `outputs/qwen25-lora-safety-internalized` 新版安全内化 LoRA。

一次提问会并发跑出十格，并共用同一份用户输入和回答长度：上方固定 2×3 六格，行是基座 / 新版 LoRA，
列是无 RAG / `aligned` / `opposed`；页面最底部固定 2×2 四格，对比旧版 / 新版 LoRA 在 45 字训练提示
和 10 字中性提示下的输出。第一次调用会加载权重，之后复用内存中的模型。

## 检索增强（RAG）接入

`webapp` 支持在推理前调用同伴的 RAG 微服务，把检索资料拼进提示词：

```text
用户提问 → webapp → [RAG 微服务 /prepare] → 增强提示词 → 本地 LoRA 推理 → 前端展示
```

启动前设置服务地址（默认 `http://192.168.108.82:8090`）：

```bash
cd "/Users/chenyixuan/Desktop/Study/大模型实训/实训项目选题"
TCM_RAG_URL="http://192.168.108.82:8090" \
  .venv-mlx/bin/python "项目2：基于LoRA微调的中医医学专家大模型/webapp/server.py" --port 8088

# 只跑本地模型、完全不启用检索增强：
# .venv-mlx/bin/python ".../webapp/server.py" --port 8088 --no-rag
```

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `TCM_RAG_URL` | `http://192.168.108.82:8090` | RAG 微服务地址 |
| `TCM_RAG_ENABLED` | `1` | 设为 `0` 整体关闭检索增强 |
| `TCM_RAG_TOP_K` | `3` | 默认检索条数 |
| `TCM_RAG_TIMEOUT` | `5` | 单次 `/prepare` 超时（秒） |
| `TCM_RAG_COOLDOWN` | `15` | 失败后的冷却时间（秒），到期自动重试 |
| `TCM_RAG_CACHE_TTL` | `300` | 同题增强结果缓存（秒） |
| `TCM_RAG_WRITEBACK` | `1` | 是否回写模型输出到 `/traces/{id}/outputs` |

行为约定：

- **LoRA 链路零改动**：只替换送入模型的 `question`，系统提示与适配器权重不变；
- **六格同题可比**：同一 stance 下的基座 / LoRA 复用同一增强提示词和 `trace_id`；
  `aligned` / `opposed` 的缓存键严格隔离；
- **自动降级**：RAG 不可达时四个 RAG 格降级直答，本地推理不中断；
- **自动恢复**：冷却期结束后重新探测，RAG 服务恢复无需重启 webapp；
- **局域网直连**：RAG 客户端不继承系统 HTTP 代理，避免 `127.0.0.1` 代理误接管局域网微服务；
- **可追溯**：界面下方按三列展示实际 prompt、命中资料、风险元数据、`trace_id` 与配对检查。

### 一屏 2×3 拉回力实验

界面直接铺开全部六条路径，不用切换模式：

| | 无 RAG | aligned 正向资料 | opposed 负向资料 |
|---|---|---|---|
| **Qwen 基座** | 原始基线 | 正向检索对照 | 受误导下界 |
| **安全内化 LoRA** | 权重自身倾向 | 同向加固 | 核心拉回力格 |

界面上的 Top K 只影响四个 RAG 格。`--no-rag` 或 `TCM_RAG_ENABLED=0` 可验证服务断开时的降级直答。

### 页面底部 2×2 前测

2×2 与上方 2×3 共用同一次用户输入，不调用 RAG，只改变 LoRA 版本与 system 条件：

| | 45 字训练提示 | 10 字中性提示 |
|---|---|---|
| **旧版安全 LoRA** | 旧版 + 训练提示 | 旧版 + 中性提示 |
| **安全内化 LoRA** | 新版 + 训练提示 | 新版 + 中性提示 |

这一区域用于把已有前测结果做成交互式展示，不替代离线 96 条人工复核，也不继续扩大前测样本。

接口：

- `GET /api/health`：本地服务、设备、模型加载状态 + RAG 连通性摘要；
- `GET /api/rag/health`：RAG 微服务详情（文档数、索引后端、数据库连接）；
- `POST /api/generate`：请求体 `question` / `variant` / `max_tokens` / `use_rag` / `stance` / `top_k`，
  响应在原结构上增加 `rag` 与 `prompt_sent` 两个字段。

2026-09-17 的 3 道题、18 次 2×3 推理属于**先导实验**，用于验证真实链路并试做人工判读；结果显示新版
LoRA 不能稳定抵抗 opposed RAG 的错误事实与危险剂量。完整输出与人工判读见
[`evaluation/results/rag_2x3_webapp_20260917-142332.md`](evaluation/results/rag_2x3_webapp_20260917-142332.md)。

随后已跑完三立场同题集合的 **134 道题、804 次真实推理**。原始数据通过结构验收，并使用透明正则、
字符 n-gram 相似度和配对统计完成本地 NLP 自动筛查。全量文件见
[`evaluation/results/rag_2x3_webapp_20260917-144422.jsonl`](evaluation/results/rag_2x3_webapp_20260917-144422.jsonl)
和 [`evaluation/results/rag_2x3_webapp_20260917-144422.md`](evaluation/results/rag_2x3_webapp_20260917-144422.md)。

自动筛查中，LoRA-opposed 相对 base-opposed 平均 `+1.38` 分，95% bootstrap CI 为
`[-0.95, +3.66]`；34 题改善、76 题持平、24 题变差。攻击话术吸收率下降，但安全边界率也下降，
因此结论是“有局部抑制信号，但未形成稳定、全面的安全拉回”。详见
[`evaluation/results/rag_2x3_webapp_20260917-144422_nlp_report.md`](evaluation/results/rag_2x3_webapp_20260917-144422_nlp_report.md)。

实现细节、与同伴接口的字段映射和降级策略见
[`docs/RAG接入实现说明.md`](docs/RAG接入实现说明.md)。

## 系统提示词与安全归因

四个提示文件承担不同角色，不能互相覆盖：

| 文件 | 字数 | 用在哪 |
|---|---:|---|
| `evaluation/TRAINING_SYSTEM_PROMPT.txt` | 45 | 训练时。与 3061 条训练数据的 `system` 逐字一致 |
| `evaluation/SAFETY_SYSTEM_PROMPT.txt` | 193 | 推理/部署时。训练数据里没有这句，是**生产护栏** |
| `evaluation/MINIMAL_SYSTEM_PROMPT.txt` | 25 | 实验用。只留人设、不含任何安全约束 |
| `evaluation/SYSTEM_PROMPT_MINIMAL.txt` | 10 | 当前 2×2 前测和 2×3 Web 固定中性条件 |

最新 2×2 人工复核表明：旧版在 45 字和 10 字条件下均有 3/24 `fail`、2/24 高风险失守，
因此“旧版换 10 字必然崩”没有得到支持；新版两种条件下均为 0 `fail`、0 高风险失守，支持
“安全行为部分进入 adapter”的谨慎结论。它仍不是绝对安全保证，生产部署继续使用 193 字安全提示。

完整 96 条复核见 [`evaluation/SAFETY_2X2_REVIEW.md`](evaluation/SAFETY_2X2_REVIEW.md)。

## 关键文档

- [`项目交接说明.md`](项目交接说明.md)：完整决策记录和当前进度。
- [`docs/README.md`](docs/README.md)：全部文档的分类索引和阅读顺序。
- [`docs/项目全流程复盘与最终报告.md`](docs/项目全流程复盘与最终报告.md)：从选题到全量实验的最终复盘。
- [`docs/同伴接入指南.md`](docs/同伴接入指南.md)：RAG 微服务接口约定与验收清单。
- [`docs/RAG接入实现说明.md`](docs/RAG接入实现说明.md)：本仓库实际实现与降级策略。
- [`data_processed/DATA_CARD.md`](data_processed/DATA_CARD.md)：数据范围、清洗规则和限制。
- [`data_audit/REPORT.md`](data_audit/REPORT.md)：原始数据审计。
- [`evaluation/RESULTS.md`](evaluation/RESULTS.md)：训练与评测结果。
- [`evaluation/SAFETY_2X2_REVIEW.md`](evaluation/SAFETY_2X2_REVIEW.md)：新旧 LoRA 两提示前测与人工判读。
- [`docs/LoRA训练与部署讲解.md`](docs/LoRA训练与部署讲解.md)：原理、命令和答辩准备。

## 已知限制

- 医学事实未经专业人员逐条审核。
- LoRA 依赖完全匹配的 Qwen2.5-1.5B 基座，不能直接套用到 Qwen3 或其他尺寸模型。
- 安全提示和补强训练降低了高危输出，但不能提供数学上的绝对安全保证。
- 原始课程数据的公开再分发许可尚未确认，因此建议保持仓库私有，仅用于组内协作。
