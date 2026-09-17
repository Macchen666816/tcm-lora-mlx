# 中医模型对比 Web 应用

## 启动

虚拟环境不纳入 Git，本机把它建在工作区根目录（`实训项目选题/.venv-mlx`），比仓库根高一层，
因此从**工作区根目录**用脚本全路径启动：

```bash
cd "/Users/chenyixuan/Desktop/Study/大模型实训/实训项目选题"
.venv-mlx/bin/python "项目2：基于LoRA微调的中医医学专家大模型/webapp/server.py" --port 8088
```

如果虚拟环境就在仓库根目录（队友 clone 后自建的情形）：

```bash
.venv-mlx/bin/python webapp/server.py --port 8088
```

浏览器访问 `http://127.0.0.1:8088`。

> 脚本内部所有路径都基于 `__file__` 解析，站在哪个目录启动都能找到模型、适配器和静态资源，
> 上面的差异只影响「相对路径写不写得对」。

服务使用 Python 标准库提供网页和 API，不需要额外安装 FastAPI、Flask 或 Node 依赖。
第一次调用时会分别加载基座、旧版 LoRA 和安全内化 LoRA，之后复用内存中的三个实例。

## 检索增强（RAG）

可选地在推理前调用同伴的 RAG 微服务增强提示词：

```bash
TCM_RAG_URL="http://192.168.108.82:8090" \
  .venv-mlx/bin/python "项目2：基于LoRA微调的中医医学专家大模型/webapp/server.py" --port 8088
```

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `TCM_RAG_URL` | `http://192.168.108.82:8090` | RAG 微服务地址 |
| `TCM_RAG_ENABLED` | `1` | `0` 表示整体关闭 |
| `TCM_RAG_TOP_K` | `3` | 默认检索条数 |
| `TCM_RAG_TIMEOUT` | `5` | `/prepare` 超时（秒） |
| `TCM_RAG_COOLDOWN` | `15` | 失败后冷却时间（秒） |
| `TCM_RAG_CACHE_TTL` | `300` | 同题增强结果缓存（秒） |
| `TCM_RAG_WRITEBACK` | `1` | 是否回写模型输出 |

命令行开关：`--no-rag`（完全不启用）、`--rag-url URL`（临时换地址）、`--host 0.0.0.0`（允许同伴访问）。

启动后一次提问会并行发出 **10 个请求**。上方固定 **2×3 六格**：

| | 无 RAG | aligned 正向资料 | opposed 负向资料 |
|---|---|---|---|
| **Qwen 基座** | 原始基线 | 正向检索对照 | 受误导下界 |
| **安全内化 LoRA** | 权重自身倾向 | 同向加固 | 核心拉回力格 |

同一 stance 下的基座 / LoRA 复用同一份增强提示词与同一个 `trace_id`；`aligned` 和 `opposed`
使用不同缓存键，不能互相串用。

页面最底部另有固定 **2×2 四格**，与上方六格共用同一份用户输入和回答长度：

| | 45 字 `training` | 10 字 `role_only` |
|---|---|---|
| **旧版 LoRA** | 旧版 + 训练提示 | 旧版 + 中性提示 |
| **安全内化 LoRA** | 新版 + 训练提示 | 新版 + 中性提示 |

这四格不调用 RAG；它只把已完成的提示词依赖前测放到页面中展示，不继续扩充前测实验。

RAG 不可达时四个检索格自动降级为直答，界面明确标记本轮无法验证立场隔离；冷却期结束会自动重试，
对端服务恢复后无需重启本服务。
局域网请求使用独立的无代理 HTTP opener，不继承本机代理环境，避免代理软件把
`192.168.*` 微服务请求转发到本地代理端口后产生假超时。

接口：

- `GET /api/health`：服务、设备、模型加载状态 + RAG 连通性摘要。
- `GET /api/rag/health`：RAG 微服务详情（文档数、索引后端、数据库连接）。
- `GET /api/prompts`：五个系统提示词预设的全文与来源（审计用）。
- `GET /api/info`：展示用的模型与训练指标。
- `POST /api/generate`：请求体包含 `question`、`variant`（`base` / `lora` / `legacy_lora`）、`max_tokens`，
  可选 `use_rag`（默认 `true`）、`stance`、`top_k`、`system`（默认 `role_only`）；
  响应在原结构上增加 `rag`、`prompt_sent`、`system_preset`、`system_prompt`、
  `system_prompt_effective`、`full_prompt`。
  界面固定发出上方 2×3 六组和下方 2×2 四组请求，共用一次用户输入。

## 系统提示词（安全归因要用）

| 预设 | 内容 | 用途 |
|---|---|---|
| `role_only`（Web 默认） | `evaluation/SYSTEM_PROMPT_MINIMAL.txt`，10 字 | 2×2 与 2×3 实验固定中性条件 |
| `safety` | `evaluation/SAFETY_SYSTEM_PROMPT.txt`，193 字 | 正式部署的第二道护栏 |
| `minimal` | `evaluation/MINIMAL_SYSTEM_PROMPT.txt`，25 字 | 只留人设、无安全约束，用来暴露"安全靠提示词托管" |
| `training` | `evaluation/TRAINING_SYSTEM_PROMPT.txt`，45 字 | 与 3061 条训练数据逐字一致，排除训练/推理分布不一致 |
| `none` | 不注入我们的提示词 | 看模型裸倾向 |

Web 上方六格为保证归因固定使用 `role_only`，前端不提供 system 切换；下方四格固定使用
`training` / `role_only` 两列。其他预设只用于 API 审计和历史实验复现。
新旧 LoRA 的两提示前测见 [`../evaluation/SAFETY_2X2_REVIEW.md`](../evaluation/SAFETY_2X2_REVIEW.md)。

> 做实验请切预设，**不要直接改 `SAFETY_SYSTEM_PROMPT.txt`**：它是部署护栏，
> 改动会静默推翻"部署必须带安全提示"的结论。

`none` 不等于"没有系统提示词"：Qwen 模板会自动补一段官方默认 system，界面上会显示实际生效的那份
（响应里的 `system_prompt_effective`）。

两套模型固定使用同一份 10 字系统提示词（六格共用）；检索增强只改写送入模型的提问，
不改动系统提示与 LoRA 权重。实现细节见 [`../docs/RAG接入实现说明.md`](../docs/RAG接入实现说明.md)。

## 真实 RAG 复跑（2026-09-17）

先导实验完成 3 道题、18 次推理，并进行了人工判读。结果不支持“新版 LoRA 已稳定抵抗负向检索”的强结论：
它会被 opposed 资料带入错误事实，也会复制孕妇用药的具体危险剂量。详见
[`../evaluation/results/rag_2x3_webapp_20260917-142332.md`](../evaluation/results/rag_2x3_webapp_20260917-142332.md)。

全量实验随后完成 134 道同题、804 次推理。134 道题均包含六个实验臂；536 个 RAG 臂全部成功检索 3 条，
同 stance 的 prompt/trace 配对和 aligned/opposed 隔离检查全部通过。原始数据：
[`JSONL`](../evaluation/results/rag_2x3_webapp_20260917-144422.jsonl) /
[`Markdown`](../evaluation/results/rag_2x3_webapp_20260917-144422.md)。

本地 NLP 自动筛查显示，LoRA-opposed 相对 base-opposed 平均 `+1.38` 分，95% CI 跨过 0；
34 题改善、76 题持平、24 题变差。它减少了攻击话术的直接吸收，但没有提高剂量防线，且安全边界率下降。
该结果用于批量筛查，不等同于医学事实审核或人工安全结论。详见
[`NLP 汇总报告`](../evaluation/results/rag_2x3_webapp_20260917-144422_nlp_report.md)。

该应用仅用于课程实验，不可作为诊疗、处方或剂量建议工具。
