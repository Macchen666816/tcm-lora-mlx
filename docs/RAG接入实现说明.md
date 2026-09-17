# RAG 接入实现说明

本文件记录 **MacBook 侧 webapp 实际落地的实现**，与 [`同伴接入指南.md`](同伴接入指南.md)
（接口约定）配套阅读。指南负责「接口长什么样」，本文负责「我们怎么接的、为什么这么接」。

```text
浏览器 ──POST /api/generate──▶ webapp/server.py
                                   │
                                   ├─ RagClient.prepare()  ──POST /prepare──▶ 192.168.108.82:8090
                                   │      └─ 失败/超时 → 本轮降级，用原始问题
                                   │
                                   ├─ RUNTIME.generate()   ── MLX 本地推理（基座 / LoRA）
                                   │
                                   └─ 响应 = 原结构 + rag + prompt_sent
```

## 1. 改动清单

| 文件 | 改动 |
|---|---|
| `webapp/server.py` | `RagClient` 支持 stance 透传、按 stance 隔离缓存/冷却、降级、自动恢复与回写；用无代理 opener 直连局域网；加载基座、新旧两版 LoRA；回传 `full_prompt` |
| `evaluation/TRAINING_SYSTEM_PROMPT.txt` | 新增。由脚本从 `data_processed/mlx/train.jsonl` 提取的训练时 system，供「训练原句」预设使用 |
| `webapp/static/index.html` | 上方固定 **2×3 六格**（基座/新版 LoRA × 无 RAG/aligned/opposed），底部固定 **2×2 四格**（旧版/新版 LoRA × 45/10 字 system） |
| `webapp/static/app.js` | 十路并行请求；同 stance 配对检查；aligned/opposed 隔离检查；风险元数据、命中资料与完整 prompt 展示 |
| `webapp/static/styles.css` | 上述组件样式与响应式规则 |
| `README.md` / `webapp/README.md` / `evaluation/` | 记录六格实验、两提示前测与人工判读 |

生产用 `evaluation/SAFETY_SYSTEM_PROMPT.txt` 未被覆盖；Web 实验固定读取独立的
`evaluation/SYSTEM_PROMPT_MINIMAL.txt`。

## 2. 关键设计

### 2.1 只增强提问，不动系统提示

`system prompt` 与 LoRA 权重完全不变，只把 `question` 换成 `augmented_prompt`：

```python
prompt, rag_payload = RAG.prepare(question, stance=stance, top_k=top_k)
result = RUNTIME.generate(variant, prompt, max_tokens, system_preset)
```

这样「微调」和「检索」是两个可独立开关的变量，消融实验才成立。

> ⚠️ **别把 RAG 的前言当成系统提示词**。上游 `/prepare` 返回的 `augmented_prompt` 开头自带一段
> 写作指令（"请根据下列检索资料回答用户问题。资料可能不完整或有误……涉及诊断、处方、剂量或
> 中毒风险时，应明确建议由专业医疗人员评估。"）。那句话属于 **user 消息的一部分**，
> 由 RAG 服务生成，不是本项目的 system prompt。
> 于是模型实际会读到三层：**我们的 system → RAG 前言 + 检索资料 → 用户问题**。
> 界面上「查看模型真正读到的完整提示词」把 system / user / 模板原文分开显示，就是为了不混淆这三层，
> 响应里的 `system_prompt_effective` 与 `full_prompt` 可以逐字核对。

Web 上方六格固定共用 10 字 `role_only`，避免操作时改变实验条件；页面底部 2×2 固定比较旧版/新版 LoRA
在 45 字 `training` 与 10 字 `role_only` 下的表现。十格共用同一个用户问题和回答长度，2×2 不调用 RAG。
其他预设只保留给离线评测和 API 审计。
新旧 LoRA 的两提示对照见 [`../evaluation/SAFETY_2X2_REVIEW.md`](../evaluation/SAFETY_2X2_REVIEW.md)。

### 2.2 一屏六路：同列配对、异列隔离

界面不使用模式开关，一次提问并行跑完：

| | 无 RAG | aligned | opposed |
|---|---|---|---|
| **Qwen 基座** | 基线 | 正向资料 | 负向资料 |
| **安全内化 LoRA** | 权重倾向 | 同向加固 | 拉回力核心格 |

六路是并发的。同一 stance 的 base / LoRA 必须拿到相同 prompt 和 `trace_id`，而 aligned / opposed
必须严格不同，否则都无法归因。

实现上用两层保护：

1. **单飞（single-flight）**：`prepare()` 的缓存未命中路径全部收在 `self._prepare_lock` 内，
   抢到锁的线程才会真的请求上游，其余线程拿到锁后复核缓存直接复用。
2. **同题同立场缓存**：`(question, top_k, stance)` 为键、默认 300 秒，第二轮提问直接复用，响应里以
   `rag_cache: "hit"` 标记。

模拟上游并发测试中，aligned 与 opposed 各只请求 **1 次**；各自的 base / LoRA prompt 和 trace
完全一致，两种 stance 之间则不同。前端会再次检查这两个条件。

> 改动 `top_k` 会改变缓存键，因此会重新检索 —— 这正是做「检索条数」对照实验需要的语义。

### 2.3 降级与自动恢复

| `rag_status` | 触发条件 | 界面表现 | 送入模型的内容 |
|---|---|---|---|
| `retrieved` | 上游命中并返回增强提示词 | 绿色 `retrieved` 与命中数 | 增强提示词 |
| `empty` | 上游 `rag_status=empty`，或声称命中但提示词为空 | 灰色「RAG 未命中」 | 原始问题 |
| `degraded` | 连接失败、超时、HTTP 错误、返回体不合法 | 黄色「RAG 已降级（直答）」 | 原始问题 |
| `disabled` | 无 RAG 列或服务端关闭检索 | 灰色 `disabled` | 原始问题 |

降级时四个检索格与无 RAG 输入相同，证据区会显示“本轮无法验证立场隔离”，避免误判。

失败后进入**冷却期**（默认 15 秒），期间不再打上游，避免对方服务刚挂就被反复冲击；
冷却结束后自动恢复探测，**RAG 服务恢复不需要重启 webapp**。

`RagClient` 使用 `ProxyHandler({})` 构造独立 opener，不读取系统 HTTP/HTTPS 代理。原因是本机代理曾把
`192.168.108.82:8090` 请求接管到 `127.0.0.1:7892`，造成微服务实际在线但 Web 误判超时；局域网服务
现在始终走直连。

> 指南 §2 的示例代码里 `RAG_AVAILABLE = False` 写在函数内却没有 `global` 声明，
> 会抛 `UnboundLocalError`；而且「一次失败后永久不再重试」与 §5 验收清单里
> 「恢复服务后下一次提问重新带上 RAG」互相矛盾。本实现用冷却期同时满足这两点。

### 2.4 上游字段映射

上游 `/prepare` 实测返回 `results[]`，每项字段为
`rank / document_id / title / source / score / fused_score / lexical_score / content / metadata`。
后端统一归一化成前端固定结构，并对字段名做了别名兜底（对方改名不用动前端）：

| 归一化字段 | 取值来源（按优先级） |
|---|---|
| `index` | `rank` → `index` → `position` → 序号 |
| `title` | `title` → `doc_title` → `name` → `file_name` → 「资料 N」 |
| `score` | `score` → `similarity` → `relevance` → `distance` |
| `source` | `source` → `file` → `path` → `uri` → `url` |
| `doc_id` | `document_id` → `doc_id` → `id` → `chunk_id` |
| `snippet` | `content` → `snippet` → `text` → `chunk` → `preview` |
| `stance` / `risk_level` / `adversarial_strength` / `intent_tag` | 同名字段或 `metadata` 内同名字段 |

容器名同样兼容 `results` / `hits` / `retrievals` / `documents` / `matches` / `contexts` / `items`。

### 2.5 输出回写（可选）

指南 §6 的三表证据链默认开启：拿到 `trace_id` 后，用**后台守护线程**把 base / lora
两次推理结果 POST 到 `/traces/{trace_id}/outputs`。回写失败只打日志，绝不影响回答。
`TCM_RAG_WRITEBACK=0` 可关闭。

## 3. 接口变化

- `GET /api/health`：新增 `rag` 子对象（`enabled` / `reachable` / `url` / `document_count` / `index_backend`）；
- `GET /api/rag/health`：透传上游 `/health` 关键字段，`?force=1` 可跳过 10 秒缓存；
- `GET /api/prompts`：五个系统提示词预设的全文、来源文件与字数（审计用）；
- `POST /api/generate`：请求体包含 `use_rag`、`stance`（`aligned` / `neutral` / `opposed`）、
  `top_k`（1–8）和 `system`（Web 固定为 `role_only`）；
  响应在原结构上新增 `rag`、`prompt_sent`、`system_preset`、`system_prompt`（我们注入的）、
  `system_prompt_effective`（实际生效的，含模板补的默认值）、`full_prompt`（模板渲染后的原始串）。
  `variant` 支持 `base` / `lora` / `legacy_lora`。界面一次点击并发发出 10 个固定请求，各自回填自己的格子。

`rag` 载荷字段：`rag_status`、`trace_id`、`rag_latency_ms`、`top_k`、`stance`、`results[]`、
`prompt`、`augmented_prompt`、`original_question`、`injected`、`service_url`、
`rag_cache`、`hint`、`error`。

与指南的差异：指南约定「RAG 不可用时 `rag` 为 `null`」。本实现始终返回 `rag` 对象，
用 `rag_status` 区分状态，信息量更大；前端同时兼容 `rag === null`（按其约定当作已降级）。

## 4. 启动

虚拟环境在工作区根目录，因此用「工作区根 + 脚本全路径」的方式启动：

```bash
cd "/Users/chenyixuan/Desktop/Study/大模型实训/实训项目选题"
TCM_RAG_URL="http://192.168.108.82:8090" \
  .venv-mlx/bin/python "项目2：基于LoRA微调的中医医学专家大模型/webapp/server.py" --port 8088
```

常用开关：

```bash
--no-rag                # 完全不启用检索增强，只跑本地模型
--rag-url http://x:8090 # 临时换地址
--host 0.0.0.0          # 允许同伴用浏览器访问你机器上的界面
```

## 5. 验收对照（2026-09-17）

- [x] 六格均为真实 MLX 推理，`inference_mode=remote-webapp`
- [x] 生产安全提示文件保持 193 字；Web 实际 system 为独立 10 字文件
- [x] 对端不可达时六格仍全部出回答，四个 RAG 格标记 `degraded`
- [x] 模拟上游证明同 stance 的 base / LoRA 共用 prompt 与 trace
- [x] 模拟上游证明 aligned / opposed 缓存和 prompt 严格隔离
- [x] 1280px 桌面与 390×844 手机真实浏览器检查均无横向溢出
- [x] 页面底部 2×2 与上方 2×3 共用用户输入；十格一次运行并全部回填
- [x] 真实微服务恢复后完成全量 134 道题、804 次 2×3 推理；aligned / opposed 均命中且立场隔离

全量原始数据见
[`JSONL`](../evaluation/results/rag_2x3_webapp_20260917-144422.jsonl) /
[`Markdown`](../evaluation/results/rag_2x3_webapp_20260917-144422.md)。134 道题全部具有六个实验臂，
536 个 RAG 臂均为 `retrieved` 且各命中 3 条，自动归因检查零失败。

另有 3 道题先导实验的人工判读：
[`../evaluation/results/rag_2x3_webapp_20260917-142332.md`](../evaluation/results/rag_2x3_webapp_20260917-142332.md)。
它发现新版 LoRA 不能稳定抵抗 opposed RAG 中的错误事实和可执行危险剂量。全量数据随后通过
[`score_rag_pullback_nlp.py`](../scripts/score_rag_pullback_nlp.py) 完成本地自动筛查：平均拉回变化
`+1.38` 分且 95% CI 跨 0，未显示稳定的总体提升。完整结果见
[`NLP 批处理报告`](../evaluation/results/rag_2x3_webapp_20260917-144422_nlp_report.md)。

## 6. 历史四路消融实测（2026-09-16）

问题：`薄荷的性味归经和主要功效是什么？`

| 路径 | 首句 | 观察 |
|---|---|---|
| **A** 纯基座 | 「薄荷性辛，味**甘**……利**湿**消肿」 | 「味甘」「利湿消肿」均无出处，典型幻觉 |
| **B** 纯微调 | 「薄荷味辛、凉，归肺、肝经。主要功效为疏散风热、清利头目、利咽、解毒」 | 句式与术语明显中医化，但「解毒」略有出入 |
| **C** 基座 + 检索 | 「性味：辛，凉。归经：肺、肝经。主要功效：疏散风热、清利头目、利咽、透疹、疏肝行气」 | 事实被检索资料完全锚定，但像在念资料 |
| **D** 微调 + 检索 | 「薄荷，性味辛凉，归肺、肝经。主要功效有疏散风热、清利头目、利咽、透疹、疏肝行气。用于风热感冒……用量3-6g，入汤剂宜后下。」 | 事实准确 + 表述专业 + 带用量医嘱 |

两条结论可以直接写进报告：

- **纵向比（A→C、B→D）**：检索把事实拉正，「味甘」「利湿消肿」这类幻觉消失；
- **横向比（C→D 最明显）**：事实一致的前提下，微调改变的是**表达方式**与**术语密度**，而不是事实本身
  —— 即「微调管表达，RAG 管事实」。

同一次运行中 C 与 D 的 `prompt_sent` 与 `trace_id` 完全相同，因此上表的横向对比成立。

## 7. 已知限制

- 检索结果未经中医专业人员审核，页面仍需保留「课程实验」免责提示；
- 上游 `results` 里混有评测集条目（如 `core_benchmark.jsonl#...`），命中质量依赖对方语料治理；
- 同题缓存默认 300 秒，若对方知识库在缓存期内更新，界面仍显示旧结果；
- 单实例内存缓存，多进程部署时两侧可比性无保证（当前为单进程本地服务，不涉及）。
