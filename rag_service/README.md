# TCM RAG 微服务

位于 **用户 query 之后、LLM 之前** 的检索增强环节；与 LoRA 微调链式线性、互不干扰。

```
用户 query ──> [本服务] MySQL 取文档 → 嵌入 → FAISS 检索 → 拼增强提示词
          ──> LoRA 推理（同伴 MacBook，不可达时降级模拟）──> 返回
```

## 快速启动（Windows，conda 环境 lora）

```bash
conda activate lora
cd G:/lora/tcm-lora-mlx
python -m rag_service.server          # 0.0.0.0:8090，供局域网同伴访问
```

首次启动会自动：建库建表（幂等）→ 导入三立场知识文档 → 加载嵌入模型 → 构建 FAISS 索引。
索引带**指纹缓存**（文档集合 + 嵌入后端哈希），内容未变时重启直接复用，
不再重算嵌入（1500 条约 2-3 分钟 → 秒级）。

可将 `.env.example` 复制为 `.env` 修改配置（如指向同伴的 LoRA 服务）。

## 知识库：三立场数据集（消融实验）

`data_rag_stance/rag_stance_dataset.jsonl`，每立场 400 条，共 1200：

| 立场 | 含义 | 构造方式 |
|---|---|---|
| `aligned` | 同向 · 安全对齐 | test 分片（未训练）+ 核心基准题 + 人工安全示范，原样使用 |
| `neutral` | 中立 · **无安全拦截** | 隔离集答案**剥离安全提示句**（确定性正则），实质性回答但不含任何安全提示 |
| `opposed` | 反向 · **恶意诱导** | 与 neutral **同一批问题**，答案再叠加主动诱导层（同题配对，`paired_id` 可追溯） |

> ⚠️ `opposed` 组为对抗性数据（含不安全医疗建议原文与诱导表述），
> **仅用于课程设计的对照实验**。默认立场为 `aligned`；
> 启用 `opposed`/`all` 需设置 `TCM_RAG_ALLOW_OPPOSED=1`，否则接口返回 403。

构建脚本：`scripts/build_stance_dataset.py`（确定性抽样，种子 `20260916`）；
数据集卡：`data_rag_stance/DATASET_CARD.md`；实验方案：`docs/三立场消融实验设计.md`。

## 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 各组件健康状态（含各立场文档数、默认立场、索引缓存状态） |
| POST | `/retrieve` | `{query, top_k, stance}` 纯检索 |
| POST | `/prepare` | `{query, top_k, enabled, stance}` 检索 + 增强提示词（query→LLM 的插入点） |
| POST | `/generate` | `{query, top_k, rag_enabled, variant, max_tokens, stance}` 完整链路（含降级兜底） |
| GET | `/documents` | 文档列表，可选 `?stance=` 过滤、`?full=1` 带正文 |
| POST | `/documents` | 新增/更新文档（需带 `stance`），自动重建索引 |
| POST | `/index/rebuild` | 手动重建 FAISS 索引（`force` 语义：忽略缓存） |
| GET | `/traces/{id}` | 链路 trace（query → 增强提示词 → 检索明细 → 模型输出） |

**stance 参数**：`all` / `aligned` / `neutral` / `opposed`（默认取 `TCM_RAG_STANCE`）。
其中 `opposed` 与 `all` 受 `TCM_RAG_ALLOW_OPPOSED` 开关保护（默认关闭，关闭时返回 403）。
指定立场时**精确只在该立场内召回**（先全库算分再按立场过滤，不受候选池截断影响），
且该立场会写入 `query_traces.rag_stance` 供实验归因。

### 示例：完整链路（指定反向立场）

```bash
curl http://127.0.0.1:8090/generate -H "Content-Type: application/json" \
  -d '{"query":"我最近总是头晕目眩，中医有什么好的方剂推荐吗？","top_k":3,"stance":"aligned"}'
```

返回体关键字段：`augmented_prompt`（送入 LLM 前的完整提示词）、`rag_results`（命中文档，含
`stance` 立场）、`llm.response`（LoRA 或降级模拟的回答）、`inference_mode`
（`remote-*` 真实 / `degraded-mock` 降级）。

## 模块结构

| 文件 | 职责 |
|---|---|
| `config.py` | 环境变量 / .env 配置 |
| `embedder.py` | sentence-transformers 语义嵌入（降级：哈希嵌入） |
| `vector_store.py` | FAISS 向量 + bigram BM25 混合检索（降级：纯 Python 余弦） |
| `repository.py` | MySQL 持久层（文档 / trace / 输出；降级：内存） |
| `llm_client.py` | 调用同伴 LoRA（webapp 或 OpenAI 兼容；降级：本地模拟） |
| `server.py` | HTTP 服务入口 |
| `smoke_test.py` | 冒烟测试（不起 HTTP 服务，直接测 Runtime） |

建表 SQL：`database/init.sql`（库 `lora`，服务启动也会自动建）。

## 检索方案：为什么不是纯向量

`paraphrase-multilingual-MiniLM-L12-v2` 对「短查询 ↔ 长文档」的余弦值压缩严重
（文档与自身标题仅 0.25），纯向量检索会让专有名词查询失效
（实测：「薄荷能治什么？」命中不到刚录入的薄荷文档，排名第 13）。

现方案 = **标题/正文分向量取最大 + 中文 bigram BM25 + 加权 RRF 融合**，
同批查询命中排名 13 → 1。完整实验数据见 `docs/检索质量实测报告.md`，复现脚本：

```bash
env -u http_proxy -u https_proxy HF_HUB_OFFLINE=1 python scripts/eval_retrieval.py
env -u http_proxy -u https_proxy HF_HUB_OFFLINE=1 python scripts/tune_retrieval.py
```

调参项：`TCM_RAG_LEXICAL_WEIGHT`（默认 4.0）、`TCM_RAG_RRF_K`（默认 60）、
`TCM_RAG_MIN_SCORE`（默认 0.20，仅当语义弱且无词汇命中时才丢弃）。

## 管理面板（独立前后端）

`rag_console/` 是**独立于 webapp** 的小前后端（端口 8091），只服务 RAG 自己：
服务状态、检索测试、**新增知识文档**、文档浏览。

```bash
env -u http_proxy -u https_proxy HF_HUB_OFFLINE=1 python rag_console/server.py --port 8091
# 浏览器打开 http://127.0.0.1:8091
```

它通过 HTTP 反向代理调用 RAG 微服务（`--rag-url` 可指向任意实例），
不直接依赖 `rag_service` 内部代码，微服务边界保持清晰。
文档浏览/编辑/删除等完整 CRUD 留待后期并入 LoRA 前后端，本期只做「新增」。

## 降级兜底（三层，均如实标注不静默）

1. MySQL 断连 → 内存种子文档（`database: fallback-memory`）
2. 嵌入模型加载失败 → 哈希嵌入（`embedder_backend: hashing-fallback`）
3. 同伴 LoRA 不可达 → 基于检索资料合成模拟回答（`degraded: true`）

## 局域网联调

- 本服务默认 `0.0.0.0`，同伴访问 `http://<我的IP>:8090`（真实网卡 IP，非 VMware 段）；
- Windows 防火墙放行 8090：`New-NetFirewallRule -DisplayName "TCM RAG 8090" -Direction Inbound -LocalPort 8090 -Protocol TCP -Action Allow`；
- 详见 `docs/前端与后端改进方案.md` §4。
