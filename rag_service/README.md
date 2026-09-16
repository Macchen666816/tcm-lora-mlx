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

首次启动会自动：建库建表（幂等）→ 导入种子知识文档 → 加载嵌入模型 → 构建 FAISS 索引。
可将 `.env.example` 复制为 `.env` 修改配置（如指向同伴的 LoRA 服务）。

## 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 各组件健康状态 |
| POST | `/retrieve` | `{query, top_k}` 纯检索 |
| POST | `/prepare` | `{query, top_k, enabled}` 检索 + 增强提示词（query→LLM 的插入点） |
| POST | `/generate` | `{query, top_k, rag_enabled, variant, max_tokens}` 完整链路（含降级兜底） |
| GET | `/documents` | 知识文档列表（向量化前，MySQL 持久层） |
| POST | `/documents` | 新增/更新文档，自动重建索引 |
| POST | `/index/rebuild` | 手动重建 FAISS 索引 |
| GET | `/traces/{id}` | 链路 trace（query → 增强提示词 → 检索明细 → 模型输出） |

### 示例：完整链路

```bash
curl http://127.0.0.1:8090/generate -H "Content-Type: application/json" \
  -d '{"query":"薄荷的性味归经和主要功效是什么？","top_k":3}'
```

返回体关键字段：`augmented_prompt`（送入 LLM 前的完整提示词）、`rag_results`（命中文档）、
`llm.response`（LoRA 或降级模拟的回答）、`inference_mode`（`remote-*` 真实 / `degraded-mock` 降级）。

## 模块结构

| 文件 | 职责 |
|---|---|
| `config.py` | 环境变量 / .env 配置 |
| `embedder.py` | sentence-transformers 语义嵌入（降级：哈希嵌入） |
| `vector_store.py` | FAISS 向量存取（降级：纯 Python 余弦） |
| `repository.py` | MySQL 持久层（文档 / trace / 输出；降级：内存） |
| `llm_client.py` | 调用同伴 LoRA（webapp 或 OpenAI 兼容；降级：本地模拟） |
| `server.py` | HTTP 服务入口 |
| `smoke_test.py` | 冒烟测试（不起 HTTP 服务，直接测 Runtime） |

建表 SQL：`database/init.sql`（库 `lora`，服务启动也会自动建）。

## 降级兜底（三层，均如实标注不静默）

1. MySQL 断连 → 内存种子文档（`database: fallback-memory`）
2. 嵌入模型加载失败 → 哈希嵌入（`embedder_backend: hashing-fallback`）
3. 同伴 LoRA 不可达 → 基于检索资料合成模拟回答（`degraded: true`）

## 局域网联调

- 本服务默认 `0.0.0.0`，同伴访问 `http://<我的IP>:8090`（真实网卡 IP，非 VMware 段）；
- Windows 防火墙放行 8090：`New-NetFirewallRule -DisplayName "TCM RAG 8090" -Direction Inbound -LocalPort 8090 -Protocol TCP -Action Allow`；
- 详见 `docs/前端与后端改进方案.md` §4。
