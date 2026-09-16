#!/usr/bin/env python3
"""TCM RAG 微服务：位于 用户 query 之后、LLM 之前 的检索增强环节。

链路（链式线性，RAG 与 LoRA 微调先后互不干扰）：
    用户 query ──> [本服务：MySQL 取文档 → 嵌入 → FAISS 检索 → 拼增强提示词]
              ──> LoRA 推理（同伴 MacBook，不可达时降级模拟）──> 返回

启动：
    python -m rag_service.server            # 默认 0.0.0.0:8090（供局域网同伴访问）
    python rag_service/server.py --port 8090

接口：
    GET  /health                健康检查（各组件状态）
    POST /retrieve              纯检索 {query, top_k}
    POST /prepare               检索 + 生成增强提示词 {query, top_k, enabled}
    POST /generate              完整链路：RAG → LoRA（含降级兜底）
    GET  /documents             列出知识文档（向量化前，存 MySQL）
    POST /documents             新增/更新知识文档（自动入 MySQL 并触发重建索引）
    POST /index/rebuild         手动重建 FAISS 索引
    GET  /traces/{trace_id}     查询链路 trace
    POST /traces/{trace_id}/outputs   回写模型输出（webapp 侧调用真实 LoRA 后可选回写）
"""

from __future__ import annotations

import argparse
import hmac
import json
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

if __package__:
    from . import config
    from .embedder import create_embedder
    from .llm_client import LlmClient
    from .repository import DocumentRepository
    from .vector_store import VectorStore
else:  # 直接以脚本方式运行：python rag_service/server.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rag_service import config
    from rag_service.embedder import create_embedder
    from rag_service.llm_client import LlmClient
    from rag_service.repository import DocumentRepository
    from rag_service.vector_store import VectorStore


RAG_INSTRUCTION = (
    "请根据下列检索资料回答用户问题。资料可能不完整或有误，不要把资料中没有的信息"
    "当作事实；涉及诊断、处方、剂量或中毒风险时，应明确建议由专业医疗人员评估。"
)


class RagRuntime:
    """组合持久层（MySQL）、向量层（FAISS）、嵌入层与 LLM 客户端。"""

    def __init__(self) -> None:
        self.embedder, self.embedder_backend, self.embedder_warning = create_embedder(
            config.EMBED_MODEL_NAME, config.EMBED_DEVICE, config.EMBED_BATCH_SIZE
        )
        self.repository = DocumentRepository()
        self.store = VectorStore(config.INDEX_DIR, self.embedder)
        self.llm = LlmClient()
        self._lock = threading.Lock()
        self.index_info = self.rebuild()

    # ---------- 索引 ----------

    def rebuild(self) -> dict:
        with self._lock:
            return self.store.rebuild(self.repository.list_documents())

    # ---------- 检索 ----------

    def retrieve(self, query: str, top_k: int) -> dict:
        started = time.monotonic()
        with self._lock:
            results = self.store.search(query, top_k)
        results = [item for item in results if item["score"] >= config.MIN_SCORE]
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "latency_ms": round((time.monotonic() - started) * 1000),
            "index_backend": self.store.backend,
            "embedder_backend": self.embedder_backend,
            "min_score": config.MIN_SCORE,
        }

    # ---------- 增强提示词（query → LLM 前的插入点） ----------

    def prepare(self, query: str, top_k: int, enabled: bool = True) -> dict:
        if enabled:
            retrieval = self.retrieve(query, top_k)
            results = retrieval["results"]
            context_blocks = [
                f"[资料 {item['rank']}] {item['title']}\n{item['content']}\n来源：{item['source']}"
                for item in results
            ]
            augmented_prompt = (
                f"{RAG_INSTRUCTION}\n\n" + "\n\n".join(context_blocks) + f"\n\n[用户问题]\n{query}"
                if results
                else query
            )
            status = "retrieved" if results else "empty"
            latency_ms = retrieval["latency_ms"]
        else:
            results = []
            augmented_prompt = query
            status = "disabled"
            latency_ms = 0

        trace_id = self.repository.create_trace(
            query, augmented_prompt, enabled, status, latency_ms, results
        )
        return {
            "trace_id": trace_id,
            "query": query,
            "augmented_prompt": augmented_prompt,
            "rag_enabled": enabled,
            "rag_status": status,
            "rag_latency_ms": latency_ms,
            "results": results,
            "index_backend": self.store.backend,
            "embedder_backend": self.embedder_backend,
            "database": self.repository.db_status,
        }

    # ---------- 完整链路：RAG → LoRA（降级兜底） ----------

    def generate(
        self,
        query: str,
        top_k: int,
        rag_enabled: bool,
        variant: str,
        max_tokens: int,
    ) -> dict:
        preparation = self.prepare(query, top_k, rag_enabled)
        llm_result = self.llm.generate(
            preparation["augmented_prompt"],
            variant=variant,
            max_tokens=max_tokens,
            retrieval_results=preparation["results"],
        )
        self.repository.record_output(
            preparation["trace_id"],
            {**llm_result, "trace_id": preparation["trace_id"]},
        )
        return {
            "trace_id": preparation["trace_id"],
            "query": query,
            "augmented_prompt": preparation["augmented_prompt"],
            "rag_status": preparation["rag_status"],
            "rag_results": preparation["results"],
            "rag_latency_ms": preparation["rag_latency_ms"],
            "llm": llm_result,
            "degraded": llm_result.get("degraded", False),
            "degraded_reason": llm_result.get("degraded_reason", ""),
            "inference_mode": llm_result.get("inference_mode", "unknown"),
        }

    # ---------- 状态 ----------

    def health(self) -> dict:
        return {
            "status": "ok",
            "service": "tcm-rag",
            "document_count": len(self.store.documents),
            "index_backend": self.store.backend,
            "embedder_backend": self.embedder_backend,
            "embedder_warning": self.embedder_warning,
            "database": self.repository.db_status,
            "database_error": self.repository.db_error,
            "llm_mode": self.llm.mode,
            "llm_url": self.llm.base_url or "(未配置，仅降级模拟)",
            "index_info": self.index_info,
        }


RUNTIME: RagRuntime | None = None


class RagHandler(BaseHTTPRequestHandler):
    server_version = "TCMRag/2.0"

    def log_message(self, format_string: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}", flush=True)

    # ---------- 基础工具 ----------

    def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")  # 同伴 webapp 跨域调用
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-RAG-API-Key")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 1_048_576:
            raise ValueError("请求内容大小不合法")
        return json.loads(self.rfile.read(length))

    def _authorized(self) -> bool:
        expected = config.API_KEY
        provided = self.headers.get("X-RAG-API-Key", "")
        return not expected or hmac.compare_digest(provided, expected)

    def do_OPTIONS(self) -> None:  # CORS 预检
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-RAG-API-Key")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    # ---------- 路由 ----------

    def do_GET(self) -> None:
        assert RUNTIME is not None
        path = urlparse(self.path).path
        if path == "/health":
            self._json(RUNTIME.health())
        elif not self._authorized():
            self._json({"error": "Unauthorized"}, HTTPStatus.UNAUTHORIZED)
        elif path == "/documents":
            documents = RUNTIME.repository.list_documents()
            compact = [
                {
                    "external_id": doc["external_id"],
                    "title": doc["title"],
                    "source": doc.get("source", ""),
                    "content_length": len(doc["content"]),
                }
                for doc in documents
            ]
            self._json({"documents": compact, "count": len(compact)})
        elif path.startswith("/traces/"):
            trace = RUNTIME.repository.get_trace(path.removeprefix("/traces/"))
            self._json(
                {"trace": trace} if trace else {"error": "Not found"},
                HTTPStatus.OK if trace else HTTPStatus.NOT_FOUND,
            )
        else:
            self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        assert RUNTIME is not None
        path = urlparse(self.path).path
        if not self._authorized():
            self._json({"error": "Unauthorized"}, HTTPStatus.UNAUTHORIZED)
            return
        try:
            payload = self._body()
            if path == "/retrieve":
                query, top_k = self._validate_query(payload)
                self._json(RUNTIME.retrieve(query, top_k))
            elif path == "/prepare":
                query, top_k = self._validate_query(payload)
                enabled = bool(payload.get("enabled", True))
                self._json(RUNTIME.prepare(query, top_k, enabled))
            elif path == "/generate":
                query, top_k = self._validate_query(payload)
                rag_enabled = bool(payload.get("rag_enabled", True))
                variant = str(payload.get("variant", config.LLM_VARIANT))
                if variant not in {"base", "lora"}:
                    raise ValueError("variant 必须是 base 或 lora")
                max_tokens = int(payload.get("max_tokens", config.LLM_MAX_TOKENS))
                if not 32 <= max_tokens <= 1024:
                    raise ValueError("max_tokens 必须在 32 到 1024 之间")
                self._json(RUNTIME.generate(query, top_k, rag_enabled, variant, max_tokens))
            elif path == "/documents":
                document = RUNTIME.repository.upsert(payload)
                rebuild_info = RUNTIME.rebuild()
                self._json(
                    {"document": document, "index": rebuild_info}, HTTPStatus.CREATED
                )
            elif path == "/index/rebuild":
                self._json(RUNTIME.rebuild())
            elif path.startswith("/traces/") and path.endswith("/outputs"):
                trace_id = path.removeprefix("/traces/").removesuffix("/outputs").strip("/")
                RUNTIME.repository.record_output(trace_id, payload)
                self._json({"stored": True, "database": RUNTIME.repository.db_status})
            else:
                self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, TypeError, json.JSONDecodeError, KeyError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._json({"error": f"RAG 服务失败: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    @staticmethod
    def _validate_query(payload: dict) -> tuple[str, int]:
        query = str(payload.get("query", "")).strip()
        top_k = int(payload.get("top_k", config.DEFAULT_TOP_K))
        if not query:
            raise ValueError("query 不能为空")
        if len(query) > 2000:
            raise ValueError("query 不能超过 2000 个字符")
        if not 1 <= top_k <= 8:
            raise ValueError("top_k 必须在 1 到 8 之间")
        return query, top_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TCM RAG microservice")
    parser.add_argument("--host", default=config.SERVICE_HOST)
    parser.add_argument("--port", type=int, default=config.SERVICE_PORT)
    return parser.parse_args()


def main() -> None:
    global RUNTIME
    config.load_env_file()
    args = parse_args()

    print("初始化 RAG 微服务：加载嵌入模型、连接 MySQL、构建 FAISS 索引……", flush=True)
    RUNTIME = RagRuntime()
    if RUNTIME.embedder_warning:
        print(f"⚠️  {RUNTIME.embedder_warning}", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), RagHandler)
    print(f"TCM RAG 微服务已启动: http://{args.host}:{args.port}", flush=True)
    print(
        f"  知识文档 {len(RUNTIME.store.documents)} 条 | 索引 {RUNTIME.store.backend} | "
        f"嵌入 {RUNTIME.embedder_backend} | 数据库 {RUNTIME.repository.db_status}",
        flush=True,
    )
    print(
        f"  LLM：mode={RUNTIME.llm.mode}, url={RUNTIME.llm.base_url or '(未配置 → 降级模拟)'}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
