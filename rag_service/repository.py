"""持久层：MySQL 存放向量化前的知识文档与链路 trace，断连时降级内存。

表结构见 database/init.sql；本模块在启动时自动建库建表（幂等），
并自动把种子 jsonl 导入 knowledge_documents（按 content_hash 幂等）。
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from pathlib import Path

try:
    import pymysql
except ImportError:  # pragma: no cover
    pymysql = None

from . import config


class DocumentRepository:
    def __init__(self, seed_files: list[Path] | None = None) -> None:
        self.seed_files = seed_files or list(config.SEED_FILES)
        self._documents: dict[str, dict] = {}
        self._traces: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.db_status = "not-configured"
        self.db_error = ""
        self._load_seed_memory()
        self._probe_database()

    # ---------- 连接 ----------

    def _connect(self):
        if pymysql is None:
            raise RuntimeError("PyMySQL 未安装（pip install pymysql）")
        return pymysql.connect(
            host=config.MYSQL_HOST,
            port=config.MYSQL_PORT,
            user=config.MYSQL_USER,
            password=config.MYSQL_PASSWORD,
            database=config.MYSQL_DATABASE,
            charset="utf8mb4",
            connect_timeout=2,
            read_timeout=5,
            write_timeout=5,
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )

    def _probe_database(self) -> None:
        if not config.MYSQL_ENABLED:
            self.db_status = "disabled"
            return
        try:
            with self._connect():
                pass
        except pymysql.err.OperationalError as exc:
            code = exc.args[0] if exc.args else 0
            if code in (1044, 1049):  # 库不存在 → 自动建库
                self._create_database()
            else:
                raise
        self._ensure_tables()
        self._sync_seed_database()
        self.db_status = "connected"
        self.db_error = ""

    def _create_database(self) -> None:
        if pymysql is None:
            raise RuntimeError("PyMySQL 未安装")
        connection = pymysql.connect(
            host=config.MYSQL_HOST,
            port=config.MYSQL_PORT,
            user=config.MYSQL_USER,
            password=config.MYSQL_PASSWORD,
            charset="utf8mb4",
            autocommit=True,
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "CREATE DATABASE IF NOT EXISTS lora "
                    "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
        finally:
            connection.close()

    def _ensure_tables(self) -> None:
        sql_file = config.PROJECT_DIR / "database" / "init.sql"
        statements = []
        if sql_file.exists():
            for raw in sql_file.read_text(encoding="utf-8").split(";"):
                stmt = "\n".join(
                    line for line in raw.splitlines()
                    if not line.strip().startswith("--") and line.strip() not in ("USE lora;",)
                ).strip()
                if stmt and not stmt.upper().startswith("CREATE DATABASE"):
                    statements.append(stmt)
        with self._connect() as connection, connection.cursor() as cursor:
            for stmt in statements:
                cursor.execute(stmt)

    # ---------- 种子数据 ----------

    def _load_seed_memory(self) -> None:
        """把种子 jsonl 读进内存（同时作为 MySQL 不可用时的兜底知识库）。"""
        for seed_file in self.seed_files:
            if not seed_file.exists():
                continue
            with seed_file.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    external_id = str(item.get("id", item.get("external_id", "")))
                    if not external_id:
                        continue
                    content = str(item.get("reference_answer", item.get("answer", item.get("content", ""))))
                    if not content:
                        continue
                    self._documents[external_id] = {
                        "external_id": external_id,
                        "title": str(item.get("question", item.get("title", external_id))),
                        "content": content,
                        "source": f"{seed_file.name}#{external_id}",
                        "metadata": {
                            "topic": item.get("topic", item.get("category", "")),
                            "review_status": "course-data-unreviewed",
                        },
                    }

    def _sync_seed_database(self) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            for document in self._documents.values():
                self._upsert_mysql(cursor, document)

    # ---------- 文档 CRUD ----------

    @staticmethod
    def _upsert_mysql(cursor, document: dict) -> None:
        content_hash = hashlib.sha256(document["content"].encode("utf-8")).hexdigest()
        cursor.execute(
            """
            INSERT INTO knowledge_documents
                (external_id, title, content, source, metadata, content_hash, enabled)
            VALUES (%s, %s, %s, %s, %s, %s, TRUE)
            ON DUPLICATE KEY UPDATE
                title=VALUES(title), content=VALUES(content), source=VALUES(source),
                metadata=VALUES(metadata), content_hash=VALUES(content_hash), enabled=TRUE
            """,
            (
                document["external_id"], document["title"], document["content"],
                document.get("source", ""),
                json.dumps(document.get("metadata", {}), ensure_ascii=False), content_hash,
            ),
        )

    def list_documents(self) -> list[dict]:
        if self.db_status == "connected":
            try:
                with self._connect() as connection, connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT external_id, title, content, source, metadata "
                        "FROM knowledge_documents WHERE enabled=TRUE ORDER BY id"
                    )
                    rows = cursor.fetchall()
                    for row in rows:
                        if isinstance(row.get("metadata"), str):
                            row["metadata"] = json.loads(row.get("metadata") or "{}")
                    return rows
            except Exception as exc:  # noqa: BLE001 —— 查询失败降级内存
                self.db_status = "fallback-memory"
                self.db_error = str(exc)
        with self._lock:
            return list(self._documents.values())

    def upsert(self, document: dict) -> dict:
        normalized = {
            "external_id": str(document.get("external_id", "")).strip(),
            "title": str(document.get("title", "")).strip(),
            "content": str(document.get("content", "")).strip(),
            "source": str(document.get("source", "manual")).strip(),
            "metadata": document.get("metadata", {}) or {},
        }
        if not all(normalized[key] for key in ("external_id", "title", "content")):
            raise ValueError("external_id、title 和 content 不能为空")
        with self._lock:
            self._documents[normalized["external_id"]] = normalized
        if self.db_status == "connected":
            with self._connect() as connection, connection.cursor() as cursor:
                self._upsert_mysql(cursor, normalized)
        return normalized

    # ---------- 链路 trace ----------

    def create_trace(
        self,
        query: str,
        augmented_prompt: str,
        rag_enabled: bool,
        rag_status: str,
        latency_ms: int,
        results: list[dict],
    ) -> str:
        trace_id = str(uuid.uuid4())
        trace = {
            "id": trace_id,
            "query_text": query,
            "augmented_prompt": augmented_prompt,
            "rag_enabled": rag_enabled,
            "rag_status": rag_status,
            "rag_latency_ms": latency_ms,
            "results": results,
            "outputs": {},
        }
        with self._lock:
            self._traces[trace_id] = trace

        if self.db_status == "connected":
            try:
                with self._connect() as connection, connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO query_traces
                            (id, query_text, augmented_prompt, rag_enabled, rag_status, rag_latency_ms)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (trace_id, query, augmented_prompt, rag_enabled, rag_status, latency_ms),
                    )
                    for result in results:
                        cursor.execute(
                            """
                            INSERT INTO rag_retrievals
                                (trace_id, rank_no, document_external_id, title, source, score, excerpt)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                trace_id, result["rank"], result["document_id"], result["title"],
                                result.get("source", ""), result["score"], result["content"][:2000],
                            ),
                        )
            except Exception as exc:  # noqa: BLE001
                self.db_status = "fallback-memory"
                self.db_error = str(exc)
        return trace_id

    def record_output(self, trace_id: str, output: dict) -> None:
        variant = str(output.get("variant", ""))
        if variant not in {"base", "lora"}:
            raise ValueError("variant 必须是 base 或 lora")
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is not None:
                trace.setdefault("outputs", {})[variant] = output

        if trace is None:
            if self.db_status == "connected":
                with self._connect() as connection, connection.cursor() as cursor:
                    cursor.execute("SELECT id FROM query_traces WHERE id=%s", (trace_id,))
                    if cursor.fetchone() is None:
                        raise KeyError("trace_id 不存在")
            else:
                raise KeyError("trace_id 不存在")

        if self.db_status == "connected":
            try:
                with self._connect() as connection, connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO model_outputs
                            (trace_id, variant, response, elapsed_seconds, character_count, inference_mode)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                            response=VALUES(response), elapsed_seconds=VALUES(elapsed_seconds),
                            character_count=VALUES(character_count), inference_mode=VALUES(inference_mode)
                        """,
                        (
                            trace_id, variant, str(output.get("response", "")),
                            float(output.get("elapsed_seconds", 0)),
                            int(output.get("character_count", 0)),
                            str(output.get("inference_mode", "unknown")),
                        ),
                    )
            except Exception as exc:  # noqa: BLE001
                self.db_status = "fallback-memory"
                self.db_error = str(exc)

    def get_trace(self, trace_id: str) -> dict | None:
        with self._lock:
            trace = self._traces.get(trace_id)
        if trace is not None:
            return trace
        if self.db_status == "connected":
            try:
                with self._connect() as connection, connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT id, query_text, augmented_prompt, rag_enabled, rag_status, "
                        "rag_latency_ms FROM query_traces WHERE id=%s",
                        (trace_id,),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        return None
                    cursor.execute(
                        "SELECT rank_no, document_external_id, title, source, score "
                        "FROM rag_retrievals WHERE trace_id=%s ORDER BY rank_no",
                        (trace_id,),
                    )
                    retrievals = cursor.fetchall()
                    cursor.execute(
                        "SELECT variant, response, elapsed_seconds, character_count, inference_mode "
                        "FROM model_outputs WHERE trace_id=%s",
                        (trace_id,),
                    )
                    outputs = {
                        row2["variant"]: row2 for row2 in cursor.fetchall()
                    }
                    return {
                        **row,
                        "results": [
                            {
                                "rank": item["rank_no"],
                                "document_id": item["document_external_id"],
                                "title": item["title"],
                                "source": item["source"],
                                "score": item["score"],
                            }
                            for item in retrievals
                        ],
                        "outputs": outputs,
                    }
            except Exception as exc:  # noqa: BLE001
                self.db_status = "fallback-memory"
                self.db_error = str(exc)
        return None
