"""持久层：MySQL 存放三立场知识文档与链路 trace，断连时降级内存。

表结构见 database/init.sql（库 `lora`）：
- `rag_stance_documents` —— 向量化前的知识文档，带 stance 立场列
- `query_traces` / `rag_retrievals` / `model_outputs` —— 链路 trace 与输出

旧表 `knowledge_documents`（无立场字段）已在本版本删除。
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

STANCES = ("aligned", "neutral", "opposed")  # 同向 / 中立（无安全拦截）/ 反向（恶意诱导）


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
            if code in (1044, 1049):
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

    @staticmethod
    def _statements_from_sql(path: Path) -> list[str]:
        if not path.exists():
            return []
        statements = []
        for raw in path.read_text(encoding="utf-8").split(";"):
            lines = [
                line for line in raw.splitlines()
                if not line.strip().startswith("--") and line.strip().upper() != "USE LORA"
            ]
            statement = "\n".join(lines).strip()
            if statement and not statement.upper().startswith("CREATE DATABASE"):
                statements.append(statement)
        return statements

    def _ensure_tables(self) -> None:
        sql_file = config.PROJECT_DIR / "database" / "init.sql"
        with self._connect() as connection, connection.cursor() as cursor:
            for statement in self._statements_from_sql(sql_file):
                cursor.execute(statement)
            # 老库升级：trace 表补 rag_stance 列（MySQL 8 无 ADD COLUMN IF NOT EXISTS）
            cursor.execute(
                "SELECT COUNT(*) AS n FROM information_schema.columns "
                "WHERE table_schema = %s AND table_name = 'query_traces' "
                "AND column_name = 'rag_stance'",
                (config.MYSQL_DATABASE,),
            )
            if not cursor.fetchone()["n"]:
                cursor.execute(
                    "ALTER TABLE query_traces ADD COLUMN rag_stance VARCHAR(16) "
                    "NOT NULL DEFAULT 'all' AFTER rag_status"
                )
                cursor.execute("ALTER TABLE query_traces ADD KEY idx_stance (rag_stance)")
            cursor.execute(
                "SELECT COUNT(*) AS n FROM information_schema.columns "
                "WHERE table_schema = %s AND table_name = 'rag_retrievals' "
                "AND column_name = 'stance'",
                (config.MYSQL_DATABASE,),
            )
            if not cursor.fetchone()["n"]:
                cursor.execute(
                    "ALTER TABLE rag_retrievals ADD COLUMN stance VARCHAR(16) "
                    "NOT NULL DEFAULT '' AFTER source"
                )

    # ---------- 种子数据 ----------

    def _load_seed_memory(self) -> None:
        """把三立场数据集读进内存（同时作为 MySQL 不可用时的兜底知识库）。"""
        for seed_file in self.seed_files:
            if not seed_file.exists():
                continue
            with seed_file.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    external_id = str(item.get("external_id", item.get("id", "")))
                    content = str(item.get("content", item.get("answer", "")))
                    if not external_id or not content:
                        continue
                    self._documents[external_id] = {
                        "external_id": external_id,
                        "stance": str(item.get("stance", "aligned")),
                        "risk_level": str(item.get("risk_level", "")),
                        "intent_tag": str(item.get("intent_tag", "")),
                        "paired_id": str(item.get("paired_id", "")),
                        "title": str(item.get("title", item.get("question", external_id))),
                        "content": content,
                        "topic": str(item.get("topic", "")),
                        "origin": str(item.get("origin", seed_file.name)),
                        "origin_id": str(item.get("origin_id", "")),
                        "origin_split": str(item.get("origin_split", "")),
                        "exclusion_reason": str(item.get("exclusion_reason", "")),
                        "stance_note": str(item.get("stance_note", "")),
                        "content_hash": str(
                            item.get("content_hash")
                            or hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
                        ),
                        "source": f"{seed_file.name}#{external_id}",
                    }

    def _sync_seed_database(self) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            for document in self._documents.values():
                self._upsert_mysql(cursor, document)

    # ---------- 文档 CRUD ----------

    @staticmethod
    def _upsert_mysql(cursor, document: dict) -> None:
        cursor.execute(
            """
            INSERT INTO rag_stance_documents
                (external_id, stance, title, content, topic, origin, origin_id,
                 origin_split, exclusion_reason, stance_note, risk_level, intent_tag,
                 paired_id, content_hash, enabled)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE)
            ON DUPLICATE KEY UPDATE
                stance=VALUES(stance), title=VALUES(title), content=VALUES(content),
                topic=VALUES(topic), origin=VALUES(origin), origin_id=VALUES(origin_id),
                origin_split=VALUES(origin_split), exclusion_reason=VALUES(exclusion_reason),
                stance_note=VALUES(stance_note), risk_level=VALUES(risk_level),
                intent_tag=VALUES(intent_tag), paired_id=VALUES(paired_id),
                content_hash=VALUES(content_hash), enabled=TRUE
            """,
            (
                document["external_id"], document.get("stance", "aligned"),
                document["title"], document["content"],
                document.get("topic", ""), document.get("origin", ""),
                document.get("origin_id", ""), document.get("origin_split", ""),
                document.get("exclusion_reason", ""), document.get("stance_note", ""),
                document.get("risk_level", ""), document.get("intent_tag", ""),
                document.get("paired_id", ""), document.get("content_hash", ""),
            ),
        )

    def list_documents(self, stance: str | None = None) -> list[dict]:
        """stance=None 取全部；否则只取指定立场（aligned/ambiguous/opposed）。"""
        if self.db_status == "connected":
            try:
                sql = (
                    "SELECT external_id, stance, title, content, topic, origin, origin_id, "
                    "origin_split, exclusion_reason, stance_note, risk_level, intent_tag, "
                    "paired_id FROM rag_stance_documents WHERE enabled=TRUE"
                )
                params: tuple = ()
                if stance and stance != "all":
                    sql += " AND stance=%s"
                    params = (stance,)
                sql += " ORDER BY id"
                with self._connect() as connection, connection.cursor() as cursor:
                    cursor.execute(sql, params)
                    rows = cursor.fetchall()
                    for row in rows:
                        row["source"] = f"{row.get('origin', '')}#{row['external_id']}"
                    return rows
            except Exception as exc:  # noqa: BLE001 —— 查询失败降级内存
                self.db_status = "fallback-memory"
                self.db_error = str(exc)
        with self._lock:
            documents = list(self._documents.values())
        if stance and stance != "all":
            documents = [doc for doc in documents if doc.get("stance") == stance]
        return documents

    def upsert(self, document: dict) -> dict:
        stance = str(document.get("stance", "")).strip()
        if stance not in STANCES:
            raise ValueError("stance 必须是 aligned / ambiguous / opposed 之一")
        content = str(document.get("content", "")).strip()
        normalized = {
            "external_id": str(document.get("external_id", "")).strip(),
            "stance": stance,
            "title": str(document.get("title", "")).strip(),
            "content": content,
            "topic": str(document.get("topic", "manual")).strip(),
            "origin": str(document.get("origin", "manual")).strip(),
            "origin_id": str(document.get("origin_id", "")).strip(),
            "origin_split": str(document.get("origin_split", "")).strip(),
            "exclusion_reason": str(document.get("exclusion_reason", "")).strip(),
            "stance_note": str(document.get("stance_note", "")).strip(),
            "risk_level": str(document.get("risk_level", "")).strip(),
            "intent_tag": str(document.get("intent_tag", "")).strip(),
            "paired_id": str(document.get("paired_id", "")).strip(),
            "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest()[:16],
        }
        if not all(normalized[key] for key in ("external_id", "title", "content")):
            raise ValueError("external_id、title 和 content 不能为空")
        normalized["source"] = f"{normalized['origin']}#{normalized['external_id']}"
        with self._lock:
            self._documents[normalized["external_id"]] = normalized
        if self.db_status == "connected":
            with self._connect() as connection, connection.cursor() as cursor:
                self._upsert_mysql(cursor, normalized)
        return normalized

    def stance_counts(self) -> dict[str, int]:
        documents = self.list_documents()
        counts = {stance: 0 for stance in STANCES}
        for doc in documents:
            counts[doc.get("stance", "aligned")] = counts.get(doc.get("stance", "aligned"), 0) + 1
        return counts

    # ---------- 链路 trace ----------

    def create_trace(
        self,
        query: str,
        augmented_prompt: str,
        rag_enabled: bool,
        rag_status: str,
        latency_ms: int,
        results: list[dict],
        stance: str = "all",
    ) -> str:
        trace_id = str(uuid.uuid4())
        trace = {
            "id": trace_id,
            "query_text": query,
            "augmented_prompt": augmented_prompt,
            "rag_enabled": rag_enabled,
            "rag_status": rag_status,
            "rag_stance": stance,
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
                            (id, query_text, augmented_prompt, rag_enabled, rag_status,
                             rag_stance, rag_latency_ms)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (trace_id, query, augmented_prompt, rag_enabled, rag_status,
                         stance, latency_ms),
                    )
                    for result in results:
                        cursor.execute(
                            """
                            INSERT INTO rag_retrievals
                                (trace_id, rank_no, document_external_id, title, source,
                                 stance, score, excerpt)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            """,
                            (
                                trace_id, result["rank"], result["document_id"], result["title"],
                                result.get("source", ""), result.get("stance", ""),
                                result["score"], result["content"][:2000],
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
                        "rag_stance, rag_latency_ms FROM query_traces WHERE id=%s",
                        (trace_id,),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        return None
                    cursor.execute(
                        "SELECT rank_no, document_external_id, title, source, stance, score "
                        "FROM rag_retrievals WHERE trace_id=%s ORDER BY rank_no",
                        (trace_id,),
                    )
                    retrievals = cursor.fetchall()
                    cursor.execute(
                        "SELECT variant, response, elapsed_seconds, character_count, inference_mode "
                        "FROM model_outputs WHERE trace_id=%s",
                        (trace_id,),
                    )
                    outputs = {item["variant"]: item for item in cursor.fetchall()}
                    return {
                        **row,
                        "results": [
                            {
                                "rank": item["rank_no"],
                                "document_id": item["document_external_id"],
                                "title": item["title"],
                                "source": item["source"],
                                "stance": item.get("stance", ""),
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
