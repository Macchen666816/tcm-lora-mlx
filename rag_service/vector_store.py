"""向量层：FAISS 存放向量化后的向量，MySQL/JSON 存放 doc_id → 原文的映射。

设计要点：
- IndexFlatIP + 归一化向量 == 精确余弦相似度检索（库只有几千条，无需近似索引）；
- 索引可持久化到 index/ 目录，重启免重建（除非文档变更触发 rebuild）；
- faiss 未安装时降级为纯 Python 余弦（backend 字段如实标注）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Protocol

try:
    import faiss  # type: ignore
    import numpy as np
except ImportError:  # pragma: no cover
    faiss = None
    np = None


class EmbedderProtocol(Protocol):
    dimensions: int
    backend: str

    def embed(self, text: str) -> list[float]: ...


class VectorStore:
    def __init__(self, index_dir: Path, embedder: EmbedderProtocol) -> None:
        self.index_dir = index_dir
        self.index_path = index_dir / "knowledge.faiss"
        self.metadata_path = index_dir / "knowledge.metadata.json"
        self.embedder = embedder
        self.documents: list[dict] = []
        self.vectors: list[list[float]] = []
        self.index = None
        self.built_at: float | None = None
        self.build_ms: int | None = None

    @property
    def backend(self) -> str:
        return "faiss" if faiss is not None else "python-cosine"

    @property
    def dimensions(self) -> int:
        return self.embedder.dimensions

    def _embed_text(self, doc: dict) -> str:
        """文档的向量化输入：标题 + 正文。"""
        return f"{doc['title']}\n{doc['content']}"

    def rebuild(self, documents: Iterable[dict]) -> dict:
        """全量重建：MySQL 里的 enabled 文档 → 嵌入 → FAISS 索引 → 持久化。"""
        started = time.monotonic()
        self.documents = list(documents)
        texts = [self._embed_text(doc) for doc in self.documents]
        self.vectors = (
            self.embedder.embed_batch(texts)
            if hasattr(self.embedder, "embed_batch") and texts
            else [self.embedder.embed(text) for text in texts]
        )
        self.index_dir.mkdir(parents=True, exist_ok=True)

        if faiss is not None and np is not None and self.vectors:
            self.index = faiss.IndexFlatIP(self.embedder.dimensions)
            self.index.add(np.asarray(self.vectors, dtype="float32"))
            faiss.write_index(self.index, str(self.index_path))
        else:
            self.index = None

        self.metadata_path.write_text(
            json.dumps(
                {
                    "embedder_backend": self.embedder.backend,
                    "dimensions": self.embedder.dimensions,
                    "document_count": len(self.documents),
                    "document_ids": [doc["external_id"] for doc in self.documents],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.built_at = time.time()
        self.build_ms = int((time.monotonic() - started) * 1000)
        return {
            "indexed_documents": len(self.documents),
            "index_backend": self.backend,
            "embedder_backend": self.embedder.backend,
            "build_ms": self.build_ms,
        }

    def search(self, query: str, top_k: int) -> list[dict]:
        if not self.documents or top_k <= 0:
            return []
        query_vector = self.embedder.embed(query)

        if self.index is not None and np is not None:
            scores, indices = self.index.search(
                np.asarray([query_vector], dtype="float32"),
                min(top_k, len(self.documents)),
            )
            pairs = list(zip(indices[0].tolist(), scores[0].tolist()))
        else:
            scored = [
                (i, sum(a * b for a, b in zip(query_vector, vec))) for i, vec in enumerate(self.vectors)
            ]
            pairs = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

        results = []
        for rank, (position, score) in enumerate(pairs, start=1):
            if position < 0 or position >= len(self.documents):
                continue
            doc = self.documents[position]
            results.append(
                {
                    "rank": rank,
                    "document_id": doc["external_id"],
                    "title": doc["title"],
                    "source": doc.get("source", ""),
                    "score": round(float(score), 6),
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {}) or {},
                }
            )
        return results
