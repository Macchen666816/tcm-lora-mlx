"""向量层：FAISS 存向量化后的向量 + BM25 词汇路，加权 RRF 融合。

为什么不是纯向量检索（实测结论，见 scripts/eval_retrieval.py）：
    paraphrase-multilingual-MiniLM-L12-v2 对「短查询 vs 长文档」的余弦值压缩严重——
    文档与自身标题的相似度只有 0.25，而同领域风格相似的无关文档能到 0.65，
    导致「薄荷能治什么？」检索不到刚录入的薄荷文档（排名第 13）。
    实测对比 7 种打分方案后，采用「分向量取最大 + bigram BM25 + 加权 RRF」，
    同批查询命中排名从 13 → 1，且不牺牲语义检索能力。

索引结构：每个文档进两条向量（标题、正文各一条），检索时取二者较大值。
    FAISS 位置 p → 文档下标 p // 2，0=标题向量，1=正文向量
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Iterable, Protocol

try:
    import faiss  # type: ignore
    import numpy as np
except ImportError:  # pragma: no cover
    faiss = None
    np = None

CJK_RANGE = ("\u3400", "\u9fff")


class EmbedderProtocol(Protocol):
    dimensions: int
    backend: str

    def embed(self, text: str) -> list[float]: ...


def _is_cjk(char: str) -> bool:
    return CJK_RANGE[0] <= char <= CJK_RANGE[1]


def bigrams(text: str) -> list[str]:
    """中文字符二元组（跨非中文字符断开）：比单字更能表达词。"""
    grams: list[str] = []
    current: list[str] = []
    for char in text:
        if _is_cjk(char):
            current.append(char)
        else:
            if len(current) > 1:
                chunk = "".join(current)
                grams.extend(chunk[i:i + 2] for i in range(len(chunk) - 1))
            current = []
    if len(current) > 1:
        chunk = "".join(current)
        grams.extend(chunk[i:i + 2] for i in range(len(chunk) - 1))
    return grams


class VectorStore:
    def __init__(
        self,
        index_dir: Path,
        embedder: EmbedderProtocol,
        lexical_weight: float = 4.0,
        rrf_k: int = 60,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
    ) -> None:
        self.index_dir = index_dir
        self.index_path = index_dir / "knowledge.faiss"
        self.metadata_path = index_dir / "knowledge.metadata.json"
        self.embedder = embedder
        self.documents: list[dict] = []
        self.title_vectors: list[list[float]] = []
        self.content_vectors: list[list[float]] = []
        self.index = None
        self.built_at: float | None = None
        self.build_ms: int | None = None
        self.lexical_weight = lexical_weight
        self.rrf_k = rrf_k
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        # BM25 统计量
        self._gram_counts: list[dict[str, int]] = []
        self._gram_lengths: list[int] = []
        self._doc_freq: dict[str, int] = {}
        self._average_length = 1.0

    # ---------- 属性 ----------

    @property
    def backend(self) -> str:
        return "faiss" if faiss is not None else "python-cosine"

    @property
    def retriever(self) -> str:
        return f"hybrid(vector+bm25-rrf, w={self.lexical_weight})"

    @property
    def dimensions(self) -> int:
        return self.embedder.dimensions

    # ---------- 嵌入 ----------

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if hasattr(self.embedder, "embed_batch"):
            return self.embedder.embed_batch(texts)
        return [self.embedder.embed(text) for text in texts]

    # ---------- 构建索引 ----------

    def rebuild(self, documents: Iterable[dict]) -> dict:
        started = time.monotonic()
        self.documents = list(documents)
        self.title_vectors = self._embed_texts([doc["title"] for doc in self.documents])
        self.content_vectors = self._embed_texts([doc["content"] for doc in self.documents])
        self._build_bm25_stats()
        self.index_dir.mkdir(parents=True, exist_ok=True)

        if faiss is not None and np is not None and self.documents:
            self.index = faiss.IndexFlatIP(self.embedder.dimensions)
            # 交错排列：偶数位=标题向量，奇数位=正文向量
            matrix = np.empty((len(self.documents) * 2, self.embedder.dimensions), dtype="float32")
            for i, (title_vector, content_vector) in enumerate(
                zip(self.title_vectors, self.content_vectors)
            ):
                matrix[i * 2] = title_vector
                matrix[i * 2 + 1] = content_vector
            self.index.add(matrix)
            faiss.write_index(self.index, str(self.index_path))
        else:
            self.index = None

        self.metadata_path.write_text(
            json.dumps(
                {
                    "embedder_backend": self.embedder.backend,
                    "retriever": self.retriever,
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
            "retriever": self.retriever,
            "embedder_backend": self.embedder.backend,
            "build_ms": self.build_ms,
        }

    def _build_bm25_stats(self) -> None:
        self._gram_counts = []
        self._gram_lengths = []
        self._doc_freq = {}
        for doc in self.documents:
            # 标题词权重 ×2：查询里的实词命中标题时应当强于仅在正文出现
            grams = bigrams(doc["title"]) * 2 + bigrams(doc["content"])
            counts: dict[str, int] = {}
            for gram in grams:
                counts[gram] = counts.get(gram, 0) + 1
            self._gram_counts.append(counts)
            self._gram_lengths.append(len(grams) or 1)
            for gram in counts:
                self._doc_freq[gram] = self._doc_freq.get(gram, 0) + 1
        total_length = sum(self._gram_lengths)
        self._average_length = total_length / len(self._gram_lengths) if self._gram_lengths else 1.0

    # ---------- 两路召回 ----------

    def _vector_scores(self, query_vector: list[float], pool: int) -> list[float]:
        """每篇文档取「标题向量 / 正文向量」中较大的余弦值。"""
        if self.index is not None and np is not None:
            size = min(pool, len(self.documents) * 2)
            scores, indices = self.index.search(
                np.asarray([query_vector], dtype="float32"), size
            )
            best = [float("-inf")] * len(self.documents)
            for position, score in zip(indices[0].tolist(), scores[0].tolist()):
                if position < 0:
                    continue
                doc_index = position // 2
                if score > best[doc_index]:
                    best[doc_index] = float(score)
            return [value if value != float("-inf") else 0.0 for value in best]

        return [
            max(
                sum(a * b for a, b in zip(query_vector, title_vector)),
                sum(a * b for a, b in zip(query_vector, content_vector)),
            )
            for title_vector, content_vector in zip(self.title_vectors, self.content_vectors)
        ]

    def _bm25_scores(self, query: str) -> list[float]:
        query_grams = bigrams(query)
        if not query_grams:
            return [0.0] * len(self.documents)
        total = len(self.documents)
        scores: list[float] = []
        for counts, length in zip(self._gram_counts, self._gram_lengths):
            score = 0.0
            for gram in query_grams:
                tf = counts.get(gram, 0)
                if not tf:
                    continue
                df = self._doc_freq.get(gram, 0)
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                score += idf * (tf * (self.bm25_k1 + 1)) / (
                    tf + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * length / self._average_length)
                )
            scores.append(score)
        return scores

    @staticmethod
    def _ranks(scores: list[float]) -> list[int]:
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranks = [0] * len(scores)
        for position, index in enumerate(order, start=1):
            ranks[index] = position
        return ranks

    def _fuse_with(
        self, vector_scores: list[float], lexical_scores: list[float], lexical_weight: float
    ) -> list[float]:
        """加权 RRF：小模型对短查询排序偏弱，给词汇路更高权重。"""
        vector_rank = self._ranks(vector_scores)
        lexical_rank = self._ranks(lexical_scores)
        return [
            1 / (self.rrf_k + vector_rank[i]) + lexical_weight / (self.rrf_k + lexical_rank[i])
            for i in range(len(vector_scores))
        ]

    def _fuse(self, vector_scores: list[float], lexical_scores: list[float]) -> list[float]:
        return self._fuse_with(vector_scores, lexical_scores, self.lexical_weight)

    # ---------- 检索 ----------

    def search(self, query: str, top_k: int, min_vector_score: float = 0.0) -> list[dict]:
        if not self.documents or top_k <= 0:
            return []
        query_vector = self.embedder.embed(query)
        pool = min(len(self.documents) * 2, max(top_k * 6, 40))

        vector_scores = self._vector_scores(query_vector, pool)
        lexical_scores = self._bm25_scores(query)
        fused_scores = self._fuse(vector_scores, lexical_scores)

        order = sorted(range(len(self.documents)), key=lambda i: fused_scores[i], reverse=True)

        results = []
        for index in order[: top_k * 2]:
            doc = self.documents[index]
            if vector_scores[index] < min_vector_score and lexical_scores[index] <= 0:
                continue  # 语义弱且无词汇命中 → 视为噪声
            results.append(
                {
                    "rank": len(results) + 1,
                    "document_id": doc["external_id"],
                    "title": doc["title"],
                    "source": doc.get("source", ""),
                    "score": round(float(vector_scores[index]), 6),
                    "fused_score": round(float(fused_scores[index]), 6),
                    "lexical_score": round(float(lexical_scores[index]), 4),
                    "content": doc["content"],
                    "metadata": doc.get("metadata", {}) or {},
                }
            )
            if len(results) >= top_k:
                break
        return results
