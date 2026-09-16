"""嵌入层：本地 sentence-transformers 语义嵌入，加载失败时降级为哈希嵌入。

首选模型：paraphrase-multilingual-MiniLM-L12-v2（384 维，多语，中文效果好，
模型小 ~120MB，纯 CPU 可跑）。降级嵌入仅保证链路不中断，语义质量明显更差，
通过 backend 字段暴露给调用方，绝不静默冒充真实语义检索。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable, Sequence

TOKEN_PATTERN = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9]+")


class HashingEmbedder:
    """离线兜底嵌入：字符/词 n-gram 哈希投影，无需下载任何模型。"""

    backend = "hashing-fallback"

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        tokens = TOKEN_PATTERN.findall(text.lower())
        features = tokens + [tokens[i] + tokens[i + 1] for i in range(len(tokens) - 1)]
        vector = [0.0] * self.dimensions
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            vector[value % self.dimensions] += -1.0 if value & 1 else 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector


class SentenceTransformerEmbedder:
    """真实语义嵌入：sentence-transformers + 本地 HuggingFace 模型。"""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer  # 延迟导入，失败可降级

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.dimensions = int(self.model.get_sentence_embedding_dimension())
        self.backend = f"sentence-transformers:{model_name}"

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,  # 归一化后内积 == 余弦相似度
            show_progress_bar=False,
        )
        return vectors.tolist()


def create_embedder(model_name: str, device: str, batch_size: int):
    """尝试加载语义嵌入模型；失败（未装包/无网络/无缓存）则降级。

    Returns:
        (embedder, backend, warning) —— warning 为空字符串表示真实模型加载成功。
    """
    try:
        embedder = SentenceTransformerEmbedder(model_name, device, batch_size)
        return embedder, embedder.backend, ""
    except Exception as exc:  # noqa: BLE001 —— 任何失败都必须降级而不是崩溃
        fallback = HashingEmbedder()
        return fallback, fallback.backend, f"语义嵌入模型加载失败，已降级哈希嵌入：{exc}"
