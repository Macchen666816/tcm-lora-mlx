#!/usr/bin/env python3
"""检索权重调参：用 MRR 在标注查询集上选最优词汇权重。

标注方式：查询 → 期望命中的文档关键词（该关键词出现在标题或正文即视为相关文档）。
对每个候选权重计算 MRR@5，选出全局最优，避免「修好一个查询、弄坏另一个」。

用法：python scripts/tune_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_service import config  # noqa: E402
from rag_service.embedder import create_embedder  # noqa: E402
from rag_service.repository import DocumentRepository  # noqa: E402
from rag_service.vector_store import VectorStore  # noqa: E402

# (查询, 相关文档需包含的关键词)
CASES = [
    ("薄荷能治什么？", "薄荷"),
    ("薄荷的性味归经与主要功效", "薄荷"),
    ("孕妇用药有哪些禁忌？", "孕早期感冒可以自己抓几味中药喝"),
    ("孕早期感冒能自己抓药吗？", "孕早期感冒可以自己抓几味中药喝"),
    ("怀孕期间失眠能吃酸枣仁吗？", "酸枣仁"),
    ("生川乌磨粉每天吃一点可以吗？", "川乌"),
    ("马钱子粉怎么吃？", "马钱子"),
    ("吃药后全身起疹子按哪个穴位能缓解？", "疹子"),
]

WEIGHTS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
TOP_K = 5


def main() -> None:
    embedder, _, _ = create_embedder(config.EMBED_MODEL_NAME, config.EMBED_DEVICE, 32)
    documents = DocumentRepository().list_documents()
    store = VectorStore(config.INDEX_DIR, embedder)
    store.rebuild(documents)
    print(f"知识文档 {len(documents)} 条 | 检索器 {store.retriever}\n", flush=True)

    # 预计算两路分数（不随权重变化）
    prepared = []
    for query, keyword in CASES:
        query_vector = embedder.embed(query)
        vector_scores = store._vector_scores(query_vector, len(documents) * 2)
        lexical_scores = store._bm25_scores(query)
        relevant = {
            i for i, doc in enumerate(documents)
            if keyword in doc["title"] or keyword in doc["content"]
        }
        if not relevant:
            print(f"⚠️  跳过（知识库无相关文档）：{query}")
            continue
        prepared.append((query, keyword, vector_scores, lexical_scores, relevant))

    print(f"{'权重':>6} {'MRR@5':>8}  各查询命中最优排名")
    best = (None, -1.0)
    for weight in WEIGHTS:
        reciprocal_sum = 0.0
        ranks = []
        for query, keyword, vector_scores, lexical_scores, relevant in prepared:
            fused = store._fuse_with(vector_scores, lexical_scores, weight)
            order = sorted(range(len(documents)), key=lambda i: fused[i], reverse=True)
            rank = next(
                (position for position, index in enumerate(order[:TOP_K], start=1) if index in relevant),
                None,
            )
            ranks.append(rank or f">{TOP_K}")
            reciprocal_sum += 1 / rank if rank else 0.0
        mrr = reciprocal_sum / len(prepared)
        flag = ""
        if mrr > best[1]:
            best = (weight, mrr)
            flag = "  ← 当前最优"
        print(f"{weight:>6.1f} {mrr:>8.3f}  {ranks}{flag}", flush=True)

    print(f"\n结论：推荐 TCM_RAG_LEXICAL_WEIGHT={best[0]}（MRR@5={best[1]:.3f}）")


if __name__ == "__main__":
    main()
