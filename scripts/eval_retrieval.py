#!/usr/bin/env python3
"""检索方案对比实验：定位并解决小模型相似度压缩问题。

背景：paraphrase-multilingual-MiniLM-L12-v2 对「短查询 vs 长文档」的余弦值压缩严重
（文档与自身标题仅 0.25），造成同领域风格相似的无关文档压过真正该命中的文档。

本脚本对比三种打分方案在同一批查询上的命中排名：
  A. 现状：cos(query, embed(title + "\n" + content))
  B. 分向量：max(cos(query, embed(title)), cos(query, embed(content)))
  C. 混合：B + 词汇重叠加成（查询中的实词在文档里出现则加分）

用法：python scripts/eval_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_service import config  # noqa: E402
from rag_service.embedder import create_embedder  # noqa: E402
from rag_service.repository import DocumentRepository  # noqa: E402
from rag_service.vector_store import VectorStore  # noqa: E402

# (查询, 期望命中文档标题包含的关键词)
CASES = [
    ("薄荷能治什么？", "薄荷"),
    ("薄荷的性味归经与主要功效", "薄荷"),
    ("孕妇用药有哪些禁忌？", "孕"),
    ("孕早期感冒能自己抓药吗？", "孕"),
]

LEXICAL_WEIGHT = 0.20


def lexical_overlap(query: str, doc_text: str) -> float:
    """查询字符（去标点）在文档中的覆盖率，用于修正小模型的相似度压缩。"""
    chars = {c for c in query if "\u4e00" <= c <= "\u9fff"}
    if not chars:
        return 0.0
    hit = sum(1 for c in chars if c in doc_text)
    return hit / len(chars)


def bigrams(text: str) -> list[str]:
    """中文字符二元组（跨标点断开），比单字更能表达词。"""
    chunks, current = [], []
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            current.append(char)
        else:
            if len(current) > 1:
                chunks.append("".join(current))
            current = []
    if len(current) > 1:
        chunks.append("".join(current))
    return [gram for chunk in chunks for gram in (chunk[i:i + 2] for i in range(len(chunk) - 1))]


def build_idf(documents: list[dict]) -> dict[str, float]:
    import math

    df: dict[str, int] = {}
    for doc in documents:
        for gram in set(bigrams(f"{doc['title']}{doc['content']}")):
            df[gram] = df.get(gram, 0) + 1
    total = len(documents)
    return {gram: math.log(1 + total / (1 + count)) for gram, count in df.items()}


def lexical_idf(query: str, doc_text: str, idf: dict[str, float]) -> float:
    """IDF 加权的 bigram 覆盖率：稀有词（薄荷）权重高，高频词（什么）权重低。"""
    grams = bigrams(query)
    if not grams:
        return 0.0
    total = sum(idf.get(gram, 1.0) for gram in grams)
    hit = sum(idf.get(gram, 1.0) for gram in grams if gram in doc_text)
    return hit / total if total else 0.0


def rrf(vector_scores: list[float], lexical_scores: list[float], k: int = 60) -> list[float]:
    """Reciprocal Rank Fusion：把两路检索的排名融合，规避分数尺度不可比的问题。"""
    def ranks(scores: list[float]) -> list[int]:
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        result = [0] * len(scores)
        for position, index in enumerate(order, start=1):
            result[index] = position
        return result

    vector_rank, lexical_rank = ranks(vector_scores), ranks(lexical_scores)
    return [1 / (k + vector_rank[i]) + 1 / (k + lexical_rank[i]) for i in range(len(vector_scores))]


def weighted_rrf(vector_scores: list[float], lexical_scores: list[float],
                 k: int = 60, lexical_weight: float = 2.0) -> list[float]:
    """加权 RRF：小模型对短查询排序弱，给词汇路更高权重。"""
    def ranks(scores: list[float]) -> list[int]:
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        result = [0] * len(scores)
        for position, index in enumerate(order, start=1):
            result[index] = position
        return result

    vector_rank, lexical_rank = ranks(vector_scores), ranks(lexical_scores)
    return [
        1 / (k + vector_rank[i]) + lexical_weight / (k + lexical_rank[i])
        for i in range(len(vector_scores))
    ]


def bm25_scores(query: str, documents: list[dict], k1: float = 1.2, b: float = 0.75) -> list[float]:
    """bigram BM25：小语料下比「覆盖率」更能突出稀有实词（如「薄荷」）。"""
    import math

    corpus = [bigrams(f"{d['title']}{d['content']}") for d in documents]
    lengths = [len(grams) for grams in corpus]
    average_length = sum(lengths) / len(lengths) if lengths else 1.0
    df: dict[str, int] = {}
    for grams in corpus:
        for gram in set(grams):
            df[gram] = df.get(gram, 0) + 1
    total = len(documents)
    query_grams = bigrams(query)

    scores = []
    for grams, length in zip(corpus, lengths):
        counts: dict[str, int] = {}
        for gram in grams:
            counts[gram] = counts.get(gram, 0) + 1
        score = 0.0
        for gram in query_grams:
            if gram not in counts:
                continue
            idf = math.log(1 + (total - df[gram] + 0.5) / (df[gram] + 0.5))
            tf = counts[gram]
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * length / average_length))
        scores.append(score)
    return scores


def main() -> None:
    embedder, _, _ = create_embedder(config.EMBED_MODEL_NAME, config.EMBED_DEVICE, 32)
    documents = DocumentRepository().list_documents()
    print(f"知识文档 {len(documents)} 条\n", flush=True)

    # 预计算：合并向量 / 标题向量 / 正文向量
    combined = [embedder.embed(f"{d['title']}\n{d['content']}") for d in documents]
    titles = [embedder.embed(d["title"]) for d in documents]
    contents = [embedder.embed(d["content"]) for d in documents]
    idf = build_idf(documents)

    def rank(scores: list[float], keyword: str) -> tuple[int, float, str]:
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for position, index in enumerate(order, start=1):
            if keyword in documents[index]["title"] or keyword in documents[index]["content"]:
                return position, scores[index], documents[index]["title"][:26]
        return -1, 0.0, "(未命中)"

    for query, keyword in CASES:
        q = embedder.embed(query)
        score_a = [sum(x * y for x, y in zip(q, v)) for v in combined]
        score_b = [max(sum(x * y for x, y in zip(q, t)), sum(x * y for x, y in zip(q, c)))
                   for t, c in zip(titles, contents)]
        score_c = [
            b + LEXICAL_WEIGHT * lexical_overlap(query, f"{d['title']}{d['content']}")
            for b, d in zip(score_b, documents)
        ]
        score_d = [
            b + LEXICAL_WEIGHT * lexical_idf(query, f"{d['title']}{d['content']}", idf)
            for b, d in zip(score_b, documents)
        ]
        # E. RRF：向量检索排名 + 词汇检索排名 融合（k=60，业界常用）
        score_e = rrf(score_b, [
            lexical_idf(query, f"{d['title']}{d['content']}", idf) for d in documents
        ])
        # F/G. BM25 词汇路 + 等权/加权 RRF
        bm25 = bm25_scores(query, documents)
        score_f = rrf(score_b, bm25)
        score_g = weighted_rrf(score_b, bm25, lexical_weight=2.0)
        score_h = weighted_rrf(score_b, bm25, lexical_weight=4.0)

        print(f"查询：{query}   期望命中含「{keyword}」的文档")
        for name, scores in (
            ("A 现状(合并向量)", score_a),
            ("B 分向量取最大", score_b),
            ("D 分向量+bigram-IDF", score_d),
            ("E +RRF(等权)", score_e),
            ("F +BM25-RRF(等权)", score_f),
            ("G +BM25-RRF(词2.0)", score_g),
            ("H +BM25-RRF(词4.0)", score_h),
        ):
            position, score, title = rank(scores, keyword)
            mark = "✅" if position == 1 else ("⚠️" if position > 0 else "❌")
            print(f"   {mark} {name:<18} 排名 {position:>2}  分数 {score:.4f}  {title}")
        print(flush=True)

    # 顺带验证 VectorStore 本身无 bug：命中条数与向量数一致
    store = VectorStore(config.INDEX_DIR, embedder)
    store.rebuild(documents)
    hits = store.search("薄荷能治什么？", 3)
    print(f"VectorStore 自检：文档 {len(store.documents)} / 向量 {len(store.vectors)} / 命中 {len(hits)}")


if __name__ == "__main__":
    main()
