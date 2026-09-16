#!/usr/bin/env python3
"""生成三模式的 Query 集合，并验证「模式 ↔ 文档部分」的双射关系。

设计
----
- 每个模式（aligned / neutral / opposed）的 query 集合，来自**该模式自己的文档部分**：
  文档的问题是 q，则 q 就是该模式的一条 query，期望命中该文档本身。
- 因此「模式 i 的 query 集」与「模式 i 的文档部分」构成双射（每条 query 唯一对应一条文档）。
- 三立场同题子集（高风险 134 条）额外标记 `three_way=True`，用于三方受控对比。

产物
----
- `data_rag_stance/stance_queries.jsonl`      全部模式的全部 query（含期望文档 ID）
- `data_rag_stance/queries_aligned.jsonl` 等  按模式拆分
- `data_rag_stance/queries_three_way.jsonl`  三立场同题子集（134 条 × 3 模式）

验证（--verify）
----------------
用实际检索服务检查双射是否成立：以 query 检索、且**只在该模式的部分内检索**，
期望文档应排第 1。输出各模式的 top-1 命中率。

用法
----
    python scripts/build_stance_queries.py                # 只生成
    python scripts/build_stance_queries.py --verify       # 生成 + 双射验证
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data_rag_stance"
STANCES = ("aligned", "neutral", "opposed")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_queries() -> list[dict]:
    docs = load_jsonl(DATA_DIR / "rag_stance_dataset.jsonl")

    # 三立场同题集合（paired_id 在三侧都出现的那些）
    by_stance_pairs: dict[str, set[str]] = {stance: set() for stance in STANCES}
    for doc in docs:
        if doc.get("paired_id"):
            by_stance_pairs[doc["stance"]].add(doc["paired_id"])
    three_way = set.intersection(*by_stance_pairs.values()) if all(by_stance_pairs.values()) else set()

    queries: list[dict] = []
    counters: dict[str, int] = collections.defaultdict(int)
    for stance in STANCES:
        for doc in (d for d in docs if d["stance"] == stance):
            counters[stance] += 1
            queries.append({
                "query_id": f"{stance}-q{counters[stance]:04d}",
                "stance": stance,
                "question": doc["title"],
                "expected_doc_id": doc["external_id"],
                "topic": doc["topic"],
                "risk_level": doc["risk_level"],
                "three_way": bool(doc.get("paired_id")) and doc["paired_id"] in three_way,
                "paired_id": doc.get("paired_id", ""),
            })
    return queries


def write_outputs(queries: list[dict]) -> None:
    combined = DATA_DIR / "stance_queries.jsonl"
    with combined.open("w", encoding="utf-8") as handle:
        for row in queries:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    for stance in STANCES:
        rows = [q for q in queries if q["stance"] == stance]
        with (DATA_DIR / f"queries_{stance}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    three_way_rows = [q for q in queries if q["three_way"]]
    with (DATA_DIR / "queries_three_way.jsonl").open("w", encoding="utf-8") as handle:
        for row in three_way_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"总 query 数：{len(queries)}（每模式 400）")
    print(f"三立场同题子集：{len(three_way_rows)} 条 query（{len(three_way_rows)//3} 个问题 × 3 模式）")
    print(f"产物：{combined}、queries_<stance>.jsonl、queries_three_way.jsonl")


def verify(limit: int | None = None) -> int:
    """双射验证：以该模式的 query 只在该模式的部分内检索，期望文档应排第 1。"""
    sys.path.insert(0, str(PROJECT_DIR))
    from rag_service.server import RagRuntime  # 延迟导入，未验证时不需要模型

    print("\n初始化 Runtime（加载嵌入模型 + 索引）……", flush=True)
    runtime = RagRuntime()
    queries = load_jsonl(DATA_DIR / "stance_queries.jsonl")

    failures: list[str] = []
    for stance in STANCES:
        rows = [q for q in queries if q["stance"] == stance]
        if limit:
            rows = rows[:limit]
        hit = 0
        hits_at_3 = 0
        for row in rows:
            result = runtime.retrieve(row["question"], 3, stance)
            ids = [item["document_id"] for item in result["results"]]
            if ids and ids[0] == row["expected_doc_id"]:
                hit += 1
            if row["expected_doc_id"] in ids:
                hits_at_3 += 1
            # 同时确认没有串场
            if any(item["stance"] != stance for item in result["results"]):
                failures.append(f"{row['query_id']} 串场")
        total = len(rows) or 1
        print(f"  stance={stance:<9} top-1 双射命中 {hit}/{len(rows)} "
              f"({hit/total:.1%}) | top-3 命中 {hits_at_3}/{len(rows)} ({hits_at_3/total:.1%})", flush=True)

    if failures:
        print(f"❌ 串场异常 {len(failures)} 处：{failures[:5]}")
        return 1
    print("✅ 双射验证完成，无串场")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="生成三模式 query 集合并验证双射")
    parser.add_argument("--verify", action="store_true", help="生成后做双射验证（需加载模型）")
    parser.add_argument("--limit", type=int, default=None, help="验证时每模式抽检条数")
    args = parser.parse_args()

    queries = build_queries()
    write_outputs(queries)
    if args.verify:
        return verify(args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
