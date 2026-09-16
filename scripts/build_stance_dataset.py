#!/usr/bin/env python3
"""构建三立场 RAG 数据集：同向 / 模糊 / 反向（等量、按主题分层）。

实验目的
--------
LoRA 已被微调出一种明确立场（传统中医知识问答、安全优先、拒绝可执行剂量、
不把无来源结论当事实）。本脚本把 RAG 知识库切成三等份，使检索到的资料
分别与该立场 **同向 / 模糊 / 反向**，从而测量「RAG 资料是否会把 LoRA 带偏」。

三立场的数据来源与依据
----------------------
| 立场 | 来源 | 依据 |
|---|---|---|
| aligned 同向 | `mlx` test 分片（未参与训练）+ 核心基准题 + 安全示范 | 与微调同分布、同立场，且**不泄漏训练数据** |
| ambiguous 模糊 | `excluded.jsonl` 中 out_of_scope 任务与信息不足样本 | 话题相关但**不表达可被引用的立场**（只给标题/只做分类/只改写） |
| opposed 反向 | `safety_quarantine` + `research_quarantine` | 正是训练流程**因立场冲突剔除**的样本：缺安全提示的个人用药建议、无来源支撑的研究结论 |

⚠️ 安全声明：opposed 组含**不安全医疗建议原文**（来自本项目自身的数据审计隔离集，
非新造内容），仅用于课程设计的对照实验。加载后必须仅在实验模式下启用，
不得用于任何面向真实用户的问答服务。

用法
----
    python scripts/build_stance_dataset.py                # 每立场 500 条
    python scripts/build_stance_dataset.py --per-stance 300
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data_processed"
OUT_DIR = PROJECT_DIR / "data_rag_stance"
SEED = 20260916

# excluded.jsonl 中属于「话题相关但立场中性/信息不足」的剔除原因
AMBIGUOUS_REASONS = {
    "out_of_scope_task:title_generation",
    "out_of_scope_task:extraction_classification",
    "out_of_scope_task:long_form_generation",
    "out_of_scope_task:rewriting_editing",
    "hard:title_only_task_mismatch",
    "hard:heading_only_task_mismatch",
    "policy:very_short_output",
}

# safety_quarantine 中与立场冲突的核心原因（逆向立场）
OPPOSED_SAFETY_REASONS = {
    "policy:personal_medical_advice_without_safety_language",
    "policy:toxic_herb_without_safety_language",
    "policy:vulnerable_population_without_safety_language",
    "policy:serious_condition_without_safety_language",
    "policy:dosage_without_safety_language",
    "policy:unsafe_absolute_claim",
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"缺少数据文件：{path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_pools() -> dict[str, list[dict]]:
    """把三类来源整理成统一的候选池结构。"""
    pools: dict[str, list[dict]] = {"aligned": [], "ambiguous": [], "opposed": []}
    seen: set[str] = set()

    def add(stance: str, doc: dict) -> None:
        key = content_hash(f"{doc['question']}|{doc['answer']}")
        if key in seen:
            return
        seen.add(key)
        doc["content_hash"] = key
        pools[stance].append(doc)

    # ---------- 同向：test 分片（未训练）+ 核心基准题 + 安全示范 ----------
    lineage = [d for d in load_jsonl(DATA_DIR / "lineage.jsonl") if d["final_split"] == "test"]
    alpaca_test = load_jsonl(DATA_DIR / "alpaca" / "test.jsonl")
    if len(lineage) != len(alpaca_test):
        print(f"⚠️  lineage(test)={len(lineage)} 与 alpaca/test={len(alpaca_test)} 数量不一致，"
              "按较短长度对齐", flush=True)
    for meta, item in zip(lineage, alpaca_test):
        add("aligned", {
            "question": item["instruction"],
            "answer": item["output"],
            "topic": meta.get("topic", "unknown"),
            "origin": "mlx/test.jsonl",
            "origin_id": meta["id"],
            "origin_split": "test",
            "exclusion_reason": "",
            "stance_note": "与微调同分布同立场，且未进入训练集（无泄漏）",
        })
    for item in load_jsonl(DATA_DIR / "evaluation" / "core_benchmark.jsonl"):
        add("aligned", {
            "question": item["question"],
            "answer": item["reference_answer"],
            "topic": item.get("topic", "unknown"),
            "origin": "evaluation/core_benchmark.jsonl",
            "origin_id": item["id"],
            "origin_split": "test",
            "exclusion_reason": "",
            "stance_note": "固定核心基准题，作为参考答案",
        })
    for item in load_jsonl(PROJECT_DIR / "data_safety_alignment" / "seed_examples.jsonl"):
        add("aligned", {
            "question": item["question"],
            "answer": item["answer"],
            "topic": "safety_alignment",
            "origin": "data_safety_alignment/seed_examples.jsonl",
            "origin_id": item["id"],
            "origin_split": item.get("split", "train"),
            "exclusion_reason": "",
            "stance_note": "人工撰写的安全示范，微调立场的标杆表述",
        })

    # ---------- 模糊：话题相关但立场中性 / 信息不足 ----------
    for item in load_jsonl(DATA_DIR / "excluded.jsonl"):
        reasons = set(item["exclusion_reasons"])
        hit = reasons & AMBIGUOUS_REASONS
        if not hit:
            continue
        add("ambiguous", {
            "question": item["instruction"],
            "answer": item["output"],
            "topic": item.get("topic", "unknown"),
            "origin": "data_processed/excluded.jsonl",
            "origin_id": item["id"],
            "origin_split": item.get("split", "unknown"),
            "exclusion_reason": ",".join(sorted(hit)),
            "stance_note": "话题相关但不表达可引用立场（标题式/分类式/改写式/信息不足）",
        })

    # ---------- 反向：训练流程因立场冲突剔除的样本 ----------
    for item in load_jsonl(DATA_DIR / "safety_quarantine.jsonl"):
        reasons = set(item["exclusion_reasons"])
        hit = reasons & OPPOSED_SAFETY_REASONS
        if not hit:
            continue
        add("opposed", {
            "question": item["instruction"],
            "answer": item["output"],
            "topic": item.get("topic", "unknown"),
            "origin": "data_processed/safety_quarantine.jsonl",
            "origin_id": item["id"],
            "origin_split": item.get("split", "unknown"),
            "exclusion_reason": ",".join(sorted(hit)),
            "stance_note": "与「安全优先、不给可执行剂量」立场直接冲突",
        })
    for item in load_jsonl(DATA_DIR / "research_quarantine.jsonl"):
        reasons = set(item["exclusion_reasons"])
        if "policy:unverified_research_claim" not in reasons:
            continue
        add("opposed", {
            "question": item["instruction"],
            "answer": item["output"],
            "topic": item.get("topic", "unknown"),
            "origin": "data_processed/research_quarantine.jsonl",
            "origin_id": item["id"],
            "origin_split": item.get("split", "unknown"),
            "exclusion_reason": "policy:unverified_research_claim",
            "stance_note": "无来源支撑的研究结论，与「不把无来源结论当事实」立场冲突",
        })

    return pools


def stratify_sample(pools: dict[str, list[dict]], per_stance: int) -> dict[str, list[dict]]:
    """按同向组的主题分布分配配额，各立场等量抽样（确定性随机）。"""
    rng = random.Random(SEED)
    for docs in pools.values():
        rng.shuffle(docs)

    aligned_topics = collections.Counter(doc["topic"] for doc in pools["aligned"])
    total_aligned = sum(aligned_topics.values()) or 1

    def quota(topic: str) -> int:
        return int(round(per_stance * aligned_topics.get(topic, 0) / total_aligned))

    selected: dict[str, list[dict]] = {}
    for stance, docs in pools.items():
        by_topic: dict[str, list[dict]] = collections.defaultdict(list)
        for doc in docs:
            by_topic[doc["topic"]].append(doc)
        picked: list[dict] = []
        for topic, items in by_topic.items():
            picked.extend(items[: quota(topic)])
        # 配额未填满（该主题库存不足）→ 按剩余库存补齐
        if len(picked) < per_stance:
            chosen = {doc["content_hash"] for doc in picked}
            rest = [doc for doc in docs if doc["content_hash"] not in chosen]
            picked.extend(rest[: per_stance - len(picked)])
        selected[stance] = picked[:per_stance]
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="构建三立场 RAG 数据集")
    parser.add_argument("--per-stance", type=int, default=500, help="每个立场的文档数（默认 500）")
    args = parser.parse_args()

    pools = build_pools()
    print("候选池规模：", {k: len(v) for k, v in pools.items()}, flush=True)

    limit = min(len(v) for v in pools.values())
    per_stance = min(args.per_stance, limit)
    if per_stance < args.per_stance:
        print(f"⚠️  最小池只有 {limit} 条，每立场调整为 {per_stance} 条", flush=True)

    selected = stratify_sample(pools, per_stance)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = OUT_DIR / "rag_stance_dataset.jsonl"
    rows: list[dict] = []
    for stance in ("aligned", "ambiguous", "opposed"):
        docs = selected[stance]
        stance_rows: list[dict] = []
        for index, doc in enumerate(docs, start=1):
            stance_rows.append({
                "external_id": f"{stance}-{index:04d}-{doc['content_hash']}",
                "stance": stance,
                "title": doc["question"],
                "content": doc["answer"],
                "topic": doc["topic"],
                "origin": doc["origin"],
                "origin_id": doc["origin_id"],
                "origin_split": doc["origin_split"],
                "exclusion_reason": doc["exclusion_reason"],
                "stance_note": doc["stance_note"],
                "content_hash": doc["content_hash"],
            })
        rows.extend(stance_rows)
        with (OUT_DIR / f"{stance}.jsonl").open("w", encoding="utf-8") as handle:
            for row in stance_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with combined_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---------- 统计与数据集卡 ----------
    print(f"\n输出：{combined_path}（{len(rows)} 条，每立场 {per_stance} 条）")
    lines = ["# 三立场 RAG 数据集（消融实验用）", "",
             f"- 每立场文档数：**{per_stance}**，合计 **{len(rows)}**",
             f"- 随机种子：`{SEED}`（确定性抽样，可复现）",
             "- 生成脚本：`scripts/build_stance_dataset.py`", ""]
    for stance in ("aligned", "ambiguous", "opposed"):
        docs = selected[stance]
        topics = collections.Counter(doc["topic"] for doc in docs)
        origins = collections.Counter(doc["origin"] for doc in docs)
        reasons = collections.Counter(
            r for doc in docs for r in (doc["exclusion_reason"].split(",") if doc["exclusion_reason"] else [])
        )
        print(f"\n[{stance}] {len(docs)} 条")
        print("   主题:", dict(topics.most_common()))
        print("   来源:", dict(origins.most_common()))
        if reasons:
            print("   剔除原因:", dict(reasons.most_common(5)))
        lines += [f"## {stance}（{len(docs)} 条）", "",
                  "| 主题 | 条数 |", "|---|---:|"]
        lines += [f"| {k} | {v} |" for k, v in topics.most_common()]
        lines += ["", "| 来源 | 条数 |", "|---|---:|"]
        lines += [f"| {k} | {v} |" for k, v in origins.most_common()]
        if reasons:
            lines += ["", "| 剔除原因（立场冲突标签） | 条数 |", "|---|---:|"]
            lines += [f"| `{k}` | {v} |" for k, v in reasons.most_common()]
        lines.append("")

    (OUT_DIR / "DATASET_CARD.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "stance_stats.json").write_text(
        json.dumps(
            {
                "per_stance": per_stance,
                "total": len(rows),
                "seed": SEED,
                "pool_sizes": {k: len(v) for k, v in pools.items()},
                "topic_distribution": {
                    stance: dict(collections.Counter(d["topic"] for d in selected[stance]))
                    for stance in selected
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n数据集卡：{OUT_DIR / 'DATASET_CARD.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
