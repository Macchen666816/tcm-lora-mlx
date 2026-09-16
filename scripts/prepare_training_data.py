#!/usr/bin/env python3
"""Build reproducible MLX and Alpaca datasets from audited TCM records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


POLICY_REJECT_FLAGS = {
    "dosage_without_safety_language",
    "latin_text_fragment",
    "personal_medical_advice_without_safety_language",
    "serious_condition_without_safety_language",
    "toxic_herb_without_safety_language",
    "unverified_research_claim",
    "unsafe_absolute_claim",
    "very_short_output",
    "vulnerable_population_without_safety_language",
}

TOPIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("acupuncture_and_meridians", re.compile(r"针灸|针刺|穴位|经穴|经络|拔罐|推拿|艾灸")),
    (
        "materia_medica_and_formulas",
        re.compile(r"中药|药材|方剂|方药|药方|性味|归经|功效|配伍|煎煮|汤|丸|散|膏"),
    ),
    (
        "syndromes_and_theory",
        re.compile(r"证候|证型|病因病机|阴阳|五行|气血|阴虚|阳虚|气虚|血虚|气滞|血瘀|寒热|虚实"),
    ),
    ("modern_research", re.compile(r"现代研究|药理|作用机制|研究进展|临床研究|疗效")),
    ("clinical_conditions", re.compile(r"症状|疾病|病症|诊断|治疗|调理|疼痛|炎|癌|病")),
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-file", type=Path, default=project_dir / "data_audit" / "audit_records.jsonl"
    )
    parser.add_argument("--output-dir", type=Path, default=project_dir / "data_processed")
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--review-sample-per-group", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260915)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected object")
                records.append(value)
    return records


def normalized_for_dedup(value: str) -> str:
    return re.sub(r"[\s'\"“”‘’]+", "", value).lower()


def classify_topic(record: dict[str, Any]) -> str:
    text = f"{record['instruction']} {record['output']}"
    for topic, pattern in TOPIC_PATTERNS:
        if pattern.search(text):
            return topic
    return "general_tcm_knowledge"


def base_exclusion_reasons(record: dict[str, Any]) -> list[str]:
    audit = record["audit"]
    reasons: list[str] = []
    if audit["task_category"] != "knowledge_qa":
        reasons.append(f"out_of_scope_task:{audit['task_category']}")
    reasons.extend(f"hard:{reason}" for reason in audit["hard_exclusion_reasons"])
    reasons.extend(
        f"policy:{flag}" for flag in sorted(set(audit["flags"]) & POLICY_REJECT_FLAGS)
    )
    return sorted(set(reasons))


def remove_duplicates(
    records: list[dict[str, Any]], excluded: list[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    seen_instructions: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    seen_outputs: set[str] = set()
    for record in sorted(records, key=lambda item: (item["source_line"], item["id"])):
        instruction_key = normalized_for_dedup(record["instruction"])
        output_key = normalized_for_dedup(record["output"])
        pair_key = (instruction_key, output_key)
        reasons: list[str] = []
        if instruction_key in seen_instructions:
            reasons.append("dedup:instruction")
        if pair_key in seen_pairs:
            reasons.append("dedup:instruction_output_pair")
        if output_key in seen_outputs:
            reasons.append("dedup:output")
        if reasons:
            rejected = dict(record)
            rejected["exclusion_reasons"] = reasons
            excluded.append(rejected)
            continue
        seen_instructions.add(instruction_key)
        seen_pairs.add(pair_key)
        seen_outputs.add(output_key)
        accepted.append(record)
    return accepted


def stable_validation_split(
    records: list[dict[str, Any]], ratio: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_topic[record["topic"]].append(record)

    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for topic in sorted(by_topic):
        group = sorted(
            by_topic[topic],
            key=lambda item: hashlib.sha256(item["id"].encode("utf-8")).hexdigest(),
        )
        validation_count = max(1, round(len(group) * ratio)) if len(group) > 1 else 0
        valid.extend(group[:validation_count])
        train.extend(group[validation_count:])
    return sorted(train, key=lambda item: item["id"]), sorted(valid, key=lambda item: item["id"])


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            count += 1
    return count


def mlx_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": record["input"]},
            {"role": "user", "content": record["instruction"]},
            {"role": "assistant", "content": record["output"]},
        ]
    }


def alpaca_record(record: dict[str, Any]) -> dict[str, str]:
    return {key: record[key] for key in ("instruction", "input", "output")}


def percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * quantile)
    return ordered[index]


def split_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    output_lengths = [len(record["output"]) for record in records]
    topics = Counter(record["topic"] for record in records)
    return {
        "records": len(records),
        "topics": dict(sorted(topics.items())),
        "output_chars": {
            "min": min(output_lengths, default=0),
            "median": percentile(output_lengths, 0.5),
            "p90": percentile(output_lengths, 0.9),
            "max": max(output_lengths, default=0),
        },
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_for_review(
    splits: dict[str, list[dict[str, Any]]], per_group: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    for split in sorted(splits):
        by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in splits[split]:
            by_topic[record["topic"]].append(record)
        for topic in sorted(by_topic):
            candidates = by_topic[topic]
            chosen.extend(rng.sample(candidates, min(per_group, len(candidates))))
    return sorted(chosen, key=lambda item: (item["final_split"], item["topic"], item["id"]))


def write_review_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = (
        "decision",
        "medical_review_notes",
        "id",
        "final_split",
        "topic",
        "source_line",
        "instruction",
        "output",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "decision": "",
                    "medical_review_notes": "",
                    **{field: record[field] for field in fieldnames if field not in {"decision", "medical_review_notes"}},
                }
            )


def benchmark_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "topic": record["topic"],
        "source_line": record["source_line"],
        "system": record["input"],
        "question": record["instruction"],
        "reference_answer": record["output"],
    }


def data_card(manifest: dict[str, Any]) -> str:
    split_rows = []
    for split in ("train", "valid", "test"):
        stats = manifest["final_splits"][split]
        split_rows.append(
            f"| {split} | {stats['records']} | {stats['output_chars']['min']} | "
            f"{stats['output_chars']['median']} | {stats['output_chars']['p90']} | "
            f"{stats['output_chars']['max']} |"
        )

    exclusion_rows = [
        f"| `{reason}` | {count} |" for reason, count in manifest["exclusions_by_reason"].items()
    ]
    return "\n".join(
        [
            "# 中医问答训练数据卡",
            "",
            "> 生成日期：2026-09-15。该数据包用于课程实验，不构成医疗知识库或诊疗依据。",
            "",
            "## 范围",
            "",
            "最终数据只保留中医知识问答。标题生成、长文写作、术语改写、抽取分类等混合任务被移出训练范围。",
            "原始回答未被自动改写；规则无法证明医学事实正确，因此保留了来源追踪和抽检表。",
            "",
            "## 最终划分",
            "",
            "| 划分 | 条数 | 输出最短 | 输出中位 | 输出 P90 | 输出最长 |",
            "|---|---:|---:|---:|---:|---:|",
            *split_rows,
            "",
            "- `train` 与 `valid` 均来自原始训练集，并按主题做确定性 9:1 划分。",
            "- `test` 只来自原始测试集，没有进入训练或验证集。",
            "- 三份数据均已做规范化去重；测试集与训练/验证集无重复问题。",
            "",
            "## 排除策略",
            "",
            "| 原因 | 条数 |",
            "|---|---:|",
            *exclusion_rows,
            "",
            "安全策略会隔离以下回答：针对个人症状给出用药但缺少就医提示、特殊人群缺少安全提示、涉及潜在毒性药材却无警示、剂量问题缺少安全约束。",
            "短于 40 字不是自动删除条件；只有极短回答、占位回答和标题式错配会被排除。",
            "",
            "## 文件",
            "",
            "- `mlx/*.jsonl`：MLX-LM 对话格式。",
            "- `alpaca/*.jsonl`：保留原始三字段格式，便于其他训练框架使用。",
            "- `lineage.jsonl`：最终样本与原始行号、主题、审计标记的映射。",
            "- `excluded.jsonl`：全部未入选样本及明确排除原因。",
            "- `safety_quarantine.jsonl`：因医疗安全规则被隔离的样本。",
            "- `research_quarantine.jsonl`：要求现代研究、药理或疗效证据但缺少可核查来源的样本。",
            "- `medical_spotcheck.csv`：分层抽检表，供后续人工或专业人员填写。",
            "- `evaluation/core_benchmark.jsonl`：仅从测试集抽取的固定主题基准题，不进入训练。",
            "- `manifest.json`：统计、规则版本和文件校验值。",
            "",
            "## 已知限制",
            "",
            "自动处理可以解决格式、任务混杂、重复和显式安全措辞问题，但无法系统判断药性、归经、方剂配伍及现代研究结论是否正确。",
            "因此本数据包应称为“规则清洗后的训练候选集”，不能称为“医学专家审核数据集”。",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_ratio < 0.5:
        raise ValueError("--validation-ratio must be between 0 and 0.5")
    if args.review_sample_per_group < 1:
        raise ValueError("--review-sample-per-group must be at least 1")

    audited = read_jsonl(args.audit_file)
    excluded: list[dict[str, Any]] = []
    eligible_by_source: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    for record in audited:
        reasons = base_exclusion_reasons(record)
        enriched = dict(record)
        enriched["topic"] = classify_topic(record)
        if reasons:
            enriched["exclusion_reasons"] = reasons
            excluded.append(enriched)
        else:
            eligible_by_source[record["split"]].append(enriched)

    source_train = remove_duplicates(eligible_by_source["train"], excluded, "train")
    source_test = remove_duplicates(eligible_by_source["test"], excluded, "test")
    train, valid = stable_validation_split(source_train, args.validation_ratio)

    train_validation_instructions = {
        normalized_for_dedup(record["instruction"]) for record in train + valid
    }
    train_validation_outputs = {normalized_for_dedup(record["output"]) for record in train + valid}
    test: list[dict[str, Any]] = []
    for record in source_test:
        reasons: list[str] = []
        if normalized_for_dedup(record["instruction"]) in train_validation_instructions:
            reasons.append("leakage:instruction_overlap_with_train")
        if normalized_for_dedup(record["output"]) in train_validation_outputs:
            reasons.append("leakage:output_overlap_with_train")
        if reasons:
            rejected = dict(record)
            rejected["exclusion_reasons"] = reasons
            excluded.append(rejected)
        else:
            test.append(record)

    splits = {"train": train, "valid": valid, "test": test}
    for split, records in splits.items():
        for record in records:
            record["final_split"] = split

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        write_jsonl(args.output_dir / "mlx" / f"{split}.jsonl", map(mlx_record, records))
        write_jsonl(args.output_dir / "alpaca" / f"{split}.jsonl", map(alpaca_record, records))

    lineage = sorted(
        (
            {
                "id": record["id"],
                "source_split": record["split"],
                "source_line": record["source_line"],
                "final_split": record["final_split"],
                "topic": record["topic"],
                "audit_flags": record["audit"]["flags"],
            }
            for records in splits.values()
            for record in records
        ),
        key=lambda item: (item["final_split"], item["id"]),
    )
    write_jsonl(args.output_dir / "lineage.jsonl", lineage)
    excluded = sorted(excluded, key=lambda item: (item["split"], item["source_line"]))
    write_jsonl(args.output_dir / "excluded.jsonl", excluded)

    safety_reason_prefixes = {
        f"policy:{flag}"
        for flag in POLICY_REJECT_FLAGS
        if "safety" in flag or "advice" in flag or "toxic" in flag or "vulnerable" in flag
    }
    safety_quarantine = [
        record
        for record in excluded
        if set(record["exclusion_reasons"]) & safety_reason_prefixes
    ]
    write_jsonl(args.output_dir / "safety_quarantine.jsonl", safety_quarantine)
    research_quarantine = [
        record
        for record in excluded
        if "policy:unverified_research_claim" in record["exclusion_reasons"]
    ]
    write_jsonl(args.output_dir / "research_quarantine.jsonl", research_quarantine)

    review_sample = sample_for_review(splits, args.review_sample_per_group, args.seed)
    write_review_csv(args.output_dir / "medical_spotcheck.csv", review_sample)
    benchmark = sample_for_review({"test": test}, 10, args.seed + 1)
    write_jsonl(
        args.output_dir / "evaluation" / "core_benchmark.jsonl",
        map(benchmark_record, benchmark),
    )

    exclusion_counts = Counter(reason for record in excluded for reason in record["exclusion_reasons"])
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generated_on": "2026-09-15",
        "scope": "Chinese traditional medicine knowledge question answering",
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "policy_reject_flags": sorted(POLICY_REJECT_FLAGS),
        "source_records": Counter(record["split"] for record in audited),
        "final_splits": {split: split_stats(records) for split, records in splits.items()},
        "excluded_records": len(excluded),
        "safety_quarantine_records": len(safety_quarantine),
        "research_quarantine_records": len(research_quarantine),
        "medical_spotcheck_records": len(review_sample),
        "core_benchmark_records": len(benchmark),
        "exclusions_by_reason": dict(sorted(exclusion_counts.items())),
        "source_sha256": {},
        "output_sha256": {},
    }
    project_dir = args.audit_file.parent.parent
    for name in ("chinese_medical_train.jsonl", "chinese_medical_test.jsonl"):
        path = project_dir / "data" / name
        manifest["source_sha256"][name] = sha256_file(path)
    for relative in (
        "mlx/train.jsonl",
        "mlx/valid.jsonl",
        "mlx/test.jsonl",
        "alpaca/train.jsonl",
        "alpaca/valid.jsonl",
        "alpaca/test.jsonl",
        "lineage.jsonl",
        "excluded.jsonl",
        "safety_quarantine.jsonl",
        "research_quarantine.jsonl",
        "medical_spotcheck.csv",
        "evaluation/core_benchmark.jsonl",
    ):
        path = args.output_dir / relative
        manifest["output_sha256"][relative] = sha256_file(path)

    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    (args.output_dir / "DATA_CARD.md").write_text(data_card(manifest), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
