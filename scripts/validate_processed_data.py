#!/usr/bin/env python3
"""Validate processed dataset schemas, checksums, lineage, and split isolation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=project_dir)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise AssertionError(f"{path}:{line_number}: blank line")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise AssertionError(f"{path}:{line_number}: expected object")
            records.append(value)
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(value: str) -> str:
    return re.sub(r"[\s'\"“”‘’]+", "", value).lower()


def validate_alpaca(records: list[dict[str, Any]], path: Path) -> None:
    required = {"instruction", "input", "output"}
    for index, record in enumerate(records, start=1):
        if set(record) != required:
            raise AssertionError(f"{path}:{index}: unexpected keys {sorted(record)}")
        for field in required:
            if not isinstance(record[field], str) or not record[field].strip():
                raise AssertionError(f"{path}:{index}: empty or non-string {field}")


def validate_mlx(records: list[dict[str, Any]], path: Path) -> None:
    expected_roles = ["system", "user", "assistant"]
    for index, record in enumerate(records, start=1):
        if set(record) != {"messages"} or not isinstance(record["messages"], list):
            raise AssertionError(f"{path}:{index}: invalid messages object")
        roles = [message.get("role") for message in record["messages"]]
        if roles != expected_roles:
            raise AssertionError(f"{path}:{index}: roles are {roles}")
        for message in record["messages"]:
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                raise AssertionError(f"{path}:{index}: empty message content")


def main() -> None:
    args = parse_args()
    processed_dir = args.project_dir / "data_processed"
    manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))

    alpaca_by_split: dict[str, list[dict[str, Any]]] = {}
    mlx_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "valid", "test"):
        alpaca_path = processed_dir / "alpaca" / f"{split}.jsonl"
        mlx_path = processed_dir / "mlx" / f"{split}.jsonl"
        alpaca_records = read_jsonl(alpaca_path)
        mlx_records = read_jsonl(mlx_path)
        validate_alpaca(alpaca_records, alpaca_path)
        validate_mlx(mlx_records, mlx_path)
        if len(alpaca_records) != len(mlx_records):
            raise AssertionError(f"{split}: MLX/Alpaca record count mismatch")
        if len(alpaca_records) != manifest["final_splits"][split]["records"]:
            raise AssertionError(f"{split}: manifest record count mismatch")
        for alpaca, mlx in zip(alpaca_records, mlx_records):
            contents = [message["content"] for message in mlx["messages"]]
            if contents != [alpaca["input"], alpaca["instruction"], alpaca["output"]]:
                raise AssertionError(f"{split}: MLX/Alpaca content mismatch")
        alpaca_by_split[split] = alpaca_records
        mlx_by_split[split] = mlx_records

    instruction_sets = {
        split: {normalize(record["instruction"]) for record in records}
        for split, records in alpaca_by_split.items()
    }
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = instruction_sets[left] & instruction_sets[right]
        if overlap:
            raise AssertionError(f"{left}/{right}: {len(overlap)} normalized instruction overlaps")

    all_instructions = [
        normalize(record["instruction"])
        for records in alpaca_by_split.values()
        for record in records
    ]
    if len(all_instructions) != len(set(all_instructions)):
        raise AssertionError("duplicate normalized instructions remain")

    system_prompts = {
        record["input"] for records in alpaca_by_split.values() for record in records
    }
    if len(system_prompts) != 1:
        raise AssertionError(f"expected one system prompt, found {len(system_prompts)}")

    lineage = read_jsonl(processed_dir / "lineage.jsonl")
    expected_lineage_count = sum(len(records) for records in alpaca_by_split.values())
    if len(lineage) != expected_lineage_count:
        raise AssertionError("lineage record count mismatch")
    lineage_ids = [record["id"] for record in lineage]
    if len(lineage_ids) != len(set(lineage_ids)):
        raise AssertionError("duplicate lineage ids")
    for record in lineage:
        expected_source = "test" if record["final_split"] == "test" else "train"
        if record["source_split"] != expected_source:
            raise AssertionError(f"{record['id']}: source/final split violation")

    raw_count = sum(manifest["source_records"].values())
    excluded = read_jsonl(processed_dir / "excluded.jsonl")
    if len(lineage) + len(excluded) != raw_count:
        raise AssertionError("included + excluded does not equal raw record count")

    for relative, expected_hash in manifest["output_sha256"].items():
        actual_hash = sha256_file(processed_dir / relative)
        if actual_hash != expected_hash:
            raise AssertionError(f"checksum mismatch: {relative}")
    for name, expected_hash in manifest["source_sha256"].items():
        actual_hash = sha256_file(args.project_dir / "data" / name)
        if actual_hash != expected_hash:
            raise AssertionError(f"source data changed: {name}")

    with (processed_dir / "medical_spotcheck.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        review_rows = list(csv.DictReader(handle))
    if len(review_rows) != manifest["medical_spotcheck_records"]:
        raise AssertionError("medical spot-check count mismatch")

    benchmark = read_jsonl(processed_dir / "evaluation" / "core_benchmark.jsonl")
    if len(benchmark) != manifest["core_benchmark_records"]:
        raise AssertionError("core benchmark count mismatch")
    test_questions = {record["instruction"] for record in alpaca_by_split["test"]}
    benchmark_questions = [record.get("question") for record in benchmark]
    if len(benchmark_questions) != len(set(benchmark_questions)):
        raise AssertionError("duplicate questions in core benchmark")
    if not set(benchmark_questions).issubset(test_questions):
        raise AssertionError("core benchmark contains a non-test question")

    adversarial_path = args.project_dir / "evaluation" / "adversarial_safety.jsonl"
    adversarial = read_jsonl(adversarial_path)
    adversarial_ids = [record.get("id") for record in adversarial]
    if len(adversarial_ids) != len(set(adversarial_ids)):
        raise AssertionError("duplicate adversarial test ids")
    for index, record in enumerate(adversarial, start=1):
        required = {
            "id",
            "category",
            "question",
            "expected_behaviors",
            "prohibited_behaviors",
        }
        if set(record) != required:
            raise AssertionError(f"{adversarial_path}:{index}: invalid fields")
        if not record["question"].strip():
            raise AssertionError(f"{adversarial_path}:{index}: empty question")
        if not record["expected_behaviors"] or not record["prohibited_behaviors"]:
            raise AssertionError(f"{adversarial_path}:{index}: empty behavior criteria")

    print(
        json.dumps(
            {
                "status": "ok",
                "records": {split: len(records) for split, records in alpaca_by_split.items()},
                "excluded": len(excluded),
                "lineage": len(lineage),
                "unique_system_prompts": len(system_prompts),
                "cross_split_instruction_overlap": 0,
                "checksums_verified": len(manifest["output_sha256"]) + len(manifest["source_sha256"]),
                "core_benchmark": len(benchmark),
                "adversarial_safety_tests": len(adversarial),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
