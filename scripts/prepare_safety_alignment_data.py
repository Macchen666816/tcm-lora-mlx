#!/usr/bin/env python3
"""Build a safety-alignment mix without exposing adversarial test prompts."""

from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data_safety_alignment" / "seed_examples.jsonl"
GENERAL = ROOT / "data_processed" / "mlx"
OUTPUT = ROOT / "data_safety_alignment" / "mlx"
SYSTEM = (
    "你是中医药知识助手，回答应专业、准确、完整且简洁。你只提供一般知识，不替代医生诊断。"
    "涉及急症、孕妇、儿童、毒性药材、精确处方剂量或药物相互作用时，应明确风险，"
    "不提供可直接执行的危险方案，并建议寻求合适的专业医疗帮助。"
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safety_message(record: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": record["question"]},
            {"role": "assistant", "content": record["answer"]},
        ]
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    rng = random.Random(20260915)
    seeds = read_jsonl(SEED)
    safety_train = [safety_message(row) for row in seeds if row["split"] == "train"]
    safety_valid = [safety_message(row) for row in seeds if row["split"] == "valid"]
    general_train = read_jsonl(GENERAL / "train.jsonl")
    general_valid = read_jsonl(GENERAL / "valid.jsonl")

    # About 30% safety examples: enough to teach boundaries while retaining QA style.
    train = rng.sample(general_train, 240) + safety_train * 3
    valid = rng.sample(general_valid, 32) + safety_valid
    rng.shuffle(train)
    rng.shuffle(valid)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT / "train.jsonl", train)
    write_jsonl(OUTPUT / "valid.jsonl", valid)
    write_jsonl(OUTPUT / "test.jsonl", valid)
    print(json.dumps({"train": len(train), "valid": len(valid), "safety_train_unique": len(safety_train), "safety_valid_unique": len(safety_valid)}))


if __name__ == "__main__":
    main()
