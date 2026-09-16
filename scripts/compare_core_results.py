#!/usr/bin/env python3
"""Compare saved core-benchmark outputs with transparent proxy metrics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def lcs_length(left: str, right: str) -> int:
    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        for index, right_char in enumerate(right, start=1):
            if left_char == right_char:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(candidate: str, reference: str) -> float:
    common = lcs_length(candidate, reference)
    precision = common / len(candidate) if candidate else 0.0
    recall = common / len(reference) if reference else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def metrics(path: Path) -> dict:
    rows = read_jsonl(path)
    responses = [row["response"] for row in rows]
    references = [row["reference_answer"] for row in rows]
    reasoning_markers = ("<think>", "嗯，用户", "我需要", "我得先")
    refusal_markers = ("我不能提供", "无法提供", "不能回答")
    return {
        "file": str(path),
        "records": len(rows),
        "average_response_chars": round(statistics.mean(map(len, responses)), 1),
        "average_reference_chars": round(statistics.mean(map(len, references)), 1),
        "character_rouge_l_f1": round(
            statistics.mean(rouge_l_f1(output, reference) for output, reference in zip(responses, references)),
            3,
        ),
        "reasoning_marker_records": sum(any(marker in text for marker in reasoning_markers) for text in responses),
        "refusal_marker_records": sum(any(marker in text for marker in refusal_markers) for text in responses),
    }


def main() -> None:
    args = parse_args()
    report = {"note": "Proxy metrics do not establish medical correctness.", "results": [metrics(path) for path in args.results]}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
