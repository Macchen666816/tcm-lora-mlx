#!/usr/bin/env python3
"""Generate reproducible model outputs for core and safety evaluation sets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--system-prompt")
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260915)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    if args.system_prompt and args.system_prompt_file:
        raise SystemExit("Use only one of --system-prompt or --system-prompt-file")
    if args.system_prompt_file:
        args.system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()
    cases = read_jsonl(args.cases)
    model, tokenizer = load(args.model, adapter_path=args.adapter_path)
    sampler = make_sampler(temp=0.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_chars = 0
    reasoning_markers = 0
    started = time.monotonic()
    with args.output.open("w", encoding="utf-8") as handle:
        for index, case in enumerate(cases, start=1):
            messages = []
            system_prompt = args.system_prompt or case.get("system")
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": case["question"]})
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            mx.random.seed(args.seed)
            response = generate(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                sampler=sampler,
                verbose=False,
            ).strip()
            total_chars += len(response)
            if any(marker in response for marker in ("<think>", "嗯，用户", "我需要", "我得先")):
                reasoning_markers += 1

            result = dict(case)
            result["model"] = args.model
            result["adapter_path"] = args.adapter_path
            result["response"] = response
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(cases)}] {case.get('id', index)}: {len(response)} chars")

    elapsed = time.monotonic() - started
    print(
        json.dumps(
            {
                "records": len(cases),
                "average_response_chars": round(total_chars / max(len(cases), 1), 1),
                "reasoning_marker_records": reasoning_markers,
                "elapsed_seconds": round(elapsed, 1),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
