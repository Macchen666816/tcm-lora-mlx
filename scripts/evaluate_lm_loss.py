#!/usr/bin/env python3
"""Evaluate a saved MLX-LM model/adapter on a local split without training."""

from __future__ import annotations

import argparse
import json
import math
import types
from pathlib import Path

from mlx_lm import load
from mlx_lm.tuner.datasets import CacheDataset, load_dataset
from mlx_lm.tuner.trainer import evaluate
from mlx_lm.tuner.utils import load_adapters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = load(args.model)
    mask_prompt = False
    if args.adapter_path:
        adapter_config = json.loads(
            (Path(args.adapter_path) / "adapter_config.json").read_text(encoding="utf-8")
        )
        mask_prompt = bool(adapter_config.get("mask_prompt", False))
        load_adapters(model, args.adapter_path)

    dataset_args = types.SimpleNamespace(
        data=str(args.data),
        train=False,
        test=True,
        mask_prompt=mask_prompt,
    )
    _, _, test_set = load_dataset(dataset_args, tokenizer)
    selected = {"train": None, "valid": None, "test": test_set}[args.split]
    if args.split != "test":
        train_set, valid_set, _ = load_dataset(dataset_args, tokenizer)
        selected = train_set if args.split == "train" else valid_set

    loss = evaluate(
        model=model,
        dataset=CacheDataset(selected),
        batch_size=args.batch_size,
        num_batches=-1,
        max_seq_length=args.max_seq_length,
    )
    result = {
        "model": args.model,
        "adapter_path": args.adapter_path,
        "data": str(args.data),
        "split": args.split,
        "records": len(selected),
        "loss": round(loss, 6),
        "perplexity": round(math.exp(loss), 6),
        "mask_prompt": mask_prompt,
        "note": "Evaluation only; this script never updates or saves model weights.",
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
