#!/usr/bin/env python3
"""Download official Qwen2.5 weights from ModelScope and convert to MLX 4-bit."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_DIR / "models" / "Qwen2.5-1.5B-Instruct"
HF_DIR = MODEL_ROOT / "hf"
MLX_DIR = MODEL_ROOT / "mlx-4bit"
BASE_URL = "https://www.modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct/resolve/master"

FILES = {
    "config.json": "98d2ff8cc47488d08a2b0b3acf4eb99ef210779b42bd48605f6b8e36acdbf670",
    "configuration.json": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
    "generation_config.json": "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "model.safetensors": "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
    "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(filename: str, expected_sha256: str) -> None:
    target = HF_DIR / filename
    if target.exists() and sha256(target) == expected_sha256:
        print(f"verified: {filename}")
        return

    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "tcm-lora-mlx/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(f"{BASE_URL}/{filename}", headers=headers)

    print(f"downloading: {filename} (resume={offset} bytes)")
    with urlopen(request, timeout=60) as response:
        append = offset > 0 and response.status == 206
        with partial.open("ab" if append else "wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    partial.replace(target)

    actual = sha256(target)
    if actual != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {filename}: {actual}")
    print(f"verified: {filename}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    HF_DIR.mkdir(parents=True, exist_ok=True)
    for filename, checksum in FILES.items():
        download(filename, checksum)

    if args.download_only:
        return
    if (MLX_DIR / "model.safetensors").exists():
        print(f"MLX model already exists: {MLX_DIR}")
        return

    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_lm.convert",
            "--hf-path",
            str(HF_DIR),
            "--mlx-path",
            str(MLX_DIR),
            "--quantize",
            "--q-bits",
            "4",
            "--q-group-size",
            "64",
        ],
        check=True,
        cwd=PROJECT_DIR,
    )
    print(f"ready: {MLX_DIR}")


if __name__ == "__main__":
    main()
