#!/usr/bin/env python3
"""Local comparison server for the Qwen base model and the trained LoRA."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
LOCAL_MODEL_DIR = PROJECT_DIR / "models" / "Qwen2.5-1.5B-Instruct" / "mlx-4bit"
LEGACY_MODEL_DIR = WORKSPACE_DIR / "models" / "Qwen2.5-1.5B-Instruct" / "mlx-4bit"
MODEL_DIR = LOCAL_MODEL_DIR if LOCAL_MODEL_DIR.exists() else LEGACY_MODEL_DIR
ADAPTER_DIR = PROJECT_DIR / "outputs" / "qwen25-lora-safety"
SYSTEM_PROMPT_FILE = PROJECT_DIR / "evaluation" / "SAFETY_SYSTEM_PROMPT.txt"


class ModelRuntime:
    def __init__(self) -> None:
        self._models: dict[str, tuple] = {}
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self.system_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()

    @property
    def loaded_variants(self) -> list[str]:
        return sorted(self._models)

    def _load_variant(self, variant: str):
        if variant not in {"base", "lora"}:
            raise ValueError("variant must be 'base' or 'lora'")
        if variant in self._models:
            return self._models[variant]

        with self._load_lock:
            if variant not in self._models:
                adapter_path = str(ADAPTER_DIR) if variant == "lora" else None
                started = time.monotonic()
                model, tokenizer = load(str(MODEL_DIR), adapter_path=adapter_path)
                self._models[variant] = (model, tokenizer)
                print(f"Loaded {variant} model in {time.monotonic() - started:.2f}s", flush=True)
        return self._models[variant]

    def generate(self, variant: str, question: str, max_tokens: int) -> dict:
        model, tokenizer = self._load_variant(variant)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        with self._generate_lock:
            mx.random.seed(20260915)
            started = time.monotonic()
            response = generate(
                model,
                tokenizer,
                prompt,
                max_tokens=max_tokens,
                sampler=make_sampler(temp=0.0),
                verbose=False,
            ).strip()
            elapsed = time.monotonic() - started

        return {
            "variant": variant,
            "label": "Qwen2.5 基座" if variant == "base" else "中医 LoRA",
            "response": response,
            "elapsed_seconds": round(elapsed, 2),
            "character_count": len(response),
            "max_tokens": max_tokens,
        }


RUNTIME = ModelRuntime()


def validate_files() -> None:
    required = [
        MODEL_DIR / "model.safetensors",
        MODEL_DIR / "config.json",
        MODEL_DIR / "tokenizer.json",
        ADAPTER_DIR / "adapters.safetensors",
        ADAPTER_DIR / "adapter_config.json",
        SYSTEM_PROMPT_FILE,
        STATIC_DIR / "index.html",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


class AppHandler(BaseHTTPRequestHandler):
    server_version = "TCMCompare/1.0"

    def log_message(self, format_string: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}", flush=True)

    def _security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'",
        )

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path == "/" else unquote(request_path.lstrip("/"))
        requested = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in requested.parents and requested != STATIC_DIR.resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not requested.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        body = requested.read_bytes()
        content_type, _ = mimetypes.guess_type(requested.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type or 'application/octet-stream'}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self._security_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json(
                {
                    "status": "ok",
                    "device": str(mx.default_device()),
                    "loaded_variants": RUNTIME.loaded_variants,
                    "model_available": MODEL_DIR.exists(),
                    "adapter_available": ADAPTER_DIR.exists(),
                }
            )
            return
        if path == "/api/info":
            self._send_json(
                {
                    "base_model": "Qwen2.5-1.5B-Instruct · MLX 4-bit",
                    "adapter": "中医 LoRA · rank 8 · 安全补强版",
                    "training_records": 3061,
                    "test_loss": 1.369,
                    "test_perplexity": 3.931,
                    "core_rouge_before": 0.208,
                    "core_rouge_after": 0.336,
                }
            )
            return
        self._serve_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/generate":
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 16_384:
                raise ValueError("请求内容大小不合法")
            payload = json.loads(self.rfile.read(content_length))
            question = str(payload.get("question", "")).strip()
            variant = str(payload.get("variant", ""))
            max_tokens = int(payload.get("max_tokens", 256))

            if not question:
                raise ValueError("请输入问题")
            if len(question) > 1000:
                raise ValueError("问题不能超过 1000 个字符")
            if variant not in {"base", "lora"}:
                raise ValueError("未知模型类型")
            if max_tokens not in {128, 256, 384}:
                raise ValueError("输出长度参数不合法")

            self._send_json(RUNTIME.generate(variant, question, max_tokens))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception:
            traceback.print_exc()
            self._send_json({"error": "本地模型推理失败，请查看服务端日志"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local TCM LoRA comparison app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--eager", action="store_true", help="Load both models before accepting requests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_files()
    if args.eager:
        RUNTIME._load_variant("base")
        RUNTIME._load_variant("lora")
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"TCM comparison app: http://{args.host}:{args.port}", flush=True)
    print("Models load lazily on first use." if not args.eager else "Both models are ready.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
