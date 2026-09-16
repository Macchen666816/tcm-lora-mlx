#!/usr/bin/env python3
"""RAG 管理面板后端：静态页面 + 反向代理 RAG 微服务 API。

与 webapp（LoRA 并排对比界面）完全独立，端口也不同（默认 8091），
只服务于 RAG 微服务本身的可视化与知识文档维护。

启动：
    python rag_console/server.py --port 8091
    # 或：python -m rag_console.server
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

STATIC_DIR = Path(__file__).resolve().parent / "static"
RAG_URL = os.getenv("TCM_RAG_URL", "http://127.0.0.1:8090")
PROXY_TIMEOUT = 20

# 面板允许访问的 RAG 接口（POST 目标）
PROXY_POST_ROUTES = {"/documents", "/retrieve", "/prepare", "/index/rebuild"}
PROXY_GET_ROUTES = {"/health", "/documents"}


class ConsoleHandler(BaseHTTPRequestHandler):
    server_version = "RagConsole/1.0"

    def log_message(self, format_string: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}", flush=True)

    # ---------- 输出 ----------

    def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def _static(self, request_path: str) -> None:
        relative = "index.html" if request_path == "/" else request_path.lstrip("/")
        requested = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in requested.parents and requested != STATIC_DIR.resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not requested.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type, _ = mimetypes.guess_type(requested.name)
        self._send(
            requested.read_bytes(),
            f"{content_type or 'application/octet-stream'}; charset=utf-8",
        )

    # ---------- 代理 ----------

    def _proxy(self, rag_path: str, method: str) -> None:
        url = f"{RAG_URL}{rag_path}"
        data = None
        headers = {}
        if method == "POST":
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_048_576:
                self._json({"error": "请求内容大小不合法"}, HTTPStatus.BAD_REQUEST)
                return
            data = self.rfile.read(length)
            headers["Content-Type"] = "application/json"

        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=PROXY_TIMEOUT) as response:
                payload = response.read().decode("utf-8")
                status = HTTPStatus(response.status)
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            status = HTTPStatus(exc.code)
        except Exception as exc:  # noqa: BLE001 —— RAG 服务未启动是常见情况
            self._json(
                {
                    "error": f"无法连接 RAG 微服务（{RAG_URL}）：{exc}",
                    "hint": "请先启动：python -m rag_service.server",
                },
                HTTPStatus.BAD_GATEWAY,
            )
            return

        self._send(payload.encode("utf-8"), "application/json; charset=utf-8", status)

    # ---------- 路由 ----------

    def do_GET(self) -> None:
        target = urlparse(self.path)
        path = target.path
        if path.startswith("/api/"):
            rag_path = path.removeprefix("/api")
            if rag_path not in PROXY_GET_ROUTES:
                self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            if target.query:
                rag_path = f"{rag_path}?{target.query}"
            self._proxy(rag_path, "GET")
            return
        self._static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if not path.startswith("/api/"):
            self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        rag_path = path.removeprefix("/api")
        if rag_path not in PROXY_POST_ROUTES:
            self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        self._proxy(rag_path, "POST")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAG admin console")
    parser.add_argument("--host", default=os.getenv("TCM_CONSOLE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("TCM_CONSOLE_PORT", "8091")))
    parser.add_argument("--rag-url", default=RAG_URL)
    return parser.parse_args()


def main() -> None:
    global RAG_URL
    args = parse_args()
    RAG_URL = args.rag_url
    if not STATIC_DIR.exists():
        raise FileNotFoundError(f"缺少静态目录：{STATIC_DIR}")

    server = ThreadingHTTPServer((args.host, args.port), ConsoleHandler)
    print(f"RAG 管理面板: http://{args.host}:{args.port}", flush=True)
    print(f"  代理的 RAG 微服务: {RAG_URL}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
