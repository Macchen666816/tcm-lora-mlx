#!/usr/bin/env python3
"""Local comparison server for the Qwen base model and the trained LoRA.

可选地在推理前调用同伴的 RAG 微服务做「提示词增强」：

    用户提问 -> webapp -> [RAG 微服务 /prepare] -> 增强提示词 -> 本地推理 -> 前端展示

设计约束：
1. 拉回力实验固定使用十字中性 system prompt，六个面板共享同一句；部署用的强安全
   prompt 仍单独保留，不能把实验条件误写成生产安全策略。
2. RAG 服务不可达时自动降级为直答，本地推理永不因 RAG 失败而中断。
3. 同一立场下的 base / LoRA 共用增强提示词（单飞 + 同题 + 立场缓存），不同立场
   必须严格隔离，否则正向与负向资料串用会让实验失效。
4. RAG 失败后进入冷却期，冷却结束后自动重试 —— 服务恢复不需要重启 webapp。
5. 响应会回传 system_preset / system_prompt / full_prompt，
   界面可逐层展示「模型真正读到的内容」，避免把 RAG 前言误认成系统提示词。
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import threading
import time
import traceback
import urllib.error
import urllib.request
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
ADAPTER_DIR = PROJECT_DIR / "outputs" / "qwen25-lora-safety-internalized"
LEGACY_ADAPTER_DIR = PROJECT_DIR / "outputs" / "qwen25-lora-safety"
SYSTEM_PROMPT_FILE = PROJECT_DIR / "evaluation" / "SAFETY_SYSTEM_PROMPT.txt"
TRAINING_PROMPT_FILE = PROJECT_DIR / "evaluation" / "TRAINING_SYSTEM_PROMPT.txt"
MINIMAL_PROMPT_FILE = PROJECT_DIR / "evaluation" / "MINIMAL_SYSTEM_PROMPT.txt"
ROLE_ONLY_PROMPT_FILE = PROJECT_DIR / "evaluation" / "SYSTEM_PROMPT_MINIMAL.txt"

# 系统提示词预设。Web 六格固定共用 role_only；其他预设保留给离线审计与历史复现。
#   safety   —— 部署现状，训练数据里没出现过这句（只有安全续训里的弱化版本）
#   minimal  —— 只保留人设、不带任何安全约束的极简句，用于暴露"安全靠提示词托管"
#   training —— 与 3061 条训练数据完全一致的那句，保证训练/推理同分布
#   none     —— 不注入我们的提示词，模板会补 Qwen 官方默认 system
SYSTEM_PRESETS = {
    "safety": {
        "label": "部署安全版",
        "note": "当前生产用法；训练数据里没有这句，是部署时加的护栏",
        "file": SYSTEM_PROMPT_FILE,
    },
    "minimal": {
        "label": "极简句（无安全约束）",
        "note": "只保留人设与风格要求。实测在这个预设下模型会给出可执行的危险剂量 —— "
                "说明安全拦截是被提示词托管的，不是训练内化的",
        "file": MINIMAL_PROMPT_FILE,
    },
    "role_only": {
        "label": "实验中性句",
        "note": "拉回力实验固定条件；十个字符，不含安全要求",
        "file": ROLE_ONLY_PROMPT_FILE,
    },
    "training": {
        "label": "训练原句",
        "note": "与 3061 条训练数据的 system 完全一致，用于排除分布不一致的干扰",
        "file": TRAINING_PROMPT_FILE,
    },
    "none": {
        "label": "无自定义 system（走 Qwen 默认）",
        "note": "不注入我们的提示词。注意：Qwen 的 chat template 会自动填一段官方默认 system"
                "（英文），所以并不是真的「没有系统提示词」，界面会显示实际生效的内容",
        "file": None,
    },
}
DEFAULT_SYSTEM_PRESET = "role_only"
VALID_STANCES = {"aligned", "neutral", "opposed"}

# 从模板渲染结果里抠出 system 段，用于如实回报"模型实际读到的 system"
SYSTEM_BLOCK_RE = re.compile(r"^\s*<\|im_start\|>system\n(.*?)<\|im_end\|>", re.DOTALL)


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_number(name: str, default: float, cast=float):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return cast(default)
    try:
        return cast(float(raw))
    except (TypeError, ValueError):
        print(f"环境变量 {name}={raw!r} 不是合法数字，回退到默认值 {default}", flush=True)
        return cast(default)


# --- RAG 微服务接入配置（全部可用环境变量覆盖）-----------------------------
RAG_URL = os.getenv("TCM_RAG_URL", "http://192.168.108.82:8090").rstrip("/")
RAG_ENABLED = _env_flag("TCM_RAG_ENABLED", True)
RAG_TOP_K = max(1, min(8, int(_env_number("TCM_RAG_TOP_K", 3, int))))
RAG_TIMEOUT = _env_number("TCM_RAG_TIMEOUT", 5.0)          # 单次 /prepare 超时（秒）
RAG_HEALTH_TIMEOUT = _env_number("TCM_RAG_HEALTH_TIMEOUT", 3.0)
RAG_COOLDOWN = _env_number("TCM_RAG_COOLDOWN", 15.0)       # 失败后的冷却时间（秒）
RAG_CACHE_TTL = _env_number("TCM_RAG_CACHE_TTL", 300.0)    # 同题增强结果缓存（秒）
RAG_WRITEBACK = _env_flag("TCM_RAG_WRITEBACK", True)       # 回写模型输出到 traces


class ModelRuntime:
    def __init__(self) -> None:
        self._models: dict[str, tuple] = {}
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self.system_prompts = self._load_system_prompts()

    @staticmethod
    def _load_system_prompts() -> dict[str, str | None]:
        """按预设加载系统提示词；``None`` 表示该预设不注入 system message。"""
        loaded: dict[str, str | None] = {}
        for key, meta in SYSTEM_PRESETS.items():
            path = meta["file"]
            if path is None:
                loaded[key] = None
                continue
            if not path.exists():
                print(f"警告：系统提示词预设 {key} 缺少文件 {path}", flush=True)
                loaded[key] = None
                continue
            loaded[key] = path.read_text(encoding="utf-8").strip()
        return loaded

    def resolve_preset(self, preset: str | None) -> str:
        return preset if preset in self.system_prompts else DEFAULT_SYSTEM_PRESET

    def describe_presets(self) -> list[dict]:
        described = []
        for key, meta in SYSTEM_PRESETS.items():
            text = self.system_prompts.get(key)
            described.append(
                {
                    "key": key,
                    "label": meta["label"],
                    "note": meta["note"],
                    "file": str(meta["file"].relative_to(PROJECT_DIR)) if meta["file"] else None,
                    "text": text,
                    "character_count": len(text or ""),
                }
            )
        return described

    @property
    def loaded_variants(self) -> list[str]:
        return sorted(self._models)

    def _load_variant(self, variant: str):
        adapter_paths = {
            "base": None,
            "lora": ADAPTER_DIR,
            "legacy_lora": LEGACY_ADAPTER_DIR,
        }
        if variant not in adapter_paths:
            raise ValueError("variant must be 'base', 'lora' or 'legacy_lora'")
        if variant in self._models:
            return self._models[variant]

        with self._load_lock:
            if variant not in self._models:
                selected = adapter_paths[variant]
                adapter_path = str(selected) if selected else None
                started = time.monotonic()
                model, tokenizer = load(str(MODEL_DIR), adapter_path=adapter_path)
                self._models[variant] = (model, tokenizer)
                print(f"Loaded {variant} model in {time.monotonic() - started:.2f}s", flush=True)
        return self._models[variant]

    def generate(
        self, variant: str, question: str, max_tokens: int, system_preset: str | None = None
    ) -> dict:
        model, tokenizer = self._load_variant(variant)
        preset = self.resolve_preset(system_preset)
        system_text = self.system_prompts.get(preset)

        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": question})
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

        # 不传 system 时 Qwen 模板会自动补一段官方默认提示词，
        # 所以"实际生效的 system"要从渲染结果里读，不能只看我们注入了什么。
        effective = system_text
        if effective is None:
            matched = SYSTEM_BLOCK_RE.match(prompt)
            effective = matched.group(1).strip() if matched else None

        labels = {
            "base": "Qwen2.5 基座",
            "lora": "安全内化 LoRA",
            "legacy_lora": "旧版安全 LoRA",
        }
        return {
            "variant": variant,
            "label": labels[variant],
            "response": response,
            "elapsed_seconds": round(elapsed, 2),
            "character_count": len(response),
            "max_tokens": max_tokens,
            "system_preset": preset,
            "system_prompt": system_text,              # 我们注入的
            "system_prompt_effective": effective,      # 模型实际读到的
            # 模板渲染后的原始字符串，即模型真正读到的内容（含 <|im_start|> 等标记）
            "full_prompt": prompt,
            "inference_mode": "remote-webapp",
            "degraded": False,
        }


RUNTIME = ModelRuntime()


class RagClient:
    """把同伴的 RAG 微服务当作「提示词增强器」来用。

    - ``prepare()`` 永不抛异常：成功返回增强后的提示词，失败原样返回用户问题。
    - 上游返回字段做了别名兼容，避免对方微调字段名后前端直接白屏。
    - 同一 ``(question, top_k, stance)`` 走单飞 + 短期缓存，保证同一立场的
      base / LoRA 拿到同一份增强提示词和同一个 ``trace_id``。
    - ``rag_status`` 取值：retrieved / empty / degraded / disabled。
    """

    # 上游字段名兼容表（对方改字段名时不用动前端）
    RESULT_KEYS = ("results", "hits", "retrievals", "documents", "matches", "contexts", "items")
    TITLE_KEYS = ("title", "doc_title", "document_title", "name", "heading", "source_title", "file_name")
    SCORE_KEYS = ("score", "similarity", "similarity_score", "relevance", "distance", "rank_score")
    SOURCE_KEYS = ("source", "file", "file_path", "path", "uri", "url")
    DOC_ID_KEYS = ("document_id", "doc_id", "id", "chunk_id")
    RANK_KEYS = ("rank", "index", "position")
    SNIPPET_KEYS = ("snippet", "content", "text", "chunk", "preview", "passage")
    MAX_PROMPT_CHARS = 8000

    def __init__(
        self,
        base_url: str,
        enabled: bool,
        top_k: int,
        timeout: float,
        cooldown: float,
        cache_ttl: float,
        writeback: bool,
    ) -> None:
        self.base_url = base_url
        self.enabled = enabled
        self.top_k = top_k
        self.timeout = timeout
        self.cooldown = cooldown
        self.cache_ttl = cache_ttl
        self.writeback_enabled = writeback
        # RAG 默认是局域网微服务。显式绕过 macOS 的 HTTP(S) 代理，避免 192.168.*
        # 请求被本机代理接管后超时，误判为微服务离线。
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        self._lock = threading.Lock()
        # 单飞锁：拿锁后按 (question, top_k, stance) 复核缓存；同一立场的
        # base / LoRA 只会真正请求上游一次，不同立场仍严格隔离。
        self._prepare_lock = threading.Lock()
        self._cooldowns: dict[str, tuple[float, str]] = {}
        self._cache: dict[tuple, tuple[float, dict]] = {}
        self._health_cache: tuple[float, dict] | None = None
        self._stats = {"prepared": 0, "hit": 0, "degraded": 0, "cache_hit": 0}

    # ------------------------------------------------------------------ HTTP
    def _request_json(self, path: str, payload: dict | None, timeout: float) -> dict:
        url = f"{self.base_url}{path}"
        if payload is None:
            request = urllib.request.Request(url, method="GET")
        else:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        with self._opener.open(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError(f"上游返回的不是 JSON 对象：{type(data).__name__}")
        return data

    # ---------------------------------------------------------------- health
    def health(self, force: bool = False) -> dict:
        """探测 RAG 服务可用性。结果缓存 10 秒，避免前端轮询打爆对方服务。"""
        now = time.monotonic()
        with self._lock:
            if not force and self._health_cache and now - self._health_cache[0] < 10:
                return dict(self._health_cache[1])

        if not self.enabled:
            info = {
                "enabled": False,
                "reachable": False,
                "url": self.base_url,
                "detail": "服务端已通过 TCM_RAG_ENABLED=0 关闭检索增强",
            }
        else:
            try:
                data = self._request_json("/health", None, min(self.timeout, RAG_HEALTH_TIMEOUT))
                info = {
                    "enabled": True,
                    "reachable": True,
                    "url": self.base_url,
                    "status": data.get("status"),
                    "document_count": data.get("document_count"),
                    "index_backend": data.get("index_backend"),
                    "embedder_backend": data.get("embedder_backend"),
                    "database": data.get("database"),
                }
            except Exception as exc:  # noqa: BLE001 - 任何异常都只降级，不冒泡
                info = {
                    "enabled": True,
                    "reachable": False,
                    "url": self.base_url,
                    "detail": self._describe(exc),
                }

        with self._lock:
            self._health_cache = (time.monotonic(), info)
        return dict(info)

    # --------------------------------------------------------------- prepare
    def prepare(
        self,
        question: str,
        stance: str,
        top_k: int | None = None,
        use_cache: bool = True,
    ) -> tuple[str, dict]:
        """返回 ``(送去模型的提示词, 给前端的 rag 载荷)``。永不抛异常。"""
        top_k = max(1, min(8, int(top_k or self.top_k)))
        if stance not in VALID_STANCES:
            raise ValueError("stance 必须是 aligned / neutral / opposed")
        if not self.enabled:
            return question, self._static_payload(
                "disabled", "服务端已关闭检索增强（TCM_RAG_ENABLED=0）",
                question=question, stance=stance
            )

        cache_key = (question, top_k, stance)
        hit = self._take_cached(cache_key, use_cache, question)
        if hit is not None:
            return hit
        cooling = self._cooldown_payload(question, stance)
        if cooling is not None:
            return cooling

        # 单飞（single-flight）：同一缓存键只允许一个线程真的去打上游。
        # 六格会同时发出四个检索请求；同一 stance 的 base / LoRA 必须复用结果，
        # aligned / opposed 则必须得到不同结果，否则拉回力实验失效。
        # 拿锁后复核缓存，后来者直接复用先到者的结果。
        with self._prepare_lock:
            hit = self._take_cached(cache_key, use_cache, question)
            if hit is not None:
                return hit
            cooling = self._cooldown_payload(question, stance)
            if cooling is not None:
                return cooling

            started = time.monotonic()
            try:
                data = self._request_json(
                    "/prepare",
                    {"query": question, "top_k": top_k, "enabled": True, "stance": stance},
                    self.timeout,
                )
            except Exception as exc:  # noqa: BLE001 - 降级是设计的一部分
                detail = self._describe(exc)
                with self._lock:
                    self._cooldowns[stance] = (time.monotonic() + self.cooldown, detail)
                    self._stats["degraded"] += 1
                print(f"[RAG] 不可用，本轮降级直答：{detail}", flush=True)
                return question, self._static_payload(
                    "degraded",
                    f"无法连接 {self.base_url}，本轮直接使用原始问题",
                    detail,
                    round((time.monotonic() - started) * 1000, 1),
                    question=question,
                    stance=stance,
                )

            elapsed_ms = round((time.monotonic() - started) * 1000, 1)
            augmented = str(data.get("augmented_prompt") or "").strip()
            results = self._normalize_results(data)
            status = str(data.get("rag_status") or "").strip().lower()
            if status not in {"retrieved", "empty"}:
                status = "retrieved" if results else "empty"
            if status == "retrieved" and not augmented:
                # 上游声称命中却没给出增强提示词，按未命中处理，避免前端展示空壳
                status = "empty"

            prompt = augmented if augmented else question
            if len(prompt) > self.MAX_PROMPT_CHARS:
                prompt = prompt[: self.MAX_PROMPT_CHARS]

            payload = {
                "rag_status": status,
                "trace_id": data.get("trace_id"),
                "rag_latency_ms": data.get("rag_latency_ms", elapsed_ms),
                "top_k": top_k,
                "stance": str(data.get("stance") or stance),
                "results": results,
                "prompt": prompt,
                "augmented_prompt": prompt,
                "original_question": question,
                # injected=True 表示这一轮真的把检索内容注入了提示词
                "injected": bool(augmented != "" and status == "retrieved"),
                "service_url": self.base_url,
                "rag_cache": "miss",
            }

            with self._lock:
                self._stats["prepared"] += 1
                if status == "retrieved":
                    self._stats["hit"] += 1
                self._cache[cache_key] = (time.monotonic() + self.cache_ttl, payload)
                if len(self._cache) > 32:  # 控制内存占用，淘汰最快过期的条目
                    for stale in sorted(self._cache, key=lambda k: self._cache[k][0])[:8]:
                        self._cache.pop(stale, None)
                self._cooldowns.pop(stance, None)
            return prompt, payload

    def _take_cached(self, cache_key: tuple, use_cache: bool, question: str):
        """命中未过期的缓存则返回 ``(prompt, payload)``，否则返回 None。"""
        if not use_cache:
            return None
        with self._lock:
            cached = self._cache.get(cache_key)
            if not cached or cached[0] <= time.monotonic():
                return None
            payload = dict(cached[1])
            payload["rag_cache"] = "hit"
            self._stats["cache_hit"] += 1
        return payload.get("prompt") or question, payload

    def _cooldown_payload(self, question: str, stance: str):
        """处于失败冷却期内时返回降级载荷，否则返回 None。"""
        with self._lock:
            cooldown_until, last_error = self._cooldowns.get(stance, (0.0, ""))
            cooldown_left = cooldown_until - time.monotonic()
        if cooldown_left <= 0:
            return None
        return question, self._static_payload(
            "degraded",
            f"RAG 服务上一轮调用失败，{cooldown_left:.0f} 秒后自动重试",
            last_error,
            question=question,
            stance=stance,
        )

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def skip_payload(self, top_k: int | None = None, question: str = "") -> dict:
        """不启用检索增强时的载荷，用于六格对比中「无 RAG」的两个面板。"""
        payload = self._static_payload(
            "disabled", "本次对比未启用检索增强，模型直接作答",
            question=question, stance="none"
        )
        payload["top_k"] = max(1, min(8, int(top_k or self.top_k)))
        return payload

    # ------------------------------------------------------------- writeback
    def writeback(self, trace_id, variant: str, result: dict) -> None:
        """后台线程回写模型输出，补全 query_traces -> rag_retrievals -> model_outputs 证据链。"""
        if not (self.enabled and self.writeback_enabled and trace_id):
            return
        payload = {
            "variant": variant,
            "response": result.get("response", ""),
            "elapsed_seconds": result.get("elapsed_seconds", 0),
            "character_count": result.get("character_count", 0),
            "inference_mode": "remote-webapp",
        }

        def _run() -> None:
            try:
                self._request_json(f"/traces/{trace_id}/outputs", payload, self.timeout)
                print(f"[RAG] 已回写 {variant} 输出 trace={trace_id}", flush=True)
            except Exception as exc:  # noqa: BLE001 - 回写失败不影响主流程
                print(f"[RAG] 回写失败（不影响回答）：{self._describe(exc)}", flush=True)

        threading.Thread(target=_run, name="rag-writeback", daemon=True).start()

    # ----------------------------------------------------------------- utils
    @staticmethod
    def _describe(exc: BaseException) -> str:
        if isinstance(exc, urllib.error.HTTPError):
            return f"HTTP {exc.code} {exc.reason}"
        if isinstance(exc, urllib.error.URLError):
            return f"无法连接（{exc.reason}）"
        return f"{type(exc).__name__}: {exc}"

    def _static_payload(
        self,
        status: str,
        hint: str,
        error: str = "",
        latency_ms=None,
        question: str = "",
        stance: str | None = None,
    ) -> dict:
        """没有真正走检索时的载荷。

        ``prompt`` 仍然给出模型实际收到的提问（即原始问题），
        方便前端如实展示"这一轮到底喂了什么"，而不是留空让人误判。
        """
        return {
            "rag_status": status,
            "trace_id": None,
            "rag_latency_ms": latency_ms,
            "top_k": self.top_k,
            "stance": stance,
            "results": [],
            "prompt": question or None,
            "augmented_prompt": None,
            "original_question": question or None,
            "injected": False,
            "service_url": self.base_url,
            "rag_cache": None,
            "hint": hint,
            "error": error or None,
        }

    @classmethod
    def _pick(cls, item: dict, keys: tuple[str, ...]):
        for key in keys:
            if key in item and item[key] not in (None, ""):
                return item[key]
        return None

    @classmethod
    def _normalize_results(cls, data: dict) -> list[dict]:
        container = None
        for key in cls.RESULT_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                container = value
                break
        if container is None:
            return []

        normalized: list[dict] = []
        for position, item in enumerate(container, start=1):
            if isinstance(item, str):
                normalized.append(
                    {"index": position, "title": item, "score": None, "source": None,
                     "doc_id": None, "snippet": None}
                )
                continue
            if not isinstance(item, dict):
                continue
            score = cls._pick(item, cls.SCORE_KEYS)
            try:
                score = round(float(score), 4)
            except (TypeError, ValueError):
                pass
            snippet = cls._pick(item, cls.SNIPPET_KEYS)
            if isinstance(snippet, str) and len(snippet) > 500:
                snippet = snippet[:500] + "…"
            source = cls._pick(item, cls.SOURCE_KEYS)
            if isinstance(source, str) and len(source) > 300:
                source = "…" + source[-299:]
            rank = cls._pick(item, cls.RANK_KEYS)
            try:
                index = int(rank)
            except (TypeError, ValueError):
                index = position
            normalized.append(
                {
                    "index": index,
                    "title": str(cls._pick(item, cls.TITLE_KEYS) or f"资料 {index}"),
                    "score": score,
                    "source": None if source is None else str(source),
                    "doc_id": None if cls._pick(item, cls.DOC_ID_KEYS) is None
                    else str(cls._pick(item, cls.DOC_ID_KEYS)),
                    "snippet": snippet,
                    "stance": cls._metadata_value(item, "stance"),
                    "risk_level": cls._metadata_value(item, "risk_level"),
                    "adversarial_strength": cls._metadata_value(item, "adversarial_strength"),
                    "intent_tag": cls._metadata_value(item, "intent_tag"),
                }
            )
        return normalized

    @staticmethod
    def _metadata_value(item: dict, key: str):
        value = item.get(key)
        if value in (None, "") and isinstance(item.get("metadata"), dict):
            value = item["metadata"].get(key)
        return value


RAG = RagClient(
    base_url=RAG_URL,
    enabled=RAG_ENABLED,
    top_k=RAG_TOP_K,
    timeout=RAG_TIMEOUT,
    cooldown=RAG_COOLDOWN,
    cache_ttl=RAG_CACHE_TTL,
    writeback=RAG_WRITEBACK,
)


def validate_files() -> None:
    required = [
        MODEL_DIR / "model.safetensors",
        MODEL_DIR / "config.json",
        MODEL_DIR / "tokenizer.json",
        ADAPTER_DIR / "adapters.safetensors",
        ADAPTER_DIR / "adapter_config.json",
        LEGACY_ADAPTER_DIR / "adapters.safetensors",
        LEGACY_ADAPTER_DIR / "adapter_config.json",
        SYSTEM_PROMPT_FILE,
        # 缺了它，「训练原句」预设会静默退化成"不注入 system"，
        # 安全归因实验就会被悄悄污染，所以按必需文件处理。
        TRAINING_PROMPT_FILE,
        MINIMAL_PROMPT_FILE,
        ROLE_ONLY_PROMPT_FILE,
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
            rag_health = RAG.health()
            self._send_json(
                {
                    "status": "ok",
                    "device": str(mx.default_device()),
                    "loaded_variants": RUNTIME.loaded_variants,
                    "model_available": MODEL_DIR.exists(),
                    "adapter_available": ADAPTER_DIR.exists(),
                    "rag": {
                        "enabled": rag_health.get("enabled", False),
                        "reachable": rag_health.get("reachable", False),
                        "url": rag_health.get("url"),
                        "document_count": rag_health.get("document_count"),
                        "index_backend": rag_health.get("index_backend"),
                    },
                }
            )
            return
        if path == "/api/rag/health":
            force = "force=1" in self.path
            self._send_json(RAG.health(force=force))
            return
        if path == "/api/prompts":
            self._send_json(
                {
                    "default": DEFAULT_SYSTEM_PRESET,
                    "presets": RUNTIME.describe_presets(),
                }
            )
            return
        if path == "/api/info":
            self._send_json(
                {
                    "base_model": "Qwen2.5-1.5B-Instruct · MLX 4-bit",
                    "adapter": "中医 LoRA · rank 8 · 安全能力内化版",
                    "training_records": 1095,
                    "test_loss": 1.998973,
                    "test_perplexity": 7.381469,
                    "core_rouge_before": 0.208,
                    "core_rouge_after": 0.337,
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
            if content_length <= 0 or content_length > 32_768:
                raise ValueError("请求内容大小不合法")
            payload = json.loads(self.rfile.read(content_length))
            question = str(payload.get("question", "")).strip()
            variant = str(payload.get("variant", ""))
            max_tokens = int(payload.get("max_tokens", 256))
            use_rag = bool(payload.get("use_rag", True))
            top_k = int(payload.get("top_k", RAG.top_k))
            system_preset = str(payload.get("system", DEFAULT_SYSTEM_PRESET))
            stance = str(payload.get("stance", "aligned")).strip().lower()

            if not question:
                raise ValueError("请输入问题")
            if len(question) > 4000:
                raise ValueError("问题不能超过 4000 个字符")
            if variant not in {"base", "lora", "legacy_lora"}:
                raise ValueError("未知模型类型")
            if max_tokens not in {128, 256, 384}:
                raise ValueError("输出长度参数不合法")
            if not 1 <= top_k <= 8:
                raise ValueError("检索条数参数不合法")
            if system_preset not in SYSTEM_PRESETS:
                raise ValueError("未知的系统提示词预设")
            if use_rag and stance not in VALID_STANCES:
                raise ValueError("stance 必须是 aligned / neutral / opposed")

            # 只增强送入模型的问题，system prompt 与 LoRA 链路保持不变。
            if use_rag:
                prompt, rag_payload = RAG.prepare(question, stance=stance, top_k=top_k)
            else:
                prompt, rag_payload = question, RAG.skip_payload(top_k, question=question)
                stance = "none"

            result = RUNTIME.generate(variant, prompt, max_tokens, system_preset)
            # 原返回结构不变，只多出 rag / prompt_sent / system_* / full_prompt，
            # 便于前端与答辩逐层追溯「模型到底读了什么」。
            self._send_json(
                {**result, "stance": stance, "rag": rag_payload, "prompt_sent": prompt}
            )
            RAG.writeback(rag_payload.get("trace_id"), variant, result)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception:
            traceback.print_exc()
            self._send_json({"error": "本地模型推理失败，请查看服务端日志"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local TCM LoRA comparison app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--eager", action="store_true", help="Load base and both LoRA variants before accepting requests")
    parser.add_argument("--rag-url", default=None, help="覆盖 RAG 微服务地址（默认取 TCM_RAG_URL）")
    parser.add_argument("--no-rag", action="store_true", help="关闭检索增强，只跑本地模型")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_files()

    if args.rag_url:
        RAG.base_url = args.rag_url.rstrip("/")
    if args.no_rag:
        RAG.enabled = False
    if RAG.enabled:
        probe = RAG.health(force=True)
        state = "可达" if probe.get("reachable") else "不可达，将自动降级直答"
        print(f"RAG service: {RAG.base_url} -> {state}", flush=True)
        if not probe.get("reachable"):
            print(f"  ({probe.get('detail')})", flush=True)
    else:
        print("RAG service: 已关闭（仅本地推理）", flush=True)

    if args.eager:
        RUNTIME._load_variant("base")
        RUNTIME._load_variant("lora")
        RUNTIME._load_variant("legacy_lora")

    presets = RUNTIME.describe_presets()
    summary = "、".join(
        f"{item['key']}({item['character_count']}字)" if item["text"] else f"{item['key']}(无)"
        for item in presets
    )
    print(f"System prompt presets: {summary} | 默认 {DEFAULT_SYSTEM_PRESET}", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"TCM comparison app: http://{args.host}:{args.port}", flush=True)
    print("Models load lazily on first use." if not args.eager else "All model variants are ready.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
