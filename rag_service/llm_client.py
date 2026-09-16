"""LLM 客户端：调用同伴 MacBook 上的 LoRA 推理服务，不可达时降级本地模拟。

链路位置（链式线性，互不干扰）：
    用户 query ──> RAG 检索/增强 ──> LoRA(同伴电脑) ──> 返回
                                    └─ 连不上 ──> degraded-mock（本地模拟）

真实调用支持两种同伴侧服务形态：
- webapp：tcm-lora-mlx/webapp/server.py 的 POST /api/generate（并排对比界面后端）
- openai：mlx_lm.server 的 OpenAI 兼容 POST /v1/chat/completions

降级原则：只降级、不报错、不中断链路；返回体带 degraded=True 与 inference_mode
="degraded-mock"，让前端/答辩演示能明确区分真实 LoRA 输出与模拟输出。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from . import config


class LlmClient:
    def __init__(
        self,
        mode: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.mode = (mode or config.LLM_MODE).lower()
        self.base_url = (base_url or config.LLM_URL).rstrip("/")
        self.timeout = timeout if timeout is not None else config.LLM_TIMEOUT
        self.last_error = ""

    @property
    def configured(self) -> bool:
        return bool(self.base_url) or self.mode == "mock"

    # ---------- 对外主入口 ----------

    def generate(
        self,
        augmented_prompt: str,
        variant: str | None = None,
        max_tokens: int | None = None,
        retrieval_results: list[dict] | None = None,
    ) -> dict:
        """返回结构与 webapp /api/generate 对齐，另加 degraded / inference_mode 字段。"""
        variant = variant or config.LLM_VARIANT
        max_tokens = max_tokens or config.LLM_MAX_TOKENS

        if self.mode == "mock":
            return self._mock(augmented_prompt, variant, max_tokens, retrieval_results, reason="mock 模式（强制离线）")

        if not self.base_url:
            return self._mock(augmented_prompt, variant, max_tokens, retrieval_results, reason="未配置 TCM_LLM_URL")

        if self.mode in ("auto", "webapp"):
            outcome = self._call_webapp(augmented_prompt, variant, max_tokens)
            if outcome is not None:
                return outcome
            if self.mode == "webapp":
                return self._mock(augmented_prompt, variant, max_tokens, retrieval_results, reason=self.last_error)

        if self.mode in ("auto", "openai"):
            outcome = self._call_openai(augmented_prompt, variant, max_tokens)
            if outcome is not None:
                return outcome
            return self._mock(augmented_prompt, variant, max_tokens, retrieval_results, reason=self.last_error)

        return self._mock(augmented_prompt, variant, max_tokens, retrieval_results, reason=f"未知 mode={self.mode}")

    # ---------- 真实调用：webapp /api/generate ----------

    def _call_webapp(self, prompt: str, variant: str, max_tokens: int) -> dict | None:
        url = f"{self.base_url}/api/generate"
        body = json.dumps(
            {"question": prompt, "variant": variant, "max_tokens": max_tokens},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            started = time.monotonic()
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if "response" not in payload:
                raise ValueError("响应缺少 response 字段")
            return {
                "variant": variant,
                "label": payload.get("label", "中医 LoRA"),
                "response": payload["response"],
                "elapsed_seconds": payload.get("elapsed_seconds", round(time.monotonic() - started, 2)),
                "character_count": payload.get("character_count", len(payload["response"])),
                "max_tokens": max_tokens,
                "inference_mode": "remote-webapp",
                "degraded": False,
                "degraded_reason": "",
            }
        except Exception as exc:  # noqa: BLE001 —— 网络不通是预期内的常态，必须降级
            self.last_error = f"webapp 调用失败（{self.base_url}）：{exc}"
            return None

    # ---------- 真实调用：OpenAI 兼容 /v1/chat/completions ----------

    def _call_openai(self, prompt: str, variant: str, max_tokens: int) -> dict | None:
        url = f"{self.base_url}/v1/chat/completions"
        body = json.dumps(
            {
                "model": "lora" if variant == "lora" else "base",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            started = time.monotonic()
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            response_text = payload["choices"][0]["message"]["content"].strip()
            return {
                "variant": variant,
                "label": "中医 LoRA" if variant == "lora" else "Qwen2.5 基座",
                "response": response_text,
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "character_count": len(response_text),
                "max_tokens": max_tokens,
                "inference_mode": "remote-openai",
                "degraded": False,
                "degraded_reason": "",
            }
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"OpenAI 兼容调用失败（{self.base_url}）：{exc}"
            return None

    # ---------- 降级：本地模拟 ----------

    def _mock(
        self,
        prompt: str,
        variant: str,
        max_tokens: int,
        retrieval_results: list[dict] | None,
        reason: str,
    ) -> dict:
        """模拟 LoRA 返回：按真实 webapp 的返回结构拼一份基于检索资料的回答。

        模拟依据（读 webapp/server.py 得到的真实行为）：
        - LoRA 微调目标 = 简洁直接的中医专家风格（系统提示语约束）；
        - 推理温度 0、max_tokens 上限 → 回复偏短；
        - 因此这里取检索 Top1/Top2 的内容做要点压缩，模拟"模型参考知识作答"。
        """
        results = retrieval_results or []
        if results:
            bullet_points = []
            for item in results[:2]:
                content = item["content"].strip().replace("\n", " ")
                if len(content) > 160:
                    content = content[:160] + "……"
                bullet_points.append(f"{item['title']}：{content}")
            body = "\n".join(bullet_points)
        else:
            body = "（未检索到相关资料，无法给出参考答案。）"

        response = (
            f"{body}\n\n"
            "【降级说明】当前 LoRA 推理服务（同伴电脑）不可达，以上为 RAG 微服务的"
            "本地降级模拟输出（基于检索资料合成，非模型生成）。"
            f"降级原因：{reason}"
        )
        return {
            "variant": variant,
            "label": "中医 LoRA（降级模拟）",
            "response": response,
            "elapsed_seconds": 0.0,
            "character_count": len(response),
            "max_tokens": max_tokens,
            "inference_mode": "degraded-mock",
            "degraded": True,
            "degraded_reason": reason,
        }
