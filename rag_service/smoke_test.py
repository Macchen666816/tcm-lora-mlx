#!/usr/bin/env python3
"""RAG 微服务冒烟测试：不起 HTTP 服务，直接驱动 RagRuntime 走完整链路。

用法（conda 环境 lora）：
    python -m rag_service.smoke_test
    python rag_service/smoke_test.py

覆盖点：
1. MySQL 连接与种子文档导入数量；
2. 嵌入层是否为真实语义模型（sentence-transformers）；
3. FAISS 检索：中医问题应命中语义相关文档；
4. /prepare 增强提示词格式；
5. /generate 完整链路 + LLM 降级兜底（未配置 TCM_LLM_URL 时应 degraded=true）；
6. trace 落库可查。

退出码：全部通过为 0，任何失败为 1。
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

if __package__:
    from . import config
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from rag_service import config


def main() -> int:
    config.load_env_file()

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'✅' if ok else '❌'} {name}" + (f" —— {detail}" if detail else ""), flush=True)
        if not ok:
            failures.append(name)

    print("== 1. 初始化 RagRuntime（MySQL + 嵌入模型 + FAISS）==", flush=True)
    if __package__:
        from .server import RagRuntime
    else:
        from rag_service.server import RagRuntime

    runtime = RagRuntime()
    print(flush=True)

    print("== 2. 组件状态 ==", flush=True)
    check("MySQL 已连接", runtime.repository.db_status == "connected",
          f"db_status={runtime.repository.db_status} err={runtime.repository.db_error}")
    check("知识文档非空", len(runtime.store.documents) > 0,
          f"{len(runtime.store.documents)} 条")
    check("真实语义嵌入", "sentence-transformers" in runtime.embedder_backend,
          runtime.embedder_backend + (f" | {runtime.embedder_warning}" if runtime.embedder_warning else ""))
    check("FAISS 后端", runtime.store.backend == "faiss", runtime.store.backend)
    print(flush=True)

    print("== 3. 检索质量（中医语义问题）==", flush=True)
    retrieval = runtime.retrieve("薄荷的性味归经和主要功效是什么？", 3)
    check("检索有命中", retrieval["count"] > 0, f"count={retrieval['count']}, latency={retrieval['latency_ms']}ms")
    top_hit = retrieval["results"][0] if retrieval["results"] else {}
    top_text = top_hit.get("title", "") + top_hit.get("content", "")
    relevant = any(k in top_text for k in ("薄荷", "辛", "凉", "疏散风热", "清利头目"))
    check("Top1 语义相关（谈薄荷）", relevant, top_hit.get("title", "")[:40])
    print(flush=True)

    print("== 4. 增强提示词 ==", flush=True)
    preparation = runtime.prepare("薄荷的性味归经和主要功效是什么？", 3, True)
    check("rag_status=retrieved", preparation["rag_status"] == "retrieved", preparation["rag_status"])
    check("提示词含指令与资料块",
          "[资料 1]" in preparation["augmented_prompt"] and "[用户问题]" in preparation["augmented_prompt"])
    check("提示词含原问题", "薄荷" in preparation["augmented_prompt"])
    print(flush=True)

    print("== 5. 完整链路 + LLM 降级兜底 ==", flush=True)
    generation = runtime.generate(
        "薄荷的性味归经和主要功效是什么？", 3, True, "lora", 256
    )
    llm = generation["llm"]
    check("LLM 有返回", bool(llm.get("response")))
    check("降级如实标注（未配置 TCM_LLM_URL 时）",
          (llm.get("degraded") is True) == (not runtime.llm.configured or runtime.llm.mode == "mock"),
          f"inference_mode={llm.get('inference_mode')}, degraded={llm.get('degraded')}, "
          f"url={runtime.llm.base_url or '(未配置)'}")
    print(flush=True)

    print("== 6. trace 落库 ==", flush=True)
    trace = runtime.repository.get_trace(generation["trace_id"])
    check("trace 可查", trace is not None, generation["trace_id"][:8] + "…")
    if trace:
        check("trace 含增强提示词", "augmented_prompt" in trace and bool(trace["augmented_prompt"]))
        check("trace 含模型输出", "outputs" in trace and "lora" in trace.get("outputs", {}))
    print(flush=True)

    if failures:
        print(f"冒烟测试失败：{len(failures)} 项 —— {failures}", flush=True)
        return 1
    print("冒烟测试全部通过 ✅", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
