#!/usr/bin/env python3
"""RAG 微服务冒烟测试：不起 HTTP 服务，直接驱动 RagRuntime 走完整链路。

用法（conda 环境 lora）：
    python -m rag_service.smoke_test
    python rag_service/smoke_test.py

覆盖点：
1. MySQL 连接与三立场数据集导入数量（同向/模糊/反向 各 500）；
2. 嵌入层是否为真实语义模型；FAISS 后端；
3. 三立场检索过滤是否精确（各立场只返回本立场文档）；
4. 检索质量：专有名词查询能命中对应文档（混合检索生效）；
5. /prepare 增强提示词格式，且不同立场的提示词内容不同；
6. /generate 完整链路 + LLM 降级兜底如实标注；
7. trace 落库并记录本次立场（消融实验归因依赖此字段）。

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
    counts = runtime.store.stance_counts()
    check("三立场数据各 500 条",
          all(counts.get(stance, 0) == 500 for stance in ("aligned", "ambiguous", "opposed")),
          str(counts))
    check("真实语义嵌入", "sentence-transformers" in runtime.embedder_backend,
          runtime.embedder_backend + (f" | {runtime.embedder_warning}" if runtime.embedder_warning else ""))
    check("FAISS 后端", runtime.store.backend == "faiss", runtime.store.backend)
    print(flush=True)

    print("== 3. 三立场检索过滤（每立场各查一次）==", flush=True)
    query = "我最近总是头晕目眩，中医有什么好的方剂推荐吗？"
    prompts: dict[str, str] = {}
    for stance in ("aligned", "ambiguous", "opposed", "all"):
        retrieval = runtime.retrieve(query, 3, stance)
        stances = {item["stance"] for item in retrieval["results"]}
        if stance == "all":
            check("stance=all 可跨立场召回", retrieval["count"] > 0,
                  f"{retrieval['count']} 条，立场 {sorted(stances)}")
        else:
            check(f"stance={stance} 只返回本立场", stances == {stance},
                  f"{retrieval['count']} 条，立场 {sorted(stances)}，{retrieval['latency_ms']}ms")
    print(flush=True)

    print("== 4. 检索质量（专有名词必须命中）==", flush=True)
    retrieval = runtime.retrieve("灵芝的药理作用是什么？", 3, "opposed")
    titles = [item["title"] for item in retrieval["results"]]
    check("反向立场命中灵芝相关文档", any("灵芝" in title for title in titles), " | ".join(t[:20] for t in titles))
    print(flush=True)

    print("== 5. 增强提示词随立场变化 ==", flush=True)
    for stance in ("aligned", "ambiguous", "opposed"):
        preparation = runtime.prepare(query, 3, True, stance)
        prompts[stance] = preparation["augmented_prompt"]
        ok = (
            preparation["rag_status"] == "retrieved"
            and "[资料 1]" in preparation["augmented_prompt"]
            and "[用户问题]" in preparation["augmented_prompt"]
            and preparation["stance"] == stance
        )
        check(f"stance={stance} 提示词格式正确", ok, preparation["rag_status"])
    check("三立场提示词内容互不相同", len(set(prompts.values())) == 3)
    print(flush=True)

    print("== 6. 完整链路 + LLM 降级兜底 ==", flush=True)
    generation = runtime.generate(query, 3, True, "lora", 256, "aligned")
    llm = generation["llm"]
    check("LLM 有返回", bool(llm.get("response")))
    check("降级如实标注（未配置 TCM_LLM_URL 时）",
          (llm.get("degraded") is True) == (not runtime.llm.configured or runtime.llm.mode == "mock"),
          f"inference_mode={llm.get('inference_mode')}, degraded={llm.get('degraded')}")
    print(flush=True)

    print("== 7. trace 落库（含立场字段）==", flush=True)
    trace = runtime.repository.get_trace(generation["trace_id"])
    check("trace 可查", trace is not None, generation["trace_id"][:8] + "…")
    if trace:
        check("trace 记录立场 aligned", trace.get("rag_stance") == "aligned", str(trace.get("rag_stance")))
        check("trace 含增强提示词", bool(trace.get("augmented_prompt")))
        check("trace 含检索明细立场", all(
            item.get("stance") for item in trace.get("results", [])
        ), f"{len(trace.get('results', []))} 条命中")
        check("trace 含模型输出", "lora" in trace.get("outputs", {}))
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
