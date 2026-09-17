#!/usr/bin/env python3
"""2×3 拉回力拔河实验跑批：模型变体 × RAG 模式，一次跑全。

实验矩阵（8 格）
----------------
                    无 RAG        积极引导 RAG    模糊·无拦截 RAG   恶意误导 RAG
    Qwen2.5 基座     base-none     base-aligned    base-neutral     base-opposed
    中医 LoRA        lora-none     lora-aligned    lora-neutral     lora-opposed

- 「无 RAG」= `rag_enabled: false`（不检索，直接问模型）——用于看微调本身带来的立场
- 三种 RAG 模式 = `stance: aligned / neutral / opposed`
- 提示词层（RAG 前置指令、webapp system prompt）必须**中性且固定**，见
  `evaluation/SYSTEM_PROMPT_VARIANTS.md`；否则拔河结果无法归因

输出
----
- `evaluation/ablation8_<mode>_<时间戳>.jsonl`  每问题 8 格原始结果（含 trace_id）
- `evaluation/ablation8_<mode>_<时间戳>.md`     每问题一张 2×4 矩阵表 + RAG 证据（top-k 与完整 prompt）

用法
----
    python scripts/run_stance_ablation.py --limit 3                 # 试跑 3 题（prepare：只测检索侧 6 格）
    python scripts/run_stance_ablation.py --all --mode generate     # 全量题目，跑完整链路（需 LoRA 在线）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
QUERY_FILE = PROJECT_DIR / "data_rag_stance" / "queries_three_way.jsonl"
OUT_DIR = PROJECT_DIR / "evaluation"
RAG_STANCES = ("aligned", "opposed")  # 中性档已剔除，拉回力实验只留正向/负向
VARIANTS = ("base", "lora")
STANCE_LABEL = {"aligned": "正向引导", "opposed": "负向误导"}
VARIANT_LABEL = {"base": "Qwen2.5 基座", "lora": "中医 LoRA"}


def load_questions(limit: int | None) -> list[dict]:
    with QUERY_FILE.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    seen: dict[str, dict] = {}
    for row in rows:
        seen.setdefault(row["question"], row)
    questions = list(seen.values())
    return questions[:limit] if limit else questions


def call(rag_url: str, path: str, payload: dict, timeout: float = 180) -> dict:
    request = urllib.request.Request(
        f"{rag_url}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:200]}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def fetch_service_state(rag_url: str) -> dict:
    """取服务端配置（前置指令开关等），写进结果用于实验归因。"""
    try:
        with urllib.request.urlopen(f"{rag_url}/health", timeout=10) as response:
            health = json.loads(response.read().decode("utf-8"))
        return {
            "rag_instruction_enabled": health.get("rag_instruction_enabled"),
            "default_stance": health.get("default_stance"),
            "document_count": health.get("document_count"),
            "stance_counts": health.get("stance_counts", {}),
            "retriever": health.get("retriever", ""),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def run(rag_url: str, questions: list[dict], mode: str, top_k: int) -> list[dict]:
    """generate=6 格（2 无 RAG + 4 有 RAG）；prepare=4 格（无 RAG 时检索侧无内容可看）。"""
    cells = [(variant, stance) for variant in VARIANTS for stance in RAG_STANCES]
    if mode == "generate":
        cells = [(variant, None) for variant in VARIANTS] + cells
    total = len(questions) * len(cells)
    results: list[dict] = []
    done = 0

    for item in questions:
        question = item["question"]
        record: dict = {"question": question, "paired_id": item.get("paired_id", ""), "arms": {}}
        for variant, stance in cells:
            key = f"{variant}-{stance or 'none'}"
            payload: dict = {"query": question, "top_k": top_k, "variant": variant}
            if stance is None:
                payload["rag_enabled"] = False
            else:
                payload["rag_enabled"] = True
                payload["stance"] = stance
            if mode == "generate":
                payload["max_tokens"] = 256
                response = call(rag_url, "/generate", payload)
                llm = response.get("llm") or {}
                record["arms"][key] = {
                    "variant": variant,
                    "stance": stance,
                    "trace_id": response.get("trace_id", ""),
                    "answer": llm.get("response", ""),
                    "inference_mode": llm.get("inference_mode", ""),
                    "degraded": llm.get("degraded"),
                    "character_count": llm.get("character_count", 0),
                    "rag_status": response.get("rag_status", ""),
                    "docs": [
                        {"rank": d["rank"], "title": d["title"], "risk_level": d.get("risk_level", ""),
                         "adversarial_strength": d.get("adversarial_strength", ""),
                         "intent_tag": d.get("intent_tag", "")}
                        for d in response.get("rag_results", [])
                    ],
                    "augmented_prompt": response.get("augmented_prompt", ""),
                    "error": response.get("error", ""),
                }
            else:
                response = call(rag_url, "/prepare", payload)
                record["arms"][key] = {
                    "variant": variant,
                    "stance": stance,
                    "trace_id": response.get("trace_id", ""),
                    "rag_status": response.get("rag_status", ""),
                    "docs": [
                        {"rank": d["rank"], "title": d["title"], "risk_level": d.get("risk_level", ""),
                         "adversarial_strength": d.get("adversarial_strength", ""),
                         "intent_tag": d.get("intent_tag", "")}
                        for d in response.get("results", [])
                    ],
                    "augmented_prompt": response.get("augmented_prompt", ""),
                    "error": response.get("error", ""),
                }
            done += 1
            print(f"\r  进度 {done}/{total}", end="", flush=True)
        results.append(record)
    print()
    return results


def _clip(text: str, limit: int = 300) -> str:
    text = (text or "").replace("\n", " ").replace("|", "｜")
    return text[:limit] + "……" if len(text) > limit else text


def write_markdown(results: list[dict], mode: str, path: Path, service_state: dict) -> None:
    instruction = service_state.get("rag_instruction_enabled")
    lines = [
        f"# 拉回力实验 · 2×3（{mode}）",
        "",
        f"- 问题数：{len(results)}，每题 {6 if mode == 'generate' else 4} 次调用",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- RAG 前置指令：{'已开启' if instruction else '**已关闭（纯资料 + 问题）**'}"
        + f"｜默认立场 {service_state.get('default_stance', '—')}"
        + f"｜文档 {service_state.get('document_count', '—')} 条",
        "",
        "> 矩阵：行 = 模型变体（基座/LoRA），列 = RAG（正向/负向/无）。中性档已剔除。",
        "> 提示词层（RAG 前置指令 / webapp system prompt）必须中性且固定，否则结果无法归因。",
        "",
    ]

    for index, record in enumerate(results, start=1):
        arms = record["arms"]
        lines += [f"## {index}. {record['question']}", ""]
        lines += ["| 模型 \\ RAG | 无 RAG | 正向引导 | 负向误导 |", "|---|---|---|---|"]
        for variant in VARIANTS:
            cells = []
            for stance in (None, *RAG_STANCES):
                arm = arms.get(f"{variant}-{stance or 'none'}", {})
                if mode == "generate":
                    body = _clip(arm.get("answer", ""), 160)
                    tag = "⚠️降级" if arm.get("degraded") else ""
                    cells.append(f"{body} {tag}".strip() or (arm.get("error") or "—"))
                else:
                    docs = arm.get("docs") or []
                    top = docs[0] if docs else {}
                    marks = "/".join(filter(None, [top.get("risk_level", ""),
                                                   top.get("adversarial_strength", ""),
                                                   top.get("intent_tag", "")]))
                    cells.append(f"{str(top.get('title', '—'))[:24]} {('· ' + marks) if marks else ''}"
                                 if docs else (arm.get("error") or "—"))
            lines.append(f"| **{VARIANT_LABEL[variant]}** | " + " | ".join(cells) + " |")
        lines.append("")

        # RAG 证据：top-k 命中文档 + 完整增强提示词
        lines += ["<details><summary>RAG 证据（top-k 命中文档 + 完整 prompt）</summary>", ""]
        for stance in (None, *RAG_STANCES):
            arm = arms.get(f"lora-{stance or 'none'}", {})
            label = "无 RAG" if stance is None else STANCE_LABEL[stance]
            lines += [f"**{label}** —— " + ", ".join(
                f"[{d['rank']}] {d['title'][:26]}"
                + (f"（{d.get('adversarial_strength') or d.get('risk_level')}）" if d.get("adversarial_strength") or d.get("risk_level") else "")
                for d in (arm.get("docs") or [])
            ) or f"**{label}** —— 未检索（rag_enabled=false）", ""]
            if arm.get("augmented_prompt"):
                lines += ["```", arm["augmented_prompt"][:1200], "```", ""]
        lines += ["</details>", ""]

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="拉回力实验（2×3）跑批")
    parser.add_argument("--rag-url", default="http://127.0.0.1:8090")
    parser.add_argument("--mode", choices=("prepare", "generate"), default="prepare")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--all", action="store_true", help="跑全部题目（134 个问题）")
    args = parser.parse_args()

    questions = load_questions(None if args.all else args.limit)
    cells = 6 if args.mode == "generate" else 4
    service_state = fetch_service_state(args.rag_url)
    print(f"服务端状态：{service_state}")
    print(f"模式：{args.mode}｜问题数：{len(questions)}｜每题 {cells} 格｜共 {len(questions) * cells} 次请求")

    results = run(args.rag_url, questions, args.mode, args.top_k)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    jsonl_path = OUT_DIR / f"ablation23_{args.mode}_{stamp}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"_service_state": service_state}, ensure_ascii=False) + "\n")
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    md_path = OUT_DIR / f"ablation23_{args.mode}_{stamp}.md"
    write_markdown(results, args.mode, md_path, service_state)

    print(f"原始结果：{jsonl_path}")
    print(f"矩阵表格：{md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
