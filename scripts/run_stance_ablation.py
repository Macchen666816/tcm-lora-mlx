#!/usr/bin/env python3
"""三模式消融实验跑批：同一批问题 × 三种立场，输出并排对比结果。

回答"实机测试时如何区分三种模式"：
    模式不是模型选的，而是**调用方在请求里指定**（`stance` 参数）。
    本脚本对每个问题发出 **3 次独立请求**（aligned / neutral / opposed），
    把三份结果并排落盘 —— 这就是"平铺三种"的正确实现。

两种运行模式
------------
- `--mode prepare`（默认）：只跑检索侧，输出三次「命中文档 + 增强提示词」。
  同伴的 LoRA 不可达时也能跑，用于核对实验配置是否正确。
- `--mode generate`：跑完整链路（RAG → LoRA），输出三份模型回答。
  需要同伴 MacBook 上的 LoRA 服务在线（`TCM_LLM_URL`），否则返回的是降级模拟。

产物
----
- `evaluation/stance_ablation_<mode>_<时间戳>.jsonl`  逐条原始结果（含 trace_id）
- `evaluation/stance_ablation_<mode>_<时间戳>.md`     并排对比表格（可直接贴报告）

用法
----
    python scripts/run_stance_ablation.py --limit 5                 # 试跑 5 条
    python scripts/run_stance_ablation.py --all --mode generate     # 全量 134 题，跑完整链路
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
STANCES = ("aligned", "neutral", "opposed")
DEFAULT_RAG_URL = "http://127.0.0.1:8090"


def load_questions(limit: int | None) -> list[dict]:
    """从三立场同题子集里取问题（同一问题在每个模式下各跑一次）。"""
    with QUERY_FILE.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    seen: dict[str, dict] = {}
    for row in rows:
        seen.setdefault(row["question"], row)
    questions = list(seen.values())
    return questions[:limit] if limit else questions


def call(rag_url: str, path: str, payload: dict, timeout: float = 120) -> dict:
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


def run(rag_url: str, questions: list[dict], mode: str, top_k: int) -> list[dict]:
    results: list[dict] = []
    total = len(questions) * len(STANCES)
    done = 0
    for item in questions:
        question = item["question"]
        record: dict = {"question": question, "paired_id": item.get("paired_id", ""), "arms": {}}
        for stance in STANCES:
            payload = {"query": question, "top_k": top_k, "stance": stance}
            if mode == "generate":
                payload.update({"rag_enabled": True, "variant": "lora", "max_tokens": 256})
                response = call(rag_url, "/generate", payload)
                arm = {
                    "stance": stance,
                    "trace_id": response.get("trace_id", ""),
                    "docs": [
                        {
                            "rank": d["rank"],
                            "title": d["title"],
                            "risk_level": d.get("risk_level", ""),
                            "adversarial_strength": d.get("adversarial_strength", ""),
                            "intent_tag": d.get("intent_tag", ""),
                        }
                        for d in response.get("rag_results", [])
                    ],
                    "answer": (response.get("llm") or {}).get("response", ""),
                    "inference_mode": (response.get("llm") or {}).get("inference_mode", ""),
                    "character_count": (response.get("llm") or {}).get("character_count", 0),
                    "error": response.get("error", ""),
                }
            else:
                response = call(rag_url, "/prepare", payload)
                arm = {
                    "stance": stance,
                    "trace_id": response.get("trace_id", ""),
                    "docs": [
                        {
                            "rank": d["rank"],
                            "title": d["title"],
                            "risk_level": d.get("risk_level", ""),
                            "adversarial_strength": d.get("adversarial_strength", ""),
                            "intent_tag": d.get("intent_tag", ""),
                        }
                        for d in response.get("results", [])
                    ],
                    "augmented_prompt": response.get("augmented_prompt", ""),
                    "error": response.get("error", ""),
                }
            record["arms"][stance] = arm
            done += 1
            print(f"\r  进度 {done}/{total}", end="", flush=True)
        results.append(record)
    print()
    return results


def write_markdown(results: list[dict], mode: str, path: Path) -> None:
    lines = [
        f"# 三模式消融对比（{mode}）",
        "",
        f"- 问题数：{len(results)}，每个问题 3 次调用（aligned / neutral / opposed）",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for index, record in enumerate(results, start=1):
        lines += [f"## {index}. {record['question']}", ""]
        lines += ["| 模式 | 命中文档（Top1） | 风险 | 强度/注入 | " +
                  ("回答" if mode == "generate" else "增强提示词（截断）") + " |",
                  "|---|---|---|---|---|"]
        for stance in STANCES:
            arm = record["arms"].get(stance, {})
            docs = arm.get("docs") or []
            top = docs[0] if docs else {}
            payload_text = arm.get("answer") if mode == "generate" else arm.get("augmented_prompt", "")
            payload_text = (payload_text or "").replace("\n", " ").replace("|", "｜")
            if len(payload_text) > 220:
                payload_text = payload_text[:220] + "……"
            strength = "/".join(filter(None, [top.get("adversarial_strength", ""), top.get("intent_tag", "")]))
            lines.append(
                f"| **{stance}** | {str(top.get('title', '—'))[:28]} | {top.get('risk_level', '—')} | "
                f"{strength or '—'} | {payload_text} |"
            )
        if mode == "generate":
            modes = {record["arms"].get(s, {}).get("inference_mode", "") for s in STANCES}
            if modes == {"degraded-mock"}:
                lines += ["", "> ⚠️ 本轮三臂均为 `degraded-mock`（LoRA 不可达），仅验证链路，不可用于结论。"]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="三模式消融实验跑批")
    parser.add_argument("--rag-url", default=DEFAULT_RAG_URL)
    parser.add_argument("--mode", choices=("prepare", "generate"), default="prepare")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=5, help="问题数上限（默认 5，便于试跑）")
    parser.add_argument("--all", action="store_true", help="跑全部 134 题")
    args = parser.parse_args()

    questions = load_questions(None if args.all else args.limit)
    print(f"模式：{args.mode}｜问题数：{len(questions)}｜每问题 3 次调用｜共 {len(questions)*3} 次请求")
    results = run(args.rag_url, questions, args.mode, args.top_k)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    jsonl_path = OUT_DIR / f"stance_ablation_{args.mode}_{stamp}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in results:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    md_path = OUT_DIR / f"stance_ablation_{args.mode}_{stamp}.md"
    write_markdown(results, args.mode, md_path)

    print(f"原始结果：{jsonl_path}")
    print(f"对比表格：{md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
