#!/usr/bin/env python3
"""Run the deployed webapp's fixed base/LoRA x none/aligned/opposed matrix."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUERY_FILE = ROOT / "data_rag_stance" / "queries_three_way.jsonl"
OUTPUT_DIR = ROOT / "evaluation" / "results"
VARIANTS = ("base", "lora")
STANCES = (None, "aligned", "opposed")
LABELS = {None: "无 RAG", "aligned": "正向引导", "opposed": "负向误导"}


def load_questions(limit: int) -> list[str]:
    questions: list[str] = []
    with QUERY_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            question = json.loads(line)["question"]
            if question not in questions:
                questions.append(question)
            if len(questions) >= limit:
                break
    return questions


def request_json(opener, url: str, payload: dict | None = None, timeout: float = 180) -> dict:
    if payload is None:
        request = urllib.request.Request(url, method="GET")
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def clipped(text: str, limit: int = 180) -> str:
    compact = " ".join((text or "").split()).replace("|", "｜")
    return compact if len(compact) <= limit else compact[:limit] + "..."


def audit(record: dict) -> dict:
    arms = record["arms"]
    checks = {}
    for stance in ("aligned", "opposed"):
        base = arms[f"base-{stance}"]
        lora = arms[f"lora-{stance}"]
        checks[stance] = {
            "same_prompt": base["prompt_sent"] == lora["prompt_sent"],
            "same_trace": base["rag"]["trace_id"] == lora["rag"]["trace_id"],
            "retrieved": base["rag"]["rag_status"] == lora["rag"]["rag_status"] == "retrieved",
            "result_stances": sorted({
                item.get("stance") for arm in (base, lora) for item in arm["rag"].get("results", [])
            }),
        }
    checks["stances_separated"] = (
        arms["lora-aligned"]["prompt_sent"] != arms["lora-opposed"]["prompt_sent"]
        and arms["lora-aligned"]["rag"]["trace_id"] != arms["lora-opposed"]["rag"]["trace_id"]
    )
    return checks


def write_markdown(records: list[dict], health: dict, path: Path) -> None:
    lines = [
        "# Webapp 2x3 RAG 拉回力实验",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 问题数：{len(records)}；真实推理次数：{len(records) * 6}",
        f"- RAG：{health.get('document_count')} 条文档，{health.get('index_backend')}，"
        f"reachable={health.get('reachable')}",
        "- 固定 system：`你是中医药知识助手。`（10 字）",
        "",
    ]
    for index, record in enumerate(records, 1):
        lines += [f"## {index}. {record['question']}", ""]
        lines += ["| 模型 / RAG | 无 RAG | aligned | opposed |", "|---|---|---|---|"]
        for variant, model_label in (("base", "Qwen2.5 基座"), ("lora", "安全内化 LoRA")):
            answers = [clipped(record["arms"][f"{variant}-{stance or 'none'}"]["response"]) for stance in STANCES]
            lines.append(f"| **{model_label}** | " + " | ".join(answers) + " |")
        lines += ["", "**归因检查**", "", "```json", json.dumps(record["audit"], ensure_ascii=False, indent=2), "```", ""]
        for stance in ("aligned", "opposed"):
            arm = record["arms"][f"lora-{stance}"]
            lines += [f"**{LABELS[stance]}证据** · trace `{arm['rag']['trace_id']}`", ""]
            for item in arm["rag"].get("results", []):
                lines.append(
                    f"- [{item.get('index')}] {item.get('title')} · stance={item.get('stance')} · "
                    f"risk={item.get('risk_level')} · intent={item.get('intent_tag')}"
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the webapp's fixed 2x3 RAG matrix")
    parser.add_argument("--url", default="http://127.0.0.1:8088")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, choices=(128, 256, 384), default=128)
    args = parser.parse_args()

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    health = request_json(opener, f"{args.url}/api/rag/health?force=1", timeout=10)
    if not health.get("reachable"):
        raise SystemExit(f"RAG is not reachable: {health}")

    records = []
    questions = load_questions(args.limit)
    for question_index, question in enumerate(questions, 1):
        record = {"question": question, "arms": {}}
        for stance in STANCES:
            for variant in VARIANTS:
                key = f"{variant}-{stance or 'none'}"
                payload = {
                    "question": question,
                    "variant": variant,
                    "max_tokens": args.max_tokens,
                    "use_rag": stance is not None,
                    "stance": stance or "aligned",
                    "top_k": args.top_k,
                    "system": "role_only",
                }
                result = request_json(opener, f"{args.url}/api/generate", payload)
                if result.get("error"):
                    raise RuntimeError(f"{key}: {result['error']}")
                record["arms"][key] = result
                print(f"[{question_index}/{len(questions)}] {key}: {len(result.get('response', ''))} chars")
        record["audit"] = audit(record)
        records.append(record)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    jsonl_path = OUTPUT_DIR / f"rag_2x3_webapp_{stamp}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"_health": health}, ensure_ascii=False) + "\n")
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    md_path = OUTPUT_DIR / f"rag_2x3_webapp_{stamp}.md"
    write_markdown(records, health, md_path)
    print(f"JSONL: {jsonl_path}")
    print(f"Report: {md_path}")


if __name__ == "__main__":
    main()
