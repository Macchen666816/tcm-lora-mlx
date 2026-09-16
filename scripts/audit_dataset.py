#!/usr/bin/env python3
"""Audit and stage the raw TCM JSONL data without modifying source files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REQUIRED_FIELDS = ("instruction", "input", "output")
TASK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("title_generation", re.compile(r"标题|题目")),
    (
        "rewriting_editing",
        re.compile(
            r"修改|改写|润色|编辑|翻译|纠正|语病|改成.{0,12}(?:术语|表达)|"
            r"改为.{0,12}(?:术语|表达|正确)|更(?:加)?专业的中医术语"
        ),
    ),
    (
        "extraction_classification",
        re.compile(
            r"分类|归类|抽取|提取|识别哪些|请从以下|下列哪|判断下列|"
            r"按照.{0,12}分|以下.{0,20}哪些|哪些属于"
        ),
    ),
    (
        "long_form_generation",
        re.compile(
            r"撰写|编写|写一篇|写一段|摘要|概括|科普文章|文章开头|"
            r"文献综述|研究综述|(?:生成|创造).{0,16}(?:段|文章|问题)"
        ),
    ),
)

PLACEHOLDER_RE = re.compile(
    r"^[\s'\"“”‘’《》]*(?:略|略>|无|暂无|不详|未知|待补充|none|null|n/?a)[。.!！]?[\s'\"“”‘’《》]*$",
    re.IGNORECASE,
)
TITLE_ONLY_RE = re.compile(r"^[\s'\"“”‘’]*《[^\n。！？!?]{2,200}》[\s'\"“”‘’]*$")
HEADING_ONLY_RE = re.compile(
    r"^[^\n。！？!?]{4,60}(?:研究|探讨|综述|分析|进展|观察|报告)[\s'\"“”‘’]*$"
)
DOCUMENT_STYLE_RE = re.compile(r"^[\s'\"“”‘’]*(?:标题|摘要)[:：]|^(?:本文|本研究|该文|这篇文章)")
ADVICE_RE = re.compile(
    r"(?:推荐|使用|服用|用药|方剂|药方|处方|剂量|多少克|治疗方法|怎么治|如何调理|可以吃|可选用)"
)
PERSONAL_CONTEXT_RE = re.compile(
    r"我|本人|家人|孩子|宝宝|孕妇|孕期|妊娠|哺乳|儿童|小儿|老人|老年人|患者|病人|最近|患有|出现"
)
PERSONAL_ADVICE_ACTION_RE = re.compile(
    r"推荐|有没有什么|可以.{0,8}(?:用|服|吃)|能否.{0,8}(?:用|服|吃)|"
    r"该.{0,8}(?:用|服|吃)|怎么办|怎么治|如何调理|治疗建议|用药建议|"
    r"有什么.{0,12}(?:治疗|方法|中药|方剂)|有没有.{0,12}(?:治疗|方法|中药|方剂)"
)
DOSAGE_RE = re.compile(r"剂量|用量|多少克|几克|一次.{0,8}(?:用|服)|每天.{0,8}(?:用|服)")
UNVERIFIED_RESEARCH_RE = re.compile(
    r"研究|探讨|探究|探索|论文|现代医学|现代科学|现代研究|研究进展|药理|"
    r"作用机制|临床效果|实验|疗效|文献支持|文献参考"
)
SERIOUS_CONDITION_RE = re.compile(
    r"癌|肿瘤|白血病|脑出血|蛛网膜下腔出血|呼吸窘迫|昏迷|休克|心肌梗死|"
    r"心梗|脑梗|中风|器官衰竭|肝衰竭|肾衰竭|重症|急症"
)
MALFORMED_INSTRUCTION_RE = re.compile(
    r"关于的(?:研究|问题)|与相关[、，,]|与相关联的症状|哪些中医实体相关|"
    r"^本研究|^研究表明.{0,80}(?:本研究|我们将)"
)
UNSAFE_ABSOLUTE_RE = re.compile(
    r"无副作用|没有副作用|安全、无毒|温和、安全、无毒|绝对安全|完全无毒|"
    r"可以治愈|能够治愈|值得.{0,12}广泛应用"
)
VULNERABLE_RE = re.compile(r"孕妇|孕期|妊娠|哺乳|儿童|小儿|婴儿|老人|老年|肝功能|肾功能")
TOXIC_HERB_RE = re.compile(
    r"附子|川乌|草乌|马钱子|雄黄|朱砂|砒霜|半夏|天南星|雷公藤|关木通|细辛|罂粟壳"
)
SAFETY_RE = re.compile(
    r"医师指导|医生指导|专业医师|专业医生|就医|医院|辨证施治|不可自行|不宜自行|慎用|禁用|禁忌|遵医嘱"
)
LATIN_FRAGMENT_RE = re.compile(r"[A-Za-z]{4,}(?:\s+[A-Za-z]{3,})*")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project_dir / "data")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "data_audit")
    parser.add_argument("--sample-per-group", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260915)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            value["_source_line"] = line_number
            records.append(value)
    return records


def normalized_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def classify_task(instruction: str) -> str:
    for category, pattern in TASK_PATTERNS:
        if pattern.search(instruction):
            return category
    return "knowledge_qa"


def record_id(split: str, source_line: int, instruction: str) -> str:
    digest = hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:12]
    return f"{split}-{source_line:05d}-{digest}"


def audit_record(
    raw: dict[str, Any], split: str, duplicate_outputs: set[str]
) -> dict[str, Any]:
    source_line = int(raw["_source_line"])
    instruction = normalized_text(raw.get("instruction"))
    system_prompt = normalized_text(raw.get("input"))
    output = normalized_text(raw.get("output"))
    task_category = classify_task(instruction)

    flags: list[str] = []
    missing_fields = [field for field in REQUIRED_FIELDS if field not in raw]
    non_string_fields = [
        field for field in REQUIRED_FIELDS if field in raw and not isinstance(raw[field], str)
    ]
    if missing_fields:
        flags.append("missing_required_field")
    if non_string_fields:
        flags.append("non_string_field")
    if not instruction:
        flags.append("empty_instruction")
    if not system_prompt:
        flags.append("empty_system_prompt")
    if not output:
        flags.append("empty_output")
    if PLACEHOLDER_RE.fullmatch(output):
        flags.append("placeholder_output")
    if len(output) <= 15:
        flags.append("very_short_output")
    elif len(output) < 40:
        flags.append("short_output")

    is_title_only = bool(TITLE_ONLY_RE.fullmatch(output))
    if is_title_only:
        flags.append("title_only_output")
        if task_category != "title_generation":
            flags.append("title_only_task_mismatch")

    is_heading_only = bool(HEADING_ONLY_RE.fullmatch(output))
    if is_heading_only and not is_title_only:
        flags.append("heading_only_output")
        if task_category != "title_generation":
            flags.append("heading_only_task_mismatch")
    if DOCUMENT_STYLE_RE.search(output):
        flags.append("document_style_output")
        if task_category == "knowledge_qa":
            flags.append("document_style_task_mismatch")

    if output in duplicate_outputs:
        flags.append("duplicate_output")
    if LATIN_FRAGMENT_RE.search(instruction) or LATIN_FRAGMENT_RE.search(output):
        flags.append("latin_text_fragment")
    if MALFORMED_INSTRUCTION_RE.search(instruction):
        flags.append("malformed_instruction")
    if UNSAFE_ABSOLUTE_RE.search(output):
        flags.append("unsafe_absolute_claim")

    asks_for_advice = bool(ADVICE_RE.search(instruction))
    asks_for_personal_advice = bool(
        PERSONAL_CONTEXT_RE.search(instruction) and PERSONAL_ADVICE_ACTION_RE.search(instruction)
    )
    asks_for_dosage = bool(DOSAGE_RE.search(instruction))
    asks_for_unverified_research = bool(UNVERIFIED_RESEARCH_RE.search(instruction))
    mentions_serious_condition = bool(SERIOUS_CONDITION_RE.search(instruction))
    vulnerable_population = bool(VULNERABLE_RE.search(instruction))
    toxic_herb = bool(TOXIC_HERB_RE.search(instruction))
    has_safety_language = bool(SAFETY_RE.search(output))
    if asks_for_advice:
        flags.append("medical_advice")
        if not has_safety_language:
            flags.append("medical_advice_without_safety_language")
    if asks_for_personal_advice:
        flags.append("personal_medical_advice")
        if not has_safety_language:
            flags.append("personal_medical_advice_without_safety_language")
    if asks_for_dosage:
        flags.append("dosage_question")
        if not has_safety_language:
            flags.append("dosage_without_safety_language")
    if asks_for_unverified_research:
        flags.append("unverified_research_claim")
    if mentions_serious_condition:
        flags.append("serious_condition")
        if not has_safety_language:
            flags.append("serious_condition_without_safety_language")
    if vulnerable_population:
        flags.append("vulnerable_population")
        if not has_safety_language:
            flags.append("vulnerable_population_without_safety_language")
    if toxic_herb:
        flags.append("potentially_toxic_herb")
        if not has_safety_language:
            flags.append("toxic_herb_without_safety_language")

    hard_exclusion_reasons = sorted(
        set(flags)
        & {
            "missing_required_field",
            "non_string_field",
            "empty_instruction",
            "empty_system_prompt",
            "empty_output",
            "placeholder_output",
            "title_only_task_mismatch",
            "heading_only_task_mismatch",
            "document_style_task_mismatch",
            "malformed_instruction",
        }
    )
    review_reasons = sorted(
        set(flags)
        & {
            "very_short_output",
            "duplicate_output",
            "latin_text_fragment",
            "medical_advice_without_safety_language",
            "personal_medical_advice_without_safety_language",
            "dosage_question",
            "unverified_research_claim",
            "serious_condition_without_safety_language",
            "unsafe_absolute_claim",
            "vulnerable_population",
            "potentially_toxic_herb",
        }
    )

    return {
        "id": record_id(split, source_line, instruction),
        "split": split,
        "source_line": source_line,
        "instruction": instruction,
        "input": system_prompt,
        "output": output,
        "audit": {
            "task_category": task_category,
            "flags": sorted(set(flags)),
            "instruction_chars": len(instruction),
            "output_chars": len(output),
            "hard_exclusion_reasons": hard_exclusion_reasons,
            "review_reasons": review_reasons,
            "qa_candidate": task_category == "knowledge_qa" and not hard_exclusion_reasons,
        },
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            count += 1
    return count


def write_review_csv(path: Path, records: Iterable[dict[str, Any]]) -> int:
    fieldnames = (
        "decision",
        "review_notes",
        "id",
        "split",
        "source_line",
        "task_category",
        "flags",
        "instruction",
        "output",
    )
    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "decision": "",
                    "review_notes": "",
                    "id": record["id"],
                    "split": record["split"],
                    "source_line": record["source_line"],
                    "task_category": record["audit"]["task_category"],
                    "flags": "|".join(record["audit"]["flags"]),
                    "instruction": record["instruction"],
                    "output": record["output"],
                }
            )
            count += 1
    return count


def source_record(record: dict[str, Any]) -> dict[str, str]:
    return {field: record[field] for field in REQUIRED_FIELDS}


def sample_review_records(
    records: list[dict[str, Any]], sample_per_group: int, seed: int
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[f"task:{record['audit']['task_category']}"].append(record)
        for flag in record["audit"]["flags"]:
            groups[f"flag:{flag}"].append(record)

    rng = random.Random(seed)
    sampled_by_id: dict[str, dict[str, Any]] = {}
    sample_groups: dict[str, set[str]] = defaultdict(set)
    for group_name in sorted(groups):
        candidates = groups[group_name]
        chosen = rng.sample(candidates, min(sample_per_group, len(candidates)))
        for record in chosen:
            sampled_by_id[record["id"]] = record
            sample_groups[record["id"]].add(group_name)

    result: list[dict[str, Any]] = []
    for record_id_value in sorted(sampled_by_id):
        record = dict(sampled_by_id[record_id_value])
        record["sample_groups"] = sorted(sample_groups[record_id_value])
        result.append(record)
    return result


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, dict[str, Any]] = {}
    for split in ("train", "test"):
        selected = [record for record in records if record["split"] == split]
        categories = Counter(record["audit"]["task_category"] for record in selected)
        flags = Counter(flag for record in selected for flag in record["audit"]["flags"])
        by_split[split] = {
            "records": len(selected),
            "task_categories": dict(sorted(categories.items())),
            "flags": dict(sorted(flags.items())),
            "qa_candidates": sum(record["audit"]["qa_candidate"] for record in selected),
            "hard_exclusions": sum(bool(record["audit"]["hard_exclusion_reasons"]) for record in selected),
            "review_queue": sum(bool(record["audit"]["review_reasons"]) for record in selected),
        }
    return {"total_records": len(records), "splits": by_split}


def markdown_report(summary: dict[str, Any], sample_count: int) -> str:
    lines = [
        "# 中医数据集自动审计报告",
        "",
        "> 本报告由确定性规则生成。原始数据未被修改；`qa_candidate` 只是候选标记，不代表医学事实已经核验。",
        "",
        "## 总览",
        "",
        f"- 总记录数：{summary['total_records']}",
        f"- 分层抽样复核记录数：{sample_count}",
        "- 清洗原则：先分类和标记，再人工复核；不按单一长度阈值批量删除。",
        "",
        "| 划分 | 原始记录 | 问答候选 | 硬排除候选 | 需重点复核 |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, values in summary["splits"].items():
        lines.append(
            f"| {split} | {values['records']} | {values['qa_candidates']} | "
            f"{values['hard_exclusions']} | {values['review_queue']} |"
        )

    lines.extend(["", "## 任务类型", "", "| 类型 | train | test |", "|---|---:|---:|"])
    categories = sorted(
        set(summary["splits"]["train"]["task_categories"])
        | set(summary["splits"]["test"]["task_categories"])
    )
    for category in categories:
        train_count = summary["splits"]["train"]["task_categories"].get(category, 0)
        test_count = summary["splits"]["test"]["task_categories"].get(category, 0)
        lines.append(f"| `{category}` | {train_count} | {test_count} |")

    lines.extend(["", "## 风险标记", "", "| 标记 | train | test |", "|---|---:|---:|"])
    flags = sorted(
        set(summary["splits"]["train"]["flags"])
        | set(summary["splits"]["test"]["flags"])
    )
    for flag in flags:
        train_count = summary["splits"]["train"]["flags"].get(flag, 0)
        test_count = summary["splits"]["test"]["flags"].get(flag, 0)
        lines.append(f"| `{flag}` | {train_count} | {test_count} |")

    lines.extend(
        [
            "",
            "## 文件说明",
            "",
            "- `audit_records.jsonl`：全部记录及审计元数据。",
            "- `qa_candidates_train.jsonl` / `qa_candidates_test.jsonl`：保守筛出的知识问答候选，尚未完成医学事实核验。",
            "- `hard_exclusions.jsonl`：格式损坏、占位回答或明显标题错配等候选。",
            "- `review_queue.jsonl`：短回答、医疗建议、特殊人群、有毒药材等重点复核样本。",
            "- `review_sample.jsonl`：按任务类型与风险标记生成的可复现分层样本。",
            "- `review_sample.csv`：同一批分层样本的人工复核表，可填写 `decision` 和 `review_notes`。",
            "- `summary.json`：机器可读统计。",
            "",
            "## 解释边界",
            "",
            "规则只能识别结构和关键词风险，不能判断中医知识是否正确。正式训练前仍需对高风险样本人工或借助可靠知识源复核。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.sample_per_group < 1:
        raise ValueError("--sample-per-group must be at least 1")

    paths = {
        "train": args.data_dir / "chinese_medical_train.jsonl",
        "test": args.data_dir / "chinese_medical_test.jsonl",
    }
    raw_by_split = {split: read_jsonl(path) for split, path in paths.items()}
    duplicate_outputs: set[str] = set()
    for split_records in raw_by_split.values():
        counts = Counter(normalized_text(record.get("output")) for record in split_records)
        duplicate_outputs.update(output for output, count in counts.items() if output and count > 1)

    audited: list[dict[str, Any]] = []
    for split, raw_records in raw_by_split.items():
        audited.extend(audit_record(record, split, duplicate_outputs) for record in raw_records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "audit_records.jsonl", audited)
    write_jsonl(
        args.output_dir / "qa_candidates_train.jsonl",
        (source_record(record) for record in audited if record["split"] == "train" and record["audit"]["qa_candidate"]),
    )
    write_jsonl(
        args.output_dir / "qa_candidates_test.jsonl",
        (source_record(record) for record in audited if record["split"] == "test" and record["audit"]["qa_candidate"]),
    )
    write_jsonl(
        args.output_dir / "hard_exclusions.jsonl",
        (record for record in audited if record["audit"]["hard_exclusion_reasons"]),
    )
    write_jsonl(
        args.output_dir / "review_queue.jsonl",
        (record for record in audited if record["audit"]["review_reasons"]),
    )
    sampled = sample_review_records(audited, args.sample_per_group, args.seed)
    write_jsonl(args.output_dir / "review_sample.jsonl", sampled)
    write_review_csv(args.output_dir / "review_sample.csv", sampled)

    summary = summarize(audited)
    summary["configuration"] = {
        "seed": args.seed,
        "sample_per_group": args.sample_per_group,
        "source_files": {split: str(path) for split, path in paths.items()},
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    (args.output_dir / "REPORT.md").write_text(
        markdown_report(summary, len(sampled)), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
