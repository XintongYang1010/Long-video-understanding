#!/usr/bin/env python3
"""
Prepare human-audit files for historical-memory feasibility v1.

This script only reads existing CSV outputs and writes derived manual-audit
files under outputs/historical_v1/manual_audit/. It does not run models,
download videos, or modify original MA-EgoQA data or historical_v1 source CSVs.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORICAL_DIR = ROOT / "outputs" / "historical_v1"
DEFAULT_OUTPUT_DIR = DEFAULT_HISTORICAL_DIR / "manual_audit"

AUDIT_TABLE_NAME = "historical_memory_candidate_audit_table.csv"
RETRIEVAL_NAME = "historical_retrieval_candidates.csv"
SELECTED_NAME = "selected_historical_candidate_questions.csv"

AGENT_NAMES = [
    "Jake",
    "Jack",
    "Alice",
    "Tasha",
    "Lucia",
    "Lucy",
    "Katrina",
    "Shure",
    "Nicous",
    "Violet",
    "Luyue",
    "Choiszt",
]

MANUAL_FIELDNAMES = [
    "case_id",
    "question_id",
    "category",
    "question",
    "answer",
    "context_day",
    "context_start",
    "context_end",
    "agents_in_context",
    "provisional_query_user",
    "query_user_status",
    "top_self_current_caption",
    "top_self_history_caption",
    "top_external_current_caption",
    "top_external_history_caption",
    "top_all_agents_current_caption",
    "top_all_agents_history_caption",
    "human_query_user",
    "query_user_valid",
    "is_self_first_suitable",
    "self_current_sufficient",
    "self_history_sufficient",
    "external_current_sufficient",
    "external_history_sufficient",
    "historical_memory_helpful",
    "final_case_type",
    "oracle_source_agent",
    "oracle_time_window",
    "oracle_caption",
    "audit_notes",
    "keep_for_demo",
]

MANUAL_LABEL_FIELDS = [
    "human_query_user",
    "query_user_valid",
    "is_self_first_suitable",
    "self_current_sufficient",
    "self_history_sufficient",
    "external_current_sufficient",
    "external_history_sufficient",
    "historical_memory_helpful",
    "final_case_type",
    "oracle_source_agent",
    "oracle_time_window",
    "oracle_caption",
    "audit_notes",
    "keep_for_demo",
]

MODE_TO_TOP_FIELD = {
    "self_current": "top_self_current_caption",
    "self_history": "top_self_history_caption",
    "external_current": "top_external_current_caption",
    "external_history": "top_external_history_caption",
    "all_agents_current": "top_all_agents_current_caption",
    "all_agents_history": "top_all_agents_history_caption",
}

CATEGORY_PRIORITY = {
    "theory_of_mind": 24,
    "task_coordination": 20,
    "temporal_reasoning": 18,
    "social_interaction": 8,
    "environmental_interaction": 0,
}

STOPWORDS = {
    "about",
    "after",
    "again",
    "ahead",
    "all",
    "along",
    "also",
    "among",
    "around",
    "before",
    "behind",
    "being",
    "between",
    "could",
    "during",
    "each",
    "everyone",
    "following",
    "from",
    "first",
    "have",
    "her",
    "helped",
    "into",
    "like",
    "look",
    "made",
    "make",
    "most",
    "next",
    "other",
    "others",
    "over",
    "participants",
    "people",
    "person",
    "seemed",
    "someone",
    "something",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "time",
    "used",
    "using",
    "was",
    "what",
    "when",
    "where",
    "which",
    "while",
    "whose",
    "why",
    "with",
    "would",
}

STATISTICAL_PATTERNS = [
    re.compile(r"\bwho\s+used\b.*\bthe\s+most\b", flags=re.IGNORECASE),
    re.compile(r"\bthe\s+most\b", flags=re.IGNORECASE),
    re.compile(r"\bhow\s+many\b", flags=re.IGNORECASE),
    re.compile(r"\bnumber\s+of\b", flags=re.IGNORECASE),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def normalize_ws(text: Any) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())


def trunc(text: Any, limit: int = 280) -> str:
    clean = normalize_ws(text)
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def case_sort_key(case_id: str) -> tuple[str, int]:
    match = re.search(r"(\d+)$", case_id)
    if match:
        return (case_id[: match.start()], int(match.group(1)))
    return (case_id, 0)


def coalesce(*values: Any) -> str:
    for value in values:
        if value is not None and str(value) != "":
            return str(value)
    return ""


def rank_as_int(row: dict[str, str]) -> int:
    try:
        return int(row.get("rank", "999999"))
    except ValueError:
        return 999999


def score_as_float(row: dict[str, str]) -> float:
    try:
        return float(row.get("score", "0"))
    except ValueError:
        return 0.0


def top_retrieval_rows(retrieval_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in retrieval_rows:
        grouped[row.get("retrieval_mode", "")].append(row)

    tops = {}
    for mode, rows in grouped.items():
        rows = sorted(rows, key=lambda row: (rank_as_int(row), -score_as_float(row)))
        if rows:
            tops[mode] = rows[0]
    return tops


def format_top_caption(row: dict[str, str] | None) -> str:
    if not row:
        return ""
    meta = (
        f"{row.get('source_agent', '')} D{row.get('day', '')} "
        f"{row.get('start_time', '')}-{row.get('end_time', '')} "
        f"{row.get('granularity', '')} rank={row.get('rank', '')} score={row.get('score', '')}"
    )
    return f"{normalize_ws(meta)} | {normalize_ws(row.get('caption_text', ''))}"


def tokenize(text: Any) -> set[str]:
    tokens = set()
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9']+", normalize_ws(text).lower()):
        token = token.strip("'")
        if len(token) < 3 or token in STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def overlap_terms(question: str, caption: str) -> set[str]:
    return tokenize(question) & tokenize(caption)


def has_explicit_person_name(question: str) -> bool:
    return any(re.search(rf"\b{re.escape(name)}\b", question, flags=re.IGNORECASE) for name in AGENT_NAMES)


def is_statistical_question(question: str) -> bool:
    return any(pattern.search(question) for pattern in STATISTICAL_PATTERNS)


def caption_from_manual(row: dict[str, str], field: str) -> str:
    return row.get(field, "")


def build_manual_rows(
    selected_rows: list[dict[str, str]],
    audit_rows: list[dict[str, str]],
    retrieval_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[str, list[dict[str, str]]]]:
    selected_by_case = {row["case_id"]: row for row in selected_rows}
    audit_by_case = {row["case_id"]: row for row in audit_rows}
    retrieval_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in retrieval_rows:
        retrieval_by_case[row.get("case_id", "")].append(row)

    case_ids = sorted(set(selected_by_case) | set(audit_by_case), key=case_sort_key)
    manual_rows = []
    for case_id in case_ids:
        selected = selected_by_case.get(case_id, {})
        audit = audit_by_case.get(case_id, {})
        tops = top_retrieval_rows(retrieval_by_case.get(case_id, []))

        manual = {
            "case_id": case_id,
            "question_id": coalesce(selected.get("question_id"), audit.get("question_id")),
            "category": coalesce(selected.get("category"), audit.get("category")),
            "question": coalesce(selected.get("question"), audit.get("question")),
            "answer": coalesce(selected.get("answer"), audit.get("answer")),
            "context_day": coalesce(selected.get("context_day"), audit.get("context_day")),
            "context_start": coalesce(selected.get("context_start"), audit.get("context_start")),
            "context_end": coalesce(selected.get("context_end"), audit.get("context_end")),
            "agents_in_context": coalesce(selected.get("agents_in_context"), audit.get("agents_in_context")),
            "provisional_query_user": coalesce(selected.get("query_user"), audit.get("query_user")),
            "query_user_status": coalesce(selected.get("query_user_status"), audit.get("query_user_status")),
        }

        for mode, field in MODE_TO_TOP_FIELD.items():
            manual[field] = format_top_caption(tops.get(mode))

        for field in MANUAL_LABEL_FIELDS:
            manual[field] = ""
        manual_rows.append(manual)

    return manual_rows, retrieval_by_case


def relevance_metrics(row: dict[str, str]) -> dict[str, Any]:
    question = row["question"]
    fields = [
        "top_self_current_caption",
        "top_self_history_caption",
        "top_external_current_caption",
        "top_external_history_caption",
        "top_all_agents_current_caption",
        "top_all_agents_history_caption",
    ]
    overlaps = {field: overlap_terms(question, row.get(field, "")) for field in fields}
    return {
        "overlaps": overlaps,
        "self_current_rel": len(overlaps["top_self_current_caption"]),
        "self_history_rel": len(overlaps["top_self_history_caption"]),
        "external_current_rel": len(overlaps["top_external_current_caption"]),
        "external_history_rel": len(overlaps["top_external_history_caption"]),
        "all_current_rel": len(overlaps["top_all_agents_current_caption"]),
        "all_history_rel": len(overlaps["top_all_agents_history_caption"]),
        "has_person": has_explicit_person_name(question),
        "is_statistical": is_statistical_question(question),
    }


def priority_score_and_reasons(row: dict[str, str]) -> tuple[int, list[str], str, str]:
    metrics = relevance_metrics(row)
    category = row.get("category", "")
    query_status = row.get("query_user_status", "")
    query_user = row.get("provisional_query_user", "")
    score = 0
    reasons: list[str] = []

    if query_status == "inferred_weak":
        score += 35
        reasons.append("query_user 是 inferred_weak，适合优先人工确认 self-first 起点")
    elif query_user and query_user != "UNKNOWN":
        score += 12
        reasons.append("已有非 UNKNOWN 的 provisional query_user")
    else:
        score -= 8
        reasons.append("query_user 仍是 UNKNOWN，需要先判断是否适合 self-first")

    category_bonus = CATEGORY_PRIORITY.get(category, 0)
    score += category_bonus
    if category_bonus:
        reasons.append(f"类别 {category} 是本轮优先审查类型")

    if metrics["has_person"]:
        score += 12
        reasons.append("问题中有明确人物名")

    self_hist_rel = metrics["self_history_rel"]
    ext_hist_rel = metrics["external_history_rel"]
    all_hist_rel = metrics["all_history_rel"]
    self_cur_rel = metrics["self_current_rel"]
    ext_cur_rel = metrics["external_current_rel"]
    all_cur_rel = metrics["all_current_rel"]
    hist_rel = max(self_hist_rel, ext_hist_rel, all_hist_rel)
    cur_rel = max(self_cur_rel, ext_cur_rel, all_cur_rel)

    if hist_rel >= 3:
        score += 18
        terms = sorted(
            max(metrics["overlaps"].values(), key=lambda terms: (len(terms), sorted(terms)))
        )[:6]
        reasons.append(f"history top caption 与问题有词面重合：{', '.join(terms)}")
    elif hist_rel >= 1:
        score += 6
        reasons.append("history top caption 有少量词面相关信号")

    if hist_rel > cur_rel:
        score += 12
        reasons.append("history 检索信号看起来强于 current，适合检查历史记忆是否补证据")
    elif hist_rel >= 2 and cur_rel <= 1:
        score += 10
        reasons.append("current 词面信号弱但 history 有相关信号")

    if row.get("top_self_history_caption") and self_hist_rel >= 2:
        score += 13
        reasons.append("self_history top caption 看起来相关")
    if row.get("top_external_history_caption") and ext_hist_rel >= 2:
        score += 10
        reasons.append("external_history top caption 看起来相关")
    if self_hist_rel >= 2 and self_cur_rel <= 1:
        score += 10
        reasons.append("self_current 可能不足，self_history 可能补充")
    if ext_hist_rel >= 2 and ext_cur_rel <= 1:
        score += 8
        reasons.append("external_current 可能不足，external_history 可能补充")

    if metrics["is_statistical"]:
        if hist_rel >= 4:
            score -= 8
            reasons.append("问题偏统计型，但 history 信号较明显，保留为次优先")
        else:
            score -= 35
            reasons.append("问题偏纯统计/most 型且 caption 难直接支持，降低优先级")

    potential_hint = potential_final_label_hint(row, metrics)
    human_check = human_check_prompt(row, metrics)
    return score, reasons, potential_hint, human_check


def potential_final_label_hint(row: dict[str, str], metrics: dict[str, Any]) -> str:
    query_user = row.get("provisional_query_user", "")
    self_hist_rel = metrics["self_history_rel"]
    ext_hist_rel = metrics["external_history_rel"]
    self_cur_rel = metrics["self_current_rel"]
    ext_cur_rel = metrics["external_current_rel"]
    all_hist_rel = metrics["all_history_rel"]
    all_cur_rel = metrics["all_current_rel"]

    if not query_user or query_user == "UNKNOWN":
        if all_hist_rel > all_cur_rel and all_hist_rel >= 2:
            return "人工待定：先确认 query_user；若无法确认，可能是 not_self_first 或 reject_unclear"
        return "人工待定：query_user 不清楚，优先检查是否应标 not_self_first/reject_unclear"
    if self_hist_rel >= 2 and self_cur_rel <= 1:
        return "人工待定：重点检查是否为 self_history_needed"
    if ext_hist_rel >= 2 and ext_cur_rel <= 1:
        return "人工待定：重点检查是否为 external_history_needed"
    if self_cur_rel >= 2:
        return "人工待定：可能是 self_current_sufficient，也可能需要历史校验"
    if ext_cur_rel >= 2:
        return "人工待定：可能是 external_current_needed"
    return "人工待定：证据可能不足，检查是否 reject_unclear"


def human_check_prompt(row: dict[str, str], metrics: dict[str, Any]) -> str:
    query_user = row.get("provisional_query_user", "")
    parts = []
    if not query_user or query_user == "UNKNOWN":
        parts.append("先判断是否能从问题/上下文确认 human_query_user")
    else:
        parts.append(f"确认 {query_user} 是否真的是 query_user")
    parts.append("比较 self/external 的 current 与 history caption 是否能支持答案所需证据")
    if metrics["is_statistical"]:
        parts.append("这是统计/most 型问题，必须检查 caption 是否真的覆盖足够事件，不能只看 top1")
    parts.append("不要把 BM25 分数或排序当作最终标签")
    return "；".join(parts)


def build_priority_subset(manual_rows: list[dict[str, str]], limit: int = 10) -> list[dict[str, Any]]:
    scored = []
    for row in manual_rows:
        score, reasons, potential_hint, human_check = priority_score_and_reasons(row)
        scored.append(
            {
                "row": row,
                "priority_score": score,
                "priority_reasons": " | ".join(reasons),
                "potential_final_label_hint": potential_hint,
                "what_human_should_check": human_check,
            }
        )

    scored.sort(key=lambda item: (-item["priority_score"], case_sort_key(item["row"]["case_id"])))
    subset = []
    for rank, item in enumerate(scored[:limit], start=1):
        out = {
            "priority_rank": rank,
            "priority_score": item["priority_score"],
            "priority_reasons": item["priority_reasons"],
            "potential_final_label_hint": item["potential_final_label_hint"],
            "what_human_should_check": item["what_human_should_check"],
        }
        out.update(item["row"])
        for field in MANUAL_LABEL_FIELDS:
            out[field] = ""
        subset.append(out)
    return subset


def brief_for_case(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"Case ID: {row['case_id']}",
            f"Question: {row['question']}",
            f"Answer: {row['answer']}",
            f"Provisional query_user: {row['provisional_query_user']} ({row['query_user_status']})",
            f"Why this case may be useful: {row['priority_reasons']}",
            f"Top self current: {trunc(row['top_self_current_caption'], 360) or '(empty)'}",
            f"Top self history: {trunc(row['top_self_history_caption'], 360) or '(empty)'}",
            f"Top external current: {trunc(row['top_external_current_caption'], 360) or '(empty)'}",
            f"Top external history: {trunc(row['top_external_history_caption'], 360) or '(empty)'}",
            f"What human should check: {row['what_human_should_check']}",
            f"Potential final label: {row['potential_final_label_hint']}",
        ]
    )


def write_audit_brief(path: Path, priority_rows: list[dict[str, Any]]) -> None:
    sections = [
        "# Historical Memory Manual Audit Brief v1",
        "",
        "说明：以下 10 个 case 是按文本启发式排序出的人工优先审查对象。这里的 Potential final label 只是待检查方向，不是自动标签。",
        "",
    ]
    for row in priority_rows:
        sections.append(brief_for_case(row))
        sections.append("")
        sections.append("---")
        sections.append("")
    path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")


def write_manual_instructions(path: Path) -> None:
    text = """# manual_historical_case_audit_v1 标注说明

本轮只做人审准备，不判断答案正确性。CSV 里的检索 caption 和 BM25 分数只是帮助你定位证据的线索，不是最终 label。

## 基本原则

- 每行是一个候选 case。
- `provisional_query_user` 是脚本弱推断的候选用户，不一定正确。
- 所有人工字段初始为空；请只在人工读过问题、上下文和 caption 后填写。
- 不要把 BM25 score、rank 或 top1 caption 当成真实标签。它们只能提示“先看哪里”。
- 不要 claim answer accuracy；本表标的是 source routing / historical memory 是否有证据支持。

## 人工字段怎么填

`human_query_user`：你人工判断的查询用户/第一人称主体。无法判断就留空或写 `unclear`。

`query_user_valid`：`yes` 表示 provisional query_user 与你判断一致；`no` 表示不一致；`unclear` 表示无法从材料判断。

`is_self_first_suitable`：这个问题是否适合先查 query_user 自己的视角。若问题是全局统计、多人整体比较、没有明确第一人称主体，或一开始就必须依赖其他人视角，通常填 `no`。

`self_current_sufficient`：在确认 human_query_user 后，只看该用户 current/context window 内的 caption，是否足以支持回答所需的关键证据。若 current caption 只部分相关、缺少关键事件、时间不对，或必须依赖之前记忆/他人视角，填 `no`。证据不够清楚填 `unclear`。

`self_history_sufficient`：该用户在 context window 之前的历史 caption，是否提供了 current 缺失的关键证据。只有当 self history 能独立或明确补足关键事实时填 `yes`；如果只是词面相关但不能支持判断，填 `no` 或 `unclear`。

`external_current_sufficient`：非 query_user 的 current/context window caption 是否足以支持关键证据。若需要当前其他人的视角才能回答，且 caption 清楚支持，填 `yes`。

`external_history_sufficient`：非 query_user 的历史 caption 是否足以支持关键证据。若 evidence 来自别人过去看到/说过/做过的内容，且能支持问题，填 `yes`。

`historical_memory_helpful`：如果 self_history 或 external_history 相比 current caption 明显补充了必要证据，填 `yes`。如果 current 已足够、history 只是冗余或无关，填 `no`。无法判断填 `unclear`。

`final_case_type` 只能填以下之一：

- `self_current_sufficient`
- `self_history_needed`
- `external_current_needed`
- `external_history_needed`
- `not_self_first`
- `reject_unclear`

`oracle_source_agent`：你认为最关键证据来自哪个 agent。多个 agent 可用分号分隔。

`oracle_time_window`：关键证据的大致时间窗口，例如 `DAY3 18:10-18:20`。

`oracle_caption`：最能支持人工判断的 caption 摘要或原文片段。

`audit_notes`：写下为什么这样标，尤其是 query_user 不清、caption 不足、时间窗口不匹配等情况。

`keep_for_demo`：适合放进论文/报告 demo 的填 `yes`；不适合填 `no`；暂不确定填 `unclear`。

## 什么时候标 reject_unclear

- 无法确认 query_user，且问题是否 self-first 也说不清。
- 检索 caption 只有词面相关，无法支持答案所需事实。
- 时间窗口明显不匹配，无法判断 current/history 的边界。
- 问题本身依赖未提供的视频细节，caption 不足以审。
- 答案可能对但当前证据链不清楚；不要为了保留 case 硬标。

## 什么时候不适合 self-first

- 问题问的是全体成员统计或全局比较，例如“who used X the most”，且没有明确 query_user。
- 问题天然需要多人的外部视角或旁观者视角，query_user 自己不可能优先提供主要证据。
- `human_query_user` 无法确定，或 provisional query_user 只是名字出现在问题里但不是实际主体。
- 答案需要聚合多个 agent 的经历，而不是先从某个 self stream 开始扩展。
"""
    path.write_text(text, encoding="utf-8")


def write_outputs(historical_dir: Path, output_dir: Path) -> dict[str, Any]:
    selected_rows = read_csv(historical_dir / SELECTED_NAME)
    audit_rows = read_csv(historical_dir / AUDIT_TABLE_NAME)
    retrieval_rows = read_csv(historical_dir / RETRIEVAL_NAME)

    manual_rows, _ = build_manual_rows(selected_rows, audit_rows, retrieval_rows)
    priority_rows = build_priority_subset(manual_rows, limit=10)

    manual_path = output_dir / "manual_historical_case_audit_v1.csv"
    priority_path = output_dir / "audit_priority_subset.csv"
    brief_path = output_dir / "audit_brief.md"
    instructions_path = output_dir / "manual_audit_instructions.md"

    write_csv(manual_path, manual_rows, MANUAL_FIELDNAMES)
    priority_fieldnames = [
        "priority_rank",
        "priority_score",
        "priority_reasons",
        "potential_final_label_hint",
        "what_human_should_check",
        *MANUAL_FIELDNAMES,
    ]
    write_csv(priority_path, priority_rows, priority_fieldnames)
    write_audit_brief(brief_path, priority_rows)
    write_manual_instructions(instructions_path)

    return {
        "manual_path": manual_path,
        "priority_path": priority_path,
        "brief_path": brief_path,
        "instructions_path": instructions_path,
        "manual_rows": manual_rows,
        "priority_rows": priority_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical-dir",
        type=Path,
        default=DEFAULT_HISTORICAL_DIR,
        help="Directory containing historical_v1 input CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for manual audit outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_outputs(args.historical_dir, args.output_dir)

    print("Generated files:")
    for key in ["manual_path", "priority_path", "brief_path", "instructions_path"]:
        print(f"- {result[key]}")

    print("\nPriority 10 cases:")
    for row in result["priority_rows"]:
        print(f"- {row['case_id']} / Q{row['question_id']} / {row['question']}")

    print("\nTop 5 to inspect first:")
    for row in result["priority_rows"][:5]:
        print(f"- {row['case_id']} / Q{row['question_id']}: {row['priority_reasons']}")


if __name__ == "__main__":
    main()
