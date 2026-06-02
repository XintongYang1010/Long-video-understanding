#!/usr/bin/env python3
"""
Build Dataset V0.1 from MA-EgoQA historical V2 full screening outputs.

This creates caption-only, auto-screened candidate cases for evidence-scope
comparison. It does not download video, run VLMs, run LLM/API calls, modify
original MA-EgoQA files, claim answer accuracy, or treat labels as final.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
V2_DIR = ROOT / "outputs" / "historical_v2_fullscreen"
OUTPUT_DIR = V2_DIR / "dataset_v0_1"

FULLSCREEN_PATH = V2_DIR / "evidence_scope_fullscreen_v2.csv"
TOPK_EVIDENCE_PATH = V2_DIR / "evidence_scope_fullscreen_v2_topk_evidence.csv"
TIER_A_PATH = V2_DIR / "tier_A_demo_ready_cases_v2.csv"
TIER_B_PATH = V2_DIR / "tier_B_promising_history_gain_cases_v2.csv"
TIER_C_PATH = V2_DIR / "tier_C_current_sufficient_controls_v2.csv"

DATASET_FIELDNAMES = [
    "case_id",
    "source_question_id",
    "source_tier",
    "question",
    "answer",
    "category",
    "subcategory",
    "case_type",
    "current_only_context",
    "history_only_context",
    "current_plus_historical_context",
    "current_only_answerability",
    "history_only_answerability",
    "current_plus_history_answerability",
    "current_plus_history_gain",
    "expected_comparison",
    "expected_result",
    "evidence_sources",
    "evidence_time_windows",
    "confidence",
    "why_selected",
    "potential_issue",
    "label_status",
    "needs_human_check",
]

ANSWERABILITY_RANK = {
    "not_answerable": 0,
    "unclear": 1,
    "partially_answerable": 2,
    "likely_answerable": 3,
}

PREFERRED_DEMO_CATEGORIES = ["theory_of_mind", "task_coordination", "social_interaction"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: safe_csv_value(row.get(field, "")) for field in fieldnames})


def safe_csv_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
    return value


def normalize_ws(text: Any) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())


def trunc(text: Any, limit: int = 800) -> str:
    clean = normalize_ws(text)
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def split_hits(value: str) -> list[str]:
    return [item for item in str(value or "").split(";") if item]


def is_exhaustive_global_question(question: str) -> bool:
    lower = question.lower()
    patterns = [
        r"\bwho\s+used\b.*\bthe\s+most\b",
        r"\bused\s+the\s+most\b",
        r"\bfirst\s+time\b",
        r"\blast\s+time\b",
        r"\bwhich\s+of\s+the\s+following\s+happened\s+(first|last)\b",
        r"\bcorrect\s+sequence\s+of\s+events\b",
        r"\bhow\s+many\b",
        r"\bnumber\s+of\b",
        r"\bfirst\s+person\b",
        r"\blast\s+person\b",
        r"\bwho\s+was\s+the\s+(first|last)\b",
        r"\bwhen\s+was\b.*\bused\s+the\s+most\b",
    ]
    return any(re.search(pattern, lower) for pattern in patterns)


def answerability_rank(row: dict[str, str], field: str) -> int:
    return ANSWERABILITY_RANK.get(row.get(field, ""), 0)


def is_strong_partial(row: dict[str, str], prefix: str = "current_plus_history") -> bool:
    label = row.get(f"{prefix}_answerability", "")
    coverage = parse_float(row.get(f"{prefix}_answer_coverage", "0"))
    overlap = parse_int(row.get(f"{prefix}_question_overlap", "0"))
    hits = split_hits(row.get(f"{prefix}_answer_hits", ""))
    return label == "partially_answerable" and coverage >= 0.45 and overlap >= 3 and len(hits) >= 2


def current_is_weak(row: dict[str, str]) -> bool:
    label = row.get("current_only_answerability", "")
    if label in {"not_answerable", "unclear"}:
        return True
    if label == "partially_answerable":
        return not is_strong_partial(row, "current_only")
    return False


def history_relevant(row: dict[str, str]) -> bool:
    label = row.get("history_only_answerability", "")
    coverage = parse_float(row.get("history_only_answer_coverage", "0"))
    overlap = parse_int(row.get("history_only_question_overlap", "0"))
    hits = split_hits(row.get("history_only_answer_hits", ""))
    return label in {"likely_answerable", "partially_answerable"} or (
        label == "unclear" and (coverage >= 0.25 or overlap >= 4 or len(hits) >= 1)
    )


def evidence_readable(row: dict[str, str]) -> bool:
    current = normalize_ws(row.get("best_current_evidence_summary", ""))
    history = normalize_ws(row.get("best_history_evidence_summary", ""))
    plus = normalize_ws(row.get("best_current_plus_history_summary", ""))
    return len(current) >= 80 and len(history) >= 120 and len(plus) >= 180


def evidence_has_substance(row: dict[str, str]) -> bool:
    hits = split_hits(row.get("current_plus_history_answer_hits", ""))
    coverage = parse_float(row.get("current_plus_history_answer_coverage", "0"))
    overlap = parse_int(row.get("current_plus_history_question_overlap", "0"))
    history_score = parse_float(row.get("history_only_top_score", "0"))
    return len(hits) >= 2 and coverage >= 0.35 and overlap >= 3 and history_score > 0


def demo_eligible(row: dict[str, str]) -> bool:
    if row.get("source_tier") != "tier_A_demo_ready":
        return False
    if row.get("current_plus_history_gain") != "yes":
        return False
    if is_exhaustive_global_question(row.get("question", "")) or row.get("is_pure_global_statistical") == "yes":
        return False
    plus_ok = row.get("current_plus_history_answerability") == "likely_answerable" or is_strong_partial(row)
    return plus_ok and current_is_weak(row) and history_relevant(row) and evidence_readable(row) and evidence_has_substance(row)


def eval_eligible(row: dict[str, str]) -> bool:
    if row.get("source_tier") not in {"tier_A_demo_ready", "tier_B_promising_history_gain"}:
        return False
    if row.get("current_plus_history_gain") not in {"yes", "unclear_but_promising"}:
        return False
    if is_exhaustive_global_question(row.get("question", "")) or row.get("is_pure_global_statistical") == "yes":
        return False
    return history_relevant(row) and evidence_readable(row) and evidence_has_substance(row)


def medium_eval_eligible(row: dict[str, str]) -> bool:
    if row.get("source_tier") not in {"tier_A_demo_ready", "tier_B_promising_history_gain"}:
        return False
    if row.get("current_plus_history_gain") == "no":
        return False
    if is_exhaustive_global_question(row.get("question", "")) or row.get("is_pure_global_statistical") == "yes":
        return False
    return evidence_readable(row) and history_relevant(row)


def control_eligible(row: dict[str, str]) -> bool:
    if row.get("source_tier") != "tier_C_current_sufficient_control":
        return False
    if row.get("current_only_answerability") != "likely_answerable":
        return False
    if row.get("current_plus_history_gain") != "no":
        return False
    if is_exhaustive_global_question(row.get("question", "")) or row.get("is_pure_global_statistical") == "yes":
        return False
    return evidence_readable(row)


def demo_score(row: dict[str, str]) -> tuple[float, int, int, int]:
    plus_cov = parse_float(row.get("current_plus_history_answer_coverage", "0"))
    hist_cov = parse_float(row.get("history_only_answer_coverage", "0"))
    current_cov = parse_float(row.get("current_only_answer_coverage", "0"))
    plus_overlap = parse_int(row.get("current_plus_history_question_overlap", "0"))
    hist_score = parse_float(row.get("history_only_top_score", "0"))
    preferred_bonus = 0.8 if row.get("category") in PREFERRED_DEMO_CATEGORIES else 0.0
    weak_bonus = 1.0 if row.get("current_only_answerability") in {"not_answerable", "unclear"} else 0.4
    score = preferred_bonus + weak_bonus + plus_cov + 0.6 * hist_cov - 0.3 * current_cov + min(hist_score / 25.0, 1.2)
    return (score, plus_overlap, len(split_hits(row.get("current_plus_history_answer_hits", ""))), -parse_int(row["question_id"]))


def eval_score(row: dict[str, str]) -> tuple[int, float, float, int, int]:
    gain_rank = {"yes": 2, "unclear_but_promising": 1, "unclear": 0}.get(row.get("current_plus_history_gain", ""), 0)
    category_bonus = 1 if row.get("category") in PREFERRED_DEMO_CATEGORIES else 0
    score = parse_float(row.get("current_plus_history_top_score", "0"))
    coverage = parse_float(row.get("current_plus_history_answer_coverage", "0"))
    overlap = parse_int(row.get("current_plus_history_question_overlap", "0"))
    return (gain_rank, category_bonus + coverage, score, overlap, -parse_int(row["question_id"]))


def control_score(row: dict[str, str]) -> tuple[float, int, int]:
    current_score = parse_float(row.get("current_only_top_score", "0"))
    current_cov = parse_float(row.get("current_only_answer_coverage", "0"))
    current_overlap = parse_int(row.get("current_only_question_overlap", "0"))
    return (current_cov + min(current_score / 25.0, 1.2), current_overlap, -parse_int(row["question_id"]))


def with_source(rows: list[dict[str, str]], source_tier: str) -> list[dict[str, str]]:
    return [dict(row, source_tier=source_tier) for row in rows]


def load_candidates() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    fullscreen = read_csv(FULLSCREEN_PATH)
    tier_a = with_source(read_csv(TIER_A_PATH), "tier_A_demo_ready")
    tier_b = with_source(read_csv(TIER_B_PATH), "tier_B_promising_history_gain")
    tier_c = with_source(read_csv(TIER_C_PATH), "tier_C_current_sufficient_control")
    stats = {
        "processed_questions": len(fullscreen),
        "tier_counts": Counter(row.get("evidence_scope_bucket", "") for row in fullscreen),
        "gain_counts": Counter(row.get("current_plus_history_gain", "") for row in fullscreen),
        "tier_A_input": len(tier_a),
        "tier_B_input": len(tier_b),
        "tier_C_input": len(tier_c),
    }
    return tier_a, tier_b, tier_c, stats


def select_demo(tier_a: list[dict[str, str]], target: int = 10) -> list[dict[str, str]]:
    eligible = sorted([row for row in tier_a if demo_eligible(row)], key=demo_score, reverse=True)
    selected: list[dict[str, str]] = []
    used = set()
    for category in PREFERRED_DEMO_CATEGORIES:
        for row in eligible:
            if row["question_id"] not in used and row.get("category") == category:
                selected.append(row)
                used.add(row["question_id"])
                break
    for row in eligible:
        if len(selected) >= target:
            break
        if row["question_id"] in used:
            continue
        selected.append(row)
        used.add(row["question_id"])
    return selected[:target]


def select_eval_lite(
    demo_rows: list[dict[str, str]],
    tier_a: list[dict[str, str]],
    tier_b: list[dict[str, str]],
    target: int = 35,
) -> list[dict[str, str]]:
    selected = list(demo_rows)
    used = {row["question_id"] for row in selected}

    gain_yes_pool = [
        row
        for row in tier_a + tier_b
        if row["question_id"] not in used
        and row.get("current_plus_history_gain") == "yes"
        and eval_eligible(row)
    ]
    promising_pool = [
        row
        for row in tier_a + tier_b
        if row["question_id"] not in used
        and row.get("current_plus_history_gain") == "unclear_but_promising"
        and medium_eval_eligible(row)
    ]
    gain_yes_pool.sort(key=eval_score, reverse=True)
    promising_pool.sort(key=eval_score, reverse=True)

    max_gain_yes_before_promising = 25
    for row in gain_yes_pool:
        if len(selected) >= min(max_gain_yes_before_promising, target):
            break
        selected.append(row)
        used.add(row["question_id"])

    min_promising = min(8, len(promising_pool))
    for row in promising_pool:
        if row["question_id"] in used:
            continue
        if sum(1 for item in selected if item.get("current_plus_history_gain") == "unclear_but_promising") >= min_promising:
            break
        if len(selected) >= target:
            break
        selected.append(row)
        used.add(row["question_id"])

    remaining = [
        row
        for row in gain_yes_pool + promising_pool
        if row["question_id"] not in used and (eval_eligible(row) or medium_eval_eligible(row))
    ]
    remaining.sort(key=eval_score, reverse=True)
    for row in remaining:
        if len(selected) >= target:
            break
        selected.append(row)
        used.add(row["question_id"])

    if len(selected) < 20:
        relaxed = [row for row in tier_a + tier_b if row["question_id"] not in used and medium_eval_eligible(row)]
        relaxed.sort(key=eval_score, reverse=True)
        for row in relaxed:
            if len(selected) >= min(target, 40):
                break
            selected.append(row)
            used.add(row["question_id"])
    return selected[:target]


def select_control(tier_c: list[dict[str, str]], target: int = 15) -> list[dict[str, str]]:
    eligible = sorted([row for row in tier_c if control_eligible(row)], key=control_score, reverse=True)
    return eligible[:target]


def load_evidence(question_ids: set[str]) -> dict[str, dict[str, list[dict[str, str]]]]:
    evidence: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    with TOPK_EVIDENCE_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("question_id", "")
            if qid not in question_ids:
                continue
            evidence[qid][row.get("evidence_scope", "")].append(row)
    for by_scope in evidence.values():
        for rows in by_scope.values():
            rows.sort(key=lambda row: parse_int(row.get("rank", "999")))
    return evidence


def format_evidence_rows(rows: list[dict[str, str]], limit_rows: int = 5, caption_limit: int = 620) -> str:
    parts = []
    for row in rows[:limit_rows]:
        parts.append(
            f"[{row.get('evidence_scope', '')} rank={row.get('rank', '')} score={row.get('score', '')} "
            f"source={row.get('source_agent', '')} D{row.get('day', '')} "
            f"{row.get('start_time', '')}-{row.get('end_time', '')} {row.get('granularity', '')}] "
            f"{trunc(row.get('caption_text', ''), caption_limit)}"
        )
    return "\n".join(parts)


def combined_context(current_context: str, history_context: str) -> str:
    return f"CURRENT EVIDENCE:\n{current_context}\n\nHISTORICAL EVIDENCE:\n{history_context}".strip()


def collect_sources_and_windows(rows: list[dict[str, str]]) -> tuple[str, str]:
    sources: list[str] = []
    windows: list[str] = []
    for row in rows:
        source = f"{row.get('evidence_scope', '')}:{row.get('source_agent', '')}:{row.get('granularity', '')}"
        window = (
            f"{row.get('source_agent', '')} D{row.get('day', '')} "
            f"{row.get('start_time', '')}-{row.get('end_time', '')} {row.get('granularity', '')}"
        )
        if source not in sources:
            sources.append(source)
        if window not in windows:
            windows.append(window)
    return "; ".join(sources), "; ".join(windows)


def confidence(row: dict[str, str], dataset_kind: str) -> str:
    if dataset_kind == "control":
        return "high"
    if dataset_kind == "demo":
        return "high"
    if row.get("current_plus_history_gain") == "yes" and row.get("current_plus_history_answerability") == "likely_answerable":
        return "medium"
    return "medium" if row.get("current_plus_history_gain") == "unclear_but_promising" else "low"


def case_type(row: dict[str, str], dataset_kind: str) -> str:
    if dataset_kind == "control":
        return "current_sufficient_control"
    if row.get("current_plus_history_gain") == "yes":
        return "current_plus_history_gain"
    return "promising_history_gain"


def expected_result(row: dict[str, str], dataset_kind: str) -> str:
    if dataset_kind == "control":
        return "current_only_sufficient"
    if row.get("current_plus_history_gain") == "yes":
        return "current_plus_history_better"
    return "current_plus_history_may_help_needs_human_check"


def why_selected(row: dict[str, str], dataset_kind: str) -> str:
    if dataset_kind == "control":
        return "Tier C current-only sufficient control; current-only is likely_answerable and current+historical has no automatic gain."
    if dataset_kind == "demo":
        return "Tier A case with readable current+historical evidence gain over weaker current-only evidence."
    return "Tier A/B case with automatic current+historical gain or promising history-gain signal; retained for lightweight comparison after human spot-check."


def potential_issue(row: dict[str, str], dataset_kind: str) -> str:
    if row.get("current_plus_history_gain") == "unclear_but_promising":
        return "History evidence is relevant but the automatic screen cannot verify necessity; human spot-check required."
    if dataset_kind == "control":
        return "Control case may still benefit from history semantically, but current-only appears sufficient in caption evidence."
    return "Auto-screened caption evidence may be lexically aligned without fully proving the answer; human spot-check required."


def dataset_row(
    row: dict[str, str],
    evidence: dict[str, dict[str, list[dict[str, str]]]],
    case_id: str,
    dataset_kind: str,
) -> dict[str, str]:
    qid = row["question_id"]
    by_scope = evidence.get(qid, {})
    current_rows = by_scope.get("current_only", [])
    history_rows = by_scope.get("history_only", [])
    plus_rows = by_scope.get("current_plus_historical", [])

    current_context = format_evidence_rows(current_rows, limit_rows=5) or row.get("best_current_evidence_summary", "")
    history_context = format_evidence_rows(history_rows, limit_rows=5) or row.get("best_history_evidence_summary", "")
    plus_context = combined_context(current_context, history_context)
    all_rows = current_rows[:5] + history_rows[:5] + plus_rows[:5]
    sources, windows = collect_sources_and_windows(all_rows)

    return {
        "case_id": case_id,
        "source_question_id": qid,
        "source_tier": row.get("source_tier", ""),
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "category": row.get("category", ""),
        "subcategory": row.get("subcategory", ""),
        "case_type": case_type(row, dataset_kind),
        "current_only_context": current_context,
        "history_only_context": history_context,
        "current_plus_historical_context": plus_context,
        "current_only_answerability": row.get("current_only_answerability", ""),
        "history_only_answerability": row.get("history_only_answerability", ""),
        "current_plus_history_answerability": row.get("current_plus_history_answerability", ""),
        "current_plus_history_gain": row.get("current_plus_history_gain", ""),
        "expected_comparison": "current_only_vs_current_plus_history",
        "expected_result": expected_result(row, dataset_kind),
        "evidence_sources": sources,
        "evidence_time_windows": windows,
        "confidence": confidence(row, dataset_kind),
        "why_selected": why_selected(row, dataset_kind),
        "potential_issue": potential_issue(row, dataset_kind),
        "label_status": "auto_screened_needs_human_check",
        "needs_human_check": "yes",
    }


def assign_rows(
    demo_source: list[dict[str, str]],
    eval_source: list[dict[str, str]],
    control_source: list[dict[str, str]],
    evidence: dict[str, dict[str, list[dict[str, str]]]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    demo_rows = [
        dataset_row(row, evidence, f"DV01_DEMO_{idx:03d}", "demo")
        for idx, row in enumerate(demo_source, start=1)
    ]
    demo_id_by_qid = {row["source_question_id"]: row["case_id"] for row in demo_rows}

    eval_rows = []
    next_eval_id = 1
    for row in eval_source:
        qid = row["question_id"]
        if qid in demo_id_by_qid:
            case_id = demo_id_by_qid[qid]
        else:
            case_id = f"DV01_EVAL_{next_eval_id:03d}"
            next_eval_id += 1
        eval_rows.append(dataset_row(row, evidence, case_id, "eval"))

    control_rows = [
        dataset_row(row, evidence, f"DV01_CONTROL_{idx:03d}", "control")
        for idx, row in enumerate(control_source, start=1)
    ]
    return demo_rows, eval_rows, control_rows


def scope_context(case: dict[str, str], scope: str) -> str:
    if scope == "current_only":
        return case["current_only_context"]
    if scope == "history_only":
        return case["history_only_context"]
    return case["current_plus_historical_context"]


def write_model_inputs(path: Path, cases: list[dict[str, str]]) -> int:
    seen_case_ids = set()
    unique_cases = []
    for case in cases:
        if case["case_id"] in seen_case_ids:
            continue
        seen_case_ids.add(case["case_id"])
        unique_cases.append(case)

    count = 0
    with path.open("w", encoding="utf-8") as f:
        for case in unique_cases:
            for scope in ["current_only", "history_only", "current_plus_historical"]:
                item = {
                    "case_id": case["case_id"],
                    "question_id": case["source_question_id"],
                    "evidence_scope": scope,
                    "question": case["question"],
                    "answer": case["answer"],
                    "category": case["category"],
                    "evidence_context": scope_context(case, scope),
                    "evidence_sources": case["evidence_sources"],
                    "expected_comparison": "current_only_vs_current_plus_history",
                    "label_status": "auto_screened_needs_human_check",
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
    return count


def write_readable(path: Path, demo_rows: list[dict[str, str]]) -> None:
    lines = [
        "# Dataset V0.1 Demo Cases",
        "",
        "These are caption-only, auto-screened candidate cases for evidence-scope comparison. They require human spot-check before use as labels.",
        "",
    ]
    for row in demo_rows:
        lines += [
            f"## {row['case_id']}",
            "",
            f"Question: {row['question']}",
            "",
            f"Answer: {row['answer']}",
            "",
            f"Why selected: {row['why_selected']}",
            "",
            f"Current-only evidence: {trunc(row['current_only_context'], 950)}",
            "",
            f"Historical evidence: {trunc(row['history_only_context'], 950)}",
            "",
            f"Current+historical evidence: {trunc(row['current_plus_historical_context'], 1200)}",
            "",
            f"Expected result: {row['expected_result']}",
            "",
            "Why this supports historical memory: current+historical evidence includes before-context captions that the automatic screen judged more answer-relevant than current-only evidence.",
            "",
            f"Potential issue: {row['potential_issue']}",
            "",
        ]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def source_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(Counter(row["source_tier"] for row in rows))


def write_report(
    path: Path,
    *,
    stats: dict[str, Any],
    demo_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    control_rows: list[dict[str, str]],
    model_input_count: int,
    recommended: list[dict[str, str]],
) -> None:
    tier_counts = stats["tier_counts"]
    gain_counts = stats["gain_counts"]
    lines = [
        "# Dataset V0.1 Report",
        "",
        "Dataset V0.1 contains auto-screened candidate cases for evidence-scope comparison. It repurposes MA-EgoQA/EgoLife caption evidence and is not a final benchmark.",
        "",
        "## Input Full Screening Counts",
        "",
        f"- Processed questions: {stats['processed_questions']}.",
        f"- Tier A: {tier_counts.get('tier_A_demo_ready', 0)}.",
        f"- Tier B: {tier_counts.get('tier_B_promising_history_gain', 0)}.",
        f"- Tier C: {tier_counts.get('tier_C_current_sufficient_control', 0)}.",
        f"- current+historical better than current-only: {gain_counts.get('yes', 0)} yes + {gain_counts.get('unclear_but_promising', 0)} unclear_but_promising.",
        "",
        "## Dataset Counts",
        "",
        f"- demo set count: {len(demo_rows)}.",
        f"- eval-lite set count: {len(eval_rows)}.",
        f"- control set count: {len(control_rows)}.",
        f"- model input JSONL rows: {model_input_count}.",
        "",
        "## Sources",
        "",
        f"- demo set sources: {source_counts(demo_rows)}.",
        f"- eval-lite set sources: {source_counts(eval_rows)}.",
        f"- control set sources: {source_counts(control_rows)}.",
        "",
        "## Recommended Human Spot-Check",
        "",
    ]
    for row in recommended[:10]:
        lines.append(
            f"- {row['case_id']} / Q{row['source_question_id']} / {row['case_type']} / "
            f"{row['confidence']} / {trunc(row['question'], 180)}"
        )

    lines += [
        "",
        "## Why This Only Shows Feasibility",
        "",
        "- The cases are selected by caption-level retrieval and heuristic answerability, not by final human annotation.",
        "- The original MA-EgoQA questions were not designed specifically for this self-first historical-memory idea.",
        "- Evidence gain here means candidate evidence coverage gain, not model correctness or causal proof.",
        "- Each case keeps `label_status=auto_screened_needs_human_check`.",
        "",
        "## Current Claims",
        "",
        "- Dataset V0.1 contains auto-screened candidate cases for evidence-scope comparison.",
        "- The dataset includes candidate cases where current+historical evidence may improve coverage over current-only evidence.",
        "",
        "## Claims Not Supported",
        "",
        "- Final labels.",
        "- Benchmark complete.",
        "- Method works.",
        "- Accuracy improved.",
        "- Historical memory proven useful.",
        "",
        "## Next Model Comparison Step",
        "",
        "1. Use `dataset_v0_1_model_inputs.jsonl` to create three evidence-scope prompts per case: current_only, history_only, and current_plus_historical.",
        "2. Compare model outputs for current_only vs current_plus_historical under the same prompt and decoding settings.",
        "3. Treat demo/eval-lite results as evidence-scope diagnostics only until manual spot-check confirms the cases.",
        "4. Use control cases to verify the system can stop at current evidence when current-only is already sufficient.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_outputs(output_dir: Path) -> dict[str, Any]:
    for path in [FULLSCREEN_PATH, TOPK_EVIDENCE_PATH, TIER_A_PATH, TIER_B_PATH, TIER_C_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    tier_a, tier_b, tier_c, stats = load_candidates()
    demo_source = select_demo(tier_a, target=10)
    eval_source = select_eval_lite(demo_source, tier_a, tier_b, target=35)
    control_source = select_control(tier_c, target=15)

    selected_qids = {row["question_id"] for row in demo_source + eval_source + control_source}
    evidence = load_evidence(selected_qids)
    demo_rows, eval_rows, control_rows = assign_rows(demo_source, eval_source, control_source, evidence)

    demo_path = output_dir / "dataset_v0_1_demo.csv"
    eval_path = output_dir / "dataset_v0_1_eval_lite.csv"
    control_path = output_dir / "dataset_v0_1_control.csv"
    model_path = output_dir / "dataset_v0_1_model_inputs.jsonl"
    readable_path = output_dir / "dataset_v0_1_readable.md"
    report_path = output_dir / "dataset_v0_1_report.md"

    write_csv(demo_path, demo_rows, DATASET_FIELDNAMES)
    write_csv(eval_path, eval_rows, DATASET_FIELDNAMES)
    write_csv(control_path, control_rows, DATASET_FIELDNAMES)
    model_input_count = write_model_inputs(model_path, eval_rows + control_rows)
    write_readable(readable_path, demo_rows)
    recommended = demo_rows + [row for row in eval_rows if row["case_id"] not in {d["case_id"] for d in demo_rows}]
    write_report(
        report_path,
        stats=stats,
        demo_rows=demo_rows,
        eval_rows=eval_rows,
        control_rows=control_rows,
        model_input_count=model_input_count,
        recommended=recommended,
    )

    return {
        "output_dir": output_dir,
        "demo_path": demo_path,
        "eval_path": eval_path,
        "control_path": control_path,
        "model_path": model_path,
        "readable_path": readable_path,
        "report_path": report_path,
        "demo_rows": demo_rows,
        "eval_rows": eval_rows,
        "control_rows": control_rows,
        "model_input_count": model_input_count,
        "recommended": recommended[:10],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_outputs(args.output_dir)
    print(f"Output directory: {result['output_dir']}")
    print(f"Demo set count: {len(result['demo_rows'])}")
    print(f"Eval-lite set count: {len(result['eval_rows'])}")
    print(f"Control set count: {len(result['control_rows'])}")
    print("Top 10 demo/eval cases to spot-check first:")
    for row in result["recommended"]:
        print(
            f"- {row['case_id']} / Q{row['source_question_id']} / {row['case_type']} / "
            f"{row['confidence']} / {trunc(row['question'], 140)}"
        )


if __name__ == "__main__":
    main()
