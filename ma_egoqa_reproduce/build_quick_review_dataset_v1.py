#!/usr/bin/env python3
"""
Build quick-review datasets from MA-EgoQA historical V2 full screening.

This script only reads caption-level V2 screening outputs and writes derived
CSV/Markdown files for human spot-check. It does not download video, run VLMs,
run LLM/API calls, modify original MA-EgoQA files, claim answer accuracy, or
treat automatic decisions as final human labels.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
V2_DIR = ROOT / "outputs" / "historical_v2_fullscreen"
OUTPUT_DIR = V2_DIR / "quick_review_dataset_v1"

FULLSCREEN_PATH = V2_DIR / "evidence_scope_fullscreen_v2.csv"
TOPK_EVIDENCE_PATH = V2_DIR / "evidence_scope_fullscreen_v2_topk_evidence.csv"
TIER_A_PATH = V2_DIR / "tier_A_demo_ready_cases_v2.csv"
TIER_B_PATH = V2_DIR / "tier_B_promising_history_gain_cases_v2.csv"
TIER_C_PATH = V2_DIR / "tier_C_current_sufficient_controls_v2.csv"

SIDE_BY_SIDE_FIELDNAMES = [
    "review_id",
    "source_tier",
    "question_id",
    "category",
    "question",
    "answer",
    "current_only_answerability",
    "history_only_answerability",
    "current_plus_history_answerability",
    "current_plus_history_gain",
    "best_current_evidence",
    "best_history_evidence",
    "best_current_plus_history_evidence",
    "evidence_sources",
    "evidence_time_windows",
    "auto_keep_decision",
    "auto_decision_reason",
    "needs_user_spot_check",
]

DATASET_FIELDNAMES = [
    "dataset_case_id",
    "source_question_id",
    "question",
    "answer",
    "category",
    "case_type",
    "current_only_context",
    "historical_context",
    "current_plus_historical_context",
    "expected_comparison",
    "expected_result",
    "evidence_sources",
    "evidence_time_windows",
    "confidence",
    "needs_user_final_check",
]

ANSWERABILITY_RANK = {
    "not_answerable": 0,
    "unclear": 1,
    "partially_answerable": 2,
    "likely_answerable": 3,
}

DECISION_ORDER = {
    "keep_demo": 0,
    "keep_eval": 1,
    "keep_control": 2,
    "reject": 3,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_csv_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: safe_csv_value(row.get(field, "")) for field in fieldnames})


def normalize_ws(text: Any) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())


def trunc(text: Any, limit: int = 700) -> str:
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


def answer_rank(row: dict[str, str], field: str) -> int:
    return ANSWERABILITY_RANK.get(row.get(field, ""), 0)


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
    return len(history) >= 120 and len(plus) >= 180 and len(current) >= 80


def evidence_not_just_lexical(row: dict[str, str]) -> bool:
    plus_hits = split_hits(row.get("current_plus_history_answer_hits", ""))
    plus_coverage = parse_float(row.get("current_plus_history_answer_coverage", "0"))
    plus_overlap = parse_int(row.get("current_plus_history_question_overlap", "0"))
    history_score = parse_float(row.get("history_only_top_score", "0"))
    return len(plus_hits) >= 2 and plus_coverage >= 0.35 and plus_overlap >= 3 and history_score > 0


def demo_score(row: dict[str, str]) -> tuple[float, int, int, int]:
    plus_cov = parse_float(row.get("current_plus_history_answer_coverage", "0"))
    hist_cov = parse_float(row.get("history_only_answer_coverage", "0"))
    current_cov = parse_float(row.get("current_only_answer_coverage", "0"))
    plus_overlap = parse_int(row.get("current_plus_history_question_overlap", "0"))
    hist_score = parse_float(row.get("history_only_top_score", "0"))
    gain_bonus = 2.0 if row.get("current_plus_history_gain") == "yes" else 0.8
    weakness_bonus = 1.0 if row.get("current_only_answerability") in {"not_answerable", "unclear"} else 0.4
    score = gain_bonus + weakness_bonus + plus_cov + 0.5 * hist_cov - 0.35 * current_cov + min(hist_score / 20.0, 1.5)
    return (score, plus_overlap, len(split_hits(row.get("current_plus_history_answer_hits", ""))), -parse_int(row["question_id"]))


def eval_score(row: dict[str, str]) -> tuple[int, float, int, int]:
    gain_rank = {"yes": 2, "unclear_but_promising": 1, "unclear": 0, "no": -1}.get(
        row.get("current_plus_history_gain", ""), 0
    )
    score = parse_float(row.get("current_plus_history_top_score", "0"))
    overlap = parse_int(row.get("current_plus_history_question_overlap", "0"))
    return (gain_rank, score, overlap, -parse_int(row["question_id"]))


def control_score(row: dict[str, str]) -> tuple[float, int, int]:
    current_score = parse_float(row.get("current_only_top_score", "0"))
    current_cov = parse_float(row.get("current_only_answer_coverage", "0"))
    current_overlap = parse_int(row.get("current_only_question_overlap", "0"))
    return (current_cov + min(current_score / 20.0, 1.5), current_overlap, -parse_int(row["question_id"]))


def format_evidence_rows(rows: list[dict[str, str]], limit_rows: int, caption_limit: int) -> str:
    parts = []
    for row in sorted(rows, key=lambda item: parse_int(item.get("rank", "999")))[:limit_rows]:
        parts.append(
            f"[{row.get('evidence_scope', '')} rank={row.get('rank', '')} score={row.get('score', '')} "
            f"source={row.get('source_agent', '')} D{row.get('day', '')} "
            f"{row.get('start_time', '')}-{row.get('end_time', '')} {row.get('granularity', '')}] "
            f"{trunc(row.get('caption_text', ''), caption_limit)}"
        )
    return " || ".join(parts)


def collect_sources_and_windows(rows: list[dict[str, str]]) -> tuple[str, str]:
    sources = []
    windows = []
    for row in sorted(rows, key=lambda item: (item.get("evidence_scope", ""), parse_int(item.get("rank", "999")))):
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


def load_review_seed_rows() -> tuple[list[dict[str, str]], dict[str, int]]:
    tier_a = read_csv(TIER_A_PATH)
    tier_b = read_csv(TIER_B_PATH)
    tier_c = read_csv(TIER_C_PATH)
    rows = []
    seen = set()
    for source_tier, selected in [
        ("tier_A_demo_ready", tier_a),
        ("tier_B_promising_history_gain", tier_b[:30]),
        ("tier_C_current_sufficient_control", tier_c[:20]),
    ]:
        for row in selected:
            qid = row["question_id"]
            if qid in seen:
                continue
            seen.add(qid)
            enriched = dict(row)
            enriched["source_tier"] = source_tier
            rows.append(enriched)
    return rows, {
        "tier_A_input_count": len(tier_a),
        "tier_B_input_count": len(tier_b),
        "tier_C_input_count": len(tier_c),
    }


def load_selected_evidence(question_ids: set[str]) -> dict[str, dict[str, list[dict[str, str]]]]:
    evidence: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    with TOPK_EVIDENCE_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("question_id", "")
            if qid not in question_ids:
                continue
            evidence[qid][row.get("evidence_scope", "")].append(row)
    return evidence


def auto_reason(row: dict[str, str], decision: str) -> str:
    global_flag = is_exhaustive_global_question(row["question"]) or row.get("is_pure_global_statistical") == "yes"
    pieces = [
        f"source={row['source_tier']}",
        f"current={row['current_only_answerability']}",
        f"history={row['history_only_answerability']}",
        f"current+history={row['current_plus_history_answerability']}",
        f"gain={row['current_plus_history_gain']}",
    ]
    if decision == "keep_demo":
        pieces.append("clear current+historical gain with readable non-global caption evidence")
    elif decision == "keep_eval":
        pieces.append("candidate or promising history gain, but still requires human spot-check")
    elif decision == "keep_control":
        pieces.append("current-only already appears sufficient and is useful as a control")
    else:
        if global_flag:
            pieces.append("rejected because question needs exhaustive global/time/order coverage")
        elif not evidence_not_just_lexical(row):
            pieces.append("rejected because evidence appears weak or mostly lexical")
        elif not evidence_readable(row):
            pieces.append("rejected because evidence summary is not readable enough")
        else:
            pieces.append("rejected by conservative quick-review rule")
    pieces.append("automatic caption-only quick review; not a final label or answer-accuracy claim")
    return " | ".join(pieces)


def assign_decisions(seed_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    demo_eligible = []
    for row in seed_rows:
        if row["source_tier"] != "tier_A_demo_ready":
            continue
        if is_exhaustive_global_question(row["question"]) or row.get("is_pure_global_statistical") == "yes":
            continue
        plus_ok = row.get("current_plus_history_answerability") == "likely_answerable" or is_strong_partial(row)
        if plus_ok and current_is_weak(row) and history_relevant(row) and evidence_readable(row) and evidence_not_just_lexical(row):
            demo_eligible.append(row)

    demo_eligible = sorted(demo_eligible, key=demo_score, reverse=True)
    demo_count = min(10, max(5, len(demo_eligible))) if demo_eligible else 0
    demo_ids = {row["question_id"] for row in demo_eligible[:demo_count]}

    decided = []
    for row in seed_rows:
        decision = "reject"
        if row["question_id"] in demo_ids:
            decision = "keep_demo"
        elif row["source_tier"] == "tier_C_current_sufficient_control":
            if (
                row.get("current_only_answerability") == "likely_answerable"
                and row.get("current_plus_history_gain") == "no"
                and not is_exhaustive_global_question(row["question"])
                and evidence_readable(row)
            ):
                decision = "keep_control"
        elif row["source_tier"] in {"tier_A_demo_ready", "tier_B_promising_history_gain"}:
            if (
                row.get("current_plus_history_gain") in {"yes", "unclear_but_promising"}
                and history_relevant(row)
                and evidence_not_just_lexical(row)
                and evidence_readable(row)
                and not is_exhaustive_global_question(row["question"])
                and row.get("is_pure_global_statistical") != "yes"
            ):
                decision = "keep_eval"
        enriched = dict(row)
        enriched["auto_keep_decision"] = decision
        enriched["auto_decision_reason"] = auto_reason(row, decision)
        decided.append(enriched)
    return decided


def build_review_rows(decided_rows: list[dict[str, str]], evidence: dict[str, dict[str, list[dict[str, str]]]]) -> list[dict[str, str]]:
    review_rows = []
    for idx, row in enumerate(decided_rows, start=1):
        qid = row["question_id"]
        by_scope = evidence.get(qid, {})
        current_rows = by_scope.get("current_only", [])
        history_rows = by_scope.get("history_only", [])
        plus_rows = by_scope.get("current_plus_historical", [])
        sources, windows = collect_sources_and_windows(current_rows[:2] + history_rows[:3] + plus_rows[:4])
        review_rows.append(
            {
                "review_id": f"QRV1_{idx:03d}",
                "source_tier": row["source_tier"],
                "question_id": qid,
                "category": row.get("category", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "current_only_answerability": row.get("current_only_answerability", ""),
                "history_only_answerability": row.get("history_only_answerability", ""),
                "current_plus_history_answerability": row.get("current_plus_history_answerability", ""),
                "current_plus_history_gain": row.get("current_plus_history_gain", ""),
                "best_current_evidence": format_evidence_rows(current_rows, 3, 520)
                or row.get("best_current_evidence_summary", ""),
                "best_history_evidence": format_evidence_rows(history_rows, 3, 520)
                or row.get("best_history_evidence_summary", ""),
                "best_current_plus_history_evidence": format_evidence_rows(plus_rows, 4, 450)
                or row.get("best_current_plus_history_summary", ""),
                "evidence_sources": sources,
                "evidence_time_windows": windows,
                "auto_keep_decision": row["auto_keep_decision"],
                "auto_decision_reason": row["auto_decision_reason"],
                "needs_user_spot_check": "yes",
            }
        )
    return review_rows


def confidence(row: dict[str, str]) -> str:
    if row["auto_keep_decision"] == "keep_demo":
        return "high"
    if row["auto_keep_decision"] == "keep_control":
        return "high" if row.get("current_only_answerability") == "likely_answerable" else "medium"
    if row.get("current_plus_history_gain") == "yes" and row.get("current_plus_history_answerability") == "likely_answerable":
        return "medium"
    return "low"


def case_type(row: dict[str, str]) -> str:
    if row["auto_keep_decision"] == "keep_control":
        return "current_sufficient_control"
    if row.get("current_plus_history_gain") == "yes":
        return "current_plus_history_gain"
    if row.get("current_plus_history_gain") == "unclear_but_promising":
        return "promising_history_gain"
    return "promising_history_gain"


def expected_result(row: dict[str, str]) -> str:
    ctype = case_type(row)
    if ctype == "current_sufficient_control":
        return "current_only_sufficient"
    if ctype == "current_plus_history_gain":
        return "current_plus_history_better"
    return "current_plus_history_promising_needs_spot_check"


def dataset_row(row: dict[str, str], dataset_case_id: str) -> dict[str, str]:
    return {
        "dataset_case_id": dataset_case_id,
        "source_question_id": row["question_id"],
        "question": row["question"],
        "answer": row["answer"],
        "category": row["category"],
        "case_type": case_type(row),
        "current_only_context": row["best_current_evidence"],
        "historical_context": row["best_history_evidence"],
        "current_plus_historical_context": row["best_current_plus_history_evidence"],
        "expected_comparison": "current_only_vs_current_plus_history",
        "expected_result": expected_result(row),
        "evidence_sources": row["evidence_sources"],
        "evidence_time_windows": row["evidence_time_windows"],
        "confidence": confidence(row),
        "needs_user_final_check": "yes",
    }


def select_datasets(review_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    keep_demo = [row for row in review_rows if row["auto_keep_decision"] == "keep_demo"]
    keep_eval = [row for row in review_rows if row["auto_keep_decision"] == "keep_eval"]
    keep_control = [row for row in review_rows if row["auto_keep_decision"] == "keep_control"]

    keep_demo.sort(key=lambda row: (confidence(row) != "high", -parse_float(row.get("current_plus_history_answerability", "0")), parse_int(row["question_id"])))
    gain_eval = keep_demo + sorted(keep_eval, key=lambda row: eval_score(row), reverse=True)
    control_sorted = sorted(keep_control, key=lambda row: control_score(row), reverse=True)

    demo_rows = keep_demo[:10]
    eval_seed = gain_eval[:30] + control_sorted[:5]
    eval_rows = eval_seed[:40]
    control_rows = control_sorted[:15]

    demo_dataset = [dataset_row(row, f"DEMO_V1_{idx:03d}") for idx, row in enumerate(demo_rows, start=1)]
    eval_dataset = [dataset_row(row, f"EVAL_LITE_V1_{idx:03d}") for idx, row in enumerate(eval_rows, start=1)]
    control_dataset = [dataset_row(row, f"CONTROL_V1_{idx:03d}") for idx, row in enumerate(control_rows, start=1)]
    return demo_dataset, eval_dataset, control_dataset


def markdown_case(row: dict[str, str], include_decision: bool = True) -> list[str]:
    lines = [
        f"### Q{row['source_question_id'] if 'source_question_id' in row else row['question_id']}",
        "",
        f"- Question: {row['question']}",
        f"- Answer: {row['answer']}",
    ]
    if "case_type" in row:
        lines.append(f"- Case type: {row['case_type']}")
        lines.append(f"- Expected result: {row['expected_result']}")
        lines.append(f"- Current-only evidence: {trunc(row['current_only_context'], 520)}")
        lines.append(f"- Historical evidence: {trunc(row['historical_context'], 520)}")
        lines.append(f"- Why keep / reject: confidence={row['confidence']}; needs_user_final_check=yes")
    else:
        lines.append(f"- Current-only evidence: {trunc(row['best_current_evidence'], 520)}")
        lines.append(f"- Historical evidence: {trunc(row['best_history_evidence'], 520)}")
        if include_decision:
            lines.append(f"- Why keep / reject: {row['auto_decision_reason']}")
        lines.append(
            f"- Expected result: {'current_only_sufficient' if row['auto_keep_decision'] == 'keep_control' else 'current_plus_history_better_or_promising'}"
        )
    lines.append("")
    return lines


def write_brief(
    path: Path,
    demo_dataset: list[dict[str, str]],
    eval_dataset: list[dict[str, str]],
    control_dataset: list[dict[str, str]],
    rejected_rows: list[dict[str, str]],
) -> None:
    lines = [
        "# Quick Review Brief V1",
        "",
        "Caption-only quick-review artifacts for human spot-check. Automatic decisions are not final labels and do not claim answer accuracy.",
        "",
        "## Demo Set",
        "",
    ]
    for row in demo_dataset:
        lines.extend(markdown_case(row, include_decision=False))

    lines += ["## Eval Lite Set: First 20", ""]
    for row in eval_dataset[:20]:
        lines.extend(markdown_case(row, include_decision=False))

    lines += ["## Control Set: First 10", ""]
    for row in control_dataset[:10]:
        lines.extend(markdown_case(row, include_decision=False))

    lines += ["## Rejected Examples: First 10", ""]
    for row in rejected_rows[:10]:
        lines.extend(markdown_case(row))

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_report(
    path: Path,
    *,
    input_counts: dict[str, int],
    review_rows: list[dict[str, str]],
    demo_dataset: list[dict[str, str]],
    eval_dataset: list[dict[str, str]],
    control_dataset: list[dict[str, str]],
    recommended: list[dict[str, str]],
) -> None:
    decision_counts = Counter(row["auto_keep_decision"] for row in review_rows)
    lines = [
        "# Quick Review Dataset V1 Report",
        "",
        "This report summarizes a caption-only quick-review dataset built from V2 full screening. No video, VLM, LLM, or API calls were used.",
        "",
        "## Counts",
        "",
        f"1. Tier A input count: {input_counts['tier_A_input_count']}.",
        f"1. Tier B input count: {input_counts['tier_B_input_count']} (top 30 used for side-by-side review).",
        f"1. Tier C input count: {input_counts['tier_C_input_count']} (top 20 used for side-by-side review).",
        f"1. Quick review decisions: {dict(decision_counts)}.",
        f"1. demo_set_v1 count: {len(demo_dataset)}.",
        f"1. eval_lite_set_v1 count: {len(eval_dataset)}.",
        f"1. control_set_v1 count: {len(control_dataset)}.",
        "",
        "## Recommended Spot-Check Cases",
        "",
    ]
    for row in recommended[:10]:
        lines.append(
            f"- Q{row['question_id']} [{row['auto_keep_decision']}; {row['source_tier']}; "
            f"gain={row['current_plus_history_gain']}] {trunc(row['question'], 180)}"
        )

    lines += [
        "",
        "## Current Claims",
        "",
        "- We constructed a caption-only quick-review dataset for evidence-scope comparison.",
        "- The dataset includes candidate cases where current+historical evidence may improve coverage over current-only.",
        "",
        "## Claims Not Supported",
        "",
        "- The final benchmark is complete.",
        "- Labels are final.",
        "- Model accuracy improved.",
        "- Historical memory is proven useful.",
        "- Self-first routing is solved.",
        "",
        "## Next Step For Comparison",
        "",
        "1. For each dataset case, prepare two model input packets with identical question/options/answer metadata: current-only context and current+historical context.",
        "2. Keep historical context limited to captions before context_start to avoid future leakage.",
        "3. Run the same evaluation protocol on both packets and compare output consistency or answer selection.",
        "4. Treat these cases as candidate evidence-scope diagnostics until manual spot-check finalizes labels.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_outputs(output_dir: Path) -> dict[str, Any]:
    for path in [FULLSCREEN_PATH, TOPK_EVIDENCE_PATH, TIER_A_PATH, TIER_B_PATH, TIER_C_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows, input_counts = load_review_seed_rows()
    question_ids = {row["question_id"] for row in seed_rows}
    evidence = load_selected_evidence(question_ids)
    decided_rows = assign_decisions(seed_rows)
    review_rows = build_review_rows(decided_rows, evidence)

    review_rows.sort(
        key=lambda row: (
            DECISION_ORDER.get(row["auto_keep_decision"], 99),
            {"tier_A_demo_ready": 0, "tier_B_promising_history_gain": 1, "tier_C_current_sufficient_control": 2}.get(
                row["source_tier"], 9
            ),
            parse_int(row["question_id"]),
        )
    )
    for idx, row in enumerate(review_rows, start=1):
        row["review_id"] = f"QRV1_{idx:03d}"

    demo_dataset, eval_dataset, control_dataset = select_datasets(review_rows)
    rejected_rows = [row for row in review_rows if row["auto_keep_decision"] == "reject"]
    recommended = [
        row
        for row in review_rows
        if row["auto_keep_decision"] in {"keep_demo", "keep_eval"}
    ][:10]

    side_path = output_dir / "quick_review_side_by_side_v1.csv"
    demo_path = output_dir / "demo_set_v1.csv"
    eval_path = output_dir / "eval_lite_set_v1.csv"
    control_path = output_dir / "control_set_v1.csv"
    brief_path = output_dir / "quick_review_brief_v1.md"
    report_path = output_dir / "quick_review_dataset_v1_report.md"

    write_csv(side_path, review_rows, SIDE_BY_SIDE_FIELDNAMES)
    write_csv(demo_path, demo_dataset, DATASET_FIELDNAMES)
    write_csv(eval_path, eval_dataset, DATASET_FIELDNAMES)
    write_csv(control_path, control_dataset, DATASET_FIELDNAMES)
    write_brief(brief_path, demo_dataset, eval_dataset, control_dataset, rejected_rows)
    write_report(
        report_path,
        input_counts=input_counts,
        review_rows=review_rows,
        demo_dataset=demo_dataset,
        eval_dataset=eval_dataset,
        control_dataset=control_dataset,
        recommended=recommended,
    )

    return {
        "output_dir": output_dir,
        "side_path": side_path,
        "demo_path": demo_path,
        "eval_path": eval_path,
        "control_path": control_path,
        "brief_path": brief_path,
        "report_path": report_path,
        "review_rows": review_rows,
        "demo_dataset": demo_dataset,
        "eval_dataset": eval_dataset,
        "control_dataset": control_dataset,
        "reject_count": len(rejected_rows),
        "recommended": recommended,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_outputs(args.output_dir)
    print(f"Output directory: {result['output_dir']}")
    print(f"demo_set_v1.csv: {result['demo_path']} rows={len(result['demo_dataset'])}")
    print(f"eval_lite_set_v1.csv: {result['eval_path']} rows={len(result['eval_dataset'])}")
    print(f"control_set_v1.csv: {result['control_path']} rows={len(result['control_dataset'])}")
    print(f"Reject count: {result['reject_count']}")
    print("Top 10 cases to inspect first:")
    for row in result["recommended"][:10]:
        print(
            f"- Q{row['question_id']} / {row['auto_keep_decision']} / {row['source_tier']} / "
            f"gain={row['current_plus_history_gain']} / {trunc(row['question'], 140)}"
        )


if __name__ == "__main__":
    main()
