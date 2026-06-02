#!/usr/bin/env python3
"""
Build caption-only QA packets for current-only, history-only, and
current+history evidence-scope diagnostics.

This script does not download video, run VLM/LLM inference, call APIs, or edit
original MA-EgoQA files. It only reads existing historical_v1 CSV/HTML outputs
and writes derived packet files under outputs/historical_v1/evidence_scope_packets_v1/.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
HISTORICAL_DIR = ROOT / "outputs" / "historical_v1"
MANUAL_DIR = HISTORICAL_DIR / "manual_audit"
OUTPUT_DIR = HISTORICAL_DIR / "evidence_scope_packets_v1"

MANUAL_AUDIT_PATH = MANUAL_DIR / "manual_historical_case_audit_v1.csv"
PRIORITY_PATH = MANUAL_DIR / "audit_priority_subset.csv"
RETRIEVAL_PATH = HISTORICAL_DIR / "historical_retrieval_candidates.csv"
SELECTED_PATH = HISTORICAL_DIR / "selected_historical_candidate_questions.csv"
GALLERY_PATH = HISTORICAL_DIR / "historical_memory_candidate_gallery.html"

CURRENT_MODES = ["all_agents_current", "self_current", "external_current"]
HISTORY_MODES = ["all_agents_history", "self_history", "external_history"]
MODE_ORDER = {
    "all_agents_current": 0,
    "self_current": 1,
    "external_current": 2,
    "all_agents_history": 3,
    "self_history": 4,
    "external_history": 5,
}

PACKET_FIELDNAMES = [
    "qa_packet_id",
    "source_case_id",
    "question_id",
    "evidence_scope",
    "question",
    "answer",
    "category",
    "provisional_query_user",
    "query_user_status",
    "evidence_text",
    "evidence_sources",
    "evidence_time_windows",
    "evidence_modes",
    "expected_use",
    "auto_answerability_draft",
    "auto_answerability_reason",
    "needs_user_check",
]

COMPARISON_FIELDNAMES = [
    "case_id",
    "question_id",
    "question",
    "answer",
    "category",
    "provisional_query_user",
    "query_user_status",
    "current_only_answerability",
    "historical_only_answerability",
    "current_plus_historical_answerability",
    "current_to_current_plus_gain",
    "likely_case_bucket",
    "best_evidence_scope",
    "best_evidence_summary",
    "recommended_keep_for_next_stage",
    "reason",
]

CONSTRUCTED_FIELDNAMES = [
    "constructed_qa_id",
    "source_case_id",
    "question",
    "answer",
    "category",
    "query_user",
    "best_evidence_scope",
    "current_only_context",
    "historical_context",
    "current_plus_historical_context",
    "why_current_only_is_or_is_not_enough",
    "why_historical_memory_adds_value",
    "oracle_evidence_sources",
    "oracle_time_windows",
    "expected_experiment",
    "confidence",
    "needs_user_final_check",
]

STOPWORDS = {
    "about",
    "after",
    "again",
    "ahead",
    "also",
    "among",
    "around",
    "because",
    "before",
    "behind",
    "being",
    "between",
    "but",
    "could",
    "did",
    "does",
    "during",
    "each",
    "else",
    "everyone",
    "following",
    "from",
    "had",
    "have",
    "helped",
    "her",
    "him",
    "his",
    "how",
    "include",
    "included",
    "including",
    "into",
    "later",
    "like",
    "look",
    "made",
    "make",
    "most",
    "next",
    "only",
    "other",
    "others",
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
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "whose",
    "with",
    "would",
}

AGENT_NAMES = {
    "alice",
    "jake",
    "jack",
    "tasha",
    "lucia",
    "lucy",
    "katrina",
    "shure",
    "nicous",
    "violet",
    "choiszt",
    "luyue",
}

ANSWERABILITY_RANK = {
    "not_answerable": 0,
    "unclear": 1,
    "partially_answerable": 2,
    "likely_answerable": 3,
}


@dataclass(frozen=True)
class Answerability:
    label: str
    reason: str
    answer_coverage: float
    question_overlap: int
    answer_hits: list[str]


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


def trunc(text: Any, limit: int = 900) -> str:
    clean = normalize_ws(text)
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def tokenize(text: Any) -> set[str]:
    tokens = set()
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9']+", normalize_ws(text).lower()):
        token = token.strip("'")
        if len(token) < 3 or token in STOPWORDS:
            continue
        stemmed = stem_token(token)
        if stemmed in STOPWORDS:
            continue
        tokens.add(stemmed)
    return tokens


def stem_token(token: str) -> str:
    """Very small normalizer for caption-only heuristics, not linguistic labeling."""
    replacements = {
        "sodas": "soda",
        "mismatched": "mismatch",
        "mismatches": "mismatch",
        "mismatching": "mismatch",
        "deviates": "deviate",
        "deviated": "deviate",
        "deviating": "deviate",
        "pieces": "piece",
        "drinks": "drink",
        "beverages": "beverage",
        "cubes": "cube",
        "packs": "pack",
    }
    if token in replacements:
        return replacements[token]
    for suffix in ["ing", "ed", "es", "s"]:
        if len(token) > 5 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def case_sort_key(case_id: str) -> tuple[str, int]:
    match = re.search(r"(\d+)$", case_id)
    if match:
        return (case_id[: match.start()], int(match.group(1)))
    return (case_id, 0)


def rank_int(row: dict[str, str]) -> int:
    try:
        return int(row.get("rank", "999999"))
    except ValueError:
        return 999999


def score_float(row: dict[str, str]) -> float:
    try:
        return float(row.get("score", "0"))
    except ValueError:
        return 0.0


def parse_answer_option(question: str, answer: str) -> str:
    answer = normalize_ws(answer)
    if not re.fullmatch(r"[A-E](?:-[A-E])*", answer):
        return answer

    option_map = {}
    matches = list(re.finditer(r"\b([A-E])\)\s*", question))
    for idx, match in enumerate(matches):
        letter = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(question)
        option_map[letter] = normalize_ws(question[start:end].strip(" ,.;"))

    parts = [option_map.get(letter, "") for letter in answer.split("-")]
    expanded = " ".join(part for part in parts if part)
    return expanded or answer


def answer_keywords(question: str, answer: str) -> set[str]:
    expanded = parse_answer_option(question, answer)
    tokens = tokenize(expanded)
    raw = normalize_ws(answer).lower()
    if raw and len(raw) > 2 and not re.fullmatch(r"[a-e](?:-[a-e])*", raw):
        tokens |= tokenize(raw)
    return tokens


def question_keywords(question: str) -> set[str]:
    return tokenize(question)


def is_global_stat_or_temporal(question: str) -> bool:
    lower = question.lower()
    patterns = [
        r"\bwho\s+used\b.*\bthe\s+most\b",
        r"\bthe\s+most\b",
        r"\bwhich\s+of\s+the\s+following\s+happened\s+(first|last)\b",
        r"\bcorrect\s+sequence\s+of\s+events\b",
        r"\bhow\s+many\b",
        r"\bnumber\s+of\b",
    ]
    return any(re.search(pattern, lower) for pattern in patterns)


def has_explicit_person(question: str) -> bool:
    return bool(tokenize(question) & AGENT_NAMES)


def expected_use(scope: str) -> str:
    if scope == "current_only":
        return "diagnose whether current context captions alone cover the answer evidence"
    if scope == "history_only":
        return "diagnose whether before-context historical captions alone contain relevant memory"
    return "test whether combining current context with before-context historical memory improves evidence coverage"


def allowed_modes_for_scope(query_user: str, scope: str) -> list[str]:
    current = ["all_agents_current"]
    history = ["all_agents_history"]
    if query_user and query_user != "UNKNOWN":
        current += ["self_current", "external_current"]
        history += ["self_history", "external_history"]
    if scope == "current_only":
        return current
    if scope == "history_only":
        return history
    return current + history


def filter_rows_for_scope(rows: list[dict[str, str]], query_user: str, scope: str) -> list[dict[str, str]]:
    modes = set(allowed_modes_for_scope(query_user, scope))
    filtered = []
    for row in rows:
        mode = row.get("retrieval_mode", "")
        if mode not in modes:
            continue
        if scope == "current_only" and row.get("is_current") != "yes":
            continue
        if scope == "history_only" and row.get("is_history") != "yes":
            continue
        filtered.append(row)
    return sorted(filtered, key=lambda row: (MODE_ORDER.get(row.get("retrieval_mode", ""), 99), rank_int(row), -score_float(row)))


def format_evidence_line(row: dict[str, str]) -> str:
    return (
        f"[{row.get('retrieval_mode', '')} rank={row.get('rank', '')} score={row.get('score', '')} "
        f"source={row.get('source_agent', '')} D{row.get('day', '')} "
        f"{row.get('start_time', '')}-{row.get('end_time', '')}] "
        f"{normalize_ws(row.get('caption_text', ''))}"
    )


def build_evidence_fields(rows: list[dict[str, str]]) -> dict[str, str]:
    evidence_text = "\n".join(format_evidence_line(row) for row in rows)
    sources = []
    windows = []
    modes = []
    for row in rows:
        source = f"{row.get('retrieval_mode', '')}:{row.get('source_agent', '')}"
        window = f"{row.get('source_agent', '')} D{row.get('day', '')} {row.get('start_time', '')}-{row.get('end_time', '')}"
        mode = row.get("retrieval_mode", "")
        if source not in sources:
            sources.append(source)
        if window not in windows:
            windows.append(window)
        if mode not in modes:
            modes.append(mode)
    return {
        "evidence_text": evidence_text,
        "evidence_sources": "; ".join(sources),
        "evidence_time_windows": "; ".join(windows),
        "evidence_modes": "; ".join(modes),
    }


def assess_answerability(question: str, answer: str, evidence_text: str) -> Answerability:
    evidence_tokens = tokenize(evidence_text)
    q_tokens = question_keywords(question)
    a_tokens = answer_keywords(question, answer)
    q_overlap = len(q_tokens & evidence_tokens)
    answer_hits = sorted(a_tokens & evidence_tokens)
    coverage = len(answer_hits) / max(1, len(a_tokens))
    lower_evidence = normalize_ws(evidence_text).lower()
    lower_answer = normalize_ws(answer).lower()
    direct_answer_hit = bool(lower_answer and len(lower_answer) > 2 and lower_answer in lower_evidence)
    global_or_temporal = is_global_stat_or_temporal(question)

    if not evidence_text.strip():
        return Answerability("not_answerable", "no retrieved caption evidence for this scope", 0.0, 0, [])

    if global_or_temporal:
        if coverage >= 0.60 and q_overlap >= 3:
            return Answerability(
                "partially_answerable",
                "global/statistical/temporal question; captions mention answer-related terms but may not cover enough events to verify the full comparison/order",
                coverage,
                q_overlap,
                answer_hits,
            )
        if q_overlap >= 4 or coverage >= 0.25:
            return Answerability(
                "unclear",
                "global/statistical/temporal question with partial lexical overlap; caption coverage may be too local",
                coverage,
                q_overlap,
                answer_hits,
            )
        return Answerability(
            "not_answerable",
            "global/statistical/temporal question and this evidence scope does not expose enough answer-related coverage",
            coverage,
            q_overlap,
            answer_hits,
        )

    if direct_answer_hit and q_overlap >= 2:
        return Answerability(
            "likely_answerable",
            "evidence contains the answer phrase and several question-relevant terms",
            coverage,
            q_overlap,
            answer_hits,
        )
    if coverage >= 0.67 and q_overlap >= 2:
        return Answerability(
            "likely_answerable",
            "evidence covers most answer keywords and overlaps with the question context",
            coverage,
            q_overlap,
            answer_hits,
        )
    if coverage >= 0.50 and q_overlap >= 5:
        return Answerability(
            "likely_answerable",
            "evidence covers key answer terms and strongly overlaps with the question context",
            coverage,
            q_overlap,
            answer_hits,
        )
    if coverage >= 0.35 and q_overlap >= 2:
        return Answerability(
            "partially_answerable",
            "evidence covers some answer keywords but may not fully support the answer",
            coverage,
            q_overlap,
            answer_hits,
        )
    if coverage > 0 or q_overlap >= 4:
        return Answerability(
            "unclear",
            "evidence has lexical overlap but does not clearly support the answer",
            coverage,
            q_overlap,
            answer_hits,
        )
    return Answerability(
        "not_answerable",
        "evidence does not contain enough answer-relevant caption content",
        coverage,
        q_overlap,
        answer_hits,
    )


def packet_id(scope: str, index: int) -> str:
    prefix = {
        "current_only": "CUR",
        "history_only": "HIS",
        "current_plus_historical": "CURHIS",
    }[scope]
    return f"ESQP_{prefix}_{index:03d}"


def build_packet(case: dict[str, str], rows: list[dict[str, str]], scope: str, index: int) -> tuple[dict[str, Any], Answerability]:
    evidence_fields = build_evidence_fields(rows)
    answerability = assess_answerability(case["question"], case["answer"], evidence_fields["evidence_text"])
    reason = (
        f"{answerability.reason}; answer_hits={','.join(answerability.answer_hits) or 'none'}; "
        f"answer_coverage={answerability.answer_coverage:.2f}; question_overlap={answerability.question_overlap}"
    )
    packet = {
        "qa_packet_id": packet_id(scope, index),
        "source_case_id": case["case_id"],
        "question_id": case["question_id"],
        "evidence_scope": scope,
        "question": case["question"],
        "answer": case["answer"],
        "category": case["category"],
        "provisional_query_user": case["provisional_query_user"],
        "query_user_status": case["query_user_status"],
        **evidence_fields,
        "expected_use": expected_use(scope),
        "auto_answerability_draft": answerability.label,
        "auto_answerability_reason": reason,
        "needs_user_check": "yes",
    }
    return packet, answerability


def best_scope(packet_by_scope: dict[str, dict[str, Any]]) -> str:
    order = ["current_plus_historical", "current_only", "history_only"]
    return max(order, key=lambda scope: ANSWERABILITY_RANK[packet_by_scope[scope]["auto_answerability_draft"]])


def summarize_evidence(packet: dict[str, Any], limit: int = 650) -> str:
    text = normalize_ws(packet.get("evidence_text", ""))
    return trunc(text, limit)


def gain_label(current: Answerability, history: Answerability, plus: Answerability) -> str:
    current_rank = ANSWERABILITY_RANK[current.label]
    history_rank = ANSWERABILITY_RANK[history.label]
    plus_rank = ANSWERABILITY_RANK[plus.label]
    if current.label == "likely_answerable":
        return "no"
    if plus_rank > current_rank and plus_rank >= 2 and history_rank >= 1:
        return "yes"
    history_adds_answer_hits = set(plus.answer_hits) - set(current.answer_hits)
    if plus_rank >= 2 and history_rank >= 2 and history_adds_answer_hits:
        return "unclear-but-promising"
    if plus_rank == current_rank:
        return "no"
    return "unclear"


def likely_bucket(case: dict[str, str], current: Answerability, history: Answerability, plus: Answerability, gain: str) -> str:
    if is_global_stat_or_temporal(case["question"]) or (
        case["provisional_query_user"] == "UNKNOWN" and not has_explicit_person(case["question"])
    ):
        return "not_self_first_or_unclear"
    if current.label == "likely_answerable":
        if gain == "unclear-but-promising":
            return "history_adds_useful_context"
        return "current_sufficient"
    if gain == "yes" and plus.label in {"likely_answerable", "partially_answerable"}:
        return "current_plus_history_needed"
    if history.label in {"likely_answerable", "partially_answerable"} and plus.label in {"likely_answerable", "partially_answerable"}:
        return "history_adds_useful_context"
    if history.label in {"not_answerable", "unclear"}:
        return "history_irrelevant"
    return "not_self_first_or_unclear"


def keep_recommendation(bucket: str, gain: str, plus: Answerability) -> str:
    if bucket in {"current_plus_history_needed", "history_adds_useful_context"} and plus.label in {
        "likely_answerable",
        "partially_answerable",
    }:
        return "yes" if gain == "yes" else "unclear"
    if bucket == "current_sufficient" and plus.label == "likely_answerable":
        return "unclear"
    return "no"


def confidence_for_constructed(row: dict[str, str], current: Answerability, history: Answerability, plus: Answerability, gain: str) -> str:
    if plus.label == "likely_answerable" and gain == "yes" and current.label in {"not_answerable", "unclear"}:
        return "high"
    if plus.label in {"likely_answerable", "partially_answerable"} and gain in {"yes", "unclear-but-promising"}:
        return "medium"
    return "low"


def build_case_rows(manual_rows: list[dict[str, str]], selected_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected_by_case = {row["case_id"]: row for row in selected_rows}
    case_rows = []
    for manual in sorted(manual_rows, key=lambda row: case_sort_key(row["case_id"])):
        selected = selected_by_case.get(manual["case_id"], {})
        case_rows.append(
            {
                "case_id": manual["case_id"],
                "question_id": manual["question_id"],
                "question": manual["question"],
                "answer": manual["answer"],
                "category": manual["category"],
                "context_day": manual["context_day"],
                "context_start": manual["context_start"],
                "context_end": manual["context_end"],
                "agents_in_context": manual["agents_in_context"],
                "provisional_query_user": manual["provisional_query_user"]
                or selected.get("query_user", "UNKNOWN"),
                "query_user_status": manual["query_user_status"] or selected.get("query_user_status", ""),
            }
        )
    return case_rows


def reason_for_comparison(
    case: dict[str, str],
    current: Answerability,
    history: Answerability,
    plus: Answerability,
    gain: str,
    bucket: str,
) -> str:
    pieces = [
        f"current={current.label} ({current.reason})",
        f"history={history.label} ({history.reason})",
        f"current+history={plus.label} ({plus.reason})",
        f"gain={gain}",
        f"bucket={bucket}",
    ]
    if is_global_stat_or_temporal(case["question"]):
        pieces.append("global/statistical/temporal question requires conservative answerability")
    if case["provisional_query_user"] == "UNKNOWN":
        pieces.append("query_user is UNKNOWN, so self-first interpretation remains provisional")
    return " | ".join(pieces)


def constructed_reason_current(case: dict[str, str], current: Answerability, plus: Answerability) -> str:
    if current.label == "likely_answerable":
        return "current-only already has a likely answerable caption signal; use this case mainly to test whether history changes or stabilizes evidence coverage"
    if current.label == "partially_answerable":
        return "current-only has partial evidence but appears to miss some answer context"
    return "current-only evidence is weak or unclear under caption-only heuristics"


def constructed_reason_history(history: Answerability, gain: str) -> str:
    if gain == "yes":
        return "historical captions add answer-related terms or context that improves current+history answerability"
    if gain == "unclear-but-promising":
        return "historical captions add related context, but the draft cannot confirm that history is necessary"
    if history.label in {"likely_answerable", "partially_answerable"}:
        return "historical captions are relevant, but the improvement over current-only is unclear"
    return "historical captions do not clearly add answer-supporting content"


def build_outputs(output_dir: Path) -> dict[str, Any]:
    for path in [MANUAL_AUDIT_PATH, PRIORITY_PATH, RETRIEVAL_PATH, SELECTED_PATH, GALLERY_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    manual_rows = read_csv(MANUAL_AUDIT_PATH)
    selected_rows = read_csv(SELECTED_PATH)
    retrieval_rows = read_csv(RETRIEVAL_PATH)

    retrieval_by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in retrieval_rows:
        retrieval_by_case[row["case_id"]].append(row)

    case_rows = build_case_rows(manual_rows, selected_rows)
    packets_by_scope: dict[str, list[dict[str, Any]]] = {
        "current_only": [],
        "history_only": [],
        "current_plus_historical": [],
    }
    answerability_by_case: dict[str, dict[str, Answerability]] = defaultdict(dict)
    packet_by_case_scope: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for idx, case in enumerate(case_rows, start=1):
        for scope in ["current_only", "history_only", "current_plus_historical"]:
            rows = filter_rows_for_scope(
                retrieval_by_case.get(case["case_id"], []),
                case["provisional_query_user"],
                scope,
            )
            packet, answerability = build_packet(case, rows, scope, idx)
            packets_by_scope[scope].append(packet)
            answerability_by_case[case["case_id"]][scope] = answerability
            packet_by_case_scope[case["case_id"]][scope] = packet

    comparison_rows = []
    constructed_rows = []
    constructed_idx = 1
    for case in case_rows:
        cid = case["case_id"]
        current = answerability_by_case[cid]["current_only"]
        history = answerability_by_case[cid]["history_only"]
        plus = answerability_by_case[cid]["current_plus_historical"]
        gain = gain_label(current, history, plus)
        bucket = likely_bucket(case, current, history, plus, gain)
        packet_scope = best_scope(packet_by_case_scope[cid])
        best_packet = packet_by_case_scope[cid][packet_scope]
        keep = keep_recommendation(bucket, gain, plus)
        reason = reason_for_comparison(case, current, history, plus, gain, bucket)

        comparison_rows.append(
            {
                "case_id": cid,
                "question_id": case["question_id"],
                "question": case["question"],
                "answer": case["answer"],
                "category": case["category"],
                "provisional_query_user": case["provisional_query_user"],
                "query_user_status": case["query_user_status"],
                "current_only_answerability": current.label,
                "historical_only_answerability": history.label,
                "current_plus_historical_answerability": plus.label,
                "current_to_current_plus_gain": gain,
                "likely_case_bucket": bucket,
                "best_evidence_scope": packet_scope,
                "best_evidence_summary": summarize_evidence(best_packet),
                "recommended_keep_for_next_stage": keep,
                "reason": reason,
            }
        )

        is_constructable = (
            plus.label in {"likely_answerable", "partially_answerable"}
            and gain in {"yes", "unclear-but-promising"}
            and not is_global_stat_or_temporal(case["question"])
            and bucket != "not_self_first_or_unclear"
            and keep in {"yes", "unclear"}
        )
        if is_constructable:
            confidence = confidence_for_constructed(case, current, history, plus, gain)
            current_packet = packet_by_case_scope[cid]["current_only"]
            history_packet = packet_by_case_scope[cid]["history_only"]
            plus_packet = packet_by_case_scope[cid]["current_plus_historical"]
            constructed_rows.append(
                {
                    "constructed_qa_id": f"SCOPEQA_{constructed_idx:03d}",
                    "source_case_id": cid,
                    "question": case["question"],
                    "answer": case["answer"],
                    "category": case["category"],
                    "query_user": case["provisional_query_user"],
                    "best_evidence_scope": packet_scope,
                    "current_only_context": summarize_evidence(current_packet, 1200),
                    "historical_context": summarize_evidence(history_packet, 1200),
                    "current_plus_historical_context": summarize_evidence(plus_packet, 1600),
                    "why_current_only_is_or_is_not_enough": constructed_reason_current(case, current, plus),
                    "why_historical_memory_adds_value": constructed_reason_history(history, gain),
                    "oracle_evidence_sources": plus_packet["evidence_sources"],
                    "oracle_time_windows": plus_packet["evidence_time_windows"],
                    "expected_experiment": "compare_current_only_vs_current_plus_history",
                    "confidence": confidence,
                    "needs_user_final_check": "yes",
                }
            )
            constructed_idx += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    current_path = output_dir / "qa_current_only_v1.csv"
    history_path = output_dir / "qa_history_only_v1.csv"
    current_history_path = output_dir / "qa_current_history_v1.csv"
    comparison_path = output_dir / "qa_evidence_scope_comparison_v1.csv"
    constructed_path = output_dir / "constructed_scope_qa_cases_v1.csv"
    report_path = output_dir / "evidence_scope_qa_construction_report.md"

    write_csv(current_path, packets_by_scope["current_only"], PACKET_FIELDNAMES)
    write_csv(history_path, packets_by_scope["history_only"], PACKET_FIELDNAMES)
    write_csv(current_history_path, packets_by_scope["current_plus_historical"], PACKET_FIELDNAMES)
    write_csv(comparison_path, comparison_rows, COMPARISON_FIELDNAMES)
    write_csv(constructed_path, constructed_rows, CONSTRUCTED_FIELDNAMES)
    write_report(report_path, comparison_rows, constructed_rows)

    return {
        "output_dir": output_dir,
        "current_path": current_path,
        "history_path": history_path,
        "current_history_path": current_history_path,
        "comparison_path": comparison_path,
        "constructed_path": constructed_path,
        "report_path": report_path,
        "comparison_rows": comparison_rows,
        "constructed_rows": constructed_rows,
    }


def write_report(path: Path, comparison_rows: list[dict[str, Any]], constructed_rows: list[dict[str, Any]]) -> None:
    total = len(comparison_rows)
    current_counts = Counter(row["current_only_answerability"] for row in comparison_rows)
    history_counts = Counter(row["historical_only_answerability"] for row in comparison_rows)
    plus_counts = Counter(row["current_plus_historical_answerability"] for row in comparison_rows)
    gain_counts = Counter(row["current_to_current_plus_gain"] for row in comparison_rows)
    bucket_counts = Counter(row["likely_case_bucket"] for row in comparison_rows)
    keep_counts = Counter(row["recommended_keep_for_next_stage"] for row in comparison_rows)

    demo_rows = sorted(
        constructed_rows,
        key=lambda row: ({"high": 0, "medium": 1, "low": 2}.get(row["confidence"], 3), row["source_case_id"]),
    )[:8]
    reject_rows = [
        row
        for row in comparison_rows
        if row["recommended_keep_for_next_stage"] == "no"
        or row["likely_case_bucket"] == "not_self_first_or_unclear"
    ]

    lines = [
        "# Evidence Scope QA Construction Report v1",
        "",
        "This is a caption-only evidence-scope packet construction draft. It does not use video, VLM inference, LLM API calls, or final human labels.",
        "",
        "## Counts",
        "",
        f"- Total candidate cases reviewed: {total}",
        f"- Constructed high-confidence/medium-confidence QA subset size: {len(constructed_rows)}",
        f"- current-only likely_answerable: {current_counts.get('likely_answerable', 0)}",
        f"- history-only likely_answerable: {history_counts.get('likely_answerable', 0)}",
        f"- current+historical likely_answerable: {plus_counts.get('likely_answerable', 0)}",
        f"- Cases where current+historical appears better than current-only: {gain_counts.get('yes', 0)} yes; {gain_counts.get('unclear-but-promising', 0)} unclear-but-promising",
        "",
        "## Answerability Distributions",
        "",
        f"- current-only: {dict(current_counts)}",
        f"- history-only: {dict(history_counts)}",
        f"- current+historical: {dict(plus_counts)}",
        f"- gain labels: {dict(gain_counts)}",
        f"- likely buckets: {dict(bucket_counts)}",
        f"- keep recommendations: {dict(keep_counts)}",
        "",
        "## Best Demo Candidates",
        "",
    ]
    if demo_rows:
        for row in demo_rows:
            lines.append(
                f"- {row['source_case_id']} ({row['confidence']}): {row['question']} "
                f"| best_scope={row['best_evidence_scope']}"
            )
    else:
        lines.append("- No case met the constructed subset criteria under conservative caption-only heuristics.")

    lines += [
        "",
        "## Likely Rejects",
        "",
    ]
    if reject_rows:
        for row in reject_rows[:12]:
            lines.append(
                f"- {row['case_id']}: bucket={row['likely_case_bucket']}; keep={row['recommended_keep_for_next_stage']}; reason={trunc(row['reason'], 260)}"
            )
    else:
        lines.append("- No clear rejects under the current conservative heuristic.")

    lines += [
        "",
        "## What This Draft Suggests",
        "",
        "- We built a caption-only evidence-scope QA packet set.",
        "- The draft suggests whether adding historical memory may improve evidence coverage for selected cases.",
        "- The current result is useful for deciding which cases deserve manual packet inspection and later model-input construction.",
        "",
        "## What This Draft Cannot Claim",
        "",
        "- It does not show that the benchmark is complete.",
        "- It does not provide final labels.",
        "- It does not show model accuracy improved.",
        "- It does not prove historical memory is useful.",
        "- It does not solve self-first routing.",
        "- It does not claim answer accuracy for any MA-EgoQA item.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_outputs(args.output_dir)
    constructed_rows = result["constructed_rows"]
    recommended = sorted(
        constructed_rows,
        key=lambda row: ({"high": 0, "medium": 1, "low": 2}.get(row["confidence"], 3), row["source_case_id"]),
    )[:5]

    print(f"Output directory: {result['output_dir']}")
    print(f"QA current-only: {result['current_path']}")
    print(f"QA history-only: {result['history_path']}")
    print(f"QA current+history: {result['current_history_path']}")
    print(f"Comparison: {result['comparison_path']}")
    print(f"Constructed subset: {result['constructed_path']}")
    print(f"High-confidence/medium-confidence retained cases: {len(constructed_rows)}")
    print("Recommended cases to inspect first:")
    for row in recommended:
        print(f"- {row['source_case_id']} / confidence={row['confidence']} / best_scope={row['best_evidence_scope']} / {row['question']}")


if __name__ == "__main__":
    main()
