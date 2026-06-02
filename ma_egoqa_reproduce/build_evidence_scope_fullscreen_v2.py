#!/usr/bin/env python3
"""
Evidence-scope full screening V2 for MA-EgoQA.

This script performs caption-only candidate screening over all parseable
MA-EgoQA questions. It does not download video, run VLM/LLM inference, call
APIs, edit the original MA-EgoQA files, or produce final benchmark labels.

The goal is to identify current-only vs current+historical evidence coverage
candidate cases for later human inspection. Historical evidence is strictly
limited to captions before the parsed context_start.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
HISTORICAL_V1_DIR = ROOT / "outputs" / "historical_v1"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "historical_v2_fullscreen"

QA_PATH_CANDIDATES = [
    ROOT / "data" / "MA-EgoQA.json",
    ROOT / "MA-EgoQA" / "data" / "MA-EgoQA.json",
]
CAPTION_30SEC_CANDIDATES = [
    ROOT / "data" / "caption" / "30sec",
    ROOT / "MA-EgoQA" / "data" / "caption" / "30sec",
]
CAPTION_10MIN_CANDIDATES = [
    ROOT / "data" / "caption" / "10min",
    ROOT / "MA-EgoQA" / "data" / "caption" / "10min",
]

CONTEXT_PARSE_PATH = HISTORICAL_V1_DIR / "maegoqa_context_parse.csv"
MEMORY_INDEX_PATH = HISTORICAL_V1_DIR / "source_isolated_caption_memory_index.csv"

TOP_K = 10
CURRENT_MARGIN_SECONDS = 120.0

AGENT_ORDER = ["Jake", "Alice", "Tasha", "Lucia", "Katrina", "Shure"]
AGENT_LOWER_TO_CANONICAL = {name.lower(): name for name in AGENT_ORDER}

ANSWERABILITY_RANK = {
    "not_answerable": 0,
    "unclear": 1,
    "partially_answerable": 2,
    "likely_answerable": 3,
}

TIER_ORDER = {
    "tier_A_demo_ready": 0,
    "tier_B_promising_history_gain": 1,
    "tier_C_current_sufficient_control": 2,
    "tier_D_reject_unclear": 3,
    "tier_E_not_suitable": 4,
}

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

RETRIEVAL_STOPWORDS = STOPWORDS | {
    "are",
    "can",
    "for",
    "has",
    "its",
    "not",
    "one",
    "out",
    "she",
    "who",
    "why",
}

AGENT_NAMES = {name.lower() for name in AGENT_ORDER} | {
    "jack",
    "lucy",
    "nicous",
    "violet",
    "choiszt",
    "luyue",
}

EXPLICIT_QUERY_PATTERNS = [
    re.compile(r"\bfrom\s+(Jake|Alice|Tasha|Lucia|Katrina|Shure)'s\s+(?:view|perspective|camera|video)\b", re.I),
    re.compile(r"\baccording\s+to\s+(Jake|Alice|Tasha|Lucia|Katrina|Shure)\b", re.I),
    re.compile(r"\bin\s+(Jake|Alice|Tasha|Lucia|Katrina|Shure)'s\s+(?:view|perspective|camera|video)\b", re.I),
]

FULLSCREEN_FIELDNAMES = [
    "question_id",
    "category",
    "subcategory",
    "question",
    "answer",
    "options_json",
    "contexts_json",
    "parse_status",
    "context_day",
    "context_start",
    "context_end",
    "context_start_hms",
    "context_end_hms",
    "agents_in_context",
    "parsed_contexts_json",
    "provisional_query_user",
    "query_user_status",
    "current_only_answerability",
    "history_only_answerability",
    "current_plus_history_answerability",
    "current_plus_history_gain",
    "evidence_scope_bucket",
    "best_evidence_scope",
    "best_current_evidence_summary",
    "best_history_evidence_summary",
    "best_current_plus_history_summary",
    "reason",
    "needs_user_check",
    "current_only_top_score",
    "history_only_top_score",
    "current_plus_history_top_score",
    "current_only_answer_coverage",
    "history_only_answer_coverage",
    "current_plus_history_answer_coverage",
    "current_only_question_overlap",
    "history_only_question_overlap",
    "current_plus_history_question_overlap",
    "current_only_answer_hits",
    "history_only_answer_hits",
    "current_plus_history_answer_hits",
    "is_global_or_temporal",
    "is_pure_global_statistical",
    "current_pool_size",
    "history_pool_size",
    "current_plus_history_pool_size",
]

EVIDENCE_FIELDNAMES = [
    "question_id",
    "evidence_scope",
    "rank",
    "retrieval_mode",
    "source_agent",
    "day",
    "start_time",
    "end_time",
    "start_seconds",
    "end_seconds",
    "granularity",
    "caption_text",
    "score",
    "memory_id",
    "raw_key",
    "source_file",
    "is_history_before_context",
    "is_current_near_context",
]


@dataclass(frozen=True)
class MemoryChunk:
    memory_id: str
    source_agent: str
    day: int
    start_time: float
    end_time: float
    start_hms: str
    end_hms: str
    granularity: str
    caption_text: str
    raw_key: str
    source_file: str


@dataclass(frozen=True)
class RankedEvidence:
    chunk_index: int
    score: float


@dataclass(frozen=True)
class Answerability:
    label: str
    reason: str
    answer_coverage: float
    question_overlap: int
    answer_hits: tuple[str, ...]
    direct_answer_hit: bool
    evidence_rows: int


def first_existing(candidates: Iterable[Path], description: str) -> Path:
    for path in candidates:
        if path.exists():
            return path
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {description}. Tried: {joined}")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


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


def seconds_to_hms(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = max(0.0, float(seconds))
    hour = int(seconds // 3600)
    minute = int((seconds % 3600) // 60)
    sec = seconds - hour * 3600 - minute * 60
    return f"{hour:02d}:{minute:02d}:{sec:05.2f}"


def normalize_agent(name: Any) -> str | None:
    return AGENT_LOWER_TO_CANONICAL.get(str(name or "").strip().lower())


def extract_agents_from_text(text: Any) -> set[str]:
    found = set()
    for name in AGENT_ORDER:
        if re.search(rf"\b{re.escape(name)}\b", str(text or ""), flags=re.IGNORECASE):
            found.add(name)
    return found


def stem_token(token: str) -> str:
    replacements = {
        "beverages": "beverage",
        "cubes": "cube",
        "drinks": "drink",
        "mismatched": "mismatch",
        "mismatches": "mismatch",
        "mismatching": "mismatch",
        "pieces": "piece",
        "sodas": "soda",
        "deviates": "deviate",
        "deviated": "deviate",
        "deviating": "deviate",
    }
    if token in replacements:
        return replacements[token]
    for suffix in ["ing", "ed", "es", "s"]:
        if len(token) > 5 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def tokenize_semantic(text: Any) -> set[str]:
    tokens = set()
    for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9']+", normalize_ws(text).lower()):
        token = token.strip("'")
        if len(token) < 2 or token in STOPWORDS:
            continue
        stemmed = stem_token(token)
        if stemmed in STOPWORDS:
            continue
        tokens.add(stemmed)
    return tokens


def tokenize_retrieval(text: Any) -> list[str]:
    tokens = []
    for token in re.findall(r"[a-z0-9][a-z0-9']+", normalize_ws(text).lower()):
        token = token.strip("'")
        if len(token) < 2 or token in RETRIEVAL_STOPWORDS:
            continue
        tokens.append(stem_token(token))
    return tokens


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

    expanded = " ".join(option_map.get(letter, "") for letter in answer.split("-"))
    return expanded or answer


def answer_keywords(question: str, answer: str) -> set[str]:
    expanded = parse_answer_option(question, answer)
    tokens = tokenize_semantic(expanded)
    raw = normalize_ws(answer).lower()
    if raw and len(raw) > 1 and not re.fullmatch(r"[a-e](?:-[a-e])*", raw):
        tokens |= tokenize_semantic(raw)
    return tokens


def is_global_or_temporal(question: str) -> bool:
    lower = question.lower()
    patterns = [
        r"\bwho\s+used\b.*\bthe\s+most\b",
        r"\bthe\s+most\b",
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
        r"\bbefore\s+\d{1,2}\s*(?:am|pm)\b",
        r"\bafter\s+\d{1,2}\s*(?:am|pm)\b",
    ]
    return any(re.search(pattern, lower) for pattern in patterns)


def requires_exhaustive_temporal_or_count_coverage(question: str) -> bool:
    lower = question.lower()
    patterns = [
        r"\bwho\s+used\b.*\bthe\s+most\b",
        r"\bwhen\s+was\b.*\bused\s+the\s+most\b",
        r"\bfirst\s+time\b",
        r"\blast\s+time\b",
        r"\bwhich\s+of\s+the\s+following\s+happened\s+(first|last)\b",
        r"\bcorrect\s+sequence\s+of\s+events\b",
        r"\bhow\s+many\b",
        r"\bnumber\s+of\b",
        r"\bfirst\s+person\b",
        r"\blast\s+person\b",
        r"\bwho\s+was\s+the\s+(first|last)\b",
    ]
    return any(re.search(pattern, lower) for pattern in patterns)


def is_pure_global_statistical(question: str, agents_in_context: str) -> bool:
    lower = question.lower()
    agents = [agent for agent in agents_in_context.split(";") if agent]
    global_patterns = [
        r"\bhow\s+many\s+people\s+used\b",
        r"\bwho\s+used\b.*\bthe\s+most\b",
        r"\bwhen\s+was\b.*\bused\s+the\s+most\b",
        r"\bwho\s+was\s+the\s+(first|last)\s+person\s+to\s+use\b",
    ]
    if any(re.search(pattern, lower) for pattern in global_patterns):
        return len(agents) >= 2 or not extract_agents_from_text(question)
    return False


def has_explicit_person(question: str) -> bool:
    return bool(tokenize_semantic(question) & AGENT_NAMES)


def infer_query_user(qa: dict[str, Any], parse_row: dict[str, str]) -> tuple[str, str]:
    question = str(qa.get("question", ""))
    for pattern in EXPLICIT_QUERY_PATTERNS:
        match = pattern.search(question)
        if match:
            return normalize_agent(match.group(1)) or match.group(1), "explicit"

    question_agents = sorted(extract_agents_from_text(question), key=AGENT_ORDER.index)
    if len(question_agents) == 1:
        return question_agents[0], "inferred_weak"

    agents = [agent for agent in str(parse_row.get("agents_in_context", "")).split(";") if agent]
    if len(agents) == 1:
        return agents[0], "inferred_weak"

    parsed_windows = parse_windows(parse_row)
    speakers = {
        normalize_agent(window.get("speaker"))
        for window in parsed_windows
        if normalize_agent(window.get("speaker")) is not None
    }
    if len(speakers) == 1:
        return next(iter(speakers)), "inferred_weak"

    return "UNKNOWN", "unknown"


def parse_windows(parse_row: dict[str, str]) -> list[dict[str, Any]]:
    raw = parse_row.get("parsed_contexts_json", "")
    windows: list[dict[str, Any]] = []
    if raw:
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, list):
                for item in loaded:
                    if not isinstance(item, dict):
                        continue
                    day = parse_int(item.get("day"), default=-1)
                    start = parse_float(item.get("start_time"), default=-1.0)
                    end = parse_float(item.get("end_time"), default=-1.0)
                    if day >= 0 and start >= 0 and end >= 0:
                        window = dict(item)
                        window["day"] = day
                        window["start_time"] = start
                        window["end_time"] = end
                        windows.append(window)
        except json.JSONDecodeError:
            pass

    if not windows:
        day = parse_int(parse_row.get("context_day"), default=-1)
        start = parse_float(parse_row.get("context_start"), default=-1.0)
        end = parse_float(parse_row.get("context_end"), default=-1.0)
        if day >= 0 and start >= 0 and end >= 0:
            windows.append({"day": day, "start_time": start, "end_time": end})
    return windows


def active_agents(parse_row: dict[str, str]) -> set[str]:
    agents = {agent for agent in str(parse_row.get("agents_in_context", "")).split(";") if agent}
    return agents or set(AGENT_ORDER)


def load_memory_index(path: Path) -> list[MemoryChunk]:
    rows = read_csv(path)
    memory = []
    for row in rows:
        text = row.get("caption_text", "").strip()
        if not text:
            continue
        memory.append(
            MemoryChunk(
                memory_id=row.get("memory_id", ""),
                source_agent=row.get("source_agent", ""),
                day=parse_int(row.get("day"), default=-1),
                start_time=parse_float(row.get("start_seconds"), default=-1.0),
                end_time=parse_float(row.get("end_seconds"), default=-1.0),
                start_hms=row.get("start_time", ""),
                end_hms=row.get("end_time", ""),
                granularity=row.get("granularity", ""),
                caption_text=text,
                raw_key=row.get("raw_key", ""),
                source_file=row.get("source_file", ""),
            )
        )
    return [chunk for chunk in memory if chunk.day >= 0 and chunk.start_time >= 0 and chunk.end_time >= 0]


class BM25Index:
    def __init__(self, chunks: list[MemoryChunk]) -> None:
        self.chunks = chunks
        self.doc_tokens: list[list[str]] = []
        self.doc_lens: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for idx, chunk in enumerate(chunks):
            tokens = tokenize_retrieval(chunk.caption_text)
            self.doc_tokens.append(tokens)
            self.doc_lens.append(len(tokens))
            for term, freq in Counter(tokens).items():
                self.postings[term].append((idx, freq))

    def rank(self, query: str, pool_indices: list[int], top_k: int) -> list[RankedEvidence]:
        if not pool_indices:
            return []

        query_counts = Counter(tokenize_retrieval(query))
        if not query_counts:
            return []

        unique_pool = list(dict.fromkeys(pool_indices))
        pool_set = set(unique_pool)
        n_docs = len(unique_pool)
        avgdl = sum(self.doc_lens[idx] for idx in unique_pool) / max(1, n_docs)
        if avgdl <= 0:
            return []

        k1 = 1.5
        b = 0.75
        scores: defaultdict[int, float] = defaultdict(float)

        for term, qtf in query_counts.items():
            term_postings = self.postings.get(term, [])
            matches = [(idx, freq) for idx, freq in term_postings if idx in pool_set]
            df = len(matches)
            if df == 0:
                continue
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for idx, freq in matches:
                doc_len = self.doc_lens[idx]
                denom = freq + k1 * (1.0 - b + b * doc_len / avgdl)
                scores[idx] += idf * (freq * (k1 + 1.0) / denom) * min(qtf, 3)

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [RankedEvidence(chunk_index=idx, score=score) for idx, score in ranked if score > 0]


def overlaps_any_window(chunk: MemoryChunk, windows: list[dict[str, Any]], margin: float) -> bool:
    for window in windows:
        if chunk.day != int(window["day"]):
            continue
        start = float(window["start_time"])
        end = float(window["end_time"])
        if chunk.start_time <= end + margin and chunk.end_time >= start - margin:
            return True
    return False


def is_before_context_start(chunk: MemoryChunk, context_day: int, context_start: float) -> bool:
    if chunk.day < context_day:
        return True
    if chunk.day == context_day and chunk.end_time <= context_start:
        return True
    return False


def build_pools(
    memory: list[MemoryChunk],
    parse_row: dict[str, str],
    margin: float,
) -> tuple[list[int], list[int], list[int]]:
    windows = parse_windows(parse_row)
    context_day = parse_int(parse_row.get("context_day"), default=-1)
    context_start = parse_float(parse_row.get("context_start"), default=-1.0)
    agents = active_agents(parse_row)

    current_indices = []
    history_indices = []
    for idx, chunk in enumerate(memory):
        if chunk.source_agent not in agents:
            continue
        is_current = overlaps_any_window(chunk, windows, margin)
        is_history = is_before_context_start(chunk, context_day, context_start)
        if is_current:
            current_indices.append(idx)
        if is_history:
            history_indices.append(idx)

    plus_indices = list(dict.fromkeys(current_indices + history_indices))
    return current_indices, history_indices, plus_indices


def evidence_text(ranked: list[RankedEvidence], memory: list[MemoryChunk]) -> str:
    return "\n".join(memory[item.chunk_index].caption_text for item in ranked)


def top_score(ranked: list[RankedEvidence]) -> float:
    return max((item.score for item in ranked), default=0.0)


def assess_answerability(question: str, answer: str, ranked: list[RankedEvidence], memory: list[MemoryChunk]) -> Answerability:
    text = evidence_text(ranked, memory)
    if not text.strip():
        return Answerability("not_answerable", "no retrieved caption evidence for this scope", 0.0, 0, tuple(), False, 0)

    evidence_tokens = tokenize_semantic(text)
    q_tokens = tokenize_semantic(question)
    a_tokens = answer_keywords(question, answer)
    q_overlap = len(q_tokens & evidence_tokens)
    answer_hits = tuple(sorted(a_tokens & evidence_tokens))
    coverage = len(answer_hits) / max(1, len(a_tokens))
    lower_evidence = normalize_ws(text).lower()
    lower_answer = normalize_ws(answer).lower()
    direct_answer_hit = bool(
        lower_answer
        and len(lower_answer) > 2
        and not re.fullmatch(r"\d+(?:\.\d+)?", lower_answer)
        and lower_answer in lower_evidence
    )

    global_or_temporal = is_global_or_temporal(question)
    exhaustive_temporal_or_count = requires_exhaustive_temporal_or_count_coverage(question)
    if global_or_temporal:
        if not exhaustive_temporal_or_count and coverage >= 0.85 and q_overlap >= 6 and len(ranked) >= 6:
            return Answerability(
                "likely_answerable",
                "global/statistical/temporal question with unusually broad answer-keyword and question coverage in captions",
                coverage,
                q_overlap,
                answer_hits,
                direct_answer_hit,
                len(ranked),
            )
        if coverage >= 0.55 and q_overlap >= 3:
            reason = (
                "global/statistical/temporal question; captions mention answer-related terms but may not cover enough events to verify the full comparison/order"
            )
            if exhaustive_temporal_or_count:
                reason = (
                    "question appears to require exhaustive temporal/count/order coverage; captions mention answer-related terms but top-k evidence cannot verify absence/order exhaustively"
                )
            return Answerability(
                "partially_answerable",
                reason,
                coverage,
                q_overlap,
                answer_hits,
                direct_answer_hit,
                len(ranked),
            )
        if q_overlap >= 4 or coverage >= 0.25:
            return Answerability(
                "unclear",
                "global/statistical/temporal question with partial lexical overlap; caption coverage may be too local",
                coverage,
                q_overlap,
                answer_hits,
                direct_answer_hit,
                len(ranked),
            )
        return Answerability(
            "not_answerable",
            "global/statistical/temporal question and this evidence scope does not expose enough answer-related coverage",
            coverage,
            q_overlap,
            answer_hits,
            direct_answer_hit,
            len(ranked),
        )

    if direct_answer_hit and q_overlap >= 2:
        return Answerability(
            "likely_answerable",
            "evidence contains the answer phrase and several question-relevant terms",
            coverage,
            q_overlap,
            answer_hits,
            direct_answer_hit,
            len(ranked),
        )
    if coverage >= 0.70 and q_overlap >= 2:
        return Answerability(
            "likely_answerable",
            "evidence covers most answer keywords and overlaps with the question context",
            coverage,
            q_overlap,
            answer_hits,
            direct_answer_hit,
            len(ranked),
        )
    if coverage >= 0.50 and q_overlap >= 5:
        return Answerability(
            "likely_answerable",
            "evidence covers key answer terms and strongly overlaps with the question context",
            coverage,
            q_overlap,
            answer_hits,
            direct_answer_hit,
            len(ranked),
        )
    if coverage >= 0.35 and q_overlap >= 2:
        return Answerability(
            "partially_answerable",
            "evidence covers some answer keywords but may not fully support the answer",
            coverage,
            q_overlap,
            answer_hits,
            direct_answer_hit,
            len(ranked),
        )
    if coverage > 0 or q_overlap >= 4:
        return Answerability(
            "unclear",
            "evidence has lexical overlap but does not clearly support the answer",
            coverage,
            q_overlap,
            answer_hits,
            direct_answer_hit,
            len(ranked),
        )
    return Answerability(
        "not_answerable",
        "evidence does not contain enough answer-relevant caption content",
        coverage,
        q_overlap,
        answer_hits,
        direct_answer_hit,
        len(ranked),
    )


def is_strong_partial(answerability: Answerability) -> bool:
    return (
        answerability.label == "partially_answerable"
        and answerability.answer_coverage >= 0.45
        and answerability.question_overlap >= 3
        and (answerability.direct_answer_hit or len(answerability.answer_hits) >= 2)
    )


def history_relevant(history: Answerability) -> bool:
    return history.label in {"likely_answerable", "partially_answerable"} or (
        history.label == "unclear" and (history.answer_coverage >= 0.20 or history.question_overlap >= 4)
    )


def history_adds_concrete_evidence(current: Answerability, history: Answerability, plus: Answerability) -> bool:
    current_hits = set(current.answer_hits)
    history_hits = set(history.answer_hits)
    plus_hits = set(plus.answer_hits)
    if history.label in {"likely_answerable", "partially_answerable"} and (history_hits or history.question_overlap >= 3):
        return True
    if plus_hits - current_hits:
        return True
    if history.answer_coverage >= 0.25 and history.question_overlap >= 3:
        return True
    return False


def gain_label(current: Answerability, history: Answerability, plus: Answerability) -> str:
    current_rank = ANSWERABILITY_RANK[current.label]
    plus_rank = ANSWERABILITY_RANK[plus.label]
    relevant_history = history_relevant(history)

    if plus_rank > current_rank and plus_rank >= 2 and relevant_history:
        return "yes"
    if plus_rank > current_rank and relevant_history:
        return "unclear_but_promising"
    if current.label == "likely_answerable" and plus_rank <= current_rank:
        return "no"

    added_hits = set(plus.answer_hits) - set(current.answer_hits)
    if (
        plus_rank >= current_rank
        and plus.label in {"likely_answerable", "partially_answerable", "unclear"}
        and relevant_history
        and (added_hits or plus.question_overlap >= current.question_overlap + 2)
    ):
        return "unclear_but_promising"

    if plus_rank == current_rank:
        return "no"
    return "unclear"


def best_scope(
    current: Answerability,
    history: Answerability,
    plus: Answerability,
) -> str:
    candidates = [
        ("current_plus_historical", plus),
        ("current_only", current),
        ("history_only", history),
    ]
    return max(candidates, key=lambda item: (ANSWERABILITY_RANK[item[1].label], -candidates.index(item)))[0]


def evidence_chain_readable(plus: Answerability, current_summary: str, history_summary: str) -> bool:
    return (
        plus.label in {"likely_answerable", "partially_answerable"}
        and plus.question_overlap >= 3
        and bool(history_summary)
        and len(current_summary + history_summary) >= 120
    )


def tier_bucket(
    question: str,
    agents_in_context: str,
    current: Answerability,
    history: Answerability,
    plus: Answerability,
    gain: str,
    current_summary: str,
    history_summary: str,
) -> str:
    global_temporal = is_global_or_temporal(question)
    pure_global = is_pure_global_statistical(question, agents_in_context)
    concrete_history = history_adds_concrete_evidence(current, history, plus)
    plus_good = plus.label == "likely_answerable" or is_strong_partial(plus)
    current_weak = current.label in {"not_answerable", "unclear"} or (
        current.label == "partially_answerable" and not is_strong_partial(current)
    )
    readable = evidence_chain_readable(plus, current_summary, history_summary)

    if pure_global and plus.label != "likely_answerable":
        return "tier_E_not_suitable"

    if plus_good and current_weak and concrete_history and readable and not (global_temporal and plus.label != "likely_answerable"):
        return "tier_A_demo_ready"

    if (
        plus.label in {"likely_answerable", "partially_answerable", "unclear"}
        and gain in {"yes", "unclear_but_promising"}
        and concrete_history
        and not pure_global
    ):
        return "tier_B_promising_history_gain"

    if current.label == "likely_answerable":
        return "tier_C_current_sufficient_control"

    if pure_global or (not has_explicit_person(question) and plus.label in {"not_answerable", "unclear"} and history.label == "not_answerable"):
        return "tier_E_not_suitable"

    if plus.label in {"not_answerable", "unclear"}:
        return "tier_D_reject_unclear"

    return "tier_D_reject_unclear"


def format_evidence_summary(ranked: list[RankedEvidence], memory: list[MemoryChunk], scope: str, limit_rows: int = 3) -> str:
    parts = []
    for rank, item in enumerate(ranked[:limit_rows], start=1):
        chunk = memory[item.chunk_index]
        parts.append(
            f"[{scope} rank={rank} score={item.score:.4f} source={chunk.source_agent} "
            f"D{chunk.day} {chunk.start_hms}-{chunk.end_hms} {chunk.granularity}] "
            f"{trunc(chunk.caption_text, 260)}"
        )
    return " || ".join(parts)


def reason_for_row(
    current: Answerability,
    history: Answerability,
    plus: Answerability,
    gain: str,
    bucket: str,
    global_temporal: bool,
    pure_global: bool,
) -> str:
    pieces = [
        f"current={current.label} ({current.reason}; answer_hits={','.join(current.answer_hits) or 'none'}; coverage={current.answer_coverage:.2f}; q_overlap={current.question_overlap})",
        f"history={history.label} ({history.reason}; answer_hits={','.join(history.answer_hits) or 'none'}; coverage={history.answer_coverage:.2f}; q_overlap={history.question_overlap})",
        f"current+history={plus.label} ({plus.reason}; answer_hits={','.join(plus.answer_hits) or 'none'}; coverage={plus.answer_coverage:.2f}; q_overlap={plus.question_overlap})",
        f"gain={gain}",
        f"bucket={bucket}",
    ]
    if global_temporal:
        pieces.append("global/statistical/temporal question handled conservatively")
    if pure_global:
        pieces.append("pure global all-agent statistical/use question is not ideal for this historical-memory study")
    pieces.append("caption-only heuristic; not a final label and not answer accuracy")
    return " | ".join(pieces)


def build_evidence_rows(
    question_id: int,
    scope: str,
    ranked: list[RankedEvidence],
    memory: list[MemoryChunk],
    parse_row: dict[str, str],
) -> list[dict[str, Any]]:
    context_day = parse_int(parse_row.get("context_day"), default=-1)
    context_start = parse_float(parse_row.get("context_start"), default=-1.0)
    windows = parse_windows(parse_row)
    rows = []
    for rank, item in enumerate(ranked, start=1):
        chunk = memory[item.chunk_index]
        rows.append(
            {
                "question_id": question_id,
                "evidence_scope": scope,
                "rank": rank,
                "retrieval_mode": scope,
                "source_agent": chunk.source_agent,
                "day": chunk.day,
                "start_time": chunk.start_hms,
                "end_time": chunk.end_hms,
                "start_seconds": f"{chunk.start_time:.2f}",
                "end_seconds": f"{chunk.end_time:.2f}",
                "granularity": chunk.granularity,
                "caption_text": chunk.caption_text,
                "score": f"{item.score:.4f}",
                "memory_id": chunk.memory_id,
                "raw_key": chunk.raw_key,
                "source_file": chunk.source_file,
                "is_history_before_context": "yes" if is_before_context_start(chunk, context_day, context_start) else "no",
                "is_current_near_context": "yes" if overlaps_any_window(chunk, windows, CURRENT_MARGIN_SECONDS) else "no",
            }
        )
    return rows


def make_retrieval_query(qa: dict[str, Any]) -> str:
    question = normalize_ws(qa.get("question", ""))
    answer = normalize_ws(qa.get("answer", ""))
    return f"{question} {answer}".strip()


def validate_inputs() -> dict[str, Path]:
    qa_path = first_existing(QA_PATH_CANDIDATES, "MA-EgoQA.json")
    caption_30sec_dir = first_existing(CAPTION_30SEC_CANDIDATES, "30sec caption directory")
    caption_10min_dir = first_existing(CAPTION_10MIN_CANDIDATES, "10min caption directory")
    if not CONTEXT_PARSE_PATH.exists():
        raise FileNotFoundError(CONTEXT_PARSE_PATH)
    if not MEMORY_INDEX_PATH.exists():
        raise FileNotFoundError(MEMORY_INDEX_PATH)
    return {
        "qa_path": qa_path,
        "caption_30sec_dir": caption_30sec_dir,
        "caption_10min_dir": caption_10min_dir,
        "context_parse_path": CONTEXT_PARSE_PATH,
        "memory_index_path": MEMORY_INDEX_PATH,
    }


def sort_recommended(row: dict[str, Any]) -> tuple[int, int, int, float, int]:
    return (
        TIER_ORDER.get(row["evidence_scope_bucket"], 99),
        0 if row["current_plus_history_gain"] == "yes" else 1,
        -ANSWERABILITY_RANK.get(row["current_plus_history_answerability"], 0),
        -parse_float(row["current_plus_history_answer_coverage"]),
        parse_int(row["question_id"]),
    )


def build_outputs(output_dir: Path, top_k: int = TOP_K, margin: float = CURRENT_MARGIN_SECONDS) -> dict[str, Any]:
    paths = validate_inputs()
    output_dir.mkdir(parents=True, exist_ok=True)

    qas = load_json(paths["qa_path"])
    if not isinstance(qas, list):
        raise RuntimeError(f"Expected a list in {paths['qa_path']}")

    parse_rows = read_csv(paths["context_parse_path"])
    parse_by_id = {parse_int(row.get("question_id"), default=-1): row for row in parse_rows}
    memory = load_memory_index(paths["memory_index_path"])
    bm25 = BM25Index(memory)

    fullscreen_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    processed = 0
    query_user_status_counts: Counter[str] = Counter()

    for question_id, qa in enumerate(qas):
        parse_row = parse_by_id.get(question_id)
        if parse_row is None or parse_row.get("parse_status") != "ok":
            continue

        processed += 1
        current_pool, history_pool, plus_pool = build_pools(memory, parse_row, margin)
        retrieval_query = make_retrieval_query(qa)

        current_ranked = bm25.rank(retrieval_query, current_pool, top_k)
        history_ranked = bm25.rank(retrieval_query, history_pool, top_k)
        plus_ranked = bm25.rank(retrieval_query, plus_pool, top_k)

        current = assess_answerability(str(qa.get("question", "")), str(qa.get("answer", "")), current_ranked, memory)
        history = assess_answerability(str(qa.get("question", "")), str(qa.get("answer", "")), history_ranked, memory)
        plus = assess_answerability(str(qa.get("question", "")), str(qa.get("answer", "")), plus_ranked, memory)
        gain = gain_label(current, history, plus)
        best = best_scope(current, history, plus)
        query_user, query_user_status = infer_query_user(qa, parse_row)
        query_user_status_counts[query_user_status] += 1

        current_summary = format_evidence_summary(current_ranked, memory, "current_only", limit_rows=3)
        history_summary = format_evidence_summary(history_ranked, memory, "history_only", limit_rows=3)
        plus_summary = format_evidence_summary(plus_ranked, memory, "current_plus_historical", limit_rows=4)
        global_temporal = is_global_or_temporal(str(qa.get("question", "")))
        pure_global = is_pure_global_statistical(str(qa.get("question", "")), parse_row.get("agents_in_context", ""))
        bucket = tier_bucket(
            str(qa.get("question", "")),
            parse_row.get("agents_in_context", ""),
            current,
            history,
            plus,
            gain,
            current_summary,
            history_summary,
        )

        row = {
            "question_id": question_id,
            "category": qa.get("category", ""),
            "subcategory": qa.get("subcategory", ""),
            "question": qa.get("question", ""),
            "answer": qa.get("answer", ""),
            "options_json": json_dumps(qa.get("options", [])),
            "contexts_json": json_dumps(qa.get("contexts", {})),
            "parse_status": parse_row.get("parse_status", ""),
            "context_day": parse_row.get("context_day", ""),
            "context_start": parse_row.get("context_start", ""),
            "context_end": parse_row.get("context_end", ""),
            "context_start_hms": parse_row.get("context_start_hms", seconds_to_hms(parse_float(parse_row.get("context_start")))),
            "context_end_hms": parse_row.get("context_end_hms", seconds_to_hms(parse_float(parse_row.get("context_end")))),
            "agents_in_context": parse_row.get("agents_in_context", ""),
            "parsed_contexts_json": parse_row.get("parsed_contexts_json", ""),
            "provisional_query_user": query_user,
            "query_user_status": query_user_status,
            "current_only_answerability": current.label,
            "history_only_answerability": history.label,
            "current_plus_history_answerability": plus.label,
            "current_plus_history_gain": gain,
            "evidence_scope_bucket": bucket,
            "best_evidence_scope": best,
            "best_current_evidence_summary": current_summary,
            "best_history_evidence_summary": history_summary,
            "best_current_plus_history_summary": plus_summary,
            "reason": reason_for_row(current, history, plus, gain, bucket, global_temporal, pure_global),
            "needs_user_check": "yes",
            "current_only_top_score": f"{top_score(current_ranked):.4f}",
            "history_only_top_score": f"{top_score(history_ranked):.4f}",
            "current_plus_history_top_score": f"{top_score(plus_ranked):.4f}",
            "current_only_answer_coverage": f"{current.answer_coverage:.4f}",
            "history_only_answer_coverage": f"{history.answer_coverage:.4f}",
            "current_plus_history_answer_coverage": f"{plus.answer_coverage:.4f}",
            "current_only_question_overlap": current.question_overlap,
            "history_only_question_overlap": history.question_overlap,
            "current_plus_history_question_overlap": plus.question_overlap,
            "current_only_answer_hits": ";".join(current.answer_hits),
            "history_only_answer_hits": ";".join(history.answer_hits),
            "current_plus_history_answer_hits": ";".join(plus.answer_hits),
            "is_global_or_temporal": "yes" if global_temporal else "no",
            "is_pure_global_statistical": "yes" if pure_global else "no",
            "current_pool_size": len(current_pool),
            "history_pool_size": len(history_pool),
            "current_plus_history_pool_size": len(plus_pool),
        }
        fullscreen_rows.append(row)

        evidence_rows.extend(build_evidence_rows(question_id, "current_only", current_ranked, memory, parse_row))
        evidence_rows.extend(build_evidence_rows(question_id, "history_only", history_ranked, memory, parse_row))
        evidence_rows.extend(build_evidence_rows(question_id, "current_plus_historical", plus_ranked, memory, parse_row))

    fullscreen_rows.sort(key=lambda row: parse_int(row["question_id"]))
    evidence_rows.sort(
        key=lambda row: (
            parse_int(row["question_id"]),
            {"current_only": 0, "history_only": 1, "current_plus_historical": 2}.get(row["evidence_scope"], 9),
            parse_int(row["rank"]),
        )
    )

    fullscreen_path = output_dir / "evidence_scope_fullscreen_v2.csv"
    evidence_path = output_dir / "evidence_scope_fullscreen_v2_topk_evidence.csv"
    write_csv(fullscreen_path, fullscreen_rows, FULLSCREEN_FIELDNAMES)
    write_csv(evidence_path, evidence_rows, EVIDENCE_FIELDNAMES)

    tier_paths = {
        "tier_A_demo_ready": output_dir / "tier_A_demo_ready_cases_v2.csv",
        "tier_B_promising_history_gain": output_dir / "tier_B_promising_history_gain_cases_v2.csv",
        "tier_C_current_sufficient_control": output_dir / "tier_C_current_sufficient_controls_v2.csv",
        "tier_D_reject_unclear": output_dir / "tier_D_reject_unclear_cases_v2.csv",
        "tier_E_not_suitable": output_dir / "tier_E_not_suitable_cases_v2.csv",
    }
    for tier, path in tier_paths.items():
        tier_rows = [row for row in fullscreen_rows if row["evidence_scope_bucket"] == tier]
        tier_rows.sort(key=sort_recommended)
        write_csv(path, tier_rows, FULLSCREEN_FIELDNAMES)

    report_path = output_dir / "evidence_scope_fullscreen_v2_report.md"
    write_report(
        report_path,
        qas=qas,
        parse_rows=parse_rows,
        processed=processed,
        fullscreen_rows=fullscreen_rows,
        evidence_path=evidence_path,
        top_k=top_k,
        margin=margin,
        query_user_status_counts=query_user_status_counts,
    )

    tier_counts = Counter(row["evidence_scope_bucket"] for row in fullscreen_rows)
    recommended = sorted(
        [
            row
            for row in fullscreen_rows
            if row["evidence_scope_bucket"] in {"tier_A_demo_ready", "tier_B_promising_history_gain"}
        ],
        key=sort_recommended,
    )[:10]

    return {
        "output_dir": output_dir,
        "fullscreen_path": fullscreen_path,
        "evidence_path": evidence_path,
        "report_path": report_path,
        "tier_paths": tier_paths,
        "questions_loaded": len(qas),
        "parseable_questions": sum(1 for row in parse_rows if row.get("parse_status") == "ok"),
        "processed": processed,
        "tier_counts": tier_counts,
        "recommended": recommended,
    }


def write_report(
    path: Path,
    *,
    qas: list[dict[str, Any]],
    parse_rows: list[dict[str, str]],
    processed: int,
    fullscreen_rows: list[dict[str, Any]],
    evidence_path: Path,
    top_k: int,
    margin: float,
    query_user_status_counts: Counter[str],
) -> None:
    parseable = sum(1 for row in parse_rows if row.get("parse_status") == "ok")
    tier_counts = Counter(row["evidence_scope_bucket"] for row in fullscreen_rows)
    current_counts = Counter(row["current_only_answerability"] for row in fullscreen_rows)
    history_counts = Counter(row["history_only_answerability"] for row in fullscreen_rows)
    plus_counts = Counter(row["current_plus_history_answerability"] for row in fullscreen_rows)
    gain_counts = Counter(row["current_plus_history_gain"] for row in fullscreen_rows)
    better_count = gain_counts.get("yes", 0)
    promising_count = gain_counts.get("unclear_but_promising", 0)
    top_cases = sorted(
        [
            row
            for row in fullscreen_rows
            if row["evidence_scope_bucket"] in {"tier_A_demo_ready", "tier_B_promising_history_gain"}
        ],
        key=sort_recommended,
    )[:20]

    lines = [
        "# Evidence Scope Full Screening V2 Report",
        "",
        "Scope: caption-only full screening over MA-EgoQA questions to find candidate current-only vs current+historical evidence coverage cases. No videos were downloaded, no VLM/LLM/API was used, original MA-EgoQA files were not modified, and these rows are not final labels.",
        "",
        "## Configuration",
        "",
        f"- Retrieval query mode: question + gold answer text, for candidate evidence discovery only.",
        f"- Top-k per scope: {top_k}.",
        f"- Current caption margin around parsed context windows: {margin:.0f} seconds.",
        f"- Historical evidence rule: caption end_time must be before parsed context_start; same-day future captions are excluded from history.",
        f"- Top-k provenance table: `{evidence_path.name}`.",
        "",
        "## Required Counts",
        "",
        f"1. Total MA-EgoQA questions loaded: {len(qas)}.",
        f"2. Parseable questions: {parseable}.",
        f"3. Processed questions: {processed}.",
        f"4. Tier A/B/C/D/E counts: A={tier_counts.get('tier_A_demo_ready', 0)}, B={tier_counts.get('tier_B_promising_history_gain', 0)}, C={tier_counts.get('tier_C_current_sufficient_control', 0)}, D={tier_counts.get('tier_D_reject_unclear', 0)}, E={tier_counts.get('tier_E_not_suitable', 0)}.",
        f"5. current-only answerability distribution: {dict(current_counts)}.",
        f"6. history-only answerability distribution: {dict(history_counts)}.",
        f"7. current+historical answerability distribution: {dict(plus_counts)}.",
        f"8. current+historical better than current-only count: {better_count} yes; {promising_count} unclear_but_promising.",
        "",
        "## Top 20 Tier A/B Cases To Inspect",
        "",
    ]
    if top_cases:
        for row in top_cases:
            lines.append(
                f"- Q{row['question_id']} [{row['evidence_scope_bucket']}; gain={row['current_plus_history_gain']}; "
                f"current={row['current_only_answerability']}; plus={row['current_plus_history_answerability']}] "
                f"{trunc(row['question'], 170)} | history: {trunc(row['best_history_evidence_summary'], 220)}"
            )
    else:
        lines.append("- No Tier A/B cases met the current heuristics.")

    lines += [
        "",
        "## Why V1 Only Produced 1 Constructed Case",
        "",
        "- V1 first narrowed the problem to 30 heuristic candidates instead of screening all 1,741 questions.",
        "- It used conservative gates around known/provisional query_user, non-global questions, current+history gain, and constructed-case confidence.",
        "- The constructed subset required readable caption evidence and rejected many global/statistical/order questions where caption retrieval looked too local.",
        "- Therefore V1 was useful as a pilot, but its candidate pool and final construction criteria were too small and conservative to test whether historical evidence has broader signal.",
        "",
        "## Preliminary Signal",
        "",
        (
            "- Caption-only full screening suggests candidate cases where current+historical evidence may improve coverage over current-only "
            f"({better_count} yes and {promising_count} unclear_but_promising under the heuristic)."
        ),
        "- This is preliminary evidence coverage signal only. It should be used to choose cases for manual inspection and later controlled packet construction.",
        "",
        "## Current Claim Boundary",
        "",
        "- Can claim: caption-only full screening suggests candidate cases where current+historical evidence may improve coverage over current-only.",
        "- Cannot claim: benchmark complete, labels final, model accuracy improves, historical memory proven useful, or self-first routing solved.",
        "- Each row keeps `needs_user_check=yes`; answerability is a conservative draft, not answer accuracy.",
        "",
        "## Additional Diagnostics",
        "",
        f"- query_user status distribution: {dict(query_user_status_counts)}.",
        f"- gain distribution: {dict(gain_counts)}.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def likely_tier_a_shortfall_reason(tier_counts: Counter[str]) -> str:
    if tier_counts.get("tier_A_demo_ready", 0) >= 5:
        return ""
    if tier_counts.get("tier_B_promising_history_gain", 0) >= 20:
        return "filtering too conservative: many cases are promising but do not pass the demo-ready gates."
    if tier_counts.get("tier_D_reject_unclear", 0) + tier_counts.get("tier_E_not_suitable", 0) > 1000:
        return "caption granularity and global/statistical question types likely make evidence chains hard to verify automatically."
    return "possible data/caption granularity issue or the historical-memory signal is weak under caption-only heuristics."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--current-margin-seconds", type=float, default=CURRENT_MARGIN_SECONDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_outputs(args.output_dir, top_k=args.top_k, margin=args.current_margin_seconds)
    tier_counts = result["tier_counts"]

    print(f"Output directory: {result['output_dir']}")
    print(f"Processed question count: {result['processed']}")
    print(f"Tier A count: {tier_counts.get('tier_A_demo_ready', 0)}")
    print(f"Tier B count: {tier_counts.get('tier_B_promising_history_gain', 0)}")
    print(f"Tier C count: {tier_counts.get('tier_C_current_sufficient_control', 0)}")
    print("Top 10 recommended cases:")
    if result["recommended"]:
        for row in result["recommended"]:
            print(
                f"- Q{row['question_id']} / {row['evidence_scope_bucket']} / gain={row['current_plus_history_gain']} / "
                f"current={row['current_only_answerability']} / plus={row['current_plus_history_answerability']} / "
                f"{trunc(row['question'], 150)}"
            )
    else:
        print("- No Tier A/B cases met the current heuristics.")

    shortfall = likely_tier_a_shortfall_reason(tier_counts)
    if shortfall:
        print(f"Tier A < 5 likely reason: {shortfall}")


if __name__ == "__main__":
    main()
