#!/usr/bin/env python3
"""
Text-only historical-memory feasibility inventory for MA-EgoQA / EgoLife.

This script intentionally does not download video, run VLM/LLM inference, or
modify original MA-EgoQA files. It builds source-isolated caption memory pools
and produces human-audit-ready candidate tables for self-first incremental
source-access analysis with historical memory.
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "MA-EgoQA" / "data"
OUTPUT_DIR = ROOT / "outputs" / "historical_v1"

QA_PATH = DATA_DIR / "MA-EgoQA.json"
CAPTION_30SEC_DIR = DATA_DIR / "caption" / "30sec"
CAPTION_10MIN_DIR = DATA_DIR / "caption" / "10min"

AGENT_ORDER = ["Jake", "Alice", "Tasha", "Lucia", "Katrina", "Shure"]
AGENT_UPPER_TO_CANONICAL = {name.upper(): name for name in AGENT_ORDER}
AGENT_LOWER_TO_CANONICAL = {name.lower(): name for name in AGENT_ORDER}

CURRENT_MARGIN_SECONDS = 120.0
TOP_K = 5
MAX_CANDIDATES = 30


@dataclass(frozen=True)
class MemoryChunk:
    memory_id: str
    source_agent: str
    day: int
    start_time: float
    end_time: float
    granularity: str
    caption_text: str
    raw_key: str
    source_file: str


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    def safe_value(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: safe_value(value) for key, value in row.items()})


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalize_agent(name: str) -> str | None:
    return AGENT_LOWER_TO_CANONICAL.get(name.strip().lower())


def extract_agents_from_text(text: str) -> set[str]:
    agents = set()
    for name in AGENT_ORDER:
        if re.search(rf"\b{re.escape(name)}\b", text, flags=re.IGNORECASE):
            agents.add(name)
    return agents


def parse_hhmmss_compact(value: str) -> float | None:
    digits = re.sub(r"\D", "", str(value))
    if len(digits) < 6:
        return None
    hour = int(digits[0:2])
    minute = int(digits[2:4])
    second = int(digits[4:6])
    centiseconds = int(digits[6:8]) if len(digits) >= 8 else 0
    if minute >= 60 or second >= 60:
        return None
    return hour * 3600 + minute * 60 + second + centiseconds / 100.0


def parse_colon_time(value: str) -> float | None:
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{1,2}):(\d{1,2})(?:\.(\d+))?\s*", value)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    second = int(match.group(3))
    fraction = match.group(4) or ""
    if minute >= 60 or second >= 60:
        return None
    frac_seconds = float(f"0.{fraction}") if fraction else 0.0
    return hour * 3600 + minute * 60 + second + frac_seconds


def seconds_to_hms(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = max(0.0, seconds)
    hour = int(seconds // 3600)
    minute = int((seconds % 3600) // 60)
    sec = seconds - hour * 3600 - minute * 60
    return f"{hour:02d}:{minute:02d}:{sec:05.2f}"


def parse_context_key(key: str) -> dict[str, Any] | None:
    match = re.fullmatch(r"DAY(\d+)_(\d{6,8})_(\d{6,8})", str(key))
    if not match:
        return None
    start = parse_hhmmss_compact(match.group(2))
    end = parse_hhmmss_compact(match.group(3))
    if start is None or end is None:
        return None
    return {
        "day": int(match.group(1)),
        "start_time": start,
        "end_time": end,
        "raw": key,
    }


TIMESTAMP_RE = re.compile(
    r"DAY(?P<day>\d+)\s+"
    r"(?P<start>\d{1,2}:\d{1,2}:\d{1,2}(?:\.\d+)?)\s*-\s*"
    r"(?P<end>\d{1,2}:\d{1,2}:\d{1,2}(?:\.\d+)?)"
    r"(?:,\s*(?P<speaker>[^:,\n]+)\s*:)?",
    flags=re.IGNORECASE,
)


def parse_timestamped_context(text: str) -> list[dict[str, Any]]:
    parsed = []
    for match in TIMESTAMP_RE.finditer(text):
        start = parse_colon_time(match.group("start"))
        end = parse_colon_time(match.group("end"))
        if start is None or end is None:
            continue
        speaker_raw = match.group("speaker") or ""
        speaker = normalize_agent(speaker_raw)
        parsed.append(
            {
                "day": int(match.group("day")),
                "start_time": start,
                "end_time": end,
                "raw": match.group(0),
                "speaker": speaker or speaker_raw.strip(),
            }
        )
    return parsed


def parse_contexts(question_id: int, qa: dict[str, Any]) -> dict[str, Any]:
    contexts = qa.get("contexts")
    parsed_windows: list[dict[str, Any]] = []
    agents: set[str] = set()
    raw_context_text = json_dumps(contexts)

    if isinstance(contexts, dict):
        for key, value in contexts.items():
            parsed = parse_context_key(key)
            if parsed is not None:
                parsed_windows.append(parsed)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        canonical = normalize_agent(item)
                        if canonical:
                            agents.add(canonical)
            elif isinstance(value, str):
                agents.update(extract_agents_from_text(value))
                for item in parse_timestamped_context(value):
                    parsed_windows.append(item)
                    if normalize_agent(str(item.get("speaker", ""))):
                        agents.add(normalize_agent(str(item["speaker"])) or "")
    elif isinstance(contexts, list):
        for item in contexts:
            text = str(item)
            agents.update(extract_agents_from_text(text))
            for parsed in parse_timestamped_context(text):
                parsed_windows.append(parsed)
                speaker = normalize_agent(str(parsed.get("speaker", "")))
                if speaker:
                    agents.add(speaker)
    else:
        agents.update(extract_agents_from_text(raw_context_text))
        for parsed in parse_timestamped_context(raw_context_text):
            parsed_windows.append(parsed)

    question_agents = extract_agents_from_text(str(qa.get("question", "")))
    answer_agents = extract_agents_from_text(str(qa.get("answer", "")))
    all_agents = sorted(agents | question_agents | answer_agents, key=AGENT_ORDER.index)

    valid_windows = [w for w in parsed_windows if w.get("day") is not None and w.get("start_time") is not None]
    if valid_windows:
        days = sorted({int(w["day"]) for w in valid_windows})
        primary_day = days[0]
        same_day_windows = [w for w in valid_windows if int(w["day"]) == primary_day]
        broad_start = min(float(w["start_time"]) for w in same_day_windows)
        broad_end = max(float(w["end_time"]) for w in same_day_windows)
        parse_status = "ok"
    else:
        primary_day = None
        broad_start = None
        broad_end = None
        parse_status = "failed"

    return {
        "question_id": question_id,
        "parse_status": parse_status,
        "context_type": type(contexts).__name__,
        "parsed_context_count": len(valid_windows),
        "context_day": primary_day,
        "context_start": broad_start,
        "context_end": broad_end,
        "context_start_hms": seconds_to_hms(broad_start),
        "context_end_hms": seconds_to_hms(broad_end),
        "agents_in_context": ";".join(all_agents),
        "agents_count": len(all_agents),
        "parsed_contexts_json": json_dumps(valid_windows),
        "raw_contexts_json": raw_context_text,
    }


def parse_caption_filename(path: Path) -> tuple[str, int, str] | None:
    match = re.search(r"_captions_A\d+_([A-Z]+)_DAY(\d+)\.json$", path.name)
    if not match:
        return None
    agent = normalize_agent(match.group(1))
    if agent is None:
        return None
    return agent, int(match.group(2)), path.name


def parse_30sec_key(key: str, fallback_day: int) -> tuple[int, float, float] | None:
    match = re.fullmatch(r"DAY(\d+)_A\d+_[A-Z]+_(\d{6,8})\.mp4", key)
    if not match:
        return None
    start = parse_hhmmss_compact(match.group(2))
    if start is None:
        return None
    return int(match.group(1)), start, start + 30.0


def parse_10min_key(key: str, fallback_day: int) -> tuple[int, float, float] | None:
    match = re.fullmatch(r"DAY(\d+)_A\d+_[A-Z]+_(\d{6,8})_(\d{6,8})", key)
    if not match:
        return None
    start = parse_hhmmss_compact(match.group(2))
    end = parse_hhmmss_compact(match.group(3))
    if start is None or end is None:
        return None
    return int(match.group(1)), start, end


def build_caption_memory_index() -> list[MemoryChunk]:
    chunks: list[MemoryChunk] = []
    counter = 0

    for granularity, directory, key_parser in [
        ("30sec", CAPTION_30SEC_DIR, parse_30sec_key),
        ("10min", CAPTION_10MIN_DIR, parse_10min_key),
    ]:
        for path in sorted(directory.glob("*.json")):
            parsed_file = parse_caption_filename(path)
            if parsed_file is None:
                continue
            agent, file_day, _ = parsed_file
            data = load_json(path)
            if not isinstance(data, dict):
                continue
            for raw_key, caption in data.items():
                parsed_key = key_parser(str(raw_key), file_day)
                if parsed_key is None:
                    continue
                day, start_time, end_time = parsed_key
                text = str(caption).strip()
                if not text:
                    continue
                counter += 1
                chunks.append(
                    MemoryChunk(
                        memory_id=f"M{counter:06d}",
                        source_agent=agent,
                        day=day,
                        start_time=start_time,
                        end_time=end_time,
                        granularity=granularity,
                        caption_text=text,
                        raw_key=str(raw_key),
                        source_file=str(path.relative_to(ROOT)),
                    )
                )
    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_rank(query: str, docs: list[MemoryChunk], top_k: int = TOP_K) -> list[tuple[MemoryChunk, float]]:
    if not docs:
        return []
    query_terms = tokenize(query)
    if not query_terms:
        return []
    query_counts = Counter(query_terms)

    doc_tokens = [tokenize(doc.caption_text) for doc in docs]
    doc_lens = [len(tokens) for tokens in doc_tokens]
    avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 0.0
    if avgdl <= 0:
        return []

    df: Counter[str] = Counter()
    query_vocab = set(query_counts)
    for tokens in doc_tokens:
        df.update(set(tokens) & query_vocab)

    n_docs = len(docs)
    k1 = 1.5
    b = 0.75
    scored: list[tuple[int, float]] = []
    for idx, tokens in enumerate(doc_tokens):
        if not tokens:
            continue
        tf = Counter(tokens)
        score = 0.0
        for term, qtf in query_counts.items():
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            idf = math.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            denom = freq + k1 * (1.0 - b + b * doc_lens[idx] / avgdl)
            score += idf * (freq * (k1 + 1.0) / denom) * min(qtf, 3)
        if score > 0:
            scored.append((idx, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return [(docs[idx], score) for idx, score in scored[:top_k]]


HISTORY_TERMS = {
    "first",
    "most",
    "again",
    "previous",
    "before",
    "after",
    "later",
    "when",
    "who",
    "what",
    "where",
    "used",
    "use",
    "object",
    "task",
    "role",
    "relationship",
    "remember",
    "same",
    "last",
    "order",
    "sequence",
    "finish",
    "finalize",
    "helped",
    "suggested",
}


def candidate_score(qa: dict[str, Any], parse_row: dict[str, Any]) -> int:
    text = f"{qa.get('question', '')} {qa.get('answer', '')}".lower()
    score = 0
    for term in HISTORY_TERMS:
        if term in text:
            score += 1
    category = qa.get("category", "")
    if category in {"temporal_reasoning", "environmental_interaction", "task_coordination"}:
        score += 3
    if parse_row["context_type"] == "list":
        score += 2
    score += min(int(parse_row["agents_count"]), 6)
    return score


def select_candidate_questions(
    qas: list[dict[str, Any]], context_rows: list[dict[str, Any]], max_candidates: int = MAX_CANDIDATES
) -> list[dict[str, Any]]:
    parse_by_id = {int(row["question_id"]): row for row in context_rows}
    eligible = []
    for question_id, qa in enumerate(qas):
        parse_row = parse_by_id[question_id]
        if parse_row["parse_status"] != "ok":
            continue
        combined_agents = set(parse_row["agents_in_context"].split(";")) if parse_row["agents_in_context"] else set()
        combined_agents |= extract_agents_from_text(str(qa.get("question", "")))
        if len([a for a in combined_agents if a]) < 2:
            continue
        score = candidate_score(qa, parse_row)
        eligible.append((qa.get("category", "unknown"), -score, question_id, qa, parse_row))

    by_category: dict[str, list[tuple[str, int, int, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for item in eligible:
        by_category[item[0]].append(item)
    for category in by_category:
        by_category[category].sort(key=lambda item: (item[1], item[2]))

    selected = []
    categories = sorted(by_category, key=lambda cat: len(by_category[cat]), reverse=True)
    per_category_cap = max(1, math.ceil(max_candidates / max(1, len(categories))))
    for category in categories:
        selected.extend(by_category[category][:per_category_cap])

    if len(selected) < max_candidates:
        used_ids = {item[2] for item in selected}
        remaining = sorted([item for item in eligible if item[2] not in used_ids], key=lambda item: (item[1], item[2]))
        selected.extend(remaining[: max_candidates - len(selected)])

    selected = sorted(selected, key=lambda item: (item[1], item[2]))[:max_candidates]
    return [
        {
            "question_id": question_id,
            "qa": qa,
            "parse": parse_row,
            "selection_score": -neg_score,
        }
        for _, neg_score, question_id, qa, parse_row in selected
    ]


EXPLICIT_QUERY_PATTERNS = [
    re.compile(r"\bfrom\s+(Jake|Alice|Tasha|Lucia|Katrina|Shure)'s\s+(?:view|perspective|camera|video)\b", re.I),
    re.compile(r"\baccording\s+to\s+(Jake|Alice|Tasha|Lucia|Katrina|Shure)\b", re.I),
    re.compile(r"\bin\s+(Jake|Alice|Tasha|Lucia|Katrina|Shure)'s\s+(?:view|perspective|camera|video)\b", re.I),
]


def infer_query_user(qa: dict[str, Any], parse_row: dict[str, Any]) -> tuple[str, str]:
    question = str(qa.get("question", ""))
    for pattern in EXPLICIT_QUERY_PATTERNS:
        match = pattern.search(question)
        if match:
            return normalize_agent(match.group(1)) or match.group(1), "explicit"

    question_agents = sorted(extract_agents_from_text(question), key=AGENT_ORDER.index)
    if len(question_agents) == 1:
        return question_agents[0], "inferred_weak"

    context_agents = [agent for agent in str(parse_row.get("agents_in_context", "")).split(";") if agent]
    if len(context_agents) == 1:
        return context_agents[0], "inferred_weak"

    return "UNKNOWN", "unknown"


def overlaps_window(chunk: MemoryChunk, day: int, start: float, end: float, margin: float) -> bool:
    if chunk.day != day:
        return False
    return chunk.start_time <= end + margin and chunk.end_time >= start - margin


def is_before_context(chunk: MemoryChunk, day: int, start: float) -> bool:
    if chunk.day < day:
        return True
    if chunk.day == day and chunk.end_time <= start:
        return True
    return False


def pool_chunks(
    memory: list[MemoryChunk],
    agents: set[str],
    day: int,
    start: float,
    end: float,
    query_user: str,
) -> dict[str, list[MemoryChunk]]:
    active_agents = agents or set(AGENT_ORDER)
    base_current = [
        chunk for chunk in memory if chunk.source_agent in active_agents and overlaps_window(chunk, day, start, end, CURRENT_MARGIN_SECONDS)
    ]
    base_history = [chunk for chunk in memory if chunk.source_agent in active_agents and is_before_context(chunk, day, start)]
    pools = {
        "all_agents_current": base_current,
        "all_agents_history": base_history,
    }
    if query_user != "UNKNOWN":
        pools["self_current"] = [chunk for chunk in base_current if chunk.source_agent == query_user]
        pools["self_history"] = [chunk for chunk in base_history if chunk.source_agent == query_user]
        pools["external_current"] = [chunk for chunk in base_current if chunk.source_agent != query_user]
        pools["external_history"] = [chunk for chunk in base_history if chunk.source_agent != query_user]
    return pools


def trunc(text: str, limit: int = 260) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def mode_summary(rows: list[dict[str, Any]], mode: str, limit: int = 2) -> str:
    selected = [row for row in rows if row["retrieval_mode"] == mode and int(row["rank"]) <= limit]
    parts = []
    for row in selected:
        parts.append(
            f"{row['source_agent']} D{row['day']} {row['start_time']}-{row['end_time']}: {trunc(row['caption_text'], 140)}"
        )
    return " || ".join(parts)


def positive_top_score(rows: list[dict[str, Any]], mode: str) -> float:
    scores = [float(row["score"]) for row in rows if row["retrieval_mode"] == mode]
    return max(scores) if scores else 0.0


def likely_case_type(rows: list[dict[str, Any]], query_user: str, question: str) -> tuple[str, str]:
    lower_q = question.lower()
    historical_trigger = any(term in lower_q for term in HISTORY_TERMS)
    if query_user == "UNKNOWN":
        if positive_top_score(rows, "all_agents_history") > 0 and historical_trigger:
            return "needs_manual_review", "possible"
        return "needs_manual_review", "no_signal"

    scores = {
        "self_current": positive_top_score(rows, "self_current"),
        "self_history": positive_top_score(rows, "self_history"),
        "external_current": positive_top_score(rows, "external_current"),
        "external_history": positive_top_score(rows, "external_history"),
    }
    best_mode = max(scores, key=scores.get)
    if scores[best_mode] <= 0:
        return "needs_manual_review", "no_signal"
    if historical_trigger and best_mode == "self_history":
        return "self_history_needed_candidate", "possible"
    if historical_trigger and best_mode == "external_history":
        return "external_history_needed_candidate", "possible"
    if best_mode == "self_current":
        return "self_current_sufficient_candidate", "unknown"
    if best_mode == "external_current":
        return "external_current_needed_candidate", "unknown"
    return "needs_manual_review", "possible" if "history" in best_mode else "unknown"


def build_outputs() -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    qas = load_json(QA_PATH)
    if not isinstance(qas, list):
        raise RuntimeError(f"Expected list in {QA_PATH}")

    inventory_rows = []
    context_rows = []
    for question_id, qa in enumerate(qas):
        inventory_rows.append(
            {
                "question_id": question_id,
                "category": qa.get("category", ""),
                "subcategory": qa.get("subcategory", ""),
                "question": qa.get("question", ""),
                "options_json": json_dumps(qa.get("options", [])),
                "answer": qa.get("answer", ""),
                "contexts_json": json_dumps(qa.get("contexts", {})),
            }
        )
        context_rows.append(parse_contexts(question_id, qa))

    write_csv(
        OUTPUT_DIR / "maegoqa_question_inventory.csv",
        inventory_rows,
        ["question_id", "category", "subcategory", "question", "options_json", "answer", "contexts_json"],
    )
    write_csv(
        OUTPUT_DIR / "maegoqa_context_parse.csv",
        context_rows,
        [
            "question_id",
            "parse_status",
            "context_type",
            "parsed_context_count",
            "context_day",
            "context_start",
            "context_end",
            "context_start_hms",
            "context_end_hms",
            "agents_in_context",
            "agents_count",
            "parsed_contexts_json",
            "raw_contexts_json",
        ],
    )

    memory = build_caption_memory_index()
    memory_rows = [
        {
            "memory_id": chunk.memory_id,
            "source_agent": chunk.source_agent,
            "day": chunk.day,
            "start_time": seconds_to_hms(chunk.start_time),
            "end_time": seconds_to_hms(chunk.end_time),
            "start_seconds": f"{chunk.start_time:.2f}",
            "end_seconds": f"{chunk.end_time:.2f}",
            "granularity": chunk.granularity,
            "caption_text": chunk.caption_text,
            "raw_key": chunk.raw_key,
            "source_file": chunk.source_file,
        }
        for chunk in memory
    ]
    write_csv(
        OUTPUT_DIR / "source_isolated_caption_memory_index.csv",
        memory_rows,
        [
            "memory_id",
            "source_agent",
            "day",
            "start_time",
            "end_time",
            "start_seconds",
            "end_seconds",
            "granularity",
            "caption_text",
            "raw_key",
            "source_file",
        ],
    )

    selected = select_candidate_questions(qas, context_rows)
    selected_rows = []
    retrieval_rows = []
    audit_rows = []
    report_case_summaries = []
    history_nonempty_count = 0
    query_status_counter: Counter[str] = Counter()
    history_signal_count = 0
    self_history_signal_count = 0
    external_history_signal_count = 0

    for case_idx, item in enumerate(selected, start=1):
        qa = item["qa"]
        parse_row = item["parse"]
        question_id = int(item["question_id"])
        case_id = f"HISTV1_{case_idx:03d}"
        context_day = int(parse_row["context_day"])
        context_start = float(parse_row["context_start"])
        context_end = float(parse_row["context_end"])
        agents = {agent for agent in str(parse_row["agents_in_context"]).split(";") if agent}
        query_user, query_user_status = infer_query_user(qa, parse_row)
        query_status_counter[query_user_status] += 1

        pools = pool_chunks(memory, agents, context_day, context_start, context_end, query_user)
        if pools.get("all_agents_history"):
            history_nonempty_count += 1

        selected_rows.append(
            {
                "case_id": case_id,
                "question_id": question_id,
                "category": qa.get("category", ""),
                "subcategory": qa.get("subcategory", ""),
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "options_json": json_dumps(qa.get("options", [])),
                "context_day": context_day,
                "context_start": seconds_to_hms(context_start),
                "context_end": seconds_to_hms(context_end),
                "agents_in_context": ";".join(sorted(agents, key=AGENT_ORDER.index)),
                "selection_score": item["selection_score"],
                "query_user": query_user,
                "query_user_status": query_user_status,
                "needs_human_review": "yes",
                "notes": "query_user is provisional; candidate selected by text heuristics only",
            }
        )

        case_retrieval_rows = []
        for mode in [
            "all_agents_current",
            "all_agents_history",
            "self_current",
            "self_history",
            "external_current",
            "external_history",
        ]:
            if mode not in pools:
                continue
            ranked = bm25_rank(str(qa.get("question", "")), pools[mode], top_k=TOP_K)
            for rank, (chunk, score) in enumerate(ranked, start=1):
                is_current = "current" in mode
                is_history = "history" in mode
                is_self = mode.startswith("self")
                is_external = mode.startswith("external")
                row = {
                    "case_id": case_id,
                    "question_id": question_id,
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "category": qa.get("category", ""),
                    "query_user": query_user,
                    "query_user_status": query_user_status,
                    "retrieval_mode": mode,
                    "source_agent": chunk.source_agent,
                    "day": chunk.day,
                    "start_time": seconds_to_hms(chunk.start_time),
                    "end_time": seconds_to_hms(chunk.end_time),
                    "granularity": chunk.granularity,
                    "rank": rank,
                    "score": f"{score:.4f}",
                    "caption_text": chunk.caption_text,
                    "raw_key": chunk.raw_key,
                    "is_current": "yes" if is_current else "no",
                    "is_history": "yes" if is_history else "no",
                    "is_self": "yes" if is_self else "no",
                    "is_external": "yes" if is_external else "no",
                    "needs_human_review": "yes",
                }
                retrieval_rows.append(row)
                case_retrieval_rows.append(row)

        if positive_top_score(case_retrieval_rows, "all_agents_history") > 0:
            history_signal_count += 1
        if positive_top_score(case_retrieval_rows, "self_history") > 0:
            self_history_signal_count += 1
        if positive_top_score(case_retrieval_rows, "external_history") > 0:
            external_history_signal_count += 1

        likely_type, historical_helpful = likely_case_type(case_retrieval_rows, query_user, str(qa.get("question", "")))
        audit_row = {
            "case_id": case_id,
            "question_id": question_id,
            "question": qa.get("question", ""),
            "answer": qa.get("answer", ""),
            "category": qa.get("category", ""),
            "context_day": context_day,
            "context_start": seconds_to_hms(context_start),
            "context_end": seconds_to_hms(context_end),
            "agents_in_context": ";".join(sorted(agents, key=AGENT_ORDER.index)),
            "query_user": query_user,
            "query_user_status": query_user_status,
            "top_self_current_summary": mode_summary(case_retrieval_rows, "self_current"),
            "top_self_history_summary": mode_summary(case_retrieval_rows, "self_history"),
            "top_external_current_summary": mode_summary(case_retrieval_rows, "external_current"),
            "top_external_history_summary": mode_summary(case_retrieval_rows, "external_history"),
            "candidate_historical_helpful": historical_helpful,
            "likely_case_type": likely_type,
            "notes": "Heuristic retrieval signal only; do not treat as answer accuracy or final source label.",
            "needs_human_review": "yes",
        }
        audit_rows.append(audit_row)
        report_case_summaries.append((audit_row, case_retrieval_rows))

    write_csv(
        OUTPUT_DIR / "selected_historical_candidate_questions.csv",
        selected_rows,
        [
            "case_id",
            "question_id",
            "category",
            "subcategory",
            "question",
            "answer",
            "options_json",
            "context_day",
            "context_start",
            "context_end",
            "agents_in_context",
            "selection_score",
            "query_user",
            "query_user_status",
            "needs_human_review",
            "notes",
        ],
    )
    write_csv(
        OUTPUT_DIR / "historical_retrieval_candidates.csv",
        retrieval_rows,
        [
            "case_id",
            "question_id",
            "question",
            "answer",
            "category",
            "query_user",
            "query_user_status",
            "retrieval_mode",
            "source_agent",
            "day",
            "start_time",
            "end_time",
            "granularity",
            "rank",
            "score",
            "caption_text",
            "raw_key",
            "is_current",
            "is_history",
            "is_self",
            "is_external",
            "needs_human_review",
        ],
    )
    write_csv(
        OUTPUT_DIR / "historical_memory_candidate_audit_table.csv",
        audit_rows,
        [
            "case_id",
            "question_id",
            "question",
            "answer",
            "category",
            "context_day",
            "context_start",
            "context_end",
            "agents_in_context",
            "query_user",
            "query_user_status",
            "top_self_current_summary",
            "top_self_history_summary",
            "top_external_current_summary",
            "top_external_history_summary",
            "candidate_historical_helpful",
            "likely_case_type",
            "notes",
            "needs_human_review",
        ],
    )

    write_html_report(report_case_summaries)

    parseable_count = sum(1 for row in context_rows if row["parse_status"] == "ok")
    category_counts = Counter(row["category"] for row in selected_rows)
    promising = choose_promising_cases(audit_rows, report_case_summaries)
    write_markdown_report(
        qas=qas,
        context_rows=context_rows,
        memory=memory,
        selected_rows=selected_rows,
        retrieval_rows=retrieval_rows,
        audit_rows=audit_rows,
        query_status_counter=query_status_counter,
        history_nonempty_count=history_nonempty_count,
        history_signal_count=history_signal_count,
        self_history_signal_count=self_history_signal_count,
        external_history_signal_count=external_history_signal_count,
        promising=promising,
        category_counts=category_counts,
    )

    return {
        "output_dir": str(OUTPUT_DIR),
        "questions_loaded": len(qas),
        "parseable_contexts": parseable_count,
        "memory_chunks": len(memory),
        "selected_candidates": len(selected_rows),
        "query_status_counts": dict(query_status_counter),
        "historical_pool_nonempty": history_nonempty_count,
        "history_signal_count": history_signal_count,
        "self_history_signal_count": self_history_signal_count,
        "external_history_signal_count": external_history_signal_count,
        "promising": promising,
    }


def choose_promising_cases(
    audit_rows: list[dict[str, Any]], report_case_summaries: list[tuple[dict[str, Any], list[dict[str, Any]]]]
) -> list[dict[str, str]]:
    rows_by_case = {audit["case_id"]: retrieval_rows for audit, retrieval_rows in report_case_summaries}

    def rank(audit: dict[str, Any]) -> tuple[int, float]:
        rows = rows_by_case.get(audit["case_id"], [])
        hist_score = max(
            positive_top_score(rows, "self_history"),
            positive_top_score(rows, "external_history"),
            positive_top_score(rows, "all_agents_history"),
        )
        has_hist_label = 1 if audit["candidate_historical_helpful"] == "possible" else 0
        known_query = 1 if audit["query_user"] != "UNKNOWN" else 0
        return (has_hist_label + known_query, hist_score)

    ranked = sorted(audit_rows, key=rank, reverse=True)[:5]
    summaries = []
    for audit in ranked:
        summaries.append(
            {
                "case_id": audit["case_id"],
                "question_id": str(audit["question_id"]),
                "question": trunc(audit["question"], 160),
                "category": audit["category"],
                "query_user": audit["query_user"],
                "likely_case_type": audit["likely_case_type"],
                "why_promising": trunc(
                    audit["top_self_history_summary"]
                    or audit["top_external_history_summary"]
                    or audit["top_external_current_summary"]
                    or audit["top_self_current_summary"]
                    or "no retrieved summary",
                    260,
                ),
            }
        )
    return summaries


def write_html_report(report_case_summaries: list[tuple[dict[str, Any], list[dict[str, Any]]]]) -> None:
    path = OUTPUT_DIR / "historical_memory_candidate_gallery.html"
    sections = []
    for audit, retrieval_rows in report_case_summaries:
        mode_blocks = []
        for mode in [
            "all_agents_current",
            "all_agents_history",
            "self_current",
            "self_history",
            "external_current",
            "external_history",
        ]:
            rows = [row for row in retrieval_rows if row["retrieval_mode"] == mode]
            if not rows:
                continue
            items = []
            for row in rows:
                items.append(
                    "<li>"
                    f"<strong>#{html.escape(str(row['rank']))} "
                    f"{html.escape(row['source_agent'])} D{html.escape(str(row['day']))} "
                    f"{html.escape(row['start_time'])}-{html.escape(row['end_time'])} "
                    f"{html.escape(row['granularity'])} score={html.escape(row['score'])}</strong>"
                    f"<div>{html.escape(trunc(row['caption_text'], 600))}</div>"
                    "</li>"
                )
            mode_blocks.append(f"<h4>{html.escape(mode)}</h4><ol>{''.join(items)}</ol>")
        sections.append(
            "<section>"
            f"<h2>{html.escape(audit['case_id'])} / Q{html.escape(str(audit['question_id']))}</h2>"
            f"<p><strong>Warning:</strong> retrieval and labels are provisional human-audit signals only. "
            f"Do not treat them as answer accuracy or ground-truth source labels.</p>"
            f"<p><strong>Question:</strong> {html.escape(str(audit['question']))}</p>"
            f"<p><strong>Answer:</strong> {html.escape(str(audit['answer']))}</p>"
            f"<p><strong>Category:</strong> {html.escape(str(audit['category']))}</p>"
            f"<p><strong>Parsed context:</strong> DAY{html.escape(str(audit['context_day']))} "
            f"{html.escape(str(audit['context_start']))}-{html.escape(str(audit['context_end']))}; "
            f"agents={html.escape(str(audit['agents_in_context']))}</p>"
            f"<p><strong>Provisional query_user:</strong> {html.escape(str(audit['query_user']))} "
            f"({html.escape(str(audit['query_user_status']))}); needs_human_review=yes</p>"
            f"{''.join(mode_blocks)}"
            "</section>"
        )
    page = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Historical Memory Candidate Gallery</title>
  <style>
    body { font-family: Arial, sans-serif; line-height: 1.45; margin: 24px; max-width: 1200px; }
    section { border-top: 1px solid #ccc; padding: 18px 0; }
    h1, h2, h3, h4 { margin-bottom: 6px; }
    li { margin-bottom: 10px; }
    strong { color: #222; }
  </style>
</head>
<body>
<h1>Historical Memory Candidate Gallery</h1>
<p>This is a text-only, human-audit report. No videos, VLMs, or LLMs were used. Labels are not final.</p>
""" + "\n".join(sections) + "\n</body>\n</html>\n"
    path.write_text(page, encoding="utf-8")


def write_markdown_report(
    *,
    qas: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
    memory: list[MemoryChunk],
    selected_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    query_status_counter: Counter[str],
    history_nonempty_count: int,
    history_signal_count: int,
    self_history_signal_count: int,
    external_history_signal_count: int,
    promising: list[dict[str, str]],
    category_counts: Counter[str],
) -> None:
    parseable_count = sum(1 for row in context_rows if row["parse_status"] == "ok")
    selected_count = len(selected_rows)
    unknown_count = query_status_counter.get("unknown", 0)
    inferred_count = query_status_counter.get("inferred_weak", 0)
    explicit_count = query_status_counter.get("explicit", 0)
    retrieval_by_mode = Counter(row["retrieval_mode"] for row in retrieval_rows)

    lines = [
        "# Historical Memory Feasibility Report V1",
        "",
        "Scope: text/caption-only candidate generation for human audit. No videos were downloaded and no VLM/LLM was run.",
        "",
        "## Required Questions",
        "",
        f"1. MA-EgoQA questions loaded: {len(qas)}.",
        f"2. Contexts parseable: {parseable_count}/{len(context_rows)}.",
        f"3. Source-isolated caption memory chunks indexed: {len(memory)}.",
        f"4. Candidate questions selected: {selected_count}. Category mix: {dict(category_counts)}.",
        f"5. query_user status: unknown={unknown_count}, inferred_weak={inferred_count}, explicit={explicit_count}. All selected cases have needs_human_review=yes.",
        f"6. historical_pool non-empty: {history_nonempty_count}/{selected_count}.",
        (
            "7. Historical retrieval signal: "
            f"all_agents_history positive in {history_signal_count}/{selected_count}; "
            f"self_history positive in {self_history_signal_count}/{selected_count}; "
            f"external_history positive in {external_history_signal_count}/{selected_count}. "
            "These are lexical retrieval signals, not answer accuracy."
        ),
        "8. Five promising cases for manual audit:",
        "",
    ]
    for case in promising:
        lines.append(
            f"- {case['case_id']} Q{case['question_id']} [{case['category']}] "
            f"query_user={case['query_user']} likely={case['likely_case_type']}: {case['question']} "
            f"Signal: {case['why_promising']}"
        )

    lines.extend(
        [
            "",
            "9. Main blockers:",
            "",
            "- query_user missing: yes. Most or all query users are unknown or weakly inferred; this blocks final self/external claims.",
            "- context parsing hard: partly. Object-style MA-EgoQA contexts parse cleanly, but timestamp-list contexts include malformed timestamps and mixed natural-language evidence strings.",
            "- captions too coarse: partly. 30s captions are useful; 10min captions help history but can blur source/event boundaries.",
            "- no frames: yes. This run is text-only; no local EgoLife clips/frames were found in the inventory.",
            "- lack of human labels: yes. All case types are provisional and need manual audit.",
            "",
            "10. Recommended next step:",
            "",
            "Manually audit the selected 30 cases, first checking whether the provisional query_user is meaningful. Then label only the evidence availability columns: self_current, self_history, external_current, external_history. Do not run models until a smaller verified case set exists.",
            "",
            "## Retrieval Rows By Mode",
            "",
        ]
    )
    for mode, count in sorted(retrieval_by_mode.items()):
        lines.append(f"- {mode}: {count}")

    (OUTPUT_DIR / "historical_memory_feasibility_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = build_outputs()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
