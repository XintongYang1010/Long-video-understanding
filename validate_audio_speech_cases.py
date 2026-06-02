#!/usr/bin/env python3
"""Validate audio/speech source-access cases using local transcripts first.

This script does not decode video. It searches local transcript JSON files if
they are mounted; otherwise it uses the CASTLE inventory to identify candidate
transcript paths and marks cases as candidate-only.
"""

from __future__ import annotations

import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SEED_CSV = ROOT / "source_access_seed_cases_v0_3.csv"
INVENTORY_CSV = ROOT / "castle_modality_inventory.csv"

OUT_CSV = ROOT / "audio_validation_v0_3.csv"
OUT_MD = ROOT / "audio_validation_snippets.md"

COLUMNS = [
    "case_id",
    "question",
    "answer_options",
    "search_terms",
    "transcript_hit",
    "source_path",
    "timestamp_or_window",
    "speaker_if_available",
    "matched_snippet",
    "answer_supported",
    "verified_route",
    "status",
    "confidence",
    "next_action",
]

PEOPLE = {
    "allie",
    "bao",
    "bjorn",
    "cathal",
    "florian",
    "klaus",
    "linh",
    "luca",
    "onanong",
    "ononang",
    "stevan",
    "tien",
    "werner",
}

ROOM_HINTS = {
    "kitchen": ["Kitchen"],
    "baking": ["Kitchen"],
    "ginger": ["Kitchen"],
    "cameras": ["Meeting"],
    "camera": ["Meeting"],
    "power outage": ["Meeting"],
    "quiz": ["Meeting", "Living1", "Living2"],
}

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "what",
    "who",
    "how",
    "many",
    "was",
    "were",
    "did",
    "does",
    "when",
    "where",
    "which",
    "that",
    "this",
    "from",
    "into",
    "onto",
    "over",
    "under",
    "there",
    "their",
    "during",
    "while",
    "first",
    "second",
    "third",
    "round",
    "day",
    "morning",
    "answer",
    "options",
    "him",
    "her",
    "his",
    "she",
    "they",
    "them",
    "are",
    "had",
    "who",
    "what",
    "did",
    "was",
    "were",
    "from",
    "according",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def terms(text: str) -> list[str]:
    out: list[str] = []
    for token in re.findall(r"[a-z0-9]+", norm(text)):
        keep_short = token.isdigit() or token in {"km", "ai", "sd", "qr"}
        if (len(token) >= 3 or keep_short) and token not in STOPWORDS and token not in out:
            out.append(token)
    return out


def parse_answer_terms(answer_options: str) -> list[str]:
    # Handles "a: Foo | b: Bar" style options and keeps meaningful option words.
    cleaned = re.sub(r"\b[a-d]:", " ", answer_options.lower())
    return terms(cleaned)


def people_in(text: str) -> list[str]:
    found = []
    token_set = set(terms(text))
    for person in PEOPLE:
        if person in token_set or person in norm(text):
            canonical = "onanong" if person == "ononang" else person
            if canonical not in found:
                found.append(canonical)
    return sorted(found)


def day_hint(text: str) -> str | None:
    t = norm(text)
    if "day 1" in t or "first day" in t:
        return "day1"
    if "day 2" in t or "second day" in t:
        return "day2"
    if "day 3" in t or "third day" in t:
        return "day3"
    if "day 4" in t or "fourth day" in t:
        return "day4"
    return None


def room_hints(text: str) -> list[str]:
    t = norm(text)
    rooms: list[str] = []
    for key, vals in ROOM_HINTS.items():
        if key in t:
            for val in vals:
                if val not in rooms:
                    rooms.append(val)
    return rooms


def timestamp_hint(text: str, candidate_paths: list[str]) -> str:
    day = day_hint(text)
    pieces: list[str] = []
    if day:
        pieces.append(day.upper())
    if "morning" in norm(text):
        pieces.append("morning; prioritize hourly transcript files around 09-12")
    hours = []
    for p in candidate_paths:
        m = re.search(r"/transcript/(\d{2})\.json$", p)
        if m:
            hours.append(m.group(1))
    if hours:
        unique_hours = sorted(set(hours), key=hours.index)
        pieces.append("candidate transcript hours: " + ",".join(unique_hours[:8]))
    return "; ".join(pieces) if pieces else "unknown; needs event/window lookup"


def local_transcript_files() -> list[Path]:
    files: list[Path] = []
    for base in [ROOT / "main", ROOT / "castle_poc" / "main", ROOT / "castle_hpc" / "main"]:
        if base.exists():
            files.extend(base.glob("**/transcript/*.json"))
    # Avoid recursive cache traversal; allow explicit top-level transcript JSON.
    files.extend(p for p in ROOT.glob("*transcript*.json") if p.is_file())
    return sorted(set(files))


def flatten_json_text(obj: Any) -> list[dict[str, str]]:
    """Return transcript-like entries with text/time/speaker if detectable."""
    entries: list[dict[str, str]] = []

    def visit(x: Any, inherited: dict[str, str] | None = None) -> None:
        inherited = inherited or {}
        if isinstance(x, dict):
            current = dict(inherited)
            for key in ["speaker", "speaker_id", "name", "agent", "participant"]:
                if key in x and isinstance(x[key], (str, int, float)):
                    current["speaker"] = str(x[key])
                    break
            for key in ["timestamp", "time", "start", "start_time", "end", "end_time", "window"]:
                if key in x and isinstance(x[key], (str, int, float)):
                    current.setdefault("timestamp", str(x[key]))
            text_value = None
            for key in ["text", "transcript", "utterance", "caption", "sentence"]:
                if key in x and isinstance(x[key], str):
                    text_value = x[key]
                    break
            if text_value:
                entries.append(
                    {
                        "text": text_value,
                        "speaker": current.get("speaker", ""),
                        "timestamp": current.get("timestamp", ""),
                    }
                )
            for value in x.values():
                visit(value, current)
        elif isinstance(x, list):
            for item in x:
                visit(item, inherited)
        elif isinstance(x, str):
            if len(x.strip()) > 10:
                entries.append({"text": x, "speaker": inherited.get("speaker", ""), "timestamp": inherited.get("timestamp", "")})

    visit(obj)
    return entries


def score_text(text: str, query_terms: list[str], answer_terms: list[str]) -> tuple[int, float, list[str]]:
    text_l = norm(text)
    exact = [term for term in query_terms + answer_terms if term in text_l]
    # Fuzzy score over individual terms; cheap and deterministic.
    fuzzy_hits = []
    text_words = set(re.findall(r"[a-z0-9]+", text_l))
    for term in query_terms + answer_terms:
        if term in exact:
            continue
        best = max((SequenceMatcher(None, term, w).ratio() for w in text_words), default=0.0)
        if best >= 0.84:
            fuzzy_hits.append(term)
    score = len(exact) * 3 + len(fuzzy_hits)
    answer_overlap = sum(1 for term in answer_terms if term in text_l) / max(1, len(answer_terms))
    return score, answer_overlap, exact + fuzzy_hits


def search_local_transcripts(
    files: list[Path],
    query_terms: list[str],
    answer_terms: list[str],
) -> dict[str, str]:
    best: dict[str, Any] | None = None
    for path in files:
        try:
            obj = json.loads(path.read_text(errors="ignore"))
            entries = flatten_json_text(obj)
        except Exception:
            text = path.read_text(errors="ignore")
            entries = [{"text": text, "speaker": "", "timestamp": ""}]
        for entry in entries:
            score, answer_overlap, matched = score_text(entry["text"], query_terms, answer_terms)
            if score <= 0:
                continue
            current = {
                "score": score,
                "answer_overlap": answer_overlap,
                "source_path": str(path.relative_to(ROOT)),
                "timestamp": entry.get("timestamp", ""),
                "speaker": entry.get("speaker", ""),
                "snippet": make_snippet(entry["text"], matched),
                "matched_terms": matched,
            }
            if best is None or (current["score"], current["answer_overlap"]) > (best["score"], best["answer_overlap"]):
                best = current
    if best is None:
        return {}
    return {
        "source_path": best["source_path"],
        "timestamp": best["timestamp"],
        "speaker": best["speaker"],
        "snippet": best["snippet"],
        "matched_terms": ", ".join(best["matched_terms"]),
        "answer_supported": "yes" if best["answer_overlap"] >= 0.25 else "ambiguous",
        "score": str(best["score"]),
    }


def make_snippet(text: str, matched_terms: list[str]) -> str:
    text_norm = re.sub(r"\s+", " ", text).strip()
    lower = text_norm.lower()
    pos = -1
    for term in matched_terms:
        pos = lower.find(term.lower())
        if pos >= 0:
            break
    if pos < 0:
        return text_norm[:300]
    return text_norm[max(0, pos - 120) : min(len(text_norm), pos + 220)]


def inventory_transcripts(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        r
        for r in rows
        if r.get("inferred_modality") == "transcript_audio" and not r.get("path", "").endswith(".sha256")
    ]


def inventory_score(row: dict[str, str], names: list[str], rooms: list[str], day: str | None, question: str) -> int:
    p = row.get("path", "").lower()
    meta = row.get("participant_or_camera_id_if_parseable", "").lower()
    score = 0
    for name in names:
        if f"/{name}/" in p or f"participant={name}" in meta:
            score += 5
    for room in rooms:
        room_l = room.lower()
        if f"/{room_l}/" in p or f"participant={room_l}" in meta:
            score += 4
    if day and f"/{day}/" in p:
        score += 3
    q = norm(question)
    if "morning" in q and re.search(r"/transcript/(09|10|11|12)\.json$", p):
        score += 2
    if not names and not rooms and not day:
        score = 1
    return score


def candidate_inventory_paths(
    transcript_rows: list[dict[str, str]],
    question: str,
    limit: int = 12,
) -> list[str]:
    names = people_in(question)
    rooms = room_hints(question)
    day = day_hint(question)
    scored = []
    for row in transcript_rows:
        score = inventory_score(row, names, rooms, day, question)
        if score > 0:
            scored.append((score, row["path"]))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out: list[str] = []
    for _, path in scored:
        if path not in out:
            out.append(path)
        if len(out) >= limit:
            break
    return out


def selected_audio_cases(seed_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in seed_rows:
        bucket = row.get("bucket") or row.get("primary_category", "")
        route = row.get("likely_minimal_evidence_route", "")
        if bucket == "audio_speech" or "self_audio" in route or "external_user_audio" in route:
            out.append(row)
    return out


def validate_case(
    case: dict[str, str],
    transcript_files: list[Path],
    transcript_rows: list[dict[str, str]],
) -> dict[str, str]:
    question = case["question"]
    answer_options = case["answer_options"]
    query_terms = terms(question)
    option_terms = parse_answer_terms(answer_options)
    search_terms = ";".join(query_terms + [t for t in option_terms if t not in query_terms])
    candidate_paths = candidate_inventory_paths(transcript_rows, question)

    hit = search_local_transcripts(transcript_files, query_terms, option_terms) if transcript_files else {}
    if hit:
        answer_supported = hit["answer_supported"]
        status = "verified" if answer_supported == "yes" else "candidate_only"
        confidence = "medium" if answer_supported == "yes" else "low"
        next_action = (
            "Frames not needed for this pass; manually confirm the matched transcript window."
            if answer_supported == "yes"
            else "Matched transcript is ambiguous; inspect neighboring transcript lines before considering a short audio/video clip."
        )
        return {
            "case_id": case["case_id"],
            "question": question,
            "answer_options": answer_options,
            "search_terms": search_terms,
            "transcript_hit": "1",
            "source_path": hit.get("source_path", ""),
            "timestamp_or_window": hit.get("timestamp") or timestamp_hint(question, [hit.get("source_path", "")]),
            "speaker_if_available": hit.get("speaker", ""),
            "matched_snippet": hit.get("snippet", ""),
            "answer_supported": answer_supported,
            "verified_route": "self_audio",
            "status": status,
            "confidence": confidence,
            "next_action": next_action,
        }

    missing_reason = "No local transcript JSON content files were found; inventory transcript paths only."
    if transcript_files:
        missing_reason = "Local transcript files were searched, but no exact/fuzzy keyword match was found."
    return {
        "case_id": case["case_id"],
        "question": question,
        "answer_options": answer_options,
        "search_terms": search_terms,
        "transcript_hit": "0",
        "source_path": "; ".join(candidate_paths),
        "timestamp_or_window": timestamp_hint(question, candidate_paths),
        "speaker_if_available": "",
        "matched_snippet": missing_reason,
        "answer_supported": "unknown",
        "verified_route": "self_audio",
        "status": "candidate_only",
        "confidence": "low",
        "next_action": (
            "Locate or mount the listed transcript JSON files and search them first. "
            "Do not claim external audio is needed yet. If transcript remains missing or ambiguous, use a short_clip only for the narrow event window."
        ),
    }


def write_markdown(rows: list[dict[str, str]], transcript_file_count: int) -> None:
    lines = [
        "# Audio/Speech Validation v0.3",
        "",
        "Transcript-first validation for source-access seed cases. No video decoding, frame extraction, or VLM inference was used.",
        "",
        f"- Audio-route cases selected: {len(rows)}",
        f"- Local transcript JSON content files searched: {transcript_file_count}",
        f"- Transcript hits: {sum(int(r['transcript_hit']) for r in rows)}",
        f"- Direct answer support from transcript: {sum(1 for r in rows if r['answer_supported'] == 'yes')}",
        "",
        "Important rule applied: external audio is not marked as needed unless self/local transcript is unavailable or insufficient after inspection. In this pass, local transcript content was unavailable, so all cases remain candidate-only.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['case_id']}",
                "",
                f"- Question: {row['question']}",
                f"- Answer options: {row['answer_options']}",
                f"- Search terms: `{row['search_terms']}`",
                f"- Transcript hit: {row['transcript_hit']}",
                f"- Candidate/source path: `{row['source_path']}`",
                f"- Timestamp/window: {row['timestamp_or_window']}",
                f"- Speaker: {row['speaker_if_available'] or 'not available'}",
                f"- Answer supported: {row['answer_supported']}",
                f"- Verified route: {row['verified_route']}",
                f"- Status: {row['status']}",
                f"- Confidence: {row['confidence']}",
                f"- Snippet/status: {row['matched_snippet']}",
                f"- Next action: {row['next_action']}",
                "",
            ]
        )
    OUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    seed_rows = read_csv(SEED_CSV)
    inventory_rows = read_csv(INVENTORY_CSV)
    transcript_rows = inventory_transcripts(inventory_rows)
    transcript_files = local_transcript_files()
    audio_cases = selected_audio_cases(seed_rows)
    rows = [validate_case(case, transcript_files, transcript_rows) for case in audio_cases]
    write_csv(OUT_CSV, rows, COLUMNS)
    write_markdown(rows, len(transcript_files))
    print("Generated audio validation outputs.")
    print(f"Audio-route cases: {len(audio_cases)}")
    print(f"Inventory transcript paths: {len(transcript_rows)}")
    print(f"Local transcript files searched: {len(transcript_files)}")
    print(f"Transcript hits: {sum(int(r['transcript_hit']) for r in rows)}")


if __name__ == "__main__":
    main()
