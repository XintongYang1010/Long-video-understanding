#!/usr/bin/env python3
"""
Build targeted visual audit packets for Dataset V0.1.

This script is intentionally conservative:
- no VLM/LLM calls
- no full-video downloads
- no edits to Dataset V0.1 source files
- only targeted frame extraction when a local frame/video, or an explicit remote
  URL with ffmpeg support, is available

When frames are unavailable, it still builds a caption-only audit packet with
explicit missing-source status so the failure mode is visible.
"""

from __future__ import annotations

import argparse
import binascii
import csv
import html
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import textwrap
import unicodedata
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "outputs" / "historical_v2_fullscreen" / "dataset_v0_1"
DEFAULT_OUTPUT_DIR = DATASET_DIR / "visual_audit_v1"

DEMO_CSV = DATASET_DIR / "dataset_v0_1_demo.csv"
EVAL_LITE_CSV = DATASET_DIR / "dataset_v0_1_eval_lite.csv"
CONTROL_CSV = DATASET_DIR / "dataset_v0_1_control.csv"
MODEL_INPUTS_JSONL = DATASET_DIR / "dataset_v0_1_model_inputs.jsonl"
TOPK_EVIDENCE_CSV = ROOT / "outputs" / "historical_v2_fullscreen" / "evidence_scope_fullscreen_v2_topk_evidence.csv"
MEMORY_INDEX_CSV = ROOT / "outputs" / "historical_v1" / "source_isolated_caption_memory_index.csv"

CASE_SUBSET_FIELDS = [
    "visual_case_id",
    "source_split",
    "dataset_case_id",
    "question_id",
    "question",
    "answer",
    "category",
    "expected_result",
    "current_only_answerability",
    "current_plus_history_answerability",
    "current_plus_history_gain",
    "reason_selected_for_visual_audit",
]

FRAME_PLAN_FIELDS = [
    "visual_case_id",
    "dataset_case_id",
    "question_id",
    "evidence_scope",
    "source_agent",
    "day",
    "start_time",
    "end_time",
    "granularity",
    "caption_text",
    "target_frame_time",
    "raw_key",
    "video_path_or_url",
    "extraction_status_planned",
    "reason",
]

AUDIT_TABLE_FIELDS = [
    "visual_case_id",
    "dataset_case_id",
    "question_id",
    "split",
    "question",
    "answer",
    "category",
    "current_only_caption",
    "historical_caption",
    "contact_sheet_path",
    "extracted_frame_paths",
    "extraction_status",
    "qa_answer_visually_plausible",
    "caption_matches_frames",
    "current_only_enough",
    "history_adds_value",
    "keep_for_ppt",
    "notes",
]

EVIDENCE_RE = re.compile(
    r"\[(?P<scope>current_only|history_only|current_plus_historical)\s+"
    r"rank=(?P<rank>\d+)\s+score=(?P<score>[-+]?\d+(?:\.\d+)?)\s+"
    r"source=(?P<agent>[A-Za-z]+)\s+D(?P<day>\d+)\s+"
    r"(?P<start>[0-9:.]+)-(?P<end>[0-9:.]+)\s+"
    r"(?P<granularity>[A-Za-z0-9]+)\]\s+"
    r"(?P<caption>.*?)(?=\n\[(?:current_only|history_only|current_plus_historical)\s+rank=|\Z)",
    re.S,
)

URL_RE = re.compile(r"^https?://", re.I)
MEDIA_SUFFIXES_VIDEO = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
MEDIA_SUFFIXES_IMAGE = {".jpg", ".jpeg", ".png", ".webp"}
SKIP_DIRS = {".git", "__pycache__", ".cache", "envs", "hf_cache"}


@dataclass
class SelectedCase:
    visual_case_id: str
    source_split: str
    row: dict[str, str]
    reason: str
    contact_sheet_path: str = ""
    extraction_status: str = "not_started"
    extracted_frame_paths: list[str] = field(default_factory=list)

    @property
    def dataset_case_id(self) -> str:
        return self.row.get("case_id", "")

    @property
    def question_id(self) -> str:
        return self.row.get("source_question_id", self.row.get("question_id", ""))


@dataclass
class EvidenceWindow:
    visual_case_id: str
    dataset_case_id: str
    question_id: str
    evidence_scope: str
    source_agent: str
    day: str
    start_time: str
    end_time: str
    start_seconds: float
    end_seconds: float
    granularity: str
    caption_text: str
    rank: int
    score: float
    raw_key: str = ""
    source_file: str = ""
    selection_reason: str = ""


@dataclass
class FrameTarget:
    window: EvidenceWindow
    target_seconds: float
    target_frame_time: str
    target_time_token: str
    output_path: Path
    video_path_or_url: str = ""
    extraction_status_planned: str = "missing_video_source"
    extraction_status_actual: str = "missing_video_source"
    reason: str = ""
    extracted_frame_path: str = ""


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean_csv_value(row.get(field, "")) for field in fieldnames})


def clean_csv_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
    return value


def normalize_ws(text: Any) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())


def trunc(text: Any, limit: int) -> str:
    clean = normalize_ws(text)
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def confidence_rank(value: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(str(value or "").lower(), 0)


def get_question_id(row: dict[str, str]) -> str:
    return row.get("source_question_id", row.get("question_id", ""))


def evidence_pattern(row: dict[str, str]) -> str:
    sources = row.get("evidence_sources", "")
    tokens = [item.strip() for item in sources.split(";") if item.strip()]
    cur_gran = sorted({tok.split(":")[-1] for tok in tokens if tok.startswith("current_only:")})
    hist_gran = sorted({tok.split(":")[-1] for tok in tokens if tok.startswith("history_only:")})
    return "|".join(
        [
            row.get("case_type", ""),
            row.get("expected_result", ""),
            "gain=" + row.get("current_plus_history_gain", ""),
            "cur=" + ",".join(cur_gran),
            "hist=" + ",".join(hist_gran),
        ]
    )


def select_diverse_rows(candidates: list[dict[str, str]], n: int) -> list[dict[str, str]]:
    scored = sorted(
        candidates,
        key=lambda r: (
            -confidence_rank(r.get("confidence", "")),
            0 if r.get("current_plus_history_gain") == "yes" else 1,
            r.get("category", ""),
            parse_int(get_question_id(r), 10**9),
        ),
    )
    chosen: list[dict[str, str]] = []
    used_categories: set[str] = set()
    used_patterns: set[str] = set()

    def pick(predicate: Any) -> None:
        nonlocal chosen
        for row in scored:
            if len(chosen) >= n:
                return
            if row in chosen:
                continue
            if predicate(row):
                chosen.append(row)
                used_categories.add(row.get("category", ""))
                used_patterns.add(evidence_pattern(row))

    pick(lambda r: r.get("category", "") not in used_categories and evidence_pattern(r) not in used_patterns)
    pick(lambda r: r.get("category", "") not in used_categories or evidence_pattern(r) not in used_patterns)
    pick(lambda r: True)
    return chosen[:n]


def select_visual_cases(
    demo_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
    control_rows: list[dict[str, str]],
) -> list[SelectedCase]:
    selected: list[SelectedCase] = []
    selected_case_ids: set[str] = set()

    for idx, row in enumerate(demo_rows, start=1):
        selected.append(
            SelectedCase(
                visual_case_id=f"VA_DEMO_{idx:03d}",
                source_split="demo",
                row=row,
                reason="demo set full inclusion for visual audit",
            )
        )
        selected_case_ids.add(row.get("case_id", ""))

    eval_candidates = [row for row in eval_rows if row.get("case_id", "") not in selected_case_ids]
    eval_selected = select_diverse_rows(eval_candidates, 5)
    if len(eval_selected) < 5:
        fallback = [row for row in eval_rows if row not in eval_selected]
        eval_selected.extend(select_diverse_rows(fallback, 5 - len(eval_selected)))

    for idx, row in enumerate(eval_selected[:5], start=1):
        selected.append(
            SelectedCase(
                visual_case_id=f"VA_EVAL_{idx:03d}",
                source_split="eval_lite",
                row=row,
                reason=(
                    "eval-lite diversity sample: "
                    f"category={row.get('category', '')}; pattern={evidence_pattern(row)}"
                ),
            )
        )
        selected_case_ids.add(row.get("case_id", ""))

    control_candidates = [
        row
        for row in control_rows
        if row.get("case_id", "") not in selected_case_ids
        and row.get("expected_result") == "current_only_sufficient"
        and row.get("current_only_answerability") == "likely_answerable"
    ]
    control_selected = select_diverse_rows(control_candidates, 5)
    if len(control_selected) < 5:
        fallback = [row for row in control_rows if row not in control_selected]
        control_selected.extend(select_diverse_rows(fallback, 5 - len(control_selected)))

    for idx, row in enumerate(control_selected[:5], start=1):
        selected.append(
            SelectedCase(
                visual_case_id=f"VA_CONTROL_{idx:03d}",
                source_split="control",
                row=row,
                reason=(
                    "current-only sufficient control sample: "
                    f"category={row.get('category', '')}; pattern={evidence_pattern(row)}"
                ),
            )
        )

    return selected


def read_model_inputs(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            out[(str(item.get("case_id", "")), str(item.get("evidence_scope", "")))] = item
    return out


def load_topk_for_questions(path: Path, question_ids: set[str]) -> dict[tuple[str, str], list[dict[str, str]]]:
    out: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    if not path.exists():
        return out
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = str(row.get("question_id", ""))
            scope = str(row.get("evidence_scope", ""))
            if qid in question_ids and scope in {"current_only", "history_only"}:
                out[(qid, scope)].append(row)
    for rows in out.values():
        rows.sort(key=lambda r: parse_int(r.get("rank", "9999"), 9999))
    return out


def memory_lookup_key(
    source_agent: str,
    day: Any,
    start_time: str,
    end_time: str,
    granularity: str,
) -> tuple[str, str, str, str, str]:
    return (
        str(source_agent or "").strip().lower(),
        str(day or "").strip(),
        str(start_time or "").strip(),
        str(end_time or "").strip(),
        str(granularity or "").strip().lower(),
    )


def load_memory_lookup(path: Path) -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    out: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = memory_lookup_key(
                row.get("source_agent", ""),
                row.get("day", ""),
                row.get("start_time", ""),
                row.get("end_time", ""),
                row.get("granularity", ""),
            )
            out[key] = row
    return out


def hms_to_seconds(value: str) -> float:
    parts = str(value or "").split(":")
    if len(parts) != 3:
        return 0.0
    hours = parse_float(parts[0])
    minutes = parse_float(parts[1])
    seconds = parse_float(parts[2])
    return hours * 3600.0 + minutes * 60.0 + seconds


def seconds_to_hms(value: float) -> str:
    value = max(0.0, float(value))
    hours = int(value // 3600)
    value -= hours * 3600
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


def seconds_to_token(value: float) -> str:
    hms = seconds_to_hms(value)
    return hms.replace(":", "").replace(".", "")


def row_to_window(selected: SelectedCase, scope: str, row: dict[str, str], reason: str) -> EvidenceWindow:
    start_seconds = parse_float(row.get("start_seconds", ""), hms_to_seconds(row.get("start_time", "")))
    end_seconds = parse_float(row.get("end_seconds", ""), hms_to_seconds(row.get("end_time", "")))
    return EvidenceWindow(
        visual_case_id=selected.visual_case_id,
        dataset_case_id=selected.dataset_case_id,
        question_id=selected.question_id,
        evidence_scope=scope,
        source_agent=row.get("source_agent", ""),
        day=str(row.get("day", "")),
        start_time=row.get("start_time", ""),
        end_time=row.get("end_time", ""),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        granularity=row.get("granularity", ""),
        caption_text=row.get("caption_text", ""),
        rank=parse_int(row.get("rank", "9999"), 9999),
        score=parse_float(row.get("score", "0")),
        raw_key=row.get("raw_key", ""),
        source_file=row.get("source_file", ""),
        selection_reason=reason,
    )


def parse_context_windows(
    selected: SelectedCase,
    scope: str,
    memory_lookup: dict[tuple[str, str, str, str, str], dict[str, str]],
) -> list[EvidenceWindow]:
    field_name = "current_only_context" if scope == "current_only" else "history_only_context"
    text = selected.row.get(field_name, "")
    windows: list[EvidenceWindow] = []
    for match in EVIDENCE_RE.finditer(text):
        if match.group("scope") != scope:
            continue
        start = match.group("start")
        end = match.group("end")
        granularity = match.group("granularity")
        key = memory_lookup_key(match.group("agent"), match.group("day"), start, end, granularity)
        memory_row = memory_lookup.get(key, {})
        row = {
            "source_agent": match.group("agent"),
            "day": match.group("day"),
            "start_time": start,
            "end_time": end,
            "start_seconds": str(hms_to_seconds(start)),
            "end_seconds": str(hms_to_seconds(end)),
            "granularity": granularity,
            "caption_text": normalize_ws(match.group("caption")),
            "rank": match.group("rank"),
            "score": match.group("score"),
            "raw_key": memory_row.get("raw_key", ""),
            "source_file": memory_row.get("source_file", ""),
        }
        windows.append(row_to_window(selected, scope, row, "parsed from dataset evidence_context"))
    return windows


def select_scope_windows(
    selected: SelectedCase,
    scope: str,
    topk_by_q_scope: dict[tuple[str, str], list[dict[str, str]]],
    memory_lookup: dict[tuple[str, str, str, str, str], dict[str, str]],
    windows_per_scope: int,
) -> list[EvidenceWindow]:
    topk_rows = topk_by_q_scope.get((selected.question_id, scope), [])
    windows = [
        row_to_window(selected, scope, row, "selected from top-k evidence")
        for row in topk_rows
    ]
    if not windows:
        windows = parse_context_windows(selected, scope, memory_lookup)

    seen: set[tuple[str, str, str, str, str]] = set()
    unique: list[EvidenceWindow] = []
    for window in sorted(windows, key=lambda w: w.rank):
        sig = (
            window.evidence_scope,
            window.source_agent,
            window.day,
            window.start_time,
            window.end_time,
        )
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(window)

    selected_windows: list[EvidenceWindow] = []
    for window in unique:
        if window.granularity.lower() == "30sec":
            window.selection_reason += "; preferred 30sec evidence for visual audit"
            selected_windows.append(window)
        if len(selected_windows) >= windows_per_scope:
            break

    if len(selected_windows) < windows_per_scope:
        selected_sigs = {
            (w.evidence_scope, w.source_agent, w.day, w.start_time, w.end_time)
            for w in selected_windows
        }
        for window in unique:
            sig = (window.evidence_scope, window.source_agent, window.day, window.start_time, window.end_time)
            if sig in selected_sigs:
                continue
            if window.granularity.lower() == "10min":
                window.selection_reason += "; coarse_10min fallback because fewer than two 30sec windows were available"
            selected_windows.append(window)
            selected_sigs.add(sig)
            if len(selected_windows) >= windows_per_scope:
                break

    return selected_windows[:windows_per_scope]


def build_evidence_windows(
    selected_cases: list[SelectedCase],
    topk_by_q_scope: dict[tuple[str, str], list[dict[str, str]]],
    memory_lookup: dict[tuple[str, str, str, str, str], dict[str, str]],
    windows_per_scope: int,
) -> list[EvidenceWindow]:
    windows: list[EvidenceWindow] = []
    for selected in selected_cases:
        for scope in ("current_only", "history_only"):
            windows.extend(
                select_scope_windows(
                    selected,
                    scope,
                    topk_by_q_scope,
                    memory_lookup,
                    windows_per_scope,
                )
            )
    return windows


def target_times_for_window(window: EvidenceWindow) -> list[tuple[float, str]]:
    duration = max(0.0, window.end_seconds - window.start_seconds)
    granularity = window.granularity.lower()
    if granularity == "30sec" or 25.0 <= duration <= 35.0:
        offsets = [5.0, 15.0, 25.0]
        reason = "30sec_window: target times at start+5s/start+15s/start+25s"
        return [(min(window.end_seconds, window.start_seconds + offset), reason) for offset in offsets]
    if granularity == "10min":
        center = window.start_seconds + duration / 2.0
        return [
            (max(window.start_seconds, center - 10.0), "coarse_10min: no finer window selected; target near center"),
            (center, "coarse_10min: no finer window selected; target at center"),
            (min(window.end_seconds, center + 10.0), "coarse_10min: no finer window selected; target near center"),
        ]
    return [
        (window.start_seconds + duration * 0.25, "generic_window: target at start+25% duration"),
        (window.start_seconds + duration * 0.50, "generic_window: target at center"),
        (window.start_seconds + duration * 0.75, "generic_window: target at start+75% duration"),
    ]


def safe_token(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "unknown"


def output_frame_name(window: EvidenceWindow, target_seconds: float) -> str:
    return (
        f"{safe_token(window.visual_case_id)}_"
        f"{safe_token(window.evidence_scope)}_"
        f"{safe_token(window.source_agent)}_"
        f"D{safe_token(window.day)}_"
        f"{seconds_to_token(target_seconds)}.jpg"
    )


def is_url(value: str) -> bool:
    return bool(URL_RE.match(str(value or "")))


def build_media_indexes(media_roots: list[Path]) -> tuple[dict[str, list[Path]], dict[str, list[Path]], list[Path]]:
    image_index: dict[str, list[Path]] = defaultdict(list)
    video_index: dict[str, list[Path]] = defaultdict(list)
    all_images: list[Path] = []
    seen_roots: set[Path] = set()
    for root in media_roots:
        root = root.expanduser().resolve()
        if root in seen_roots or not root.exists():
            continue
        seen_roots.add(root)
        if root.is_file():
            paths = [root]
        else:
            paths = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
                for filename in filenames:
                    paths.append(Path(dirpath) / filename)
        for path in paths:
            suffix = path.suffix.lower()
            name = path.name.lower()
            if suffix in MEDIA_SUFFIXES_IMAGE:
                image_index[name].append(path)
                all_images.append(path)
            elif suffix in MEDIA_SUFFIXES_VIDEO:
                video_index[name].append(path)
    return image_index, video_index, all_images


def raw_key_candidates(raw_key: str) -> list[str]:
    raw_key = str(raw_key or "").strip()
    if not raw_key:
        return []
    names = {Path(raw_key).name}
    stem = Path(raw_key).stem
    suffix = Path(raw_key).suffix
    if not suffix:
        names.add(stem + ".mp4")
        names.add(stem + ".mov")
        names.add(stem + ".mkv")
    return [name for name in names if name]


def resolve_video_source(window: EvidenceWindow, video_index: dict[str, list[Path]]) -> tuple[str, str, str]:
    for candidate in raw_key_candidates(window.raw_key):
        matches = video_index.get(candidate.lower(), [])
        if matches:
            return str(matches[0]), "planned_local_video_targeted_ffmpeg", "local video matched raw_key"

    for value in (window.raw_key, window.source_file):
        if is_url(value):
            return value, "planned_remote_video_targeted_ffmpeg", "remote URL available; extraction requires byte-range capable ffmpeg"

    return "", "missing_video_source", "no local frame/video or remote video URL found for this evidence window"


def image_matches_target(path: Path, target: FrameTarget) -> bool:
    name = path.name.lower()
    agent = target.window.source_agent.lower()
    day = str(target.window.day).lower()
    time_token = target.target_time_token.lower()
    raw_stem = Path(target.window.raw_key).stem.lower()
    has_time = time_token in name or time_token[:6] in name
    has_agent_or_raw = (agent and agent in name) or (raw_stem and raw_stem in name)
    has_day = f"d{day}" in name or f"day{day}" in name
    return has_time and has_agent_or_raw and (has_day or raw_stem)


def find_existing_frame(target: FrameTarget, image_index: dict[str, list[Path]], all_images: list[Path]) -> Path | None:
    exact = image_index.get(target.output_path.name.lower(), [])
    if exact:
        return exact[0]
    for path in all_images:
        if image_matches_target(path, target):
            return path
    return None


def build_frame_targets(
    windows: list[EvidenceWindow],
    frames_dir: Path,
    image_index: dict[str, list[Path]],
    video_index: dict[str, list[Path]],
    all_images: list[Path],
) -> list[FrameTarget]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    targets: list[FrameTarget] = []
    for window in windows:
        source, planned_status, source_reason = resolve_video_source(window, video_index)
        for target_seconds, timing_reason in target_times_for_window(window):
            output_path = frames_dir / output_frame_name(window, target_seconds)
            target = FrameTarget(
                window=window,
                target_seconds=target_seconds,
                target_frame_time=seconds_to_hms(target_seconds),
                target_time_token=seconds_to_token(target_seconds),
                output_path=output_path,
                video_path_or_url=source,
                extraction_status_planned=planned_status,
                reason=f"{window.selection_reason}; {timing_reason}; {source_reason}".strip("; "),
            )
            existing = find_existing_frame(target, image_index, all_images)
            if existing:
                target.video_path_or_url = str(existing)
                target.extraction_status_planned = "planned_reuse_existing_frame"
                target.reason += "; existing local frame matched target time"
            targets.append(target)
    return targets


def target_offset_for_video(target: FrameTarget) -> float:
    source = target.video_path_or_url
    raw_stem = Path(target.window.raw_key).stem
    source_stem = Path(source).stem
    if raw_stem and raw_stem == source_stem:
        return max(0.0, target.target_seconds - target.window.start_seconds)
    return max(0.0, target.target_seconds)


def copy_existing_frame(source: Path, target: FrameTarget) -> tuple[str, str]:
    if source.resolve() == target.output_path.resolve():
        return str(target.output_path), "reused_existing_frame"
    target.output_path.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copy2(source, target.output_path)
        return str(target.output_path), "reused_existing_frame"

    png_output = target.output_path.with_suffix(source.suffix.lower())
    shutil.copy2(source, png_output)
    return str(png_output), "reused_existing_frame_non_jpeg"


def run_ffmpeg_extract(target: FrameTarget, ffmpeg_path: str) -> tuple[bool, str]:
    target.output_path.parent.mkdir(parents=True, exist_ok=True)
    seek = f"{target_offset_for_video(target):.3f}"
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        seek,
        "-i",
        target.video_path_or_url,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(target.output_path),
    ]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if completed.returncode == 0 and target.output_path.exists():
        return True, ""
    message = normalize_ws(completed.stderr or completed.stdout or f"ffmpeg exited with {completed.returncode}")
    return False, message


def extract_frames(targets: list[FrameTarget], allow_remote: bool) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    image_index: dict[str, list[Path]] = defaultdict(list)
    all_images: list[Path] = []
    for target in targets:
        if target.extraction_status_planned == "planned_reuse_existing_frame":
            source = Path(target.video_path_or_url)
            try:
                copied_path, status = copy_existing_frame(source, target)
                target.extracted_frame_path = copied_path
                target.extraction_status_actual = status
            except OSError as exc:
                target.extraction_status_actual = "existing_frame_copy_failed"
                target.reason += f"; copy failed: {exc}"
            continue

        if target.extraction_status_planned == "missing_video_source":
            target.extraction_status_actual = "missing_video_source"
            continue

        if target.extraction_status_planned == "planned_remote_video_targeted_ffmpeg" and not allow_remote:
            target.extraction_status_actual = "remote_extraction_disabled"
            target.reason += "; remote extraction disabled to avoid unintended network/video download"
            continue

        if not ffmpeg_path:
            if target.extraction_status_planned.startswith("planned_local_video"):
                target.extraction_status_actual = "local_video_no_ffmpeg"
            else:
                target.extraction_status_actual = "remote_url_no_ffmpeg"
            target.reason += "; ffmpeg executable not available"
            continue

        ok, message = run_ffmpeg_extract(target, ffmpeg_path)
        if ok:
            if is_url(target.video_path_or_url):
                target.extraction_status_actual = "extracted_from_remote_video"
            else:
                target.extraction_status_actual = "extracted_from_local_video"
            target.extracted_frame_path = str(target.output_path)
            image_index[target.output_path.name.lower()].append(target.output_path)
            all_images.append(target.output_path)
        else:
            target.extraction_status_actual = "ffmpeg_extract_failed"
            target.reason += f"; ffmpeg failed: {trunc(message, 240)}"


# 5x7 uppercase bitmap font for dependency-free placeholder PNGs.
FONT_5X7: dict[str, list[str]] = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["11111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ",": ["00000", "00000", "00000", "00000", "01100", "00100", "01000"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    ";": ["00000", "01100", "01100", "00000", "01100", "00100", "01000"],
    "!": ["00100", "00100", "00100", "00100", "00100", "00000", "00100"],
    "?": ["01110", "10001", "00001", "00010", "00100", "00000", "00100"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    "\\": ["10000", "01000", "01000", "00100", "00010", "00010", "00001"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    "[": ["01110", "01000", "01000", "01000", "01000", "01000", "01110"],
    "]": ["01110", "00010", "00010", "00010", "00010", "00010", "01110"],
    "'": ["00100", "00100", "01000", "00000", "00000", "00000", "00000"],
    '"': ["01010", "01010", "01010", "00000", "00000", "00000", "00000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "=": ["00000", "00000", "11111", "00000", "11111", "00000", "00000"],
    "%": ["11001", "11010", "00010", "00100", "01000", "01011", "10011"],
    "#": ["01010", "01010", "11111", "01010", "11111", "01010", "01010"],
    "&": ["01100", "10010", "10100", "01000", "10101", "10010", "01101"],
    "@": ["01110", "10001", "10111", "10101", "10111", "10000", "01110"],
    "*": ["00000", "10101", "01110", "11111", "01110", "10101", "00000"],
    "<": ["00010", "00100", "01000", "10000", "01000", "00100", "00010"],
    ">": ["01000", "00100", "00010", "00001", "00010", "00100", "01000"],
    "|": ["00100", "00100", "00100", "00100", "00100", "00100", "00100"],
}
FONT_5X7["?"] = FONT_5X7["?"]


class SimplePNG:
    def __init__(self, width: int, height: int, bg: tuple[int, int, int] = (255, 255, 255)):
        self.width = width
        self.height = height
        self.data = bytearray(width * height * 3)
        self.fill_rect(0, 0, width, height, bg)

    def fill_rect(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int]) -> None:
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(self.width, int(x + w))
        y1 = min(self.height, int(y + h))
        r, g, b = color
        for yy in range(y0, y1):
            row = yy * self.width * 3
            for xx in range(x0, x1):
                idx = row + xx * 3
                self.data[idx] = r
                self.data[idx + 1] = g
                self.data[idx + 2] = b

    def stroke_rect(self, x: int, y: int, w: int, h: int, color: tuple[int, int, int], thickness: int = 2) -> None:
        self.fill_rect(x, y, w, thickness, color)
        self.fill_rect(x, y + h - thickness, w, thickness, color)
        self.fill_rect(x, y, thickness, h, color)
        self.fill_rect(x + w - thickness, y, thickness, h, color)

    def draw_text(self, text: str, x: int, y: int, color: tuple[int, int, int] = (0, 0, 0), scale: int = 3) -> None:
        text = ascii_upper(text)
        cursor_x = int(x)
        cursor_y = int(y)
        for ch in text:
            if ch == "\n":
                cursor_x = int(x)
                cursor_y += 8 * scale
                continue
            glyph = FONT_5X7.get(ch, FONT_5X7["?"])
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        self.fill_rect(cursor_x + gx * scale, cursor_y + gy * scale, scale, scale, color)
            cursor_x += 6 * scale

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = bytearray()
        for y in range(self.height):
            raw.append(0)
            row_start = y * self.width * 3
            raw.extend(self.data[row_start : row_start + self.width * 3])
        compressed = zlib.compress(bytes(raw), level=6)

        def chunk(kind: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + kind
                + payload
                + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
            )

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", compressed)
        png += chunk(b"IEND", b"")
        path.write_bytes(png)


def ascii_upper(text: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.replace("`", "'").replace("~", "-")
    return "".join(ch if ch in FONT_5X7 or ch == "\n" else "?" for ch in ascii_text.upper())


def wrapped_lines(text: str, max_chars: int, max_lines: int) -> list[str]:
    clean = normalize_ws(text)
    lines = textwrap.wrap(clean, width=max(8, max_chars), break_long_words=True, break_on_hyphens=False)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: max(0, max_chars - 3)].rstrip() + "..."
    return lines


def draw_wrapped(
    canvas: SimplePNG,
    text: str,
    x: int,
    y: int,
    width: int,
    max_lines: int,
    color: tuple[int, int, int] = (0, 0, 0),
    scale: int = 3,
) -> int:
    max_chars = max(8, width // (6 * scale))
    line_height = 9 * scale
    for line in wrapped_lines(text, max_chars, max_lines):
        canvas.draw_text(line, x, y, color=color, scale=scale)
        y += line_height
    return y


def grouped_targets_for_case(targets: list[FrameTarget], visual_case_id: str) -> dict[str, list[FrameTarget]]:
    grouped: dict[str, list[FrameTarget]] = defaultdict(list)
    for target in targets:
        if target.window.visual_case_id == visual_case_id:
            grouped[target.window.evidence_scope].append(target)
    return grouped


def caption_summary_for_case(windows: list[EvidenceWindow], visual_case_id: str, scope: str, limit: int = 1400) -> str:
    items = [
        w
        for w in windows
        if w.visual_case_id == visual_case_id and w.evidence_scope == scope
    ]
    parts = [
        f"[{w.evidence_scope} rank={w.rank} source={w.source_agent} D{w.day} "
        f"{w.start_time}-{w.end_time} {w.granularity}] {w.caption_text}"
        for w in sorted(items, key=lambda x: x.rank)
    ]
    return trunc("\n".join(parts), limit)


def status_for_target(target: FrameTarget) -> str:
    if target.extracted_frame_path:
        return f"{target.extraction_status_actual}: {Path(target.extracted_frame_path).name}"
    return target.extraction_status_actual


def generate_placeholder_contact_sheet(
    selected: SelectedCase,
    targets: list[FrameTarget],
    windows: list[EvidenceWindow],
    contact_dir: Path,
) -> Path:
    path = contact_dir / f"{selected.visual_case_id}.png"
    canvas = SimplePNG(1800, 2600, bg=(250, 250, 248))
    navy = (22, 40, 64)
    muted = (89, 99, 110)
    blue = (36, 86, 148)
    green = (32, 115, 84)
    red = (155, 54, 54)
    border = (180, 185, 190)
    pale_blue = (229, 239, 250)
    pale_green = (230, 244, 236)
    pale_red = (249, 232, 232)

    y = 42
    canvas.draw_text(selected.visual_case_id, 42, y, navy, scale=4)
    canvas.draw_text(f"SPLIT {selected.source_split}  CATEGORY {selected.row.get('category', '')}", 42, y + 44, muted, scale=3)
    y += 92
    y = draw_wrapped(canvas, f"Q: {selected.row.get('question', '')}", 42, y, 1700, 5, navy, scale=3)
    y += 8
    y = draw_wrapped(canvas, f"A: {selected.row.get('answer', '')}", 42, y, 1700, 4, (40, 40, 40), scale=3)
    y += 8
    meta = (
        f"EXPECTED: {selected.row.get('expected_result', '')} | "
        f"CUR: {selected.row.get('current_only_answerability', '')} | "
        f"CUR+HIST: {selected.row.get('current_plus_history_answerability', '')} | "
        f"GAIN: {selected.row.get('current_plus_history_gain', '')}"
    )
    y = draw_wrapped(canvas, meta, 42, y, 1700, 3, muted, scale=3)
    y += 24

    grouped = grouped_targets_for_case(targets, selected.visual_case_id)
    for scope, title, fill, title_color in [
        ("current_only", "CURRENT-ONLY EVIDENCE FRAMES", pale_blue, blue),
        ("history_only", "HISTORICAL EVIDENCE FRAMES", pale_green, green),
    ]:
        canvas.fill_rect(42, y, 1716, 38, fill)
        canvas.stroke_rect(42, y, 1716, 38, border, thickness=2)
        canvas.draw_text(title, 58, y + 8, title_color, scale=3)
        y += 54
        scope_targets = grouped.get(scope, [])
        if not scope_targets:
            canvas.fill_rect(58, y, 1680, 90, pale_red)
            canvas.stroke_rect(58, y, 1680, 90, border, thickness=2)
            canvas.draw_text("NO FRAME TARGETS FOUND FOR THIS SCOPE", 78, y + 30, red, scale=3)
            y += 112
            continue

        box_w = 540
        box_h = 142
        x0 = 58
        for idx, target in enumerate(scope_targets[:6]):
            col = idx % 3
            row = idx // 3
            x = x0 + col * (box_w + 22)
            yy = y + row * (box_h + 18)
            canvas.fill_rect(x, yy, box_w, box_h, (255, 255, 255))
            canvas.stroke_rect(x, yy, box_w, box_h, border, thickness=2)
            label = (
                f"{target.window.source_agent} D{target.window.day} "
                f"{target.target_frame_time} R{target.window.rank} {target.window.granularity}"
            )
            canvas.draw_text(label, x + 14, yy + 14, title_color, scale=2)
            status = status_for_target(target)
            status_color = red if "missing" in status or "failed" in status or "disabled" in status else muted
            draw_wrapped(canvas, status, x + 14, yy + 44, box_w - 28, 3, status_color, scale=2)
        y += 2 * (box_h + 18) + 14

        caption = caption_summary_for_case(windows, selected.visual_case_id, scope, limit=900)
        y = draw_wrapped(canvas, f"CAPTION: {caption}", 58, y, 1660, 8, (35, 35, 35), scale=2)
        y += 26

    extraction_counts = Counter(
        target.extraction_status_actual
        for target in targets
        if target.window.visual_case_id == selected.visual_case_id
    )
    status_text = " | ".join(f"{key}: {value}" for key, value in sorted(extraction_counts.items())) or "no targets"
    canvas.fill_rect(42, min(y, 2440), 1716, 96, (245, 245, 245))
    canvas.stroke_rect(42, min(y, 2440), 1716, 96, border, thickness=2)
    draw_wrapped(canvas, f"EXTRACTION STATUS: {status_text}", 58, min(y, 2440) + 24, 1660, 2, muted, scale=3)
    canvas.save(path)
    return path


def try_import_pillow() -> tuple[Any, Any, Any] | tuple[None, None, None]:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ModuleNotFoundError:
        return None, None, None
    return Image, ImageDraw, ImageFont


def pil_font(ImageFont: Any, size: int, bold: bool = False) -> Any:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
    return ImageFont.load_default()


def pil_text_width(draw: Any, text: str, font: Any) -> int:
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return int(box[2] - box[0])
    except AttributeError:
        return int(draw.textsize(text, font=font)[0])


def pil_line_height(draw: Any, font: Any) -> int:
    try:
        box = draw.textbbox((0, 0), "Ag", font=font)
        return int(box[3] - box[1]) + 6
    except AttributeError:
        return int(draw.textsize("Ag", font=font)[1]) + 6


def draw_pil_wrapped(
    draw: Any,
    text: str,
    x: int,
    y: int,
    width: int,
    font: Any,
    fill: tuple[int, int, int],
    max_lines: int,
) -> int:
    words = normalize_ws(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        proposed = word if not current else f"{current} {word}"
        if pil_text_width(draw, proposed, font) <= width:
            current = proposed
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(". ") + "..."
    line_height = pil_line_height(draw, font)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def generate_pil_contact_sheet(
    selected: SelectedCase,
    targets: list[FrameTarget],
    windows: list[EvidenceWindow],
    contact_dir: Path,
) -> Path | None:
    Image, ImageDraw, ImageFont = try_import_pillow()
    if Image is None:
        return None

    case_targets = [target for target in targets if target.window.visual_case_id == selected.visual_case_id]
    if not any(target.extracted_frame_path for target in case_targets):
        return None

    path = contact_dir / f"{selected.visual_case_id}.png"
    image = Image.new("RGB", (1800, 2600), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    title_font = pil_font(ImageFont, 42, bold=True)
    meta_font = pil_font(ImageFont, 24)
    body_font = pil_font(ImageFont, 24)
    small_font = pil_font(ImageFont, 18)
    navy = (22, 40, 64)
    muted = (89, 99, 110)
    blue = (36, 86, 148)
    green = (32, 115, 84)
    red = (155, 54, 54)
    border = (180, 185, 190)

    y = 42
    draw.text((42, y), selected.visual_case_id, font=title_font, fill=navy)
    y += 50
    draw.text(
        (42, y),
        f"Split {selected.source_split} | Category {selected.row.get('category', '')}",
        font=meta_font,
        fill=muted,
    )
    y += 42
    y = draw_pil_wrapped(draw, f"Q: {selected.row.get('question', '')}", 42, y, 1700, body_font, navy, 4)
    y = draw_pil_wrapped(draw, f"A: {selected.row.get('answer', '')}", 42, y + 6, 1700, body_font, (40, 40, 40), 3)
    meta = (
        f"Expected: {selected.row.get('expected_result', '')} | "
        f"Current: {selected.row.get('current_only_answerability', '')} | "
        f"Current+history: {selected.row.get('current_plus_history_answerability', '')} | "
        f"Gain: {selected.row.get('current_plus_history_gain', '')}"
    )
    y = draw_pil_wrapped(draw, meta, 42, y + 8, 1700, meta_font, muted, 2) + 18

    grouped = grouped_targets_for_case(targets, selected.visual_case_id)
    for scope, title, fill, title_color in [
        ("current_only", "Current-only evidence frames", (229, 239, 250), blue),
        ("history_only", "Historical evidence frames", (230, 244, 236), green),
    ]:
        draw.rectangle((42, y, 1758, y + 42), fill=fill, outline=border, width=2)
        draw.text((58, y + 8), title, font=meta_font, fill=title_color)
        y += 58
        scope_targets = grouped.get(scope, [])
        box_w = 540
        box_h = 184
        x0 = 58
        for idx, target in enumerate(scope_targets[:6]):
            col = idx % 3
            row = idx // 3
            x = x0 + col * (box_w + 22)
            yy = y + row * (box_h + 18)
            draw.rectangle((x, yy, x + box_w, yy + box_h), fill=(255, 255, 255), outline=border, width=2)
            frame_path = Path(target.extracted_frame_path) if target.extracted_frame_path else None
            if frame_path and frame_path.exists():
                try:
                    thumb = Image.open(frame_path).convert("RGB")
                    thumb.thumbnail((box_w - 20, 122))
                    px = x + (box_w - thumb.width) // 2
                    image.paste(thumb, (px, yy + 10))
                except OSError:
                    draw.rectangle((x + 12, yy + 12, x + box_w - 12, yy + 128), fill=(249, 232, 232), outline=border)
                    draw.text((x + 22, yy + 56), "Frame open failed", font=small_font, fill=red)
            else:
                draw.rectangle((x + 12, yy + 12, x + box_w - 12, yy + 128), fill=(249, 232, 232), outline=border)
                draw.text((x + 22, yy + 56), target.extraction_status_actual, font=small_font, fill=red)
            label = (
                f"{target.window.source_agent} D{target.window.day} {target.target_frame_time} "
                f"R{target.window.rank} {target.window.granularity}"
            )
            draw_pil_wrapped(draw, label, x + 14, yy + 136, box_w - 28, small_font, title_color, 2)
        y += 2 * (box_h + 18) + 12
        caption = caption_summary_for_case(windows, selected.visual_case_id, scope, limit=900)
        y = draw_pil_wrapped(draw, f"Caption: {caption}", 58, y, 1660, small_font, (35, 35, 35), 6) + 22

    extraction_counts = Counter(
        target.extraction_status_actual
        for target in targets
        if target.window.visual_case_id == selected.visual_case_id
    )
    status_text = " | ".join(f"{key}: {value}" for key, value in sorted(extraction_counts.items())) or "no targets"
    draw.rectangle((42, min(y, 2440), 1758, min(y, 2440) + 96), fill=(245, 245, 245), outline=border, width=2)
    draw_pil_wrapped(draw, f"Extraction status: {status_text}", 58, min(y, 2440) + 24, 1660, meta_font, muted, 2)
    image.save(path)
    return path


def generate_contact_sheets(
    selected_cases: list[SelectedCase],
    targets: list[FrameTarget],
    windows: list[EvidenceWindow],
    contact_dir: Path,
) -> None:
    contact_dir.mkdir(parents=True, exist_ok=True)
    for selected in selected_cases:
        sheet_path = generate_pil_contact_sheet(selected, targets, windows, contact_dir)
        if sheet_path is None:
            sheet_path = generate_placeholder_contact_sheet(selected, targets, windows, contact_dir)
        selected.contact_sheet_path = str(sheet_path)


def rel_to_output(path: str | Path, output_dir: Path) -> str:
    if not path:
        return ""
    path_obj = Path(path)
    try:
        return str(path_obj.resolve().relative_to(output_dir.resolve()))
    except ValueError:
        return str(path_obj)


def aggregate_case_status(targets: list[FrameTarget], visual_case_id: str) -> tuple[str, list[str]]:
    case_targets = [t for t in targets if t.window.visual_case_id == visual_case_id]
    frame_paths = [t.extracted_frame_path for t in case_targets if t.extracted_frame_path]
    if not case_targets:
        return "no_frame_targets", frame_paths
    if len(frame_paths) == len(case_targets):
        return "all_frames_available", frame_paths
    if frame_paths:
        return "partial_frames_available", frame_paths
    return "frames_missing", frame_paths


def build_case_subset_rows(selected_cases: list[SelectedCase]) -> list[dict[str, str]]:
    rows = []
    for selected in selected_cases:
        row = selected.row
        rows.append(
            {
                "visual_case_id": selected.visual_case_id,
                "source_split": selected.source_split,
                "dataset_case_id": selected.dataset_case_id,
                "question_id": selected.question_id,
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "category": row.get("category", ""),
                "expected_result": row.get("expected_result", ""),
                "current_only_answerability": row.get("current_only_answerability", ""),
                "current_plus_history_answerability": row.get("current_plus_history_answerability", ""),
                "current_plus_history_gain": row.get("current_plus_history_gain", ""),
                "reason_selected_for_visual_audit": selected.reason,
            }
        )
    return rows


def build_frame_plan_rows(targets: list[FrameTarget]) -> list[dict[str, str]]:
    rows = []
    for target in targets:
        window = target.window
        rows.append(
            {
                "visual_case_id": window.visual_case_id,
                "dataset_case_id": window.dataset_case_id,
                "question_id": window.question_id,
                "evidence_scope": window.evidence_scope,
                "source_agent": window.source_agent,
                "day": window.day,
                "start_time": window.start_time,
                "end_time": window.end_time,
                "granularity": window.granularity,
                "caption_text": window.caption_text,
                "target_frame_time": target.target_frame_time,
                "raw_key": window.raw_key,
                "video_path_or_url": target.video_path_or_url,
                "extraction_status_planned": target.extraction_status_planned,
                "reason": target.reason,
            }
        )
    return rows


def build_audit_table_rows(
    selected_cases: list[SelectedCase],
    targets: list[FrameTarget],
    windows: list[EvidenceWindow],
    output_dir: Path,
) -> list[dict[str, str]]:
    rows = []
    for selected in selected_cases:
        status, frame_paths = aggregate_case_status(targets, selected.visual_case_id)
        selected.extraction_status = status
        selected.extracted_frame_paths = frame_paths
        rows.append(
            {
                "visual_case_id": selected.visual_case_id,
                "dataset_case_id": selected.dataset_case_id,
                "question_id": selected.question_id,
                "split": selected.source_split,
                "question": selected.row.get("question", ""),
                "answer": selected.row.get("answer", ""),
                "category": selected.row.get("category", ""),
                "current_only_caption": caption_summary_for_case(windows, selected.visual_case_id, "current_only"),
                "historical_caption": caption_summary_for_case(windows, selected.visual_case_id, "history_only"),
                "contact_sheet_path": rel_to_output(selected.contact_sheet_path, output_dir),
                "extracted_frame_paths": "; ".join(rel_to_output(p, output_dir) for p in frame_paths),
                "extraction_status": status,
                "qa_answer_visually_plausible": "",
                "caption_matches_frames": "",
                "current_only_enough": "",
                "history_adds_value": "",
                "keep_for_ppt": "",
                "notes": "",
            }
        )
    return rows


def combined_story(selected: SelectedCase, model_inputs: dict[tuple[str, str], dict[str, Any]]) -> str:
    text = selected.row.get("current_plus_historical_context", "")
    if text:
        return text
    item = model_inputs.get((selected.dataset_case_id, "current_plus_historical"))
    if item:
        return str(item.get("evidence_context", ""))
    return ""


def html_escape_text(text: Any) -> str:
    return html.escape(str(text or ""))


def frame_gallery_html(selected: SelectedCase, targets: list[FrameTarget], output_dir: Path) -> str:
    frame_targets = [
        target
        for target in targets
        if target.window.visual_case_id == selected.visual_case_id and target.extracted_frame_path
    ]
    if not frame_targets:
        return '<p class="frames-none">No extracted frames available for this case.</p>'
    cards = []
    for target in frame_targets:
        rel = rel_to_output(target.extracted_frame_path, output_dir)
        label = (
            f"{target.window.evidence_scope} | {target.window.source_agent} D{target.window.day} "
            f"{target.target_frame_time} | rank {target.window.rank}"
        )
        cards.append(
            f"""
      <figure>
        <img src="{html_escape_text(rel)}" alt="{html_escape_text(label)}">
        <figcaption>{html_escape_text(label)}</figcaption>
      </figure>
"""
        )
    return f'<div class="frames">{"".join(cards)}</div>'


def build_html_gallery(
    selected_cases: list[SelectedCase],
    targets: list[FrameTarget],
    windows: list[EvidenceWindow],
    model_inputs: dict[tuple[str, str], dict[str, Any]],
    output_dir: Path,
) -> None:
    rows: list[str] = []
    for selected in selected_cases:
        status_counts = Counter(
            t.extraction_status_actual
            for t in targets
            if t.window.visual_case_id == selected.visual_case_id
        )
        status_text = ", ".join(f"{k}: {v}" for k, v in sorted(status_counts.items())) or "no targets"
        contact_rel = rel_to_output(selected.contact_sheet_path, output_dir)
        current_caption = caption_summary_for_case(windows, selected.visual_case_id, "current_only", limit=2400)
        historical_caption = caption_summary_for_case(windows, selected.visual_case_id, "history_only", limit=2400)
        story = trunc(combined_story(selected, model_inputs), 3600)
        frames_html = frame_gallery_html(selected, targets, output_dir)
        rows.append(
            f"""
<section class="case">
  <h2>{html_escape_text(selected.visual_case_id)} <span>{html_escape_text(selected.source_split)}</span></h2>
  <div class="meta">
    <b>Dataset case:</b> {html_escape_text(selected.dataset_case_id)}
    <b>Question ID:</b> {html_escape_text(selected.question_id)}
    <b>Category:</b> {html_escape_text(selected.row.get('category', ''))}
    <b>Expected:</b> {html_escape_text(selected.row.get('expected_result', ''))}
  </div>
  <p><b>Question:</b> {html_escape_text(selected.row.get('question', ''))}</p>
  <p><b>Answer:</b> {html_escape_text(selected.row.get('answer', ''))}</p>
  <p><b>Extraction status:</b> {html_escape_text(status_text)}</p>
  <div class="sheet"><img src="{html_escape_text(contact_rel)}" alt="{html_escape_text(selected.visual_case_id)} contact sheet"></div>
  <h3>Extracted frames</h3>
  {frames_html}
  <div class="cols">
    <div>
      <h3>Current-only caption evidence</h3>
      <pre>{html_escape_text(current_caption)}</pre>
    </div>
    <div>
      <h3>Historical caption evidence</h3>
      <pre>{html_escape_text(historical_caption)}</pre>
    </div>
  </div>
  <details>
    <summary>Current+historical combined story</summary>
    <pre>{html_escape_text(story)}</pre>
  </details>
  <table class="check">
    <tr><th>qa_answer_visually_plausible</th><td></td></tr>
    <tr><th>caption_matches_frames</th><td></td></tr>
    <tr><th>current_only_enough</th><td></td></tr>
    <tr><th>history_adds_value</th><td></td></tr>
    <tr><th>keep_for_ppt</th><td></td></tr>
    <tr><th>notes</th><td></td></tr>
  </table>
</section>
"""
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dataset V0.1 Visual Audit Packet</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; background: #f7f7f5; }}
h1 {{ margin: 0 0 8px; }}
.claim {{ color: #55616f; margin-bottom: 24px; }}
.case {{ background: #fff; border: 1px solid #d7dce0; margin: 0 0 28px; padding: 18px; }}
h2 {{ margin: 0 0 8px; color: #18324d; }}
h2 span {{ font-size: 14px; color: #64707d; margin-left: 8px; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 8px 16px; color: #4a5562; font-size: 14px; }}
.sheet img {{ max-width: 100%; border: 1px solid #cfd6dd; margin: 12px 0; background: #fff; }}
.frames {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }}
.frames figure {{ margin: 0; border: 1px solid #d4dbe2; padding: 6px; background: #fbfbfb; }}
.frames img {{ width: 100%; height: auto; display: block; }}
.frames figcaption, .frames-none {{ color: #64707d; font-size: 13px; }}
.cols {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
pre {{ white-space: pre-wrap; font-size: 13px; line-height: 1.35; background: #f2f4f6; padding: 12px; overflow-wrap: anywhere; }}
.check {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
.check th, .check td {{ border: 1px solid #ccd3da; padding: 8px; text-align: left; height: 28px; }}
.check th {{ width: 240px; background: #eef2f5; }}
</style>
</head>
<body>
<h1>Dataset V0.1 Visual Audit Packet</h1>
<p class="claim">Targeted human-inspection packet. Caption-only candidates are not final QA labels.</p>
{''.join(rows)}
</body>
</html>
"""
    (output_dir / "visual_audit_packet_gallery.html").write_text(document, encoding="utf-8")


def rank_ppt_candidates(selected_cases: list[SelectedCase]) -> list[SelectedCase]:
    def score(selected: SelectedCase) -> int:
        row = selected.row
        return sum(
            [
                4 if selected.source_split == "demo" else 0,
                3 if row.get("expected_result") == "current_plus_history_better" else 0,
                2 if row.get("current_plus_history_gain") == "yes" else 0,
                confidence_rank(row.get("confidence", "")),
            ]
        )

    return [
        selected
        for _, selected in sorted(
            enumerate(selected_cases),
            key=lambda item: (-score(item[1]), item[0]),
        )[:5]
    ]


def build_report(
    selected_cases: list[SelectedCase],
    windows: list[EvidenceWindow],
    targets: list[FrameTarget],
    output_dir: Path,
    media_roots: list[Path],
) -> str:
    extracted = [
        t for t in targets
        if t.extraction_status_actual in {"extracted_from_local_video", "extracted_from_remote_video"}
    ]
    reused = [t for t in targets if t.extraction_status_actual.startswith("reused_existing_frame")]
    available = extracted + reused
    missing = [t for t in targets if not t.extracted_frame_path]
    status_counts = Counter(t.extraction_status_actual for t in targets)
    planned_counts = Counter(t.extraction_status_planned for t in targets)
    coarse_10min = [w for w in windows if w.granularity.lower() == "10min"]
    contact_count = sum(1 for s in selected_cases if s.contact_sheet_path and Path(s.contact_sheet_path).exists())
    ppt_cases = rank_ppt_candidates(selected_cases)

    no_video = status_counts.get("missing_video_source", 0) == len(targets) if targets else False
    blockers = []
    if no_video:
        blockers.append("No local video/frame source was found for the selected MA-EgoQA evidence raw_keys under the searched media roots.")
    if not available:
        blockers.append("No reusable frame mapping was found; contact sheets are caption-only placeholders.")
    if coarse_10min:
        blockers.append(f"{len(coarse_10min)} selected evidence windows are 10min granularity and are marked coarse_10min.")
    blockers.append("Visual frames alone cannot verify spoken dialogue, speaker identity, or intent; they only support visual plausibility checks.")

    report = f"""# Dataset V0.1 Visual Audit Report

## Counts

- Selected visual audit cases count: {len(selected_cases)}
- Frame extraction targets count: {len(targets)}
- Extracted frames count: {len(extracted)}
- Reused existing frames count: {len(reused)}
- Missing/unavailable frames count: {len(missing)}
- Contact sheets generated count: {contact_count}

## Planned Extraction Status

{format_counter(planned_counts)}

## Actual Extraction Status

{format_counter(status_counts)}

## Best Cases For PPT Inspection

These are the first cases to inspect manually. They are not yet PPT-ready visual claims unless frames are later attached and human-verified.

{format_ppt_cases(ppt_cases)}

## Blockers

{format_bullets(blockers)}

Searched media roots:

{format_bullets([str(p) for p in media_roots])}

## What Can Be Claimed

- We built a targeted visual audit packet for Dataset V0.1.
- The packet allows human inspection of whether QA/caption evidence is visually plausible.
- Missing visual sources are explicitly marked instead of treated as successful frame extraction.

## What Cannot Be Claimed

- QA labels are final.
- Dataset V0.1 is fully verified.
- Model accuracy improves.
- Historical memory is proven useful.

## Output Files

- `visual_audit_case_subset.csv`
- `visual_frame_extraction_plan.csv`
- `visual_audit_table.csv`
- `visual_audit_packet_gallery.html`
- `visual_audit_report.md`
- `contact_sheets/`
- `frames/`
"""
    (output_dir / "visual_audit_report.md").write_text(report, encoding="utf-8")
    return report


def format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "- none"
    return "\n".join(f"- {key}: {value}" for key, value in sorted(counter.items()))


def format_bullets(items: list[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {item}" for item in items)


def format_ppt_cases(cases: list[SelectedCase]) -> str:
    if not cases:
        return "- none"
    lines = []
    for case in cases:
        lines.append(
            f"- {case.visual_case_id} / {case.dataset_case_id} / Q{case.question_id}: "
            f"{trunc(case.row.get('question', ''), 140)}"
        )
    return "\n".join(lines)


def validate_inputs(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Dataset V0.1 visual audit packet.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--media-root",
        action="append",
        type=Path,
        default=[],
        help="Optional local root to search for existing frames/videos. Can be repeated.",
    )
    parser.add_argument("--windows-per-scope", type=int, default=2)
    parser.add_argument(
        "--allow-remote-ffmpeg",
        action="store_true",
        help="Allow targeted ffmpeg extraction from explicit remote URLs. Off by default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    frames_dir = output_dir / "frames"
    contact_dir = output_dir / "contact_sheets"

    validate_inputs([DEMO_CSV, EVAL_LITE_CSV, CONTROL_CSV, MODEL_INPUTS_JSONL, TOPK_EVIDENCE_CSV, MEMORY_INDEX_CSV])
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_rows = read_csv(DEMO_CSV)
    eval_rows = read_csv(EVAL_LITE_CSV)
    control_rows = read_csv(CONTROL_CSV)
    model_inputs = read_model_inputs(MODEL_INPUTS_JSONL)

    selected_cases = select_visual_cases(demo_rows, eval_rows, control_rows)
    question_ids = {case.question_id for case in selected_cases}
    topk_by_q_scope = load_topk_for_questions(TOPK_EVIDENCE_CSV, question_ids)
    memory_lookup = load_memory_lookup(MEMORY_INDEX_CSV)

    media_roots = [ROOT] + list(args.media_root)
    image_index, video_index, all_images = build_media_indexes(media_roots)

    windows = build_evidence_windows(
        selected_cases,
        topk_by_q_scope,
        memory_lookup,
        windows_per_scope=max(1, args.windows_per_scope),
    )
    targets = build_frame_targets(windows, frames_dir, image_index, video_index, all_images)
    extract_frames(targets, allow_remote=args.allow_remote_ffmpeg)
    generate_contact_sheets(selected_cases, targets, windows, contact_dir)

    write_csv(output_dir / "visual_audit_case_subset.csv", build_case_subset_rows(selected_cases), CASE_SUBSET_FIELDS)
    write_csv(output_dir / "visual_frame_extraction_plan.csv", build_frame_plan_rows(targets), FRAME_PLAN_FIELDS)
    write_csv(output_dir / "visual_audit_table.csv", build_audit_table_rows(selected_cases, targets, windows, output_dir), AUDIT_TABLE_FIELDS)
    build_html_gallery(selected_cases, targets, windows, model_inputs, output_dir)
    build_report(selected_cases, windows, targets, output_dir, media_roots)

    extracted_count = sum(
        1 for target in targets
        if target.extraction_status_actual in {"extracted_from_local_video", "extracted_from_remote_video"}
    )
    contact_count = sum(1 for case in selected_cases if case.contact_sheet_path and Path(case.contact_sheet_path).exists())
    missing_count = sum(1 for target in targets if not target.extracted_frame_path)
    failed_reasons = Counter(target.extraction_status_actual for target in targets if not target.extracted_frame_path)

    print(f"output directory: {output_dir}")
    print(f"selected case count: {len(selected_cases)}")
    print(f"extracted frame count: {extracted_count}")
    print(f"contact sheet count: {contact_count}")
    print(f"missing frame count: {missing_count}")
    print("5 best PPT candidate cases:")
    for case in rank_ppt_candidates(selected_cases):
        print(f"- {case.visual_case_id} {case.dataset_case_id} Q{case.question_id}: {trunc(case.row.get('question', ''), 120)}")
    print("failed extraction reasons:")
    if failed_reasons:
        for reason, count in sorted(failed_reasons.items()):
            print(f"- {reason}: {count}")
    else:
        print("- none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
