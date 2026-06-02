#!/usr/bin/env python3
"""
CASTLE incremental-evidence QA proof-of-concept.

This metadata-driven pipeline tests whether views beyond a querying user's own
ego view can add evidence for QA. It does not download full videos and does not
use hf_hub_download for videos. All frame access is via ffmpeg remote URLs.

Default first run:
  - select 3 high-overlap 5-minute windows
  - extract one overview frame from all active streams in each selected window
  - write annotation/QA templates and strawman comparison artifacts
  - keep tracked local artifacts under 500 MB
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from castle_event_relevant_view_selection import (
    Interval,
    StreamFrame,
    Window,
    clip_embeddings,
    cosine,
    ffmpeg_extract_frame,
    make_overview_sheet,
    make_stream_frame,
    parse_clock_seconds,
    read_intervals,
    replace_status,
    seconds_to_time,
    tracked_size,
    write_stream_frame_log,
)


REPO_ID = "CASTLE-Dataset/CASTLE2024"
REPO_TYPE = "dataset"
DEFAULT_OUTPUT_DIR = Path("castle_incremental_qa_poc")
WINDOW_MINUTES = 5
RANDOM_SEED = 42
STATIC_PRIORITY = ["Meeting", "Living1", "Living2", "Kitchen", "Reading"]
RELEVANCE_LEVELS = {
    "direct_event_detail",
    "direct_static_event",
    "spatial_context_only",
    "off_event",
    "invalid_or_calibration",
    "unclear",
}
DIRECT_EVIDENCE_LEVELS = {"direct_event_detail", "direct_static_event"}
INCREMENTAL_EVIDENCE_LEVELS = {"direct_event_detail", "direct_static_event", "spatial_context_only"}
OFF_EVENT_LEVELS = {"off_event", "invalid_or_calibration"}


@dataclass(frozen=True)
class CandidateWindow:
    window_id: str
    day: str
    period: str
    start_seconds: float
    end_seconds: float
    active: tuple[Interval, ...]

    @property
    def ego(self) -> list[Interval]:
        return [item for item in self.active if item.stream_type == "ego"]

    @property
    def static(self) -> list[Interval]:
        return [item for item in self.active if item.stream_type == "static"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CASTLE incremental QA POC from multi-stream overview frames.")
    parser.add_argument("--interval-table", type=Path, default=Path("castle_main_video_interval_table.csv"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--repo-type", default=REPO_TYPE)
    parser.add_argument("--num-windows", type=int, default=3)
    parser.add_argument("--window-minutes", type=int, default=WINDOW_MINUTES)
    parser.add_argument("--min-active-streams", type=int, default=6)
    parser.add_argument("--min-static-streams", type=int, default=1)
    parser.add_argument("--overview-target-offset-sec", type=float, default=60.0)
    parser.add_argument("--max-local-mb", type=float, default=500.0)
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=3,
        help="Stop a window after this many consecutive failed frame extractions.",
    )
    parser.add_argument("--skip-clip", action="store_true", help="Skip CLIP-based strawman ranking.")
    parser.add_argument("--dry-run", action="store_true", help="Select windows/templates without running ffmpeg.")
    return parser.parse_args()


def period_for_seconds(seconds: float) -> str:
    hour = (seconds % (24 * 3600)) / 3600
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 24:
        return "evening"
    return "night"


def window_id(day: str, start_seconds: float) -> str:
    return f"{day}_{seconds_to_time(start_seconds).replace(':', '').replace('.', '')}"


def build_candidate_windows(
    intervals: list[Interval],
    *,
    window_minutes: int,
    min_active_streams: int,
    min_static_streams: int,
) -> list[CandidateWindow]:
    by_day: dict[str, list[Interval]] = defaultdict(list)
    for interval in intervals:
        by_day[interval.day].append(interval)

    bin_seconds = window_minutes * 60
    candidates = []
    for day in sorted(by_day):
        for bin_index in range((24 * 60) // window_minutes):
            start = bin_index * bin_seconds
            end = start + bin_seconds
            active = tuple(item for item in by_day[day] if item.start_seconds <= start and item.end_seconds >= end)
            ego_count = sum(1 for item in active if item.stream_type == "ego")
            static_count = sum(1 for item in active if item.stream_type == "static")
            if ego_count + static_count >= min_active_streams and static_count >= min_static_streams:
                candidates.append(
                    CandidateWindow(
                        window_id=window_id(day, start),
                        day=day,
                        period=period_for_seconds(start),
                        start_seconds=start,
                        end_seconds=end,
                        active=active,
                    )
                )
    return sorted(candidates, key=rank_window)


def rank_window(window: CandidateWindow) -> tuple[int, int, int, float, str]:
    return (
        -(len(window.active)),
        -len(window.ego),
        -len(window.static),
        window.start_seconds,
        window.day,
    )


def select_diverse_windows(candidates: list[CandidateWindow], num_windows: int) -> list[CandidateWindow]:
    selected: list[CandidateWindow] = []
    selected_ids = set()
    selected_days = set()
    desired_periods = ["morning", "afternoon", "evening", "night"]

    for period in desired_periods:
        period_candidates = [
            item
            for item in candidates
            if item.period == period and item.window_id not in selected_ids and item.day not in selected_days
        ]
        if not period_candidates:
            period_candidates = [item for item in candidates if item.period == period and item.window_id not in selected_ids]
        if period_candidates:
            chosen = sorted(period_candidates, key=rank_window)[0]
            selected.append(chosen)
            selected_ids.add(chosen.window_id)
            selected_days.add(chosen.day)
        if len(selected) >= num_windows:
            return selected

    for candidate in candidates:
        if candidate.window_id in selected_ids:
            continue
        # Avoid choosing windows that are effectively adjacent unless needed.
        too_close = any(candidate.day == item.day and abs(candidate.start_seconds - item.start_seconds) < 30 * 60 for item in selected)
        if too_close and len(candidates) - len(selected) > num_windows - len(selected):
            continue
        selected.append(candidate)
        selected_ids.add(candidate.window_id)
        if len(selected) >= num_windows:
            break
    return selected


def to_window(candidate: CandidateWindow) -> Window:
    return Window(
        window_id=candidate.window_id,
        day=candidate.day,
        period=candidate.period,
        start_time=seconds_to_time(candidate.start_seconds),
        end_time=seconds_to_time(candidate.end_seconds),
        start_seconds=candidate.start_seconds,
        end_seconds=candidate.end_seconds,
        active_ego_entities=tuple(item.entity for item in candidate.ego),
        active_static_entities=tuple(item.entity for item in candidate.static),
    )


def write_selected_windows(windows: list[CandidateWindow], output_path: Path) -> None:
    fieldnames = [
        "window_id",
        "day",
        "period",
        "window_start",
        "window_end",
        "num_active_ego",
        "num_active_static",
        "total_active_streams",
        "active_ego_entities",
        "active_static_entities",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for window in windows:
            writer.writerow(
                {
                    "window_id": window.window_id,
                    "day": window.day,
                    "period": window.period,
                    "window_start": seconds_to_time(window.start_seconds),
                    "window_end": seconds_to_time(window.end_seconds),
                    "num_active_ego": len(window.ego),
                    "num_active_static": len(window.static),
                    "total_active_streams": len(window.active),
                    "active_ego_entities": ";".join(item.entity for item in window.ego),
                    "active_static_entities": ";".join(item.entity for item in window.static),
                }
            )


def extraction_paths(output_dir: Path) -> tuple[Path, Path]:
    return output_dir / "frames", output_dir / "ffmpeg_logs"


def extract_or_reuse(
    item: StreamFrame,
    *,
    timeout_sec: int,
    tracked_paths: list[Path],
    max_bytes: int,
    dry_run: bool,
) -> StreamFrame:
    if item.frame_path.exists() and item.frame_path.stat().st_size > 0:
        return replace_status(item, "ok")
    return ffmpeg_extract_frame(item, timeout_sec, tracked_paths, max_bytes, dry_run)


def extract_window_overview(
    window: CandidateWindow,
    *,
    repo_id: str,
    repo_type: str,
    frames_dir: Path,
    logs_dir: Path,
    timeout_sec: int,
    tracked_paths: list[Path],
    max_bytes: int,
    target_offset_sec: float,
    dry_run: bool,
    max_consecutive_failures: int,
) -> list[StreamFrame]:
    w = to_window(window)
    target_clock = window.start_seconds + min(target_offset_sec, (window.end_seconds - window.start_seconds) / 2)
    active_intervals = sorted(window.active, key=lambda item: (0 if item.stream_type == "ego" else 1, item.entity))
    outputs = []
    consecutive_failures = 0
    stop_after_index: int | None = None
    stop_status = "not_attempted_after_failure"
    for index, interval in enumerate(active_intervals):
        frame = make_stream_frame(
            w,
            interval,
            target_clock,
            repo_id=repo_id,
            repo_type=repo_type,
            frames_dir=frames_dir,
            logs_dir=logs_dir,
            label="overview",
        )
        extracted = extract_or_reuse(frame, timeout_sec=timeout_sec, tracked_paths=tracked_paths, max_bytes=max_bytes, dry_run=dry_run)
        outputs.append(extracted)
        if extracted.status in {"ok", "dry_run_not_extracted"}:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
        if extracted.status in {"ffmpeg_timeout", "ffmpeg_missing"}:
            stop_status = f"not_attempted_after_{extracted.status}"
            stop_after_index = index + 1
            break
        if max_consecutive_failures > 0 and consecutive_failures >= max_consecutive_failures:
            stop_status = "not_attempted_after_consecutive_failures"
            stop_after_index = index + 1
            break
    if stop_after_index is not None:
        for interval in active_intervals[stop_after_index:]:
            frame = make_stream_frame(
                w,
                interval,
                target_clock,
                repo_id=repo_id,
                repo_type=repo_type,
                frames_dir=frames_dir,
                logs_dir=logs_dir,
                label="overview",
            )
            outputs.append(replace_status(frame, stop_status))
    return outputs


def write_all_extraction_logs(rows: list[StreamFrame], output_path: Path) -> None:
    write_stream_frame_log(rows, output_path)


def manual_seed_label(window_id_value: str, entity: str) -> tuple[str, str, str, str]:
    if window_id_value != "DAY1_100500000":
        return "unclear", "", "unlabeled high-overlap view; needs manual review", "yes"
    seeds = {
        "Onanong": (
            "direct_event_detail",
            "speaker_visible;audience_visible;object_visible",
            "seeded from prior manual inspection: ego view aligned to living-room presentation event",
            "no",
        ),
        "Tien": (
            "direct_event_detail",
            "speaker_visible;audience_visible;object_visible",
            "seeded from prior manual inspection: ego view aligned to living-room presentation event",
            "no",
        ),
        "Meeting": (
            "direct_static_event",
            "speaker_visible;audience_visible;room_layout_only",
            "seeded from prior manual inspection: static view directly relevant to group presentation event",
            "no",
        ),
        "Living1": (
            "spatial_context_only",
            "audience_visible;room_layout_only",
            "seeded from prior manual inspection: broad spatial context for the event",
            "no",
        ),
        "Living2": (
            "spatial_context_only",
            "room_layout_only",
            "seeded from prior manual inspection: broad spatial context, less direct than Meeting",
            "no",
        ),
        "Kitchen": (
            "off_event",
            "empty_room;kitchen_scene",
            "seeded from prior manual inspection: active stream but irrelevant empty kitchen",
            "no",
        ),
        "Reading": (
            "off_event",
            "empty_room;room_layout_only",
            "seeded from prior manual inspection: active stream but off-event reading room",
            "no",
        ),
        "Klaus": (
            "invalid_or_calibration",
            "calibration_pattern",
            "seeded from prior manual inspection: test/calibration-like view",
            "no",
        ),
    }
    return seeds.get(entity, ("unclear", "", "not labeled yet; needs manual review", "yes"))


def write_view_annotation(rows: list[StreamFrame], output_path: Path) -> None:
    fieldnames = [
        "window_id",
        "day",
        "clock_time",
        "entity",
        "stream_type",
        "video_path",
        "frame_path",
        "relevance_level",
        "visible_cues",
        "notes",
        "needs_manual_review",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            if item.status != "ok":
                relevance, cues, notes, review = "unclear", "", f"frame extraction status: {item.status}", "yes"
            else:
                relevance, cues, notes, review = manual_seed_label(item.window_id, item.entity)
            writer.writerow(
                {
                    "window_id": item.window_id,
                    "day": item.day,
                    "clock_time": item.target_clock_time,
                    "entity": item.entity,
                    "stream_type": item.stream_type,
                    "video_path": item.video_path,
                    "frame_path": str(item.frame_path),
                    "relevance_level": relevance,
                    "visible_cues": cues,
                    "notes": notes,
                    "needs_manual_review": review,
                }
            )


def write_qa_template(windows: list[CandidateWindow], output_path: Path, questions_per_window: int = 3) -> None:
    fieldnames = [
        "question_id",
        "window_id",
        "querying_user",
        "question",
        "answer",
        "self_view_sufficient",
        "oracle_incremental_views",
        "oracle_static_views",
        "expected_failure_self_only",
        "notes",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for window in windows:
            for index in range(questions_per_window):
                writer.writerow(
                    {
                        "question_id": f"{window.window_id}_Q{index + 1}",
                        "window_id": window.window_id,
                        "querying_user": "",
                        "question": "",
                        "answer": "",
                        "self_view_sufficient": "unclear",
                        "oracle_incremental_views": "",
                        "oracle_static_views": "",
                        "expected_failure_self_only": "",
                        "notes": "Fill manually after inspecting overview/contact sheets.",
                    }
                )


def read_annotation(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def try_clip_embeddings_local_only(items: list[StreamFrame], skip_clip: bool) -> tuple[dict[str, Any], str]:
    if skip_clip:
        return {}, "CLIP skipped by --skip-clip"
    # Use the existing CLIP helper, but report clearly if model/deps are missing.
    return clip_embeddings(items, skip_clip=False)


def embedding_key(item: StreamFrame) -> str:
    return f"{item.window_id}::{item.entity}"


def window_rows(items: list[StreamFrame], window_id_value: str) -> list[StreamFrame]:
    return [item for item in items if item.window_id == window_id_value and item.status == "ok"]


def annotation_index(annotation_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["window_id"], row["entity"]): row for row in annotation_rows}


def default_querying_user(rows: list[StreamFrame], annotations: dict[tuple[str, str], dict[str, str]]) -> str:
    direct_egos = [
        item.entity
        for item in rows
        if item.stream_type == "ego"
        and annotations.get((item.window_id, item.entity), {}).get("relevance_level") == "direct_event_detail"
    ]
    if direct_egos:
        return direct_egos[0]
    egos = [item.entity for item in rows if item.stream_type == "ego"]
    return egos[0] if egos else ""


def similarity(embeddings: dict[str, Any], window_id_value: str, a: str, b: str) -> float | None:
    key_a = f"{window_id_value}::{a}"
    key_b = f"{window_id_value}::{b}"
    if key_a not in embeddings or key_b not in embeddings:
        return None
    return cosine(embeddings[key_a], embeddings[key_b])


def convert_embedding_keys(items: list[StreamFrame], raw_embeddings: dict[str, Any]) -> dict[str, Any]:
    # clip_embeddings keys by entity; this is safe for one window only. For
    # multiple windows, compute per window and namespace keys here.
    converted = {}
    for item in items:
        if item.entity in raw_embeddings:
            converted[embedding_key(item)] = raw_embeddings[item.entity]
    return converted


def compute_embeddings_by_window(items: list[StreamFrame], skip_clip: bool) -> tuple[dict[str, Any], str]:
    all_embeddings = {}
    statuses = []
    for window_id_value in sorted({item.window_id for item in items}):
        subset = window_rows(items, window_id_value)
        raw, status = try_clip_embeddings_local_only(subset, skip_clip)
        statuses.append(f"{window_id_value}: {status}")
        all_embeddings.update(convert_embedding_keys(subset, raw))
    return all_embeddings, " | ".join(statuses)


def choose_other_ego(rows: list[StreamFrame], querying_user: str) -> list[str]:
    return [item.entity for item in rows if item.stream_type == "ego" and item.entity != querying_user]


def choose_best_static(
    rows: list[StreamFrame],
    querying_user: str,
    annotations: dict[tuple[str, str], dict[str, str]],
    embeddings: dict[str, Any],
) -> str:
    direct_static = [
        item.entity
        for item in rows
        if item.stream_type == "static"
        and annotations.get((item.window_id, item.entity), {}).get("relevance_level") == "direct_static_event"
    ]
    if direct_static:
        return direct_static[0]
    static_rows = [item for item in rows if item.stream_type == "static"]
    scored = []
    for item in static_rows:
        sim = similarity(embeddings, item.window_id, querying_user, item.entity)
        scored.append((-(sim if sim is not None else -1), STATIC_PRIORITY.index(item.entity) if item.entity in STATIC_PRIORITY else 99, item.entity))
    return sorted(scored)[0][2] if scored else ""


def strawman_selections(
    windows: list[CandidateWindow],
    overview_rows: list[StreamFrame],
    annotation_rows: list[dict[str, str]],
    embeddings: dict[str, Any],
) -> list[dict[str, str]]:
    random.seed(RANDOM_SEED)
    annotations = annotation_index(annotation_rows)
    output = []
    for window in windows:
        rows = window_rows(overview_rows, window.window_id)
        if not rows:
            continue
        querying_user = default_querying_user(rows, annotations)
        other_egos = choose_other_ego(rows, querying_user)
        random_other = random.choice(other_egos) if other_egos else ""

        most_similar = ""
        if other_egos:
            scored = [
                (-(similarity(embeddings, window.window_id, querying_user, entity) or -1), entity)
                for entity in other_egos
            ]
            most_similar = sorted(scored)[0][1]

        mmr_other = ""
        if other_egos:
            scored = [
                ((similarity(embeddings, window.window_id, querying_user, entity) if similarity(embeddings, window.window_id, querying_user, entity) is not None else 1), entity)
                for entity in other_egos
            ]
            mmr_other = sorted(scored)[0][1]

        best_static = choose_best_static(rows, querying_user, annotations, embeddings)
        all_entities = [item.entity for item in rows]
        oracle_entities = [
            item.entity
            for item in rows
            if annotations.get((window.window_id, item.entity), {}).get("relevance_level") in INCREMENTAL_EVIDENCE_LEVELS
        ]
        if querying_user and querying_user not in oracle_entities:
            oracle_entities.insert(0, querying_user)

        method_to_entities = {
            "self-only": [querying_user] if querying_user else [],
            "self + random other ego": [querying_user, random_other] if random_other else [querying_user],
            "self + most CLIP-similar other ego": [querying_user, most_similar] if most_similar else [querying_user],
            "self + MMR incremental other ego": [querying_user, mmr_other] if mmr_other else [querying_user],
            "self + best static": [querying_user, best_static] if best_static else [querying_user],
            "all active views": all_entities,
            "oracle views": oracle_entities,
        }

        path_by_entity = {item.entity: item.video_path for item in rows}
        frame_by_entity = {item.entity: str(item.frame_path) for item in rows}
        for method, entities in method_to_entities.items():
            deduped = []
            for entity in entities:
                if entity and entity not in deduped:
                    deduped.append(entity)
            output.append(
                {
                    "window_id": window.window_id,
                    "day": window.day,
                    "clock_time": seconds_to_time(window.start_seconds + 60),
                    "querying_user": querying_user,
                    "method": method,
                    "selected_entities": ";".join(deduped),
                    "selected_video_paths": ";".join(path_by_entity.get(entity, "") for entity in deduped),
                    "selected_frame_paths": ";".join(frame_by_entity.get(entity, "") for entity in deduped),
                    "num_views_used": str(len(deduped)),
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def metric_value(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def compute_window_metric(
    candidate: dict[str, str],
    annotations: dict[tuple[str, str], dict[str, str]],
    embeddings: dict[str, Any],
) -> dict[str, str]:
    window_id_value = candidate["window_id"]
    selected = [entity for entity in candidate["selected_entities"].split(";") if entity]
    querying_user = candidate["querying_user"]
    window_annotations = {
        entity: row
        for (wid, entity), row in annotations.items()
        if wid == window_id_value and row.get("relevance_level") in RELEVANCE_LEVELS
    }
    has_manualish_labels = any(row.get("needs_manual_review") == "no" for row in window_annotations.values())
    direct = {entity for entity, row in window_annotations.items() if row.get("relevance_level") in DIRECT_EVIDENCE_LEVELS}
    incremental = {entity for entity, row in window_annotations.items() if row.get("relevance_level") in INCREMENTAL_EVIDENCE_LEVELS and entity != querying_user}
    off_event = {entity for entity, row in window_annotations.items() if row.get("relevance_level") in OFF_EVENT_LEVELS}

    direct_recall = len(set(selected) & direct) / len(direct) if direct else None
    incremental_recall = len(set(selected) & incremental) / len(incremental) if incremental else None
    off_event_rate = len(set(selected) & off_event) / len(selected) if selected and has_manualish_labels else None
    sims = [
        similarity(embeddings, window_id_value, querying_user, entity)
        for entity in selected
        if entity != querying_user and similarity(embeddings, window_id_value, querying_user, entity) is not None
    ]
    duplicate_rate = sum(sims) / len(sims) if sims else None
    return {
        "window_id": window_id_value,
        "method": candidate["method"],
        "querying_user": querying_user,
        "num_views_used": candidate["num_views_used"],
        "answer_accuracy_if_labels_available": "",
        "evidence_sufficiency_manual": "",
        "direct_evidence_recall_at_k": metric_value(direct_recall),
        "incremental_evidence_recall_at_k": metric_value(incremental_recall),
        "duplicate_rate_with_self": metric_value(duplicate_rate),
        "off_event_access_rate": metric_value(off_event_rate),
        "average_number_of_views_used": candidate["num_views_used"],
        "notes": "Manual QA labels not yet filled; accuracy/sufficiency require annotation.",
    }


def write_metrics(
    candidate_rows: list[dict[str, str]],
    annotation_rows: list[dict[str, str]],
    embeddings: dict[str, Any],
    window_output_path: Path,
    aggregate_output_path: Path,
) -> None:
    annotations = annotation_index(annotation_rows)
    window_metrics = [compute_window_metric(row, annotations, embeddings) for row in candidate_rows]
    metric_fields = [
        "window_id",
        "method",
        "querying_user",
        "num_views_used",
        "answer_accuracy_if_labels_available",
        "evidence_sufficiency_manual",
        "direct_evidence_recall_at_k",
        "incremental_evidence_recall_at_k",
        "duplicate_rate_with_self",
        "off_event_access_rate",
        "average_number_of_views_used",
        "notes",
    ]
    write_csv(window_output_path, window_metrics, metric_fields)

    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in window_metrics:
        by_method[row["method"]].append(row)

    aggregate_rows = []
    for method, rows in sorted(by_method.items()):
        aggregate_rows.append(
            {
                "method": method,
                "num_windows": str(len(rows)),
                "mean_direct_evidence_recall_at_k": average_field(rows, "direct_evidence_recall_at_k"),
                "mean_incremental_evidence_recall_at_k": average_field(rows, "incremental_evidence_recall_at_k"),
                "mean_duplicate_rate_with_self": average_field(rows, "duplicate_rate_with_self"),
                "mean_off_event_access_rate": average_field(rows, "off_event_access_rate"),
                "average_number_of_views_used": average_field(rows, "average_number_of_views_used"),
                "answer_accuracy_if_labels_available": "",
                "evidence_sufficiency_manual": "",
                "notes": "Aggregate scaffold; fill QA/evidence labels for final comparison.",
            }
        )
    write_csv(
        aggregate_output_path,
        aggregate_rows,
        [
            "method",
            "num_windows",
            "mean_direct_evidence_recall_at_k",
            "mean_incremental_evidence_recall_at_k",
            "mean_duplicate_rate_with_self",
            "mean_off_event_access_rate",
            "average_number_of_views_used",
            "answer_accuracy_if_labels_available",
            "evidence_sufficiency_manual",
            "notes",
        ],
    )


def average_field(rows: list[dict[str, str]], field: str) -> str:
    values = []
    for row in rows:
        value = row.get(field, "")
        if value == "":
            continue
        try:
            values.append(float(value))
        except ValueError:
            continue
    if not values:
        return ""
    return f"{sum(values) / len(values):.6f}"


def write_report(
    *,
    output_path: Path,
    selected_windows: list[CandidateWindow],
    overview_rows: list[StreamFrame],
    clip_status: str,
    local_storage_bytes: int,
    output_dir: Path,
) -> None:
    ok = sum(1 for row in overview_rows if row.status == "ok")
    total = len(overview_rows)
    lines = [
        "CASTLE incremental QA POC report",
        "",
        f"Selected windows: {len(selected_windows)}",
        f"Overview frames extracted: {ok}/{total}",
        f"Local tracked storage: {local_storage_bytes / (1024**2):.3f} MB",
        f"CLIP status: {clip_status}",
        "hf_hub_download for videos: NO",
        "",
        "Selected windows:",
    ]
    for window in selected_windows:
        lines.append(
            f"  {window.window_id}: {window.day} {window.period} "
            f"{seconds_to_time(window.start_seconds)}-{seconds_to_time(window.end_seconds)} "
            f"active={len(window.active)} ego={len(window.ego)} static={len(window.static)}"
        )
    lines.extend(
        [
            "",
            "POC interpretation:",
            "  The generated annotation CSV is the manual labeling surface for direct/off-event evidence.",
            "  The QA template is the manual question/answer surface.",
            "  Strawman metrics are scaffolded now; answer accuracy and evidence sufficiency become meaningful after QA labels are filled.",
            "",
            f"Output directory: {output_dir}",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir, logs_dir = extraction_paths(args.output_dir)
    max_bytes = int(args.max_local_mb * 1024 * 1024)
    tracked_paths = [frames_dir, logs_dir, args.output_dir]

    intervals = read_intervals(args.interval_table)
    candidates = build_candidate_windows(
        intervals,
        window_minutes=args.window_minutes,
        min_active_streams=args.min_active_streams,
        min_static_streams=args.min_static_streams,
    )
    selected_windows = select_diverse_windows(candidates, args.num_windows)
    write_selected_windows(selected_windows, args.output_dir / "castle_poc_selected_windows.csv")

    overview_rows: list[StreamFrame] = []
    for window in selected_windows:
        extracted = extract_window_overview(
            window,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            frames_dir=frames_dir,
            logs_dir=logs_dir,
            timeout_sec=args.timeout_sec,
            tracked_paths=tracked_paths,
            max_bytes=max_bytes,
            target_offset_sec=args.overview_target_offset_sec,
            dry_run=args.dry_run,
            max_consecutive_failures=args.max_consecutive_failures,
        )
        overview_rows.extend(extracted)
        write_stream_frame_log(extracted, args.output_dir / f"castle_poc_overview_extraction_log_{window.window_id}.csv")
        make_overview_sheet(extracted, args.output_dir / f"castle_poc_overview_contact_sheet_{window.window_id}.png")
        if any(item.status in {"ffmpeg_timeout", "ffmpeg_missing"} for item in extracted):
            break
        if (
            not args.dry_run
            and args.max_consecutive_failures > 0
            and extracted
            and all(item.status != "ok" for item in extracted)
        ):
            break

    write_all_extraction_logs(overview_rows, args.output_dir / "castle_poc_overview_extraction_log_all.csv")
    write_view_annotation(overview_rows, args.output_dir / "castle_poc_view_annotation.csv")
    write_qa_template(selected_windows, args.output_dir / "castle_self_vs_other_qa_template.csv")

    annotation_rows = read_annotation(args.output_dir / "castle_poc_view_annotation.csv")
    embeddings, clip_status = compute_embeddings_by_window(overview_rows, args.skip_clip or args.dry_run)
    candidate_rows = strawman_selections(selected_windows, overview_rows, annotation_rows, embeddings)
    write_csv(
        args.output_dir / "castle_poc_strawman_view_candidates.csv",
        candidate_rows,
        [
            "window_id",
            "day",
            "clock_time",
            "querying_user",
            "method",
            "selected_entities",
            "selected_video_paths",
            "selected_frame_paths",
            "num_views_used",
        ],
    )
    write_metrics(
        candidate_rows,
        annotation_rows,
        embeddings,
        args.output_dir / "castle_poc_strawman_window_metrics.csv",
        args.output_dir / "castle_poc_strawman_metrics.csv",
    )
    local_storage = tracked_size([args.output_dir])
    write_report(
        output_path=args.output_dir / "castle_incremental_qa_poc_report.txt",
        selected_windows=selected_windows,
        overview_rows=overview_rows,
        clip_status=clip_status,
        local_storage_bytes=local_storage,
        output_dir=args.output_dir,
    )

    print(f"Selected windows: {len(selected_windows)}")
    print(f"Overview frames extracted: {sum(1 for row in overview_rows if row.status == 'ok')}/{len(overview_rows)}")
    print(f"CLIP status: {clip_status}")
    print(f"Local tracked storage: {local_storage / (1024**2):.3f} MB")
    print("hf_hub_download for videos: NO")
    print(f"Wrote {args.output_dir / 'castle_poc_selected_windows.csv'}")
    print(f"Wrote {args.output_dir / 'castle_poc_view_annotation.csv'}")
    print(f"Wrote {args.output_dir / 'castle_self_vs_other_qa_template.csv'}")
    print(f"Wrote {args.output_dir / 'castle_poc_strawman_view_candidates.csv'}")
    print(f"Wrote {args.output_dir / 'castle_poc_strawman_metrics.csv'}")
    print(f"Wrote {args.output_dir / 'castle_incremental_qa_poc_report.txt'}")


if __name__ == "__main__":
    main()
