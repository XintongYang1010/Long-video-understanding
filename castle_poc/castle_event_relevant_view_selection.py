#!/usr/bin/env python3
"""
CASTLE event-relevant multi-view selection for one clock-time window.

Pipeline:
  1. Extract one overview frame from every active stream in a selected window.
  2. Build an overview contact sheet.
  3. Optionally compute CLIP image similarity between views.
  4. Select event-relevant living-room views anchored by Onanong/Tien.
  5. Extract three frames for selected views and create a contact sheet.

Safety:
  - Never calls hf_hub_download for videos.
  - Uses ffmpeg remote URLs from hf_hub_url.
  - Uses clock-time offsets from castle_main_video_interval_table.csv.
  - Stops if tracked local artifacts exceed --max-local-mb.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_url


REPO_ID = "CASTLE-Dataset/CASTLE2024"
REPO_TYPE = "dataset"
DEFAULT_WINDOW_ID = "DAY1_100500000"
STATIC_DISPLAY_PRIORITY = ["Meeting", "Living1", "Living2", "Kitchen", "Reading"]
ANCHOR_EGOS = ["Onanong", "Tien"]
DEFAULT_EXCLUDED_ENTITIES = {
    "Kitchen": "excluded: active but off-event kitchen view with no visible living-room presentation",
    "Reading": "excluded: active but off-event/empty reading-room view",
    "Klaus": "excluded: test/calibration-like ego view, not part of the living-room event",
}


@dataclass(frozen=True)
class Interval:
    day: str
    entity: str
    stream_type: str
    segment_id: str
    video_path: str
    start_time: str
    end_time: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float


@dataclass(frozen=True)
class Window:
    window_id: str
    day: str
    period: str
    start_time: str
    end_time: str
    start_seconds: float
    end_seconds: float
    active_ego_entities: tuple[str, ...]
    active_static_entities: tuple[str, ...]


@dataclass(frozen=True)
class StreamFrame:
    window_id: str
    day: str
    entity: str
    stream_type: str
    segment_id: str
    video_path: str
    video_start_time: str
    video_end_time: str
    target_clock_time: str
    offset_seconds: float
    remote_url: str
    frame_path: Path
    log_path: Path
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select event-relevant CASTLE views from one clock-time overlap window.")
    parser.add_argument("--interval-table", type=Path, default=Path("castle_main_video_interval_table.csv"))
    parser.add_argument("--selected-windows", type=Path, default=Path("castle_selected_multiview_windows.csv"))
    parser.add_argument("--window-id", default=DEFAULT_WINDOW_ID)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--repo-type", default=REPO_TYPE)
    parser.add_argument("--frames-dir", type=Path, default=Path("castle_event_relevant_frames"))
    parser.add_argument("--logs-dir", type=Path, default=Path("castle_event_relevant_ffmpeg_logs"))
    parser.add_argument("--max-local-mb", type=float, default=500.0)
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--max-active-streams", type=int, default=15)
    parser.add_argument("--overview-target-offset-sec", type=float, default=60.0)
    parser.add_argument("--event-offsets-sec", type=float, nargs="+", default=[60.0, 120.0, 180.0])
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument(
        "--manual-selected",
        nargs="*",
        default=[],
        help="Manual event-relevant entities to select when CLIP is skipped, e.g. Onanong Tien Meeting Living1.",
    )
    parser.add_argument(
        "--manual-excluded",
        nargs="*",
        default=[],
        help="Additional entities to force-exclude from event-relevant selection.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build selections without running ffmpeg.")
    return parser.parse_args()


def parse_clock_seconds(value: str) -> float | None:
    match = re.search(r"\b(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d+))?\b", str(value))
    if not match:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    fraction = match.group(4) or ""
    fractional_seconds = float(f"0.{fraction}") if fraction else 0.0
    return hours * 3600 + minutes * 60 + seconds + fractional_seconds


def seconds_to_time(seconds: float) -> str:
    seconds = seconds % (24 * 3600)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    whole_seconds = int(seconds % 60)
    millis = int(round((seconds - math.floor(seconds)) * 1000))
    if millis == 1000:
        whole_seconds += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def safe_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def read_intervals(path: Path) -> list[Interval]:
    intervals = []
    for row in read_csv_rows(path):
        if row["time_parse_status"] != "ok":
            continue
        start = parse_clock_seconds(row["start_time"])
        end = parse_clock_seconds(row["end_time"])
        if start is None or end is None:
            continue
        if end < start:
            end += 24 * 3600
        intervals.append(
            Interval(
                day=row["day"],
                entity=row["entity"],
                stream_type=row["stream_type"],
                segment_id=row["segment_id"],
                video_path=row["video_path"],
                start_time=row["start_time"],
                end_time=row["end_time"],
                start_seconds=start,
                end_seconds=end,
                duration_seconds=safe_float(row["duration_seconds"]),
            )
        )
    return intervals


def load_window(path: Path, window_id: str) -> Window:
    rows = read_csv_rows(path)
    row = next((item for item in rows if item["window_id"] == window_id), None)
    if row is None:
        raise RuntimeError(f"window_id not found: {window_id}")
    start = parse_clock_seconds(row["window_start"])
    end = parse_clock_seconds(row["window_end"])
    if start is None or end is None:
        raise RuntimeError(f"cannot parse window times for {window_id}")
    return Window(
        window_id=row["window_id"],
        day=row["day"],
        period=row["period"],
        start_time=row["window_start"],
        end_time=row["window_end"],
        start_seconds=start,
        end_seconds=end,
        active_ego_entities=tuple(entity for entity in row["active_ego_entities"].split(";") if entity),
        active_static_entities=tuple(entity for entity in row["active_static_entities"].split(";") if entity),
    )


def active_intervals_for_window(window: Window, intervals: list[Interval], max_streams: int) -> list[Interval]:
    active_entities = set(window.active_ego_entities + window.active_static_entities)
    active = [
        interval
        for interval in intervals
        if interval.day == window.day
        and interval.entity in active_entities
        and interval.start_seconds <= window.start_seconds
        and interval.end_seconds >= window.end_seconds
    ]
    ego_rank = {entity: index for index, entity in enumerate(window.active_ego_entities)}
    static_rank = {entity: index for index, entity in enumerate(STATIC_DISPLAY_PRIORITY)}
    active.sort(
        key=lambda item: (
            0 if item.stream_type == "ego" else 1,
            ego_rank.get(item.entity, 99) if item.stream_type == "ego" else static_rank.get(item.entity, 99),
            item.entity,
        )
    )
    return active[:max_streams]


def make_stream_frame(
    window: Window,
    interval: Interval,
    target_clock_seconds: float,
    *,
    repo_id: str,
    repo_type: str,
    frames_dir: Path,
    logs_dir: Path,
    label: str,
    status: str = "pending",
) -> StreamFrame:
    target_clock_time = seconds_to_time(target_clock_seconds)
    safe_time = target_clock_time.replace(":", "").replace(".", "")
    frame_path = frames_dir / window.window_id / label / f"{interval.entity}_{safe_time}.jpg"
    log_path = logs_dir / window.window_id / label / f"{interval.entity}_{safe_time}.stderr.txt"
    return StreamFrame(
        window_id=window.window_id,
        day=interval.day,
        entity=interval.entity,
        stream_type=interval.stream_type,
        segment_id=interval.segment_id,
        video_path=interval.video_path,
        video_start_time=interval.start_time,
        video_end_time=interval.end_time,
        target_clock_time=target_clock_time,
        offset_seconds=target_clock_seconds - interval.start_seconds,
        remote_url=hf_hub_url(repo_id=repo_id, filename=interval.video_path, repo_type=repo_type),
        frame_path=frame_path,
        log_path=log_path,
        status=status,
    )


def tracked_size(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            total += path.stat().st_size
        else:
            total += sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
    return total


def enforce_storage_cap(paths: list[Path], max_bytes: int) -> None:
    size = tracked_size(paths)
    if size > max_bytes:
        raise RuntimeError(f"local storage cap exceeded: {size} bytes > {max_bytes} bytes")


def redact_command(command: list[str]) -> str:
    output = []
    redact_next = False
    for part in command:
        if redact_next:
            output.append("Authorization: Bearer ***")
            redact_next = False
            continue
        output.append(part)
        if part == "-headers":
            redact_next = True
    return " ".join(output)


def ffmpeg_extract_frame(item: StreamFrame, timeout_sec: int, tracked_paths: list[Path], max_bytes: int, dry_run: bool) -> StreamFrame:
    if item.offset_seconds < 0:
        return replace_status(item, "skip_negative_offset")
    enforce_storage_cap(tracked_paths, max_bytes)
    if dry_run:
        return replace_status(item, "dry_run_not_extracted")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return replace_status(item, "ffmpeg_missing")

    item.frame_path.parent.mkdir(parents=True, exist_ok=True)
    item.log_path.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN", "").strip()
    command = [
        ffmpeg,
        "-hide_banner",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-rw_timeout",
        "30000000",
        "-reconnect",
        "1",
        "-reconnect_on_network_error",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        "5",
        "-multiple_requests",
        "1",
        "-seekable",
        "1",
    ]
    if token:
        command.extend(["-headers", f"Authorization: Bearer {token}"])
    command.extend(
        [
            "-ss",
            f"{item.offset_seconds:.3f}",
            "-i",
            item.remote_url,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(item.frame_path),
        ]
    )

    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_sec, check=False)
        status = "ok" if completed.returncode == 0 and item.frame_path.exists() and item.frame_path.stat().st_size > 0 else "ffmpeg_failed"
        item.log_path.write_text(
            "\n".join(
                [
                    f"COMMAND: {redact_command(command)}",
                    f"RETURN_CODE: {completed.returncode}",
                    "STDOUT:",
                    completed.stdout or "",
                    "STDERR:",
                    completed.stderr or "",
                ]
            ),
            encoding="utf-8",
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        status = "ffmpeg_timeout"
        item.log_path.write_text(
            f"COMMAND: {redact_command(command)}\nTIMEOUT_AFTER_SEC: {timeout_sec}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}",
            encoding="utf-8",
        )
    enforce_storage_cap(tracked_paths, max_bytes)
    return replace_status(item, status)


def extract_frames_sequential(
    frames: list[StreamFrame],
    timeout_sec: int,
    tracked_paths: list[Path],
    max_bytes: int,
    dry_run: bool,
) -> list[StreamFrame]:
    outputs = []
    for frame in frames:
        extracted = ffmpeg_extract_frame(frame, timeout_sec, tracked_paths, max_bytes, dry_run)
        outputs.append(extracted)
        if extracted.status in {"ffmpeg_timeout", "ffmpeg_missing"}:
            break
    return outputs


def replace_status(item: StreamFrame, status: str) -> StreamFrame:
    return StreamFrame(
        window_id=item.window_id,
        day=item.day,
        entity=item.entity,
        stream_type=item.stream_type,
        segment_id=item.segment_id,
        video_path=item.video_path,
        video_start_time=item.video_start_time,
        video_end_time=item.video_end_time,
        target_clock_time=item.target_clock_time,
        offset_seconds=item.offset_seconds,
        remote_url=item.remote_url,
        frame_path=item.frame_path,
        log_path=item.log_path,
        status=status,
    )


def write_stream_frame_log(items: list[StreamFrame], output_path: Path) -> None:
    fieldnames = [
        "window_id",
        "day",
        "entity",
        "stream_type",
        "segment_id",
        "video_path",
        "video_start_time",
        "video_end_time",
        "target_clock_time",
        "offset_seconds",
        "frame_path",
        "log_path",
        "status",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "window_id": item.window_id,
                    "day": item.day,
                    "entity": item.entity,
                    "stream_type": item.stream_type,
                    "segment_id": item.segment_id,
                    "video_path": item.video_path,
                    "video_start_time": item.video_start_time,
                    "video_end_time": item.video_end_time,
                    "target_clock_time": item.target_clock_time,
                    "offset_seconds": f"{item.offset_seconds:.3f}",
                    "frame_path": str(item.frame_path),
                    "log_path": str(item.log_path),
                    "status": item.status,
                }
            )


def make_overview_sheet(items: list[StreamFrame], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    ok_items = [item for item in items if item.status == "ok" and item.frame_path.exists()]
    if not ok_items:
        return
    cols = 3
    thumb_w, thumb_h = 340, 200
    card_w, card_h = 420, 300
    rows = math.ceil(len(ok_items) / cols)
    sheet = Image.new("RGB", (cols * card_w, rows * card_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(ok_items):
        col = index % cols
        row = index // cols
        x = col * card_w
        y = row * card_h
        draw.rectangle((x, y, x + card_w - 1, y + card_h - 1), outline="gray")
        image = Image.open(item.frame_path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        sheet.paste(image, (x + (card_w - image.width) // 2, y + 8))
        label = f"{item.entity} ({item.stream_type})\n{item.target_clock_time}\n{item.video_path}"
        draw.text((x + 8, y + thumb_h + 22), label, fill="black")
    sheet.save(output_path)


def clip_embeddings(items: list[StreamFrame], skip_clip: bool) -> tuple[dict[str, Any], str]:
    if skip_clip:
        return {}, "CLIP skipped by --skip-clip"
    ok_items = [item for item in items if item.status == "ok" and item.frame_path.exists()]
    if not ok_items:
        return {}, "CLIP skipped: no extracted overview frames"
    try:
        import numpy as np
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except Exception as exc:  # noqa: BLE001
        return {}, f"CLIP skipped: missing dependency ({exc})"

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        model.eval()
        embeddings: dict[str, Any] = {}
        with torch.no_grad():
            for item in ok_items:
                image = Image.open(item.frame_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                features = model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                embeddings[item.entity] = features.detach().cpu().numpy()[0].astype(np.float32)
        return embeddings, f"CLIP embeddings computed for {len(embeddings)} frames"
    except Exception as exc:  # noqa: BLE001
        return {}, f"CLIP skipped: model load/embedding failed ({exc})"


def cosine(a: Any, b: Any) -> float:
    import numpy as np

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def write_similarity_csv(items: list[StreamFrame], embeddings: dict[str, Any], anchor_sims: dict[str, float], output_path: Path) -> None:
    fieldnames = [
        "source_entity",
        "source_stream_type",
        "source_similarity_to_anchor",
        "target_entity",
        "target_stream_type",
        "target_similarity_to_anchor",
        "cosine_similarity",
        "source_frame_path",
        "target_frame_path",
    ]
    ok_items = [item for item in items if item.entity in embeddings]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for source in ok_items:
            for target in ok_items:
                writer.writerow(
                    {
                        "source_entity": source.entity,
                        "source_stream_type": source.stream_type,
                        "source_similarity_to_anchor": f"{anchor_sims[source.entity]:.6f}" if source.entity in anchor_sims else "",
                        "target_entity": target.entity,
                        "target_stream_type": target.stream_type,
                        "target_similarity_to_anchor": f"{anchor_sims[target.entity]:.6f}" if target.entity in anchor_sims else "",
                        "cosine_similarity": f"{cosine(embeddings[source.entity], embeddings[target.entity]):.6f}",
                        "source_frame_path": str(source.frame_path),
                        "target_frame_path": str(target.frame_path),
                    }
                )


def forced_exclusions(extra_excluded: list[str]) -> dict[str, str]:
    exclusions = dict(DEFAULT_EXCLUDED_ENTITIES)
    for entity in extra_excluded:
        exclusions[entity] = "excluded by --manual-excluded"
    return exclusions


def selected_static_count(selected_entities: list[str], by_entity: dict[str, StreamFrame]) -> int:
    return sum(1 for entity in selected_entities if entity in by_entity and by_entity[entity].stream_type == "static")


def compute_anchor_similarities(ok_items: list[StreamFrame], embeddings: dict[str, Any], selected_anchor_entities: list[str]) -> dict[str, float]:
    if not embeddings or not selected_anchor_entities or not all(entity in embeddings for entity in selected_anchor_entities):
        return {}
    import numpy as np

    anchor = np.mean([embeddings[entity] for entity in selected_anchor_entities], axis=0)
    anchor = anchor / np.linalg.norm(anchor)
    return {item.entity: cosine(embeddings[item.entity], anchor) for item in ok_items if item.entity in embeddings}


def select_event_relevant_views(
    items: list[StreamFrame],
    embeddings: dict[str, Any],
    *,
    manual_selected: list[str],
    manual_excluded: list[str],
    clip_skipped: bool,
) -> tuple[list[StreamFrame], dict[str, str], dict[str, float]]:
    ok_items = [item for item in items if item.status == "ok" and item.frame_path.exists()]
    by_entity = {item.entity: item for item in ok_items}
    exclusions = forced_exclusions(manual_excluded)
    selected_entities = [entity for entity in ANCHOR_EGOS if entity in by_entity and entity not in exclusions]
    reasons = {entity: "selected: anchor ego view of the living-room presentation event" for entity in selected_entities}
    for entity, reason in exclusions.items():
        if entity in by_entity:
            reasons[entity] = reason

    anchor_sims = compute_anchor_similarities(ok_items, embeddings, selected_entities)

    if clip_skipped or not embeddings:
        for entity in manual_selected:
            if entity not in by_entity:
                continue
            if entity in exclusions:
                reasons[entity] = f"{exclusions[entity]}; not selected despite --manual-selected"
                continue
            if by_entity[entity].stream_type == "static" and selected_static_count(selected_entities, by_entity) >= 2:
                reasons[entity] = "excluded: manual selection already has two static views"
                continue
            if entity not in selected_entities:
                selected_entities.append(entity)
            if entity == "Meeting":
                reasons[entity] = "selected by --manual-selected: manual inspection indicates Meeting is directly relevant to the group event"
            elif entity in ANCHOR_EGOS:
                reasons[entity] = "selected by --manual-selected: anchor ego view of the same living-room event"
            else:
                reasons[entity] = "selected by --manual-selected: event-relevant context view"
    else:
        static_candidates = [
            item
            for item in ok_items
            if item.stream_type == "static" and item.entity not in exclusions
        ]
        static_rank = {entity: index for index, entity in enumerate(STATIC_DISPLAY_PRIORITY)}
        static_candidates.sort(
            key=lambda item: (
                -anchor_sims.get(item.entity, -1.0),
                static_rank.get(item.entity, 99),
                item.entity,
            )
        )
        for item in static_candidates:
            if selected_static_count(selected_entities, by_entity) >= 2:
                reasons[item.entity] = "excluded: lower-ranked static view after selecting two event-relevant static views"
                continue
            selected_entities.append(item.entity)
            reasons[item.entity] = (
                "selected: top static view by CLIP similarity to Onanong/Tien anchor "
                f"({anchor_sims.get(item.entity, 0.0):.3f})"
            )

    for item in ok_items:
        reasons.setdefault(item.entity, "excluded: not in event-relevant living-room group")

    deduped = []
    seen = set()
    for entity in selected_entities:
        if entity in by_entity and entity not in seen:
            deduped.append(by_entity[entity])
            seen.add(entity)
    return deduped, reasons, anchor_sims


def write_event_relevant_csv(items: list[StreamFrame], selected: list[StreamFrame], reasons: dict[str, str], anchor_sims: dict[str, float], output_path: Path) -> None:
    selected_entities = {item.entity for item in selected}
    fieldnames = [
        "selected",
        "entity",
        "stream_type",
        "video_path",
        "overview_frame_path",
        "target_clock_time",
        "similarity_to_onanong_tien_anchor",
        "reason",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            if item.status != "ok":
                continue
            writer.writerow(
                {
                    "selected": "yes" if item.entity in selected_entities else "no",
                    "entity": item.entity,
                    "stream_type": item.stream_type,
                    "video_path": item.video_path,
                    "overview_frame_path": str(item.frame_path),
                    "target_clock_time": item.target_clock_time,
                    "similarity_to_onanong_tien_anchor": f"{anchor_sims[item.entity]:.6f}" if item.entity in anchor_sims else "",
                    "reason": reasons.get(item.entity, ""),
                }
            )


def extract_event_frames(
    window: Window,
    selected: list[StreamFrame],
    *,
    repo_id: str,
    repo_type: str,
    frames_dir: Path,
    logs_dir: Path,
    event_offsets_sec: list[float],
    timeout_sec: int,
    tracked_paths: list[Path],
    max_bytes: int,
    dry_run: bool,
) -> list[StreamFrame]:
    selected_intervals = [
        Interval(
            day=item.day,
            entity=item.entity,
            stream_type=item.stream_type,
            segment_id=item.segment_id,
            video_path=item.video_path,
            start_time=item.video_start_time,
            end_time=item.video_end_time,
            start_seconds=(parse_clock_seconds(item.target_clock_time) or window.start_seconds) - item.offset_seconds,
            end_seconds=0.0,
            duration_seconds=999999.0,
        )
        for item in selected
    ]
    outputs = []
    for interval in selected_intervals:
        for offset in event_offsets_sec:
            target_clock_seconds = window.start_seconds + offset
            frame = make_stream_frame(
                window,
                interval,
                target_clock_seconds,
                repo_id=repo_id,
                repo_type=repo_type,
                frames_dir=frames_dir,
                logs_dir=logs_dir,
                label="event",
            )
            extracted = ffmpeg_extract_frame(frame, timeout_sec, tracked_paths, max_bytes, dry_run)
            outputs.append(extracted)
            if extracted.status in {"ffmpeg_timeout", "ffmpeg_missing"}:
                return outputs
    return outputs


def make_event_contact_sheet(items: list[StreamFrame], output_path: Path) -> None:
    from PIL import Image, ImageDraw

    ok_items = [item for item in items if item.status == "ok" and item.frame_path.exists()]
    if not ok_items:
        return
    entities = []
    times = []
    for item in ok_items:
        if item.entity not in entities:
            entities.append(item.entity)
        if item.target_clock_time not in times:
            times.append(item.target_clock_time)
    by_key = {(item.entity, item.target_clock_time): item for item in ok_items}
    thumb_w, thumb_h = 320, 190
    label_w = 430
    header_h = 40
    row_h = 245
    sheet = Image.new("RGB", (label_w + len(times) * thumb_w, header_h + len(entities) * row_h), "white")
    draw = ImageDraw.Draw(sheet)
    for col, time_str in enumerate(times):
        draw.text((label_w + col * thumb_w + 8, 12), time_str, fill="black")
    for row_index, entity in enumerate(entities):
        sample = next(item for item in ok_items if item.entity == entity)
        y = header_h + row_index * row_h
        label = f"{sample.entity} ({sample.stream_type})\n{sample.video_path}"
        draw.text((8, y + 12), label, fill="black")
        for col, time_str in enumerate(times):
            x = label_w + col * thumb_w
            item = by_key.get((entity, time_str))
            if not item:
                draw.rectangle((x, y, x + thumb_w - 1, y + thumb_h - 1), outline="gray")
                continue
            image = Image.open(item.frame_path).convert("RGB")
            image.thumbnail((thumb_w, thumb_h))
            sheet.paste(image, (x + (thumb_w - image.width) // 2, y + 8 + (thumb_h - image.height) // 2))
            draw.text((x + 8, y + thumb_h + 16), f"offset={item.offset_seconds:.1f}s", fill="black")
    sheet.save(output_path)


def write_report(
    *,
    window: Window,
    active: list[Interval],
    overview: list[StreamFrame],
    selected: list[StreamFrame],
    reasons: dict[str, str],
    clip_status: str,
    event_frames: list[StreamFrame],
    local_storage_bytes: int,
    output_path: Path,
) -> None:
    ok_overview = [item for item in overview if item.status == "ok"]
    selected_entities = {item.entity for item in selected}
    irrelevant = [item.entity for item in ok_overview if item.entity not in selected_entities]
    static_selected = [item.entity for item in selected if item.stream_type == "static"]
    lines = [
        "CASTLE event-relevant view selection report",
        "",
        f"Window: {window.window_id} {window.day} {window.start_time}-{window.end_time}",
        f"Overview target time: {seconds_to_time(window.start_seconds + 60)}",
        f"Local tracked storage used: {local_storage_bytes / (1024**2):.3f} MB",
        f"CLIP status: {clip_status}",
        "",
        "1. How many active streams existed in this window?",
        f"   {len(active)} active streams were eligible; {len(ok_overview)}/{len(active)} overview frames extracted successfully.",
        "",
        "2. Which streams are visually aligned to the living-room event?",
        f"   {', '.join(item.entity for item in selected) if selected else 'none selected'}",
        "",
        "3. Which streams are active but irrelevant?",
        f"   {', '.join(irrelevant) if irrelevant else 'none identified'}",
        "",
        "4. Do static views complement ego views?",
        (
            f"   Yes, selected static context: {', '.join(static_selected)}."
            if static_selected
            else "   Not confirmed; no static views were selected."
        ),
        "",
        "5. Does this support multi-view collaborative memory?",
        (
            "   Yes if the generated contact sheet shows the selected ego and static rows tracking the same event over time."
            if selected and static_selected
            else "   Weak/unclear from this run."
        ),
        "",
        "6. Should we scale this event-relevant selection to 3 windows?",
        (
            "   Yes, after manual inspection of the overview and event-relevant contact sheet."
            if selected and static_selected
            else "   Not yet; inspect failures or tune selection first."
        ),
        "",
        "Selection reasons:",
    ]
    for item in ok_overview:
        lines.append(f"   {item.entity}: {reasons.get(item.entity, '')}")
    if event_frames:
        ok_event = sum(1 for item in event_frames if item.status == "ok")
        lines.append("")
        lines.append(f"Event contact-sheet frame extractions: {ok_event}/{len(event_frames)}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    max_bytes = int(args.max_local_mb * 1024 * 1024)
    tracked_paths = [
        args.frames_dir,
        args.logs_dir,
        Path(f"castle_window_overview_{args.window_id}.png"),
        Path(f"castle_event_relevant_contact_sheet_{args.window_id}.png"),
    ]
    intervals = read_intervals(args.interval_table)
    window = load_window(args.selected_windows, args.window_id)
    active = active_intervals_for_window(window, intervals, args.max_active_streams)

    overview_target = window.start_seconds + args.overview_target_offset_sec
    overview_frames = [
        make_stream_frame(
            window,
            interval,
            overview_target,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            frames_dir=args.frames_dir,
            logs_dir=args.logs_dir,
            label="overview",
        )
        for interval in active
    ]
    extracted_overview = extract_frames_sequential(overview_frames, args.timeout_sec, tracked_paths, max_bytes, args.dry_run)
    write_stream_frame_log(extracted_overview, Path(f"castle_window_overview_extraction_log_{args.window_id}.csv"))
    make_overview_sheet(extracted_overview, Path(f"castle_window_overview_{args.window_id}.png"))

    embeddings, clip_status = clip_embeddings(extracted_overview, args.skip_clip or args.dry_run)
    selected, reasons, anchor_sims = select_event_relevant_views(
        extracted_overview,
        embeddings,
        manual_selected=args.manual_selected,
        manual_excluded=args.manual_excluded,
        clip_skipped=args.skip_clip or args.dry_run,
    )
    write_similarity_csv(extracted_overview, embeddings, anchor_sims, Path(f"castle_window_view_similarity_{args.window_id}.csv"))
    write_event_relevant_csv(
        extracted_overview,
        selected,
        reasons,
        anchor_sims,
        Path(f"castle_event_relevant_views_{args.window_id}.csv"),
    )

    event_frames = extract_event_frames(
        window,
        selected,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        frames_dir=args.frames_dir,
        logs_dir=args.logs_dir,
        event_offsets_sec=args.event_offsets_sec,
        timeout_sec=args.timeout_sec,
        tracked_paths=tracked_paths,
        max_bytes=max_bytes,
        dry_run=args.dry_run,
    )
    write_stream_frame_log(event_frames, Path(f"castle_event_relevant_extraction_log_{args.window_id}.csv"))
    make_event_contact_sheet(event_frames, Path(f"castle_event_relevant_contact_sheet_{args.window_id}.png"))
    local_storage = tracked_size(tracked_paths)
    write_report(
        window=window,
        active=active,
        overview=extracted_overview,
        selected=selected,
        reasons=reasons,
        clip_status=clip_status,
        event_frames=event_frames,
        local_storage_bytes=local_storage,
        output_path=Path("castle_event_relevant_view_selection_report.txt"),
    )

    print(f"Window: {window.window_id} {window.start_time}-{window.end_time}")
    print(f"Active streams: {len(active)}")
    print(f"Overview frames extracted: {sum(1 for item in extracted_overview if item.status == 'ok')}/{len(extracted_overview)}")
    print(f"CLIP status: {clip_status}")
    print(f"Selected event-relevant views: {', '.join(item.entity for item in selected) if selected else 'none'}")
    print(f"Event frames extracted: {sum(1 for item in event_frames if item.status == 'ok')}/{len(event_frames)}")
    print(f"Local tracked storage: {local_storage / (1024**2):.3f} MB")
    print("hf_hub_download for videos: NO")
    print(f"Wrote castle_window_overview_{args.window_id}.png")
    print(f"Wrote castle_window_view_similarity_{args.window_id}.csv")
    print(f"Wrote castle_event_relevant_views_{args.window_id}.csv")
    print(f"Wrote castle_event_relevant_contact_sheet_{args.window_id}.png")
    print("Wrote castle_event_relevant_view_selection_report.txt")


if __name__ == "__main__":
    main()
