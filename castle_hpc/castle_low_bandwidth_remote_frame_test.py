#!/usr/bin/env python3
"""
Low-bandwidth CASTLE remote-frame validation test.

This script validates one clock-time overlap window without downloading full
videos. It builds HuggingFace remote URLs with hf_hub_url and asks ffmpeg to
seek directly into the remote video URL.

Safety rules:
  - Never calls hf_hub_download for videos.
  - First run uses one overlap window, up to four streams, one frame per stream.
  - Stops if tracked local output/log storage exceeds the configured limit.
  - Stops on ffmpeg timeout or missing ffmpeg.
"""

from __future__ import annotations

import argparse
import csv
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
STATIC_PRIORITY = ["Kitchen", "Living1", "Living2", "Meeting", "Reading"]


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
class SelectedStream:
    window_id: str
    day: str
    period: str
    entity: str
    stream_type: str
    segment_id: str
    video_path: str
    video_start_time: str
    video_end_time: str
    window_start: str
    window_end: str
    target_clock_time: str
    offset_seconds: float
    remote_url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a micro remote-frame extraction test for CASTLE.")
    parser.add_argument("--interval-table", type=Path, default=Path("castle_main_video_interval_table.csv"))
    parser.add_argument("--selected-windows", type=Path, default=Path("castle_selected_multiview_windows.csv"))
    parser.add_argument("--selected-videos", type=Path, default=Path("castle_selected_validation_videos.csv"))
    parser.add_argument("--frames-dir", type=Path, default=Path("castle_low_bandwidth_remote_frames"))
    parser.add_argument("--logs-dir", type=Path, default=Path("castle_low_bandwidth_ffmpeg_logs"))
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--repo-type", default=REPO_TYPE)
    parser.add_argument("--max-local-mb", type=float, default=300.0)
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--target-offset-sec", type=float, default=60.0)
    parser.add_argument("--max-streams", type=int, default=4)
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
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        whole_seconds += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def safe_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def read_intervals(path: Path) -> dict[tuple[str, str, str], Interval]:
    intervals = {}
    with path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["time_parse_status"] != "ok":
                continue
            start = parse_clock_seconds(row["start_time"])
            end = parse_clock_seconds(row["end_time"])
            if start is None or end is None:
                continue
            if end < start:
                end += 24 * 3600
            interval = Interval(
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
            intervals[(interval.day, interval.entity, interval.video_path)] = interval
    return intervals


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def choose_micro_window(windows: list[dict[str, str]]) -> dict[str, str]:
    def rank(row: dict[str, str]) -> tuple[int, int, int, int, float, str]:
        period_rank = 0 if row["period"] == "morning" else 1
        start = parse_clock_seconds(row["window_start"]) or 0.0
        return (
            period_rank,
            -int(row["total_active_streams"]),
            -int(row["num_active_ego"]),
            -int(row["num_active_static"]),
            start,
            row["window_id"],
        )

    eligible = [
        row
        for row in windows
        if int(row["num_active_ego"]) >= 2
        and int(row["num_active_static"]) >= 1
        and safe_float(row["duration_minutes"]) >= 5
    ]
    if not eligible:
        raise RuntimeError("no eligible overlap windows found")
    return sorted(eligible, key=rank)[0]


def selected_video_paths_for_window(window: dict[str, str], selected_video_rows: list[dict[str, str]]) -> set[str]:
    window_id = window["window_id"]
    return {
        row["video_path"]
        for row in selected_video_rows
        if window_id in {part.strip() for part in row.get("used_by_windows", "").split(";") if part.strip()}
    }


def entity_order_from_window(window: dict[str, str], selected_paths: set[str], intervals: dict[tuple[str, str, str], Interval]) -> list[Interval]:
    active_ego = [item for item in window["active_ego_entities"].split(";") if item]
    active_static = [item for item in window["active_static_entities"].split(";") if item]
    window_start = parse_clock_seconds(window["window_start"]) or 0.0
    window_end = parse_clock_seconds(window["window_end"]) or 0.0

    candidates = [
        interval
        for interval in intervals.values()
        if interval.day == window["day"]
        and interval.entity in set(active_ego + active_static)
        and interval.start_seconds <= window_start
        and interval.end_seconds >= window_end
    ]

    selected_candidates = [interval for interval in candidates if interval.video_path in selected_paths]
    if selected_candidates:
        candidates = selected_candidates

    static_rank = {entity: index for index, entity in enumerate(STATIC_PRIORITY)}
    ego = sorted(
        [row for row in candidates if row.stream_type == "ego"],
        key=lambda row: (row.video_path not in selected_paths, row.entity),
    )
    static = sorted(
        [row for row in candidates if row.stream_type == "static"],
        key=lambda row: (static_rank.get(row.entity, len(STATIC_PRIORITY)), row.video_path not in selected_paths, row.entity),
    )

    chosen: list[Interval] = []
    if ego:
        chosen.append(ego[0])
    chosen.extend(static[:2])
    if len(chosen) < 4 and len(ego) > 1:
        chosen.append(ego[1])
    return chosen[:4]


def build_selection(
    window: dict[str, str],
    intervals: dict[tuple[str, str, str], Interval],
    selected_video_rows: list[dict[str, str]],
    *,
    repo_id: str,
    repo_type: str,
    target_offset_sec: float,
    max_streams: int,
) -> list[SelectedStream]:
    selected_paths = selected_video_paths_for_window(window, selected_video_rows)
    candidate_intervals = entity_order_from_window(window, selected_paths, intervals)[:max_streams]
    window_start_seconds = parse_clock_seconds(window["window_start"]) or 0.0
    target_clock_seconds = window_start_seconds + target_offset_sec
    target_clock_time = seconds_to_time(target_clock_seconds)
    selection = []
    for interval in candidate_intervals:
        offset = target_clock_seconds - interval.start_seconds
        if offset < 0 or offset > interval.duration_seconds:
            continue
        selection.append(
            SelectedStream(
                window_id=window["window_id"],
                day=interval.day,
                period=window["period"],
                entity=interval.entity,
                stream_type=interval.stream_type,
                segment_id=interval.segment_id,
                video_path=interval.video_path,
                video_start_time=interval.start_time,
                video_end_time=interval.end_time,
                window_start=window["window_start"],
                window_end=window["window_end"],
                target_clock_time=target_clock_time,
                offset_seconds=offset,
                remote_url=hf_hub_url(repo_id=repo_id, filename=interval.video_path, repo_type=repo_type),
            )
        )
    return selection


def write_selection(selection: list[SelectedStream], path: Path) -> None:
    fieldnames = [
        "window_id",
        "day",
        "period",
        "entity",
        "stream_type",
        "segment_id",
        "video_path",
        "video_start_time",
        "video_end_time",
        "window_start",
        "window_end",
        "target_clock_time",
        "offset_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in selection:
            writer.writerow(
                {
                    "window_id": item.window_id,
                    "day": item.day,
                    "period": item.period,
                    "entity": item.entity,
                    "stream_type": item.stream_type,
                    "segment_id": item.segment_id,
                    "video_path": item.video_path,
                    "video_start_time": item.video_start_time,
                    "video_end_time": item.video_end_time,
                    "window_start": item.window_start,
                    "window_end": item.window_end,
                    "target_clock_time": item.target_clock_time,
                    "offset_seconds": f"{item.offset_seconds:.3f}",
                }
            )


def write_url_table(selection: list[SelectedStream], path: Path) -> None:
    fieldnames = [
        "window_id",
        "day",
        "entity",
        "stream_type",
        "video_path",
        "target_clock_time",
        "offset_seconds",
        "remote_url",
        "uses_hf_hub_download",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in selection:
            writer.writerow(
                {
                    "window_id": item.window_id,
                    "day": item.day,
                    "entity": item.entity,
                    "stream_type": item.stream_type,
                    "video_path": item.video_path,
                    "target_clock_time": item.target_clock_time,
                    "offset_seconds": f"{item.offset_seconds:.3f}",
                    "remote_url": item.remote_url,
                    "uses_hf_hub_download": "no",
                }
            )


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def tracked_size(paths: list[Path]) -> int:
    return sum(directory_size(path) for path in paths)


def redact_command(command: list[str]) -> str:
    output = []
    skip_next = False
    for index, part in enumerate(command):
        if skip_next:
            output.append("Authorization: Bearer ***")
            skip_next = False
            continue
        output.append(part)
        if part == "-headers" and index + 1 < len(command):
            skip_next = True
    return " ".join(output)


def run_ffmpeg_extraction(
    selection: list[SelectedStream],
    *,
    frames_dir: Path,
    logs_dir: Path,
    max_local_bytes: int,
    timeout_sec: int,
) -> tuple[list[dict[str, Any]], str]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    tracked_dirs = [frames_dir, logs_dir]
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return [], "ffmpeg_missing"

    token = os.environ.get("HF_TOKEN", "")
    results = []
    for item in selection:
        before_size = tracked_size(tracked_dirs)
        if before_size > max_local_bytes:
            return results, "local_storage_limit_exceeded_before_extraction"

        output_path = frames_dir / f"{item.window_id}_{item.entity}_{item.target_clock_time.replace(':', '').replace('.', '')}.jpg"
        log_path = logs_dir / f"{item.window_id}_{item.entity}.stderr.txt"
        command = [
            ffmpeg,
            "-nostdin",
            "-y",
            "-ss",
            f"{item.offset_seconds:.3f}",
        ]
        if token:
            command.extend(["-headers", f"Authorization: Bearer {token}"])
        command.extend(
            [
                "-i",
                item.remote_url,
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(output_path),
            ]
        )

        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_sec, check=False)
            stderr = completed.stderr or ""
            stdout = completed.stdout or ""
            log_path.write_text(
                f"COMMAND: {redact_command(command)}\nRETURN_CODE: {completed.returncode}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}",
                encoding="utf-8",
            )
            status = "ok" if completed.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0 else "ffmpeg_failed"
        except subprocess.TimeoutExpired as exc:
            stderr = (exc.stderr or "").decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
            log_path.write_text(f"COMMAND: {redact_command(command)}\nTIMEOUT_AFTER_SEC: {timeout_sec}\n\nSTDERR:\n{stderr}", encoding="utf-8")
            status = "ffmpeg_timeout"

        after_size = tracked_size(tracked_dirs)
        results.append(
            {
                "window_id": item.window_id,
                "entity": item.entity,
                "stream_type": item.stream_type,
                "video_path": item.video_path,
                "target_clock_time": item.target_clock_time,
                "offset_seconds": f"{item.offset_seconds:.3f}",
                "output_frame": str(output_path) if output_path.exists() else "",
                "stderr_log": str(log_path),
                "status": status,
                "local_size_before_bytes": str(before_size),
                "local_size_after_bytes": str(after_size),
            }
        )
        if status != "ok":
            return results, status
        if after_size > max_local_bytes:
            return results, "local_storage_limit_exceeded_after_extraction"
    return results, ""


def write_extraction_log(results: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "window_id",
        "entity",
        "stream_type",
        "video_path",
        "target_clock_time",
        "offset_seconds",
        "output_frame",
        "stderr_log",
        "status",
        "local_size_before_bytes",
        "local_size_after_bytes",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def create_contact_sheet(results: list[dict[str, Any]], selection: list[SelectedStream], output_path: Path) -> bool:
    ok_results = [row for row in results if row["status"] == "ok" and row["output_frame"]]
    if not ok_results:
        return False
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False

    selection_by_entity = {item.entity: item for item in selection}
    thumb_w, thumb_h = 360, 220
    label_w = 520
    row_h = thumb_h + 50
    sheet = Image.new("RGB", (label_w + thumb_w, row_h * len(ok_results)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, result in enumerate(ok_results):
        y = row_index * row_h
        item = selection_by_entity[result["entity"]]
        label = (
            f"{item.day} {item.entity} ({item.stream_type})\n"
            f"{item.target_clock_time}, offset={item.offset_seconds:.1f}s\n"
            f"{item.video_path}"
        )
        draw.text((8, y + 10), label, fill="black")
        image = Image.open(result["output_frame"]).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        sheet.paste(image, (label_w + (thumb_w - image.width) // 2, y + (thumb_h - image.height) // 2))
    sheet.save(output_path)
    return True


def classify_failure(stopped_reason: str, results: list[dict[str, Any]]) -> str:
    if stopped_reason == "ffmpeg_missing":
        return "ffmpeg is not installed"
    if stopped_reason == "ffmpeg_timeout":
        return "ffmpeg timeout"
    if "local_storage_limit" in stopped_reason:
        return "too much local data written"
    if stopped_reason == "ffmpeg_failed" and results:
        log_path = Path(results[-1]["stderr_log"])
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        lowered = text.lower()
        if "401" in lowered or "403" in lowered or "authorization" in lowered:
            return "authentication"
        if "range" in lowered or "seek" in lowered or "partial" in lowered:
            return "HTTP seeking or remote seeking"
        if "moov atom" in lowered or "invalid data" in lowered:
            return "video format issue"
        return "ffmpeg failed; inspect stderr log"
    return "none"


def write_report(
    *,
    selection: list[SelectedStream],
    results: list[dict[str, Any]],
    stopped_reason: str,
    local_storage_bytes: int,
    contact_sheet_created: bool,
    output_path: Path,
) -> None:
    succeeded = sum(1 for row in results if row["status"] == "ok")
    failure_class = classify_failure(stopped_reason, results)
    lines = [
        "CASTLE low-bandwidth remote frame report",
        "",
        f"Selected streams: {len(selection)}",
        f"Successful frame extractions: {succeeded}",
        f"Local tracked storage used: {local_storage_bytes / (1024**2):.3f} MB",
        f"Stopped reason: {stopped_reason or 'none'}",
        f"Failure class: {failure_class}",
        f"Contact sheet created: {'yes' if contact_sheet_created else 'no'}",
        "",
        "1. Did remote frame extraction work without full video download?",
        (
            "   Yes; frames were extracted through remote URLs and no hf_hub_download video path was used."
            if succeeded == len(selection) and selection
            else "   No; the micro test did not complete all remote extractions."
        ),
        "",
        "2. How many streams succeeded?",
        f"   {succeeded}/{len(selection)}",
        "",
        "3. How much local storage was used?",
        f"   {local_storage_bytes / (1024**2):.3f} MB across frames/log outputs.",
        "",
        "4. Did the extracted frames appear to show the same event/window?",
        (
            "   Not visually evaluated because no complete contact sheet was produced."
            if not contact_sheet_created
            else "   Contact sheet produced; inspect it to confirm whether rows show the same event."
        ),
        "",
        "5. Should we scale to 3 windows and 12 streams?",
        (
            "   Not yet. Fix the remote extraction failure first."
            if succeeded < len(selection)
            else "   Yes, if the contact sheet confirms visual alignment and local storage remains low."
        ),
        "",
        "6. If failed, likely cause:",
        f"   {failure_class}",
        "",
        "Selected streams:",
    ]
    for item in selection:
        lines.append(
            f"   {item.window_id} {item.entity} {item.stream_type} "
            f"{item.target_clock_time} offset={item.offset_seconds:.3f}s {item.video_path}"
        )
    if results:
        lines.extend(["", "Extraction statuses:"])
        for row in results:
            lines.append(f"   {row['entity']}: {row['status']} log={row['stderr_log']}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    intervals = read_intervals(args.interval_table)
    windows = read_rows(args.selected_windows)
    selected_video_rows = read_rows(args.selected_videos) if args.selected_videos.exists() else []
    window = choose_micro_window(windows)
    selection = build_selection(
        window,
        intervals,
        selected_video_rows,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        target_offset_sec=args.target_offset_sec,
        max_streams=args.max_streams,
    )
    write_selection(selection, Path("castle_low_bandwidth_test_selection.csv"))
    write_url_table(selection, Path("castle_remote_url_test_table.csv"))

    max_local_bytes = int(args.max_local_mb * 1024 * 1024)
    results, stopped_reason = run_ffmpeg_extraction(
        selection,
        frames_dir=args.frames_dir,
        logs_dir=args.logs_dir,
        max_local_bytes=max_local_bytes,
        timeout_sec=args.timeout_sec,
    )
    write_extraction_log(results, Path("castle_low_bandwidth_extraction_log.csv"))

    contact_sheet = Path("castle_low_bandwidth_multiview_contact_sheet_micro.png")
    contact_sheet_created = create_contact_sheet(results, selection, contact_sheet)
    local_storage_bytes = tracked_size([args.frames_dir, args.logs_dir])
    write_report(
        selection=selection,
        results=results,
        stopped_reason=stopped_reason,
        local_storage_bytes=local_storage_bytes,
        contact_sheet_created=contact_sheet_created,
        output_path=Path("castle_low_bandwidth_remote_frame_report.txt"),
    )

    print(f"Selected micro-test window: {window['window_id']} {window['window_start']}-{window['window_end']}")
    print(f"Selected streams: {len(selection)}")
    print(f"Successful frame extractions: {sum(1 for row in results if row['status'] == 'ok')}/{len(selection)}")
    print(f"Tracked local storage: {local_storage_bytes / (1024**2):.3f} MB")
    print(f"Stopped reason: {stopped_reason or 'none'}")
    print("hf_hub_download for videos: NO")
    print("Wrote castle_low_bandwidth_test_selection.csv")
    print("Wrote castle_remote_url_test_table.csv")
    print("Wrote castle_low_bandwidth_extraction_log.csv")
    print("Wrote castle_low_bandwidth_remote_frame_report.txt")
    if contact_sheet_created:
        print("Wrote castle_low_bandwidth_multiview_contact_sheet_micro.png")


if __name__ == "__main__":
    main()
