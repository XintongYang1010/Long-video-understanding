#!/usr/bin/env python3
"""
Round-2 targeted low-bandwidth contact-sheet extraction for CASTLE true A+B
complementarity candidates.

This script:
- reads candidate windows from round2_selected_windows.txt
- uses remote ffmpeg frame extraction only
- does not use hf_hub_download
- does not run VLM/LLM/CLIP
- keeps labels conservative and human-reviewable
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_url
from PIL import Image, ImageDraw


REPO_ID = "CASTLE-Dataset/CASTLE2024"
REPO_TYPE = "dataset"
OUT_DIR = Path("candidate_true_AB_contact_sheets_round2")
CASES_CSV = Path("candidate_true_AB_cases_round2.csv")
SUMMARY_MD = Path("true_AB_search_summary_round2.md")
INTERVAL_TABLE = Path("castle_main_video_interval_table.csv")
WINDOW_TABLE = Path("castle_selected_multiview_windows.csv")
ROUND2_WINDOWS = Path("round2_selected_windows.txt")

OVERVIEW_OFFSETS = [30.0, 150.0, 270.0]
FOCUSED_OFFSETS = [30.0, 150.0, 270.0]
EXCLUDED_ROUND1_WINDOWS = {
    "DAY1_100500000",
    "DAY3_174500000",
    "DAY4_120500000",
    "DAY1_102000000",
    "DAY2_141000000",
    "DAY2_183500000",
    "DAY3_121500000",
    "DAY4_100500000",
    "DAY4_182500000",
}

PATTERNS = [
    "speaker/gesture vs screen/content",
    "object pickup vs object placement/final state",
    "ego local action vs static global layout",
    "person/object identity vs room/location",
]


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


@dataclass
class Frame:
    candidate_id: str
    window_id: str
    day: str
    entity: str
    stream_type: str
    video_path: str
    target_time: str
    offset_seconds: float
    frame_path: Path
    log_path: Path
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round-2 CASTLE true A+B candidate contact-sheet extraction.")
    parser.add_argument("--max-local-mb", type=float, default=500.0)
    parser.add_argument("--timeout-sec", type=int, default=45)
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_seconds(value: str) -> float:
    hh, mm, ss = value.split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def seconds_to_time(seconds: float) -> str:
    seconds %= 24 * 3600
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = seconds - hh * 3600 - mm * 60
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"


def safe_time(value: str) -> str:
    return value.replace(":", "").replace(".", "")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tracked_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def load_intervals() -> list[Interval]:
    intervals = []
    for row in read_csv(INTERVAL_TABLE):
        if row.get("time_parse_status") != "ok":
            continue
        intervals.append(
            Interval(
                day=row["day"],
                entity=row["entity"],
                stream_type=row["stream_type"],
                segment_id=row["segment_id"],
                video_path=row["video_path"],
                start_time=row["start_time"],
                end_time=row["end_time"],
                start_seconds=parse_seconds(row["start_time"]),
                end_seconds=parse_seconds(row["end_time"]),
            )
        )
    return intervals


def load_windows() -> dict[str, dict[str, str]]:
    return {row["window_id"]: row for row in read_csv(WINDOW_TABLE)}


def load_round2_window_ids() -> list[str]:
    if not ROUND2_WINDOWS.exists():
        raise FileNotFoundError(
            f"{ROUND2_WINDOWS} not found. Create it with one window_id per line, excluding C001-C009 windows."
        )
    ids: list[str] = []
    for line in ROUND2_WINDOWS.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        ids.append(value.split()[0])
    return ids


def choose_sources(window: dict[str, str], index: int) -> tuple[str, str, str, str]:
    egos = [x for x in window["active_ego_entities"].split(";") if x]
    statics = [x for x in window["active_static_entities"].split(";") if x]
    static_priority = ["Meeting", "Living1", "Living2", "Kitchen", "Reading"]
    static = next((s for s in static_priority if s in statics), statics[0] if statics else "")
    if len(egos) >= 2:
        source_a = egos[index % len(egos)]
        source_b = egos[(index + max(1, len(egos) // 2)) % len(egos)]
    elif egos:
        source_a = egos[0]
        source_b = static
    else:
        source_a = static
        source_b = ""
    return source_a, source_b, static, PATTERNS[index % len(PATTERNS)]


def make_candidate_config(window_id: str, window: dict[str, str], index: int) -> dict[str, str]:
    source_a, source_b, static, pattern = choose_sources(window, index)
    return {
        "candidate_id": f"R2_{index + 1:03d}",
        "window_id": window_id,
        "proposed_querying_user": source_a,
        "proposed_question_natural_language": question_for_pattern(pattern),
        "proposed_answer": "TBD after human review.",
        "source_A": source_a,
        "source_B": source_b,
        "static_source_if_any": static,
        "pattern": pattern,
        "notes": (
            "Round-2 targeted candidate from user-provided window list. Labels are intentionally conservative; "
            "manual inspection is required before any true A+B claim."
        ),
    }


def question_for_pattern(pattern: str) -> str:
    if pattern == "speaker/gesture vs screen/content":
        return "What is the person referring to, and how is it connected to the group or screen context?"
    if pattern == "object pickup vs object placement/final state":
        return "Who handled the object, and where did the object end up?"
    if pattern == "ego local action vs static global layout":
        return "What local action happened, and where in the room did it happen?"
    return "Which person or object is involved, and where is it located in the room?"


def active_intervals(intervals: list[Interval], window: dict[str, str], target_seconds: float) -> list[Interval]:
    entities = set(window["active_ego_entities"].split(";") + window["active_static_entities"].split(";"))
    output = [
        item
        for item in intervals
        if item.day == window["day"]
        and item.entity in entities
        and item.start_seconds <= target_seconds <= item.end_seconds
    ]
    return sorted(output, key=lambda item: (0 if item.stream_type == "ego" else 1, item.entity))


def selected_intervals(active: list[Interval], entities: list[str]) -> list[Interval]:
    wanted = {entity for entity in entities if entity}
    return [item for item in active if item.entity in wanted]


def extract_one(
    interval: Interval,
    candidate_id: str,
    window_id: str,
    target_seconds: float,
    ffmpeg: str,
    timeout_sec: int,
    max_bytes: int,
    dry_run: bool,
) -> Frame:
    target_time = seconds_to_time(target_seconds)
    frame_path = OUT_DIR / "frames" / window_id / f"{interval.entity}_{safe_time(target_time)}.jpg"
    log_path = OUT_DIR / "logs" / window_id / f"{interval.entity}_{safe_time(target_time)}.stderr.txt"
    offset = target_seconds - interval.start_seconds

    if frame_path.exists() and frame_path.stat().st_size > 0:
        return Frame(candidate_id, window_id, interval.day, interval.entity, interval.stream_type, interval.video_path, target_time, offset, frame_path, log_path, "ok_reused")
    if dry_run:
        return Frame(candidate_id, window_id, interval.day, interval.entity, interval.stream_type, interval.video_path, target_time, offset, frame_path, log_path, "dry_run")
    if tracked_size(OUT_DIR) > max_bytes:
        return Frame(candidate_id, window_id, interval.day, interval.entity, interval.stream_type, interval.video_path, target_time, offset, frame_path, log_path, "skip_storage_cap")

    frame_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    url = hf_hub_url(repo_id=REPO_ID, filename=interval.video_path, repo_type=REPO_TYPE)
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
        "-ss",
        f"{offset:.3f}",
        "-i",
        url,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(frame_path),
    ]
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        command[1:1] = ["-headers", f"Authorization: Bearer {token}"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_sec)
        status = "ok" if result.returncode == 0 and frame_path.exists() and frame_path.stat().st_size > 0 else f"ffmpeg_failed_{result.returncode}"
        log_path.write_text(
            "COMMAND: " + " ".join(command[: command.index("-i")] + ["-i", url, "..."]) + "\n"
            + f"RETURN_CODE: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n",
            encoding="utf-8",
        )
    except subprocess.TimeoutExpired as exc:
        status = "ffmpeg_timeout"
        log_path.write_text(f"TIMEOUT after {timeout_sec}s\n{exc}\n", encoding="utf-8")
    return Frame(candidate_id, window_id, interval.day, interval.entity, interval.stream_type, interval.video_path, target_time, offset, frame_path, log_path, status)


def make_sheet_grid(results: list[Frame], output_path: Path, title: str) -> bool:
    ok = [item for item in results if item.status.startswith("ok") and item.frame_path.exists()]
    if not ok:
        return False
    entities = sorted({item.entity for item in ok}, key=lambda e: next(i for i, item in enumerate(ok) if item.entity == e))
    times = sorted({item.target_time for item in ok})
    by_key = {(item.entity, item.target_time): item for item in ok}
    thumb_w, thumb_h = 260, 150
    label_w = 190
    header_h = 70
    row_h = 205
    col_w = 310
    width = label_w + len(times) * col_w
    height = header_h + len(entities) * row_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill="black")
    for col, time in enumerate(times):
        draw.text((label_w + col * col_w + 8, 40), time, fill="black")
    for row, entity in enumerate(entities):
        y = header_h + row * row_h
        stream_type = next((item.stream_type for item in ok if item.entity == entity), "")
        draw.rectangle((0, y, width - 1, y + row_h - 1), outline="gray")
        draw.text((8, y + 12), f"{entity}\n{stream_type}", fill="black")
        for col, time in enumerate(times):
            x = label_w + col * col_w
            draw.rectangle((x, y, x + col_w - 1, y + row_h - 1), outline="lightgray")
            item = by_key.get((entity, time))
            if not item:
                continue
            img = Image.open(item.frame_path).convert("RGB")
            img.thumbnail((thumb_w, thumb_h))
            sheet.paste(img, (x + (col_w - img.width) // 2, y + 8))
            draw.text((x + 8, y + thumb_h + 18), item.video_path, fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return True


def make_case_row(config: dict[str, str], window: dict[str, str], overview_path: Path, focused_path: Path, status: str) -> dict[str, str]:
    return {
        "candidate_id": config["candidate_id"],
        "window_id": config["window_id"],
        "day": window.get("day", ""),
        "clock_time_range": f"{window.get('window_start', '')}-{window.get('window_end', '')}",
        "proposed_querying_user": config["proposed_querying_user"],
        "proposed_question_natural_language": config["proposed_question_natural_language"],
        "proposed_answer": config["proposed_answer"],
        "source_A": config["source_A"],
        "source_A_observation": "Needs human review from focused/contact sheet.",
        "source_B": config["source_B"],
        "source_B_observation": "Needs human review from focused/contact sheet.",
        "static_source_if_any": config["static_source_if_any"],
        "static_observation": "Needs human review from focused/contact sheet.",
        "A_only_sufficient": "unclear",
        "B_only_sufficient": "unclear",
        "static_only_sufficient": "unclear",
        "A_plus_B_sufficient": "unclear",
        "A_plus_B_required": "unclear",
        "why_A_alone_insufficient": "Unknown until human inspection. Do not mark A insufficient automatically.",
        "why_B_alone_insufficient": "Unknown until human inspection. Do not mark B insufficient automatically.",
        "why_combination_is_needed": "Candidate selected for possible A+B complementarity; manual inspection required.",
        "case_type": "weak_or_unclear",
        "should_use_as_main_evidence": "unclear",
        "needs_human_review": "yes",
        "notes": f"{config['pattern']}. {config['notes']} overview_sheet={overview_path}; focused_sheet={focused_path}; extraction_status={status}",
    }


def write_summary(case_rows: list[dict[str, str]], log_rows: list[dict[str, str]], sheets_generated: list[Path]) -> None:
    ok_frames = sum(1 for row in log_rows if row["status"].startswith("ok"))
    total_frames = len(log_rows)
    visually_reviewable = [row for row in case_rows if "extraction_status=ok" in row["notes"]]
    lines = [
        "# True A+B Candidate Search Summary Round 2",
        "",
        "This is a low-bandwidth visual evidence generation step. It does not run VLM/LLM/CLIP, does not run official CASTLE QA, and does not download full videos.",
        "",
        f"- number of candidate windows selected: {len(case_rows)}",
        f"- number of windows with generated visual evidence: {len(visually_reviewable)}",
        f"- number of contact sheets generated: {len(sheets_generated)}",
        f"- frame extraction success: {ok_frames}/{total_frames}",
        f"- number of candidate cases listed: {len(case_rows)}",
        "- number of possible true_complementary_A_plus_B cases: all generated cases remain needs_human_review; 0 locked true cases",
        "",
        "## Strongest 3 candidate cases",
    ]
    for row in case_rows[:3]:
        lines.append(f"- {row['candidate_id']} {row['window_id']}: {row['proposed_question_natural_language']} ({row['case_type']})")
    lines.extend(["", "## Weak/unclear cases"])
    for row in case_rows:
        if row["case_type"] == "weak_or_unclear":
            lines.append(f"- {row['candidate_id']} {row['window_id']}: needs_human_review=yes")
    lines.extend(
        [
            "",
            "## Recommendation",
            "Manually review the generated round-2 contact sheets. Only label true_complementary_A_plus_B if A-only and B-only are both no/unclear and A+B is yes. If no such cases appear, continue targeted search rather than building an automatic selector.",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg and not args.dry_run:
        raise RuntimeError("ffmpeg not found")
    window_ids = load_round2_window_ids()
    windows = load_windows()
    intervals = load_intervals()
    max_bytes = int(args.max_local_mb * 1024 * 1024)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    configs: list[dict[str, str]] = []
    case_rows: list[dict[str, str]] = []
    log_rows: list[dict[str, str]] = []
    sheets_generated: list[Path] = []

    for window_id in window_ids:
        if window_id in EXCLUDED_ROUND1_WINDOWS:
            raise ValueError(f"{window_id} is a C001-C009/round-1 window and must not be reused in round 2.")
        if window_id not in windows:
            raise KeyError(f"{window_id} not found in {WINDOW_TABLE}")
        configs.append(make_candidate_config(window_id, windows[window_id], len(configs)))

    for config in configs:
        window = windows[config["window_id"]]
        start = parse_seconds(window["window_start"])
        overview_results: list[Frame] = []
        focused_results: list[Frame] = []
        status = "ok"

        for offset in OVERVIEW_OFFSETS:
            active = active_intervals(intervals, window, start + offset)
            for interval in active:
                result = extract_one(interval, config["candidate_id"], config["window_id"], start + offset, ffmpeg or "ffmpeg", args.timeout_sec, max_bytes, args.dry_run)
                overview_results.append(result)
                log_rows.append(result.__dict__ | {"frame_path": str(result.frame_path), "log_path": str(result.log_path)})
                if not (result.status.startswith("ok") or result.status == "dry_run"):
                    status = result.status

        for offset in FOCUSED_OFFSETS:
            active = active_intervals(intervals, window, start + offset)
            targets = [config["source_A"], config["source_B"], config["static_source_if_any"]]
            for interval in selected_intervals(active, targets):
                result = extract_one(interval, config["candidate_id"], config["window_id"], start + offset, ffmpeg or "ffmpeg", args.timeout_sec, max_bytes, args.dry_run)
                focused_results.append(result)
                log_rows.append(result.__dict__ | {"frame_path": str(result.frame_path), "log_path": str(result.log_path)})
                if not (result.status.startswith("ok") or result.status == "dry_run"):
                    status = result.status

        overview_path = OUT_DIR / f"overview_round2_{config['candidate_id']}_{config['window_id']}.png"
        focused_path = OUT_DIR / f"focused_AB_round2_{config['candidate_id']}_{config['window_id']}.png"
        if make_sheet_grid(overview_results, overview_path, f"{config['candidate_id']} {config['window_id']} overview round2"):
            sheets_generated.append(overview_path)
        if make_sheet_grid(focused_results, focused_path, f"{config['candidate_id']} {config['window_id']} focused A/B round2"):
            sheets_generated.append(focused_path)
        case_rows.append(make_case_row(config, window, overview_path, focused_path, status))

    write_csv(
        CASES_CSV,
        case_rows,
        [
            "candidate_id",
            "window_id",
            "day",
            "clock_time_range",
            "proposed_querying_user",
            "proposed_question_natural_language",
            "proposed_answer",
            "source_A",
            "source_A_observation",
            "source_B",
            "source_B_observation",
            "static_source_if_any",
            "static_observation",
            "A_only_sufficient",
            "B_only_sufficient",
            "static_only_sufficient",
            "A_plus_B_sufficient",
            "A_plus_B_required",
            "why_A_alone_insufficient",
            "why_B_alone_insufficient",
            "why_combination_is_needed",
            "case_type",
            "should_use_as_main_evidence",
            "needs_human_review",
            "notes",
        ],
    )
    if log_rows:
        write_csv(
            OUT_DIR / "extraction_log_round2.csv",
            log_rows,
            ["candidate_id", "window_id", "day", "entity", "stream_type", "video_path", "target_time", "offset_seconds", "frame_path", "log_path", "status"],
        )
    write_summary(case_rows, log_rows, sheets_generated)

    print(f"Wrote {CASES_CSV}")
    print(f"Wrote {SUMMARY_MD}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Candidate windows: {len(case_rows)}")
    print(f"Contact sheets generated: {len(sheets_generated)}")
    print(f"Frame extraction ok: {sum(1 for r in log_rows if r['status'].startswith('ok'))}/{len(log_rows)}")
    print(f"Local storage MB: {tracked_size(OUT_DIR) / (1024**2):.3f}")


if __name__ == "__main__":
    main()
