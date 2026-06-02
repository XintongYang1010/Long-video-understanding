#!/usr/bin/env python3
"""
Targeted low-bandwidth contact-sheet extraction for CASTLE true A+B candidates.

This script uses remote ffmpeg frame extraction only. It does not use
hf_hub_download, does not download full videos, and does not run VLM/CLIP/LLM.
Candidate labels remain conservative and reviewable.
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
OUT_DIR = Path("candidate_true_AB_contact_sheets")
CASES_CSV = Path("candidate_true_AB_cases.csv")
SUMMARY_MD = Path("true_AB_search_summary.md")
INTERVAL_TABLE = Path("castle_main_video_interval_table.csv")
WINDOW_TABLE = Path("castle_selected_multiview_windows.csv")

EXISTING_OFFSETS = [30.0, 150.0, 270.0]
NEW_OVERVIEW_OFFSETS = [150.0]
FOCUSED_OFFSETS = [30.0, 150.0, 270.0]


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
    parser = argparse.ArgumentParser(description="Extract CASTLE true A+B candidate contact sheets.")
    parser.add_argument("--max-local-mb", type=float, default=500.0)
    parser.add_argument("--timeout-sec", type=int, default=45)
    parser.add_argument("--max-consecutive-failures", type=int, default=5)
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
    rows = read_csv(INTERVAL_TABLE)
    intervals = []
    for row in rows:
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


def candidate_configs() -> list[dict[str, str]]:
    return [
        {
            "candidate_id": "C001",
            "window_id": "DAY1_100500000",
            "kind": "existing",
            "proposed_querying_user": "Werner",
            "proposed_question_natural_language": "Can we tell both that someone is presenting and what the group/screen context is?",
            "proposed_answer": "Known presentation/meeting window; final answer requires human review for A+B complementarity framing.",
            "source_A": "Werner",
            "source_B": "Tien",
            "static_source_if_any": "Meeting",
            "pattern": "speaker/gesture vs screen/content split",
            "notes": "Existing Q1 shows external access helps; likely not true A+B because Tien or Meeting may be sufficient alone.",
        },
        {
            "candidate_id": "C002",
            "window_id": "DAY3_174500000",
            "kind": "existing",
            "proposed_querying_user": "Cathal",
            "proposed_question_natural_language": "Can we connect tabletop activity with kitchen activity in the same mixed window?",
            "proposed_answer": "Known mixed tabletop/kitchen window; final answer requires human review.",
            "source_A": "Cathal",
            "source_B": "Florian",
            "static_source_if_any": "Kitchen",
            "pattern": "ego local activity vs kitchen/global context split",
            "notes": "Existing Q5/Q6 show query dependence; search for whether a combined tabletop+kitchen question needs both.",
        },
        {
            "candidate_id": "C003",
            "window_id": "DAY4_120500000",
            "kind": "existing",
            "proposed_querying_user": "Allie",
            "proposed_question_natural_language": "Can we identify dining activity and broader room/table context?",
            "proposed_answer": "Known dining window; likely redundancy/control rather than true A+B.",
            "source_A": "Allie",
            "source_B": "Bao",
            "static_source_if_any": "Living1",
            "pattern": "relevant-but-redundant dining views",
            "notes": "Existing Q3/Q4 mostly control for self-sufficient/redundant or calibration failure cases.",
        },
        {
            "candidate_id": "C004",
            "window_id": "DAY1_102000000",
            "kind": "new",
            "proposed_querying_user": "Werner",
            "proposed_question_natural_language": "What is the presenter referring to, and who is addressing the seated group?",
            "proposed_answer": "TBD after human review.",
            "source_A": "Werner",
            "source_B": "Tien",
            "static_source_if_any": "Meeting",
            "pattern": "speaker/gesture vs screen/content split",
            "notes": "Later presentation-like window; inspect whether one view sees speaker/gesture and another sees screen/content.",
        },
        {
            "candidate_id": "C005",
            "window_id": "DAY2_141000000",
            "kind": "new",
            "proposed_querying_user": "Allie",
            "proposed_question_natural_language": "What object/action is happening locally, and where is it located in the room?",
            "proposed_answer": "TBD after human review.",
            "source_A": "Allie",
            "source_B": "Bjorn",
            "static_source_if_any": "Living1",
            "pattern": "ego local action vs static global layout split",
            "notes": "DAY2 afternoon high-overlap candidate for object/action and layout complementarity.",
        },
        {
            "candidate_id": "C006",
            "window_id": "DAY2_183500000",
            "kind": "new",
            "proposed_querying_user": "Cathal",
            "proposed_question_natural_language": "Who is involved in the activity and where does it happen in the room?",
            "proposed_answer": "TBD after human review.",
            "source_A": "Cathal",
            "source_B": "Onanong",
            "static_source_if_any": "Meeting",
            "pattern": "person/object identity vs room/location split",
            "notes": "DAY2 evening high-overlap candidate for identity/location complementarity.",
        },
        {
            "candidate_id": "C007",
            "window_id": "DAY3_121500000",
            "kind": "new",
            "proposed_querying_user": "Allie",
            "proposed_question_natural_language": "Who manipulates the object on or near the table, and where does it end up?",
            "proposed_answer": "TBD after human review.",
            "source_A": "Allie",
            "source_B": "Cathal",
            "static_source_if_any": "Kitchen",
            "pattern": "object pickup vs object placement/final state split",
            "notes": "DAY3 midday table/object candidate.",
        },
        {
            "candidate_id": "C008",
            "window_id": "DAY4_100500000",
            "kind": "new",
            "proposed_querying_user": "Allie",
            "proposed_question_natural_language": "What local action is happening, and how does it relate to the room layout?",
            "proposed_answer": "TBD after human review.",
            "source_A": "Allie",
            "source_B": "Bao",
            "static_source_if_any": "Living1",
            "pattern": "ego local action vs static global layout split",
            "notes": "DAY4 morning candidate for local/global split.",
        },
        {
            "candidate_id": "C009",
            "window_id": "DAY4_182500000",
            "kind": "new",
            "proposed_querying_user": "Luca",
            "proposed_question_natural_language": "Can we combine a close ego view with static/global context to answer an evening activity question?",
            "proposed_answer": "TBD after human review.",
            "source_A": "Luca",
            "source_B": "Werner",
            "static_source_if_any": "Meeting",
            "pattern": "evening multi-view local/global candidate",
            "notes": "DAY4 evening high-overlap candidate; inspect for off-event/invalid controls and possible complementarity.",
        },
    ]


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


def selected_intervals(active: list[Interval], entities: list[str]) -> list[Interval]:
    wanted = set(entities)
    return [item for item in active if item.entity in wanted]


def make_case_row(config: dict[str, str], window: dict[str, str], overview_path: Path, focused_path: Path, status: str) -> dict[str, str]:
    kind = config["kind"]
    if kind == "existing":
        if config["candidate_id"] == "C001":
            case_type = "self_insufficient_other_sufficient"
            a_req = "no"
            a_suff = b_suff = static_suff = "yes"
            ab_suff = "yes"
            main = "yes"
            review = "yes"
            why_a = "A likely sees presentation evidence directly; verify if A alone is enough."
            why_b = "B likely sees presentation evidence directly; verify if B alone is enough."
            why_combo = "Combination may not be required; use sheet to check whether a stricter speaker/screen split exists over time."
        elif config["candidate_id"] == "C002":
            case_type = "query_dependent"
            a_req = "no"
            a_suff = b_suff = static_suff = "yes"
            ab_suff = "yes"
            main = "yes"
            review = "yes"
            why_a = "A may answer tabletop questions but not kitchen questions."
            why_b = "B may answer kitchen questions but not tabletop questions."
            why_combo = "Combination may help for a cross-sub-event question, but this must be verified manually."
        else:
            case_type = "redundant_control"
            a_req = "no"
            a_suff = b_suff = "yes"
            static_suff = "unclear"
            ab_suff = "yes"
            main = "no"
            review = "yes"
            why_a = "A likely already shows dining evidence."
            why_b = "B likely already shows dining evidence."
            why_combo = "Likely redundant control, not true complementarity."
    else:
        case_type = "weak_or_unclear"
        a_req = "unclear"
        a_suff = b_suff = static_suff = "unclear"
        ab_suff = "unclear"
        main = "unclear"
        review = "yes"
        why_a = "Unknown until human inspection of contact sheet."
        why_b = "Unknown until human inspection of contact sheet."
        why_combo = "Candidate was selected for possible complementary evidence pattern; manual review required."

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
        "A_only_sufficient": a_suff,
        "B_only_sufficient": b_suff,
        "static_only_sufficient": static_suff,
        "A_plus_B_sufficient": ab_suff,
        "A_plus_B_required": a_req,
        "why_A_alone_insufficient": why_a,
        "why_B_alone_insufficient": why_b,
        "why_combination_is_needed": why_combo,
        "case_type": case_type,
        "should_use_as_main_evidence": main,
        "needs_human_review": review,
        "notes": f"{config['pattern']}. {config['notes']} overview_sheet={overview_path}; focused_sheet={focused_path}; extraction_status={status}",
    }


def write_summary(case_rows: list[dict[str, str]], log_rows: list[dict[str, str]], sheets_generated: list[Path]) -> None:
    visually_reviewable = [r for r in case_rows if "extraction_status=ok" in r["notes"]]
    possible_true = [
        r
        for r in visually_reviewable
        if r["A_plus_B_required"] == "yes" or (r["case_type"] == "weak_or_unclear" and r["needs_human_review"] == "yes")
    ]
    strongest = [r for r in case_rows if r["candidate_id"] in {"C001", "C002", "C004"}]
    weak = [r for r in case_rows if r["case_type"] in {"weak_or_unclear", "redundant_control"}]
    ok_frames = sum(1 for r in log_rows if r["status"].startswith("ok"))
    total_frames = len(log_rows)
    lines = [
        "# True A+B Candidate Search Summary",
        "",
        "This is a low-bandwidth visual evidence generation step. It does not run VLM/LLM inference, does not run official CASTLE QA, and does not download full videos.",
        "",
        f"- number of candidate windows selected: {len(case_rows)}",
        f"- number of windows with generated visual evidence: {len(visually_reviewable)}",
        f"- number of contact sheets generated: {len(sheets_generated)}",
        f"- frame extraction success: {ok_frames}/{total_frames}",
        f"- number of candidate cases listed: {len(case_rows)}",
        f"- number of possible true_complementary_A_plus_B cases: {len(possible_true)} reviewable candidates, 0 locked true cases",
        "",
        "## Strongest 3 candidate cases",
    ]
    for row in strongest[:3]:
        lines.append(f"- {row['candidate_id']} {row['window_id']}: {row['proposed_question_natural_language']} ({row['case_type']})")
    lines.extend(["", "## Weak/unclear cases"])
    for row in weak:
        lines.append(f"- {row['candidate_id']} {row['window_id']}: {row['case_type']}; needs_human_review={row['needs_human_review']}")
    lines.extend(
        [
            "",
            "## Recommendation",
            "Review generated contact sheets manually before making any true A+B claims. If extraction failed or no sheets were generated, rerun this script on the NYU/conda ffmpeg environment where remote seeking previously worked.",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg and not args.dry_run:
        raise RuntimeError("ffmpeg not found")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    intervals = load_intervals()
    windows = load_windows()
    max_bytes = int(args.max_local_mb * 1024 * 1024)
    log_rows: list[dict[str, str]] = []
    case_rows: list[dict[str, str]] = []
    sheets_generated: list[Path] = []
    consecutive_failures = 0

    configs = candidate_configs()
    stop_remaining = False
    for config in configs:
        window = windows.get(config["window_id"])
        if not window:
            continue
        if stop_remaining:
            overview_path = OUT_DIR / f"overview_multi_{config['candidate_id']}_{config['window_id']}.png"
            focused_path = OUT_DIR / f"focused_AB_{config['candidate_id']}_{config['window_id']}.png"
            case_rows.append(make_case_row(config, window, overview_path, focused_path, "not_attempted_after_remote_ffmpeg_failure"))
            continue
        start = parse_seconds(window["window_start"])
        overview_offsets = EXISTING_OFFSETS if config["kind"] == "existing" else NEW_OVERVIEW_OFFSETS
        overview_results: list[Frame] = []
        focused_results: list[Frame] = []
        status = "ok"

        for offset in overview_offsets:
            active = active_intervals(intervals, window, start + offset)
            for interval in active:
                result = extract_one(interval, config["candidate_id"], config["window_id"], start + offset, ffmpeg or "ffmpeg", args.timeout_sec, max_bytes, args.dry_run)
                overview_results.append(result)
                log_rows.append(result.__dict__ | {"frame_path": str(result.frame_path), "log_path": str(result.log_path)})
                if result.status.startswith("ok") or result.status == "dry_run":
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    status = result.status
                if consecutive_failures >= args.max_consecutive_failures:
                    status = "stopped_after_consecutive_failures"
                    break
            if status == "stopped_after_consecutive_failures":
                break
        # Focused sheets reuse already extracted frames when same entity/time exists.
        if status != "stopped_after_consecutive_failures":
            for offset in FOCUSED_OFFSETS:
                active = active_intervals(intervals, window, start + offset)
                targets = [config["source_A"], config["source_B"], config["static_source_if_any"]]
                for interval in selected_intervals(active, targets):
                    result = extract_one(interval, config["candidate_id"], config["window_id"], start + offset, ffmpeg or "ffmpeg", args.timeout_sec, max_bytes, args.dry_run)
                    focused_results.append(result)
                    log_rows.append(result.__dict__ | {"frame_path": str(result.frame_path), "log_path": str(result.log_path)})
                    if result.status.startswith("ok") or result.status == "dry_run":
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        status = result.status
                    if consecutive_failures >= args.max_consecutive_failures:
                        status = "stopped_after_consecutive_failures"
                        break
                if status == "stopped_after_consecutive_failures":
                    break

        overview_path = OUT_DIR / f"overview_multi_{config['candidate_id']}_{config['window_id']}.png"
        focused_path = OUT_DIR / f"focused_AB_{config['candidate_id']}_{config['window_id']}.png"
        if make_sheet_grid(overview_results, overview_path, f"{config['candidate_id']} {config['window_id']} overview"):
            sheets_generated.append(overview_path)
        if make_sheet_grid(focused_results, focused_path, f"{config['candidate_id']} {config['window_id']} focused A/B"):
            sheets_generated.append(focused_path)
        case_rows.append(make_case_row(config, window, overview_path, focused_path, status))

        if status == "stopped_after_consecutive_failures":
            # Stop remote extraction to avoid burning time if network is failing.
            # Remaining cases are still recorded as failed/not attempted.
            stop_remaining = True

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
            OUT_DIR / "extraction_log.csv",
            log_rows,
            ["candidate_id", "window_id", "day", "entity", "stream_type", "video_path", "target_time", "offset_seconds", "frame_path", "log_path", "status"],
        )
    write_summary(case_rows, log_rows, sheets_generated)

    print(f"Wrote {CASES_CSV}")
    print(f"Wrote {SUMMARY_MD}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Windows inspected: {len(case_rows)}")
    print(f"Contact sheets generated: {len(sheets_generated)}")
    print(f"Frame extraction ok: {sum(1 for r in log_rows if r['status'].startswith('ok'))}/{len(log_rows)}")
    print(f"Local storage MB: {tracked_size(OUT_DIR) / (1024**2):.3f}")


if __name__ == "__main__":
    main()
