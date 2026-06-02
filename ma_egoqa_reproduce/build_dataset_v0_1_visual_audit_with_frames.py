#!/usr/bin/env python3
"""
Download only needed EgoLife clips and extract targeted frames for Dataset V0.1
visual audit.

No model inference is performed. This script only uses the existing visual audit
frame plan, maps each raw_key to a Hugging Face file path, downloads those exact
mp4 files when needed, and extracts one still frame per planned target time.
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import re
import shutil
import subprocess
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
AUDIT_DIR = ROOT / "outputs" / "historical_v2_fullscreen" / "dataset_v0_1" / "visual_audit_v1"
FRAME_PLAN_CSV = AUDIT_DIR / "visual_frame_extraction_plan.csv"
CASE_SUBSET_CSV = AUDIT_DIR / "visual_audit_case_subset.csv"

HF_REPO_ID = "lmms-lab/EgoLife"
HF_REPO_TYPE = "dataset"

PARTICIPANT_FOLDERS = {
    "Jake": "A1_JAKE",
    "Alice": "A2_ALICE",
    "Tasha": "A3_TASHA",
    "Lucia": "A4_LUCIA",
    "Katrina": "A5_KATRINA",
    "Shure": "A6_SHURE",
}

KNOWN_FFMPEG_CANDIDATES = [
    Path("/scratch/xy3257/castle_hpc/envs/castle/bin/ffmpeg"),
    Path("/scratch/xy3257/castle_poc/cenv/bin/ffmpeg"),
]

VIDEO_SOURCE_FIELDS = [
    "visual_case_id",
    "dataset_case_id",
    "evidence_scope",
    "source_agent",
    "participant_folder",
    "day",
    "raw_key",
    "hf_repo_id",
    "hf_path",
    "can_resolve_video",
    "reason",
]

DOWNLOAD_MANIFEST_FIELDS = [
    "hf_path",
    "local_video_path",
    "file_size_mb",
    "download_status",
    "error",
]

EXTRACTION_MANIFEST_FIELDS = [
    "visual_case_id",
    "dataset_case_id",
    "evidence_scope",
    "source_agent",
    "hf_path",
    "local_video_path",
    "target_frame_time_absolute",
    "target_frame_time_relative",
    "output_frame_path",
    "extraction_status",
    "error",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean_csv(row.get(field, "")) for field in fieldnames})


def clean_csv(value: Any) -> Any:
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


def hms_to_seconds(value: str) -> float:
    parts = str(value or "").split(":")
    if len(parts) != 3:
        return 0.0
    return float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2])


def seconds_to_hms(value: float) -> str:
    value = max(0.0, float(value))
    hours = int(value // 3600)
    value -= hours * 3600
    minutes = int(value // 60)
    seconds = value - minutes * 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


def time_token_from_hms(value: str) -> str:
    return str(value or "").replace(":", "").replace(".", "")


def safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or ""))
    return token.strip("_") or "unknown"


def parse_raw_key_start_seconds(raw_key: str) -> float | None:
    match = re.search(r"_(\d{6,8})(?:\.mp4)?$", str(raw_key or ""))
    if not match:
        return None
    digits = match.group(1)
    if len(digits) < 6:
        return None
    hours = int(digits[0:2])
    minutes = int(digits[2:4])
    seconds = int(digits[4:6])
    hundredths = int(digits[6:8]) if len(digits) >= 8 else 0
    return hours * 3600.0 + minutes * 60.0 + seconds + hundredths / 100.0


def reconstruct_raw_key(row: dict[str, str], participant_folder: str) -> str:
    start = row.get("start_time", "")
    if not start or not participant_folder:
        return ""
    seconds = hms_to_seconds(start)
    hours = int(seconds // 3600)
    seconds -= hours * 3600
    minutes = int(seconds // 60)
    sec = seconds - minutes * 60
    whole = int(sec)
    hundredths = int(round((sec - whole) * 100))
    if hundredths >= 100:
        whole += 1
        hundredths -= 100
    return f"DAY{row.get('day', '')}_{participant_folder}_{hours:02d}{minutes:02d}{whole:02d}{hundredths:02d}.mp4"


def derive_source(row: dict[str, str]) -> dict[str, str]:
    agent = row.get("source_agent", "")
    participant = PARTICIPANT_FOLDERS.get(agent, "")
    day = row.get("day", "")
    raw_key = row.get("raw_key", "").strip()
    reason_parts: list[str] = []
    if not participant:
        reason_parts.append("unknown source_agent participant mapping")
    if not day:
        reason_parts.append("missing day")
    if not raw_key:
        raw_key = reconstruct_raw_key(row, participant)
        if raw_key:
            reason_parts.append("raw_key reconstructed from source_agent/day/start_time")
        else:
            reason_parts.append("insufficient_metadata: missing raw_key and cannot reconstruct filename")
    if participant and day and raw_key:
        hf_path = f"{participant}/DAY{day}/{raw_key}"
        can_resolve = "yes"
        if not reason_parts:
            reason_parts.append("resolved from raw_key and participant mapping")
    else:
        hf_path = ""
        can_resolve = "no"
    return {
        "participant_folder": participant,
        "raw_key": raw_key,
        "hf_path": hf_path,
        "can_resolve_video": can_resolve,
        "reason": "; ".join(reason_parts),
    }


def build_video_source_plan(frame_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in frame_rows:
        source = derive_source(row)
        rows.append(
            {
                "visual_case_id": row.get("visual_case_id", ""),
                "dataset_case_id": row.get("dataset_case_id", ""),
                "evidence_scope": row.get("evidence_scope", ""),
                "source_agent": row.get("source_agent", ""),
                "participant_folder": source["participant_folder"],
                "day": row.get("day", ""),
                "raw_key": source["raw_key"],
                "hf_repo_id": HF_REPO_ID,
                "hf_path": source["hf_path"],
                "can_resolve_video": source["can_resolve_video"],
                "reason": source["reason"],
            }
        )
    return rows


def locate_ffmpeg(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    which = shutil.which("ffmpeg")
    if which:
        return Path(which)
    for path in KNOWN_FFMPEG_CANDIDATES:
        if path.exists():
            return path
    return None


def ffprobe_for(ffmpeg_path: Path | None) -> Path | None:
    if not ffmpeg_path:
        return None
    candidate = ffmpeg_path.with_name("ffprobe")
    return candidate if candidate.exists() else None


def command_first_line(cmd: list[str]) -> str:
    try:
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    return (completed.stdout or "").splitlines()[0] if completed.stdout else f"exit_code={completed.returncode}"


def download_videos(video_source_rows: list[dict[str, str]], video_cache: Path, skip_download: bool) -> list[dict[str, str]]:
    unique_paths = sorted({row["hf_path"] for row in video_source_rows if row.get("can_resolve_video") == "yes" and row.get("hf_path")})
    video_cache.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []

    if skip_download:
        for hf_path in unique_paths:
            local_path = video_cache / hf_path
            status = "cached" if local_path.exists() and local_path.stat().st_size > 0 else "skipped_download"
            size_mb = local_path.stat().st_size / (1024 * 1024) if local_path.exists() else 0.0
            manifest.append(
                {
                    "hf_path": hf_path,
                    "local_video_path": str(local_path) if local_path.exists() else "",
                    "file_size_mb": f"{size_mb:.3f}" if size_mb else "",
                    "download_status": status,
                    "error": "" if status == "cached" else "download skipped by flag",
                }
            )
        return manifest

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        return [
            {
                "hf_path": hf_path,
                "local_video_path": "",
                "file_size_mb": "",
                "download_status": "huggingface_hub_missing",
                "error": str(exc),
            }
            for hf_path in unique_paths
        ]

    for hf_path in unique_paths:
        local_path = video_cache / hf_path
        if local_path.exists() and local_path.stat().st_size > 0:
            manifest.append(
                {
                    "hf_path": hf_path,
                    "local_video_path": str(local_path),
                    "file_size_mb": f"{local_path.stat().st_size / (1024 * 1024):.3f}",
                    "download_status": "cached",
                    "error": "",
                }
            )
            continue
        try:
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=hf_path,
                repo_type=HF_REPO_TYPE,
                local_dir=str(video_cache),
            )
            downloaded_path = Path(downloaded)
            manifest.append(
                {
                    "hf_path": hf_path,
                    "local_video_path": str(downloaded_path),
                    "file_size_mb": f"{downloaded_path.stat().st_size / (1024 * 1024):.3f}",
                    "download_status": "downloaded",
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001 - errors are recorded per file.
            manifest.append(
                {
                    "hf_path": hf_path,
                    "local_video_path": "",
                    "file_size_mb": "",
                    "download_status": "download_failed",
                    "error": trunc(repr(exc), 500),
                }
            )
    return manifest


def output_frame_path(row: dict[str, str], frames_dir: Path) -> Path:
    token = time_token_from_hms(row.get("target_frame_time", ""))
    return frames_dir / (
        f"{safe_token(row.get('visual_case_id', ''))}_"
        f"{safe_token(row.get('evidence_scope', ''))}_"
        f"{safe_token(row.get('source_agent', ''))}_"
        f"D{safe_token(row.get('day', ''))}_"
        f"{safe_token(token)}.jpg"
    )


def relative_target_time(row: dict[str, str], source: dict[str, str]) -> float:
    target_abs = hms_to_seconds(row.get("target_frame_time", ""))
    raw_start = parse_raw_key_start_seconds(source.get("raw_key", ""))
    if raw_start is None:
        raw_start = hms_to_seconds(row.get("start_time", ""))
    return max(0.0, target_abs - raw_start)


def extract_frame(ffmpeg_path: Path, local_video_path: str, rel_seconds: float, out_path: Path) -> tuple[str, str]:
    if out_path.exists() and out_path.stat().st_size > 0:
        return "already_extracted", ""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg_path),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-ss",
        f"{rel_seconds:.3f}",
        "-i",
        local_video_path,
        "-frames:v",
        "1",
        "-strict",
        "unofficial",
        "-q:v",
        "2",
        str(out_path),
    ]
    try:
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, timeout=90)
    except subprocess.TimeoutExpired:
        return "ffmpeg_timeout", "ffmpeg timed out after 90s"
    except OSError as exc:
        return "ffmpeg_os_error", str(exc)
    if completed.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
        return "extracted", ""
    return "ffmpeg_failed", trunc(normalize_ws(completed.stderr or completed.stdout or f"exit_code={completed.returncode}"), 500)


def probe_duration(ffmpeg_path: Path | None, local_video_path: str, cache: dict[str, float | None]) -> float | None:
    if local_video_path in cache:
        return cache[local_video_path]
    if not ffmpeg_path:
        cache[local_video_path] = None
        return None
    probe = ffprobe_for(ffmpeg_path)
    if not probe:
        cache[local_video_path] = None
        return None
    completed = subprocess.run(
        [
            str(probe),
            "-hide_banner",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            local_video_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    try:
        duration = float((completed.stdout or "").strip())
    except ValueError:
        duration = None
    cache[local_video_path] = duration
    return duration


def build_extraction_manifest(
    frame_rows: list[dict[str, str]],
    download_manifest: list[dict[str, str]],
    frames_dir: Path,
    ffmpeg_path: Path | None,
) -> list[dict[str, str]]:
    download_by_hf = {row["hf_path"]: row for row in download_manifest}
    manifest: list[dict[str, str]] = []
    duration_cache: dict[str, float | None] = {}
    for row in frame_rows:
        source = derive_source(row)
        hf_path = source["hf_path"]
        download = download_by_hf.get(hf_path, {})
        local_video_path = download.get("local_video_path", "")
        out_path = output_frame_path(row, frames_dir)
        rel_seconds = relative_target_time(row, source) if source["can_resolve_video"] == "yes" else 0.0
        planned_rel_seconds = rel_seconds
        clamp_note = ""

        if source["can_resolve_video"] != "yes":
            status, error = "insufficient_metadata", source["reason"]
        elif not local_video_path:
            status = "video_unavailable"
            error = download.get("error", "video was not downloaded")
        elif not ffmpeg_path:
            status, error = "ffmpeg_unavailable", "no ffmpeg executable available"
        else:
            duration = probe_duration(ffmpeg_path, local_video_path, duration_cache)
            if duration is not None and duration > 0 and rel_seconds >= duration:
                rel_seconds = max(0.0, duration - 0.5)
                clamp_note = (
                    f"planned relative {planned_rel_seconds:.3f}s exceeded clip duration "
                    f"{duration:.3f}s; used {rel_seconds:.3f}s"
                )
            status, error = extract_frame(ffmpeg_path, local_video_path, rel_seconds, out_path)
            if clamp_note and status in {"extracted", "already_extracted"}:
                status = f"{status}_clamped_to_clip_end"
                error = clamp_note

        manifest.append(
            {
                "visual_case_id": row.get("visual_case_id", ""),
                "dataset_case_id": row.get("dataset_case_id", ""),
                "evidence_scope": row.get("evidence_scope", ""),
                "source_agent": row.get("source_agent", ""),
                "hf_path": hf_path,
                "local_video_path": local_video_path,
                "target_frame_time_absolute": row.get("target_frame_time", ""),
                "target_frame_time_relative": seconds_to_hms(rel_seconds),
                "output_frame_path": str(out_path) if out_path.exists() and out_path.stat().st_size > 0 else "",
                "extraction_status": status,
                "error": error,
            }
        )
    return manifest


def try_import_pillow() -> tuple[Any, Any, Any] | tuple[None, None, None]:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ModuleNotFoundError:
        return None, None, None
    return Image, ImageDraw, ImageFont


def pil_font(ImageFont: Any, size: int, bold: bool = False) -> Any:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
    return ImageFont.load_default()


def draw_wrapped(draw: Any, text: str, x: int, y: int, width: int, font: Any, fill: tuple[int, int, int], max_lines: int) -> int:
    words = normalize_ws(text).split()
    lines: list[str] = []
    line = ""
    for word in words:
        proposed = word if not line else f"{line} {word}"
        bbox = draw.textbbox((0, 0), proposed, font=font)
        if bbox[2] - bbox[0] <= width:
            line = proposed
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(". ") + "..."
    line_height = draw.textbbox((0, 0), "Ag", font=font)[3] + 8
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += line_height
    return y


def unique_caption_blocks(frame_rows: list[dict[str, str]], visual_case_id: str, scope: str, limit: int = 1200) -> str:
    seen: set[tuple[str, str, str, str]] = set()
    blocks: list[str] = []
    for row in frame_rows:
        if row.get("visual_case_id") != visual_case_id or row.get("evidence_scope") != scope:
            continue
        sig = (row.get("source_agent", ""), row.get("day", ""), row.get("start_time", ""), row.get("end_time", ""))
        if sig in seen:
            continue
        seen.add(sig)
        blocks.append(
            f"[{scope} source={row.get('source_agent')} D{row.get('day')} "
            f"{row.get('start_time')}-{row.get('end_time')} {row.get('granularity')}] "
            f"{row.get('caption_text', '')}"
        )
    return trunc("\n".join(blocks), limit)


def create_contact_sheet(
    case: dict[str, str],
    frame_rows: list[dict[str, str]],
    extraction_rows: list[dict[str, str]],
    contact_dir: Path,
) -> str:
    Image, ImageDraw, ImageFont = try_import_pillow()
    if Image is None:
        return ""
    contact_dir.mkdir(parents=True, exist_ok=True)
    visual_case_id = case["visual_case_id"]
    out_path = contact_dir / f"{visual_case_id}.png"
    image = Image.new("RGB", (1800, 2600), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    title_font = pil_font(ImageFont, 42, bold=True)
    meta_font = pil_font(ImageFont, 24)
    body_font = pil_font(ImageFont, 24)
    small_font = pil_font(ImageFont, 18)
    border = (180, 185, 190)
    navy = (22, 40, 64)
    muted = (89, 99, 110)
    blue = (36, 86, 148)
    green = (32, 115, 84)
    red = (155, 54, 54)

    y = 40
    draw.text((42, y), visual_case_id, font=title_font, fill=navy)
    y += 52
    draw.text((42, y), f"Split {case.get('source_split', '')} | Category {case.get('category', '')}", font=meta_font, fill=muted)
    y += 40
    y = draw_wrapped(draw, f"Q: {case.get('question', '')}", 42, y, 1700, body_font, navy, 4)
    y = draw_wrapped(draw, f"A: {case.get('answer', '')}", 42, y + 6, 1700, body_font, (40, 40, 40), 3)
    meta = (
        f"Expected: {case.get('expected_result', '')} | "
        f"Current: {case.get('current_only_answerability', '')} | "
        f"Current+history: {case.get('current_plus_history_answerability', '')} | "
        f"Gain: {case.get('current_plus_history_gain', '')}"
    )
    y = draw_wrapped(draw, meta, 42, y + 6, 1700, meta_font, muted, 2) + 20

    extraction_by_key = defaultdict(list)
    for row in extraction_rows:
        if row.get("visual_case_id") == visual_case_id:
            extraction_by_key[row.get("evidence_scope", "")].append(row)

    for scope, title, fill, title_color in [
        ("current_only", "Current-only evidence frames", (229, 239, 250), blue),
        ("history_only", "Historical evidence frames", (230, 244, 236), green),
    ]:
        draw.rectangle((42, y, 1758, y + 42), fill=fill, outline=border, width=2)
        draw.text((58, y + 8), title, font=meta_font, fill=title_color)
        y += 58
        rows = extraction_by_key.get(scope, [])
        box_w, box_h = 540, 182
        x0 = 58
        for idx, row in enumerate(rows[:6]):
            col, r = idx % 3, idx // 3
            x, yy = x0 + col * (box_w + 22), y + r * (box_h + 18)
            draw.rectangle((x, yy, x + box_w, yy + box_h), fill=(255, 255, 255), outline=border, width=2)
            frame_path = Path(row.get("output_frame_path", ""))
            if frame_path.exists():
                try:
                    thumb = Image.open(frame_path).convert("RGB")
                    thumb.thumbnail((box_w - 20, 122))
                    image.paste(thumb, (x + (box_w - thumb.width) // 2, yy + 10))
                except OSError:
                    draw.rectangle((x + 12, yy + 12, x + box_w - 12, yy + 128), fill=(249, 232, 232), outline=border)
                    draw.text((x + 20, yy + 55), "Frame open failed", font=small_font, fill=red)
            else:
                draw.rectangle((x + 12, yy + 12, x + box_w - 12, yy + 128), fill=(249, 232, 232), outline=border)
                draw_wrapped(draw, row.get("extraction_status", "missing"), x + 20, yy + 52, box_w - 40, small_font, red, 2)
            label = (
                f"{row.get('source_agent')} | {row.get('target_frame_time_absolute')} "
                f"| rel {row.get('target_frame_time_relative')}"
            )
            draw_wrapped(draw, label, x + 14, yy + 136, box_w - 28, small_font, title_color, 2)
        y += 2 * (box_h + 18) + 12
        caption = unique_caption_blocks(frame_rows, visual_case_id, scope, limit=900)
        y = draw_wrapped(draw, f"Caption: {caption}", 58, y, 1660, small_font, (35, 35, 35), 6) + 20

    status_counts = Counter(row.get("extraction_status", "") for row in extraction_rows if row.get("visual_case_id") == visual_case_id)
    status = " | ".join(f"{key}: {value}" for key, value in sorted(status_counts.items())) or "no targets"
    draw.rectangle((42, min(y, 2440), 1758, min(y, 2440) + 96), fill=(245, 245, 245), outline=border, width=2)
    draw_wrapped(draw, f"Extraction status: {status}", 58, min(y, 2440) + 24, 1660, meta_font, muted, 2)
    image.save(out_path)
    return str(out_path)


def rel_path(path: str | Path, base: Path = AUDIT_DIR) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def build_gallery(
    case_rows: list[dict[str, str]],
    frame_rows: list[dict[str, str]],
    extraction_rows: list[dict[str, str]],
    contact_paths: dict[str, str],
    output_path: Path,
) -> None:
    extracted_by_case = defaultdict(list)
    for row in extraction_rows:
        if row.get("output_frame_path"):
            extracted_by_case[row["visual_case_id"]].append(row)
    sections: list[str] = []
    for case in case_rows:
        visual_case_id = case["visual_case_id"]
        status_counts = Counter(row.get("extraction_status", "") for row in extraction_rows if row.get("visual_case_id") == visual_case_id)
        status = ", ".join(f"{k}: {v}" for k, v in sorted(status_counts.items())) or "no targets"
        frames = []
        for row in extracted_by_case.get(visual_case_id, []):
            rel = rel_path(row["output_frame_path"])
            label = f"{row.get('evidence_scope')} | {row.get('source_agent')} | {row.get('target_frame_time_absolute')}"
            frames.append(
                f"<figure><img src=\"{html.escape(rel)}\" alt=\"{html.escape(label)}\"><figcaption>{html.escape(label)}</figcaption></figure>"
            )
        frame_html = "<div class=\"frames\">" + "".join(frames) + "</div>" if frames else "<p class=\"none\">No extracted frames available.</p>"
        sections.append(
            f"""
<section class="case">
  <h2>{html.escape(visual_case_id)} <span>{html.escape(case.get('source_split', ''))}</span></h2>
  <p><b>Dataset case:</b> {html.escape(case.get('dataset_case_id', ''))} <b>Question ID:</b> {html.escape(case.get('question_id', ''))} <b>Category:</b> {html.escape(case.get('category', ''))}</p>
  <p><b>Question:</b> {html.escape(case.get('question', ''))}</p>
  <p><b>Answer:</b> {html.escape(case.get('answer', ''))}</p>
  <p><b>Expected result:</b> {html.escape(case.get('expected_result', ''))}</p>
  <p><b>Extraction status:</b> {html.escape(status)}</p>
  <div class="sheet"><img src="{html.escape(rel_path(contact_paths.get(visual_case_id, '')))}" alt="{html.escape(visual_case_id)} contact sheet"></div>
  <h3>Extracted frames</h3>
  {frame_html}
  <div class="cols">
    <div><h3>Current-only caption evidence</h3><pre>{html.escape(unique_caption_blocks(frame_rows, visual_case_id, 'current_only', 2400))}</pre></div>
    <div><h3>Historical caption evidence</h3><pre>{html.escape(unique_caption_blocks(frame_rows, visual_case_id, 'history_only', 2400))}</pre></div>
  </div>
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
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dataset V0.1 Visual Audit With Frames</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; background: #f7f7f5; }}
.case {{ background: #fff; border: 1px solid #d7dce0; margin: 0 0 28px; padding: 18px; }}
h2 {{ color: #18324d; margin: 0 0 8px; }}
h2 span, .none {{ color: #64707d; font-size: 14px; }}
.sheet img {{ max-width: 100%; border: 1px solid #cfd6dd; margin: 12px 0; background: #fff; }}
.frames {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; }}
.frames figure {{ margin: 0; border: 1px solid #d4dbe2; padding: 6px; background: #fbfbfb; }}
.frames img {{ width: 100%; display: block; }}
.frames figcaption {{ color: #64707d; font-size: 13px; }}
.cols {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
pre {{ white-space: pre-wrap; font-size: 13px; line-height: 1.35; background: #f2f4f6; padding: 12px; overflow-wrap: anywhere; }}
.check {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
.check th, .check td {{ border: 1px solid #ccd3da; padding: 8px; text-align: left; height: 28px; }}
.check th {{ width: 240px; background: #eef2f5; }}
</style>
</head>
<body>
<h1>Dataset V0.1 Visual Audit With Frames</h1>
<p>Targeted human-inspection packet. Extracted frames are only for caption/evidence plausibility checks.</p>
{''.join(sections)}
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")


def case_status_and_paths(extraction_rows: list[dict[str, str]], visual_case_id: str) -> tuple[str, list[str]]:
    rows = [row for row in extraction_rows if row.get("visual_case_id") == visual_case_id]
    paths = [row["output_frame_path"] for row in rows if row.get("output_frame_path")]
    if not rows:
        return "no_frame_targets", []
    if len(paths) == len(rows):
        return "all_frames_available", paths
    if paths:
        return "partial_frames_available", paths
    return "frames_missing", []


def build_audit_table(
    case_rows: list[dict[str, str]],
    frame_rows: list[dict[str, str]],
    extraction_rows: list[dict[str, str]],
    contact_paths: dict[str, str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for case in case_rows:
        visual_case_id = case["visual_case_id"]
        status, paths = case_status_and_paths(extraction_rows, visual_case_id)
        rows.append(
            {
                "visual_case_id": visual_case_id,
                "dataset_case_id": case.get("dataset_case_id", ""),
                "question_id": case.get("question_id", ""),
                "split": case.get("source_split", ""),
                "question": case.get("question", ""),
                "answer": case.get("answer", ""),
                "category": case.get("category", ""),
                "current_only_caption": unique_caption_blocks(frame_rows, visual_case_id, "current_only", 1400),
                "historical_caption": unique_caption_blocks(frame_rows, visual_case_id, "history_only", 1400),
                "contact_sheet_path": rel_path(contact_paths.get(visual_case_id, "")),
                "extracted_frame_paths": "; ".join(rel_path(path) for path in paths),
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


def confidence_score(case: dict[str, str]) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(case.get("confidence", ""), 0)


def top_ppt_cases(case_rows: list[dict[str, str]], extraction_rows: list[dict[str, str]], limit: int = 5) -> list[dict[str, str]]:
    frame_counts = Counter(row.get("visual_case_id", "") for row in extraction_rows if row.get("output_frame_path"))
    indexed = list(enumerate(case_rows))

    def score(item: tuple[int, dict[str, str]]) -> tuple[int, int, int, int, int]:
        idx, case = item
        return (
            frame_counts.get(case["visual_case_id"], 0),
            1 if case.get("source_split") == "demo" else 0,
            1 if case.get("expected_result") == "current_plus_history_better" else 0,
            confidence_score(case),
            -idx,
        )

    return [case for _, case in sorted(indexed, key=score, reverse=True)[:limit] if frame_counts.get(case["visual_case_id"], 0) > 0]


def build_report(
    ffmpeg_path: Path | None,
    video_source_rows: list[dict[str, str]],
    download_manifest: list[dict[str, str]],
    extraction_manifest: list[dict[str, str]],
    contact_paths: dict[str, str],
    case_rows: list[dict[str, str]],
    report_path: Path,
) -> None:
    unique_requested = sorted({row["hf_path"] for row in video_source_rows if row.get("can_resolve_video") == "yes" and row.get("hf_path")})
    downloaded = [row for row in download_manifest if row.get("download_status") in {"downloaded", "cached"}]
    extracted = [row for row in extraction_manifest if row.get("output_frame_path")]
    missing = [row for row in extraction_manifest if not row.get("output_frame_path")]
    clamped = [row for row in extraction_manifest if "clamped_to_clip_end" in row.get("extraction_status", "")]
    blockers: list[str] = []
    if not ffmpeg_path:
        blockers.append("ffmpeg is unavailable; no frame extraction can run.")
    if len(downloaded) < len(unique_requested):
        blockers.append("Some required EgoLife clips could not be downloaded from Hugging Face.")
    if missing:
        status_counts = Counter(row.get("extraction_status", "") for row in missing)
        blockers.append("Missing frame statuses: " + ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())))
    if clamped:
        blockers.append(f"{len(clamped)} planned frame times exceeded the source clip duration and were clamped to the clip end.")
    if not top_ppt_cases(case_rows, extraction_manifest):
        blockers.append("No cases have extracted frames yet, so there are no frame-backed PPT-ready cases.")
    blockers.append("Visual frames alone cannot verify dialogue, speaker identity, or intent.")
    ppt = top_ppt_cases(case_rows, extraction_manifest)

    text = f"""# Dataset V0.1 Visual Audit With Frames Report

## FFmpeg Status

- ffmpeg path: {ffmpeg_path or ''}
- ffmpeg version: {command_first_line([str(ffmpeg_path), '-version']) if ffmpeg_path else 'unavailable'}
- ffprobe version: {command_first_line([str(ffprobe_for(ffmpeg_path)), '-version']) if ffprobe_for(ffmpeg_path) else 'unavailable'}

## Counts

- Unique videos requested: {len(unique_requested)}
- Unique videos downloaded or cached: {len(downloaded)}
- Frames planned: {len(extraction_manifest)}
- Frames extracted: {len(extracted)}
- Frames missing: {len(missing)}
- Contact sheets generated: {sum(1 for path in contact_paths.values() if path and Path(path).exists())}

## Download Status

{format_counter(Counter(row.get('download_status', '') for row in download_manifest))}

## Extraction Status

{format_counter(Counter(row.get('extraction_status', '') for row in extraction_manifest))}

## Top 5 PPT-Ready Cases With Frames

{format_ppt_cases(ppt, extraction_manifest)}

## Blockers

{format_bullets(blockers)}

## Claim Boundary

Can say:
- We built targeted visual audit packets for a small Dataset V0.1 subset.
- Frames are used only for sanity-checking caption/evidence plausibility.

Cannot say:
- QA labels are final.
- Dataset is fully verified.
- Historical memory is proven useful.
- Model accuracy improves.
"""
    report_path.write_text(text, encoding="utf-8")


def format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "- none"
    return "\n".join(f"- {key}: {value}" for key, value in sorted(counter.items()))


def format_bullets(items: list[str]) -> str:
    if not items:
        return "- none"
    return "\n".join(f"- {item}" for item in items)


def format_ppt_cases(cases: list[dict[str, str]], extraction_rows: list[dict[str, str]]) -> str:
    if not cases:
        return "- none"
    frame_counts = Counter(row.get("visual_case_id", "") for row in extraction_rows if row.get("output_frame_path"))
    return "\n".join(
        f"- {case['visual_case_id']} / {case.get('dataset_case_id', '')} / Q{case.get('question_id', '')} "
        f"({frame_counts.get(case['visual_case_id'], 0)} frames): {trunc(case.get('question', ''), 140)}"
        for case in cases
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Dataset V0.1 visual audit packet with targeted frames.")
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--ffmpeg-path", type=str, default=None)
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit_dir: Path = args.audit_dir
    frame_plan_csv = audit_dir / "visual_frame_extraction_plan.csv"
    case_subset_csv = audit_dir / "visual_audit_case_subset.csv"
    video_cache = audit_dir / "video_cache"
    frames_dir = audit_dir / "frames"
    contact_dir = audit_dir / "contact_sheets_with_frames"

    if not frame_plan_csv.exists() or not case_subset_csv.exists():
        raise FileNotFoundError("visual_frame_extraction_plan.csv and visual_audit_case_subset.csv are required")

    frame_rows = read_csv(frame_plan_csv)
    case_rows = read_csv(case_subset_csv)
    video_source_rows = build_video_source_plan(frame_rows)
    write_csv(audit_dir / "visual_video_source_plan_v1.csv", video_source_rows, VIDEO_SOURCE_FIELDS)

    download_manifest = download_videos(video_source_rows, video_cache, args.skip_download)
    write_csv(audit_dir / "visual_video_download_manifest_v1.csv", download_manifest, DOWNLOAD_MANIFEST_FIELDS)

    ffmpeg_path = locate_ffmpeg(args.ffmpeg_path)
    extraction_manifest = build_extraction_manifest(frame_rows, download_manifest, frames_dir, ffmpeg_path)
    write_csv(audit_dir / "visual_frame_extraction_manifest_v1.csv", extraction_manifest, EXTRACTION_MANIFEST_FIELDS)

    contact_paths: dict[str, str] = {}
    for case in case_rows:
        contact_paths[case["visual_case_id"]] = create_contact_sheet(case, frame_rows, extraction_manifest, contact_dir)

    build_gallery(case_rows, frame_rows, extraction_manifest, contact_paths, audit_dir / "visual_audit_packet_gallery_with_frames.html")
    write_csv(audit_dir / "visual_audit_table_with_frames.csv", build_audit_table(case_rows, frame_rows, extraction_manifest, contact_paths), AUDIT_TABLE_FIELDS)
    build_report(
        ffmpeg_path,
        video_source_rows,
        download_manifest,
        extraction_manifest,
        contact_paths,
        case_rows,
        audit_dir / "visual_audit_with_frames_report.md",
    )

    downloaded_count = sum(1 for row in download_manifest if row.get("download_status") in {"downloaded", "cached"})
    extracted_count = sum(1 for row in extraction_manifest if row.get("output_frame_path"))
    contact_count = sum(1 for path in contact_paths.values() if path and Path(path).exists())
    source_success = sum(1 for row in video_source_rows if row.get("can_resolve_video") == "yes")
    source_failure = len(video_source_rows) - source_success
    gallery_path = audit_dir / "visual_audit_packet_gallery_with_frames.html"

    print(f"ffmpeg path: {ffmpeg_path or ''}")
    print(f"video source success/failure: {source_success}/{source_failure}")
    print(f"downloaded video count: {downloaded_count}")
    print(f"extracted frame count: {extracted_count}")
    print(f"contact sheet count: {contact_count}")
    print(f"gallery path: {gallery_path}")
    print("5 best PPT cases with extracted frames:")
    ppt = top_ppt_cases(case_rows, extraction_manifest)
    if ppt:
        frame_counts = Counter(row.get("visual_case_id", "") for row in extraction_manifest if row.get("output_frame_path"))
        for case in ppt:
            print(f"- {case['visual_case_id']} {case.get('dataset_case_id', '')} Q{case.get('question_id', '')}: {frame_counts[case['visual_case_id']]} frames")
    else:
        print("- none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
