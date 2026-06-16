"""Prepare two-user evidence packets from the EgoLife manifest."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
import shutil
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from .gaze_projection import (
    find_clip_calibration,
    load_aria_projection_calibration,
    project_gaze_csv_with_projectaria_tools,
    project_gaze_row,
    summarize_projected_gaze,
)
from .io_utils import download_file, read_json, stable_id, write_jsonl
from .manifest import seconds_from_time_token

PAIR_STRATEGIES = ("first", "balanced")
GROUP_STRATEGIES = ("chronological", "time_stratified", "random")


def group_manifest_clips(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for clip in manifest.get("clips", []):
        grouped[(clip["day"], clip["time_token"])].append(clip)

    groups = []
    for (day, time_token), clips in sorted(grouped.items()):
        unique_agents = sorted({clip["agent_dir"] for clip in clips})
        if len(unique_agents) < 2:
            continue
        groups.append(
            {
                "day": day,
                "time_token": time_token,
                "clip_clock": clips[0].get("clip_clock"),
                "agents": unique_agents,
                "clips": sorted(clips, key=lambda c: c["agent_dir"]),
            }
        )
    return groups


def _group_clock_seconds(group: dict[str, Any]) -> float | None:
    try:
        return seconds_from_time_token(str(group.get("time_token", "")))
    except ValueError:
        return None


def _passes_min_gap(
    group: dict[str, Any],
    selected: list[dict[str, Any]],
    min_clock_gap_seconds: float,
) -> bool:
    if min_clock_gap_seconds <= 0:
        return True
    seconds = _group_clock_seconds(group)
    if seconds is None:
        return True
    day = group.get("day")
    for other in selected:
        if other.get("day") != day:
            continue
        other_seconds = _group_clock_seconds(other)
        if other_seconds is None:
            continue
        if abs(seconds - other_seconds) < min_clock_gap_seconds:
            return False
    return True


def _prefix_spread_order(length: int) -> list[int]:
    """Return indices in a quantile-spread order so early prefixes cover time."""

    if length <= 0:
        return []
    order = [0]
    if length > 1:
        order.append(length - 1)
    seen = set(order)
    denominator = 2
    while len(order) < length:
        for numerator in range(1, denominator, 2):
            index = round((length - 1) * numerator / denominator)
            if index not in seen:
                order.append(index)
                seen.add(index)
                if len(order) >= length:
                    break
        denominator *= 2
    return order


def _choose_from_bin(
    groups: list[dict[str, Any]],
    indices: list[int],
    selected: list[dict[str, Any]],
    rng: random.Random,
    min_clock_gap_seconds: float,
) -> int | None:
    if not indices:
        return None
    center = indices[len(indices) // 2]
    candidates = sorted(indices, key=lambda idx: (abs(idx - center), rng.random()))
    for index in candidates:
        if _passes_min_gap(groups[index], selected, min_clock_gap_seconds):
            return index
    return None


def order_evidence_groups(
    groups: list[dict[str, Any]],
    *,
    target_count: int,
    group_strategy: str = "chronological",
    sampling_seed: int = 0,
    min_clock_gap_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    if group_strategy not in GROUP_STRATEGIES:
        raise ValueError(f"group_strategy must be one of {GROUP_STRATEGIES}")
    if target_count <= 0:
        return []
    groups = list(groups)
    if group_strategy == "chronological":
        selected: list[dict[str, Any]] = []
        for group in groups:
            if len(selected) >= target_count:
                break
            if _passes_min_gap(group, selected, min_clock_gap_seconds):
                selected.append(group)
        return selected
    if group_strategy == "random":
        rng = random.Random(sampling_seed)
        shuffled = list(groups)
        rng.shuffle(shuffled)
        selected = []
        for group in shuffled:
            if len(selected) >= target_count:
                break
            if _passes_min_gap(group, selected, min_clock_gap_seconds):
                selected.append(group)
        return selected

    rng = random.Random(sampling_seed)
    bin_count = min(target_count, len(groups))
    selected_indices: list[int] = []
    selected_groups: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    for bin_index in range(bin_count):
        start = math.floor(bin_index * len(groups) / bin_count)
        end = math.floor((bin_index + 1) * len(groups) / bin_count)
        indices = [idx for idx in range(start, end) if idx not in used_indices]
        chosen = _choose_from_bin(groups, indices, selected_groups, rng, min_clock_gap_seconds)
        if chosen is None:
            continue
        selected_indices.append(chosen)
        used_indices.add(chosen)
        selected_groups.append(groups[chosen])

    if len(selected_indices) < target_count:
        fallback_indices = sorted(
            (idx for idx in range(len(groups)) if idx not in used_indices),
            key=lambda idx: rng.random(),
        )
        for index in fallback_indices:
            if len(selected_indices) >= target_count:
                break
            group = groups[index]
            if not _passes_min_gap(group, selected_groups, min_clock_gap_seconds):
                continue
            selected_indices.append(index)
            used_indices.add(index)
            selected_groups.append(group)

    selected_indices = sorted(selected_indices)
    spread_order = _prefix_spread_order(len(selected_indices))
    return [groups[selected_indices[index]] for index in spread_order]


def _safe_rel_path(repo_path: str) -> Path:
    return Path(*Path(repo_path).parts)


def local_cache_path(cache_dir: str | Path, repo_path: str) -> Path:
    return Path(cache_dir) / _safe_rel_path(repo_path)


def ffprobe_duration(video_path: str | Path) -> float | None:
    if not shutil.which("ffprobe"):
        return None
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    frames_per_clip: int = 3,
    duration: float | None = None,
) -> list[dict[str, Any]]:
    """Extract evenly spaced frames with ffmpeg."""

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for frame extraction")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = duration if duration is not None else ffprobe_duration(video_path)
    if duration is None or duration <= 0:
        duration = 30.0
    frames_per_clip = max(1, frames_per_clip)
    timestamps = [
        min(duration - 0.05, max(0.0, duration * (idx + 1) / (frames_per_clip + 1)))
        for idx in range(frames_per_clip)
    ]
    rows: list[dict[str, Any]] = []
    for idx, timestamp in enumerate(timestamps, 1):
        frame_path = output_dir / f"frame_{idx:02d}_{timestamp:.2f}s.jpg"
        if not frame_path.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(frame_path),
                ],
                check=True,
            )
        rows.append({"timestamp_seconds": round(timestamp, 3), "path": str(frame_path)})
    return rows


def summarize_gaze_csv(
    path: str | Path,
    *,
    max_rows: int = 5000,
    calibration_path: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize EyeGaze CSV and optionally project CPF gaze to image pixels.

    EgoLife gaze CSV values are Aria CPF yaw/pitch/depth values. Without a
    calibration file, this function intentionally does not produce 2D pixel
    gaze coordinates.
    """

    yaw_values: list[float] = []
    pitch_values: list[float] = []
    depth_values: list[float] = []
    projected_points: list[dict[str, Any]] = []
    calibration = None
    projectaria_projection_summary: dict[str, Any] | None = None
    if calibration_path:
        suffix = Path(calibration_path).suffix.lower()
        if suffix in {".vrs", ".jsonl"}:
            try:
                projected_points, projectaria_projection_summary = project_gaze_csv_with_projectaria_tools(
                    path,
                    calibration_path,
                    max_rows=max_rows,
                )
            except RuntimeError as exc:
                projectaria_projection_summary = {
                    "projection_status": "projection_failed",
                    "calibration_path": str(calibration_path),
                    "projection_error": str(exc),
                }
        else:
            calibration = load_aria_projection_calibration(calibration_path)
    projection_errors: list[str] = []
    first_ts = None
    last_ts = None
    total = 0
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for row in reader:
            total += 1
            if first_ts is None:
                first_ts = row.get("tracking_timestamp_us")
            last_ts = row.get("tracking_timestamp_us")
            if total > max_rows:
                continue
            for target, values in [
                ("pitch_rads_cpf", pitch_values),
                ("depth_m", depth_values),
            ]:
                try:
                    values.append(float(row[target]))
                except (KeyError, TypeError, ValueError):
                    pass
            try:
                left = float(row["left_yaw_rads_cpf"])
                right = float(row["right_yaw_rads_cpf"])
                yaw_values.append((left + right) / 2.0)
            except (KeyError, TypeError, ValueError):
                pass
            if calibration is not None:
                try:
                    projected = project_gaze_row(row, calibration)
                    if projected is not None:
                        projected_points.append(projected)
                except ValueError as exc:
                    if len(projection_errors) < 3:
                        projection_errors.append(str(exc))

    def stats(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {
            "min": round(min(values), 5),
            "max": round(max(values), 5),
            "median": round(statistics.median(values), 5),
        }

    summary = {
        "row_count": total,
        "sampled_rows": min(total, max_rows),
        "first_tracking_timestamp_us": first_ts,
        "last_tracking_timestamp_us": last_ts,
        "columns": fields,
        "yaw_rads_summary": stats(yaw_values),
        "pitch_rads_summary": stats(pitch_values),
        "depth_m_summary": stats(depth_values),
        "gaze_coordinate_frame": "Aria Central Pupil Frame (CPF); not image pixels",
    }
    if calibration_path and projectaria_projection_summary is not None:
        summary["projection_status"] = projectaria_projection_summary["projection_status"]
        summary["projected_gaze_summary"] = projectaria_projection_summary
        summary["projected_gaze_points_sample"] = projected_points[:20]
    elif calibration is None:
        summary["projection_status"] = "missing_calibration"
        summary["projected_gaze_summary"] = None
    else:
        projection_summary = summarize_projected_gaze(projected_points, calibration)
        if projection_errors:
            projection_summary["projection_errors"] = projection_errors
        summary["projection_status"] = projection_summary["projection_status"]
        summary["projected_gaze_summary"] = projection_summary
        summary["projected_gaze_points_sample"] = projected_points[:20]
    return summary


def pair_key_for_clips(clips: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(clip["agent_dir"] for clip in sorted(clips, key=lambda c: c["agent_dir"]))


def choose_required_clips(
    group: dict[str, Any],
    users_per_case: int,
    *,
    pair_strategy: str = "first",
    pair_counts: dict[tuple[str, ...], int] | None = None,
) -> list[dict[str, Any]]:
    clips = sorted(group["clips"], key=lambda c: c["agent_dir"])
    users_per_case = max(2, users_per_case)
    if pair_strategy == "first" or len(clips) <= users_per_case:
        return clips[:users_per_case]
    if pair_strategy != "balanced":
        raise ValueError(f"unknown pair strategy: {pair_strategy}")

    counts = pair_counts or {}
    candidates = [list(combo) for combo in itertools.combinations(clips, users_per_case)]
    return min(candidates, key=lambda combo: (counts.get(pair_key_for_clips(combo), 0), pair_key_for_clips(combo)))


def build_evidence_packet(
    group: dict[str, Any],
    *,
    cache_dir: str | Path,
    output_root: str | Path,
    users_per_case: int = 2,
    frames_per_clip: int = 3,
    aria_calibration_dir: str | Path | None = None,
    download_media: bool = True,
    pair_strategy: str = "first",
    pair_counts: dict[tuple[str, ...], int] | None = None,
) -> dict[str, Any]:
    selected = choose_required_clips(
        group,
        users_per_case,
        pair_strategy=pair_strategy,
        pair_counts=pair_counts,
    )
    selected_pair_key = pair_key_for_clips(selected)
    if pair_counts is not None:
        pair_counts[selected_pair_key] = pair_counts.get(selected_pair_key, 0) + 1
    packet_id = stable_id("EGOLIFE2U", group["day"], group["time_token"], *[c["agent_id"] for c in selected])
    packet_dir = Path(output_root) / "evidence_assets" / packet_id
    clips_out = []

    for clip in selected:
        local_video = local_cache_path(cache_dir, clip["video_path"])
        local_gaze = local_cache_path(cache_dir, clip["gaze_path"])
        if download_media:
            download_file(clip["video_url"], local_video)
            download_file(clip["gaze_url"], local_gaze)
        duration = ffprobe_duration(local_video) if local_video.exists() else None
        frame_rows: list[dict[str, Any]] = []
        if local_video.exists():
            frame_dir = packet_dir / clip["agent_dir"]
            try:
                frame_rows = extract_frames(
                    local_video,
                    frame_dir,
                    frames_per_clip=frames_per_clip,
                    duration=duration,
                )
            except subprocess.CalledProcessError:
                if not download_media:
                    raise
                local_video.unlink(missing_ok=True)
                shutil.rmtree(frame_dir, ignore_errors=True)
                download_file(clip["video_url"], local_video)
                duration = ffprobe_duration(local_video) if local_video.exists() else None
                frame_rows = extract_frames(
                    local_video,
                    frame_dir,
                    frames_per_clip=frames_per_clip,
                    duration=duration,
                )
        calibration_path = find_clip_calibration(aria_calibration_dir, clip)
        gaze_summary = (
            summarize_gaze_csv(local_gaze, calibration_path=calibration_path) if local_gaze.exists() else {}
        )
        clips_out.append(
            {
                "agent_dir": clip["agent_dir"],
                "agent_id": clip["agent_id"],
                "agent_name": clip["agent_name"],
                "day": clip["day"],
                "time_token": clip["time_token"],
                "clip_clock": clip["clip_clock"],
                "duration_seconds": duration,
                "video_url": clip["video_url"],
                "gaze_url": clip["gaze_url"],
                "overlay_url": clip.get("overlay_url"),
                "local_video": str(local_video) if local_video.exists() else None,
                "local_gaze": str(local_gaze) if local_gaze.exists() else None,
                "local_calibration": str(calibration_path) if calibration_path else None,
                "frames": frame_rows,
                "gaze_summary": gaze_summary,
            }
        )

    return {
        "evidence_id": packet_id,
        "day": group["day"],
        "time_token": group["time_token"],
        "clip_clock": group.get("clip_clock"),
        "required_users": [clip["agent_name"] for clip in selected],
        "requirement": "The final question must require evidence from at least two listed users; any single listed user alone must be insufficient.",
        "pair_strategy": pair_strategy,
        "available_agents": group.get("agents", []),
        "selected_agent_dirs": list(selected_pair_key),
        "clips": clips_out,
        "source_urls": {
            "videos": [clip["video_url"] for clip in selected],
            "gazes": [clip["gaze_url"] for clip in selected],
            "overlays": [clip.get("overlay_url") for clip in selected if clip.get("overlay_url")],
        },
    }


def prepare_evidence(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    cache_dir: str | Path,
    output_root: str | Path,
    target_count: int = 20,
    users_per_case: int = 2,
    frames_per_clip: int = 3,
    aria_calibration_dir: str | Path | None = None,
    max_groups: int | None = None,
    download_media: bool = True,
    pair_strategy: str = "first",
    group_strategy: str = "chronological",
    sampling_seed: int = 0,
    min_clock_gap_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    if pair_strategy not in PAIR_STRATEGIES:
        raise ValueError(f"pair_strategy must be one of {PAIR_STRATEGIES}")
    if group_strategy not in GROUP_STRATEGIES:
        raise ValueError(f"group_strategy must be one of {GROUP_STRATEGIES}")
    manifest = read_json(manifest_path)
    groups = group_manifest_clips(manifest)
    if max_groups is not None:
        groups = groups[:max_groups]
    groups = order_evidence_groups(
        groups,
        target_count=target_count,
        group_strategy=group_strategy,
        sampling_seed=sampling_seed,
        min_clock_gap_seconds=min_clock_gap_seconds,
    )

    packets = []
    pair_counts: dict[tuple[str, ...], int] = {}
    for group in groups:
        packets.append(
            build_evidence_packet(
                group,
                cache_dir=cache_dir,
                output_root=output_root,
                users_per_case=users_per_case,
                frames_per_clip=frames_per_clip,
                aria_calibration_dir=aria_calibration_dir,
                download_media=download_media,
                pair_strategy=pair_strategy,
                pair_counts=pair_counts,
            )
        )
    write_jsonl(output_path, packets)
    return packets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare EgoLife two-user evidence packets")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=".cache/egolife_two_user_qa")
    parser.add_argument("--output-root", default="egolife_two_user_qa/outputs/pilot_20")
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--users-per-case", type=int, default=2)
    parser.add_argument("--frames-per-clip", type=int, default=3)
    parser.add_argument("--aria-calibration-dir")
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--pair-strategy", default="first", choices=PAIR_STRATEGIES)
    parser.add_argument("--group-strategy", default="chronological", choices=GROUP_STRATEGIES)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--min-clock-gap-seconds", type=float, default=0.0)
    parser.add_argument("--no-download-media", action="store_true")
    args = parser.parse_args(argv)
    packets = prepare_evidence(
        manifest_path=args.manifest,
        output_path=args.output,
        cache_dir=args.cache_dir,
        output_root=args.output_root,
        target_count=args.target_count,
        users_per_case=args.users_per_case,
        frames_per_clip=args.frames_per_clip,
        aria_calibration_dir=args.aria_calibration_dir,
        max_groups=args.max_groups,
        download_media=not args.no_download_media,
        pair_strategy=args.pair_strategy,
        group_strategy=args.group_strategy,
        sampling_seed=args.sampling_seed,
        min_clock_gap_seconds=args.min_clock_gap_seconds,
    )
    print(f"wrote {len(packets)} evidence packets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
