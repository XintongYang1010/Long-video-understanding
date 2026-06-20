#!/usr/bin/env python3
"""Prepare FR/AD/GC/GM video-condition artifacts for EgoEverything-style evals.

The arXiv source describes AD as resizing frames to 1/3 H x 1/3 W and GC as a
gaze-centered crop with the same per-frame token budget. The current camera
ready text phrases this as roughly 10% of the original frame/resolution.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


MODES = ("FR", "AD", "GC", "GM")


def codec_safe_even_dimension(value: float, max_value: int) -> int:
    """Return the nearest even dimension that common MP4 codecs preserve."""
    size = max(2, int(round(value)))
    if size > max_value:
        size = max_value
    if size % 2 == 0:
        return size

    lower = max(2, size - 1)
    upper = size + 1
    if upper <= max_value and abs(upper - value) <= abs(lower - value):
        return upper
    return lower if lower % 2 == 0 else max(2, lower - 1)


def clamp_crop(cx: int, cy: int, size: int, width: int, height: int) -> tuple[int, int, int, int]:
    half = size // 2
    x1 = max(0, min(width - size, cx - half))
    y1 = max(0, min(height - size, cy - half))
    return x1, y1, x1 + size, y1 + size


def open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def process_video(
    video_path: Path,
    tracking_path: Path,
    output_dir: Path,
    crop_ratio: float,
    ad_scale: float,
    max_frames: int | None,
) -> dict:
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not tracking_path.exists():
        raise FileNotFoundError(tracking_path)

    tracking = pd.read_csv(tracking_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    process_frames = min(frame_count, len(tracking), max_frames or frame_count)

    crop_size = codec_safe_even_dimension(min(width, height) * crop_ratio, min(width, height))
    ad_width = codec_safe_even_dimension(width * ad_scale, width)
    ad_height = codec_safe_even_dimension(height * ad_scale, height)

    outputs = {
        "FR": output_dir / "FR" / video_path.name,
        "AD": output_dir / "AD" / video_path.name,
        "GC": output_dir / "GC" / video_path.name,
        "GM": output_dir / "GM" / video_path.name,
    }

    writers = {
        "AD": open_writer(outputs["AD"], fps, ad_width, ad_height),
        "GC": open_writer(outputs["GC"], fps, crop_size, crop_size),
        "GM": open_writer(outputs["GM"], fps, width, height),
    }

    output_fr = outputs["FR"]
    output_fr.parent.mkdir(parents=True, exist_ok=True)
    if output_fr.exists() or output_fr.is_symlink():
        output_fr.unlink()
    try:
        output_fr.symlink_to(video_path)
        fr_kind = "symlink"
    except OSError:
        shutil.copy2(video_path, output_fr)
        fr_kind = "copy"

    gaze_missing = 0
    processed = 0
    for frame_idx in range(process_frames):
        ok, frame = cap.read()
        if not ok:
            break
        row = tracking.iloc[frame_idx]

        writers["AD"].write(cv2.resize(frame, (ad_width, ad_height), interpolation=cv2.INTER_AREA))

        if pd.notna(row.get("gaze_x")) and pd.notna(row.get("gaze_y")):
            cx = int(row["gaze_x"])
            cy = int(row["gaze_y"])
        else:
            cx = width // 2
            cy = height // 2
            gaze_missing += 1

        x1, y1, x2, y2 = clamp_crop(cx, cy, crop_size, width, height)
        crop = frame[y1:y2, x1:x2]
        if crop.shape[1] != crop_size or crop.shape[0] != crop_size:
            crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
        writers["GC"].write(crop)

        masked = frame.copy()
        masked[y1:y2, x1:x2] = 0
        writers["GM"].write(masked)
        processed += 1

    cap.release()
    for writer in writers.values():
        writer.release()

    artifacts = {}
    for mode, path in outputs.items():
        probe = cv2.VideoCapture(str(path))
        artifacts[mode] = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "frames": int(probe.get(cv2.CAP_PROP_FRAME_COUNT)) if probe.isOpened() else 0,
            "fps": probe.get(cv2.CAP_PROP_FPS) if probe.isOpened() else 0,
            "width": int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)) if probe.isOpened() else 0,
            "height": int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)) if probe.isOpened() else 0,
        }
        probe.release()

    return {
        "source_video": str(video_path),
        "source_tracking": str(tracking_path),
        "fr_kind": fr_kind,
        "source_width": width,
        "source_height": height,
        "source_fps": fps,
        "source_frames": frame_count,
        "processed_frames": processed,
        "gaze_missing_frames": gaze_missing,
        "crop_ratio": crop_ratio,
        "crop_size": crop_size,
        "ad_scale": ad_scale,
        "ad_width": ad_width,
        "ad_height": ad_height,
        "parameter_note": (
            "Defaults follow arXiv source comments: AD uses 1/3 H x 1/3 W and "
            "GC uses a gaze-centered square crop with the same side ratio, i.e. "
            "roughly 10% of original frame area/resolution. Dimensions are rounded "
            "to the nearest codec-safe even integer."
        ),
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Section 4 FR/AD/GC/GM video conditions")
    parser.add_argument("--sequence-id", default="loc5_script4_seq6_rec1")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--crop-ratio", type=float, default=1.0 / 3.0)
    parser.add_argument("--ad-scale", type=float, default=1.0 / 3.0)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    if not (0 < args.crop_ratio <= 1):
        raise SystemExit("--crop-ratio must be in (0, 1]")
    if not (0 < args.ad_scale <= 1):
        raise SystemExit("--ad-scale must be in (0, 1]")

    dataset_path = Path(args.dataset_path)
    seq_dir = dataset_path / args.sequence_id
    video_path = seq_dir / f"{args.sequence_id}.mp4"
    tracking_path = seq_dir / f"{args.sequence_id}_tracking.csv"
    output_root = Path(args.output_root) / args.sequence_id

    result = process_video(
        video_path=video_path,
        tracking_path=tracking_path,
        output_dir=output_root,
        crop_ratio=args.crop_ratio,
        ad_scale=args.ad_scale,
        max_frames=args.max_frames,
    )

    manifest_path = output_root / "section4_conditions_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
