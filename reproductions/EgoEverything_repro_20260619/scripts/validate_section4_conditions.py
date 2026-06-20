#!/usr/bin/env python3
"""Validate generated FR/AD/GC/GM condition videos against their manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2


MODES = ("FR", "AD", "GC", "GM")


def probe_video(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    opened = cap.isOpened()
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
    mid_sha16 = ""
    if opened and frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames // 2)
        ok, frame = cap.read()
        if ok:
            mid_sha16 = hashlib.sha256(frame.tobytes()).hexdigest()[:16]
    cap.release()
    return {
        "path": str(path),
        "exists": path.exists(),
        "opened": opened,
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "mid_sha16": mid_sha16,
    }


def nearly_equal(left: float, right: float, tolerance: float = 0.01) -> bool:
    return abs(left - right) <= tolerance


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate EgoEverything Section 4 condition videos")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    probes: dict[str, Any] = {}

    for mode in MODES:
        artifact = manifest.get("artifacts", {}).get(mode)
        if not artifact:
            errors.append(f"missing manifest artifact for {mode}")
            continue
        probe = probe_video(Path(artifact["path"]))
        probes[mode] = probe
        if not probe["exists"]:
            errors.append(f"{mode} video does not exist")
        if not probe["opened"]:
            errors.append(f"{mode} video cannot be opened by OpenCV")
        for field in ("frames", "width", "height"):
            if int(artifact.get(field, -1)) != int(probe[field]):
                errors.append(f"{mode} {field} manifest={artifact.get(field)} probe={probe[field]}")
        if not nearly_equal(float(artifact.get("fps", 0.0)), float(probe["fps"])):
            errors.append(f"{mode} fps manifest={artifact.get('fps')} probe={probe['fps']}")

    expected_frames = int(manifest.get("processed_frames", 0))
    for mode, probe in probes.items():
        if expected_frames and probe["frames"] != expected_frames:
            errors.append(f"{mode} frames={probe['frames']} expected processed_frames={expected_frames}")

    if manifest.get("gaze_missing_frames") != 0:
        errors.append(f"gaze_missing_frames={manifest.get('gaze_missing_frames')}, expected 0")

    report = {
        "manifest": str(manifest_path),
        "valid": not errors,
        "errors": errors,
        "source_video": manifest.get("source_video"),
        "source_tracking": manifest.get("source_tracking"),
        "processed_frames": manifest.get("processed_frames"),
        "gaze_missing_frames": manifest.get("gaze_missing_frames"),
        "crop_ratio": manifest.get("crop_ratio"),
        "crop_size": manifest.get("crop_size"),
        "ad_scale": manifest.get("ad_scale"),
        "ad_width": manifest.get("ad_width"),
        "ad_height": manifest.get("ad_height"),
        "probes": probes,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
