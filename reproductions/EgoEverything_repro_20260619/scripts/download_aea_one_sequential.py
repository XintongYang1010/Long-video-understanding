#!/usr/bin/env python3
"""Download and process a small AEA subset without the repo's queue race."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

repo_root = Path(os.environ.get("EGOEVERYTHING_REPO", Path.cwd()))
sys.path.insert(0, str(repo_root))

from aria_downloader_processor import process_worker


def sequence_names(manifest_path: Path, limit: int) -> list[str]:
    data = json.loads(manifest_path.read_text())
    names = list(data["sequences"].keys())
    return names[:limit]


def run_downloader(manifest_path: Path, output_dir: Path, sequence: str) -> None:
    status = output_dir / sequence / ".download_status.json"
    if status.exists():
        status.unlink()

    for data_type in ("0", "4"):
        cmd = [
            "aria_dataset_downloader",
            "-c",
            str(manifest_path),
            "-o",
            str(output_dir),
            "-l",
            sequence,
            "--data_types",
            data_type,
        ]
        print("+", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


def process_sequence(output_dir: Path, sequence: str, total: int) -> None:
    download_queue = mp.Queue()
    process_queue = mp.Queue()
    process_queue.put(sequence)
    counter = mp.Value("i", 0)

    process_worker(download_queue, process_queue, counter, total, str(output_dir))

    seq_dir = output_dir / sequence
    mp4 = seq_dir / f"{sequence}.mp4"
    tracking = seq_dir / f"{sequence}_tracking.csv"
    if not mp4.exists() or not tracking.exists():
        raise SystemExit(
            f"Processing did not produce expected files for {sequence}: "
            f"{mp4} and {tracking}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    names = sequence_names(args.json, args.limit)
    print(f"Processing {len(names)} AEA sequence(s): {', '.join(names)}", flush=True)

    for idx, sequence in enumerate(names, start=1):
        seq_dir = args.output / sequence
        mp4 = seq_dir / f"{sequence}.mp4"
        tracking = seq_dir / f"{sequence}_tracking.csv"
        if mp4.exists() and tracking.exists():
            print(f"Skipping {sequence}: processed files already exist", flush=True)
            continue
        print(f"[{idx}/{len(names)}] Downloading {sequence}", flush=True)
        run_downloader(args.json, args.output, sequence)
        print(f"[{idx}/{len(names)}] Processing {sequence}", flush=True)
        process_sequence(args.output, sequence, len(names))
        print(f"[{idx}/{len(names)}] Done {sequence}", flush=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
