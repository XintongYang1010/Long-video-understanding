#!/usr/bin/env python3
"""Check whether EgoEverything Section 4 evaluation artifacts are present."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


TASKS = ("FR", "AD", "GC", "GM", "VMP", "AMEGO")
KEY_ENVS = ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY")


def read_text_prefix(path: Path, limit: int = 256_000) -> str:
    try:
        return path.read_bytes()[:limit].decode("utf-8", "ignore")
    except Exception:
        return ""


def score_json_like(path: Path) -> dict[str, Any]:
    text = read_text_prefix(path)
    lowered = text.lower()
    score = 0
    reasons: list[str] = []

    for token in ("question", "options", "answer"):
        if token in lowered:
            score += 1
            reasons.append(f"contains {token}")
    if "video" in lowered:
        score += 1
        reasons.append("contains video")
    if "mcq" in lowered or "multiple-choice" in lowered or "multiple_choice" in lowered:
        score += 1
        reasons.append("contains mcq marker")

    return {"path": str(path), "score": score, "reasons": reasons}


def score_csv(path: Path) -> dict[str, Any]:
    score = 0
    reasons: list[str] = []
    try:
        with path.open(newline="", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, [])
    except Exception:
        header = []
    lowered = {h.strip().lower() for h in header}
    for token in ("question", "answer", "video"):
        if token in lowered:
            score += 1
            reasons.append(f"header {token}")
    if any(h in lowered for h in ("options", "choices", "choice_a", "a")):
        score += 1
        reasons.append("header options/choices")
    return {"path": str(path), "score": score, "reasons": reasons}


def find_benchmark_candidates(root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if not root.exists():
        return candidates
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".json", ".jsonl"}:
            item = score_json_like(path)
        elif path.suffix.lower() == ".csv":
            item = score_csv(path)
        else:
            continue
        name = path.name.lower()
        if any(word in name for word in ("mcq", "question", "benchmark", "eval", "ego")):
            item["score"] += 1
            item["reasons"].append("filename benchmark-like")
        if item["score"] >= 3:
            candidates.append(item)
    return sorted(candidates, key=lambda x: (-x["score"], x["path"]))


def find_answer_key_candidates(root: Path) -> list[str]:
    if not root.exists():
        return []
    hits: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv", ".tsv", ".txt"}:
            continue
        name = path.name.lower()
        if any(word in name for word in ("answer", "key", "label", "gold", "ground_truth", "gt")):
            hits.append(str(path))
    return sorted(hits)


def find_eval_runner_candidates(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    hits: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".sh", ".ipynb", ".md", ".yaml", ".yml"}:
            continue
        text = read_text_prefix(path)
        upper = text.upper()
        present = [task for task in TASKS if task in upper]
        if len(present) >= 3 and ("ACCURACY" in upper or "EVAL" in upper or "PRED" in upper):
            hits.append({"path": str(path), "methods_seen": present})
    return hits


def count_processed_videos(dataset_path: Path) -> list[str]:
    if not dataset_path.exists():
        return []
    videos = []
    for seq_dir in dataset_path.iterdir():
        if seq_dir.is_dir():
            mp4 = seq_dir / f"{seq_dir.name}.mp4"
            tracking = seq_dir / f"{seq_dir.name}_tracking.csv"
            if mp4.exists() and tracking.exists():
                videos.append(str(seq_dir))
    return sorted(videos)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check EgoEverything Section 4 artifact readiness")
    parser.add_argument("--artifact-root", default=os.environ.get("EGOEVERYTHING_SECTION4_ROOT", "section4_artifacts"))
    parser.add_argument("--dataset-path", default=os.environ.get("DATASET_PATH", "Data"))
    parser.add_argument("--repo", default=os.environ.get("EGOEVERYTHING_REPO", "."))
    parser.add_argument("--report", default=os.environ.get("SECTION4_REPORT", "section4_artifact_check.json"))
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root)
    dataset_path = Path(args.dataset_path)
    repo = Path(args.repo)

    benchmark_candidates = find_benchmark_candidates(artifact_root)
    answer_key_candidates = find_answer_key_candidates(artifact_root)
    eval_runner_candidates = find_eval_runner_candidates(artifact_root) + find_eval_runner_candidates(repo / "Evaluation")
    processed_videos = count_processed_videos(dataset_path)
    key_presence = {name: bool(os.environ.get(name)) for name in KEY_ENVS}

    missing = []
    if not benchmark_candidates:
        missing.append("official Section 4 MCQ benchmark manifest")
    if not answer_key_candidates:
        missing.append("official answer key / gold labels")
    if not eval_runner_candidates:
        missing.append("Section 4 VLM evaluation runner/prompts for FR, AD, GC, GM, VMP, AMEGO")
    if not processed_videos:
        missing.append("processed video+tracking data")
    if not any(key_presence.values()):
        missing.append("model/API credential environment variable")

    report = {
        "artifact_root": str(artifact_root),
        "dataset_path": str(dataset_path),
        "repo": str(repo),
        "ready": not missing,
        "missing": missing,
        "benchmark_candidates": benchmark_candidates[:20],
        "answer_key_candidates": answer_key_candidates[:20],
        "eval_runner_candidates": eval_runner_candidates[:20],
        "processed_video_dirs": processed_videos[:20],
        "api_env_present": key_presence,
        "required_methods": list(TASKS),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
