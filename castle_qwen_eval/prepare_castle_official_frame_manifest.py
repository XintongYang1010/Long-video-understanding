#!/usr/bin/env python3
"""Prepare an official CASTLE/EgoVis question and local-frame manifest.

The manifest uses only official question text/options. Local POC files may be
used to locate already-extracted frames, but their proposed questions/answers
are intentionally not loaded.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OFFICIAL_QA_URL = "https://castle-dataset.github.io/EgoVis2026_CVPR_Questions.json"
DEFAULT_QA_PATH = Path("EgoVis2026_CVPR_Questions.json")
DEFAULT_OUTPUT = Path("outputs/castle_frame_eval/frame_manifest.json")

DEFAULT_FRAME_ROOTS = (
    Path("castle_poc/existing6_targeted_frames_hf"),
    Path("castle_hpc/castle_event_relevant_frames"),
    Path("castle_poc/candidate_true_AB_contact_sheets/frames"),
)

STATIC_SOURCES = {"Kitchen", "Living1", "Living2", "Meeting", "Reading"}

# Heuristic links from already-extracted frame windows to official EgoVis
# questions. These are not answer labels and are kept explicit in the manifest.
HEURISTIC_FRAMESET_QIDS = {
    "EX6_HPC_Q1_PRESENTATION": ["2026_q0003"],
    "EX6_HPC_Q5_TABLETOP": ["2026_q0114"],
}
HEURISTIC_WINDOW_QIDS = {
    "DAY1_100500000": ["2026_q0003"],
    "DAY3_174500000": ["2026_q0114"],
}


@dataclass
class SourceFrames:
    source_id: str
    source_type: str
    frames: list[str]


@dataclass
class FrameSet:
    frame_set_id: str
    frame_root: str
    window_id: str
    question_ids: list[str]
    mapping_status: str
    mapping_note: str
    sources: list[SourceFrames]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def validate_official_questions(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} must contain a non-empty list")
    for item in data:
        if set(item.keys()) != {"id", "query", "answers"}:
            raise ValueError(
                f"{path} item {item!r} must contain exactly id/query/answers"
            )
        if not isinstance(item["id"], str) or not item["id"].startswith("2026_q"):
            raise ValueError(f"Invalid official question id: {item['id']!r}")
        if not isinstance(item["query"], str) or not item["query"].strip():
            raise ValueError(f"Invalid query for {item['id']}")
        answers = item["answers"]
        if not isinstance(answers, dict) or set(answers.keys()) != {"a", "b", "c", "d"}:
            raise ValueError(f"Invalid answers for {item['id']}: {answers!r}")
    return data


def ensure_official_questions(path: Path, url: str) -> list[dict[str, Any]]:
    try:
        if path.exists():
            return validate_official_questions(path)
    except Exception:
        path.unlink(missing_ok=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading_official_questions={url}", flush=True)
    urllib.request.urlretrieve(url, path)
    return validate_official_questions(path)


def source_type(source_id: str) -> str:
    return "static" if source_id in STATIC_SOURCES else "ego"


def sorted_images(path: Path) -> list[Path]:
    return sorted(
        p
        for p in path.glob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def group_source_frames(source_dirs: list[Path]) -> list[SourceFrames]:
    sources: list[SourceFrames] = []
    for source_dir in sorted(source_dirs):
        if not source_dir.is_dir():
            continue
        frames = [str(p) for p in sorted_images(source_dir)]
        if frames:
            sources.append(
                SourceFrames(
                    source_id=source_dir.name,
                    source_type=source_type(source_dir.name),
                    frames=frames,
                )
            )
    return sources


def infer_source_from_filename(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def group_flat_frames(path: Path) -> list[SourceFrames]:
    by_source: dict[str, list[str]] = {}
    for frame in sorted_images(path):
        source = infer_source_from_filename(frame)
        by_source.setdefault(source, []).append(str(frame))
    return [
        SourceFrames(source_id=source, source_type=source_type(source), frames=frames)
        for source, frames in sorted(by_source.items())
    ]


def scan_existing6(root: Path) -> list[FrameSet]:
    if not root.exists():
        return []
    frame_sets: list[FrameSet] = []
    for target_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        qids = HEURISTIC_FRAMESET_QIDS.get(target_dir.name, [])
        sources = group_source_frames([p for p in target_dir.iterdir() if p.is_dir()])
        if not sources:
            continue
        frame_sets.append(
            FrameSet(
                frame_set_id=target_dir.name,
                frame_root=str(root),
                window_id="",
                question_ids=qids,
                mapping_status="heuristic_official_question" if qids else "unmapped",
                mapping_note=(
                    "Uses official EgoVis question text/options; frame-set label only "
                    "links existing extracted frames to a likely official question."
                    if qids
                    else "No official question id could be inferred from this frame set."
                ),
                sources=sources,
            )
        )
    return frame_sets


def scan_event_relevant(root: Path) -> list[FrameSet]:
    if not root.exists():
        return []
    frame_sets: list[FrameSet] = []
    for window_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for split_dir in sorted(p for p in window_dir.iterdir() if p.is_dir()):
            sources = group_flat_frames(split_dir)
            if not sources:
                continue
            qids = HEURISTIC_WINDOW_QIDS.get(window_dir.name, [])
            frame_sets.append(
                FrameSet(
                    frame_set_id=f"{window_dir.name}_{split_dir.name}",
                    frame_root=str(root),
                    window_id=window_dir.name,
                    question_ids=qids,
                    mapping_status="heuristic_window_match" if qids else "unmapped",
                    mapping_note=(
                        "Window id overlaps an existing extracted official-question "
                        "candidate. This is a frame-location heuristic only."
                        if qids
                        else "No official question id could be inferred from this window."
                    ),
                    sources=sources,
                )
            )
    return frame_sets


def scan_contact_sheet_frames(root: Path) -> list[FrameSet]:
    if not root.exists():
        return []
    frame_sets: list[FrameSet] = []
    for window_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        sources = group_flat_frames(window_dir)
        if not sources:
            continue
        qids = HEURISTIC_WINDOW_QIDS.get(window_dir.name, [])
        frame_sets.append(
            FrameSet(
                frame_set_id=f"{window_dir.name}_contact_sheet_frames",
                frame_root=str(root),
                window_id=window_dir.name,
                question_ids=qids,
                mapping_status="heuristic_window_match" if qids else "unmapped",
                mapping_note=(
                    "Window id overlaps an existing extracted official-question "
                    "candidate. Candidate QA text/answers are not used."
                    if qids
                    else "No official question id could be inferred from this window."
                ),
                sources=sources,
            )
        )
    return frame_sets


def scan_frame_roots(roots: list[Path]) -> list[FrameSet]:
    frame_sets: list[FrameSet] = []
    for root in roots:
        root_name = root.as_posix()
        if root_name.endswith("existing6_targeted_frames_hf"):
            frame_sets.extend(scan_existing6(root))
        elif root_name.endswith("castle_event_relevant_frames"):
            frame_sets.extend(scan_event_relevant(root))
        elif root_name.endswith("candidate_true_AB_contact_sheets/frames"):
            frame_sets.extend(scan_contact_sheet_frames(root))
    return frame_sets


def summarize_frame_sets(frame_sets: list[FrameSet]) -> dict[str, Any]:
    mapped = [fs for fs in frame_sets if fs.question_ids]
    return {
        "num_frame_sets": len(frame_sets),
        "num_mapped_frame_sets": len(mapped),
        "num_sources": sum(len(fs.sources) for fs in frame_sets),
        "num_frames": sum(len(src.frames) for fs in frame_sets for src in fs.sources),
        "mapped_question_ids": sorted({qid for fs in mapped for qid in fs.question_ids}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--official-qa-path", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--official-qa-url", default=OFFICIAL_QA_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-root", action="append", default=None)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    qa_path = args.official_qa_path
    if not qa_path.is_absolute():
        qa_path = project_root / qa_path
    output = args.output
    if not output.is_absolute():
        output = project_root / output

    official_questions = ensure_official_questions(qa_path, args.official_qa_url)
    question_by_id = {q["id"]: q for q in official_questions}

    roots = [Path(r) for r in args.frame_root] if args.frame_root else list(DEFAULT_FRAME_ROOTS)
    roots = [r if r.is_absolute() else project_root / r for r in roots]
    frame_sets = scan_frame_roots(roots)
    valid_frame_sets = []
    for frame_set in frame_sets:
        frame_set.question_ids = [
            qid for qid in frame_set.question_ids if qid in question_by_id
        ]
        valid_frame_sets.append(frame_set)

    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "official_qa_url": args.official_qa_url,
        "official_qa_path": str(qa_path),
        "has_answer_key": False,
        "answer_key_note": (
            "Official CASTLE/EgoVis question JSON contains options but no correct "
            "answer key; local metrics are answer differences only."
        ),
        "questions": official_questions,
        "frame_sets": [asdict(fs) for fs in valid_frame_sets],
        "summary": summarize_frame_sets(valid_frame_sets),
    }
    write_json(output, manifest)
    print(f"manifest={output}", flush=True)
    print(json.dumps(manifest["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
