#!/usr/bin/env python3
"""Summarize official CASTLE/EgoVis frame QA predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("outputs/castle_frame_eval")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    by_unit: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_unit[(row["frame_set_id"], row["question_id"])].append(row)

    diff_rows = []
    submission: dict[str, str] = {}
    for (frame_set_id, question_id), unit_rows in sorted(by_unit.items()):
        multi_rows = [row for row in unit_rows if row["condition"] == "multi"]
        single_rows = [row for row in unit_rows if row["condition"] == "single"]
        multi_pred = multi_rows[-1]["prediction"] if multi_rows else ""
        if multi_pred and question_id not in submission:
            submission[question_id] = multi_pred.upper()

        single_counter = Counter(row["prediction"] or "INVALID" for row in single_rows)
        single_unique = sorted(single_counter)
        changed_sources = [
            row["source_ids"][0]
            for row in single_rows
            if multi_pred and row.get("prediction") != multi_pred
        ]
        diff_rows.append(
            {
                "frame_set_id": frame_set_id,
                "question_id": question_id,
                "question": unit_rows[0]["question"],
                "mapping_status": unit_rows[0].get("mapping_status", ""),
                "multi_prediction": multi_pred.upper() if multi_pred else "",
                "single_unique_predictions": ";".join(
                    f"{k.upper()}:{v}" for k, v in sorted(single_counter.items())
                ),
                "num_single_sources": len(single_rows),
                "num_changed_single_vs_multi": len(changed_sources),
                "changed_sources": ";".join(changed_sources),
                "answer_difference_rate": (
                    len(changed_sources) / len(single_rows) if single_rows else ""
                ),
                "metric_note": "No official answer key available; this is answer difference, not accuracy.",
            }
        )
    return diff_rows, submission


def write_markdown(path: Path, diff_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# CASTLE/EgoVis Qwen3-VL Frame QA Summary",
        "",
        "Official CASTLE/EgoVis questions/options were used, but no official answer key is available locally. Metrics below are answer differences only, not accuracy.",
        "",
        "| Frame Set | Question | Multi | Single Distribution | Changed Sources |",
        "|---|---|---:|---|---|",
    ]
    for row in diff_rows:
        q = row["question"].replace("|", "\\|")
        lines.append(
            "| {frame_set_id} | {question_id}: {question} | {multi_prediction} | {single_unique_predictions} | {changed_sources} |".format(
                frame_set_id=row["frame_set_id"],
                question_id=row["question_id"],
                question=q,
                multi_prediction=row["multi_prediction"],
                single_unique_predictions=row["single_unique_predictions"],
                changed_sources=row["changed_sources"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    predictions_path = args.output_root / "predictions.jsonl"
    rows = read_jsonl(predictions_path)
    diff_rows, submission = summarize(rows)

    diff_path = args.output_root / "answer_difference.csv"
    write_csv(
        diff_path,
        diff_rows,
        [
            "frame_set_id",
            "question_id",
            "question",
            "mapping_status",
            "multi_prediction",
            "single_unique_predictions",
            "num_single_sources",
            "num_changed_single_vs_multi",
            "changed_sources",
            "answer_difference_rate",
            "metric_note",
        ],
    )
    submission_path = args.output_root / "submission_partial.json"
    submission_path.write_text(
        json.dumps(submission, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path = args.output_root / "summary.md"
    write_markdown(md_path, diff_rows)

    print(f"wrote {diff_path}", flush=True)
    print(f"wrote {submission_path}", flush=True)
    print(f"wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
