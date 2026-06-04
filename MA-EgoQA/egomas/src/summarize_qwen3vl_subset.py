"""Summarize Qwen3-VL subset MA-EgoQA condition outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("outputs/qwen3vl_subset")
CONDITION_ORDER = (
    "Jack",
    "Alice",
    "Katrina",
    "Jack_Alice",
    "Jack_Katrina",
    "Alice_Katrina",
    "Jack_Alice_Katrina",
    "Lucia",
    "Tasha",
    "Shure",
    "Lucia_Tasha",
    "Lucia_Shure",
    "Tasha_Shure",
    "Lucia_Tasha_Shure",
)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def condition_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    condition = row.get("condition", "")
    try:
        return CONDITION_ORDER.index(condition), condition
    except ValueError:
        return len(CONDITION_ORDER), condition


def collect_summaries(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(output_root.glob("*/summary.json")):
        row = load_json(summary_path)
        row["summary_path"] = str(summary_path)
        row["predictions_path"] = str(summary_path.parent / "predictions.jsonl")
        agents = row.get("agents", [])
        row["agent_group"] = "+".join(agents) if isinstance(agents, list) else str(agents)
        row["agent_count"] = len(agents) if isinstance(agents, list) else 0
        rows.append(row)
    rows.sort(key=condition_sort_key)
    return rows


def write_csv(output_root: Path, rows: list[dict[str, Any]]) -> Path:
    path = output_root / "summary.csv"
    fieldnames = [
        "condition",
        "agent_group",
        "agent_count",
        "num_items",
        "correct",
        "accuracy",
        "invalid_predictions",
        "elapsed_seconds",
        "predictions_path",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def write_markdown(output_root: Path, rows: list[dict[str, Any]]) -> Path:
    path = output_root / "summary.md"
    lines = [
        "# Qwen3-VL MA-EgoQA Subset Summary",
        "",
        "| Condition | Agents | Items | Correct | Accuracy | Invalid | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        accuracy = float(row.get("accuracy", 0.0))
        elapsed = float(row.get("elapsed_seconds", 0.0))
        lines.append(
            "| {condition} | {agent_group} | {num_items} | {correct} | "
            "{accuracy:.2%} | {invalid_predictions} | {elapsed:.1f} |".format(
                condition=row.get("condition", ""),
                agent_group=row.get("agent_group", ""),
                num_items=row.get("num_items", ""),
                correct=row.get("correct", ""),
                accuracy=accuracy,
                invalid_predictions=row.get("invalid_predictions", ""),
                elapsed=elapsed,
            )
        )

    grouped_rows: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        agent_count = int(row.get("agent_count") or 0)
        grouped_rows.setdefault(agent_count, []).append(row)
    labels = {1: "Single-agent", 2: "Pair-agent", 3: "Three-agent"}
    for agent_count in sorted(grouped_rows):
        group = grouped_rows[agent_count]
        if not group:
            continue
        avg = sum(float(row.get("accuracy", 0.0)) for row in group) / len(group)
        label = labels.get(agent_count, f"{agent_count}-agent")
        lines.extend(["", f"{label} mean accuracy: {avg:.2%}"])
        if len(group) > 1:
            best = max(group, key=lambda row: float(row.get("accuracy", 0.0)))
            lines.append(
                "Best {label}: {condition} ({accuracy:.2%})".format(
                    label=label.lower(),
                    condition=best.get("condition", ""),
                    accuracy=float(best.get("accuracy", 0.0)),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = collect_summaries(args.output_root)
    if not rows:
        raise SystemExit(f"No summary.json files found under {args.output_root}")
    csv_path = write_csv(args.output_root, rows)
    md_path = write_markdown(args.output_root, rows)
    print(f"wrote {csv_path}", flush=True)
    print(f"wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
