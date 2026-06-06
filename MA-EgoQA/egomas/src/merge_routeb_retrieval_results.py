"""Merge Route B retrieval result chunks into one JSON and summary CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from egomas.src.evaluate_day1_qwen3vl_routeb_retrieval import save_outputs, summarize
from egomas.utils.io import load_json


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("question_index"),
        row.get("top_k"),
        row.get("eval_mode"),
        tuple(row.get("agents", [])),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", action="append", required=True, help="Glob for chunk JSON files; can be repeated.")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--strict", action="store_true", help="Fail if an input file has no results.")
    args = parser.parse_args()

    paths: list[Path] = []
    for pattern in args.input_glob:
        paths.extend(sorted(Path().glob(pattern) if not pattern.startswith("/") else Path("/").glob(pattern[1:])))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("No input files matched")

    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    inputs: list[dict[str, Any]] = []
    for path in paths:
        payload = load_json(str(path))
        rows = payload.get("results", []) if isinstance(payload, dict) else []
        if args.strict and not rows:
            raise SystemExit(f"Input has no results: {path}")
        inputs.append({"path": str(path), "results": len(rows)})
        for row in rows:
            merged[row_key(row)] = row

    results = sorted(
        merged.values(),
        key=lambda row: (
            row.get("question_index", -1),
            row.get("top_k", -1),
            row.get("eval_mode", ""),
            tuple(row.get("agents", [])),
        ),
    )
    payload = {
        "inputs": inputs,
        "summary": summarize(results),
        "results": results,
    }
    save_outputs(payload, args.output_path)
    print(json.dumps({"output_path": args.output_path, "inputs": inputs, "merged_results": len(results)}, indent=2))


if __name__ == "__main__":
    main()
