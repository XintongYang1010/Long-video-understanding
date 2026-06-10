"""Generation, review, and validation orchestration for EgoLife two-user QA."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .io_utils import iter_jsonl, write_json, write_jsonl
from .prompts import build_generation_prompt, build_review_prompt
from .qwen3vl_runner import DEFAULT_MODEL_ID, make_runner
from .schema import extract_json_object, load_and_validate, validate_qa_item, write_qa_csv


def image_paths_from_packet(packet: dict[str, Any]) -> list[str]:
    paths = []
    for clip in packet.get("clips", []):
        for frame in clip.get("frames", []):
            path = frame.get("path")
            if path and Path(path).exists():
                paths.append(path)
    return paths


def packet_by_id(evidence_path: str | Path) -> dict[str, dict[str, Any]]:
    return {row["evidence_id"]: row for row in iter_jsonl(evidence_path)}


def generate_qa(
    *,
    evidence_path: str | Path,
    output_path: str | Path,
    prompts_path: str | Path | None,
    backend: str,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: str = "http://127.0.0.1:8000/v1",
    target_count: int = 20,
    max_new_tokens: int = 1024,
    max_image_pixels: int = 262144,
    dtype: str = "bfloat16",
    allow_cpu: bool = False,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    runner = make_runner(
        "dry-run" if dry_run else backend,
        model_id=model_id,
        base_url=base_url,
        max_new_tokens=max_new_tokens,
        max_image_pixels=max_image_pixels,
        dtype=dtype,
        allow_cpu=allow_cpu,
    )
    rows = []
    prompt_rows = []
    for packet in iter_jsonl(evidence_path):
        if len(rows) >= target_count:
            break
        prompt = build_generation_prompt(packet)
        images = image_paths_from_packet(packet)
        prompt_rows.append(
            {
                "evidence_id": packet["evidence_id"],
                "prompt": prompt,
                "image_paths": images,
            }
        )
        if dry_run:
            continue
        raw = runner.generate(prompt, images)
        qa = extract_json_object(raw)
        qa.setdefault("qa_id", f"QA_{len(rows) + 1:03d}_{packet['evidence_id']}")
        qa["evidence_id"] = packet["evidence_id"]
        qa["model_id"] = runner.model_id
        qa["source_urls"] = packet.get("source_urls", {})
        qa.setdefault("review", {})
        qa["review"]["generator_raw_output"] = raw
        validate_errors = validate_qa_item(qa)
        qa["review"]["schema_errors"] = validate_errors
        rows.append(qa)
    if prompts_path:
        write_jsonl(prompts_path, prompt_rows)
    if not dry_run:
        write_jsonl(output_path, rows)
    return rows


def review_qa(
    *,
    qa_path: str | Path,
    evidence_path: str | Path,
    output_path: str | Path,
    backend: str,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: str = "http://127.0.0.1:8000/v1",
    max_new_tokens: int = 768,
    max_image_pixels: int = 262144,
    dtype: str = "bfloat16",
    allow_cpu: bool = False,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    packets = packet_by_id(evidence_path)
    runner = make_runner(
        "dry-run" if dry_run else backend,
        model_id=model_id,
        base_url=base_url,
        max_new_tokens=max_new_tokens,
        max_image_pixels=max_image_pixels,
        dtype=dtype,
        allow_cpu=allow_cpu,
    )
    reviewed = []
    for qa in iter_jsonl(qa_path):
        evidence_id = qa.get("evidence_id") or str(qa.get("qa_id", "")).split("_", 2)[-1]
        packet = packets.get(evidence_id)
        if not packet:
            qa.setdefault("review", {})["review_passed"] = False
            qa["review"]["review_error"] = f"evidence packet not found: {evidence_id}"
            reviewed.append(qa)
            continue
        prompt = build_review_prompt(qa, packet)
        images = image_paths_from_packet(packet)
        if dry_run:
            qa.setdefault("review", {})["review_prompt"] = prompt
            qa["review"]["review_passed"] = False
            reviewed.append(qa)
            continue
        raw = runner.generate(prompt, images)
        review = extract_json_object(raw)
        qa["review"] = {**qa.get("review", {}), **review, "review_raw_output": raw}
        reviewed.append(qa)
    write_jsonl(output_path, reviewed)
    return reviewed


def validate_outputs(
    *,
    qa_path: str | Path,
    report_path: str | Path,
    csv_path: str | Path | None = None,
    strict_review: bool = False,
) -> int:
    count, errors = load_and_validate(qa_path, strict_review=strict_review)
    if csv_path:
        write_qa_csv(qa_path, csv_path)
    report = {
        "qa_path": str(qa_path),
        "qa_count": count,
        "strict_review": strict_review,
        "passed": not errors,
        "errors": errors,
    }
    report_path = Path(report_path)
    if report_path.suffix.lower() in {".md", ".markdown"}:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# EgoLife Two-User QA Generation Report",
            "",
            f"- QA path: `{qa_path}`",
            f"- QA count: {count}",
            f"- Strict review: {strict_review}",
            f"- Passed: {not errors}",
            "",
            "## Validation Errors",
            "",
        ]
        lines.extend([f"- {error}" for error in errors] or ["- none"])
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        write_json(report_path, report)
    return 0 if not errors else 1


def add_runner_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", default="transformers-local", choices=["transformers-local", "openai-compatible-local"])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-image-pixels", type=int, default=262144)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EgoLife two-user QA generation helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate_qa")
    gen.add_argument("--evidence", required=True)
    gen.add_argument("--output", required=True)
    gen.add_argument("--prompts-output")
    gen.add_argument("--target-count", type=int, default=20)
    add_runner_args(gen)

    rev = sub.add_parser("review_qa")
    rev.add_argument("--qa", required=True)
    rev.add_argument("--evidence", required=True)
    rev.add_argument("--output", required=True)
    add_runner_args(rev)

    val = sub.add_parser("validate_outputs")
    val.add_argument("--qa", required=True)
    val.add_argument("--report", required=True)
    val.add_argument("--csv-output")
    val.add_argument("--strict-review", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "generate_qa":
        rows = generate_qa(
            evidence_path=args.evidence,
            output_path=args.output,
            prompts_path=args.prompts_output,
            backend=args.backend,
            model_id=args.model_id,
            base_url=args.base_url,
            target_count=args.target_count,
            max_new_tokens=args.max_new_tokens,
            max_image_pixels=args.max_image_pixels,
            dtype=args.dtype,
            allow_cpu=args.allow_cpu,
            dry_run=args.dry_run,
        )
        print(f"generated {len(rows)} QA rows")
        return 0
    if args.command == "review_qa":
        rows = review_qa(
            qa_path=args.qa,
            evidence_path=args.evidence,
            output_path=args.output,
            backend=args.backend,
            model_id=args.model_id,
            base_url=args.base_url,
            max_new_tokens=args.max_new_tokens,
            max_image_pixels=args.max_image_pixels,
            dtype=args.dtype,
            allow_cpu=args.allow_cpu,
            dry_run=args.dry_run,
        )
        print(f"reviewed {len(rows)} QA rows")
        return 0
    if args.command == "validate_outputs":
        return validate_outputs(
            qa_path=args.qa,
            report_path=args.report,
            csv_path=args.csv_output,
            strict_review=args.strict_review,
        )
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
