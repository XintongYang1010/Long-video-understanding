"""Unified CLI for the EgoLife two-user QA pilot."""

from __future__ import annotations

import argparse

from .evidence import prepare_evidence
from .manifest import build_manifest
from .qa_pipeline import add_runner_args, generate_qa, review_qa, validate_outputs


def _csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="egolife-two-user-qa")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("build_manifest", help="Build EgoLife video/gaze manifest")
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--agents")
    manifest.add_argument("--days")
    manifest.add_argument("--revision", default="main")
    manifest.add_argument("--max-per-agent-day", type=int)
    manifest.add_argument("--no-overlays", action="store_true")

    evidence = sub.add_parser("prepare_evidence", help="Prepare multi-user evidence packets")
    evidence.add_argument("--manifest", required=True)
    evidence.add_argument("--output", required=True)
    evidence.add_argument("--cache-dir", default=".cache/egolife_two_user_qa")
    evidence.add_argument("--output-root", default="egolife_two_user_qa/outputs/pilot_20")
    evidence.add_argument("--target-count", type=int, default=20)
    evidence.add_argument("--users-per-case", type=int, default=2)
    evidence.add_argument("--frames-per-clip", type=int, default=3)
    evidence.add_argument("--max-groups", type=int)
    evidence.add_argument("--no-download-media", action="store_true")

    gen = sub.add_parser("generate_qa", help="Generate QA from evidence packets")
    gen.add_argument("--evidence", required=True)
    gen.add_argument("--output", required=True)
    gen.add_argument("--prompts-output")
    gen.add_argument("--target-count", type=int, default=20)
    add_runner_args(gen)

    rev = sub.add_parser("review_qa", help="Review generated QA")
    rev.add_argument("--qa", required=True)
    rev.add_argument("--evidence", required=True)
    rev.add_argument("--output", required=True)
    add_runner_args(rev)

    val = sub.add_parser("validate_outputs", help="Validate QA JSONL and write report/CSV")
    val.add_argument("--qa", required=True)
    val.add_argument("--report", required=True)
    val.add_argument("--csv-output")
    val.add_argument("--strict-review", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "build_manifest":
        result = build_manifest(
            output_path=args.output,
            agents=_csv(args.agents),
            days=_csv(args.days),
            revision=args.revision,
            max_per_agent_day=args.max_per_agent_day,
            include_overlays=not args.no_overlays,
        )
        print(f"wrote {len(result['clips'])} aligned clips to {args.output}")
        return 0
    if args.command == "prepare_evidence":
        rows = prepare_evidence(
            manifest_path=args.manifest,
            output_path=args.output,
            cache_dir=args.cache_dir,
            output_root=args.output_root,
            target_count=args.target_count,
            users_per_case=args.users_per_case,
            frames_per_clip=args.frames_per_clip,
            max_groups=args.max_groups,
            download_media=not args.no_download_media,
        )
        print(f"wrote {len(rows)} evidence packets to {args.output}")
        return 0
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
