#!/usr/bin/env python3
"""Replay accepted video-first generation prompts with a Gemini backend."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from egolife_two_user_qa.io_utils import iter_jsonl, write_json, write_jsonl
from egolife_two_user_qa.prompts import build_judger_prompt
from egolife_two_user_qa.qwen3vl_runner import make_runner
from egolife_two_user_qa.schema import (
    extract_json_object,
    validate_qa_item,
    write_human_review_sheet,
    write_qa_csv,
)
from egolife_two_user_qa.video_qa_loop import (
    build_review_from_gates,
    complete_generator_metadata,
    human_audit_packet,
    judge_gate,
    qa_for_judger_prompt,
    skipped_answerability_review,
    video_evidence_for_packet,
)


PRO_25_PRICING = {
    "input_le_200k_per_million": 1.25,
    "input_gt_200k_per_million": 2.50,
    "output_le_200k_per_million": 10.00,
    "output_gt_200k_per_million": 15.00,
}

FLASH_25_PRICING = {
    "input_le_200k_per_million": 0.30,
    "input_gt_200k_per_million": 0.30,
    "output_le_200k_per_million": 2.50,
    "output_gt_200k_per_million": 2.50,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the accepted generation prompts from a previous video-first run, "
            "using Gemini as the generator and optionally as the judger."
        )
    )
    parser.add_argument("--source-qa", required=True, help="Previous accepted qa_mcq.jsonl")
    parser.add_argument("--source-prompts", required=True, help="Previous video_first_prompts.jsonl")
    parser.add_argument("--evidence", required=True, help="Previous evidence_manifest.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for replay outputs")
    parser.add_argument("--model-id", default="gemini-2.5-pro")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--limit", type=int, default=0, help="Optional first-N limit; 0 means all")
    parser.add_argument("--skip-judge", action="store_true", help="Only generate; do not run the judger")
    parser.add_argument("--plan-only", action="store_true", help="Validate replay mapping without API calls")
    return parser.parse_args()


def prompt_digest(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def index_prompts(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        try:
            attempt = int(row.get("attempt"))
        except (TypeError, ValueError):
            continue
        key = (str(row.get("stage")), str(row.get("evidence_id")), attempt)
        if key in indexed:
            raise ValueError(f"Duplicate prompt row for {key}")
        indexed[key] = row
    return indexed


def usage_from_runner(runner: Any) -> dict[str, Any]:
    usage = getattr(runner, "last_usage_metadata", {}) or {}
    return copy.deepcopy(usage)


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def pricing_for_model(model_id: str) -> tuple[dict[str, float] | None, str]:
    model = model_id.lower()
    if "gemini-2.5-pro" in model:
        return PRO_25_PRICING, "Gemini 2.5 Pro standard paid pricing"
    if "gemini-2.5-flash" in model:
        return FLASH_25_PRICING, "Gemini 2.5 Flash standard paid pricing"
    return None, "No baked-in price table for this model; usage tokens are still reported"


def estimate_usage_cost(calls: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    pricing, pricing_note = pricing_for_model(model_id)
    totals = {
        "prompt_token_count": 0,
        "candidates_token_count": 0,
        "thoughts_token_count": 0,
        "estimated_billable_output_token_count": 0,
        "total_token_count": 0,
        "estimated_cost_usd": 0.0,
    }
    per_call = []
    for call in calls:
        usage = call.get("usage_metadata") if isinstance(call.get("usage_metadata"), dict) else {}
        prompt_tokens = as_int(usage.get("promptTokenCount"))
        candidates_tokens = as_int(usage.get("candidatesTokenCount"))
        thoughts_tokens = as_int(usage.get("thoughtsTokenCount"))
        output_tokens = candidates_tokens + thoughts_tokens
        total_tokens = as_int(usage.get("totalTokenCount"))
        cost = None
        if pricing:
            input_rate = (
                pricing["input_gt_200k_per_million"]
                if prompt_tokens > 200_000
                else pricing["input_le_200k_per_million"]
            )
            output_rate = (
                pricing["output_gt_200k_per_million"]
                if prompt_tokens > 200_000
                else pricing["output_le_200k_per_million"]
            )
            cost = (prompt_tokens / 1_000_000 * input_rate) + (
                output_tokens / 1_000_000 * output_rate
            )
            totals["estimated_cost_usd"] += cost
        totals["prompt_token_count"] += prompt_tokens
        totals["candidates_token_count"] += candidates_tokens
        totals["thoughts_token_count"] += thoughts_tokens
        totals["estimated_billable_output_token_count"] += output_tokens
        totals["total_token_count"] += total_tokens
        per_call.append(
            {
                "stage": call.get("stage"),
                "index": call.get("index"),
                "qa_id": call.get("qa_id"),
                "evidence_id": call.get("evidence_id"),
                "prompt_token_count": prompt_tokens,
                "candidates_token_count": candidates_tokens,
                "thoughts_token_count": thoughts_tokens,
                "estimated_billable_output_token_count": output_tokens,
                "total_token_count": total_tokens,
                "estimated_cost_usd": cost,
            }
        )
    return {
        "model_id": model_id,
        "pricing_note": pricing_note,
        "pricing_assumption": pricing,
        "important_caveat": (
            "This is an estimate from Gemini usageMetadata. If thoughtsTokenCount is present, "
            "the script treats it as billable output in addition to candidatesTokenCount."
        ),
        "totals": totals,
        "per_call": per_call,
    }


def final_reason(review: dict[str, Any]) -> str:
    final = review.get("final_decision") if isinstance(review.get("final_decision"), dict) else {}
    return str(final.get("reason") or "")


def md_cell(value: Any, *, limit: int = 220) -> str:
    text = str(value or "").replace("\n", " ").replace("|", "\\|")
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def write_report(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    source_rows: list[dict[str, Any]],
    accepted_rows: list[dict[str, Any]],
    intermediate_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    cost: dict[str, Any],
) -> None:
    parsed_count = sum(1 for row in raw_rows if row.get("parse_error") is None)
    schema_pass_count = sum(
        1
        for row in intermediate_rows
        if (row.get("review") or {}).get("schema_validation", {}).get("passed") is True
    )
    judge_pass_count = sum(
        1
        for row in intermediate_rows
        if (row.get("review") or {}).get("judger", {}).get("gate", {}).get("passed") is True
    )
    totals = cost.get("totals", {})
    lines = [
        "# Gemini Replay Generation Report",
        "",
        f"- Model: `{args.model_id}`",
        f"- Source QA: `{args.source_qa}`",
        f"- Source prompts: `{args.source_prompts}`",
        f"- Evidence manifest: `{args.evidence}`",
        f"- Output dir: `{output_dir}`",
        f"- Source accepted rows replayed: {len(source_rows)}",
        f"- Parsed Gemini generations: {parsed_count}",
        f"- Schema-pass rows: {schema_pass_count}",
        f"- Judger-pass rows: {judge_pass_count}",
        f"- Rows written to `qa_mcq.jsonl`: {len(accepted_rows)}",
        f"- Rejected or parse-failed rows: {len(rejected_rows)}",
        f"- Estimated cost USD: {totals.get('estimated_cost_usd', 0.0):.6f}",
        f"- Prompt tokens: {totals.get('prompt_token_count', 0)}",
        f"- Output tokens estimate: {totals.get('estimated_billable_output_token_count', 0)}",
        "",
        "## Files",
        "",
        "- `qa_mcq.jsonl`: rows that passed schema and judger, still pending human review",
        "- `qa_mcq.intermediate.jsonl`: every parsed Gemini QA with review status",
        "- `qa_mcq.rejected.jsonl`: parse failures, schema failures, and judger failures",
        "- `video_first_prompts.jsonl`: replay generation prompts and new judge prompts",
        "- `gemini_replay_raw.jsonl`: raw Gemini outputs plus source Qwen comparison snippets",
        "- `cost_estimate.json`: token usage and estimated API cost",
        "",
        "## Question Comparison",
        "",
        "| # | Source Qwen QA | Gemini QA | Review status | Reason |",
        "|---|---|---|---|---|",
    ]
    by_index = {row.get("source_index"): row for row in intermediate_rows}
    raw_by_index = {row.get("source_index"): row for row in raw_rows}
    for source in source_rows:
        index = source.get("_source_index")
        generated = by_index.get(index)
        raw = raw_by_index.get(index, {})
        qwen_question = source.get("question", "")
        gemini_question = generated.get("question") if generated else raw.get("parse_error", "")
        review = generated.get("review", {}) if isinstance(generated, dict) else {}
        status = review.get("status", "parse_failed" if raw.get("parse_error") else "")
        lines.append(
            "| "
            + " | ".join(
                [
                    md_cell(index, limit=20),
                    md_cell(qwen_question),
                    md_cell(gemini_question),
                    md_cell(status, limit=80),
                    md_cell(final_reason(review), limit=180),
                ]
            )
            + " |"
        )
    output_dir.joinpath("generation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    source_rows: list[dict[str, Any]],
    accepted_rows: list[dict[str, Any]],
    intermediate_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    prompt_rows: list[dict[str, Any]],
    usage_calls: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "qa_mcq.jsonl", accepted_rows)
    write_jsonl(output_dir / "qa_mcq.intermediate.jsonl", intermediate_rows)
    write_jsonl(output_dir / "qa_mcq.rejected.jsonl", rejected_rows)
    write_jsonl(output_dir / "video_first_prompts.jsonl", prompt_rows)
    write_jsonl(output_dir / "gemini_replay_raw.jsonl", raw_rows)
    cost = estimate_usage_cost(usage_calls, args.model_id)
    write_json(output_dir / "cost_estimate.json", cost)
    write_report(
        output_dir=output_dir,
        args=args,
        source_rows=source_rows,
        accepted_rows=accepted_rows,
        intermediate_rows=intermediate_rows,
        rejected_rows=rejected_rows,
        raw_rows=raw_rows,
        cost=cost,
    )
    write_qa_csv(output_dir / "qa_mcq.jsonl", output_dir / "qa_mcq.csv")
    write_human_review_sheet(output_dir / "qa_mcq.jsonl", output_dir / "human_review_sheet.md")


def source_with_indices(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected = rows[:limit] if limit > 0 else rows
    copied = []
    for index, row in enumerate(selected, 1):
        item = copy.deepcopy(row)
        item["_source_index"] = index
        copied.append(item)
    return copied


def build_plan(
    source_rows: list[dict[str, Any]],
    prompt_index: dict[tuple[str, str, int], dict[str, Any]],
    evidence_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    plan = []
    for row in source_rows:
        evidence_id = str(row.get("evidence_id"))
        attempt = int(row.get("attempt_count") or 1)
        prompt_row = prompt_index.get(("generation", evidence_id, attempt))
        packet = evidence_index.get(evidence_id)
        if prompt_row is None:
            raise ValueError(f"No generation prompt for evidence_id={evidence_id} attempt={attempt}")
        if packet is None:
            raise ValueError(f"No evidence packet for evidence_id={evidence_id}")
        prompt = str(prompt_row.get("prompt") or "")
        plan.append(
            {
                "source_index": row["_source_index"],
                "source_qa_id": row.get("qa_id"),
                "evidence_id": evidence_id,
                "attempt": attempt,
                "question_type": prompt_row.get("question_type"),
                "generation_mode": prompt_row.get("generation_mode"),
                "prompt_sha256": prompt_digest(prompt),
                "prompt_chars": len(prompt),
                "image_paths": prompt_row.get("image_paths") or [],
                "video_paths": prompt_row.get("video_paths") or [],
                "required_users": packet.get("required_users", []),
            }
        )
    return plan


def normalize_generated_qa(
    *,
    qa: dict[str, Any],
    source_index: int,
    source_qa: dict[str, Any],
    prompt_row: dict[str, Any],
    packet: dict[str, Any],
    model_id: str,
    raw_generation: str,
    usage_metadata: dict[str, Any],
) -> dict[str, Any]:
    evidence_id = str(source_qa.get("evidence_id"))
    question_type = str(prompt_row.get("question_type") or source_qa.get("question_type") or "natural_two_user")
    qa["qa_id"] = f"GEMINI25PRO_REPLAY_{source_index:02d}_{source_qa.get('qa_id', evidence_id)}"
    qa["evidence_id"] = evidence_id
    qa["required_users"] = packet.get("required_users", qa.get("required_users", []))
    qa["model_id"] = model_id
    qa["source_urls"] = packet.get("source_urls", {})
    qa["video_evidence"] = video_evidence_for_packet(packet)
    qa.setdefault("referred_timestamps", [])
    qa["human_audit"] = human_audit_packet(packet)
    qa["attempt_count"] = int(source_qa.get("attempt_count") or 1)
    qa["source_replay"] = {
        "source_qwen_qa_id": source_qa.get("qa_id"),
        "source_qwen_model_id": source_qa.get("model_id"),
        "source_qwen_question": source_qa.get("question"),
        "source_qwen_answer": source_qa.get("answer"),
        "source_index": source_index,
        "source_attempt": qa["attempt_count"],
        "source_generation_prompt_sha256": prompt_digest(str(prompt_row.get("prompt") or "")),
        "replay_policy": "same accepted generation prompt and media; generator model changed",
    }
    qa["generation_trace"] = [
        {
            "attempt": qa["attempt_count"],
            "stage": "gemini_replay_generation",
            "question_type": question_type,
            "design_cell": prompt_row.get("design_cell"),
            "generation_mode": prompt_row.get("generation_mode"),
            "source_qwen_qa_id": source_qa.get("qa_id"),
            "prompt_sha256": qa["source_replay"]["source_generation_prompt_sha256"],
            "raw_output": raw_generation,
            "usage_metadata": usage_metadata,
            "media": {
                "image_paths": prompt_row.get("image_paths") or [],
                "video_paths": prompt_row.get("video_paths") or [],
            },
        }
    ]
    complete_generator_metadata(
        qa,
        packet=packet,
        question_type=question_type,
        design_cell=prompt_row.get("design_cell"),
    )
    return qa


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    source_rows = source_with_indices(load_jsonl(args.source_qa), args.limit)
    prompt_rows_source = load_jsonl(args.source_prompts)
    evidence_rows = load_jsonl(args.evidence)
    prompt_index = index_prompts(prompt_rows_source)
    evidence_index = {str(row.get("evidence_id")): row for row in evidence_rows}
    plan = build_plan(source_rows, prompt_index, evidence_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "replay_plan.json", plan)
    if args.plan_only:
        print(f"plan_ok rows={len(plan)} output_dir={output_dir}", flush=True)
        return

    runner = make_runner(
        "gemini-api",
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
    )
    accepted_rows: list[dict[str, Any]] = []
    intermediate_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    usage_calls: list[dict[str, Any]] = []

    for source_qa in source_rows:
        source_index = int(source_qa["_source_index"])
        evidence_id = str(source_qa.get("evidence_id"))
        attempt = int(source_qa.get("attempt_count") or 1)
        prompt_row = copy.deepcopy(prompt_index[("generation", evidence_id, attempt)])
        packet = evidence_index[evidence_id]
        image_paths = list(prompt_row.get("image_paths") or [])
        video_paths = list(prompt_row.get("video_paths") or [])
        prompt = str(prompt_row.get("prompt") or "")
        prompt_rows.append(
            {
                **prompt_row,
                "replay_model_id": args.model_id,
                "source_qwen_qa_id": source_qa.get("qa_id"),
                "source_index": source_index,
                "prompt_sha256": prompt_digest(prompt),
            }
        )
        print(
            "replay_generation_start "
            f"index={source_index} evidence_id={evidence_id} attempt={attempt} "
            f"images={len(image_paths)} videos={len(video_paths)}",
            flush=True,
        )
        stage_start = time.time()
        raw_generation = runner.generate(prompt, image_paths=image_paths, video_paths=video_paths)
        generation_seconds = time.time() - stage_start
        generation_usage = usage_from_runner(runner)
        usage_calls.append(
            {
                "stage": "generation",
                "index": source_index,
                "qa_id": source_qa.get("qa_id"),
                "evidence_id": evidence_id,
                "usage_metadata": generation_usage,
            }
        )
        raw_row: dict[str, Any] = {
            "source_index": source_index,
            "source_qwen_qa_id": source_qa.get("qa_id"),
            "source_qwen_question": source_qa.get("question"),
            "source_qwen_answer": source_qa.get("answer"),
            "evidence_id": evidence_id,
            "attempt": attempt,
            "model_id": args.model_id,
            "generation_seconds": generation_seconds,
            "generation_usage_metadata": generation_usage,
            "raw_generation": raw_generation,
            "parse_error": None,
        }
        try:
            generated_qa = extract_json_object(raw_generation)
        except Exception as exc:
            raw_row["parse_error"] = str(exc)
            rejected_rows.append(
                {
                    "source_index": source_index,
                    "source_qwen_qa_id": source_qa.get("qa_id"),
                    "evidence_id": evidence_id,
                    "stage": "generation_parse",
                    "reason": str(exc),
                    "raw_generation": raw_generation,
                }
            )
            raw_rows.append(raw_row)
            write_outputs(
                output_dir=output_dir,
                args=args,
                source_rows=source_rows,
                accepted_rows=accepted_rows,
                intermediate_rows=intermediate_rows,
                rejected_rows=rejected_rows,
                raw_rows=raw_rows,
                prompt_rows=prompt_rows,
                usage_calls=usage_calls,
            )
            continue

        qa = normalize_generated_qa(
            qa=generated_qa,
            source_index=source_index,
            source_qa=source_qa,
            prompt_row=prompt_row,
            packet=packet,
            model_id=args.model_id,
            raw_generation=raw_generation,
            usage_metadata=generation_usage,
        )
        raw_row["parsed_qa"] = {
            "qa_id": qa.get("qa_id"),
            "question": qa.get("question"),
            "answer": qa.get("answer"),
            "correct": qa.get("correct"),
            "generator_rationale": qa.get("generator_rationale"),
            "why_two_users_needed": qa.get("why_two_users_needed"),
            "per_user_evidence_claims": qa.get("per_user_evidence_claims"),
        }
        schema_errors = validate_qa_item(qa)
        if schema_errors:
            feedback = "Schema errors to fix: " + "; ".join(schema_errors)
            qa["review"] = build_review_from_gates(
                judge=None,
                answerability=None,
                schema_errors=schema_errors,
                accepted=False,
                rejection_stage="schema",
                final_reason=feedback,
            )
            intermediate_rows.append(qa)
            rejected_rows.append(
                {
                    "source_index": source_index,
                    "source_qwen_qa_id": source_qa.get("qa_id"),
                    "evidence_id": evidence_id,
                    "stage": "schema",
                    "reason": feedback,
                    "qa": qa,
                }
            )
            raw_rows.append(raw_row)
            write_outputs(
                output_dir=output_dir,
                args=args,
                source_rows=source_rows,
                accepted_rows=accepted_rows,
                intermediate_rows=intermediate_rows,
                rejected_rows=rejected_rows,
                raw_rows=raw_rows,
                prompt_rows=prompt_rows,
                usage_calls=usage_calls,
            )
            continue

        if args.skip_judge:
            answerability = skipped_answerability_review()
            qa["review"] = build_review_from_gates(
                judge={"skipped": True},
                answerability=answerability,
                schema_errors=[],
                accepted=False,
                final_reason="schema passed; judger skipped for generation-only replay",
                human_review_pending=True,
            )
            intermediate_rows.append(qa)
            accepted_rows.append(qa)
            raw_rows.append(raw_row)
            write_outputs(
                output_dir=output_dir,
                args=args,
                source_rows=source_rows,
                accepted_rows=accepted_rows,
                intermediate_rows=intermediate_rows,
                rejected_rows=rejected_rows,
                raw_rows=raw_rows,
                prompt_rows=prompt_rows,
                usage_calls=usage_calls,
            )
            continue

        judge_prompt = build_judger_prompt(qa_for_judger_prompt(qa), packet)
        judge_prompt_row = {
            "stage": "judge",
            "evidence_id": evidence_id,
            "question_type": qa.get("question_type"),
            "design_cell": qa.get("design_cell"),
            "generation_mode": prompt_row.get("generation_mode"),
            "attempt": attempt,
            "prompt": judge_prompt,
            "image_paths": image_paths,
            "video_paths": video_paths,
            "replay_model_id": args.model_id,
            "source_qwen_qa_id": source_qa.get("qa_id"),
            "source_index": source_index,
            "prompt_sha256": prompt_digest(judge_prompt),
        }
        prompt_rows.append(judge_prompt_row)
        print(
            "replay_judge_start "
            f"index={source_index} evidence_id={evidence_id} qa_id={qa.get('qa_id')} "
            f"images={len(image_paths)} videos={len(video_paths)}",
            flush=True,
        )
        stage_start = time.time()
        raw_judge = runner.generate(judge_prompt, image_paths=image_paths, video_paths=video_paths)
        judge_seconds = time.time() - stage_start
        judge_usage = usage_from_runner(runner)
        usage_calls.append(
            {
                "stage": "judge",
                "index": source_index,
                "qa_id": qa.get("qa_id"),
                "evidence_id": evidence_id,
                "usage_metadata": judge_usage,
            }
        )
        raw_row["raw_judge"] = raw_judge
        raw_row["judge_seconds"] = judge_seconds
        raw_row["judge_usage_metadata"] = judge_usage
        try:
            judge = extract_json_object(raw_judge)
        except Exception as exc:
            judge = {
                "review_passed": False,
                "feedback_to_generator": f"Judger output was not valid JSON: {exc}",
            }
        judge["raw_output"] = raw_judge
        judge["gate"] = judge_gate(judge)
        qa["generation_trace"].append(
            {
                "attempt": attempt,
                "stage": "gemini_replay_judge",
                "prompt_sha256": prompt_digest(judge_prompt),
                "raw_output": raw_judge,
                "usage_metadata": judge_usage,
            }
        )
        if judge["gate"].get("passed") is not True:
            feedback = str(
                judge.get("feedback_to_generator")
                or judge["gate"].get("reason")
                or "Judger rejected the question."
            )
            qa["review"] = build_review_from_gates(
                judge=judge,
                answerability=None,
                schema_errors=[],
                accepted=False,
                rejection_stage="judger",
                final_reason=feedback,
            )
            intermediate_rows.append(qa)
            rejected_rows.append(
                {
                    "source_index": source_index,
                    "source_qwen_qa_id": source_qa.get("qa_id"),
                    "evidence_id": evidence_id,
                    "stage": "judger",
                    "reason": feedback,
                    "qa": qa,
                }
            )
            raw_rows.append(raw_row)
            write_outputs(
                output_dir=output_dir,
                args=args,
                source_rows=source_rows,
                accepted_rows=accepted_rows,
                intermediate_rows=intermediate_rows,
                rejected_rows=rejected_rows,
                raw_rows=raw_rows,
                prompt_rows=prompt_rows,
                usage_calls=usage_calls,
            )
            continue

        answerability = skipped_answerability_review()
        qa["review"] = build_review_from_gates(
            judge=judge,
            answerability=answerability,
            schema_errors=[],
            accepted=False,
            final_reason="schema and judger passed; answerability gate skipped for human review",
            human_review_pending=True,
        )
        strict_errors = validate_qa_item(qa, strict_review=False)
        if strict_errors:
            feedback = "Post-judge validation errors: " + "; ".join(strict_errors)
            qa["review"] = build_review_from_gates(
                judge=judge,
                answerability=answerability,
                schema_errors=strict_errors,
                accepted=False,
                rejection_stage="schema",
                final_reason=feedback,
            )
            intermediate_rows.append(qa)
            rejected_rows.append(
                {
                    "source_index": source_index,
                    "source_qwen_qa_id": source_qa.get("qa_id"),
                    "evidence_id": evidence_id,
                    "stage": "post_judge_schema",
                    "reason": feedback,
                    "qa": qa,
                }
            )
            raw_rows.append(raw_row)
            write_outputs(
                output_dir=output_dir,
                args=args,
                source_rows=source_rows,
                accepted_rows=accepted_rows,
                intermediate_rows=intermediate_rows,
                rejected_rows=rejected_rows,
                raw_rows=raw_rows,
                prompt_rows=prompt_rows,
                usage_calls=usage_calls,
            )
            continue

        intermediate_rows.append(qa)
        accepted_rows.append(qa)
        raw_rows.append(raw_row)
        print(
            "replay_row_done "
            f"index={source_index} evidence_id={evidence_id} qa_id={qa.get('qa_id')} "
            f"status={qa.get('review', {}).get('status')}",
            flush=True,
        )
        write_outputs(
            output_dir=output_dir,
            args=args,
            source_rows=source_rows,
            accepted_rows=accepted_rows,
            intermediate_rows=intermediate_rows,
            rejected_rows=rejected_rows,
            raw_rows=raw_rows,
            prompt_rows=prompt_rows,
            usage_calls=usage_calls,
        )

    print(
        "replay_done "
        f"attempted={len(source_rows)} parsed={len(intermediate_rows)} "
        f"qa_mcq_rows={len(accepted_rows)} rejected={len(rejected_rows)} "
        f"output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
