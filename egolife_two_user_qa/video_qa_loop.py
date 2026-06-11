"""Video-first generation loop for EgoLife two-user MCQ construction."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

from .io_utils import iter_jsonl, write_jsonl
from .prompts import build_answerability_prompt, build_judger_prompt, build_video_generation_prompt
from .qwen3vl_runner import DEFAULT_MODEL_ID, make_runner
from .schema import extract_json_object, normalize_correct, validate_qa_item


QUESTION_TYPES = ("commonality", "difference")


def existing_path(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return str(path)
    return None


def clip_video_path(clip: dict[str, Any]) -> str | None:
    return existing_path(clip.get("local_video"))


def clip_image_paths(clip: dict[str, Any]) -> list[str]:
    paths = []
    for frame in clip.get("frames", []):
        path = existing_path(frame.get("path"))
        if path:
            paths.append(path)
    return paths


def media_for_clips(
    clips: list[dict[str, Any]],
    *,
    backend: str,
    allow_openai_video_input: bool,
) -> tuple[list[str], list[str]]:
    videos = [path for clip in clips if (path := clip_video_path(clip))]
    images = [path for clip in clips for path in clip_image_paths(clip)]
    if backend == "openai-compatible-local" and not allow_openai_video_input:
        return images, []
    return images if not videos else [], videos


def clips_for_users(packet: dict[str, Any], users: list[str]) -> list[dict[str, Any]]:
    wanted = set(users)
    return [clip for clip in packet.get("clips", []) if clip.get("agent_name") in wanted]


def target_type_counts(target_count: int) -> dict[str, int]:
    commonality = (target_count + 1) // 2
    difference = target_count - commonality
    return {"commonality": commonality, "difference": difference}


def choose_question_type(counts: dict[str, int], targets: dict[str, int]) -> str | None:
    remaining = {
        question_type: targets[question_type] - counts.get(question_type, 0)
        for question_type in QUESTION_TYPES
    }
    remaining = {key: value for key, value in remaining.items() if value > 0}
    if not remaining:
        return None
    return sorted(remaining.items(), key=lambda item: (-item[1], item[0]))[0][0]


def build_answerability_conditions(required_users: list[str]) -> list[dict[str, Any]]:
    users = list(required_users)
    conditions = [
        {
            "condition_id": f"single_user::{user}",
            "condition_type": "single_user",
            "users": [user],
        }
        for user in users
    ]
    if len(users) > 2:
        for size in range(2, len(users)):
            for combo in itertools.combinations(users, size):
                combo_users = list(combo)
                conditions.append(
                    {
                        "condition_id": "proper_subset::" + "+".join(combo_users),
                        "condition_type": "proper_subset",
                        "users": combo_users,
                    }
                )
    conditions.append(
        {
            "condition_id": "combined_all_users::" + "+".join(users),
            "condition_type": "combined_all_users",
            "users": users,
        }
    )
    return conditions


def parsed_choice(value: Any) -> tuple[str | None, bool]:
    text = str(value or "").strip()
    if text.lower() in {"insufficient", "not enough", "unknown", "cannot answer", "can't answer"}:
        return None, True
    try:
        return normalize_correct(text), False
    except ValueError:
        return None, False


def answerability_gate(qa_item: dict[str, Any], evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        correct = normalize_correct(qa_item.get("correct"))
    except ValueError as exc:
        return {"passed": False, "reason": str(exc)}

    combined = [row for row in evaluations if row.get("condition_type") == "combined_all_users"]
    if not combined:
        return {"passed": False, "reason": "missing combined_all_users evaluation"}

    combined_choice, combined_insufficient = parsed_choice(combined[-1].get("choice"))
    if combined_insufficient or combined_choice != correct:
        return {
            "passed": False,
            "reason": f"combined_all_users did not select correct answer {correct}",
        }

    leaking = []
    for row in evaluations:
        if row.get("condition_type") == "combined_all_users":
            continue
        choice, insufficient = parsed_choice(row.get("choice"))
        if not insufficient and choice == correct:
            leaking.append(row.get("condition_id"))
    if leaking:
        return {
            "passed": False,
            "reason": "single/subset condition answered correctly: " + ", ".join(str(item) for item in leaking),
        }

    return {
        "passed": True,
        "reason": "combined videos answer correctly and all single/subset conditions are insufficient or incorrect",
    }


def dry_run_qa(packet: dict[str, Any], question_type: str) -> dict[str, Any]:
    users = packet.get("required_users", [])[:2]
    return {
        "qa_id": f"DRYRUN_{packet.get('evidence_id')}_{question_type}",
        "question_type": question_type,
        "question": "Which option can be determined only after comparing what we each experienced?",
        "options": ["Option A", "Option B", "Option C", "Option D", "Option E"],
        "correct": "A",
        "answer": "Option A",
        "category": "environmental_interaction",
        "required_users": users,
        "evidence": [{"user": user, "needed_fact": "dry-run video evidence", "frames_used": []} for user in users],
        "single_user_answerability": {user: "insufficient in dry-run mode" for user in users},
        "combined_answerability": "sufficient in dry-run prompt construction only",
        "generator_rationale": "dry-run placeholder",
        "why_two_users_needed": "dry-run placeholder",
        "per_user_evidence_claims": [{"user": user, "claim": "dry-run placeholder"} for user in users],
        "judge_feedback": {},
        "answerability_eval": {},
        "attempt_count": 0,
        "review": {"review_passed": False, "status": "dry_run"},
        "model_id": "dry-run-no-model",
        "source_urls": packet.get("source_urls", {}),
    }


def run_answerability_eval(
    *,
    qa_item: dict[str, Any],
    packet: dict[str, Any],
    runner: Any,
    backend: str,
    allow_openai_video_input: bool,
    prompt_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluations = []
    for condition in build_answerability_conditions(qa_item.get("required_users", [])):
        clips = clips_for_users(packet, condition["users"])
        image_paths, video_paths = media_for_clips(
            clips,
            backend=backend,
            allow_openai_video_input=allow_openai_video_input,
        )
        prompt = build_answerability_prompt(qa_item, condition)
        prompt_rows.append(
            {
                "stage": "answerability",
                "qa_id": qa_item.get("qa_id"),
                "condition_id": condition["condition_id"],
                "prompt": prompt,
                "image_paths": image_paths,
                "video_paths": video_paths,
            }
        )
        raw = runner.generate(prompt, image_paths=image_paths, video_paths=video_paths)
        try:
            answer = extract_json_object(raw)
        except Exception as exc:
            answer = {
                "choice": "insufficient",
                "answer_text": "",
                "confidence": 0.0,
                "evidence_used": "",
                "insufficient_reason": f"parse_failed: {exc}",
            }
        evaluations.append({**condition, **answer, "raw_output": raw})
    gate = answerability_gate(qa_item, evaluations)
    return {"evaluations": evaluations, "gate": gate}


def generate_video_qa_loop(
    *,
    evidence_path: str | Path,
    output_path: str | Path,
    prompts_path: str | Path | None,
    rejected_path: str | Path | None,
    backend: str,
    model_id: str = DEFAULT_MODEL_ID,
    base_url: str = "http://127.0.0.1:8000/v1",
    target_count: int = 20,
    max_attempts: int = 3,
    max_new_tokens: int = 1536,
    max_image_pixels: int = 262144,
    dtype: str = "bfloat16",
    allow_cpu: bool = False,
    allow_openai_video_input: bool = False,
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
        allow_openai_video_input=allow_openai_video_input,
    )
    prompts = []
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    targets = target_type_counts(target_count)
    counts = {question_type: 0 for question_type in QUESTION_TYPES}

    for packet in iter_jsonl(evidence_path):
        if len(accepted) >= target_count:
            break
        question_type = choose_question_type(counts, targets)
        if question_type is None:
            break
        clips = packet.get("clips", [])
        image_paths, video_paths = media_for_clips(
            clips,
            backend=backend,
            allow_openai_video_input=allow_openai_video_input,
        )
        feedback = None
        if dry_run:
            qa = dry_run_qa(packet, question_type)
            gen_prompt = build_video_generation_prompt(packet, question_type)
            judge_prompt = build_judger_prompt(qa, packet)
            prompts.append(
                {
                    "stage": "generation",
                    "evidence_id": packet.get("evidence_id"),
                    "question_type": question_type,
                    "attempt": 1,
                    "prompt": gen_prompt,
                    "image_paths": image_paths,
                    "video_paths": video_paths,
                }
            )
            prompts.append(
                {
                    "stage": "judge",
                    "evidence_id": packet.get("evidence_id"),
                    "question_type": question_type,
                    "attempt": 1,
                    "prompt": judge_prompt,
                    "image_paths": image_paths,
                    "video_paths": video_paths,
                }
            )
            for condition in build_answerability_conditions(packet.get("required_users", [])):
                condition_clips = clips_for_users(packet, condition["users"])
                cond_images, cond_videos = media_for_clips(
                    condition_clips,
                    backend=backend,
                    allow_openai_video_input=allow_openai_video_input,
                )
                prompts.append(
                    {
                        "stage": "answerability",
                        "evidence_id": packet.get("evidence_id"),
                        "question_type": question_type,
                        "condition_id": condition["condition_id"],
                        "prompt": build_answerability_prompt(qa, condition),
                        "image_paths": cond_images,
                        "video_paths": cond_videos,
                    }
                )
            counts[question_type] += 1
            accepted.append(qa)
            continue

        packet_rejections = []
        for attempt in range(1, max_attempts + 1):
            gen_prompt = build_video_generation_prompt(packet, question_type, feedback=feedback)
            prompts.append(
                {
                    "stage": "generation",
                    "evidence_id": packet.get("evidence_id"),
                    "question_type": question_type,
                    "attempt": attempt,
                    "prompt": gen_prompt,
                    "image_paths": image_paths,
                    "video_paths": video_paths,
                }
            )
            raw_generation = runner.generate(gen_prompt, image_paths=image_paths, video_paths=video_paths)
            try:
                qa = extract_json_object(raw_generation)
            except Exception as exc:
                feedback = f"Generator output was not valid JSON: {exc}"
                packet_rejections.append({"attempt": attempt, "reason": feedback, "raw_output": raw_generation})
                continue

            qa.setdefault("qa_id", f"QA_{len(accepted) + 1:03d}_{packet.get('evidence_id')}")
            qa["evidence_id"] = packet.get("evidence_id")
            qa["question_type"] = question_type
            qa["required_users"] = packet.get("required_users", qa.get("required_users", []))
            qa["model_id"] = runner.model_id
            qa["source_urls"] = packet.get("source_urls", {})
            qa["attempt_count"] = attempt
            qa.setdefault("review", {})
            qa["review"]["generator_raw_output"] = raw_generation

            schema_errors = validate_qa_item(qa)
            if schema_errors:
                feedback = "Schema errors to fix: " + "; ".join(schema_errors)
                qa["review"]["schema_errors"] = schema_errors
                packet_rejections.append({"attempt": attempt, "reason": feedback, "qa": qa})
                continue

            judge_prompt = build_judger_prompt(qa, packet)
            prompts.append(
                {
                    "stage": "judge",
                    "evidence_id": packet.get("evidence_id"),
                    "question_type": question_type,
                    "attempt": attempt,
                    "prompt": judge_prompt,
                    "image_paths": image_paths,
                    "video_paths": video_paths,
                }
            )
            raw_judge = runner.generate(judge_prompt, image_paths=image_paths, video_paths=video_paths)
            try:
                judge = extract_json_object(raw_judge)
            except Exception as exc:
                judge = {
                    "review_passed": False,
                    "feedback_to_generator": f"Judger output was not valid JSON: {exc}",
                }
            judge["raw_output"] = raw_judge
            qa["judge_feedback"] = judge
            if judge.get("review_passed") is not True:
                feedback = str(judge.get("feedback_to_generator") or "Judger rejected the question.")
                packet_rejections.append({"attempt": attempt, "reason": feedback, "qa": qa})
                continue

            answerability = run_answerability_eval(
                qa_item=qa,
                packet=packet,
                runner=runner,
                backend=backend,
                allow_openai_video_input=allow_openai_video_input,
                prompt_rows=prompts,
            )
            qa["answerability_eval"] = answerability
            if answerability.get("gate", {}).get("passed") is not True:
                feedback = "Answerability gate failed: " + str(answerability.get("gate", {}).get("reason", ""))
                packet_rejections.append({"attempt": attempt, "reason": feedback, "qa": qa})
                continue

            qa["review"] = {
                **qa.get("review", {}),
                "review_passed": True,
                "status": "passed",
                "judger_passed": True,
                "answerability_passed": True,
            }
            strict_errors = validate_qa_item(qa, strict_review=True)
            if strict_errors:
                feedback = "Strict validation errors: " + "; ".join(strict_errors)
                qa["review"]["schema_errors"] = strict_errors
                packet_rejections.append({"attempt": attempt, "reason": feedback, "qa": qa})
                continue

            accepted.append(qa)
            counts[question_type] += 1
            break
        else:
            rejected.append(
                {
                    "evidence_id": packet.get("evidence_id"),
                    "question_type": question_type,
                    "attempts": packet_rejections,
                }
            )

    if prompts_path:
        write_jsonl(prompts_path, prompts)
    if not dry_run:
        write_jsonl(output_path, accepted)
    if rejected_path and rejected:
        write_jsonl(rejected_path, rejected)
    return accepted


def add_video_loop_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", default="transformers-local", choices=["transformers-local", "openai-compatible-local"])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--max-image-pixels", type=int, default=262144)
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--allow-openai-video-input", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Video-first EgoLife two-user QA generation loop")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts-output")
    parser.add_argument("--rejected-output")
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=3)
    add_video_loop_args(parser)
    args = parser.parse_args(argv)
    rows = generate_video_qa_loop(
        evidence_path=args.evidence,
        output_path=args.output,
        prompts_path=args.prompts_output,
        rejected_path=args.rejected_output,
        backend=args.backend,
        model_id=args.model_id,
        base_url=args.base_url,
        target_count=args.target_count,
        max_attempts=args.max_attempts,
        max_new_tokens=args.max_new_tokens,
        max_image_pixels=args.max_image_pixels,
        dtype=args.dtype,
        allow_cpu=args.allow_cpu,
        allow_openai_video_input=args.allow_openai_video_input,
        dry_run=args.dry_run,
    )
    print(f"accepted {len(rows)} video-first QA rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
