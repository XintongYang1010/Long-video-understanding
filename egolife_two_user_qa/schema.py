"""Schema validation and JSON parsing for generated QA items."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from .io_utils import iter_jsonl


OPTION_LETTERS = ("A", "B", "C", "D", "E")
REQUIRED_QA_FIELDS = {
    "qa_id",
    "question",
    "options",
    "correct",
    "answer",
    "category",
    "required_users",
    "evidence",
    "single_user_answerability",
    "combined_answerability",
    "review",
    "model_id",
    "source_urls",
}
VIDEO_FIRST_REQUIRED_FIELDS = {
    "question_type",
    "generator_rationale",
    "why_two_users_needed",
    "per_user_evidence_claims",
    "judge_feedback",
    "answerability_eval",
    "attempt_count",
}


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output")
    return json.loads(cleaned[start : end + 1])


def normalize_correct(value: Any) -> str:
    if isinstance(value, int):
        if 0 <= value < 5:
            return OPTION_LETTERS[value]
    text = str(value).strip().upper()
    if text in OPTION_LETTERS:
        return text
    match = re.search(r"\b([A-E])\b", text)
    if match:
        return match.group(1)
    raise ValueError(f"Invalid correct option: {value!r}")


def validate_qa_item(item: dict[str, Any], *, strict_review: bool = False) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_QA_FIELDS - set(item))
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    if strict_review:
        missing_video = sorted(VIDEO_FIRST_REQUIRED_FIELDS - set(item))
        if missing_video:
            errors.append(f"missing video-first fields: {', '.join(missing_video)}")

    options = item.get("options")
    if not isinstance(options, list) or len(options) != 5:
        errors.append("options must contain exactly five entries")
    elif any(not isinstance(option, str) or not option.strip() for option in options):
        errors.append("all options must be non-empty strings")

    try:
        correct = normalize_correct(item.get("correct"))
        item["correct"] = correct
        if isinstance(options, list) and len(options) == 5:
            answer = item.get("answer")
            expected_answer = options[OPTION_LETTERS.index(correct)]
            if answer != expected_answer:
                errors.append("answer must equal options[correct]")
    except ValueError as exc:
        errors.append(str(exc))

    required_users = item.get("required_users")
    if not isinstance(required_users, list) or len(required_users) < 2:
        errors.append("required_users must list at least two users")
    elif len(set(required_users)) != len(required_users):
        errors.append("required_users must not contain duplicates")

    single = item.get("single_user_answerability")
    if not isinstance(single, dict):
        errors.append("single_user_answerability must be an object")
    elif required_users:
        missing_single = [user for user in required_users if user not in single]
        if missing_single:
            errors.append(f"single_user_answerability missing required users: {missing_single}")
        for user in required_users:
            text = str(single.get(user, "")).lower()
            if "insufficient" not in text and "cannot" not in text and "not enough" not in text:
                errors.append(f"single user {user} must be marked insufficient")

    combined = str(item.get("combined_answerability", "")).lower()
    if "sufficient" not in combined and "support" not in combined:
        errors.append("combined_answerability must state sufficient support")

    question_type = item.get("question_type")
    if question_type is not None and question_type not in {"commonality", "difference"}:
        errors.append("question_type must be commonality or difference")

    answerability = item.get("answerability_eval")
    if answerability is not None:
        if not isinstance(answerability, dict):
            errors.append("answerability_eval must be an object")
        else:
            gate = answerability.get("gate")
            evaluations = answerability.get("evaluations")
            if strict_review and (not isinstance(gate, dict) or gate.get("passed") is not True):
                errors.append("answerability gate must pass in strict mode")
            if strict_review and not isinstance(evaluations, list):
                errors.append("answerability_eval.evaluations must be a list in strict mode")

    judge = item.get("judge_feedback")
    if judge is not None:
        if not isinstance(judge, dict):
            errors.append("judge_feedback must be an object")
        elif strict_review and judge.get("review_passed") is not True:
            errors.append("judge_feedback.review_passed must be true in strict mode")

    review = item.get("review")
    if not isinstance(review, dict):
        errors.append("review must be an object")
    elif strict_review and review.get("review_passed") is not True and review.get("status") != "passed":
        errors.append("review must pass in strict mode")

    return errors


def load_and_validate(path: str | Path, *, strict_review: bool = False) -> tuple[int, list[str]]:
    errors = []
    count = 0
    for count, item in enumerate(iter_jsonl(path), 1):
        row_errors = validate_qa_item(item, strict_review=strict_review)
        errors.extend([f"row {count}: {error}" for error in row_errors])
    return count, errors


def write_qa_csv(jsonl_path: str | Path, csv_path: str | Path) -> int:
    rows = list(iter_jsonl(jsonl_path))
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "qa_id",
        "question",
        "correct",
        "answer",
        "category",
        "required_users",
        "combined_answerability",
        "review_passed",
        "question_type",
        "attempt_count",
        "answerability_passed",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            review = row.get("review") if isinstance(row.get("review"), dict) else {}
            writer.writerow(
                {
                    "qa_id": row.get("qa_id", ""),
                    "question": row.get("question", ""),
                    "correct": row.get("correct", ""),
                    "answer": row.get("answer", ""),
                    "category": row.get("category", ""),
                    "required_users": ";".join(row.get("required_users", [])),
                    "combined_answerability": row.get("combined_answerability", ""),
                    "review_passed": review.get("review_passed", review.get("status", "")),
                    "question_type": row.get("question_type", ""),
                    "attempt_count": row.get("attempt_count", ""),
                    "answerability_passed": (
                        row.get("answerability_eval", {}).get("gate", {}).get("passed", "")
                        if isinstance(row.get("answerability_eval"), dict)
                        else ""
                    ),
                }
            )
    return len(rows)
