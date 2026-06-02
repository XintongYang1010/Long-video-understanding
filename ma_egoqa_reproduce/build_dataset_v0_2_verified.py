#!/usr/bin/env python3
"""
Build Dataset V0.2 Verified from Dataset V0.1 and visual-audit human labels.

This script does not run models, rescreen the full dataset, download videos, or
modify Dataset V0.1. It only packages the manually reviewed visual-audit subset.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
V01_DIR = ROOT / "outputs" / "historical_v2_fullscreen" / "dataset_v0_1"
AUDIT_DIR = V01_DIR / "visual_audit_v1"
OUT_DIR = ROOT / "outputs" / "historical_v2_fullscreen" / "dataset_v0_2_verified"

V01_SPLIT_PATHS = [
    V01_DIR / "dataset_v0_1_demo.csv",
    V01_DIR / "dataset_v0_1_eval_lite.csv",
    V01_DIR / "dataset_v0_1_control.csv",
]
AUDIT_TABLE_WITH_FRAMES = AUDIT_DIR / "visual_audit_table_with_frames.csv"
AUDIT_TABLE_CAPTION_ONLY = AUDIT_DIR / "visual_audit_table.csv"

BASE_FIELDS = [
    "visual_case_id",
    "source_split",
    "case_id",
    "source_question_id",
    "source_tier",
    "question",
    "answer",
    "category",
    "subcategory",
    "case_type",
    "current_only_context",
    "history_only_context",
    "current_plus_historical_context",
    "current_only_answerability",
    "history_only_answerability",
    "current_plus_history_answerability",
    "current_plus_history_gain",
    "expected_comparison",
    "expected_result",
    "evidence_sources",
    "evidence_time_windows",
    "confidence",
    "why_selected",
    "potential_issue",
    "label_status",
    "needs_human_check",
    "contact_sheet_path",
    "extracted_frame_paths",
    "visual_audit_status",
    "human_audit_label",
    "audit_reason",
    "ppt_suitability",
    "rejection_reason",
]

HISTORY_DEMO = {
    "VA_DEMO_001": "1078",
    "VA_DEMO_007": "523",
}

POSSIBLE_HISTORY_EVAL = {
    "VA_DEMO_003": "313",
}

CURRENT_CONTROL = {
    "VA_DEMO_005": "982",
    "VA_DEMO_006": "1073",
    "VA_DEMO_009": "433",
    "VA_DEMO_010": "419",
    "VA_EVAL_001": "20",
    "VA_EVAL_002": "408",
    "VA_EVAL_003": "871",
    "VA_CONTROL_001": "178",
    "VA_CONTROL_002": "652",
    "VA_CONTROL_003": "1262",
    "VA_CONTROL_004": "995",
    "VA_CONTROL_005": "206",
}

REJECT = {
    "VA_DEMO_002": "617",
    "VA_DEMO_004": "1040",
    "VA_DEMO_008": "437",
    "VA_EVAL_004": "1396",
    "VA_EVAL_005": "615",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: safe_csv(row.get(field, "")) for field in fields})


def safe_csv(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
    return value


def normalize_ws(text: Any) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())


def trunc(text: Any, limit: int) -> str:
    clean = normalize_ws(text)
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def load_v01_cases() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for path in V01_SPLIT_PATHS:
        for row in read_csv(path):
            out.setdefault(row["case_id"], row)
    return out


def load_audit_rows() -> dict[str, dict[str, str]]:
    path = AUDIT_TABLE_WITH_FRAMES if AUDIT_TABLE_WITH_FRAMES.exists() else AUDIT_TABLE_CAPTION_ONLY
    rows = read_csv(path)
    return {row["visual_case_id"]: row for row in rows}


def validate_human_maps(audit_by_visual: dict[str, dict[str, str]]) -> None:
    all_maps = {}
    for label_map in [HISTORY_DEMO, POSSIBLE_HISTORY_EVAL, CURRENT_CONTROL, REJECT]:
        for visual_id, qid in label_map.items():
            if visual_id in all_maps:
                raise ValueError(f"Duplicate visual case label: {visual_id}")
            all_maps[visual_id] = qid
    missing = sorted(set(all_maps) - set(audit_by_visual))
    if missing:
        raise ValueError(f"Human-labeled visual IDs missing from audit table: {missing}")
    mismatched = []
    for visual_id, expected_qid in all_maps.items():
        actual_qid = str(audit_by_visual[visual_id].get("question_id", ""))
        if actual_qid != expected_qid:
            mismatched.append(f"{visual_id}: expected Q{expected_qid}, found Q{actual_qid}")
    if mismatched:
        raise ValueError("Question ID mismatches:\n" + "\n".join(mismatched))


def make_row(
    visual_id: str,
    audit_by_visual: dict[str, dict[str, str]],
    v01_by_case: dict[str, dict[str, str]],
    visual_status: str,
    human_label: str,
    audit_reason: str,
    ppt_suitability: str,
    expected_result: str,
    rejection_reason: str = "",
) -> dict[str, str]:
    audit = audit_by_visual[visual_id]
    case_id = audit["dataset_case_id"]
    if case_id not in v01_by_case:
        raise KeyError(f"Dataset case not found in V0.1: {case_id}")
    row = dict(v01_by_case[case_id])
    row["visual_case_id"] = visual_id
    row["source_split"] = audit.get("split", "")
    row["case_id"] = case_id
    row["source_question_id"] = row.get("source_question_id", audit.get("question_id", ""))
    row["contact_sheet_path"] = audit.get("contact_sheet_path", "")
    row["extracted_frame_paths"] = audit.get("extracted_frame_paths", "")
    row["visual_audit_status"] = visual_status
    row["human_audit_label"] = human_label
    row["audit_reason"] = audit_reason
    row["ppt_suitability"] = ppt_suitability
    row["rejection_reason"] = rejection_reason
    row["expected_result"] = expected_result
    row["label_status"] = "visual_spot_checked_candidate" if not rejection_reason else "rejected_after_visual_audit"
    row["needs_human_check"] = "yes"
    return row


def build_rows(audit_by_visual: dict[str, dict[str, str]], v01_by_case: dict[str, dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    history_rows = [
        make_row(
            visual_id,
            audit_by_visual,
            v01_by_case,
            "visual_spot_checked",
            "verified_history_demo",
            "Human visual audit marked this case as PPT_HERO / verified_history_demo.",
            "ppt_ready",
            "current_plus_history_better",
        )
        for visual_id in HISTORY_DEMO
    ]
    possible_rows = [
        make_row(
            visual_id,
            audit_by_visual,
            v01_by_case,
            "visual_spot_checked",
            "possible_eval_history",
            "Human visual audit marked this case as possible_eval_history; keep as a tentative eval candidate, not a PPT hero.",
            "possible_after_more_review",
            "current_plus_history_may_help_needs_more_review",
        )
        for visual_id in POSSIBLE_HISTORY_EVAL
    ]
    control_rows = [
        make_row(
            visual_id,
            audit_by_visual,
            v01_by_case,
            "visual_spot_checked",
            "verified_current_control",
            "Human visual audit marked this case as verified_current_control.",
            "supporting_control",
            "current_only_sufficient",
        )
        for visual_id in CURRENT_CONTROL
    ]
    rejected_rows = [
        make_row(
            visual_id,
            audit_by_visual,
            v01_by_case,
            "visual_spot_checked_rejected",
            "reject",
            "Human visual audit rejected this case for Dataset V0.2 Verified.",
            "not_suitable",
            v01_by_case[audit_by_visual[visual_id]["dataset_case_id"]].get("expected_result", ""),
            "Rejected by human visual audit; visual/caption evidence was not reliable enough for V0.2 Verified.",
        )
        for visual_id in REJECT
    ]
    return history_rows, possible_rows, control_rows, rejected_rows


def scope_context(case: dict[str, str], scope: str) -> str:
    if scope == "current_only":
        return case.get("current_only_context", "")
    if scope == "history_only":
        return case.get("history_only_context", "")
    return case.get("current_plus_historical_context", "")


def write_model_inputs(path: Path, rows: list[dict[str, str]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for case in rows:
            for scope in ["current_only", "history_only", "current_plus_historical"]:
                item = {
                    "case_id": case["case_id"],
                    "visual_case_id": case["visual_case_id"],
                    "question_id": case["source_question_id"],
                    "evidence_scope": scope,
                    "question": case["question"],
                    "answer": case["answer"],
                    "category": case["category"],
                    "evidence_context": scope_context(case, scope),
                    "evidence_sources": case.get("evidence_sources", ""),
                    "expected_comparison": "current_only_vs_current_plus_history",
                    "expected_result": case.get("expected_result", ""),
                    "visual_audit_status": case.get("visual_audit_status", ""),
                    "human_audit_label": case.get("human_audit_label", ""),
                    "audit_reason": case.get("audit_reason", ""),
                    "ppt_suitability": case.get("ppt_suitability", ""),
                    "contact_sheet_path": case.get("contact_sheet_path", ""),
                    "extracted_frame_paths": case.get("extracted_frame_paths", ""),
                    "label_status": case.get("label_status", ""),
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
    return count


def write_readable(path: Path, groups: list[tuple[str, list[dict[str, str]]]]) -> None:
    lines = [
        "# Dataset V0.2 Verified",
        "",
        "Dataset V0.2 packages manually spot-checked visual-audit candidates from Dataset V0.1.",
        "It is not a final benchmark and does not claim final QA labels.",
        "",
    ]
    for title, rows in groups:
        lines += [f"## {title}", ""]
        for row in rows:
            lines += [
                f"### {row['visual_case_id']} / {row['case_id']} / Q{row['source_question_id']}",
                "",
                f"- Human label: {row['human_audit_label']}",
                f"- Expected result: {row['expected_result']}",
                f"- PPT suitability: {row['ppt_suitability']}",
                f"- Question: {row['question']}",
                f"- Answer: {row['answer']}",
                f"- Contact sheet: `{row['contact_sheet_path']}`",
                f"- Frames: `{trunc(row['extracted_frame_paths'], 400)}`",
                "",
                f"Current-only evidence: {trunc(row['current_only_context'], 900)}",
                "",
                f"Historical evidence: {trunc(row['history_only_context'], 900)}",
                "",
                f"Current+historical evidence: {trunc(row['current_plus_historical_context'], 1100)}",
                "",
            ]
            if row.get("rejection_reason"):
                lines += [f"Rejection reason: {row['rejection_reason']}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(path: Path, history_rows: list[dict[str, str]], possible_rows: list[dict[str, str]], control_rows: list[dict[str, str]], rejected_rows: list[dict[str, str]], model_count: int) -> None:
    ppt = history_rows
    lines = [
        "# Dataset V0.2 Verified Report",
        "",
        "## Counts",
        "",
        f"- History demo count: {len(history_rows)}",
        f"- Possible history eval count: {len(possible_rows)}",
        f"- Verified current control count: {len(control_rows)}",
        f"- Reject count: {len(rejected_rows)}",
        f"- Model input rows: {model_count}",
        "",
        "## PPT-Ready Case List",
        "",
    ]
    lines += [
        f"- {row['visual_case_id']} / {row['case_id']} / Q{row['source_question_id']}: {row['question']}"
        for row in ppt
    ] or ["- none"]
    lines += [
        "",
        "## Construction Notes",
        "",
        "- Source: Dataset V0.1 plus `visual_audit_v1` human spot-check results.",
        "- No VLM/LLM was run.",
        "- No full-dataset rescreening was run.",
        "- No new videos were downloaded.",
        "- Rejected cases are retained separately for provenance and are excluded from `dataset_v0_2_model_inputs.jsonl`.",
        "",
        "## Claim Boundary",
        "",
        "Can say:",
        "- Dataset V0.2 contains visually spot-checked candidate cases.",
        "",
        "Cannot say:",
        "- Final benchmark complete.",
        "- Labels are final.",
        "- Model accuracy improved.",
        "- Historical memory is proven useful.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v01_by_case = load_v01_cases()
    audit_by_visual = load_audit_rows()
    validate_human_maps(audit_by_visual)

    history_rows, possible_rows, control_rows, rejected_rows = build_rows(audit_by_visual, v01_by_case)

    write_csv(OUT_DIR / "verified_history_demo_v0_2.csv", history_rows, BASE_FIELDS)
    write_csv(OUT_DIR / "possible_history_eval_v0_2.csv", possible_rows, BASE_FIELDS)
    write_csv(OUT_DIR / "verified_current_control_v0_2.csv", control_rows, BASE_FIELDS)
    write_csv(OUT_DIR / "rejected_visual_audit_cases_v0_2.csv", rejected_rows, BASE_FIELDS)
    model_count = write_model_inputs(OUT_DIR / "dataset_v0_2_model_inputs.jsonl", history_rows + possible_rows + control_rows)
    write_readable(
        OUT_DIR / "dataset_v0_2_readable.md",
        [
            ("Verified History Demo", history_rows),
            ("Possible History Eval", possible_rows),
            ("Verified Current Control", control_rows),
            ("Rejected Visual Audit Cases", rejected_rows),
        ],
    )
    write_report(OUT_DIR / "dataset_v0_2_report.md", history_rows, possible_rows, control_rows, rejected_rows, model_count)

    print(f"history demo count: {len(history_rows)}")
    print(f"possible history eval count: {len(possible_rows)}")
    print(f"control count: {len(control_rows)}")
    print(f"reject count: {len(rejected_rows)}")
    print(f"model input rows: {model_count}")
    print("PPT-ready case list:")
    for row in history_rows:
        print(f"- {row['visual_case_id']} / {row['case_id']} / Q{row['source_question_id']}: {row['question']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
