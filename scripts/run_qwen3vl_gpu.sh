#!/usr/bin/env bash
set -euo pipefail

TARGET_COUNT=20
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
DTYPE="bfloat16"
MAX_NEW_TOKENS=1536
BACKEND="transformers-local"
BASE_URL="http://127.0.0.1:8000/v1"
OUTDIR="egolife_two_user_qa/outputs/pilot_20_video_first"
CACHE_DIR=".cache/egolife_two_user_qa"
EVIDENCE_TARGET_COUNT=""
MAX_ATTEMPTS=3
ARIA_CALIBRATION_DIR=""
ALLOW_OPENAI_VIDEO_INPUT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target-count) TARGET_COUNT="$2"; shift 2 ;;
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --evidence-target-count) EVIDENCE_TARGET_COUNT="$2"; shift 2 ;;
    --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
    --aria-calibration-dir) ARIA_CALIBRATION_DIR="$2"; shift 2 ;;
    --allow-openai-video-input) ALLOW_OPENAI_VIDEO_INPUT=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${OUTDIR}" "${CACHE_DIR}"
if [[ -z "${EVIDENCE_TARGET_COUNT}" ]]; then
  EVIDENCE_TARGET_COUNT=$((TARGET_COUNT * 4))
fi
calibration_args=()
if [[ -n "${ARIA_CALIBRATION_DIR}" ]]; then
  calibration_args=(--aria-calibration-dir "${ARIA_CALIBRATION_DIR}")
fi
video_input_args=()
if [[ "${ALLOW_OPENAI_VIDEO_INPUT}" -eq 1 ]]; then
  video_input_args=(--allow-openai-video-input)
fi

python -m egolife_two_user_qa build_manifest \
  --output "${OUTDIR}/manifest.json"

python -m egolife_two_user_qa prepare_evidence \
  --manifest "${OUTDIR}/manifest.json" \
  --output "${OUTDIR}/evidence_manifest.jsonl" \
  --cache-dir "${CACHE_DIR}" \
  --output-root "${OUTDIR}" \
  --users-per-case 2 \
  --target-count "${EVIDENCE_TARGET_COUNT}" \
  --frames-per-clip 4 \
  "${calibration_args[@]}"

python -m egolife_two_user_qa generate_video_qa_loop \
  --evidence "${OUTDIR}/evidence_manifest.jsonl" \
  --output "${OUTDIR}/qa_mcq.jsonl" \
  --prompts-output "${OUTDIR}/video_first_prompts.jsonl" \
  --rejected-output "${OUTDIR}/qa_mcq.rejected.jsonl" \
  --intermediate-output "${OUTDIR}/qa_mcq.intermediate.jsonl" \
  --target-count "${TARGET_COUNT}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --backend "${BACKEND}" \
  --base-url "${BASE_URL}" \
  --model-id "${MODEL_ID}" \
  --dtype "${DTYPE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  "${video_input_args[@]}"

python -m egolife_two_user_qa validate_outputs \
  --qa "${OUTDIR}/qa_mcq.jsonl" \
  --csv-output "${OUTDIR}/qa_mcq.csv" \
  --human-review-output "${OUTDIR}/human_review_sheet.md" \
  --report "${OUTDIR}/generation_report.md" \
  --strict-review
