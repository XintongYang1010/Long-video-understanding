#!/usr/bin/env bash
set -euo pipefail

TARGET_COUNT=20
MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
DTYPE="bfloat16"
MAX_NEW_TOKENS=1536
BACKEND="transformers-local"
BASE_URL="http://127.0.0.1:8000/v1"
OUTDIR="egolife_two_user_qa/outputs/pilot_20"
CACHE_DIR=".cache/egolife_two_user_qa"
TARGET_CLIP_COUNT=""
MAX_TIME_GAP_SECONDS=90
MIN_COMPLEMENTARITY_SCORE=5
ARIA_CALIBRATION_DIR=""

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
    --target-clip-count) TARGET_CLIP_COUNT="$2"; shift 2 ;;
    --max-time-gap-seconds) MAX_TIME_GAP_SECONDS="$2"; shift 2 ;;
    --min-complementarity-score) MIN_COMPLEMENTARITY_SCORE="$2"; shift 2 ;;
    --aria-calibration-dir) ARIA_CALIBRATION_DIR="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${OUTDIR}" "${CACHE_DIR}"
if [[ -z "${TARGET_CLIP_COUNT}" ]]; then
  TARGET_CLIP_COUNT=$((TARGET_COUNT * 8))
fi
calibration_args=()
if [[ -n "${ARIA_CALIBRATION_DIR}" ]]; then
  calibration_args=(--aria-calibration-dir "${ARIA_CALIBRATION_DIR}")
fi

python -m egolife_two_user_qa build_manifest \
  --output "${OUTDIR}/manifest.json"

python -m egolife_two_user_qa observe_clips \
  --manifest "${OUTDIR}/manifest.json" \
  --output "${OUTDIR}/observations.jsonl" \
  --prompts-output "${OUTDIR}/observation_prompts.jsonl" \
  --cache-dir "${CACHE_DIR}" \
  --output-root "${OUTDIR}" \
  --target-clip-count "${TARGET_CLIP_COUNT}" \
  --frames-per-clip 4 \
  "${calibration_args[@]}" \
  --backend "${BACKEND}" \
  --base-url "${BASE_URL}" \
  --model-id "${MODEL_ID}" \
  --dtype "${DTYPE}" \
  --max-new-tokens 768

python -m egolife_two_user_qa mine_candidates \
  --observations "${OUTDIR}/observations.jsonl" \
  --output "${OUTDIR}/evidence_manifest.jsonl" \
  --target-count "${TARGET_COUNT}" \
  --users-per-case 2 \
  --max-time-gap-seconds "${MAX_TIME_GAP_SECONDS}" \
  --min-score "${MIN_COMPLEMENTARITY_SCORE}"

python -m egolife_two_user_qa generate_qa \
  --evidence "${OUTDIR}/evidence_manifest.jsonl" \
  --output "${OUTDIR}/qa_mcq.raw.jsonl" \
  --prompts-output "${OUTDIR}/generation_prompts.jsonl" \
  --target-count "${TARGET_COUNT}" \
  --backend "${BACKEND}" \
  --base-url "${BASE_URL}" \
  --model-id "${MODEL_ID}" \
  --dtype "${DTYPE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"

python -m egolife_two_user_qa review_qa \
  --qa "${OUTDIR}/qa_mcq.raw.jsonl" \
  --evidence "${OUTDIR}/evidence_manifest.jsonl" \
  --output "${OUTDIR}/qa_mcq.jsonl" \
  --backend "${BACKEND}" \
  --base-url "${BASE_URL}" \
  --model-id "${MODEL_ID}" \
  --dtype "${DTYPE}" \
  --max-new-tokens 768

python -m egolife_two_user_qa validate_outputs \
  --qa "${OUTDIR}/qa_mcq.jsonl" \
  --csv-output "${OUTDIR}/qa_mcq.csv" \
  --report "${OUTDIR}/generation_report.md" \
  --strict-review
