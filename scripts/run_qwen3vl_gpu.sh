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
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "${OUTDIR}" "${CACHE_DIR}"

python -m egolife_two_user_qa build_manifest \
  --output "${OUTDIR}/manifest.json"

python -m egolife_two_user_qa prepare_evidence \
  --manifest "${OUTDIR}/manifest.json" \
  --output "${OUTDIR}/evidence_manifest.jsonl" \
  --cache-dir "${CACHE_DIR}" \
  --output-root "${OUTDIR}" \
  --target-count "${TARGET_COUNT}" \
  --users-per-case 2 \
  --frames-per-clip 3

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
