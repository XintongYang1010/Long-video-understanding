#!/usr/bin/env bash
#SBATCH --job-name=egoeverything-vqa
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/xy3257/EgoEverything_repro_20260619/logs/vqa-%j.out
#SBATCH --error=/scratch/xy3257/EgoEverything_repro_20260619/logs/vqa-%j.err

set -euo pipefail

BASE=/scratch/xy3257/EgoEverything_repro_20260619
source "$BASE/env.sh"

: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before submitting this job.}"

if [[ ! -f "$DATASET_JSON" ]]; then
  echo "Missing DATASET_JSON: $DATASET_JSON" >&2
  echo "Place the AEA download-links JSON there, or export DATASET_JSON=/path/to/file.json." >&2
  exit 2
fi

if ! find "$DATASET_PATH" -mindepth 2 -maxdepth 2 -name '*.mp4' -print -quit | grep -q .; then
  echo "No processed videos found under DATASET_PATH: $DATASET_PATH" >&2
  echo "Run scripts/download_aea_one.sh after providing the AEA download-links JSON." >&2
  exit 2
fi

if ! find "$DATASET_PATH" -mindepth 2 -maxdepth 2 -name '*_summary.json' -print -quit | grep -q .; then
  echo "No summary JSON files found under DATASET_PATH: $DATASET_PATH" >&2
  echo "Run scripts/run_video_summarizer_one.sbatch first, or provide precomputed sequence summaries." >&2
  exit 2
fi

python Generation/main.py \
  --api-key "$OPENROUTER_API_KEY" \
  --dataset-path "$DATASET_PATH" \
  --json-path "$DATASET_JSON" \
  --dataset-name "$DATASET_NAME" \
  --limit "${LIMIT:-1}" \
  --question-factor "${QUESTION_FACTOR:-1}" \
  --sampling-density "${SAMPLING_DENSITY:-5}" \
  --n-llms "${N_LLMS:-1}" \
  --qa-n-llms "${QA_N_LLMS:-1}" \
  --output-path "$EGOEVERYTHING_BASE/${DATASET_NAME}_vqa_limit${LIMIT:-1}.json" \
  --temp-dir "$EGOEVERYTHING_BASE/tmp_vqa" \
  --verbose
