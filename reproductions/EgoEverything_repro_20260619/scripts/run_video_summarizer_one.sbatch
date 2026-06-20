#!/usr/bin/env bash
#SBATCH --job-name=egoeverything-summary
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/xy3257/EgoEverything_repro_20260619/logs/video_summarizer-%j.out
#SBATCH --error=/scratch/xy3257/EgoEverything_repro_20260619/logs/video_summarizer-%j.err

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

python video_summarizer.py \
  --api-key "$OPENROUTER_API_KEY" \
  --dataset-path "$DATASET_PATH" \
  --json-path "$DATASET_JSON" \
  --limit "${LIMIT:-1}" \
  --n-clusters "${N_CLUSTERS:-12}" \
  --n-LLMs "${N_LLMS:-1}"
