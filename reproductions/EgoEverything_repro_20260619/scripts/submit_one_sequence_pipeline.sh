#!/usr/bin/env bash
set -euo pipefail

BASE="${EGOEVERYTHING_BASE:-/scratch/xy3257/EgoEverything_repro_20260619}"
source "$BASE/env.sh"

: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY before submitting summary/VQA jobs.}"

export LIMIT="${LIMIT:-1}"
export N_CLUSTERS="${N_CLUSTERS:-12}"
export N_LLMS="${N_LLMS:-1}"
export QA_N_LLMS="${QA_N_LLMS:-1}"
export QUESTION_FACTOR="${QUESTION_FACTOR:-1}"
export SAMPLING_DENSITY="${SAMPLING_DENSITY:-5}"

if [[ ! -f "$DATASET_JSON" ]]; then
  echo "Missing DATASET_JSON: $DATASET_JSON" >&2
  echo "Run REFRESH_AEA_MANIFEST=1 sbatch $BASE/scripts/run_download_aea_one.sbatch first." >&2
  exit 2
fi

if ! find "$DATASET_PATH" -mindepth 2 -maxdepth 2 -name '*.mp4' -print -quit | grep -q .; then
  echo "No processed videos found under DATASET_PATH: $DATASET_PATH" >&2
  echo "Run REFRESH_AEA_MANIFEST=1 sbatch $BASE/scripts/run_download_aea_one.sbatch first." >&2
  exit 2
fi

summary_job=""
if ! find "$DATASET_PATH" -mindepth 2 -maxdepth 2 -name '*_summary.json' -print -quit | grep -q .; then
  summary_job="$(sbatch --parsable --export=ALL "$BASE/scripts/run_video_summarizer_one.sbatch")"
  echo "Submitted summary job: $summary_job"
fi

if [[ -n "$summary_job" ]]; then
  vqa_job="$(sbatch --parsable --dependency="afterok:$summary_job" --export=ALL "$BASE/scripts/run_vqa_one.sbatch")"
else
  vqa_job="$(sbatch --parsable --export=ALL "$BASE/scripts/run_vqa_one.sbatch")"
fi

echo "Submitted VQA job: $vqa_job"
