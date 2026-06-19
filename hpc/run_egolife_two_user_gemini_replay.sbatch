#!/usr/bin/env bash
#SBATCH --job-name=egolife_gemini_replay
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xy3257@nyu.edu

set -euo pipefail

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/github_sync_long_video_understanding}"
SOURCE_DIR="${EGOLIFE2U_REPLAY_SOURCE_DIR:-/scratch/${USER}/egolife_two_user_qa_outputs/day1_day2/visual_truth_sameplace_qa10_20260617_045313}"
OUTDIR="${EGOLIFE2U_OUTPUT_ROOT:-/scratch/${USER}/egolife_two_user_qa_outputs/day1_day2/gemini25pro_replay_10994195_$(date +%Y%m%d_%H%M%S)}"
MODEL_ID="${EGOLIFE2U_MODEL_ID:-gemini-2.5-pro}"
MAX_NEW_TOKENS="${EGOLIFE2U_MAX_NEW_TOKENS:-1536}"
LIMIT="${EGOLIFE2U_REPLAY_LIMIT:-0}"
SKIP_JUDGE="${EGOLIFE2U_SKIP_JUDGE:-0}"

if [[ -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
  echo "ERROR: set GEMINI_API_KEY or GOOGLE_API_KEY before submitting this job." >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
mkdir -p hpc/logs "${OUTDIR}"

source "${PROJECT_ROOT}/hpc/env_qwen3vl.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "job_id=${SLURM_JOB_ID:-}"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "date_start=$(date --iso-8601=seconds)"
echo "model_id=${MODEL_ID}"
echo "source_dir=${SOURCE_DIR}"
echo "outdir=${OUTDIR}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "limit=${LIMIT}"
echo "skip_judge=${SKIP_JUDGE}"
echo "gemini_key_visible=yes"
python --version

run_args=(
  python scripts/replay_video_first_prompts_gemini.py
  --source-qa "${SOURCE_DIR}/qa_mcq.jsonl"
  --source-prompts "${SOURCE_DIR}/video_first_prompts.jsonl"
  --evidence "${SOURCE_DIR}/evidence_manifest.jsonl"
  --output-dir "${OUTDIR}"
  --model-id "${MODEL_ID}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
)

if [[ "${LIMIT}" != "0" ]]; then
  run_args+=(--limit "${LIMIT}")
fi
if [[ "${SKIP_JUDGE}" == "1" ]]; then
  run_args+=(--skip-judge)
fi

"${run_args[@]}"

echo "date_done=$(date --iso-8601=seconds)"
