#!/usr/bin/env bash
#SBATCH --job-name=egolife_2u_qwen3vl
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xy3257@nyu.edu

set -euo pipefail

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/github_sync_long_video_understanding}"
TARGET_COUNT="${EGOLIFE2U_TARGET_COUNT:-20}"
MODEL_ID="${EGOLIFE2U_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
DTYPE="${EGOLIFE2U_DTYPE:-bfloat16}"
MAX_NEW_TOKENS="${EGOLIFE2U_MAX_NEW_TOKENS:-1536}"
OUTDIR="${EGOLIFE2U_OUTPUT_ROOT:-egolife_two_user_qa/outputs/pilot_20}"
CACHE_DIR="${EGOLIFE2U_CACHE_DIR:-/scratch/${USER}/egolife_two_user_qa_cache}"
TARGET_CLIP_COUNT="${EGOLIFE2U_TARGET_CLIP_COUNT:-}"
MAX_TIME_GAP_SECONDS="${EGOLIFE2U_MAX_TIME_GAP_SECONDS:-90}"
MIN_COMPLEMENTARITY_SCORE="${EGOLIFE2U_MIN_COMPLEMENTARITY_SCORE:-5}"

cd "${PROJECT_ROOT}"
mkdir -p hpc/logs "${OUTDIR}" "${CACHE_DIR}" "/scratch/${USER}/hf_cache"

source "${PROJECT_ROOT}/hpc/env_qwen3vl.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job_id=${SLURM_JOB_ID:-}"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "date_start=$(date --iso-8601=seconds)"
echo "model_id=${MODEL_ID}"
echo "target_count=${TARGET_COUNT}"
python --version
nvidia-smi || true

run_args=(
  bash scripts/run_qwen3vl_gpu.sh
  --target-count "${TARGET_COUNT}"
  --model-id "${MODEL_ID}"
  --dtype "${DTYPE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --outdir "${OUTDIR}"
  --cache-dir "${CACHE_DIR}"
  --max-time-gap-seconds "${MAX_TIME_GAP_SECONDS}"
  --min-complementarity-score "${MIN_COMPLEMENTARITY_SCORE}"
)
if [[ -n "${TARGET_CLIP_COUNT}" ]]; then
  run_args+=(--target-clip-count "${TARGET_CLIP_COUNT}")
fi

"${run_args[@]}"

echo "date_done=$(date --iso-8601=seconds)"
