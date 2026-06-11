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
OUTDIR="${EGOLIFE2U_OUTPUT_ROOT:-egolife_two_user_qa/outputs/pilot_20_video_first}"
CACHE_DIR="${EGOLIFE2U_CACHE_DIR:-/scratch/${USER}/egolife_two_user_qa_cache}"
EVIDENCE_TARGET_COUNT="${EGOLIFE2U_EVIDENCE_TARGET_COUNT:-}"
MAX_ATTEMPTS="${EGOLIFE2U_MAX_ATTEMPTS:-3}"
ARIA_CALIBRATION_DIR="${EGOLIFE2U_ARIA_CALIBRATION_DIR:-}"
ALLOW_OPENAI_VIDEO_INPUT="${EGOLIFE2U_ALLOW_OPENAI_VIDEO_INPUT:-0}"

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
  --max-attempts "${MAX_ATTEMPTS}"
)
if [[ -n "${EVIDENCE_TARGET_COUNT}" ]]; then
  run_args+=(--evidence-target-count "${EVIDENCE_TARGET_COUNT}")
fi
if [[ -n "${ARIA_CALIBRATION_DIR}" ]]; then
  run_args+=(--aria-calibration-dir "${ARIA_CALIBRATION_DIR}")
fi
if [[ "${ALLOW_OPENAI_VIDEO_INPUT}" == "1" ]]; then
  run_args+=(--allow-openai-video-input)
fi

"${run_args[@]}"

echo "date_done=$(date --iso-8601=seconds)"
