#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_castle_frames
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xy3257@nyu.edu

set -euo pipefail

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/github_sync_long_video_understanding}"
OUTPUT_ROOT="${QWEN3VL_OUTPUT_ROOT:-outputs/castle_frame_eval}"
MANIFEST_PATH="${QWEN3VL_MANIFEST_PATH:-${OUTPUT_ROOT}/frame_manifest.json}"
MODEL_ID="${QWEN3VL_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
LIMIT="${QWEN3VL_LIMIT:-5}"
MAX_FRAMES_PER_SOURCE="${QWEN3VL_MAX_FRAMES_PER_SOURCE:-1}"
MAX_TOTAL_MULTI_FRAMES="${QWEN3VL_MAX_TOTAL_MULTI_FRAMES:-8}"
MAX_IMAGE_PIXELS="${QWEN3VL_MAX_IMAGE_PIXELS:-262144}"
MAX_NEW_TOKENS="${QWEN3VL_MAX_NEW_TOKENS:-16}"

cd "${PROJECT_ROOT}"
mkdir -p hpc/logs "${OUTPUT_ROOT}" "/scratch/${USER}/hf_cache"

source "${PROJECT_ROOT}/hpc/env_qwen3vl.sh"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job_id=${SLURM_JOB_ID:-}"
echo "job_name=${SLURM_JOB_NAME:-}"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "date_start=$(date --iso-8601=seconds)"
echo "python=$(which python)"
python --version
echo "model_id=${MODEL_ID}"
echo "limit=${LIMIT}"
echo "max_frames_per_source=${MAX_FRAMES_PER_SOURCE}"
echo "max_total_multi_frames=${MAX_TOTAL_MULTI_FRAMES}"
echo "max_image_pixels=${MAX_IMAGE_PIXELS}"
echo "output_root=${OUTPUT_ROOT}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

python - <<'PY'
import torch
print("preflight_cuda_available=", torch.cuda.is_available(), flush=True)
print("preflight_cuda_device_count=", torch.cuda.device_count(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable in Slurm job")
PY

python castle_qwen_eval/prepare_castle_official_frame_manifest.py \
  --project-root "${PROJECT_ROOT}" \
  --official-qa-path EgoVis2026_CVPR_Questions.json \
  --output "${MANIFEST_PATH}"

run_args=(
  python castle_qwen_eval/run_qwen3vl_frame_qa.py
  --manifest "${MANIFEST_PATH}"
  --output-root "${OUTPUT_ROOT}"
  --model-id "${MODEL_ID}"
  --max-frames-per-source "${MAX_FRAMES_PER_SOURCE}"
  --max-total-multi-frames "${MAX_TOTAL_MULTI_FRAMES}"
  --max-image-pixels "${MAX_IMAGE_PIXELS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --condition all
  --overwrite
)

if [[ -n "${LIMIT}" && "${LIMIT}" != "none" && "${LIMIT}" != "None" ]]; then
  run_args+=(--limit "${LIMIT}")
fi

"${run_args[@]}"

python castle_qwen_eval/summarize_frame_qa.py --output-root "${OUTPUT_ROOT}"

echo "date_done=$(date --iso-8601=seconds)"
