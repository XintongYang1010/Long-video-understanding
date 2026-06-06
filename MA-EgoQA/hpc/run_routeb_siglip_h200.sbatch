#!/usr/bin/env bash
#SBATCH --job-name=routeb_siglip
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xy3257@nyu.edu

set -euo pipefail

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/github_sync_long_video_understanding}"
MAEGOQA_ROOT="${PROJECT_ROOT}/MA-EgoQA"
BENCHMARK_PATH="${ROUTEB_BENCHMARK_PATH:-/scratch/${USER}/ma_egoqa_reproduce/MA-EgoQA/data/MA-EgoQA.json}"
VIDEO_ROOT="${ROUTEB_VIDEO_ROOT:-/scratch/${USER}/data/MaEgo}"
MODEL_ID="${ROUTEB_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
RETRIEVER_MODEL="${ROUTEB_RETRIEVER_MODEL:-google/siglip-base-patch16-224}"
OUTPUT_PATH="${ROUTEB_OUTPUT_PATH:-/scratch/${USER}/data/multiresult/routeB_siglip_day1.json}"
LIMIT="${ROUTEB_LIMIT:-0}"
START_INDEX="${ROUTEB_START_INDEX:-0}"
CANDIDATE_FPS="${ROUTEB_CANDIDATE_FPS:-1}"
TOP_KS="${ROUTEB_TOP_KS:-5 10 15}"
EVAL_MODES="${ROUTEB_EVAL_MODES:-single pair all}"
SAVE_EVERY="${ROUTEB_SAVE_EVERY:-10}"
MAX_NEW_TOKENS="${ROUTEB_MAX_NEW_TOKENS:-16}"
MMR_LAMBDA="${ROUTEB_MMR_LAMBDA:-0.7}"
TEMPORAL_NMS_SEC="${ROUTEB_TEMPORAL_NMS_SEC:-1.0}"
RETRIEVAL_BATCH_SIZE="${ROUTEB_RETRIEVAL_BATCH_SIZE:-32}"
MAX_CANDIDATES_PER_AGENT="${ROUTEB_MAX_CANDIDATES_PER_AGENT:-0}"

cd "${MAEGOQA_ROOT}"
mkdir -p hpc/logs "$(dirname "${OUTPUT_PATH}")" "/scratch/${USER}/hf_cache"

source "${PROJECT_ROOT}/hpc/env_qwen3vl.sh"
export PYTHONPATH="${MAEGOQA_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job_id=${SLURM_JOB_ID:-}"
echo "job_name=${SLURM_JOB_NAME:-}"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "date_start=$(date --iso-8601=seconds)"
echo "python=$(which python)"
python --version
echo "benchmark_path=${BENCHMARK_PATH}"
echo "video_root=${VIDEO_ROOT}"
echo "model_id=${MODEL_ID}"
echo "retriever_model=${RETRIEVER_MODEL}"
echo "output_path=${OUTPUT_PATH}"
echo "limit=${LIMIT}"
echo "top_ks=${TOP_KS}"
echo "eval_modes=${EVAL_MODES}"
echo "candidate_fps=${CANDIDATE_FPS}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

python - <<'PY'
import torch
print("preflight_cuda_available=", torch.cuda.is_available(), flush=True)
print("preflight_cuda_device_count=", torch.cuda.device_count(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable in Slurm job")
if torch.cuda.device_count() < 1:
    raise SystemExit("Expected at least 1 visible GPU")
PY

read -r -a TOP_K_ARGS <<< "${TOP_KS}"
read -r -a EVAL_MODE_ARGS <<< "${EVAL_MODES}"

args=(
  python -m egomas.src.evaluate_day1_qwen3vl_routeb_retrieval
  --benchmark-path "${BENCHMARK_PATH}"
  --video-root "${VIDEO_ROOT}"
  --output-path "${OUTPUT_PATH}"
  --model-name "${MODEL_ID}"
  --retriever-model "${RETRIEVER_MODEL}"
  --candidate-fps "${CANDIDATE_FPS}"
  --top-ks "${TOP_K_ARGS[@]}"
  --eval-modes "${EVAL_MODE_ARGS[@]}"
  --limit "${LIMIT}"
  --start-index "${START_INDEX}"
  --save-every "${SAVE_EVERY}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --mmr-lambda "${MMR_LAMBDA}"
  --temporal-nms-sec "${TEMPORAL_NMS_SEC}"
  --retrieval-batch-size "${RETRIEVAL_BATCH_SIZE}"
  --max-candidates-per-agent "${MAX_CANDIDATES_PER_AGENT}"
)

echo "running: ${args[*]}"
"${args[@]}"

echo "date_done=$(date --iso-8601=seconds)"
