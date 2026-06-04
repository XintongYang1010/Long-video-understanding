#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_maegoqa1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xy3257@nyu.edu

set -euo pipefail

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/github_sync_long_video_understanding}"
MAEGOQA_ROOT="${PROJECT_ROOT}/MA-EgoQA"
DATA_ROOT="${PROJECT_ROOT}/ma_egoqa_reproduce/MA-EgoQA/data"
BENCHMARK_PATH="${QWEN3VL_BENCHMARK_PATH:-${DATA_ROOT}/MA-EgoQA.json}"
BM25_PATH="${QWEN3VL_BM25_PATH:-${DATA_ROOT}/30sec_bm25.pkl}"
MODEL_ID="${QWEN3VL_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
LIMIT="${QWEN3VL_LIMIT:-100}"
MAX_NEW_TOKENS="${QWEN3VL_MAX_NEW_TOKENS:-16}"
OUTPUT_ROOT="${QWEN3VL_OUTPUT_ROOT:-outputs/qwen3vl_subset}"

if [[ -n "${QWEN3VL_CONDITIONS:-}" ]]; then
  read -r -a CONDITIONS <<< "${QWEN3VL_CONDITIONS}"
else
  CONDITIONS=(Lucia Tasha Shure Lucia_Tasha Lucia_Shure Tasha_Shure)
fi

cd "${MAEGOQA_ROOT}"
mkdir -p hpc/logs "${OUTPUT_ROOT}" "/scratch/${USER}/hf_cache"

source "${PROJECT_ROOT}/hpc/env_qwen3vl.sh"
export PYTHONPATH="${MAEGOQA_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS=8

echo "job_id=${SLURM_JOB_ID:-}"
echo "job_name=${SLURM_JOB_NAME:-}"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "user=${USER}"
echo "date_start=$(date --iso-8601=seconds)"
echo "python=$(which python)"
python --version
echo "benchmark_path=${BENCHMARK_PATH}"
echo "bm25_path=${BM25_PATH}"
echo "model_id=${MODEL_ID}"
echo "limit=${LIMIT}"
echo "output_root=${OUTPUT_ROOT}"
echo "conditions=${CONDITIONS[*]}"
echo "slurm_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
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

ORIGINAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
IFS=',' read -r -a ALLOCATED_GPUS <<< "${ORIGINAL_CUDA_VISIBLE_DEVICES}"
if (( ${#ALLOCATED_GPUS[@]} < 1 )); then
  echo "Need at least 1 GPU, but CUDA_VISIBLE_DEVICES=${ORIGINAL_CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

status=0
gpu_slots=${#ALLOCATED_GPUS[@]}
echo "gpu_slots=${gpu_slots}"

for ((batch_start=0; batch_start<${#CONDITIONS[@]}; batch_start+=gpu_slots)); do
  batch_end=$((batch_start + gpu_slots - 1))
  if (( batch_end >= ${#CONDITIONS[@]} )); then
    batch_end=$((${#CONDITIONS[@]} - 1))
  fi
  echo "batch_start=${batch_start} batch_end=${batch_end}"

  pids=()
  batch_conditions=()
  for ((offset=0; offset<gpu_slots; offset++)); do
    idx=$((batch_start + offset))
    if (( idx >= ${#CONDITIONS[@]} )); then
      break
    fi
    condition="${CONDITIONS[$idx]}"
    gpu_id="${ALLOCATED_GPUS[$offset]}"
    condition_output_dir="${OUTPUT_ROOT}/${condition}"
    mkdir -p "${condition_output_dir}"

    args=(
      python -m egomas.src.inference_qwen3vl_subset
      --condition "${condition}"
      --benchmark-path "${BENCHMARK_PATH}"
      --bm25-path "${BM25_PATH}"
      --model-id "${MODEL_ID}"
      --output-dir "${condition_output_dir}"
      --max-new-tokens "${MAX_NEW_TOKENS}"
      --overwrite
    )
    if [[ -n "${LIMIT}" && "${LIMIT}" != "none" && "${LIMIT}" != "None" ]]; then
      args+=(--limit "${LIMIT}")
    fi

    echo "launch condition=${condition} gpu=${gpu_id} output=${condition_output_dir}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      export OMP_NUM_THREADS=8
      "${args[@]}"
    ) > "${condition_output_dir}/run.log" 2>&1 &
    pids[$offset]=$!
    batch_conditions[$offset]="${condition}"
  done

  for offset in "${!pids[@]}"; do
    condition="${batch_conditions[$offset]}"
    if wait "${pids[$offset]}"; then
      echo "condition_done=${condition}"
    else
      child_status=$?
      echo "condition_failed=${condition} status=${child_status}" >&2
      status=1
    fi
  done
done

python -m egomas.src.summarize_qwen3vl_subset --output-root "${OUTPUT_ROOT}" || status=1

echo "date_done=$(date --iso-8601=seconds)"
exit "${status}"
