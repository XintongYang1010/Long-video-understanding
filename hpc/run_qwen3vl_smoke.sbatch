#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_smoke
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200|h100
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xy3257@nyu.edu

set -euo pipefail

cd "/scratch/${USER}/github_sync_long_video_understanding"
mkdir -p hpc/logs hpc/smoke_outputs "/scratch/${USER}/hf_cache"

source hpc/env_qwen3vl.sh

echo "job_id=${SLURM_JOB_ID:-}"
echo "job_name=${SLURM_JOB_NAME:-}"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "user=${USER}"
echo "date_start=$(date --iso-8601=seconds)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
echo "python=$(which python)"
python --version

nvidia-smi || true

python - <<'PY'
import torch
print("preflight_cuda_available=", torch.cuda.is_available(), flush=True)
print("preflight_cuda_device_count=", torch.cuda.device_count(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable in Slurm job")
PY

python hpc/qwen3vl_smoke_service.py \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --max-new-tokens 64 \
  --output-dir hpc/smoke_outputs

echo "date_done=$(date --iso-8601=seconds)"
