#!/usr/bin/env bash
set -euo pipefail

export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}}"
export PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-${SCRATCH_ROOT}/github_sync_long_video_understanding}"
export CONDA_ROOT="${CONDA_ROOT:-${SCRATCH_ROOT}/miniconda3}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-qwen3vl-smoke}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export HF_HOME="${SCRATCH_ROOT}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=True
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HUB_ENABLE_HF_TRANSFER=0

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}"
