#!/bin/bash
#SBATCH --job-name=castle_poc3
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /scratch/$USER/castle_poc
mkdir -p logs

source /scratch/$USER/castle_poc/venv_compute/bin/activate
export PATH="/scratch/$USER/castle_poc/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_CACHE=/scratch/$USER/hf_cache/hub
mkdir -p "$HF_HOME"

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "USER=$USER"
echo "PYTHON=$(which python)"
python --version
echo "FFMPEG=$(which ffmpeg)"
ffmpeg -version | head -n 3 || true

python - <<'PY'
from huggingface_hub import hf_hub_url
import pandas, PIL, numpy, tqdm, imageio_ffmpeg, requests
print("dependency check ok")
PY

python -m py_compile castle_event_relevant_view_selection.py
python -m py_compile castle_incremental_qa_poc.py

python castle_incremental_qa_poc.py \
  --num-windows 3 \
  --output-dir castle_incremental_qa_poc \
  --max-local-mb 500 \
  --skip-clip
