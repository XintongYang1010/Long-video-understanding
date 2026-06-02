#!/bin/bash
#SBATCH --job-name=setup_ffmpeg
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /scratch/$USER/castle_poc
mkdir -p logs

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"

MINIFORGE=/scratch/$USER/miniforge3
ENV=/scratch/$USER/castle_poc/cenv

if [ ! -d "$MINIFORGE" ]; then
  echo "Installing Miniforge..."
  curl -L -o Miniforge3.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3.sh -b -p "$MINIFORGE"
else
  echo "Miniforge already exists: $MINIFORGE"
fi

source "$MINIFORGE/etc/profile.d/conda.sh"

if [ ! -d "$ENV" ]; then
  echo "Creating conda env..."
  conda create -p "$ENV" -y -c conda-forge \
    python=3.10 ffmpeg=6.1 pandas pillow numpy tqdm huggingface_hub requests
else
  echo "Conda env already exists: $ENV"
fi

"$ENV/bin/python" - <<'PY'
from huggingface_hub import hf_hub_url
import pandas, PIL, numpy, tqdm, requests
print("python deps ok")
PY

"$ENV/bin/ffmpeg" -version | head -n 5

echo "SETUP_DONE"
