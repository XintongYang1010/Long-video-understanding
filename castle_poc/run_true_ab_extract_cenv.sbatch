#!/bin/bash
#SBATCH --job-name=castle_true_ab
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /scratch/$USER/castle_poc
mkdir -p logs

export PYTHONNOUSERSITE=True
export PATH="/scratch/$USER/castle_poc/bin:/scratch/$USER/castle_poc/cenv/bin:$PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=/scratch/$USER/hf_cache
export HF_HUB_CACHE=/scratch/$USER/hf_cache/hub
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

PY=/scratch/$USER/castle_poc/cenv/bin/python
FF=/scratch/$USER/castle_poc/cenv/bin/ffmpeg

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "USER=$USER"
echo "PY=$PY"
"$PY" --version
echo "FF=$FF"
"$FF" -version | head -n 3

echo "Checking dependencies..."
"$PY" - <<'PY'
from huggingface_hub import hf_hub_url
import pandas, PIL, numpy, tqdm, requests
print("dependency check ok")
PY

echo "Checking required input files..."
ls -lh castle_extract_true_AB_candidates.py
ls -lh castle_main_video_interval_table.csv
ls -lh castle_selected_multiview_windows.csv

echo "Compiling script..."
"$PY" -m py_compile castle_extract_true_AB_candidates.py

echo "Running targeted true A+B extraction..."
"$PY" castle_extract_true_AB_candidates.py \
  --max-local-mb 500 \
  --timeout-sec 45 \
  --ffmpeg "$FF"

echo "Done. Outputs:"
find . -maxdepth 4 -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.csv" -o -name "*.md" -o -name "*.txt" \) | sort | head -300
du -sh candidate_true_AB_contact_sheets 2>/dev/null || true
