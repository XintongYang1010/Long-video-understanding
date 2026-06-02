#!/bin/bash
#SBATCH --job-name=castle_ab_r2
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

echo "Checking required files..."
ls -lh castle_extract_true_AB_candidates_round2.py
ls -lh castle_main_video_interval_table.csv
ls -lh castle_selected_multiview_windows.csv
ls -lh round2_selected_windows.txt

echo "Compile check..."
"$PY" -m py_compile castle_extract_true_AB_candidates_round2.py

echo "Running round2 targeted extraction..."
"$PY" castle_extract_true_AB_candidates_round2.py \
  --max-local-mb 500 \
  --timeout-sec 45 \
  --ffmpeg "$FF"

echo "Done. Outputs:"
find candidate_true_AB_contact_sheets_round2 -maxdepth 4 -type f \( -name "*.png" -o -name "*.jpg" \) | sort | head -300
ls -lh candidate_true_AB_cases_round2.csv true_AB_search_summary_round2.md 2>/dev/null || true
du -sh candidate_true_AB_contact_sheets_round2 2>/dev/null || true
