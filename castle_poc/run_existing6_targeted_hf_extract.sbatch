#!/usr/bin/env bash
#SBATCH --job-name=ex6_hf_frames
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --account=torch_pr_674_tandon_advanced

#SBATCH --output=logs/ex6_hf_frames_%j.out
#SBATCH --error=logs/ex6_hf_frames_%j.err
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -euo pipefail

cd /scratch/xy3257/castle_poc
mkdir -p logs

echo "hostname=$(hostname)"
echo "start=$(date)"
echo "pwd=$(pwd)"

export PATH="/scratch/xy3257/castle_poc/cenv/bin:/scratch/xy3257/castle_poc/bin:$PATH"
export FFMPEG_BIN="${FFMPEG_BIN:-/scratch/xy3257/castle_poc/cenv/bin/ffmpeg}"

echo "FFMPEG_BIN=${FFMPEG_BIN}"
"${FFMPEG_BIN}" -version | head -2 || true

PROJECT_ROOT=/scratch/xy3257/castle_poc \
OUTPUT_ROOT=/scratch/xy3257/castle_poc/existing6_targeted_frames_hf \
FFMPEG_BIN="${FFMPEG_BIN}" \
bash scripts/hpc_extract_existing6_targeted_frames.sh --force

echo "done=$(date)"
echo "final_jpg_count:"
find /scratch/xy3257/castle_poc/existing6_targeted_frames_hf -type f -name '*.jpg' | wc -l
