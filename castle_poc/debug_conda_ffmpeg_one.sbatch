#!/bin/bash
#SBATCH --job-name=debug_conda_ffmpeg
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /scratch/$USER/castle_poc
mkdir -p debug_conda_ffmpeg logs

FF=/scratch/$USER/castle_poc/cenv/bin/ffmpeg
URL="https://huggingface.co/datasets/CASTLE-Dataset/CASTLE2024/resolve/main/main/day1/Allie/video/10.mp4"

echo "HOSTNAME=$(hostname)"
echo "FF=$FF"
"$FF" -version | head -n 5

echo ""
echo "===== CURL RANGE CONTROL ====="
curl -L --max-time 60 -r 0-1048575 -o debug_conda_ffmpeg/allie_1mb_probe.bin "$URL" \
  -w "http_code=%{http_code} size_download=%{size_download} time_total=%{time_total}\n" || true
ls -lh debug_conda_ffmpeg/allie_1mb_probe.bin || true

echo ""
echo "===== CONDA FFMPEG EXTRACTION TEST ====="
set +e
timeout 240 "$FF" \
  -hide_banner -loglevel info -nostdin -y \
  -rw_timeout 120000000 \
  -reconnect 1 \
  -reconnect_on_network_error 1 \
  -reconnect_streamed 1 \
  -reconnect_delay_max 5 \
  -ss 360.000 \
  -i "$URL" \
  -frames:v 1 -q:v 2 \
  debug_conda_ffmpeg/allie_conda.jpg

RC=$?
echo "RETURN_CODE=$RC"
ls -lh debug_conda_ffmpeg/allie_conda.jpg || true

echo ""
echo "===== OUTPUT FILES ====="
find debug_conda_ffmpeg -maxdepth 1 -type f -ls
