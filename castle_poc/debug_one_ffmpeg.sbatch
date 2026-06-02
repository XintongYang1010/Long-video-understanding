#!/bin/bash
#SBATCH --job-name=debug_ffmpeg
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
mkdir -p debug_ffmpeg logs

source /scratch/$USER/castle_poc/venv_compute/bin/activate
export PATH="/scratch/$USER/castle_poc/bin:$PATH"

URL="https://huggingface.co/datasets/CASTLE-Dataset/CASTLE2024/resolve/main/main/day1/Allie/video/10.mp4"

echo "HOSTNAME=$(hostname)"
echo "PYTHON=$(which python)"
python --version
echo "FFMPEG=$(which ffmpeg)"
ffmpeg -version | head -n 5

echo ""
echo ""
echo "===== CURL HEAD TEST ====="
curl -I -L --max-time 60 "$URL" | head -80 || true

echo ""
echo "===== CURL RANGE TEST ====="
curl -L --max-time 60 -r 0-1048575 -o debug_ffmpeg/allie_1mb_probe.bin "$URL" \
  -w "http_code=%{http_code} size_download=%{size_download} time_total=%{time_total}\n" || true
ls -lh debug_ffmpeg/allie_1mb_probe.bin || true

echo ""
echo "===== TEST A: original-ish command, but loglevel info ====="
set +e
timeout 180 ffmpeg \
  -hide_banner -loglevel info -nostdin -y \
  -rw_timeout 120000000 \
  -reconnect 1 \
  -reconnect_on_network_error 1 \
  -reconnect_streamed 1 \
  -reconnect_delay_max 5 \
  -multiple_requests 1 \
  -seekable 1 \
  -ss 360.000 \
  -i "$URL" \
  -frames:v 1 -q:v 2 \
  debug_ffmpeg/allie_A.jpg
echo "TEST_A_RETURN=$?"
ls -lh debug_ffmpeg/allie_A.jpg || true
echo ""
echo "===== TEST B: simplified remote ffmpeg, no multiple_requests/seekable ====="
timeout 180 ffmpeg \
  -hide_banner -loglevel info -nostdin -y \
  -rw_timeout 120000000 \
  -reconnect 1 \
  -reconnect_on_network_error 1 \
  -reconnect_streamed 1 \
  -reconnect_delay_max 5 \
  -ss 360.000 \
  -i "$URL" \
  -frames:v 1 -q:v 2 \
  debug_ffmpeg/allie_B.jpg
echo "TEST_B_RETURN=$?"
ls -lh debug_ffmpeg/allie_B.jpg || true

echo ""
echo "===== TEST C: input first, seek after input; slower but safer ====="
timeout 240 ffmpeg \
  -hide_banner -loglevel info -nostdin -y \
  -rw_timeout 120000000 \
  -reconnect 1 \
  -reconnect_on_network_error 1 \
  -reconnect_streamed 1 \
  -reconnect_delay_max 5 \
  -i "$URL" \
  -ss 360.000 \
  -frames:v 1 -q:v 2 \
debug_ffmpeg/allie_C.jpg
echo "TEST_C_RETURN=$?"
ls -lh debug_ffmpeg/allie_C.jpg || true

echo ""
echo "===== OUTPUT FILES ====="
find debug_ffmpeg -maxdepth 1 -type f -ls

