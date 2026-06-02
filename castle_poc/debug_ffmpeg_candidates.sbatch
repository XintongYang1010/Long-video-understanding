#!/bin/bash
#SBATCH --job-name=debug_ffmpeg_bins
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
mkdir -p debug_ffmpeg_bins logs

URL="https://huggingface.co/datasets/CASTLE-Dataset/CASTLE2024/resolve/main/main/day1/Allie/video/10.mp4"

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "URL=$URL"

echo ""
echo "===== MODULE AVAIL FFMPEG ====="
module avail ffmpeg 2>&1 || true
module spider ffmpeg 2>&1 || true

echo ""
echo "===== TYPE -A FFMPEG BEFORE PATH CHANGE ====="
type -a ffmpeg || true

echo ""
echo "===== CANDIDATE BINARIES ====="

CANDIDATES=()

for p in \
/usr/bin/ffmpeg \
  /usr/local/bin/ffmpeg \
  /share/apps/ffmpeg/bin/ffmpeg \
  /share/apps/ffmpeg*/bin/ffmpeg \
  /opt/ffmpeg/bin/ffmpeg \
  /scratch/$USER/castle_poc/bin/ffmpeg
do
  if [ -x "$p" ]; then
    CANDIDATES+=("$p")
  fi
done

printf '%s\n' "${CANDIDATES[@]}" || true

echo ""
echo "===== CURL RANGE CONTROL TEST ====="
curl -L --max-time 60 -r 0-1048575 -o debug_ffmpeg_bins/allie_1mb_probe.bin "$URL" \
  -w "http_code=%{http_code} size_download=%{size_download} time_total=%{time_total}\n" || true
ls -lh debug_ffmpeg_bins/allie_1mb_probe.bin || true

echo ""
echo "===== TEST EACH FFMPEG CANDIDATE ====="

set +e

for BIN in "${CANDIDATES[@]}"; do
  SAFE_NAME=$(echo "$BIN" | sed 's#[/ ]#_#g')
  OUT="debug_ffmpeg_bins/allie_${SAFE_NAME}.jpg"

  echo ""
  echo "============================================================"
  echo "BIN=$BIN"
  "$BIN" -version | head -n 3 || true

  echo "Running extraction..."
  timeout 180 "$BIN" \
    -hide_banner -loglevel info -nostdin -y \
-rw_timeout 120000000 \
    -reconnect 1 \
    -reconnect_on_network_error 1 \
    -reconnect_streamed 1 \
    -reconnect_delay_max 5 \
    -ss 360.000 \
    -i "$URL" \
    -frames:v 1 -q:v 2 \
    "$OUT"

  RC=$?
  echo "RETURN_CODE=$RC"
  ls -lh "$OUT" || true
done

echo ""
echo "===== FINAL OUTPUT FILES ====="
find debug_ffmpeg_bins -maxdepth 1 -type f -ls

