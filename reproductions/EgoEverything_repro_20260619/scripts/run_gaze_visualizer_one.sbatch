#!/usr/bin/env bash
#SBATCH --job-name=egoeverything-gaze
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xy3257/EgoEverything_repro_20260619/logs/gaze_visualizer-%j.out
#SBATCH --error=/scratch/xy3257/EgoEverything_repro_20260619/logs/gaze_visualizer-%j.err

set -euo pipefail

BASE=/scratch/xy3257/EgoEverything_repro_20260619
source "$BASE/env.sh"

SEQUENCE_ID="${SEQUENCE_ID:-loc5_script4_seq6_rec1}"
SEQ_DIR="$DATASET_PATH/$SEQUENCE_ID"
VIDEO="$SEQ_DIR/$SEQUENCE_ID.mp4"
TRACKING="$SEQ_DIR/${SEQUENCE_ID}_tracking.csv"
OUTPUT="$SEQ_DIR/${SEQUENCE_ID}_with_tracking.mp4"

if [[ ! -f "$VIDEO" ]]; then
  echo "Missing video: $VIDEO" >&2
  exit 2
fi

if [[ ! -f "$TRACKING" ]]; then
  echo "Missing tracking CSV: $TRACKING" >&2
  exit 2
fi

python gaze_hand_visualizer.py \
  --input "$VIDEO" \
  --csv "$TRACKING" \
  --output "$OUTPUT"

python - "$OUTPUT" <<'PY'
import sys
from pathlib import Path
import cv2

path = Path(sys.argv[1])
cap = cv2.VideoCapture(str(path))
if not path.exists() or path.stat().st_size <= 0 or not cap.isOpened():
    raise SystemExit(f"Invalid visualization output: {path}")
print(f"visualization_output={path}")
print(f"size_bytes={path.stat().st_size}")
print(f"frames={int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
print(f"fps={cap.get(cv2.CAP_PROP_FPS)}")
print(f"width={int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
print(f"height={int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
cap.release()
PY
