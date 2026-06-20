#!/usr/bin/env bash
set -euo pipefail

BASE="${EGOEVERYTHING_BASE:-/scratch/xy3257/EgoEverything_repro_20260619}"
source "$BASE/env.sh"

python Generation/main.py \
  --api-key dummy \
  --dataset-path "$DATASET_PATH" \
  --json-path "$BASE/configs/empty_sequences.json" \
  --dataset-name smoke_empty \
  --limit 0 \
  --output-path "$BASE/smoke_empty_vqa.json" \
  --temp-dir "$BASE/tmp_smoke"
