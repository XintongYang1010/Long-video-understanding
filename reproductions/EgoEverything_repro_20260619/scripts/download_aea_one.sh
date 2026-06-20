#!/usr/bin/env bash
set -euo pipefail

BASE="${EGOEVERYTHING_BASE:-/scratch/xy3257/EgoEverything_repro_20260619}"
source "$BASE/env.sh"

LIMIT="${LIMIT:-1}"

if [[ ! -f "$DATASET_JSON" || "${REFRESH_AEA_MANIFEST:-0}" == "1" ]]; then
  bash "$BASE/scripts/fetch_public_aea_manifest.sh"
fi

python "$BASE/scripts/download_aea_one_sequential.py" \
  --json "$DATASET_JSON" \
  --output "$DATASET_PATH" \
  --limit "$LIMIT"
