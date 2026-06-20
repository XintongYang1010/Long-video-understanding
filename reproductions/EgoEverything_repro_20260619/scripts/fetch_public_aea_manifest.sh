#!/usr/bin/env bash
set -euo pipefail

BASE="${EGOEVERYTHING_BASE:-/scratch/xy3257/EgoEverything_repro_20260619}"
source "$BASE/env.sh"

MANIFEST_URL="${AEA_MANIFEST_URL:-https://explorer.projectaria.com:443/data/aea/download_links}"
mkdir -p "$(dirname "$DATASET_JSON")"

tmp="${DATASET_JSON}.tmp"
curl -L --fail --retry 3 --retry-delay 5 \
  -H "User-Agent: EgoEverything-repro/20260619" \
  "$MANIFEST_URL" -o "$tmp"

python - "$tmp" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
sequences = data.get("sequences")
if not isinstance(sequences, dict) or not sequences:
    raise SystemExit(f"Invalid AEA manifest: {path}")
print(f"Validated {len(sequences)} AEA sequences in {path}")
PY

mv "$tmp" "$DATASET_JSON"
{
  echo "source_url=$MANIFEST_URL"
  echo "fetched_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python - "$DATASET_JSON" <<'PY'
import hashlib
import sys
from pathlib import Path
path = Path(sys.argv[1])
print(f"sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
PY
} > "${DATASET_JSON}.source.txt"

echo "Wrote $DATASET_JSON"
