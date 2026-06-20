#!/usr/bin/env bash
#SBATCH --job-name=egoeverything-audit
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/xy3257/EgoEverything_repro_20260619/logs/repro_audit-%j.out
#SBATCH --error=/scratch/xy3257/EgoEverything_repro_20260619/logs/repro_audit-%j.err

set -euo pipefail

BASE=/scratch/xy3257/EgoEverything_repro_20260619
source "$BASE/env.sh"

SEQ=${SEQUENCE_ID:-loc5_script4_seq6_rec1}
COND_MANIFEST="$BASE/section4_conditions/$SEQ/section4_conditions_manifest.json"

python "$BASE/scripts/validate_section4_conditions.py" \
  --manifest "$COND_MANIFEST" \
  --report "$BASE/section4_conditions_validation.json"

python "$BASE/scripts/check_public_release_assets.py" \
  --report "$BASE/public_release_check.json"

set +e
python "$BASE/scripts/check_section4_artifacts.py" \
  --artifact-root "$BASE/section4_artifacts" \
  --dataset-path "$DATASET_PATH" \
  --repo "$EGOEVERYTHING_REPO" \
  --report "$BASE/section4_artifact_check.json"
ARTIFACT_RC=$?
set -e

python - <<PY
import json
from pathlib import Path

base = Path("$BASE")
summary = {
    "condition_validation": json.loads((base / "section4_conditions_validation.json").read_text()),
    "public_release_check": json.loads((base / "public_release_check.json").read_text()),
    "section4_artifact_check": json.loads((base / "section4_artifact_check.json").read_text()),
    "section4_artifact_check_exit_code": $ARTIFACT_RC,
}
(base / "repro_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({
    "condition_valid": summary["condition_validation"]["valid"],
    "public_section4_assets_ready": summary["public_release_check"]["public_section4_assets_ready"],
    "section4_artifact_ready": summary["section4_artifact_check"]["ready"],
    "section4_artifact_missing": summary["section4_artifact_check"]["missing"],
}, indent=2))
PY
