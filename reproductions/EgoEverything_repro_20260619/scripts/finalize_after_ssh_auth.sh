#!/usr/bin/env bash
# Finalize the EgoEverything reproduction audit after Torch SSH auth is restored.

set -euo pipefail

LOCAL_ROOT=${LOCAL_ROOT:-/Users/xintongyang/Documents/NYU/EgoEverything_repro_support_20260619}
REMOTE_HOST=${REMOTE_HOST:-torch}
BASE=${BASE:-/scratch/xy3257/EgoEverything_repro_20260619}
AUDIT_JOB_ID=${AUDIT_JOB_ID:-11112793}

echo "[1/5] Checking SSH connectivity to ${REMOTE_HOST}"
ssh -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE_HOST" "echo connected: \$(hostname)"

echo "[2/5] Syncing support scripts and docs"
scp \
  "$LOCAL_ROOT/README_REPRO.md" \
  "$LOCAL_ROOT/SECTION4_REPRO_AUDIT.md" \
  "$LOCAL_ROOT/scripts/validate_section4_conditions.py" \
  "$LOCAL_ROOT/scripts/check_public_release_assets.py" \
  "$LOCAL_ROOT/scripts/run_repro_audit.sbatch" \
  "$REMOTE_HOST:$BASE/"

ssh "$REMOTE_HOST" "set -euo pipefail
  mv '$BASE/validate_section4_conditions.py' '$BASE/scripts/validate_section4_conditions.py'
  mv '$BASE/check_public_release_assets.py' '$BASE/scripts/check_public_release_assets.py'
  mv '$BASE/run_repro_audit.sbatch' '$BASE/scripts/run_repro_audit.sbatch'
  chmod +x '$BASE/scripts/validate_section4_conditions.py' '$BASE/scripts/check_public_release_assets.py' '$BASE/scripts/run_repro_audit.sbatch'
  rm -rf '$BASE/scripts/__pycache__'
"

echo "[3/5] Checking audit Slurm job ${AUDIT_JOB_ID}"
ssh "$REMOTE_HOST" "squeue -j '$AUDIT_JOB_ID' -o '%.18i %.9P %.32j %.8T %.10M %.10l %.6D %R' || true
sacct -j '$AUDIT_JOB_ID' --format=JobID,JobName%34,Partition,State,ExitCode,Elapsed,MaxRSS -P || true"

echo "[4/5] Reading audit reports"
ssh "$REMOTE_HOST" "set -euo pipefail
  for f in section4_conditions_validation.json public_release_check.json section4_artifact_check.json repro_audit_summary.json; do
    echo '---' \"\$f\" '---'
    if [ -f '$BASE/'\"\$f\" ]; then
      python -m json.tool '$BASE/'\"\$f\" | sed -n '1,220p'
    else
      echo 'missing'
    fi
  done
"

echo "[5/5] Final no-secret and no-active-job checks"
ssh "$REMOTE_HOST" "set -euo pipefail
  REPO='$BASE/EgoEverything'
  echo '--- squeue ---'
  squeue -u \"\$USER\" -o '%.18i %.9P %.32j %.8T %.10M %.10l %.6D %R'
  echo '--- repo status ---'
  git -C \"\$REPO\" status --short
  echo '--- secret scan ---'
  grep -RIn 'sk-or-v1-' --exclude-dir=.git \"\$REPO\" || true
"
