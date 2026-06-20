#!/usr/bin/env bash
#SBATCH --job-name=egoeverything-download
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/xy3257/EgoEverything_repro_20260619/logs/download_aea-%j.out
#SBATCH --error=/scratch/xy3257/EgoEverything_repro_20260619/logs/download_aea-%j.err

set -euo pipefail

BASE=/scratch/xy3257/EgoEverything_repro_20260619
source "$BASE/env.sh"

if [[ ! -f "$DATASET_JSON" || "${REFRESH_AEA_MANIFEST:-0}" == "1" ]]; then
  bash "$BASE/scripts/fetch_public_aea_manifest.sh"
fi

python "$BASE/scripts/download_aea_one_sequential.py" \
  --json "$DATASET_JSON" \
  --output "$DATASET_PATH" \
  --limit "${LIMIT:-1}"
