#!/usr/bin/env bash
#SBATCH --job-name=egoeverything-s4check
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/xy3257/EgoEverything_repro_20260619/logs/section4_artifact_check-%j.out
#SBATCH --error=/scratch/xy3257/EgoEverything_repro_20260619/logs/section4_artifact_check-%j.err

set -euo pipefail

BASE=/scratch/xy3257/EgoEverything_repro_20260619
source "$BASE/env.sh"

python "$BASE/scripts/check_section4_artifacts.py" \
  --artifact-root "$BASE/section4_artifacts" \
  --dataset-path "$DATASET_PATH" \
  --repo "$EGOEVERYTHING_REPO" \
  --report "$BASE/section4_artifact_check.json"
