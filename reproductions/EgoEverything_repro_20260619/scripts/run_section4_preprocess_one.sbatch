#!/usr/bin/env bash
#SBATCH --job-name=egoeverything-s4prep
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xy3257/EgoEverything_repro_20260619/logs/section4_preprocess-%j.out
#SBATCH --error=/scratch/xy3257/EgoEverything_repro_20260619/logs/section4_preprocess-%j.err

set -euo pipefail

BASE=/scratch/xy3257/EgoEverything_repro_20260619
source "$BASE/env.sh"

python "$BASE/scripts/preprocess_section4_conditions.py" \
  --sequence-id "${SEQUENCE_ID:-loc5_script4_seq6_rec1}" \
  --dataset-path "$DATASET_PATH" \
  --output-root "$BASE/section4_conditions" \
  --crop-ratio "${GC_CROP_RATIO:-0.3333333333333333}" \
  --ad-scale "${AD_SCALE:-0.3333333333333333}" \
  ${MAX_FRAMES:+--max-frames "$MAX_FRAMES"}
