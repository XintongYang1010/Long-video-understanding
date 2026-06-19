#!/usr/bin/env bash
#SBATCH --job-name=routeb_t5_t10
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export ROUTEB_TOP_KS="${ROUTEB_TOP_KS:-5 10}"
export ROUTEB_EVAL_MODES="${ROUTEB_EVAL_MODES:-single pair all}"
export ROUTEB_OUTPUT_PATH="${ROUTEB_OUTPUT_PATH:-/scratch/${USER}/data/multiresult/routeB_siglip_day1_top5_top10.json}"
export ROUTEB_SAVE_EVERY="${ROUTEB_SAVE_EVERY:-1}"

bash "$(dirname "$0")/run_routeb_siglip_h200.sbatch"
