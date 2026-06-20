#!/usr/bin/env bash
#SBATCH --job-name=egoeverything-smoke
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/xy3257/EgoEverything_repro_20260619/logs/smoke_empty-%j.out
#SBATCH --error=/scratch/xy3257/EgoEverything_repro_20260619/logs/smoke_empty-%j.err

set -euo pipefail

BASE=/scratch/xy3257/EgoEverything_repro_20260619
bash "$BASE/scripts/run_smoke_empty.sh"
