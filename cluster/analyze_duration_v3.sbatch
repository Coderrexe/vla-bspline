#!/bin/bash
# Merge and analyze disjoint task shards from one duration-v3 checkpoint.
# Usage: sbatch analyze_duration_v3.sbatch LEFT_JSON RIGHT_JSON OUT_PREFIX
#SBATCH --job-name=dur_v3_analysis
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

set -euo pipefail
LEFT=${1:?expected first immutable v3 artifact}
RIGHT=${2:?expected second immutable v3 artifact}
OUT_PREFIX=${3:?expected new output prefix}
test -s "$LEFT"
test -s "$RIGHT"
test ! -e "${OUT_PREFIX}.json"
test ! -e "${OUT_PREFIX}.md"

module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
python "$HOME/vla_bspline/scripts/analysis/duration_v3_analysis.py" \
  "$LEFT" "$RIGHT" --out-prefix "$OUT_PREFIX" --bootstrap-samples 20000
test -s "${OUT_PREFIX}.json"
test -s "${OUT_PREFIX}.md"
echo "completed duration-v3 analysis job=$SLURM_JOB_ID output=$OUT_PREFIX"
