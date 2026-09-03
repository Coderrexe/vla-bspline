#!/bin/bash
# Validate and summarize the completed two-seed fixed-clock phase experiment.
# Usage: sbatch ... analyze_robocasa_phase_replication.sbatch IMMUTABLE_ANALYSIS_SNAPSHOT
#SBATCH --job-name=rc_phase_stats
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

set -euo pipefail
SNAP=${1:?immutable analysis snapshot}
SCRIPT="$SNAP/scripts/analysis/robocasa_phase_replication_stats.py"
CONFIG="$SNAP/configs/robocasa_phase_replication_s1000_s1001.json"
ROOT="$HOME/scratch/vla_bspline/outputs"
OUTDIR="$HOME/scratch/vla_bspline/artifacts/robocasa_phase_replication"
OUT="$OUTDIR/robocasa_phase_replication_stats_20260825_${SLURM_JOB_ID}.json"

test -s "$SCRIPT"; test -s "$CONFIG"; test ! -e "$OUT"
mkdir -p "$OUTDIR"
module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export PYTHONPATH="$SNAP"
python -m pytest -q "$SNAP/tests/test_robocasa_phase_replication_stats.py"
python "$SCRIPT" --config "$CONFIG" --root "$ROOT" --out "$OUT"
sha256sum "$SCRIPT" "$CONFIG" "$OUT"
