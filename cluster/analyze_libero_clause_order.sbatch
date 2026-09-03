#!/bin/bash
#SBATCH --job-name=t04_order_stats
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# Usage: sbatch --dependency=afterok:EVAL_JOB analyze... EVAL_JOB TAG MODEL_SHA CONFIG_SHA
set -euo pipefail
EVAL_JOB=${1:?evaluation job ID}
TAG=${2:?safe tag}
MODEL_SHA=${3:?expected model SHA-256}
CONFIG_SHA=${4:?expected isolated config SHA-256}
[[ "$EVAL_JOB" =~ ^[0-9]+$ ]] || exit 64
case "$TAG" in *[!A-Za-z0-9_.-]*) exit 64 ;; esac

ROOT="$HOME/scratch/vla_bspline/outputs/clause_order_steering"
SCRIPT="$HOME/scratch/vla_bspline/code_snapshots/clause_order_stats_v2_20260822/libero_clause_order_stats.py"
OUT="$HOME/scratch/vla_bspline/outputs/clause_order_analysis/${TAG}_${EVAL_JOB}.json"
mapfile -t GATES < <(find "$ROOT" -maxdepth 1 -type f -name "*replay_gate_${EVAL_JOB}.json" -print)
[[ ${#GATES[@]} -eq 1 ]] || { echo "expected one gate, found ${#GATES[@]}" >&2; exit 66; }
test -s "$SCRIPT"; test ! -e "$OUT"
echo "2778c010e59ca8584d22cc0d522961914b0299e806235cb0f9fc2dfe26ef0244  $SCRIPT" | sha256sum --check --strict
mkdir -p "$(dirname "$OUT")"
module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
python "$SCRIPT" --gate "${GATES[0]}" --expected_states_per_task 50 \
  --expected_model_sha256 "$MODEL_SHA" --expected_config_sha256 "$CONFIG_SHA" --out "$OUT"
sha256sum "$SCRIPT" "$OUT"
