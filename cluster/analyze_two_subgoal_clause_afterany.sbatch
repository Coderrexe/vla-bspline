#!/bin/bash
# Fail-closed autonomous harvest for the complete 3-seed two-subgoal factorial.
# Usage: sbatch --dependency=afterany:<...> ... COMMA_SEPARATED_EVAL_JOB_IDS
#SBATCH --job-name=t04_clause_stats
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail

JOB_CSV=${1:?comma-separated eval job IDs required}
IFS=',' read -r -a JOB_IDS <<< "$JOB_CSV"
[[ ${#JOB_IDS[@]} -eq 30 ]] || { echo "expected 30 eval jobs" >&2; exit 64; }
ROOT="$HOME/scratch/vla_bspline/outputs/two_subgoal_eval_v2"
SCRIPT="$HOME/scratch/vla_bspline/code_snapshots/t04_clause_stats_v2_20260822/two_subgoal_clause_stats.py"
OUT_ROOT="$HOME/scratch/vla_bspline/outputs/two_subgoal_clause_analysis"
OUT="$OUT_ROOT/locked_3seed_factorial_${SLURM_JOB_ID}.json"
[[ ! -e "$OUT" ]] || { echo "refusing overwrite" >&2; exit 73; }
echo "9a1308039192b80e6fc03d4b93831120b32de0bea43c4f7acb9f0a6f4655a498  $SCRIPT" | sha256sum --check --strict

ARGS=()
declare -A SEEN=()
for JOB in "${JOB_IDS[@]}"; do
  [[ "$JOB" =~ ^[0-9]+$ ]] || { echo "invalid job ID: $JOB" >&2; exit 64; }
  [[ -z ${SEEN[$JOB]+x} ]] || { echo "duplicate job ID: $JOB" >&2; exit 64; }
  SEEN[$JOB]=1
  mapfile -t MATCHES < <(find "$ROOT" -maxdepth 1 -type f -name "*replay_gate_${JOB}.json" -print)
  [[ ${#MATCHES[@]} -eq 1 ]] || {
    echo "job $JOB has ${#MATCHES[@]} replay gates, expected one" >&2; exit 66;
  }
  ARGS+=(--gate "${MATCHES[0]}")
done
mkdir -p "$OUT_ROOT"
module load miniconda
export PATH="$HOME/.conda/envs/robocasa_env/bin:$PATH"
python "$SCRIPT" "${ARGS[@]}" --out "$OUT"
test -s "$OUT"
sha256sum "$SCRIPT" "$OUT"
