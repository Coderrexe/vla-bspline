#!/bin/bash
# CPU-only release of the exact-replay evaluation matrix after the same-allocation
# L40S C original/clause training pair completes.
#SBATCH --job-name=t04_Cpair_release
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err
set -euo pipefail

TRAIN_JOB=${1:?paired training job required}
SEED=${2:-1001}
ROOT="$HOME/scratch/vla_bspline/outputs"
ORIGINAL="$ROOT/Cwarm_t04exact_original_s${SEED}_7500_v1_l40s_samealloc/checkpoints/last/pretrained_model"
CLAUSE="$ROOT/Cwarm_t04exact_clause_s${SEED}_7500_v1_l40s_samealloc/checkpoints/last/pretrained_model"
LAUNCHER="$HOME/vla_bspline/cluster/eval_two_subgoal_clause_t04_v2.sbatch"
SOURCE="$HOME/vla_bspline/t04_evalv2_source_20260821_v3"
LEDGER="$HOME/scratch/vla_bspline/manifests/two_subgoal_clause_2x2/C_l40s_eval_release_${SLURM_JOB_ID}.txt"
test -s "$ORIGINAL/model.safetensors"
test -s "$CLAUSE/model.safetensors"
test -s "$ORIGINAL/../../../reproducibility/launch_manifest.json"
test -s "$CLAUSE/../../../reproducibility/launch_manifest.json"
test ! -e "$LEDGER"

cd "$HOME/vla_bspline/cluster"
jobs=()
for mode in fixed_clock event_clock; do
  for treatment in original clause; do
    if [[ "$treatment" == original ]]; then ckpt="$ORIGINAL"; else ckpt="$CLAUSE"; fi
    jobs+=("$(sbatch --parsable --nice=2500 \
      --export=ALL,EXPECTED_TRAIN_GPU_CLASS=L40S,T04_EVAL_V2_SOURCE_DIR="$SOURCE" \
      "$LAUNCHER" "$ckpt" "C_${treatment}_s${SEED}_l40s" "$mode" 50)")
  done
done
for treatment in original clause; do
  if [[ "$treatment" == original ]]; then ckpt="$ORIGINAL"; else ckpt="$CLAUSE"; fi
  jobs+=("$(sbatch --parsable --nice=10000 \
    --export=ALL,EXPECTED_TRAIN_GPU_CLASS=L40S,T04_EVAL_V2_SOURCE_DIR="$SOURCE" \
    "$LAUNCHER" "$ckpt" "C_${treatment}_s${SEED}_l40s" compound 50)")
done
{
  echo "schema=two_subgoal_c_l40s_eval_release_v1"
  echo "training_job=$TRAIN_JOB"
  echo "seed=$SEED"
  printf 'eval_job=%s\n' "${jobs[@]}"
} > "$LEDGER"
cat "$LEDGER"
