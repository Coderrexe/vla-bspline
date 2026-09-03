#!/bin/bash
# Failure-directed RoboCasa progress with deployable proprioceptive scheduling.
# Usage: sbatch eval_robocasa_progress_v3.sbatch CKPT TASK EPISODES MAX_STEPS TAG MODE [SEED]
# MODE is env, selfpaced, proprio, oracle, or fixedk:<positive steps>.
# Proprio uses observed Panda finger separation and command history, never
# simulator object state. Oracle remains a diagnostic upper bound.
#SBATCH --job-name=rc_prog_v3
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

CKPT=${1:?checkpoint required}
TASK=${2:?task required}
EPISODES=${3:-10}
MAX_STEPS=${4:?max steps required}
TAG=${5:-diagnostic}
MODE=${6:-env}
SEED=${7:-1000}
case "$TAG" in *[!A-Za-z0-9_.-]*) echo "unsafe tag: $TAG" >&2; exit 64 ;; esac

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/robocasa_env/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

SCRIPT="$HOME/scratch/vla_bspline/code_snapshots/robocasa_progress_eval_v3_20260821/robocasa_progress_eval.py"
OUT_DIR="$HOME/scratch/vla_bspline/outputs/robocasa_progress_v3"
SAFE_MODE=${MODE//:/_}
OUT="$OUT_DIR/${TAG}_${TASK}_${SAFE_MODE}_s${SEED}_n${EPISODES}_${SLURM_JOB_ID}.json"

[[ -f "$SCRIPT" ]] || { echo "missing frozen evaluator: $SCRIPT" >&2; exit 66; }
[[ -f "$CKPT/model.safetensors" ]] || { echo "missing checkpoint: $CKPT" >&2; exit 66; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite: $OUT" >&2; exit 73; }
mkdir -p "$OUT_DIR"

python "$SCRIPT" \
  --ckpt "$CKPT" \
  --task "$TASK" \
  --episodes "$EPISODES" \
  --max_steps "$MAX_STEPS" \
  --seed_base "$SEED" \
  --instr_mode "$MODE" \
  --out "$OUT"

test -s "$OUT"
sha256sum "$SCRIPT" "$OUT"
