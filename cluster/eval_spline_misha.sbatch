#!/bin/bash
#SBATCH --job-name=eval_spline
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# Evaluate the SmolVLA-Spline pilot in the LIBERO simulator.
# Usage: sbatch [--export=ALL,CKPT=...,TASK=...,EPISODES=...] eval_spline_misha.sbatch
set -uo pipefail
echo "=== node $(hostname) | $(date) ==="

module load miniconda
ENV=$HOME/.conda/envs/lerobot
export PATH="$ENV/bin:$PATH"
export HF_HOME=$HOME/scratch/vla_bspline/hf_cache
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
import sys, torch
assert torch.cuda.is_available(), "CUDA unavailable — aborting"
print("gpu:", torch.cuda.get_device_name(0))
PY
[ $? -eq 0 ] || exit 97

CKPT=${CKPT:-$HOME/scratch/vla_bspline/outputs/spline_pilot_20k/checkpoints/last/pretrained_model}
TASK=${TASK:-libero_object}
EPISODES=${EPISODES:-10}
CFREQ=${CFREQ:-20}
EPLEN=${EPLEN:-}   # scale with CFREQ for fair wall-clock budgets
RECORD=${RECORD:-false}
OUT=$HOME/scratch/vla_bspline/outputs/eval_$(basename $(dirname $(dirname $CKPT)))_${TASK}_f${CFREQ}_${SLURM_JOB_ID}

echo "=== CKPT=$CKPT TASK=$TASK EPISODES=$EPISODES ==="
python -m lerobot.scripts.lerobot_eval \
  --policy.path="$CKPT" \
  --output_dir="$OUT" \
  --env.type=libero \
  --env.task="$TASK" \
  --env.control_mode=relative \
  --env.control_freq=$CFREQ \
  ${EPLEN:+--env.episode_length=$EPLEN} \
  --eval.batch_size=1 \
  --eval.n_episodes=$EPISODES \
  --eval.recording=$RECORD \
  --env.max_parallel_tasks=1
echo "=== eval exit: $? | $(date) ==="
grep -rE "success|pc_success" "$OUT"/*.json 2>/dev/null | tail -5
