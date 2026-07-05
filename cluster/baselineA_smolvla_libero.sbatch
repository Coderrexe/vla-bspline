#!/bin/bash
#SBATCH --job-name=baselineA
#SBATCH --partition=gpu_devel
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# Baseline A: stock SmolVLA (waypoint output) fine-tuned on ALL 40 LIBERO tasks.
# TRAIN-ONLY for now (LIBERO sim eval dep still being fixed). gpu_devel: 6h cap, so we
# use a STABLE output dir + auto-resume, and self-resubmit if not finished (safety-capped).
set -uo pipefail
echo "=== node $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load miniconda
source activate $HOME/.conda/envs/lerobot
export HF_HOME=$HOME/scratch_pi_zy286/zs442/vla_bspline/hf_cache
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

STEPS=100000
OUT=$HOME/scratch_pi_zy286/zs442/vla_bspline/outputs/baselineA_smolvla_libero

if [ -f "$OUT/checkpoints/last/pretrained_model/train_config.json" ]; then
  # lerobot resume API: point at the saved config, don't re-pass CLI args
  echo "=== resuming from checkpoint ==="
  lerobot-train \
    --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" \
    --resume=true
else
  rm -rf "$OUT"
  echo "=== fresh start ==="
  lerobot-train \
    --policy.type=smolvla \
    --policy.push_to_hub=false \
    --policy.load_vlm_weights=true \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --output_dir="$OUT" \
    --steps=$STEPS \
    --batch_size=32 \
    --num_workers=12 \
    --save_freq=5000 \
    --log_freq=100 \
    --wandb.enable=false
fi
TRAIN_RC=$?
echo "=== train exit: $TRAIN_RC | $(date) ==="

# self-resubmit to continue past the 6h wall, capped at 5 hops to avoid runaway loops
LAST_STEP=$(ls "$OUT/checkpoints" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
LAST_STEP=${LAST_STEP:-0}; LAST_STEP=$((10#$LAST_STEP))
HOP_FILE="$OUT/.resubmit_hops"; HOPS=$(cat "$HOP_FILE" 2>/dev/null || echo 0)
echo "=== last checkpoint step: $LAST_STEP / $STEPS | hops: $HOPS ==="
if [ "$TRAIN_RC" -eq 0 ] && [ "$LAST_STEP" -lt "$STEPS" ] && [ "$HOPS" -lt 5 ]; then
  echo $((HOPS+1)) > "$HOP_FILE"
  echo "=== not finished; resubmitting for next 6h session ==="
  cd "$HOME/vla_bspline/cluster" && sbatch baselineA_smolvla_libero.sbatch
fi
