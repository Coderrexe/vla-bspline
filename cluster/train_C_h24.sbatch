#!/bin/bash
#SBATCH --job-name=C_h24
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# SmolVLA-Spline (B-spline + duration head, v1 fixed-T) on all 40 LIBERO tasks.
# Phase 1: 100-step integration smoke. Phase 2 (if clean): 20k-step pilot.
set -uo pipefail
echo "=== node $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load miniconda
ENV=$HOME/.conda/envs/lerobot
export PATH="$ENV/bin:$PATH"
export HF_HOME=$HOME/scratch/vla_bspline/hf_cache
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

# HARD GATE: never let a driver/CUDA mismatch silently train on CPU again.
python - <<'PY'
import sys, torch
ok = torch.cuda.is_available()
print(f"torch {torch.__version__} cuda-build {torch.version.cuda} available={ok}")
if not ok:
    print("FATAL: CUDA unavailable on this node — aborting instead of CPU-crawling.")
    sys.exit(97)
print("gpu:", torch.cuda.get_device_name(0))
PY
[ $? -eq 0 ] || exit 97

train () {  # $1=steps  $2=outdir
  local OUT=$2
  if [ -f "$OUT/checkpoints/last/pretrained_model/train_config.json" ]; then
    # lerobot resume API: point at the saved config, don't re-pass CLI args
    python -m lerobot.scripts.lerobot_train \
      --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" \
      --resume=true
  else
    rm -rf "$OUT"
    python -m lerobot.scripts.lerobot_train \
      --policy.type=smolvla_spline \
      --policy.predict_duration=true \
      --policy.horizon_max=24 \
      --policy.spline_stats_file_v2=spline_stats_libero_v2_h24.json \
      --policy.push_to_hub=false \
      --policy.load_vlm_weights=true \
      --dataset.repo_id=HuggingFaceVLA/libero \
      --output_dir="$OUT" \
      --steps=$1 \
      --batch_size=32 \
      --num_workers=24 \
      --save_freq=2500 \
      --log_freq=50 \
      --wandb.enable=false
  fi
}

echo "=== PHASE 1: 100-step integration smoke ==="
train 100 $HOME/scratch/vla_bspline/outputs/C_h24_smoke
RC=$?
echo "=== smoke exit: $RC ==="
if [ $RC -ne 0 ]; then echo "SMOKE FAILED — aborting pilot"; exit $RC; fi

echo "=== PHASE 2: 20k-step pilot ==="
train 20000 $HOME/scratch/vla_bspline/outputs/C_h24_20k
echo "=== pilot exit: $? | $(date) ==="
