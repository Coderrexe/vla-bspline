#!/bin/bash
#SBATCH --job-name=baselineA
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# Baseline A on Misha: stock SmolVLA (waypoint) on ALL 40 LIBERO tasks, H100.
# 48h QoS + H100 => one shot, no 6h chunking. 24 dataloader workers to beat the
# LIBERO video-decode bottleneck that data-bound the Bouchet run.
set -uo pipefail
echo "=== node $(hostname) | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module load miniconda
ENV=$HOME/.conda/envs/lerobot
export PATH="$ENV/bin:$PATH"
export HF_HOME=$HOME/scratch/vla_bspline/hf_cache
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

STEPS=100000
OUT=$HOME/scratch/vla_bspline/outputs/baselineA_smolvla_libero
if [ -d "$OUT/checkpoints/last" ]; then RESUME=true; else RESUME=false; rm -rf "$OUT"; fi
echo "=== resume=$RESUME | out=$OUT ==="

python -m lerobot.scripts.lerobot_train \
  --policy.type=smolvla \
  --policy.push_to_hub=false \
  --policy.load_vlm_weights=true \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --output_dir="$OUT" \
  --steps=$STEPS \
  --batch_size=32 \
  --num_workers=24 \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false \
  --resume=$RESUME
echo "=== train exit: $? | $(date) ==="
