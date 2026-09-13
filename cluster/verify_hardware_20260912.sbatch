#!/bin/bash
# Usage: sbatch verify_hardware_20260912.sbatch RUN_NAME TASK [FRAMES_PER_EPISODE]
#SBATCH --job-name=hw_verify
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --exclude=r817u29n05,r817u23n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/hardware_20260912/logs/%x_%j.out
set -euo pipefail
module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
HW_ROOT="$HOME/scratch/vla_bspline/hardware_20260912"
export HF_DATASETS_CACHE="$HW_ROOT/cache/datasets_${SLURM_JOB_ID}"
export PYTHONPATH="$HW_ROOT/source/src"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
python "$HW_ROOT/scripts/verify_apollo_checkpoint.py" \
  --checkpoint "$HW_ROOT/outputs/${1:?run}/checkpoints/last/pretrained_model" \
  --dataset "$HW_ROOT/datasets/${2:?task}_v1" \
  --output "$HW_ROOT/outputs/$1/offline_validation_${SLURM_JOB_ID}.json" \
  --frames-per-episode "${3:-10}"
