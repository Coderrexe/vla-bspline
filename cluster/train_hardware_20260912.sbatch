#!/bin/bash
# Usage: sbatch [Slurm overrides] train_hardware_20260912.sbatch TASK HEAD [STEPS] [TAG] [DATASET_VERSION]
#SBATCH --job-name=hw_train
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/hardware_20260912/logs/%x_%j.out
set -euo pipefail
module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
HW_ROOT="$HOME/scratch/vla_bspline/hardware_20260912"
# Concurrent exports of the same Parquet files must not race in a shared Arrow
# builder cache. Keep this small derived cache separate for each allocation.
export HF_DATASETS_CACHE="$HW_ROOT/cache/datasets_${SLURM_JOB_ID}"
export PYTHONPATH="$HW_ROOT/source/src"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
echo "GPU allocation: job=$SLURM_JOB_ID node=$SLURMD_NODENAME visible=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
test -s "$HW_ROOT/manifests/datasets_ready.json"
python "$HW_ROOT/scripts/train_apollo.py" --root "$HW_ROOT" \
  --task "${1:?task}" --head "${2:?head}" --steps "${3:-20000}" --tag "${4:-v1}" \
  --dataset-version "${5:-v1}" --verify-initialization
HW_RUN="${1}_${2}_${4:-v1}_${SLURM_JOB_ID}"
python "$HW_ROOT/scripts/verify_apollo_checkpoint.py" \
  --checkpoint "$HW_ROOT/outputs/$HW_RUN/checkpoints/last/pretrained_model" \
  --dataset "$HW_ROOT/datasets/${1}_${5:-v1}" \
  --output "$HW_ROOT/outputs/$HW_RUN/offline_validation.json"
