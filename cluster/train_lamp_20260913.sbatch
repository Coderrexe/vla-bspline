#!/bin/bash
# Usage: sbatch train_lamp_20260913.sbatch spline|waypoint
#SBATCH --job-name=lamp_train
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --exclude=r817u29n05,r817u23n05
#SBATCH --output=/gpfs/radev/home/zs442/vla_hardware_20260913/logs/%x_%j.out
set -euo pipefail
module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
HW_ROOT="$HOME/vla_hardware_20260913"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export HF_DATASETS_CACHE="$HW_ROOT/cache/datasets_${SLURM_JOB_ID}"
export PYTHONPATH="$HW_ROOT/source/src:$HW_ROOT/scripts"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 WANDB_MODE=offline PYTHONUNBUFFERED=1
test -s "$HW_ROOT/manifests/lamp_dataset_ready.json"
test -s "$HW_ROOT/manifests/lamp_visual_review.json"
HW_HEAD="${1:?spline or waypoint}"
test "$HW_HEAD" = spline || test "$HW_HEAD" = waypoint
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
python "$HW_ROOT/scripts/train_apollo.py" --root "$HW_ROOT" --task lamp_assembling \
  --head "$HW_HEAD" --steps 40000 --save-every 10000 --tag native_tcp_v1 \
  --dataset-version v1 --verify-initialization
HW_RUN="lamp_assembling_${HW_HEAD}_native_tcp_v1_${SLURM_JOB_ID}"
for HW_STEP in 010000 020000 030000 040000; do
  python "$HW_ROOT/scripts/verify_apollo_checkpoint.py" \
    --checkpoint "$HW_ROOT/outputs/$HW_RUN/checkpoints/$HW_STEP/pretrained_model" \
    --dataset "$HW_ROOT/datasets/lamp_assembling_v1" \
    --output "$HW_ROOT/outputs/$HW_RUN/offline_validation_${HW_STEP}.json"
done
