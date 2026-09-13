#!/bin/bash
# Test the exported source/config/tokenizer/checkpoint without Internet access.
#SBATCH --job-name=hw_portable_gate
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
HW_ROOT="$HOME/scratch/vla_bspline/hardware_20260912"
HW_KIT="$HW_ROOT/portable_gate_${SLURM_JOB_ID}"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
python "$HW_ROOT/scripts/package_apollo_handoff.py" --root "$HW_ROOT" \
  --output "$HW_KIT" --run cabinet_assembling_spline_smoke_2456067
mkdir "$HW_KIT/source"
tar -xzf "$HW_KIT/training_source_v1.tar.gz" -C "$HW_KIT/source"
export HF_HOME="$HW_KIT/hf_cache" HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE="$HW_ROOT/cache/portable_${SLURM_JOB_ID}"
export PYTHONPATH="$HW_KIT/source/src"
python "$HW_KIT/verify_apollo_checkpoint.py" \
  --checkpoint "$HW_KIT/models/cabinet_assembling_spline" \
  --dataset "$HW_ROOT/datasets/cabinet_assembling_v2" \
  --output "$HW_KIT/portable_verification.json" --frames-per-episode 2
