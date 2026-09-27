#!/bin/bash
#SBATCH --job-name=drawer_ckpt_audit
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=r817u29n05,r817u23n05
#SBATCH --output=/gpfs/radev/home/zs442/vla_hardware_20260913/logs/%x_%j.out
set -euo pipefail
module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
HW_ROOT="$HOME/vla_hardware_20260913"
HW_PREVIOUS="$HOME/scratch/vla_bspline/hardware_20260912"
export PYTHONPATH="$HW_ROOT/source/src:$HW_ROOT/scripts"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export HF_DATASETS_CACHE="$HW_ROOT/cache/datasets_${SLURM_JOB_ID}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
python "$HW_ROOT/scripts/prepare_apollo_checkpoint_views.py" \
  --run "$HW_PREVIOUS/outputs/drawer_assembling_spline_v1_2456086" \
  --output "$HW_ROOT/outputs/drawer_spline_views" --prefix drawer_spline
python "$HW_ROOT/scripts/prepare_apollo_checkpoint_views.py" \
  --run "$HW_PREVIOUS/outputs/drawer_assembling_waypoint_v1_2456084" \
  --output "$HW_ROOT/outputs/drawer_waypoint_views" --prefix drawer_waypoint
python "$HW_ROOT/scripts/diagnose_apollo_first_grasp.py" evaluate \
  --inputs "$HW_ROOT/reports/first_grasp_inputs_035" \
  --videos "$HW_PREVIOUS/raw/drawer_assembling" \
  --models "$HW_ROOT"/outputs/drawer_spline_views/drawer_spline_* \
           "$HW_ROOT"/outputs/drawer_waypoint_views/drawer_waypoint_* \
  --output "$HW_ROOT/reports/drawer_checkpoint_grasp_${SLURM_JOB_ID}"
