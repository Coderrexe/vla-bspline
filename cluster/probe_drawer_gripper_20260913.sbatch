#!/bin/bash
# One short offline model-behavior debug job. No hardware access.
#SBATCH --job-name=drawer_grip_probe
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/radev/home/zs442/vla_hardware_20260913/logs/%x_%j.out
set -euo pipefail
module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
HW_ROOT="$HOME/vla_hardware_20260913"
export PYTHONPATH="$HW_ROOT/source/src:$HW_ROOT/transition_scripts:$HW_ROOT/scripts"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
python "$HW_ROOT/transition_scripts/probe_apollo_gripper_conditioning.py" \
  --inputs "$HW_ROOT/reports/first_grasp_inputs_035" \
  --videos "$HOME/scratch/vla_bspline/hardware_20260912/raw/drawer_assembling" \
  --models "$HW_ROOT/outputs/drawer_spline_views/drawer_spline_020000" \
           "$HW_ROOT/outputs/drawer_waypoint_views/drawer_waypoint_020000" \
  --output "$HW_ROOT/reports/drawer_gripper_conditioning_${SLURM_JOB_ID}"
