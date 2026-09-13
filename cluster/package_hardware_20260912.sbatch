#!/bin/bash
# Run only after all four training allocations finish (afterany dependency).
#SBATCH --job-name=hw_package
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/hardware_20260912/logs/%x_%j.out
set -euo pipefail
module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
HW_ROOT="$HOME/scratch/vla_bspline/hardware_20260912"
python "$HW_ROOT/scripts/package_apollo_handoff.py" --root "$HW_ROOT" \
  --output "$HW_ROOT/handoff_${SLURM_JOB_ID}" \
  --run cabinet_assembling_waypoint_v2_2456089 \
  --run cabinet_assembling_spline_v2_2456090 \
  --run drawer_assembling_waypoint_v1_2456084 \
  --run drawer_assembling_spline_v1_2456086
