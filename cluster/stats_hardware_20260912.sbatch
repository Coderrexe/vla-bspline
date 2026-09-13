#!/bin/bash
#SBATCH --job-name=hw_stats
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/hardware_20260912/logs/%x_%j.out
set -euo pipefail
module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
HW_ROOT="$HOME/scratch/vla_bspline/hardware_20260912"
export PYTHONPATH="$HW_ROOT/source/src"
export OMP_NUM_THREADS=4
for task in cabinet_assembling drawer_assembling; do
  python "$HW_ROOT/scripts/gen_apollo_spline_stats.py" --dataset "$HW_ROOT/datasets/${task}_v1"
done
