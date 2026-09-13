#!/bin/bash
# Re-export without one episode containing a discontinuous EE observation.
# Raw recordings and v1 export are retained. Original validation IDs unchanged.
#SBATCH --job-name=hw_cab_clean
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/hardware_20260912/logs/%x_%j.out
set -euo pipefail
module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
HW_ROOT="$HOME/scratch/vla_bspline/hardware_20260912"
export PYTHONPATH="$HW_ROOT/source/src" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
python "$HW_ROOT/scripts/export_apollo_lerobot.py" \
  --source "$HW_ROOT/raw/cabinet_assembling" --output "$HW_ROOT/datasets/cabinet_assembling_v2" \
  --exclude-episode 20260911T005454.988Z-2ad672
python "$HW_ROOT/scripts/gen_apollo_spline_stats.py" --dataset "$HW_ROOT/datasets/cabinet_assembling_v2"
