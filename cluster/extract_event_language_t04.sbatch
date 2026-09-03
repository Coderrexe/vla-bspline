#!/bin/bash
#SBATCH --job-name=evlang_extract_t04
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

set -euo pipefail
module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

SNAP="$HOME/scratch/vla_bspline/code_snapshots/event_language_t04_20260821_v1"
WORK="$HOME/scratch/vla_bspline/event_language_t04_20260821_v1"
SCRIPT="$SNAP/scripts/data/molmo_segment_labels.py"
test -s "$SCRIPT"
test -s "$SNAP/policy/smolvla_spline/event_targets.py"
test ! -e "$WORK/segments.jsonl"
test ! -e "$WORK/frames"

python "$SCRIPT" --extract --dataset libero \
  --task_id 0 --task_id 4 \
  --camera observation.images.image \
  --camera observation.images.image2 \
  --horizon_max 24 --min_seg 8 --pause_frac 0.15 \
  --frames_per_seg 3 --workdir "$WORK"

test -s "$WORK/segments.jsonl"
sha256sum "$WORK/segments.jsonl" > "$WORK/segments.sha256"
echo "=== completed extraction job=$SLURM_JOB_ID time=$(date -Is) ==="
