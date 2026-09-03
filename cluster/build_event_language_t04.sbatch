#!/bin/bash
#SBATCH --job-name=evlang_build_t04
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

set -euo pipefail
module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

SNAP="$HOME/scratch/vla_bspline/code_snapshots/event_language_t04_20260821_v1"
WORK="$HOME/scratch/vla_bspline/event_language_t04_20260821_v1"
ROOT="$HOME/scratch/vla_bspline/hf_cache/lerobot/HuggingFaceVLA/libero"
OUT="$HOME/scratch/vla_bspline/libero_event_t04_epmix_v1"
SCRIPT="$SNAP/scripts/data/build_event_language_dataset.py"

# Resolve through LeRobot's cache layout once here; the builder records the
# canonical resolved path and hashes all non-language parquet inputs.
if [ ! -d "$ROOT/meta" ]; then
  ROOT=$(find "$HOME/scratch/vla_bspline/hf_cache" -path '*/datasets--HuggingFaceVLA--libero/snapshots/*' -type d | head -1)
fi
test -d "$ROOT/meta"
test -s "$WORK/seg_labels.jsonl"
sha256sum -c "$WORK/seg_labels.sha256"
test ! -e "$OUT"

python "$SCRIPT" --root "$ROOT" --labels "$WORK/seg_labels.jsonl" \
  --out "$OUT" --keep_original_fraction 0.5 --seed 20260821

test -s "$OUT/meta/event_language_provenance.json"
echo "=== completed dataset build job=$SLURM_JOB_ID time=$(date -Is) ==="
