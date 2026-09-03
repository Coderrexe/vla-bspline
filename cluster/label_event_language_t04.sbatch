#!/bin/bash
#SBATCH --job-name=evlang_label_t04
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

set -euo pipefail
module load miniconda
export PATH="$HOME/.conda/envs/molmo/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export TOKENIZERS_PARALLELISM=false

SNAP="$HOME/scratch/vla_bspline/code_snapshots/event_language_t04_20260821_v1"
WORK="$HOME/scratch/vla_bspline/event_language_t04_20260821_v1"
SCRIPT="$SNAP/scripts/data/molmo_segment_labels.py"
test -s "$WORK/segments.jsonl"
test -s "$WORK/segments.sha256"
sha256sum -c "$WORK/segments.sha256"

python "$SCRIPT" --label --label_mode phase_bank --workdir "$WORK"

test -s "$WORK/seg_labels.jsonl"
sha256sum "$WORK/seg_labels.jsonl" > "$WORK/seg_labels.sha256"
echo "=== completed labeling job=$SLURM_JOB_ID time=$(date -Is) ==="
