#!/bin/bash
#SBATCH --job-name=evlang_build_t04full
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

set -euo pipefail
module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
SNAP="$HOME/scratch/vla_bspline/code_snapshots/event_language_t04_20260821_v1"
WORK="$HOME/scratch/vla_bspline/event_language_t04_20260821_v1"
ROOT="$HOME/scratch/vla_bspline/hf_cache/hub/datasets--HuggingFaceVLA--libero/snapshots/86958911c0f959db2bbbdb107eb3e17c5f9c798e"
OUT="$HOME/scratch/vla_bspline/libero_event_t04_full_v1"
AUDIT="$HOME/scratch/vla_bspline/outputs/language_dataset_audit/event_t04_2321848.json"
test -d "$ROOT/meta"; test -s "$WORK/seg_labels.jsonl"; test -s "$AUDIT"; test ! -e "$OUT"
python - "$AUDIT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x["quality_gate_passed"], x["quality_gate_failures"]
PY
python "$SNAP/scripts/data/build_event_language_dataset.py" \
  --root "$ROOT" --labels "$WORK/seg_labels.jsonl" --out "$OUT" \
  --keep_original_fraction 0.0 --seed 20260821
test -s "$OUT/meta/event_language_provenance.json"
echo "=== completed 100%-event t0/t4 overlay job=$SLURM_JOB_ID ==="
