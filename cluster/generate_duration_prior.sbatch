#!/bin/bash
#SBATCH --job-name=duration_prior
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# Compute a fixed-duration control from every training frame. This stages a
# pinned event-target source and never modifies the installed policy package.

set -euo pipefail

CKPT=${1:?expected spline checkpoint pretrained_model directory}
TAG=${2:?expected filesystem-safe tag}
case "$TAG" in
  *[!A-Za-z0-9_.-]*) echo "unsafe tag: $TAG" >&2; exit 2 ;;
esac

module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

SCRIPT="$HOME/vla_bspline/cluster/duration_training_prior.py"
EVENT_SOURCE="$HOME/vla_bspline/cluster/event_targets_duration_prior.py"
OUT="$HOME/scratch/vla_bspline/outputs/duration_priors/${TAG}_${SLURM_JOB_ID}.json"
test -s "$CKPT/config.json"
test -s "$SCRIPT"
test -s "$EVENT_SOURCE"
test ! -e "$OUT"

python "$SCRIPT" \
  --ckpt "$CKPT" \
  --dataset_root "$HOME/scratch/vla_bspline/libero_l10gran" \
  --dataset_repo_id HuggingFaceVLA/libero \
  --event_target_source "$EVENT_SOURCE" \
  --out "$OUT"

test -s "$OUT"
OUT="$OUT" python - <<'PY'
import json
import os

result = json.load(open(os.environ["OUT"]))
assert result["complete_training_scan"] is True
assert result["dataset_frames_scanned"] == result["dataset_frames_expected"]
assert result["global_duration"] in range(result["min_seg"], result["horizon_max"] + 1)
print("validated duration prior:", result["global_duration"], "from", result["dataset_frames_scanned"], "frames")
PY
echo "=== completed duration prior job=${SLURM_JOB_ID} time=$(date -Is) ==="
