#!/bin/bash
#SBATCH --job-name=duration_parity
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# Compare the direct action-only scanner against LeRobot's actual low-level
# action query at explicit start/interior/near-end/final episode anchors.

set -euo pipefail

CKPT=${1:?expected duration-enabled checkpoint}
module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export OMP_NUM_THREADS=1

SOURCE="$HOME/vla_bspline/cluster"
STAGE="$HOME/scratch/vla_bspline/code_snapshots/duration_prior_parity_${SLURM_JOB_ID}"
OUT="$HOME/scratch/vla_bspline/outputs/duration_priors/parity_${SLURM_JOB_ID}.json"
test -s "$CKPT/config.json"
test ! -e "$STAGE"
test ! -e "$OUT"
mkdir -p "$STAGE/scripts/analysis"
touch "$STAGE/scripts/__init__.py" "$STAGE/scripts/analysis/__init__.py"
cp "$SOURCE/duration_prior_parity.py" "$STAGE/scripts/analysis/"
cp "$SOURCE/duration_training_prior.py" "$STAGE/scripts/analysis/"
cp "$SOURCE/event_targets_duration_prior.py" "$STAGE/"
(cd "$STAGE" && find . -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)

cd "$STAGE"
python -m scripts.analysis.duration_prior_parity \
  --ckpt "$CKPT" \
  --dataset_root "$HOME/scratch/vla_bspline/libero_l10gran" \
  --dataset_repo_id HuggingFaceVLA/libero \
  --event_target_source "$STAGE/event_targets_duration_prior.py" \
  --sample 0:0 --sample 0:20 --sample 0:190 --sample 0:213 \
  --sample 1:0 --sample 1:31 --sample 1:260 --sample 1:283 \
  --out "$OUT"

test -s "$OUT"
OUT="$OUT" python - <<'PY'
import json
import os

result = json.load(open(os.environ["OUT"]))
assert result["all_samples_exact"] is True
assert result["sample_count"] == 8
assert result["video_decode_called"] is False
print("validated exact duration-prior parity on", result["sample_count"], "anchors")
PY
echo "=== completed duration parity job=${SLURM_JOB_ID} time=$(date -Is) ==="
