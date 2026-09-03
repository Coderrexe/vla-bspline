#!/bin/bash
#SBATCH --job-name=dur_ablate
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# PROPOSAL ONLY (2026-08-21): do not submit until the current locked language
# factorial finishes and these sources have been reviewed/synced atomically.
#
# Usage:
#   sbatch eval_duration_ablation.sbatch CKPT TAG predicted
#   sbatch eval_duration_ablation.sbatch CKPT TAG fixed FIXED_MANIFEST.json
#   sbatch eval_duration_ablation.sbatch CKPT TAG fixed_task TASK_MAP.json
#   sbatch eval_duration_ablation.sbatch CKPT TAG shuffled PREDICTED_RESULT.json
#
# `fixed` manifests are JSON with at least {"global_duration": 24, ...} and
# training-split provenance.  `shuffled` must depend afterok on the matching
# predicted job and receive that job's exact result path.

set -euo pipefail

CKPT=${1:?expected checkpoint pretrained_model directory}
TAG=${2:?expected filesystem-safe tag}
MODE=${3:?expected predicted, fixed, fixed_task, or shuffled}
AUX=${4:-}
STATE_START=${STATE_START:-0}
STATES_PER_TASK=${STATES_PER_TASK:-50}
case "$TAG" in
  *[!A-Za-z0-9_.-]*) echo "unsafe tag: $TAG" >&2; exit 2 ;;
esac
case "$MODE" in
  predicted|fixed|fixed_task|shuffled) ;;
  *) echo "invalid duration mode: $MODE" >&2; exit 2 ;;
esac

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false

test -s "$CKPT/model.safetensors"
test -s "$CKPT/config.json"
# Stage immutable copies from the cluster launcher directory. This keeps the
# experiment independent of edits to the installed LeRobot source tree.
SOURCE="$HOME/vla_bspline/cluster"
STAGE="$HOME/scratch/vla_bspline/code_snapshots/duration_${SLURM_JOB_ID}"
OUTDIR="$HOME/scratch/vla_bspline/outputs/duration_ablation"
OUT="$OUTDIR/${TAG}_${MODE}_${SLURM_JOB_ID}.json"
test ! -e "$STAGE"
test ! -e "$OUT"
mkdir -p "$STAGE" "$OUTDIR"
cp "$SOURCE/libero_duration_ablation.py" "$STAGE/"
cp "$SOURCE/duration_intervention.py" "$STAGE/"
cp "$SOURCE/libero_locked_eval.py" "$STAGE/"
sha256sum "$STAGE"/*.py > "$STAGE/SHA256SUMS"

ARGS=(
  --ckpt "$CKPT"
  --suite libero_10
  --state_start "$STATE_START"
  --states_per_task "$STATES_PER_TASK"
  --seed_base 100000
  --control_freq 20
  --duration_mode "$MODE"
  --out "$OUT"
)
case "$MODE" in
  predicted)
    test -z "$AUX"
    ;;
  fixed)
    test -s "$AUX"
    FIXED_DURATION=$(AUX="$AUX" python - <<'PY'
import json, os
value = json.load(open(os.environ["AUX"]))["global_duration"]
if not isinstance(value, int):
    raise TypeError("global_duration must be an integer")
print(value)
PY
)
    ARGS+=(--fixed_duration "$FIXED_DURATION" --fixed_manifest "$AUX")
    ;;
  fixed_task)
    test -s "$AUX"
    ARGS+=(--task_duration_map "$AUX")
    ;;
  shuffled)
    test -s "$AUX"
    ARGS+=(--shuffle_source "$AUX" --shuffle_seed 73921)
    ;;
esac

python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    print("FATAL: CUDA unavailable", file=sys.stderr)
    raise SystemExit(97)
print("gpu:", torch.cuda.get_device_name(0))
PY

echo "=== duration tag=${TAG} mode=${MODE} job=${SLURM_JOB_ID} node=$(hostname) ==="
python "$STAGE/libero_duration_ablation.py" "${ARGS[@]}"
test -s "$OUT"
OUT="$OUT" MODE="$MODE" STATES_PER_TASK="$STATES_PER_TASK" python - <<'PY'
import json, os
result = json.load(open(os.environ["OUT"]))
assert result["duration_mode"] == os.environ["MODE"]
assert result["n_episodes"] == 10 * int(os.environ["STATES_PER_TASK"])
assert result["n_duration_calls"] > 0
print("validated:", result["n_successes"], "/", result["n_episodes"])
PY
