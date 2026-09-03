#!/bin/bash
# OSMesa repeatability gate for legacy RoboCasa spline checkpoints.  These
# checkpoints depend on the historical installed policy/stats contract; do not
# use this launcher for corrected-contract checkpoints.
# Usage: sbatch eval_robocasa_progress_v9_osmesa_legacy.sbatch \
#   CKPT TASK EPISODES MAX_STEPS TAG MODE [SEED]
#SBATCH --job-name=rc_osm_legacy
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=%x-%j.out

set -euo pipefail
CKPT=${1:?checkpoint required}
TASK=${2:?task required}
EPISODES=${3:-1}
MAX_STEPS=${4:?max steps required}
TAG=${5:?tag required}
MODE=${6:-env}
SEED=${7:-1000}
case "$TAG" in *[!A-Za-z0-9_.-]*) echo "unsafe tag: $TAG" >&2; exit 64 ;; esac
for VALUE in "$EPISODES" "$MAX_STEPS" "$SEED"; do
  case "$VALUE" in ''|*[!0-9]*) echo "invalid integer: $VALUE" >&2; exit 64 ;; esac
done

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0 Mesa/22.2.4-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/robocasa_env/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCRIPT="$HOME/scratch/vla_bspline/code_snapshots/robocasa_progress_eval_v9_20260821/robocasa_progress_eval.py"
OUT_DIR="$HOME/scratch/vla_bspline/outputs/robocasa_progress_v9_osmesa_legacy"
SAFE_MODE=${MODE//:/_}
OUT="$OUT_DIR/${TAG}_${TASK}_${SAFE_MODE}_s${SEED}_n${EPISODES}_${SLURM_JOB_ID}.json"
[[ -f "$SCRIPT" ]] || { echo "missing frozen evaluator: $SCRIPT" >&2; exit 66; }
[[ -f "$CKPT/model.safetensors" && -f "$CKPT/config.json" ]] || {
  echo "missing checkpoint artifacts: $CKPT" >&2; exit 66;
}
[[ ! -e "$OUT" ]] || { echo "refusing output overwrite: $OUT" >&2; exit 73; }
mkdir -p "$OUT_DIR"

python "$SCRIPT" --ckpt "$CKPT" --task "$TASK" --episodes "$EPISODES" \
  --max_steps "$MAX_STEPS" --seed_base "$SEED" --constructor_seed 20260821 \
  --instr_mode "$MODE" --out "$OUT"
test -s "$OUT"
python - "$OUT" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("renderer") != "osmesa":
    raise RuntimeError("result is not bound to OSMesa")
if payload.get("source_sha256") != \
        "89ca38b043538c8bb932fea7e2bbc391ae12957e5a6aa7eb30f5b9b5ebf64679":
    raise RuntimeError("unexpected evaluator source")
print(f"renderer=osmesa successes={payload['successes']}/{payload['n_episodes']}")
PY
sha256sum "$SCRIPT" "$OUT"
