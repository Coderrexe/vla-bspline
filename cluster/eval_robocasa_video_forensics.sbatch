#!/bin/bash
# One-seed failure-forensics rollout for a completed corrected RoboCasa model.
# Usage: sbatch eval_robocasa_video_forensics.sbatch CKPT TASK SEED MAX_STEPS TAG
#SBATCH --job-name=rc_video_forensics
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=%x-%j.out

set -euo pipefail
CKPT=${1:?checkpoint required}
TASK=${2:?task required}
SEED=${3:?episode seed required}
MAX_STEPS=${4:?maximum steps required}
TAG=${5:?safe tag required}
case "$TAG" in *[!A-Za-z0-9_.-]*) echo "unsafe tag" >&2; exit 64 ;; esac
for value in "$SEED" "$MAX_STEPS"; do
  case "$value" in ''|*[!0-9]*) echo "invalid integer: $value" >&2; exit 64 ;; esac
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

POLICY_SNAPSHOT="$HOME/scratch/vla_bspline/code_snapshots/robocasa_corrected_contract_v1_20260821"
SOURCE="$HOME/scratch/vla_bspline/code_snapshots/robocasa_video_forensics_v1_20260822"
SCRIPT="$SOURCE/robocasa_progress_eval.py"
export PYTHONPATH="$SOURCE:$POLICY_SNAPSHOT/src${PYTHONPATH:+:$PYTHONPATH}"
OUT_ROOT="$HOME/scratch/vla_bspline/outputs/robocasa_video_forensics"
OUT="$OUT_ROOT/${TAG}_${TASK}_seed${SEED}_${SLURM_JOB_ID}.json"
VIDEO_DIR="$OUT_ROOT/${TAG}_${TASK}_seed${SEED}_${SLURM_JOB_ID}_videos"

for path in "$CKPT/model.safetensors" "$CKPT/config.json" "$SCRIPT" "$SOURCE/robocasa_video.py"; do
  [[ -s "$path" ]] || { echo "missing forensic input: $path" >&2; exit 66; }
done
[[ ! -e "$OUT" && ! -e "$VIDEO_DIR" ]] || { echo "refusing overwrite" >&2; exit 73; }
mkdir -p "$OUT_ROOT"

python "$SCRIPT" \
  --ckpt "$CKPT" --task "$TASK" --episodes 1 --max_steps "$MAX_STEPS" \
  --seed_base "$SEED" --constructor_seed 20260821 --instr_mode env \
  --video_dir "$VIDEO_DIR" --video_stride 5 --out "$OUT"

python - "$OUT" <<'PY'
import json, os, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["n_episodes"] == 1 and len(payload["video_artifacts"]) == 1
video = payload["video_artifacts"][0]
assert os.path.isfile(video["path"]) and len(video["sha256"]) == 64
print({"successes": payload["successes"], "progress": payload["episodes"][0]["extrema"], "video": video})
PY
sha256sum "$SCRIPT" "$SOURCE/robocasa_video.py" "$OUT" "$VIDEO_DIR"/*.mp4
