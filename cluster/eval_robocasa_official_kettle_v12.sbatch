#!/bin/bash
# Target-split OSMesa evaluation for the official Kettle language experiment.
# Records every pilot rollout as video for failure-directed iteration.
# Usage: sbatch ... CKPT TAG MODE [EPISODES] [SEED]
# MODE: env | kettle_official_fixed | kettle_official_oracle
#SBATCH --job-name=rc_official_eval
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=%x-%j.out

set -euo pipefail

CKPT=${1:?checkpoint required}
TAG=${2:?tag required}
MODE=${3:?mode required}
EPISODES=${4:-10}
SEED=${5:-1000}
case "$TAG" in *[!A-Za-z0-9_.-]*) echo "unsafe tag" >&2; exit 64 ;; esac
case "$MODE" in env|kettle_official_fixed|kettle_official_oracle) ;; *) exit 64 ;; esac

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
EVAL_SNAPSHOT="$HOME/scratch/vla_bspline/code_snapshots/robocasa_progress_v12_official_20260822"
export PYTHONPATH="$POLICY_SNAPSHOT/src${PYTHONPATH:+:$PYTHONPATH}"
SCRIPT="$EVAL_SNAPSHOT/robocasa_progress_eval.py"
OUT_ROOT="$HOME/scratch/vla_bspline/outputs/robocasa_official_kettle_eval_v12"
VIDEO_DIR="$OUT_ROOT/videos/${TAG}_${MODE}_target_s${SEED}_n${EPISODES}_${SLURM_JOB_ID}"
OUT="$OUT_ROOT/${TAG}_${MODE}_target_s${SEED}_n${EPISODES}_${SLURM_JOB_ID}.json"

[[ -s "$CKPT/model.safetensors" && -s "$CKPT/config.json" ]] || exit 66
[[ ! -e "$OUT" && ! -e "$VIDEO_DIR" ]] || { echo "refusing overwrite" >&2; exit 73; }
printf '%s\n' \
  "5b73d27bef9c82455f9c2ec74a4710d93aea688f5bd066aa7d4f4fc4d85ab656  $SCRIPT" \
  "f1624203d4b954ca2a71827e3a5087a87e57f72195868af13dc09e693eae2b51  $EVAL_SNAPSHOT/robocasa_video.py" \
  "ffca203b363ddfd0055ea7f6e0b9c9e4de55c20d15f9bd57868fe205ba9af5fc  $EVAL_SNAPSHOT/robocasa_language_programs.py" | sha256sum --check --strict
mkdir -p "$OUT_ROOT"

python "$SCRIPT" \
  --ckpt "$CKPT" --task KettleBoiling --episodes "$EPISODES" \
  --max_steps 1500 --seed_base "$SEED" --constructor_seed 20260821 \
  --split target --instr_mode "$MODE" --out "$OUT" \
  --video_dir "$VIDEO_DIR" --video_stride 5

test -s "$OUT"
python - "$OUT" "$EPISODES" "$MODE" <<'PY'
import json, sys
result = json.load(open(sys.argv[1], encoding="utf-8"))
if result.get("schema") != "robocasa_progress_eval_v12_official_program":
    raise RuntimeError("unexpected result schema")
if result.get("renderer") != "osmesa" or result.get("environment_split") != "target":
    raise RuntimeError("result is not OSMesa/target")
if result.get("n_episodes") != int(sys.argv[2]):
    raise RuntimeError("incomplete result")
if result.get("instruction_mode") != sys.argv[3]:
    raise RuntimeError("mode mismatch")
if len(result.get("video_artifacts", [])) != int(sys.argv[2]):
    raise RuntimeError("missing rollout videos")
print(f"successes={result['successes']}/{result['n_episodes']}")
PY
sha256sum "$SCRIPT" "$OUT" "$VIDEO_DIR"/*.mp4
