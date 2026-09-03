#!/bin/bash
# Oracle-gated four-clause Rinse mechanism probe with failure videos.
# Usage: sbatch ... CKPT EPISODES TAG [SEED]
# This is a diagnostic upper bound: simulator predicates choose clause changes.
#SBATCH --job-name=rc_rinse_program
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=%x-%j.out

set -euo pipefail

CKPT=${1:?checkpoint required}
EPISODES=${2:-10}
TAG=${3:?tag required}
SEED=${4:-1000}
case "$TAG" in *[!A-Za-z0-9_.-]*) echo "unsafe tag: $TAG" >&2; exit 64 ;; esac

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
PROGRAM_SNAPSHOT="$HOME/scratch/vla_bspline/code_snapshots/robocasa_rinse_program_v10_20260822"
export PYTHONPATH="$POLICY_SNAPSHOT/src${PYTHONPATH:+:$PYTHONPATH}"
SCRIPT="$PROGRAM_SNAPSHOT/robocasa_progress_eval.py"
OUT_ROOT="$HOME/scratch/vla_bspline/outputs/robocasa_rinse_program_v10"
VIDEO_DIR="$OUT_ROOT/videos_${TAG}_${SLURM_JOB_ID}"
OUT="$OUT_ROOT/${TAG}_RinseSinkBasin_rinse_oracle_regions_s${SEED}_n${EPISODES}_${SLURM_JOB_ID}.json"

[[ -f "$CKPT/model.safetensors" && -f "$CKPT/config.json" ]] || {
  echo "missing checkpoint artifacts: $CKPT" >&2; exit 66;
}
[[ ! -e "$OUT" && ! -e "$VIDEO_DIR" ]] || { echo "refusing overwrite" >&2; exit 73; }
printf '%s\n' \
  "20b48665914356219ef576334fa0851318c56e6f2face6f4a717d4534d6c9ae4  $SCRIPT" \
  "f1624203d4b954ca2a71827e3a5087a87e57f72195868af13dc09e693eae2b51  $PROGRAM_SNAPSHOT/robocasa_video.py" \
  "2eca4bf5374257af8df1ec00d01685d6b355b63b375b0ae9c9a5841f5a6c30c7  $PROGRAM_SNAPSHOT/robocasa_language_programs.py" | sha256sum --check --strict
mkdir -p "$OUT_ROOT"

python "$SCRIPT" \
  --ckpt "$CKPT" --task RinseSinkBasin --episodes "$EPISODES" \
  --max_steps 1350 --seed_base "$SEED" --constructor_seed 20260821 \
  --instr_mode rinse_oracle_regions --video_dir "$VIDEO_DIR" \
  --video_stride 10 --out "$OUT"

test -s "$OUT"
python - "$OUT" "$EPISODES" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
if result.get("renderer") != "osmesa":
    raise RuntimeError("result is not OSMesa-bound")
if result.get("instruction_mode") != "rinse_oracle_regions":
    raise RuntimeError("wrong instruction program")
if result.get("n_episodes") != int(sys.argv[2]):
    raise RuntimeError("incomplete result")
print(f"successes={result['successes']}/{result['n_episodes']}")
PY
sha256sum "$SCRIPT" "$OUT"
