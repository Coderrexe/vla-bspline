#!/bin/bash
# Claim-bearing target-split Kettle evaluation with training-exact phase prompts
# and burner-specific final instructions selected at the phase transition.
# Usage: sbatch ... CKPT TAG MODE [EPISODES] [SEED]
# MODE: env | kettle_official_fixed | kettle_official_oracle |
#       kettle_official_fixed_slowplace | kettle_official_oracle_slowplace
#SBATCH --job-name=rc_kettle_v15
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
case "$TAG" in *[!A-Za-z0-9_.-]*) exit 64 ;; esac
case "$MODE" in
  env|kettle_official_fixed|kettle_official_oracle|kettle_official_fixed_slowplace|kettle_official_oracle_slowplace) ;;
  *) exit 64 ;;
esac

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0 Mesa/22.2.4-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/robocasa_env/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

POLICY="$HOME/scratch/vla_bspline/code_snapshots/robocasa_corrected_contract_v1_20260821"
EVAL="$HOME/scratch/vla_bspline/code_snapshots/robocasa_progress_v15_exact_phase_20260822"
export PYTHONPATH="$POLICY/src${PYTHONPATH:+:$PYTHONPATH}"
SCRIPT="$EVAL/robocasa_progress_eval.py"
ROOT="$HOME/scratch/vla_bspline/outputs/robocasa_official_kettle_eval_v15"
VIDEO="$ROOT/videos/${TAG}_${MODE}_target_s${SEED}_n${EPISODES}_${SLURM_JOB_ID}"
OUT="$ROOT/${TAG}_${MODE}_target_s${SEED}_n${EPISODES}_${SLURM_JOB_ID}.json"
[[ -s "$CKPT/model.safetensors" && -s "$CKPT/config.json" ]] || exit 66
[[ ! -e "$OUT" && ! -e "$VIDEO" ]] || exit 73
printf '%s\n' \
  "21d05518ef92625486f15de7799b53c0704bd1d29baf50c5d6ae66b150e75c0c  $SCRIPT" \
  "ac14c9d0f13042cfe9c7d9ed2ea7f3182c2568fd0d7ff0fe4f6ca6fff7d4b914  $EVAL/robocasa_language_programs.py" \
  "f1624203d4b954ca2a71827e3a5087a87e57f72195868af13dc09e693eae2b51  $EVAL/robocasa_video.py" | sha256sum --check --strict
mkdir -p "$ROOT"

python "$SCRIPT" --ckpt "$CKPT" --task KettleBoiling --episodes "$EPISODES" \
  --max_steps 1500 --seed_base "$SEED" --constructor_seed 20260821 --split target \
  --instr_mode "$MODE" --out "$OUT" --video_dir "$VIDEO" --video_stride 5

python - "$OUT" "$EPISODES" "$MODE" <<'PY'
import json, re, sys
d=json.load(open(sys.argv[1]))
if d.get("schema") != "robocasa_progress_eval_v15_exact_phase_prompts": raise RuntimeError("schema")
if d.get("renderer") != "osmesa" or d.get("environment_split") != "target": raise RuntimeError("domain")
if d.get("n_episodes") != int(sys.argv[2]) or d.get("instruction_mode") != sys.argv[3]: raise RuntimeError("support")
if len(d.get("video_artifacts",[])) != int(sys.argv[2]): raise RuntimeError("videos")
if sys.argv[3] != "env":
    for episode in d["episodes"]:
        prompts=episode["instruction_sentences"]
        if prompts[:2] != ["pick up the kettle from the counter","place the kettle on the stove burner"]:
            raise RuntimeError("phase prompt mismatch")
        if episode["final_instruction_pointer"] == 2 and not re.fullmatch(
            r"turn on the (?:rear-|front-)?(?:left|right|center) burner where the kettle is placed", prompts[2]
        ):
            raise RuntimeError(f"non-specific burner prompt: {prompts[2]!r}")
print(f"successes={d['successes']}/{d['n_episodes']}")
PY
sha256sum "$OUT" "$VIDEO"/*.mp4
