#!/bin/bash
#SBATCH --job-name=evlang_program_t04
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# Counterfactual long-horizon steering protocol. The two subtasks are issued as
# separate clauses; a commanded gripper close->open release advances the
# language pointer and drains residual actions generated under the old clause.
# Usage: ... CKPT TAG N_ACTION_STEPS
#   {event_clock,event_clock_swap,fixed_clock,fixed_clock_swap,first_only}
set -euo pipefail
CKPT=${1:?checkpoint}; TAG=${2:?tag}; NAS=${3:?actions/replan}; MODE=${4:?language mode}
case "$TAG" in *[!A-Za-z0-9_.-]*) exit 2 ;; esac
case "$NAS" in ''|*[!0-9]*) exit 2 ;; esac
case "$MODE" in event_clock|event_clock_swap|fixed_clock|fixed_clock_swap|first_only) ;; *) exit 2 ;; esac
STATES_PER_TASK=${STATES_PER_TASK:-50}; export STATES_PER_TASK

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0 Mesa/22.2.4-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export MUJOCO_GL=osmesa
export TOKENIZERS_PARALLELISM=false
SOURCE="$HOME/scratch/vla_bspline/code_snapshots/language_source_20260820_v2"
SNAP="$HOME/scratch/vla_bspline/code_snapshots/event_language_program_20260821_v1"
EVALUATOR="$SNAP/scripts/eval/libero_locked_eval.py"
VARIANT="$HOME/scratch/vla_bspline/outputs/event_language_variants/${TAG}_${MODE}_nas${NAS}_${SLURM_JOB_ID}"
OUT="$HOME/scratch/vla_bspline/outputs/event_language_eval/${TAG}_${MODE}_nas${NAS}_${SLURM_JOB_ID}.json"
test -s "$CKPT/model.safetensors"; test -s "$EVALUATOR"; test ! -e "$VARIANT"; test ! -e "$OUT"
export PYTHONPATH="$SOURCE/src"

CKPT="$CKPT" VARIANT="$VARIANT" NAS="$NAS" TAG="$TAG" MODE="$MODE" python - <<'PY'
import hashlib,json,os
from pathlib import Path
src=Path(os.environ["CKPT"]).resolve(); dst=Path(os.environ["VARIANT"]); dst.mkdir(parents=True)
for item in src.iterdir():
    if item.name != "config.json": (dst/item.name).symlink_to(item.resolve())
cfg=json.load(open(src/"config.json")); original=cfg.get("n_action_steps"); cfg["n_action_steps"]=int(os.environ["NAS"])
(dst/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
m={"tag":os.environ["TAG"],"language_mode":os.environ["MODE"],"source_checkpoint":str(src),
"model_sha256":sha(src/"model.safetensors"),"original_n_action_steps":original,
"evaluated_n_action_steps":cfg["n_action_steps"],"renderer":"osmesa","task_ids":[0,4],
"states_per_task":int(os.environ["STATES_PER_TASK"]),"slurm_job_id":os.environ["SLURM_JOB_ID"]}
(dst/"evaluation_manifest.json").write_text(json.dumps(m,indent=2)+"\n")
PY

python "$EVALUATOR" --ckpt "$VARIANT" --suite libero_10 --task_id 0 --task_id 4 \
  --state_start 0 --states_per_task "$STATES_PER_TASK" --seed_base 100000 \
  --control_freq 20 --language_mode "$MODE" --out "$OUT"
OUT="$OUT" MODE="$MODE" STATES_PER_TASK="$STATES_PER_TASK" python - <<'PY'
import json,os
x=json.load(open(os.environ["OUT"])); n=int(os.environ["STATES_PER_TASK"])
assert x["language_mode"]==os.environ["MODE"]; assert x["task_ids"]==[0,4]
assert x["n_episodes"]==2*n
print("validated",x["n_successes"],"/",x["n_episodes"])
PY
echo "=== completed program eval tag=$TAG mode=$MODE nas=$NAS job=$SLURM_JOB_ID ==="
