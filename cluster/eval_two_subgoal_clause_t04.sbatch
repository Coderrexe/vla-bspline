#!/bin/bash
#SBATCH --job-name=t04_clause_eval
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# Usage: ... CKPT TAG {compound,fixed_clock,event_clock} [n_action_steps=10]
set -euo pipefail
CKPT=${1:?checkpoint}; TAG=${2:?tag}; MODE=${3:?mode}; NAS=${4:-10}
case "$TAG" in *[!A-Za-z0-9_.-]*) exit 2 ;; esac
case "$MODE" in compound|fixed_clock|event_clock) ;; *) exit 2 ;; esac
case "$NAS" in ''|*[!0-9]*) exit 2 ;; esac
STATES_PER_TASK=${STATES_PER_TASK:-50}; export STATES_PER_TASK

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0 Mesa/22.2.4-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export MUJOCO_GL=osmesa
export TOKENIZERS_PARALLELISM=false
SOURCE="$HOME/scratch/vla_bspline/code_snapshots/language_source_20260820_v2"
SNAP="$HOME/scratch/vla_bspline/code_snapshots/two_subgoal_clause_20260821_v2"
EVALUATOR="$SNAP/scripts/eval/libero_locked_eval.py"
MAPPING="$HOME/scratch/vla_bspline/libero_t04_exact_clause_v1/meta/evaluator_dataset_task_mapping.json"
VARIANT="$HOME/scratch/vla_bspline/outputs/two_subgoal_eval_variants/${TAG}_${MODE}_nas${NAS}_${SLURM_JOB_ID}"
OUT="$HOME/scratch/vla_bspline/outputs/two_subgoal_eval/${TAG}_${MODE}_nas${NAS}_${SLURM_JOB_ID}.json"
TRAIN_MANIFEST="$CKPT/../../../reproducibility/launch_manifest.json"
test -s "$CKPT/model.safetensors"; test -s "$EVALUATOR"; test -s "$MAPPING"
test -s "$TRAIN_MANIFEST"; test ! -e "$VARIANT"; test ! -e "$OUT"
export PYTHONPATH="$SOURCE/src"

CKPT="$CKPT" VARIANT="$VARIANT" NAS="$NAS" TAG="$TAG" MODE="$MODE" \
MAPPING="$MAPPING" TRAIN_MANIFEST="$TRAIN_MANIFEST" python - <<'PY'
import hashlib,json,os
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""): h.update(b)
 return h.hexdigest()

src=Path(os.environ["CKPT"]).resolve(); dst=Path(os.environ["VARIANT"]); dst.mkdir(parents=True)
for item in src.iterdir():
 if item.name!="config.json": (dst/item.name).symlink_to(item.resolve())
cfg=json.load(open(src/"config.json")); original=cfg.get("n_action_steps")
cfg["n_action_steps"]=int(os.environ["NAS"]); (dst/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
mapping=json.load(open(os.environ["MAPPING"])); train=json.load(open(os.environ["TRAIN_MANIFEST"]))
assert train["mapping_sha256"]==mapping["mapping_sha256"]
gpu=os.popen("nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader").read().strip()
assert "L40S" in gpu.upper(),gpu
m={"schema":"two_subgoal_clause_eval_v1","tag":os.environ["TAG"],
"language_mode":os.environ["MODE"],"source_checkpoint":str(src),
"model_sha256":sha(src/"model.safetensors"),"original_n_action_steps":original,
"evaluated_n_action_steps":cfg["n_action_steps"],"renderer":"osmesa","task_ids":[0,4],
"states_per_task":int(os.environ["STATES_PER_TASK"]),"seed_base":100000,
"mapping_sha256":mapping["mapping_sha256"],"mapping_file_sha256":sha(os.environ["MAPPING"]),
"training_manifest_sha256":sha(os.environ["TRAIN_MANIFEST"]),"gpu":gpu,
"hardware_gate":"L40S","slurm_job_id":os.environ["SLURM_JOB_ID"]}
(dst/"evaluation_manifest.json").write_text(json.dumps(m,indent=2,sort_keys=True)+"\n")
PY

python "$EVALUATOR" --ckpt "$VARIANT" --suite libero_10 --task_id 0 --task_id 4 \
  --state_start 0 --states_per_task "$STATES_PER_TASK" --seed_base 100000 \
  --control_freq 20 --language_mode "$MODE" --out "$OUT"
OUT="$OUT" MODE="$MODE" STATES_PER_TASK="$STATES_PER_TASK" python - <<'PY'
import json,os
x=json.load(open(os.environ["OUT"])); n=int(os.environ["STATES_PER_TASK"])
assert x["language_mode"]==os.environ["MODE"] and x["task_ids"]==[0,4]
assert x["n_episodes"]==2*n and x["seed_base"]==100000
assert [t["task_description"] for t in x["tasks"]]==[
 "put both the alphabet soup and the tomato sauce in the basket",
 "put the white mug on the left plate and put the yellow and white mug on the right plate"]
print("validated",x["n_successes"],"/",x["n_episodes"])
PY
echo "=== completed exact-task eval tag=$TAG mode=$MODE nas=$NAS job=$SLURM_JOB_ID ==="
