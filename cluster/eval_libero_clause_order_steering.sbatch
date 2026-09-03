#!/bin/bash
#SBATCH --job-name=t04_order_smoke
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# Tiny-first paired clause-order evaluator. Usage:
# sbatch eval_libero_clause_order_steering.sbatch CKPT TAG [states=2] [model_sha] [source_config_sha]
set -euo pipefail

CKPT=${1:?checkpoint pretrained_model directory}
TAG=${2:?filesystem-safe tag}
STATES=${3:-2}
EXPECTED_MODEL_SHA=${4:-}
EXPECTED_SOURCE_CONFIG_SHA=${5:-}
case "$TAG" in *[!A-Za-z0-9_.-]*) echo "unsafe tag" >&2; exit 2 ;; esac
case "$STATES" in ''|*[!0-9]*) echo "invalid states" >&2; exit 2 ;; esac
[ "$STATES" -gt 0 ]

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0 Mesa/22.2.4-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

SOURCE=${ORDER_STEERING_SOURCE_DIR:-$HOME/vla_bspline/cluster}
STAGE="$HOME/scratch/vla_bspline/code_snapshots/clause_order_${SLURM_JOB_ID}"
OUTDIR="$HOME/scratch/vla_bspline/outputs/clause_order_steering"
VARIANT="$HOME/scratch/vla_bspline/outputs/clause_order_variants/${TAG}_nas10_${SLURM_JOB_ID}"
OUT0="$OUTDIR/${TAG}_n${STATES}_r0_${SLURM_JOB_ID}.json"
OUT1="$OUTDIR/${TAG}_n${STATES}_r1_${SLURM_JOB_ID}.json"
GATE="$OUTDIR/${TAG}_n${STATES}_replay_gate_${SLURM_JOB_ID}.json"
test -s "$CKPT/model.safetensors"; test -s "$CKPT/config.json"
if [ -n "$EXPECTED_MODEL_SHA" ]; then
  echo "$EXPECTED_MODEL_SHA  $CKPT/model.safetensors" | sha256sum --check --strict
fi
if [ -n "$EXPECTED_SOURCE_CONFIG_SHA" ]; then
  echo "$EXPECTED_SOURCE_CONFIG_SHA  $CKPT/config.json" | sha256sum --check --strict
fi
test ! -e "$STAGE"; test ! -e "$VARIANT"; test ! -e "$OUT0"; test ! -e "$OUT1"; test ! -e "$GATE"
mkdir -p "$STAGE" "$OUTDIR" "$VARIANT"
for name in libero_clause_order_steering.py libero_hidden_state.py libero_locked_eval_v2.py \
  libero_duration_ablation_v2.py duration_intervention_v2.py deterministic_eval.py
do
  test -s "$SOURCE/$name"; cp "$SOURCE/$name" "$STAGE/$name"
done
sha256sum "$STAGE"/*.py > "$STAGE/SHA256SUMS"

CKPT="$CKPT" VARIANT="$VARIANT" python - <<'PY'
import json,os
from pathlib import Path
source=Path(os.environ["CKPT"]).resolve(); variant=Path(os.environ["VARIANT"])
for item in source.iterdir():
    if item.name != "config.json": (variant/item.name).symlink_to(item.resolve())
config=json.load(open(source/"config.json")); config["n_action_steps"]=10
(variant/"config.json").write_text(json.dumps(config,indent=2,sort_keys=True)+"\n")
PY

for replay in 0 1; do
  if [ "$replay" = 0 ]; then OUT="$OUT0"; else OUT="$OUT1"; fi
  python "$STAGE/libero_clause_order_steering.py" --ckpt "$VARIANT" \
    --states_per_task "$STATES" --replicate_index "$replay" --out "$OUT"
done

OUT0="$OUT0" OUT1="$OUT1" GATE="$GATE" STAGE="$STAGE" TAG="$TAG" python - <<'PY'
import hashlib,json,os,tempfile
from pathlib import Path
a=json.load(open(os.environ["OUT0"])); b=json.load(open(os.environ["OUT1"])); failures=[]
for key in ("protocol","checkpoint_config_sha256","model_sha256","policy_type","n_action_steps","renderer_backend","suite","task_ids","state_ids","seed_base","control_frequency_hz","transition_oracle","transition_oracle_is_privileged","paired_initial_state_checks","paired_order_effects","summary"):
    if a.get(key)!=b.get(key): failures.append({"kind":"run_identity","key":key,"a":a.get(key),"b":b.get(key)})
def index(payload):
    result={}
    for row in payload["episodes"]:
        key=(row["task_id"],row["state_id"],row["condition"])
        if key in result: raise RuntimeError(f"duplicate {key}")
        result[key]=row
    return result
left,right=index(a),index(b)
if set(left)!=set(right): failures.append({"kind":"coordinate_grid"})
fields=("task_description","goal_order","clause_order","official_goal_predicates","initial_goal_flags","first_goal_hit_steps","first_completed_goal_indices","first_completed_goal_predicates","requested_first_goal_completed_first","switch_steps","final_goal_flags","predicate_trace","success","steps","env_seed_u32","policy_seed_u64","initial_raw_observation_sha256","initial_mujoco_integration_state_sha256","compiled_model_xml_sha256","first_action_sha256","executed_action_trace_protocol","executed_action_trace_sha256")
for coordinate in sorted(set(left)&set(right)):
    for field in fields:
        if left[coordinate].get(field)!=right[coordinate].get(field):
            failures.append({"kind":"episode_replay","coordinate":coordinate,"field":field,"a":left[coordinate].get(field),"b":right[coordinate].get(field)})
def sha(path): return hashlib.file_digest(open(path,"rb"),"sha256").hexdigest()
gate={"schema_version":1,"protocol":"libero_clause_order_exact_replay_gate_v1","passed":not failures,"tag":os.environ["TAG"],"primary_result":os.environ["OUT0"],"replay_result":os.environ["OUT1"],"primary_result_sha256":sha(os.environ["OUT0"]),"replay_result_sha256":sha(os.environ["OUT1"]),"source_snapshot_sha256s":{p.name:sha(p) for p in sorted(Path(os.environ["STAGE"]).glob("*.py"))},"n_coordinates":len(left),"failures":failures}
destination=Path(os.environ["GATE"]); fd,tmp=tempfile.mkstemp(prefix="."+destination.name+".",dir=destination.parent)
with os.fdopen(fd,"w") as stream:
    json.dump(gate,stream,indent=2,sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
try: os.link(tmp,destination)
finally: os.unlink(tmp)
print(json.dumps({"passed":gate["passed"],"n_coordinates":gate["n_coordinates"],"failures":len(failures),"summary":a["summary"]},sort_keys=True))
if failures: raise SystemExit("FATAL: clause-order replay mismatch")
PY
