#!/bin/bash
# Exact same-allocation replay gate for the t0-Scene2 soup+cream holdout.
# Usage: sbatch ... CKPT TAG {whole,clause} [states=2] [heldout|seen_t0|seen_t1]
#SBATCH --job-name=comp_holdout_eval
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

set -euo pipefail
CKPT=${1:?checkpoint pretrained_model directory}
TAG=${2:?filesystem-safe tag}
MODE=${3:?whole or clause}
STATES=${4:-2}
TASK_VARIANT=${5:-heldout}
case "$TAG" in *[!A-Za-z0-9_.-]*) echo "unsafe tag" >&2; exit 2 ;; esac
case "$MODE" in whole|clause) ;; *) echo "invalid language mode" >&2; exit 2 ;; esac
case "$TASK_VARIANT" in heldout|seen_t0|seen_t1) ;; *) echo "invalid task variant" >&2; exit 2 ;; esac
case "$STATES" in ''|*[!0-9]*) echo "invalid state count" >&2; exit 2 ;; esac
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

SOURCE=${COMPOSITIONAL_EVAL_SOURCE_DIR:?immutable evaluator source directory}
STAGE="$HOME/scratch/vla_bspline/code_snapshots/comp_eval_${SLURM_JOB_ID}"
OUTDIR="$HOME/scratch/vla_bspline/outputs/compositional_holdout_eval"
VARIANT="$HOME/scratch/vla_bspline/outputs/compositional_holdout_variants/${TAG}_nas10_${SLURM_JOB_ID}"
OUT0="$OUTDIR/${TAG}_${MODE}_${TASK_VARIANT}_n${STATES}_r0_${SLURM_JOB_ID}.json"
OUT1="$OUTDIR/${TAG}_${MODE}_${TASK_VARIANT}_n${STATES}_r1_${SLURM_JOB_ID}.json"
GATE="$OUTDIR/${TAG}_${MODE}_${TASK_VARIANT}_n${STATES}_replay_gate_${SLURM_JOB_ID}.json"
VIDEO_ROOT="$HOME/scratch/vla_bspline/outputs/compositional_holdout_videos/${TAG}_${MODE}_${TASK_VARIANT}_${SLURM_JOB_ID}"
TRAIN_MANIFEST="$CKPT/../../../reproducibility/launch_manifest.json"

test -s "$CKPT/model.safetensors"; test -s "$CKPT/config.json"
test -s "$TRAIN_MANIFEST"; test ! -e "$STAGE"; test ! -e "$VARIANT"
test ! -e "$OUT0"; test ! -e "$OUT1"; test ! -e "$GATE"
mkdir -p "$STAGE" "$OUTDIR" "$VARIANT"
for name in libero_compositional_holdout.py libero_hidden_state.py \
  libero_locked_eval_v2.py libero_duration_ablation_v2.py duration_intervention_v2.py \
  deterministic_eval.py
do
  test -s "$SOURCE/$name"; cp "$SOURCE/$name" "$STAGE/$name"
done
sha256sum "$STAGE"/*.py > "$STAGE/SHA256SUMS"

CKPT="$CKPT" VARIANT="$VARIANT" python - <<'PY'
import json,os
from pathlib import Path
source=Path(os.environ["CKPT"]).resolve(); variant=Path(os.environ["VARIANT"])
for item in source.iterdir():
    if item.name!="config.json": (variant/item.name).symlink_to(item.resolve())
config=json.load(open(source/"config.json")); config["n_action_steps"]=10
(variant/"config.json").write_text(json.dumps(config,indent=2,sort_keys=True)+"\n")
PY

TRAIN_MANIFEST="$TRAIN_MANIFEST" MODE="$MODE" python - <<'PY'
import json,os
m=json.load(open(os.environ["TRAIN_MANIFEST"]));
assert m["schema"]=="libero_compositional_holdout_train_v1"
assert m["treatment"]==os.environ["MODE"]
assert m["training_suite_tasks"]==[0,1] and m["held_out_suite_task"]==7
assert m["episode_count"]==82 and m["frame_count"]==22273
assert m["held_out_episode_intersection"]==[]
assert "no LIBERO-finetuned checkpoint" in m["initialization"]
print("training provenance gate passed")
PY

for replay in 0 1; do
  if [ "$replay" = 0 ]; then OUT="$OUT0"; else OUT="$OUT1"; fi
  VIDEO_ARGS=()
  if [ "$STATES" -le 2 ]; then VIDEO_ARGS=(--video_dir "$VIDEO_ROOT/r${replay}"); fi
  python "$STAGE/libero_compositional_holdout.py" \
    --ckpt "$VARIANT" --language_mode "$MODE" --task_variant "$TASK_VARIANT" \
    --states "$STATES" \
    --replicate_index "$replay" --out "$OUT" "${VIDEO_ARGS[@]}"
done

OUT0="$OUT0" OUT1="$OUT1" GATE="$GATE" STAGE="$STAGE" TAG="$TAG" \
MODE="$MODE" TASK_VARIANT="$TASK_VARIANT" TRAIN_MANIFEST="$TRAIN_MANIFEST" python - <<'PY'
import hashlib,json,os,tempfile
from pathlib import Path
a=json.load(open(os.environ["OUT0"])); b=json.load(open(os.environ["OUT1"])); failures=[]
variant=os.environ["TASK_VARIANT"]
variant_requirements={
  "heldout":{
    "protocol":"libero_same_scene_compositional_holdout_v1",
    "source_task_id":0,
    "transition_oracle":"held_out_current_subgoal_predicate_true_then_clear_queue_v1",
    "scene_control":"official_libero_long_t0_scene2; only terminal goal conjunction changed",
  },
  "seen_t0":{
    "protocol":"libero_compositional_seen_competence_v1",
    "source_task_id":0,
    "transition_oracle":"seen_current_subgoal_predicate_true_then_clear_queue_v1",
    "scene_control":"official_libero_long_t0_scene2; goal conjunction unchanged",
  },
  "seen_t1":{
    "protocol":"libero_compositional_seen_competence_v1",
    "source_task_id":1,
    "transition_oracle":"seen_current_subgoal_predicate_true_then_clear_queue_v1",
    "scene_control":"official_libero_long_t1_scene2; goal conjunction unchanged",
  },
}
required={**variant_requirements[variant],
          "language_mode":os.environ["MODE"],"task_variant":variant,
          "renderer_backend":"osmesa","seed_base":200000,
          "control_frequency_hz":20}
for key,expected in required.items():
    if a.get(key)!=expected or b.get(key)!=expected:
        failures.append({"kind":"required_identity","key":key})
for key in ("checkpoint_config_sha256","model_sha256","policy_type","n_action_steps",
            "state_ids","transition_oracle","scene_control","summary"):
    if a.get(key)!=b.get(key): failures.append({"kind":"run_identity","key":key})
def index(payload):
    result={}
    for row in payload["episodes"]:
        key=(row["state_id"],row["condition"])
        if key in result: raise RuntimeError(f"duplicate {key}")
        result[key]=row
    return result
left,right=index(a),index(b)
if set(left)!=set(right): failures.append({"kind":"coordinate_grid"})
fields=("source_task_description","requested_task_description","language_mode","goal_order",
        "task_variant","prompts_used_unique_ordered","source_goal_predicates","held_out_goal_predicates",
        "initial_goal_flags","first_goal_hit_steps","first_completed_goal_indices",
        "requested_first_goal_completed_first","switch_steps","final_goal_flags",
        "first_goal_retained_final","goal_ever_false_after_first_hit","predicate_trace",
        "success","steps","env_seed_u32","policy_seed_u64","initial_raw_observation_sha256",
        "initial_mujoco_integration_state_sha256","compiled_model_xml_sha256",
        "first_action_sha256","executed_action_trace_protocol","executed_action_trace_sha256")
for coordinate in sorted(set(left)&set(right)):
    for field in fields:
        if left[coordinate].get(field)!=right[coordinate].get(field):
            failures.append({"kind":"episode_replay","coordinate":coordinate,"field":field})
def sha(path):
    with open(path,"rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()
gate={"schema_version":1,"protocol":"libero_compositional_holdout_exact_replay_gate_v1",
      "passed":not failures,"tag":os.environ["TAG"],"language_mode":os.environ["MODE"],
      "task_variant":variant,
      "primary_result":os.environ["OUT0"],"replay_result":os.environ["OUT1"],
      "primary_result_sha256":sha(os.environ["OUT0"]),
      "replay_result_sha256":sha(os.environ["OUT1"]),
      "training_manifest":os.environ["TRAIN_MANIFEST"],
      "training_manifest_sha256":sha(os.environ["TRAIN_MANIFEST"]),
      "source_snapshot_sha256s":{p.name:sha(p) for p in sorted(Path(os.environ["STAGE"]).glob("*.py"))},
      "n_coordinates":len(left),"checked_episode_fields":list(fields),"failures":failures}
destination=Path(os.environ["GATE"]); fd,tmp=tempfile.mkstemp(prefix="."+destination.name+".",dir=destination.parent)
with os.fdopen(fd,"w") as stream:
    json.dump(gate,stream,indent=2,sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
try: os.link(tmp,destination)
finally: os.unlink(tmp)
print(json.dumps({"passed":gate["passed"],"coordinates":len(left),"summary":a["summary"]},sort_keys=True))
if failures: raise SystemExit("FATAL: compositional replay mismatch")
PY

sha256sum --check --strict "$STAGE/SHA256SUMS"
echo "=== passed compositional replay gate job=$SLURM_JOB_ID ==="
