#!/bin/bash
#SBATCH --job-name=t04_clause_evalv2
#SBATCH --partition=gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=05:30:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# Immutable two-replay OSMesa gate for the exact t0/t4 clause experiment.
# Usage: sbatch ... CKPT TAG {compound,fixed_clock,event_clock} [states=50]
# The first result is claim-eligible only when its same-allocation replay gate
# passes.  Both raw result files and their image-array sidecars are retained.
set -euo pipefail

CKPT=${1:?checkpoint pretrained_model directory}
TAG=${2:?filesystem-safe experiment tag}
MODE=${3:?compound, fixed_clock, or event_clock}
STATES_PER_TASK=${4:-50}
NAS=10
case "$TAG" in *[!A-Za-z0-9_.-]*) echo "unsafe tag" >&2; exit 2 ;; esac
case "$MODE" in compound|fixed_clock|event_clock|duration_clock) ;; *) echo "invalid mode" >&2; exit 2 ;; esac
case "$STATES_PER_TASK" in ''|*[!0-9]*) echo "invalid state count" >&2; exit 2 ;; esac
[ "$STATES_PER_TASK" -gt 0 ]

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0 Mesa/22.2.4-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export EXPECTED_TRAIN_GPU_CLASS=${EXPECTED_TRAIN_GPU_CLASS:-H100}

SOURCE=${T04_EVAL_V2_SOURCE_DIR:-$HOME/vla_bspline/cluster}
STAGE="$HOME/scratch/vla_bspline/code_snapshots/two_subgoal_clause_evalv2_${SLURM_JOB_ID}"
OUTDIR="$HOME/scratch/vla_bspline/outputs/two_subgoal_eval_v2"
VARIANT="$HOME/scratch/vla_bspline/outputs/two_subgoal_eval_variants_v2/${TAG}_${MODE}_nas${NAS}_${SLURM_JOB_ID}"
TRAIN_MANIFEST="$CKPT/../../../reproducibility/launch_manifest.json"
OUT_A="$OUTDIR/${TAG}_${MODE}_nas${NAS}_r0_${SLURM_JOB_ID}.json"
OUT_B="$OUTDIR/${TAG}_${MODE}_nas${NAS}_r1_${SLURM_JOB_ID}.json"
ARRAYS_A="${OUT_A%.json}.npz"
ARRAYS_B="${OUT_B%.json}.npz"
GATE="$OUTDIR/${TAG}_${MODE}_nas${NAS}_replay_gate_${SLURM_JOB_ID}.json"
test -s "$CKPT/model.safetensors"; test -s "$CKPT/config.json"; test -s "$TRAIN_MANIFEST"
test ! -e "$STAGE"; test ! -e "$VARIANT"; test ! -e "$OUT_A"; test ! -e "$OUT_B"; test ! -e "$GATE"
mkdir -p "$STAGE" "$OUTDIR"
for name in libero_renderer_probe.py libero_hidden_state.py libero_locked_eval_v2.py \
  libero_duration_ablation_v2.py duration_intervention_v2.py deterministic_eval.py
do
  test -s "$SOURCE/$name"
  cp "$SOURCE/$name" "$STAGE/$name"
done
sha256sum "$STAGE"/*.py > "$STAGE/SHA256SUMS"

# The training cadence is intentionally independent of the locked control
# cadence.  Make an isolated checkpoint view; never edit the trained config.
CKPT="$CKPT" VARIANT="$VARIANT" NAS="$NAS" python - <<'PY'
import json,os
from pathlib import Path
source=Path(os.environ["CKPT"]).resolve()
variant=Path(os.environ["VARIANT"])
variant.mkdir(parents=True)
for item in source.iterdir():
    if item.name != "config.json":
        (variant/item.name).symlink_to(item.resolve())
config=json.load(open(source/"config.json"))
config["n_action_steps"]=int(os.environ["NAS"])
(variant/"config.json").write_text(json.dumps(config,indent=2,sort_keys=True)+"\n")
PY

CKPT="$CKPT" TRAIN_MANIFEST="$TRAIN_MANIFEST" STAGE="$STAGE" python - <<'PY'
import hashlib,json,os
from pathlib import Path
train=json.load(open(os.environ["TRAIN_MANIFEST"]))
assert train["episode_count"] == 71 and train["frame_count"] == 19378
assert train["steps"] == 7500 and train["batch_size"] == 32
assert train["mapping_sha256"]
assert train["hardware_gate"] == os.environ["EXPECTED_TRAIN_GPU_CLASS"]
assert Path(os.environ["CKPT"]).resolve().as_posix().endswith("/pretrained_model")
print(json.dumps({"train_manifest_sha256": hashlib.file_digest(open(os.environ["TRAIN_MANIFEST"],"rb"),"sha256").hexdigest(), "source_snapshot_files": sorted(p.name for p in Path(os.environ["STAGE"]).glob("*.py"))}, sort_keys=True))
PY

for replay in 0 1; do
  if [ "$replay" = 0 ]; then OUT="$OUT_A"; ARRAYS="$ARRAYS_A"; else OUT="$OUT_B"; ARRAYS="$ARRAYS_B"; fi
  python "$STAGE/libero_renderer_probe.py" \
    --ckpt "$VARIANT" --suite libero_10 --task_ids 0,4 --state_start 0 \
    --states_per_task "$STATES_PER_TASK" --seed_base 100000 --control_freq 20 \
    --language_mode "$MODE" --replicate_index "$replay" --out "$OUT" --arrays_out "$ARRAYS"
  test -s "$OUT"; test -s "$ARRAYS"
done

OUT_A="$OUT_A" OUT_B="$OUT_B" GATE="$GATE" MODE="$MODE" STAGE="$STAGE" \
TRAIN_MANIFEST="$TRAIN_MANIFEST" TAG="$TAG" python - <<'PY'
import hashlib,json,os,tempfile
from pathlib import Path

first=json.load(open(os.environ["OUT_A"]))
second=json.load(open(os.environ["OUT_B"]))
required={"renderer_backend":"osmesa", "task_ids":[0,4], "state_ids": list(range(len(first["state_ids"]))), "seed_base":100000, "control_frequency_hz":20, "language_mode":os.environ["MODE"]}
failures=[]
for key, expected in required.items():
    if first.get(key) != expected or second.get(key) != expected:
        failures.append({"kind":"required_identity", "key":key, "first":first.get(key), "second":second.get(key), "expected":expected})
for key in ("checkpoint_config_sha256", "model_sha256", "suite", "source"):
    if first.get(key) != second.get(key):
        failures.append({"kind":"run_identity", "key":key, "first":first.get(key), "second":second.get(key)})
def index(payload):
    records={}
    for row in payload["episodes"]:
        coordinate=(row["task_id"], row["state_id"])
        if coordinate in records: raise RuntimeError(f"duplicate coordinate {coordinate}")
        records[coordinate]=row
    return records
left,right=index(first),index(second)
if set(left) != set(right): failures.append({"kind":"coordinate_grid", "left":sorted(left), "right":sorted(right)})
fields=("task_description", "language_mode", "reviewed_clauses", "prompts_used", "switch_steps", "final_clause_index", "duration_clock_horizon_max", "duration_clock_chunks", "duration_clock_switch_trigger", "initial_raw_observation_sha256", "initial_processed_observation_sha256", "initial_mujoco_integration_state_sha256", "compiled_model_xml_sha256", "first_action_sha256", "executed_action_trace_protocol", "executed_action_trace_sha256", "success", "steps")
for coordinate in sorted(set(left)&set(right)):
    for field in fields:
        if left[coordinate].get(field) != right[coordinate].get(field):
            failures.append({"kind":"episode_replay", "coordinate":coordinate, "field":field, "first":left[coordinate].get(field), "second":right[coordinate].get(field)})
for payload in (first,second):
    if payload["n_episodes"] != len(payload["episodes"]): failures.append({"kind":"episode_count"})
    for row in payload["episodes"]:
        if row["executed_action_trace_protocol"] != "executed_numpy_action_v1" or not isinstance(row["executed_action_trace_sha256"],str) or len(row["executed_action_trace_sha256"]) != 64:
            failures.append({"kind":"malformed_action_trace", "coordinate":[row["task_id"],row["state_id"]]})
def sha(path):
    with open(path,"rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()
gate={"schema_version":2,"protocol":"two_subgoal_t04_osmesa_exact_replay_gate_v2","passed":not failures,"tag":os.environ["TAG"],"mode":os.environ["MODE"],"primary_result":os.environ["OUT_A"],"replay_result":os.environ["OUT_B"],"primary_result_sha256":sha(os.environ["OUT_A"]),"replay_result_sha256":sha(os.environ["OUT_B"]),"training_manifest":os.environ["TRAIN_MANIFEST"],"training_manifest_sha256":sha(os.environ["TRAIN_MANIFEST"]),"source_snapshot_sha256s":{p.name:sha(p) for p in sorted(Path(os.environ["STAGE"]).glob("*.py"))},"n_coordinates":len(left),"checked_episode_fields":list(fields),"failures":failures}
destination=Path(os.environ["GATE"])
descriptor,temporary=tempfile.mkstemp(prefix="."+destination.name+".",dir=destination.parent)
with os.fdopen(descriptor,"w") as stream:
    json.dump(gate,stream,indent=2,sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
try: os.link(temporary,destination)
finally: os.unlink(temporary)
print(json.dumps({"passed":gate["passed"],"n_coordinates":gate["n_coordinates"],"n_failures":len(failures)},sort_keys=True))
if failures: raise SystemExit("FATAL: OSMesa t0/t4 replay mismatch; refusing claim-eligible result")
PY
echo "=== passed two-subgoal OSMesa replay gate tag=$TAG mode=$MODE job=$SLURM_JOB_ID ==="
