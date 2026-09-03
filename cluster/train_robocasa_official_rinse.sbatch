#!/bin/bash
# Matched Rinse original-caption vs official phase-label fine-tune.
# Usage: sbatch train_robocasa_official_rinse.sbatch original|official [seed] [steps]
#SBATCH --job-name=rc_official_rinse
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=%x-%j.out
set -euo pipefail

ARM=${1:?original or official}
SEED=${2:-1000}
STEPS=${3:-30000}
case "$ARM" in
  original) DATASET_NAME=robocasa_composite_seen ;;
  official) DATASET_NAME=robocasa_composite_seen_official_rinse_v1 ;;
  *) exit 64 ;;
esac
module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache" WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
WORKERS=${SLURM_CPUS_PER_TASK:-24}
case "$WORKERS" in ''|*[!0-9]*) echo "invalid worker count: $WORKERS" >&2; exit 64 ;; esac
SNAP="$HOME/scratch/vla_bspline/code_snapshots/robocasa_corrected_contract_v1_20260821"
export PYTHONPATH="$SNAP/src${PYTHONPATH:+:$PYTHONPATH}"
POLICY="$SNAP/src/lerobot/policies/smolvla_spline"
STATS=spline_stats_robocasa_composite_seen_all_v2_n8h24.json
DATASET="$HOME/scratch/vla_bspline/$DATASET_NAME"
INIT="$HOME/scratch/vla_bspline/outputs/rccorrected_easy4_task_C_s1000_100000_2322332/checkpoints/100000/pretrained_model"
OUT="$HOME/scratch/vla_bspline/outputs/rcofficial_rinse_${ARM}_C_s${SEED}_${STEPS}_${SLURM_JOB_ID}"
MDIR="$HOME/scratch/vla_bspline/manifests/robocasa_official_rinse_training"
MANIFEST="$MDIR/rcofficial_rinse_${ARM}_C_s${SEED}_${STEPS}_${SLURM_JOB_ID}.json"
[[ -d "$DATASET" && -s "$INIT/model.safetensors" ]] || exit 66
[[ ! -e "$OUT" && ! -e "$MANIFEST" ]] || exit 73
[[ "$ARM" != official || -s "$DATASET/official_language_overlay_manifest.json" ]] || exit 66
mkdir -p "$MDIR"
EPISODES=$(python - <<'PY'
import json
x=list(range(3528,4037)); assert len(x)==509; print(json.dumps(x,separators=(",",":")))
PY
)

python - "$MANIFEST" "$SNAP" "$DATASET" "$INIT" "$OUT" "$ARM" "$SEED" "$STEPS" <<'PY'
import hashlib, inspect, json, os, platform, subprocess, sys
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from lerobot.policies.smolvla_spline import event_targets, modeling_smolvla_spline, stats_contract
manifest,snap,dataset,init,out,arm,seed,steps=sys.argv[1:]
def sha(path):
 d=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""): d.update(b)
 return d.hexdigest()
policy=os.path.join(snap,"src/lerobot/policies/smolvla_spline")
sources={n:os.path.join(policy,f) for n,f in {
 "modeling":"modeling_smolvla_spline.py","configuration":"configuration_smolvla_spline.py",
 "event_targets":"event_targets.py","stats_contract":"stats_contract.py","corrected_stats":"spline_stats_robocasa_composite_seen_all_v2_n8h24.json"}.items()}
if os.path.realpath(inspect.getfile(modeling_smolvla_spline)) != os.path.realpath(sources["modeling"]): raise RuntimeError("source")
if sha(sources["corrected_stats"])!="287be4152eb77bd9301a8d62e26123574356f9cb543e1b8050406e783ec78007": raise RuntimeError("stats")
if sha(inspect.getfile(event_targets))!="9335cae53e277aa76975d59f43c266b0a1beee17d12d7737fd2d09db3016e9f6": raise RuntimeError("events")
episode_meta=os.path.join(dataset,"meta/episodes/chunk-000/file-000.parquet")
rows=pq.read_table(episode_meta,columns=["episode_index","source_prefix"]).to_pylist()
by={int(r["episode_index"]):str(r["source_prefix"]) for r in rows}
if {by[i].split("/")[-2] for i in range(3528,4037)}!={"RinseSinkBasin"}: raise RuntimeError("range")
cols=["index","episode_index","frame_index","timestamp","action","observation.state","next.reward","next.done"]
t=ds.dataset(os.path.join(dataset,"data"),format="parquet").to_table(
 columns=cols+["task_index"],filter=(ds.field("episode_index")>=3528)&(ds.field("episode_index")<=4036)
).sort_by([("episode_index","ascending"),("frame_index","ascending")])
if len(t)!=211036: raise RuntimeError(f"frames {len(t)}")
physical=hashlib.sha256(); language=hashlib.sha256()
for b in t.select(cols).to_batches(max_chunksize=32768): physical.update(b.serialize().to_pybytes())
for b in t.select(["episode_index","frame_index","task_index"]).to_batches(max_chunksize=32768): language.update(b.serialize().to_pybytes())
overlay=os.path.join(dataset,"official_language_overlay_manifest.json")
overlay_sha=None
if arm=="official":
 o=json.load(open(overlay));
 if o.get("task")!="RinseSinkBasin" or not o.get("physical_action_stream_matches_official"): raise RuntimeError("overlay")
 overlay_sha=sha(overlay)
d={"schema":"robocasa_official_rinse_train_v1","job_id":os.environ["SLURM_JOB_ID"],"arm":arm,
 "seed":int(seed),"steps":int(steps),"episodes":509,"frames":len(t),"episode_range":[3528,4036],
 "initialization":os.path.realpath(init),"initial_model_sha256":sha(os.path.join(init,"model.safetensors")),
 "initial_config_sha256":sha(os.path.join(init,"config.json")),"dataset":os.path.realpath(dataset),
 "dataset_tasks_sha256":sha(os.path.join(dataset,"meta/tasks.parquet")),"dataset_episode_metadata_sha256":sha(episode_meta),
 "selected_physical_transition_sha256":physical.hexdigest(),"selected_language_schedule_sha256":language.hexdigest(),
 "official_overlay_manifest_sha256":overlay_sha,"source_snapshot":os.path.realpath(snap),
 "source_sha256":{n:sha(p) for n,p in sources.items()},"normalization_contract":{"stats_sha256":sha(sources["corrected_stats"]),
 "event_boundary_semantics":stats_contract.EVENT_BOUNDARY_SEMANTICS,"checkpoint_embedding_required":True},
 "optimizer_semantics":"fresh optimizer/schedule from identical corrected task checkpoint","output":os.path.realpath(out),
 "python":sys.version,"platform":platform.platform(),"gpu":subprocess.check_output(["nvidia-smi","--query-gpu=name,uuid,driver_version","--format=csv,noheader"],text=True).strip()}
with open(manifest,"x") as f: json.dump(d,f,indent=2,sort_keys=True); f.write("\n")
PY

python -m lerobot.scripts.lerobot_train \
 --policy.type=smolvla_spline --policy.n_ctrl=8 --policy.min_seg=8 --policy.horizon_max=24 \
 --policy.action_layout=robocasa12 --policy.spline_stats_file_v2="$STATS" --policy.predict_duration=true \
 --policy.push_to_hub=false --policy.load_vlm_weights=true --policy.pretrained_path="$INIT" \
 --dataset.repo_id=ember-lab-berkeley/robocasa365-target-composite-seen --dataset.root="$DATASET" \
 --dataset.episodes="$EPISODES" --output_dir="$OUT" --steps="$STEPS" --batch_size=32 \
 --num_workers="$WORKERS" --seed="$SEED" --save_freq=10000 --log_freq=100 --wandb.enable=false
test -s "$OUT/checkpoints/last/pretrained_model/model.safetensors"
cp "$MANIFEST" "$OUT/launch_manifest.json"
