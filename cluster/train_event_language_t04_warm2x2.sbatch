#!/bin/bash
#SBATCH --job-name=evlang_warm2x2
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# Primary corrected-language 2x2.
# Usage: sbatch ... {A,C} {original,event} [seed=1000] [steps=7500]
set -euo pipefail
ARM=${1:?A or C}; TREATMENT=${2:?original or event}; SEED=${3:-1000}; STEPS=${4:-7500}
case "$ARM" in A|C) ;; *) exit 2 ;; esac
case "$TREATMENT" in original|event) ;; *) exit 2 ;; esac
case "$SEED:$STEPS" in *[!0-9:]*) exit 2 ;; esac

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
SOURCE="$HOME/scratch/vla_bspline/code_snapshots/language_source_20260820_v2"
SOURCE_TAR="$HOME/scratch/vla_bspline/code_snapshots/language_training_active_20260820_v2/lerobot_src.tar.gz"
ORIGINAL="$HOME/scratch/vla_bspline/hf_cache/hub/datasets--HuggingFaceVLA--libero/snapshots/86958911c0f959db2bbbdb107eb3e17c5f9c798e"
EVENT="$HOME/scratch/vla_bspline/libero_event_t04_full_v1"
LABELS="$HOME/scratch/vla_bspline/event_language_t04_20260821_v1/seg_labels.jsonl"
AUDIT="$HOME/scratch/vla_bspline/outputs/language_dataset_audit/event_t04_2321848.json"
DATASET="$ORIGINAL"; [ "$TREATMENT" = original ] || DATASET="$EVENT"
if [ "$ARM" = A ]; then WARM_NAME=baselineA_100k; else WARM_NAME=Cn8_100k; fi
WARM="$HOME/scratch/vla_bspline/outputs/$WARM_NAME/checkpoints/last/pretrained_model"
OUT="$HOME/scratch/vla_bspline/outputs/${ARM}warmT04_${TREATMENT}_s${SEED}_${STEPS}"
STATS="$SOURCE/src/lerobot/policies/smolvla_spline/spline_stats_libero_v2_h24_n8.json"
MANIFEST_DIR="$HOME/scratch/vla_bspline/manifests/event_language_warm2x2"
MANIFEST="$MANIFEST_DIR/${ARM}_${TREATMENT}_s${SEED}_${STEPS}_${SLURM_JOB_ID}.json"
test -s "$WARM/model.safetensors"; test -s "$AUDIT"; test -s "$LABELS"; test -d "$DATASET/meta"
test ! -e "$OUT"; test ! -e "$MANIFEST"; mkdir -p "$MANIFEST_DIR"
export PYTHONPATH="$SOURCE/src"

# Exact episode set is derived from the immutable corrected label manifest.
EPISODES=$(python - "$LABELS" <<'PY'
import json,sys
r=[json.loads(x) for x in open(sys.argv[1])]; e=sorted({int(x["episode_index"]) for x in r})
assert len(e)==81
print(json.dumps(e,separators=(",",":")))
PY
)

ARM="$ARM" TREATMENT="$TREATMENT" SEED="$SEED" STEPS="$STEPS" WARM="$WARM" \
ORIGINAL="$ORIGINAL" EVENT="$EVENT" DATASET="$DATASET" LABELS="$LABELS" AUDIT="$AUDIT" \
SOURCE_TAR="$SOURCE_TAR" MANIFEST="$MANIFEST" EPISODES="$EPISODES" python - <<'PY'
import hashlib,json,os
from pathlib import Path
import numpy as np
import pyarrow.dataset as pads

def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()

episodes=json.loads(os.environ["EPISODES"]); original=Path(os.environ["ORIGINAL"]); event=Path(os.environ["EVENT"])
audit=json.load(open(os.environ["AUDIT"])); assert audit["quality_gate_passed"]
event_prov=json.load(open(event/"meta/event_language_provenance.json"))
assert event_prov["mix"]["granular_episodes"]==81
filt=pads.field("episode_index").isin(episodes)
columns=["episode_index","frame_index","index","timestamp","action","observation.state"]
left=pads.dataset(str(original/"data"),format="parquet").to_table(columns=columns,filter=filt).to_pandas().sort_values("index")
right=pads.dataset(str(event/"data"),format="parquet").to_table(columns=columns,filter=filt).to_pandas().sort_values("index")
assert len(left)==len(right)==21301
for col in columns:
 if col in ("action","observation.state"):
  assert np.array_equal(np.stack(left[col]),np.stack(right[col])),col
 else: assert np.array_equal(left[col].to_numpy(),right[col].to_numpy()),col
warm_cfg=json.load(open(Path(os.environ["WARM"])/"config.json"))
expected="smolvla" if os.environ["ARM"]=="A" else "smolvla_spline"
assert warm_cfg["type"]==expected
gpu=os.popen("nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader").read().strip()
assert "H100" in gpu,gpu
m={"schema":"event_language_warm2x2_v1","arm":os.environ["ARM"],"treatment":os.environ["TREATMENT"],
"seed":int(os.environ["SEED"]),"steps":int(os.environ["STEPS"]),"batch_size":32,
"episodes":episodes,"episode_count":81,"frame_count":21301,
"effective_epochs":int(os.environ["STEPS"])*32/21301,
"dataset_root":str(Path(os.environ["DATASET"]).resolve()),"warm_checkpoint":str(Path(os.environ["WARM"]).resolve()),
"warm_model_sha256":sha(Path(os.environ["WARM"])/"model.safetensors"),
"warm_config_sha256":sha(Path(os.environ["WARM"])/"config.json"),
"labels_sha256":sha(os.environ["LABELS"]),"quality_audit_sha256":sha(os.environ["AUDIT"]),
"source_tar_sha256":sha(os.environ["SOURCE_TAR"]),"gpu":gpu,"slurm_job_id":os.environ["SLURM_JOB_ID"],
"paired_non_language_columns_exact":columns,
"event_dataset_provenance_sha256":sha(event/"meta/event_language_provenance.json")}
Path(os.environ["MANIFEST"]).write_text(json.dumps(m,indent=2,sort_keys=True)+"\n")
print(json.dumps(m,indent=2))
PY

COMMON=(
 --policy.push_to_hub=false --policy.load_vlm_weights=true --policy.pretrained_path="$WARM"
 --dataset.repo_id=HuggingFaceVLA/libero --dataset.root="$DATASET" --dataset.episodes="$EPISODES"
 --output_dir="$OUT" --steps="$STEPS" --batch_size=32 --num_workers=24 --seed="$SEED"
 --save_freq=2500 --log_freq=100 --wandb.enable=false
)
if [ "$ARM" = A ]; then
 python -m lerobot.scripts.lerobot_train --policy.type=smolvla --policy.n_action_steps=10 "${COMMON[@]}"
else
 python -m lerobot.scripts.lerobot_train --policy.type=smolvla_spline --policy.predict_duration=true \
  --policy.n_ctrl=8 --policy.min_seg=8 --policy.horizon_max=24 --policy.n_action_steps=5 \
  --policy.spline_stats_file_v2=spline_stats_libero_v2_h24_n8.json "${COMMON[@]}"
fi
FINAL="$OUT/checkpoints/last/pretrained_model"; test -s "$FINAL/model.safetensors"
mkdir -p "$OUT/reproducibility"
cp "$MANIFEST" "$OUT/reproducibility/launch_manifest.json"
cp "$AUDIT" "$OUT/reproducibility/label_quality_audit.json"
cp "$SOURCE_TAR" "$OUT/reproducibility/lerobot_src.tar.gz"
[ "$ARM" = A ] || cp "$STATS" "$OUT/reproducibility/spline_stats_libero_v2_h24_n8.json"
sha256sum "$FINAL/model.safetensors" > "$OUT/reproducibility/model.sha256"
echo "=== completed warm2x2 arm=$ARM treatment=$TREATMENT job=$SLURM_JOB_ID ==="
