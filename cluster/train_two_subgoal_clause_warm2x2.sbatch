#!/bin/bash
#SBATCH --job-name=t04_clause_2x2
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# Locked warm A/C x original/clause factorial.
# Usage: sbatch ... {A,C} {original,clause} [seed=1000] [steps=7500]
set -euo pipefail
ARM=${1:?A or C}; TREATMENT=${2:?original or clause}; SEED=${3:-1000}; STEPS=${4:-7500}
case "$ARM" in A|C) ;; *) exit 2 ;; esac
case "$TREATMENT" in original|clause) ;; *) exit 2 ;; esac
case "$SEED:$STEPS" in *[!0-9:]*) exit 2 ;; esac

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export EXPECTED_GPU_CLASS=${TRAIN_GPU_CLASS:-H100}
export RUN_SUFFIX=${RUN_SUFFIX:-v1}
WORKERS=${SLURM_CPUS_PER_TASK:-24}
case "$WORKERS" in ''|*[!0-9]*) exit 2 ;; esac
SOURCE="$HOME/scratch/vla_bspline/code_snapshots/language_source_20260820_v2"
SOURCE_TAR="$HOME/scratch/vla_bspline/code_snapshots/language_training_active_20260820_v2/lerobot_src.tar.gz"
ORIGINAL="$HOME/scratch/vla_bspline/libero_t04_exact_original_v1"
CLAUSE="$HOME/scratch/vla_bspline/libero_t04_exact_clause_v1"
PROV="$CLAUSE/meta/two_subgoal_provenance.json"
MAPPING="$CLAUSE/meta/evaluator_dataset_task_mapping.json"
DATASET="$ORIGINAL"; [ "$TREATMENT" = original ] || DATASET="$CLAUSE"
if [ "$ARM" = A ]; then WARM_NAME=baselineA_100k; else WARM_NAME=Cn8_100k; fi
WARM="$HOME/scratch/vla_bspline/outputs/$WARM_NAME/checkpoints/last/pretrained_model"
OUT="$HOME/scratch/vla_bspline/outputs/${ARM}warm_t04exact_${TREATMENT}_s${SEED}_${STEPS}_${RUN_SUFFIX}"
STATS="$SOURCE/src/lerobot/policies/smolvla_spline/spline_stats_libero_v2_h24_n8.json"
MANIFEST_DIR="$HOME/scratch/vla_bspline/manifests/two_subgoal_clause_2x2"
MANIFEST="$MANIFEST_DIR/${ARM}_${TREATMENT}_s${SEED}_${STEPS}_${SLURM_JOB_ID}.json"
test -s "$WARM/model.safetensors"; test -s "$PROV"; test -s "$MAPPING"
test -s "$SOURCE_TAR"; test -d "$DATASET/meta"; test ! -e "$OUT"
test ! -e "$MANIFEST"; mkdir -p "$MANIFEST_DIR"
export PYTHONPATH="$SOURCE/src"

EPISODES=$(python - "$PROV" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); e=sorted(a["episode_index"] for a in x["annotations"])
assert len(e)==71 and len(set(e))==71 and x["frame_count"]==19378
print(json.dumps(e,separators=(",",":")))
PY
)

ARM="$ARM" TREATMENT="$TREATMENT" SEED="$SEED" STEPS="$STEPS" WARM="$WARM" \
DATASET="$DATASET" ORIGINAL="$ORIGINAL" CLAUSE="$CLAUSE" PROV="$PROV" \
MAPPING="$MAPPING" SOURCE_TAR="$SOURCE_TAR" MANIFEST="$MANIFEST" EPISODES="$EPISODES" \
python - <<'PY'
import hashlib,json,os
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(8<<20),b""): h.update(b)
 return h.hexdigest()

p=json.load(open(os.environ["PROV"])); mapping=json.load(open(os.environ["MAPPING"]))
assert p["episode_count"]==71 and p["frame_count"]==19378
assert p["mapping_sha256"]==mapping["mapping_sha256"]
assert all(v["byte_identical"] for v in p["physical_non_language_verification"]["columns"].values())
warm=Path(os.environ["WARM"]); cfg=json.load(open(warm/"config.json"))
arm=os.environ["ARM"]; assert cfg["type"]==("smolvla" if arm=="A" else "smolvla_spline")
gpu=os.popen("nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader").read().strip()
expected=os.environ["EXPECTED_GPU_CLASS"]
assert expected.lower() in gpu.lower(),(expected,gpu)
m={"schema":"two_subgoal_clause_warm2x2_v1","arm":arm,
"treatment":os.environ["TREATMENT"],"seed":int(os.environ["SEED"]),
"steps":int(os.environ["STEPS"]),"batch_size":32,"episodes":json.loads(os.environ["EPISODES"]),
"episode_count":71,"frame_count":19378,"effective_epochs":int(os.environ["STEPS"])*32/19378,
"dataset_root":str(Path(os.environ["DATASET"]).resolve()),
"warm_checkpoint":str(warm.resolve()),"warm_model_sha256":sha(warm/"model.safetensors"),
"warm_config_sha256":sha(warm/"config.json"),"mapping_sha256":p["mapping_sha256"],
"dataset_provenance_sha256":sha(os.environ["PROV"]),
"mapping_file_sha256":sha(os.environ["MAPPING"]),
"source_tar_sha256":sha(os.environ["SOURCE_TAR"]),"gpu":gpu,
"slurm_job_id":os.environ["SLURM_JOB_ID"],"hardware_gate":expected,
"run_suffix":os.environ["RUN_SUFFIX"]}
Path(os.environ["MANIFEST"]).write_text(json.dumps(m,indent=2,sort_keys=True)+"\n")
print(json.dumps(m,indent=2,sort_keys=True))
PY

COMMON=(
 --policy.push_to_hub=false --policy.load_vlm_weights=true --policy.pretrained_path="$WARM"
 --dataset.repo_id=HuggingFaceVLA/libero --dataset.root="$DATASET" --dataset.episodes="$EPISODES"
 --output_dir="$OUT" --steps="$STEPS" --batch_size=32 --num_workers="$WORKERS" --seed="$SEED"
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
cp "$PROV" "$OUT/reproducibility/two_subgoal_provenance.json"
cp "$MAPPING" "$OUT/reproducibility/evaluator_dataset_task_mapping.json"
cp "$SOURCE_TAR" "$OUT/reproducibility/lerobot_src.tar.gz"
[ "$ARM" = A ] || cp "$STATS" "$OUT/reproducibility/spline_stats_libero_v2_h24_n8.json"
sha256sum "$FINAL/model.safetensors" > "$OUT/reproducibility/model.sha256"
echo "=== completed exact-task 2x2 arm=$ARM treatment=$TREATMENT job=$SLURM_JOB_ID ==="
