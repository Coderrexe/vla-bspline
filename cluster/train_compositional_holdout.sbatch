#!/bin/bash
# t0+t1-only compositional training. No LIBERO-finetuned checkpoint is loaded.
# Usage: sbatch ... {A,C} {whole,clause} [seed=1000] [steps=20000]
#SBATCH --job-name=comp_holdout_train
#SBATCH --partition=gpu
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

set -euo pipefail
ARM=${1:?A or C}
TREATMENT=${2:?whole or clause}
SEED=${3:-1000}
STEPS=${4:-20000}
case "$ARM" in A|C) ;; *) echo "invalid arm" >&2; exit 2 ;; esac
case "$TREATMENT" in whole|clause) ;; *) echo "invalid treatment" >&2; exit 2 ;; esac
case "$SEED:$STEPS" in *[!0-9:]*) echo "invalid numeric argument" >&2; exit 2 ;; esac

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

SOURCE="$HOME/scratch/vla_bspline/code_snapshots/language_source_20260820_v2"
SOURCE_TAR="$HOME/scratch/vla_bspline/code_snapshots/language_training_active_20260820_v2/lerobot_src.tar.gz"
ORIGINAL="$HOME/scratch/vla_bspline/libero_comp_t01_original_v1"
CLAUSE="$HOME/scratch/vla_bspline/libero_comp_t01_clause_v1"
PROV="$CLAUSE/meta/two_subgoal_provenance.json"
MAPPING="$CLAUSE/meta/evaluator_dataset_task_mapping.json"
STATS="$HOME/scratch/vla_bspline/artifacts/compositional_holdout_v1/spline_stats_libero_t01_v2_h24_n8.json"
DATASET="$ORIGINAL"; [ "$TREATMENT" = clause ] && DATASET="$CLAUSE"
OUT="$HOME/scratch/vla_bspline/outputs/comp_${ARM}_${TREATMENT}_s${SEED}_${STEPS}_v1"
MANIFEST="$HOME/scratch/vla_bspline/manifests/compositional_holdout/${ARM}_${TREATMENT}_s${SEED}_${STEPS}_${SLURM_JOB_ID}.json"

test -d "$SOURCE/src/lerobot"; test -s "$SOURCE_TAR"
test -s "$PROV"; test -s "$MAPPING"; test -s "$STATS"
test -d "$DATASET/data"; test ! -e "$OUT"; test ! -e "$MANIFEST"
mkdir -p "$(dirname "$MANIFEST")"
export PYTHONPATH="$SOURCE/src"

EPISODES=$(python - "$PROV" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
episodes=sorted(int(a["episode_index"]) for a in p["annotations"])
assert len(episodes)==82 and len(set(episodes))==82
assert p["frame_count"]==22273
assert {int(a["suite_task_id"]) for a in p["annotations"]}=={0,1}
print(json.dumps(episodes,separators=(",",":")))
PY
)

ARM="$ARM" TREATMENT="$TREATMENT" SEED="$SEED" STEPS="$STEPS" \
DATASET="$DATASET" PROV="$PROV" MAPPING="$MAPPING" STATS="$STATS" \
SOURCE_TAR="$SOURCE_TAR" MANIFEST="$MANIFEST" EPISODES="$EPISODES" python - <<'PY'
import hashlib,json,os
from pathlib import Path

def sha(path):
    with open(path,"rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()

p=json.load(open(os.environ["PROV"])); m=json.load(open(os.environ["MAPPING"]));
s=json.load(open(os.environ["STATS"])); episodes=json.loads(os.environ["EPISODES"])
assert p["mapping_sha256"]==m["mapping_sha256"]
assert len(episodes)==82 and p["frame_count"]==22273
assert s["episode_selection"]["episode_indices"]==episodes
dataset_stats_path=Path(os.environ["DATASET"])/"meta/stats.json"
dataset_stats=json.load(open(dataset_stats_path))
for feature in ("action","observation.state","task_index"):
    assert dataset_stats[feature]["count"]==[22273]
held_out={7,9,25,29,30,41,63,74,82,83,96,98,124,135,148,160,161,163,
          171,174,188,195,196,205,208,221,222,223,237,246,250,256,265,
          266,275,281,286,289,293,297,318,331,373}
assert not set(episodes)&held_out
gpu=os.popen("nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader").read().strip()
assert "h100" in gpu.lower(),gpu
record={
  "schema":"libero_compositional_holdout_train_v1",
  "arm":os.environ["ARM"],"treatment":os.environ["TREATMENT"],
  "seed":int(os.environ["SEED"]),"steps":int(os.environ["STEPS"]),
  "batch_size":32,"episode_count":82,"frame_count":22273,"episodes":episodes,
  "training_suite_tasks":[0,1],"held_out_suite_task":7,
  "held_out_episode_count":43,"held_out_episode_intersection":[],
  "initialization":"SmolVLM2-500M-Video-Instruct; new action expert; no LIBERO-finetuned checkpoint",
  "dataset_root":str(Path(os.environ["DATASET"]).resolve()),
  "lerobot_training_stats_sha256":sha(dataset_stats_path),
  "lerobot_training_stats_frame_count":22273,
  "dataset_provenance_sha256":sha(os.environ["PROV"]),
  "mapping_sha256":p["mapping_sha256"],"mapping_file_sha256":sha(os.environ["MAPPING"]),
  "spline_stats_sha256":sha(os.environ["STATS"]),
  "spline_stats_episode_count":s["episode_selection"]["used_count"],
  "source_tar_sha256":sha(os.environ["SOURCE_TAR"]),
  "gpu":gpu,"slurm_job_id":os.environ["SLURM_JOB_ID"],
}
Path(os.environ["MANIFEST"]).write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
print(json.dumps(record,indent=2,sort_keys=True))
PY

COMMON=(
  --policy.push_to_hub=false --policy.load_vlm_weights=true
  --dataset.repo_id=HuggingFaceVLA/libero --dataset.root="$DATASET" --dataset.episodes="$EPISODES"
  --output_dir="$OUT" --steps="$STEPS" --batch_size=32 --num_workers=24 --seed="$SEED"
  --save_freq=5000 --log_freq=100 --wandb.enable=false
)
if [ "$ARM" = A ]; then
  python -m lerobot.scripts.lerobot_train \
    --policy.type=smolvla --policy.n_action_steps=10 "${COMMON[@]}"
else
  python -m lerobot.scripts.lerobot_train \
    --policy.type=smolvla_spline --policy.predict_duration=true \
    --policy.n_ctrl=8 --policy.min_seg=8 --policy.horizon_max=24 \
    --policy.n_action_steps=5 --policy.spline_stats_file_v2="$STATS" "${COMMON[@]}"
fi

FINAL="$OUT/checkpoints/last/pretrained_model"
test -s "$FINAL/model.safetensors"; test -s "$FINAL/config.json"
mkdir -p "$OUT/reproducibility"
cp "$MANIFEST" "$OUT/reproducibility/launch_manifest.json"
cp "$PROV" "$OUT/reproducibility/two_subgoal_provenance.json"
cp "$MAPPING" "$OUT/reproducibility/evaluator_dataset_task_mapping.json"
cp "$STATS" "$OUT/reproducibility/spline_stats_libero_t01_v2_h24_n8.json"
cp "$SOURCE_TAR" "$OUT/reproducibility/lerobot_src.tar.gz"
sha256sum "$FINAL/model.safetensors" "$FINAL/config.json" \
  > "$OUT/reproducibility/checkpoint.sha256"
echo "=== completed compositional training arm=$ARM treatment=$TREATMENT job=$SLURM_JOB_ID ==="
