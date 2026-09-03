#!/bin/bash
#SBATCH --job-name=evlang_train_t04
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=09:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# Matched A/C discovery-seed training on the corrected t0/t4 event-language
# overlay.  This uses the exact frozen source snapshot captured for the active
# legacy-language factorial, preventing code drift from masquerading as a label
# improvement. Usage: sbatch train_event_language_t04.sbatch A 1000

set -euo pipefail
ARM=${1:?expected A or C}
SEED=${2:?expected integer seed}
case "$ARM" in A|C) ;; *) exit 2 ;; esac
case "$SEED" in ''|*[!0-9]*) exit 2 ;; esac

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

DATASET="$HOME/scratch/vla_bspline/libero_event_t04_epmix_v1"
SOURCE="$HOME/scratch/vla_bspline/code_snapshots/language_source_20260820_v2"
SOURCE_TAR="$HOME/scratch/vla_bspline/code_snapshots/language_training_active_20260820_v2/lerobot_src.tar.gz"
OUT="$HOME/scratch/vla_bspline/outputs/${ARM}eventT04_epmix_v1_s${SEED}"
STATS="$SOURCE/src/lerobot/policies/smolvla_spline/spline_stats_libero_v2_h24_n8.json"
PROVENANCE="$DATASET/meta/event_language_provenance.json"

test -s "$SOURCE/src/lerobot/scripts/lerobot_train.py"
test -s "$PROVENANCE"
test ! -e "$OUT"
mkdir -p "$OUT/reproducibility"
cp "$PROVENANCE" "$OUT/reproducibility/event_language_provenance.json"
cp "$SOURCE_TAR" "$OUT/reproducibility/lerobot_src.tar.gz"
[ "$ARM" = A ] || cp "$STATS" "$OUT/reproducibility/spline_stats_libero_v2_h24_n8.json"
export PYTHONPATH="$SOURCE/src"
python - <<'PY'
import lerobot, os
actual = os.path.realpath(lerobot.__file__)
expected = os.path.realpath(os.path.expanduser(
    "~/scratch/vla_bspline/code_snapshots/language_source_20260820_v2/src/lerobot"
))
assert actual.startswith(expected + os.sep), (actual, expected)
print("frozen lerobot source:", actual)
PY

COMMON_ARGS=(
  --policy.push_to_hub=false
  --policy.load_vlm_weights=true
  --dataset.repo_id=HuggingFaceVLA/libero
  --dataset.root="$DATASET"
  --output_dir="$OUT"
  --steps=100000
  --batch_size=32
  --num_workers=24
  --seed="$SEED"
  --save_freq=50000
  --log_freq=100
  --wandb.enable=false
)
if [ "$ARM" = A ]; then
  python -m lerobot.scripts.lerobot_train \
    --policy.type=smolvla --policy.n_action_steps=10 "${COMMON_ARGS[@]}"
else
  test -s "$STATS"
  python -m lerobot.scripts.lerobot_train \
    --policy.type=smolvla_spline --policy.predict_duration=true \
    --policy.n_ctrl=8 --policy.min_seg=8 --policy.horizon_max=24 \
    --policy.n_action_steps=5 \
    --policy.spline_stats_file_v2=spline_stats_libero_v2_h24_n8.json \
    "${COMMON_ARGS[@]}"
fi

FINAL="$OUT/checkpoints/last/pretrained_model"
test -s "$FINAL/model.safetensors"
ARM="$ARM" SEED="$SEED" OUT="$OUT" FINAL="$FINAL" SOURCE_TAR="$SOURCE_TAR" \
python - <<'PY'
import hashlib, json, os
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

out=Path(os.environ["OUT"]); final=Path(os.environ["FINAL"])
cfg=json.load(open(final/"config.json")); train=json.load(open(final/"train_config.json"))
arm=os.environ["ARM"]; seed=int(os.environ["SEED"])
assert int(train["seed"]) == seed
assert cfg["type"] == ("smolvla" if arm == "A" else "smolvla_spline")
manifest={
  "arm":arm, "seed":seed, "slurm_job_id":os.environ["SLURM_JOB_ID"],
  "policy_type":cfg["type"], "n_action_steps":cfg["n_action_steps"],
  "frozen_source_tar_sha256":sha(os.environ["SOURCE_TAR"]),
  "dataset_provenance_sha256":sha(out/"reproducibility/event_language_provenance.json"),
  "model_sha256":sha(final/"model.safetensors"),
}
if arm == "C":
  assert (cfg["n_ctrl"],cfg["min_seg"],cfg["horizon_max"]) == (8,8,24)
  manifest.update(n_ctrl=8,min_seg=8,horizon_max=24,
    spline_stats_sha256=sha(out/"reproducibility/spline_stats_libero_v2_h24_n8.json"))
(out/"reproducibility/manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(manifest,indent=2))
PY
echo "=== completed event-language training arm=$ARM seed=$SEED job=$SLURM_JOB_ID ==="
