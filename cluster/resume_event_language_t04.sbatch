#!/bin/bash
#SBATCH --job-name=evlang_resume_t04
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# Fail-safe resume for a timed-out event-language cell.  Submit only with an
# afternotok dependency on the primary job.
set -euo pipefail
ARM=${1:?expected A or C}; SEED=${2:?expected seed}
case "$ARM" in A|C) ;; *) exit 2 ;; esac
module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
SOURCE="$HOME/scratch/vla_bspline/code_snapshots/language_source_20260820_v2"
OUT="$HOME/scratch/vla_bspline/outputs/${ARM}eventT04_epmix_v1_s${SEED}"
FINAL="$OUT/checkpoints/last/pretrained_model"
TRAIN_CONFIG="$FINAL/train_config.json"
test -s "$TRAIN_CONFIG"
export PYTHONPATH="$SOURCE/src"
python - <<PY
import json
x=json.load(open("$TRAIN_CONFIG")); assert int(x["seed"])==int("$SEED")
PY
python -m lerobot.scripts.lerobot_train --config_path="$TRAIN_CONFIG" --resume=true
test -s "$FINAL/model.safetensors"
SOURCE_TAR="$HOME/scratch/vla_bspline/code_snapshots/language_training_active_20260820_v2/lerobot_src.tar.gz"
STATS="$SOURCE/src/lerobot/policies/smolvla_spline/spline_stats_libero_v2_h24_n8.json"
REPRO="$OUT/reproducibility"
test -s "$REPRO/event_language_provenance.json"
test -s "$REPRO/lerobot_src.tar.gz"
ARM="$ARM" SEED="$SEED" OUT="$OUT" FINAL="$FINAL" SOURCE_TAR="$SOURCE_TAR" STATS="$STATS" \
python - <<'PY'
import hashlib, json, os
from pathlib import Path
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
out=Path(os.environ["OUT"]); final=Path(os.environ["FINAL"]); arm=os.environ["ARM"]
cfg=json.load(open(final/"config.json")); train=json.load(open(final/"train_config.json"))
assert int(train["seed"])==int(os.environ["SEED"])
manifest={
    "arm":arm, "seed":int(os.environ["SEED"]),
    "resume_slurm_job_id":os.environ["SLURM_JOB_ID"],
    "policy_type":cfg["type"], "n_action_steps":cfg["n_action_steps"],
    "frozen_source_tar_sha256":sha(os.environ["SOURCE_TAR"]),
    "dataset_provenance_sha256":sha(out/"reproducibility/event_language_provenance.json"),
    "model_sha256":sha(final/"model.safetensors"),
}
if arm=="C": manifest["spline_stats_sha256"]=sha(os.environ["STATS"])
(out/"reproducibility/manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
PY
echo "=== completed event-language resume arm=$ARM seed=$SEED job=$SLURM_JOB_ID ==="
