#!/bin/bash
# Config-only decode-rate intervention on a corrected-contract RoboCasa checkpoint.
# The source checkpoint is immutable; only exec_rate_ratio is changed in a
# job-ID-keyed variant.  MODE may be oracle or a co-scaled fixedk clock.
# Usage: sbatch ... CKPT TASK EPISODES MAX_STEPS TAG MODE RATE [SEED]
# RATE > 1 executes the same predicted spline more slowly.
#SBATCH --job-name=rc_corr_rate
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --exclude=r817u29n05
#SBATCH --output=%x-%j.out

set -euo pipefail

CKPT=${1:?checkpoint required}
TASK=${2:?task required}
EPISODES=${3:?episode count required}
MAX_STEPS=${4:?max steps required}
TAG=${5:?tag required}
MODE=${6:?instruction mode required}
RATE=${7:?exec_rate_ratio required}
SEED=${8:-1000}
case "$TAG" in *[!A-Za-z0-9_.-]*) echo "unsafe tag: $TAG" >&2; exit 64 ;; esac
for VALUE in "$EPISODES" "$MAX_STEPS" "$SEED"; do
  case "$VALUE" in ''|*[!0-9]*) echo "invalid integer: $VALUE" >&2; exit 64 ;; esac
done

module load miniconda FFmpeg/4.3.2-GCCcore-12.2.0 Mesa/22.2.4-GCCcore-12.2.0
export PATH="$HOME/.conda/envs/robocasa_env/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

POLICY_SNAPSHOT="$HOME/scratch/vla_bspline/code_snapshots/robocasa_corrected_contract_v1_20260821"
export PYTHONPATH="$POLICY_SNAPSHOT/src${PYTHONPATH:+:$PYTHONPATH}"
SCRIPT="$HOME/scratch/vla_bspline/code_snapshots/robocasa_progress_eval_v9_20260821/robocasa_progress_eval.py"
VARIANT_ROOT="$HOME/scratch/vla_bspline/outputs/robocasa_corrected_rate_variants"
OUT_ROOT="$HOME/scratch/vla_bspline/outputs/robocasa_corrected_rate_eval"
SAFE_RATE=${RATE//./p}
SAFE_MODE=${MODE//:/_}
VARIANT="$VARIANT_ROOT/${TAG}_rate${SAFE_RATE}_${SLURM_JOB_ID}"
OUT="$OUT_ROOT/${TAG}_${TASK}_${SAFE_MODE}_rate${SAFE_RATE}_s${SEED}_n${EPISODES}_${SLURM_JOB_ID}.json"

[[ -f "$CKPT/model.safetensors" && -f "$CKPT/config.json" ]] || {
  echo "missing checkpoint artifacts: $CKPT" >&2; exit 66;
}
[[ -f "$SCRIPT" ]] || { echo "missing frozen evaluator: $SCRIPT" >&2; exit 66; }
[[ ! -e "$VARIANT" && ! -e "$OUT" ]] || { echo "refusing overwrite" >&2; exit 73; }
mkdir -p "$VARIANT_ROOT" "$OUT_ROOT" "$VARIANT"

CKPT="$CKPT" VARIANT="$VARIANT" RATE="$RATE" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

source = Path(os.environ["CKPT"]).resolve()
variant = Path(os.environ["VARIANT"])
rate = float(os.environ["RATE"])
if not 1.0 < rate <= 4.0:
    raise ValueError(f"rate must be in (1, 4], got {rate}")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

for item in source.iterdir():
    if item.name != "config.json":
        (variant / item.name).symlink_to(item.resolve())
config = json.loads((source / "config.json").read_text())
if "embedded_spline_stats" not in config:
    raise RuntimeError("checkpoint is not corrected-contract/self-contained")
original_rate = float(config.get("exec_rate_ratio", 1.0))
if original_rate != 1.0:
    raise RuntimeError(f"source checkpoint is already retimed: {original_rate}")
config["exec_rate_ratio"] = rate
(variant / "config.json").write_text(json.dumps(config, indent=2) + "\n")
manifest = {
    "schema": "robocasa_corrected_rate_variant_v1",
    "job_id": os.environ["SLURM_JOB_ID"],
    "source_checkpoint": str(source),
    "source_model_sha256": sha256(source / "model.safetensors"),
    "source_config_sha256": sha256(source / "config.json"),
    "variant_config_sha256": sha256(variant / "config.json"),
    "original_exec_rate_ratio": original_rate,
    "evaluated_exec_rate_ratio": rate,
    "only_semantic_change": "exec_rate_ratio",
}
(variant / "rate_variant_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(manifest, sort_keys=True))
PY

python "$SCRIPT" \
  --ckpt "$VARIANT" --task "$TASK" --episodes "$EPISODES" \
  --max_steps "$MAX_STEPS" --seed_base "$SEED" \
  --constructor_seed 20260821 --instr_mode "$MODE" --out "$OUT"

test -s "$OUT"
python - "$OUT" "$VARIANT/rate_variant_manifest.json" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
manifest = json.load(open(sys.argv[2], encoding="utf-8"))
if result.get("renderer") != "osmesa":
    raise RuntimeError("result is not OSMesa-bound")
if result.get("checkpoint_model_sha256") != manifest["source_model_sha256"]:
    raise RuntimeError("result model is not the source checkpoint model")
print(f"successes={result['successes']}/{result['n_episodes']} rate={manifest['evaluated_exec_rate_ratio']}")
PY
sha256sum "$SCRIPT" "$VARIANT/config.json" "$VARIANT/rate_variant_manifest.json" "$OUT"
