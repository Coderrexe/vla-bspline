#!/bin/bash
#SBATCH --job-name=rc_stats_v2
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

# Generate corrected RoboCasa spline-target statistics without installing them
# into the shared policy source tree. Usage:
#   sbatch generate_robocasa_stats.sbatch DATASET_ROOT TAG [EPISODE_LIMIT]
# EPISODE_LIMIT=0 uses the complete dataset.

set -euo pipefail

DATASET=${1:?expected dataset root}
TAG=${2:?expected output tag}
EPISODES=${3:-0}
case "$TAG" in
  *[!A-Za-z0-9_.-]*) echo "unsafe tag: $TAG" >&2; exit 2 ;;
esac
case "$EPISODES" in
  ''|*[!0-9]*) echo "invalid episode limit: $EPISODES" >&2; exit 2 ;;
esac

module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

STAGE="$HOME/vla_bspline/cluster/rc_stats_repair"
SCRIPT="$STAGE/scripts/data/robocasa_prep.py"
EVENTS="$STAGE/policy/smolvla_spline/event_targets.py"
OUTDIR="$HOME/scratch/vla_bspline/outputs/rc_stats_repair_20260821"
OUT="$OUTDIR/spline_stats_robocasa_${TAG}_v2_n8h24.json"
MANIFEST="$OUTDIR/spline_stats_robocasa_${TAG}_v2_n8h24.manifest.json"

test -f "$SCRIPT"
test -f "$EVENTS"
test -d "$DATASET/data"
mkdir -p "$OUTDIR"
if [ -e "$OUT" ] || [ -e "$MANIFEST" ]; then
  echo "FATAL: refusing to overwrite existing stats artifact for $TAG" >&2
  exit 42
fi

python "$SCRIPT" \
  --root "$DATASET" \
  --episodes "$EPISODES" \
  --emit 8:24 \
  --stride 8 \
  --pause_frac 0.15 \
  --sampling_seed 20260821 \
  --out "$OUT"

test -s "$OUT"
DATASET="$DATASET" TAG="$TAG" EPISODES="$EPISODES" SCRIPT="$SCRIPT" EVENTS="$EVENTS" OUT="$OUT" MANIFEST="$MANIFEST" \
python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

dataset = Path(os.environ["DATASET"]).resolve()
output = Path(os.environ["OUT"]).resolve()
stats = json.loads(output.read_text())
assert stats["n_ctrl"] == 8
assert stats["horizon_max"] == 24
assert stats["min_seg"] == 8
assert stats["segmentation"] == "production_event_index_k_exclusive"

fingerprints = {}
for relative in ("meta/info.json", "meta/tasks.parquet", "meta/stats.json"):
    path = dataset / relative
    if path.exists():
        fingerprints[relative] = sha256(path)

manifest = {
    "tag": os.environ["TAG"],
    "dataset_root": str(dataset),
    "requested_episode_limit": int(os.environ["EPISODES"]),
    "dataset_fingerprints": fingerprints,
    "stats_path": str(output),
    "stats_sha256": sha256(output),
    "generator_path": str(Path(os.environ["SCRIPT"]).resolve()),
    "generator_sha256": sha256(Path(os.environ["SCRIPT"])),
    "event_targets_path": str(Path(os.environ["EVENTS"]).resolve()),
    "event_targets_sha256": sha256(Path(os.environ["EVENTS"])),
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
}
Path(os.environ["MANIFEST"]).write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

echo "=== completed corrected RoboCasa stats: $TAG at $(date -Is) ==="
