#!/bin/bash
# Download one official July-2026 annotated RoboCasa target composite dataset
# into an immutable, job-ID-staged scratch location. Existing datasets are
# never overwritten.
# Usage: sbatch ... TASK DATE BOX_SHARED_ID
#SBATCH --job-name=rc_ann_download
#SBATCH --partition=day
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

TASK=${1:?task required}
DATE=${2:?registry date required}
BOX_ID=${3:?Box shared ID required}
case "$TASK" in
  KettleBoiling) EXPECTED_EPISODES=501 ;;
  RinseSinkBasin) EXPECTED_EPISODES=509 ;;
  *) echo "unsupported task" >&2; exit 64 ;;
esac
case "$DATE" in ''|*[!0-9]*) echo "invalid date" >&2; exit 64 ;; esac
case "$BOX_ID" in *[!A-Za-z0-9]*) echo "invalid Box ID" >&2; exit 64 ;; esac

module load miniconda
export PATH="$HOME/.conda/envs/robocasa_env/bin:$PATH"
ROOT="$HOME/scratch/vla_bspline/robocasa_official_annotated"
DEST="$ROOT/${TASK}_${DATE}"
STAGE="$ROOT/.stage_${TASK}_${SLURM_JOB_ID}"
TAR="$STAGE/lerobot.tar"
URL="https://utexas.box.com/shared/static/${BOX_ID}.tar"
[[ ! -e "$DEST" && ! -e "$STAGE" ]] || { echo "refusing overwrite" >&2; exit 73; }
mkdir -p "$STAGE/unpack"

curl --fail --location --retry 3 --retry-all-errors --connect-timeout 30 \
  --output "$TAR" "$URL"
test -s "$TAR"
tar -tf "$TAR" >/dev/null
tar -xf "$TAR" -C "$STAGE/unpack"

TASK="$TASK" DATE="$DATE" URL="$URL" TAR="$TAR" STAGE="$STAGE" DEST="$DEST" \
EXPECTED_EPISODES="$EXPECTED_EPISODES" \
ROBOCASA_COMMIT="$(git -C "$HOME/robocasa" rev-parse HEAD)" python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

task = os.environ["TASK"]
date = os.environ["DATE"]
stage = Path(os.environ["STAGE"])
dest = Path(os.environ["DEST"])
tar_path = Path(os.environ["TAR"])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

infos = list((stage / "unpack").rglob("meta/info.json"))
if len(infos) != 1:
    raise RuntimeError(f"expected exactly one LeRobot root, found {infos}")
root = infos[0].parent.parent
info = json.loads(infos[0].read_text())
expected_episodes = int(os.environ["EXPECTED_EPISODES"])
if int(info.get("total_episodes", -1)) != expected_episodes:
    raise RuntimeError(
        f"expected {expected_episodes} episodes, got {info.get('total_episodes')}"
    )
parquets = sorted(root.glob("data/chunk-*/*.parquet"))
if not parquets:
    raise RuntimeError("download contains no data Parquets")
schemas = [pq.ParquetFile(path).schema_arrow.names for path in {parquets[0], parquets[-1]}]
if schemas[0] != schemas[1]:
    raise RuntimeError("first/last Parquet schemas differ")
columns = schemas[0]
annotation_columns = [
    name for name in columns
    if any(token in name.lower() for token in ("subtask", "skill", "stage"))
]
if len(annotation_columns) < 3:
    raise RuntimeError(
        "official per-frame subtask/skill/stage annotations are missing: "
        f"columns={columns}"
    )
manifest = {
    "schema": "robocasa_official_annotated_download_v1",
    "task": task,
    "registry_date": date,
    "source_url": os.environ["URL"],
    "source_tar_sha256": sha256(tar_path),
    "robocasa_commit": os.environ["ROBOCASA_COMMIT"],
    "total_episodes": expected_episodes,
    "total_frames": int(info["total_frames"]),
    "parquet_files": len(parquets),
    "columns": columns,
    "per_frame_annotation_columns": annotation_columns,
    "info_sha256": sha256(infos[0]),
}
(root / "official_download_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
if dest.exists():
    raise FileExistsError(dest)
root.rename(dest)
tar_path.unlink()
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

test -s "$DEST/official_download_manifest.json"
sha256sum "$DEST/official_download_manifest.json" "$DEST/meta/info.json"
