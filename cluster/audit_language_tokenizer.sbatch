#!/bin/bash
#SBATCH --job-name=audit_lang_tokens
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.out
#SBATCH --error=/gpfs/radev/home/zs442/scratch/vla_bspline/logs/%x_%j.err

set -euo pipefail
module load miniconda
export PATH="$HOME/.conda/envs/lerobot/bin:$PATH"
export HF_HOME="$HOME/scratch/vla_bspline/hf_cache"

MAPPING=${1:?expected semantic-shuffle analysis JSON}
SCRIPT="$HOME/vla_bspline/cluster/audit_language_tokenizer.py"
OUT="$HOME/scratch/vla_bspline/outputs/language_dataset_audit/semshuffle_tokenizer_${SLURM_JOB_ID}.json"
test -s "$MAPPING"
test -s "$SCRIPT"
test ! -e "$OUT"

python "$SCRIPT" \
  --mapping_report "$MAPPING" \
  --tokenizer_name HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --max_length 48 \
  --out "$OUT"

test -s "$OUT"
OUT="$OUT" python - <<'PY'
import json
import os

result = json.load(open(os.environ["OUT"]))
assert result["token_id_collision_count"] == 0, result["token_id_collision_task_indices"]
print("validated tokenizer-distinct pairs:", result["pair_count"])
PY
echo "=== completed tokenizer audit job=${SLURM_JOB_ID} time=$(date -Is) ==="
