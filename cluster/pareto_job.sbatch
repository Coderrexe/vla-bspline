#!/bin/bash
#SBATCH --job-name=pareto
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out
# Duration-aware selective speedup Pareto: does a time head let C speed up
# transport (long chunks) while keeping grasps careful, beating uniform speedup?
# Metric per config: success rate + mean steps-to-completion ON SUCCESSES (the
# honest "speed" axis; failures run to the cap and are excluded from timing).
set -uo pipefail
module load miniconda
export PATH=$HOME/.conda/envs/lerobot/bin:$PATH
export HF_HOME=$HOME/scratch/vla_bspline/hf_cache MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
cd $HOME/vla_bspline
HV=$HOME/scratch/vla_bspline/outputs/hz_variants
OUT=$HOME/scratch/vla_bspline/outputs/pareto; mkdir -p $OUT

# arms span a speed axis; C-selective slows only for grasp-critical short chunks
CONFIGS="Cfull_nas5 C_selective06 C_selective04 C_uniform06 C_uniform04 B_uniform06 B_uniform04"
for CFG in $CONFIGS; do
  for TID in 0 1 2 3 4 5 6 7 8 9; do
    python record_rollouts.py --ckpt $HV/$CFG --task libero_object --task_id $TID \
      --episodes 3 --out $OUT/${CFG}_t${TID}.npz 2>&1 | tail -1
  done
done

echo "=== PARETO SUMMARY (success, mean-steps-on-success) ==="
python - <<'PYEOF'
import numpy as np, glob, os, json
OUT = os.path.expanduser("~/scratch/vla_bspline/outputs/pareto")
for cfg in "Cfull_nas5 C_selective06 C_selective04 C_uniform06 C_uniform04 B_uniform06 B_uniform04".split():
    suc, steps_succ, steps_all = [], [], []
    for f in sorted(glob.glob(f"{OUT}/{cfg}_t*.npz")):
        z = np.load(f, allow_pickle=True)
        for s, L in zip(z["successes"], z["ep_lens"]):
            suc.append(bool(s)); steps_all.append(int(L))
            if s: steps_succ.append(int(L))
    if not suc:
        print(json.dumps({"cfg": cfg, "n": 0})); continue
    print(json.dumps({"cfg": cfg, "success": round(float(np.mean(suc)),3),
                      "steps_on_success": round(float(np.mean(steps_succ)),1) if steps_succ else None,
                      "steps_all": round(float(np.mean(steps_all)),1), "n": len(suc)}))
PYEOF
echo "=== done ==="
