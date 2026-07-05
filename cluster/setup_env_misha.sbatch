#!/bin/bash
#SBATCH --job-name=lerobot_env
#SBATCH --partition=day
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
# One-shot lerobot env build on Misha, with ALL fixes learned on Bouchet baked in.
# Env -> HOME (empty, 125G, backed up). Data/checkpoints go to scratch at run time.
set -uo pipefail
echo "=== node $(hostname) | $(date) ==="
echo "--- internet check ---"; curl -sI --max-time 15 https://pypi.org 2>&1 | head -1

module load miniconda
ENV=$HOME/.conda/envs/lerobot
PROJ=$HOME/vla_bspline
PIP="$ENV/bin/pip"; PY="$ENV/bin/python"

[ -d "$ENV" ] || conda create -y -p "$ENV" python=3.12
$PIP install --upgrade pip
# build toolchain + EGL/GL headers for the LIBERO sim dep (egl_probe / hf-egl-probe)
conda install -y -p "$ENV" -c conda-forge cmake ninja mesalib libglu

cd "$PROJ"
[ -d lerobot ] || git clone --depth 1 https://github.com/huggingface/lerobot.git
cd lerobot
echo "=== lerobot commit: $(git rev-parse HEAD) ==="

# 1) core training stack (robust, wheels)
$PIP install -e ".[smolvla,training]"
echo "=== [smolvla,training] exit: $? ==="

# 2) LIBERO sim eval: the --no-build-isolation trick makes hf-egl-probe's cmake succeed
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export CMAKE_PREFIX_PATH="$ENV"
export CPATH="$ENV/include"
export LIBRARY_PATH="$ENV/lib"
export PATH="$ENV/bin:$PATH"
# both egl_probe (robosuite transitive dep) AND hf-egl-probe must be pre-installed
# no-build-isolation, else .[libero] rebuilds egl_probe in isolation and fails on `import cmake`
$PIP install --no-build-isolation egl_probe hf-egl-probe
echo "=== egl_probe+hf-egl-probe exit: $? ==="
$PIP install -e ".[libero]"
echo "=== [libero] exit: $? ==="

echo "=== VERIFY ==="
$PY - <<'PY'
import importlib
for m in ["lerobot","torch","transformers","datasets","libero"]:
    try:
        mod=importlib.import_module(m); print("OK ",m,getattr(mod,"__version__","?"))
    except Exception as e: print("ERR",m,repr(e)[:160])
import torch; print("torch cuda build:",torch.version.cuda)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; print("OK SmolVLAPolicy")
PY
echo "=== done $(date) ==="
