#!/bin/bash
#SBATCH --job-name=fix_libero_m
#SBATCH --partition=day
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
set -uo pipefail
module load miniconda
ENV=$HOME/.conda/envs/lerobot
export PATH="$ENV/bin:$PATH"
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export CMAKE_PREFIX_PATH="$ENV"
export CPATH="$ENV/include"
export LIBRARY_PATH="$ENV/lib"
cd $HOME/vla_bspline/lerobot
# pre-install BOTH probes without isolation so .[libero] doesn't rebuild egl_probe
pip install --no-build-isolation egl_probe hf-egl-probe 2>&1 | tail -8
echo "=== probes exit: $? ==="
pip install -e ".[libero]" 2>&1 | tail -12
echo "=== [libero] exit: $? ==="
python -c "import libero; print('libero OK', libero.__file__)" 2>&1
