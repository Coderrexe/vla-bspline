# LIBERO track — B-spline VLA project

Simba's LIBERO half of the time-aware B-spline action-head project (Quinten owns the
CALVIN half; the root-level `dataproccess.py` / `test_bspine_reconstruction.py` are his
CALVIN code, kept for reference). Goal: SmolVLA + B-spline + time-allocation head,
trained/evaluated on LIBERO via LeRobot.

## Environment
Local work uses a lightweight venv (numeric only, no `lerobot`/torch/sim):
```
source ~/.venvs/vla_bspline/bin/activate   # numpy scipy pandas pyarrow matplotlib huggingface_hub
```
The venv + HF data cache live OUTSIDE this OneDrive folder on purpose (avoid sync churn).
The LIBERO *simulator* + SmolVLA training run on the Yale HPC (Bouchet), not the Mac.

## Dataset
`HuggingFaceVLA/libero` — LeRobot v3.0, fps=10, 1693 episodes, 40 tasks (= the 4 eval
suites), state[8]=xyz+axisangle+2 gripper qpos, action[7]=6D delta+gripper(±1).
We fit splines to the **absolute state**, not the delta actions.

## Files
- `lerobot_io.py` — downloads only the numeric columns (~100 MB) and indexes episodes
  (works around the fact that meta `data/file_index` is NOT the parquet filename).
- `bspline_core.py` — cubic B-spline fit / reconstruct / `sample_at_hz` (the Hz-decoupling
  primitive) / RMSE. Gripper kept raw; cubic needs n_ctrl >= 4.
- `explore_and_fit.py` — visualize one episode: absolute state vs spline, raw deltas,
  n_ctrl sweep. -> `outputs/*.png`.
- `horizon_study.py` — reconstruction RMSE over (window length x n_ctrl); picks defaults.
- `build_bspline_targets.py` — the target-generation pipeline (LIBERO analog of Quinten's
  `dataproccess.py`). Emits per-chunk (control_points, duration_s, gripper) keyed by
  (episode_index, anchor_frame) -> `outputs/bspline_targets_*.parquet`.

## Open design fork (time allocation = the novelty)
Uniform-knot fixed-length chunks -> constant duration -> nothing for a time-allocation
head to learn (that's just the fixed-time B-spline baseline). Making duration meaningful
needs variable-length semantic chunks OR non-uniform knot optimization (per-segment
durations as targets). Decide before wiring the head. See `build_bspline_targets.py` docstring.
