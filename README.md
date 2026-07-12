# VLA-BSpline: Time-Aware Trajectory Action Heads for Vision-Language-Action Models

A VLA policy should output a **continuous trajectory plus explicit time allocation**,
not a fixed-rate list of waypoints. This repo implements that action head on top of
SmolVLA (LeRobot), evaluates it on LIBERO, and benchmarks it against the 2025–26 wave
of speed-adjustable-VLA methods. One sentence: *prior work makes policies faster by
changing the data, the weights, or the prompt; we change where the clock lives*
(see `docs/TIME_AUTHORITY.md`).

## Layout

```
policy/smolvla_spline/     The deployable policy package (source of truth).
                           Registered as --policy.type=smolvla_spline via
                           patch_factory.py. Also contains the study baselines:
                           smolvla_interp (waypoint resampling), smolvla_tempo
                           (speed-as-input, TempoVLA-style), smolvla_dsel
                           (data-level selective retiming), smolvla_speedaug
                           (demo-speed heterogeneity), and the per-config
                           normalization stats JSONs.

docs/                      RESULTS.md            — the results digest (start here; §0 summary)
                           BENCHMARK_DESIGN.md   — baselines, protocol, ablation matrix
                           TIME_AUTHORITY.md     — the unifying principle
                           SPEED_AUG_DESIGN.md   — shape/timing factorization study
                           C_FIX_DESIGN.md, ROADMAP.md, PROJECT*.md — history

scripts/data/              Dataset IO (lerobot_io.py) + normalization-stats
                           generators (gen_v2_stats_param.py is the canonical one).
scripts/analysis/          Offline studies & probes — every training run in this
                           project was gated by one of these first (fit_variant_study,
                           contact_diagnosis, toggle_kinematics, that_stagnation_probe,
                           calibrate_duration, ...).
scripts/eval/              record_rollouts.py (rollout harness: success, steps,
                           policy calls, T-hat traces, stall-recovery), latency +
                           smoothness benchmarks.
scripts/figures/           Paper figures (numbers hardcoded from measured results)
                           + generated outputs/.

cluster/                   Slurm batch scripts (Misha). Deployment layout on the
                           cluster is flat (~/vla_bspline/{record_rollouts.py,libero/,
                           lerobot/, cluster/}); rsync targets are documented in each
                           script header. GPU jobs hard-abort if CUDA is unavailable.

legacy/calvin_prototype/   Pre-project CALVIN spline-fitting prototypes (Windows
                           paths, per-dim splines). Kept only as a CALVIN data-format
                           reference for the port.
```

## The method in one paragraph
SmolVLA's flow-matching expert emits **n_ctrl spline tokens** instead of 50 waypoints:
clamped uniform cubic B-spline control points over the cumulative end-effector path
(both endpoints pinned), a gripper channel, and duration channel(s). Chunks terminate
at motion events (gripper toggles / pauses), so duration is a learned, calibrated
output and contact has a canonical location in chunk parameter. A decode samples the
continuous trajectory at **any** execution rate, which makes speed, deceleration,
replanning cadence, and rate transfer *decode-time knobs* — and the duration signal
doubles as a live failure monitor. Promoted config: n_ctrl=8, horizon_max=24 ("C-n8").

## Reproducing
Training/eval run on the cluster (see `cluster/*.sbatch`); the policy package is
rsynced into a LeRobot checkout as `lerobot/src/lerobot/policies/smolvla_spline/`
and registered with `patch_factory.py`. Every result in `docs/RESULTS.md` lists its
protocol; the evaluation-noise anatomy in §4c explains why all load-bearing numbers
are protocol-crossed (multiple seed sets × harnesses).
