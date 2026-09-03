# EGL vs. OSMesa: LIBERO Comparability Gate

Status: **local-only design and implementation; not synced or submitted.**

## Scientific verdict

OSMesa preserves the task definition but is not automatically a comparable
evaluation distribution.

- LIBERO's `OffScreenRenderEnv` still uses the same BDDL task, MuJoCo model,
  cameras, initial states, controls, physics, and sparse success predicate.
  MuJoCo's own documentation says its rendering code is independent of how the
  OpenGL context is created; EGL and OSMesa are alternative Linux context
  implementations. Thus an OSMesa rollout is a semantically valid LIBERO
  rollout.
- The official LIBERO reproduction README explicitly configures an EGL device
  for its reference training/evaluation path. Our historical evaluations also
  used `MUJOCO_GL=egl`. OSMesa changes the OpenGL implementation from an NVIDIA
  GPU driver to Mesa's software rasterizer. Rasterization, antialiasing,
  texture filtering, and floating-point details can change pixels even though
  the scene and camera are identical.
- Pixels are policy inputs. The task-2 replay investigation already showed
  that EGL image drift can change actions and episode length while MuJoCo's
  initial integration state is exact. Therefore renderer equivalence must be
  measured, not inferred from an identical reset or reward implementation.

Sources: [official LIBERO repository and reproduction
instructions](https://github.com/Lifelong-Robot-Learning/LIBERO), [official
MuJoCo context-creation discussion](https://github.com/google-deepmind/mujoco/blob/main/doc/programming/samples.rst),
and [MuJoCo's Python backend selector](https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/rendering/classic/gl_context.py).

The immediate consequence is strict:

> OSMesa duration arms may be compared with one another because they share the
> renderer. They must not be compared numerically with historical EGL rows
> unless the gate below passes. The safest paper table re-evaluates every
> claimed comparator under OSMesa regardless of the gate.

## Paired diagnostic

The local collector is `scripts/eval/libero_renderer_probe.py`, its reviewed
but deliberately unsubmitted launcher is `cluster/eval_renderer_probe.sbatch`,
and the fail-closed analyzer is
`scripts/analysis/renderer_comparability.py`.

Each collector process selects exactly one backend before imports. For every
locked `(task, state)` coordinate it:

1. uses the same checkpoint, explicit LIBERO initial state, environment seed,
   policy seed, control frequency, preprocessing pipeline, and cameras;
2. records the compiled-model and MuJoCo integration-state hashes;
3. stores every initial raw camera array and every processed policy image
   tensor in one compressed NPZ sidecar, with exact shape/dtype/content hashes;
4. queries the actual OpenGL vendor, renderer, and version; and
5. executes the normal closed-loop policy, recording success, steps, first
   action, and complete action-trace hashes.

The replicate index never changes simulator or policy seeds. It exists only to
measure replay variation. At least two independent fresh processes are run per
backend. This separates three quantities:

- within-EGL replay variation;
- within-OSMesa repeatability; and
- cross-renderer shift.

The analyzer validates identical checkpoint/config/source/grid identities and
array sidecar hashes, then reports per-camera and aggregate:

- uint8 MAE, RMSE, maximum error, exact-pixel fraction, error-tail fractions,
  PSNR, and SSIM when scikit-image is installed;
- processed-tensor MAE, RMSE, maximum error, exact fraction, PSNR-like dynamic
  range score, and cosine similarity;
- first-action and full action-trace agreement;
- within- and cross-backend outcome disagreement; and
- the OSMesa-minus-EGL success difference overall, per task, and with a
  fixed-suite state-stratified bootstrap interval.

Initial camera comparisons are the minimum distribution check. If they reveal
a state-dependent shift, add identical open-loop action replay and compare
synchronized frames before diagnosing policy effects. Do not compare frames
at equal wall-clock steps from already-diverged closed-loop trajectories.

## Minimum runs and gates

### Engineering screen

Use all ten LIBERO-Long tasks, ten fixed states per task, and two fresh
processes per backend: 400 closed-loop episodes plus 400 inexpensive initial
image captures. This screen can reject equivalence or size the full run; it is
not powered to establish a ±5-point performance equivalence interval.

### Paper gate

Use the standard 50 locked states per task and two fresh processes per backend:
2,000 closed-loop episodes. Keep state IDs, environment/policy seeds, node
software, checkpoint, preprocessing, action frequency, and horizons fixed.
The decision thresholds are defined in code before observing results:

- exact checkpoint/config/source/grid identities and initial MuJoCo/model
  hashes;
- exact OSMesa raw/processed arrays, first actions, full action traces, and
  outcomes across its repeated process;
- cross-renderer raw-image p95 MAE at most 1 uint8 level, p05 PSNR at least
  40 dB, and p05 SSIM at least 0.995 when SSIM is available;
- processed-image p95 MAE at most 0.01 and p05 cosine at least 0.999;
- raw and processed shift no greater than the larger of the absolute limit and
  twice within-EGL p95 replay variation;
- the fixed-suite, state-stratified 95% interval for the overall success
  difference entirely inside ±5 percentage points;
- every per-task point difference inside ±10 points (descriptive, not a
  per-task equivalence test); and
- cross-renderer outcome discord no more than five points above within-EGL
  outcome discord.

Failure to find a significant difference is not a pass. Every equivalence gate
must pass. SSIM being unavailable is reported and does not silently substitute
an invented implementation; installing scikit-image before the paper gate is
preferred.

## Reporting rules

1. Always disclose `MUJOCO_GL`, OpenGL vendor/renderer/version, MuJoCo,
   robosuite, LIBERO, CUDA/Torch, camera size/names, state grid, and seeds.
2. Label the current OSMesa experiment as an internally paired mechanism
   experiment. Its causal comparison is predicted duration versus fixed and
   shuffled duration under the same renderer.
3. Never put an OSMesa method number beside a historical EGL baseline as if
   renderer were controlled. Re-evaluate baseline and method under OSMesa for
   the primary table, or keep the OSMesa result in a renderer-controlled
   ablation table.
4. If the paper gate passes, report the diagnostic in an appendix and still
   prefer matched-renderer tables. If it fails, this does not invalidate the
   OSMesa within-run duration intervention; it invalidates mixed-backend
   numerical comparisons.
5. If OSMesa changes absolute success materially, run the headline baseline
   and B-spline arms on both renderers. A consistent within-renderer method
   effect is stronger evidence than choosing whichever renderer gives the
   higher absolute score.

This gate prevents determinism engineering from becoming an unacknowledged
test-domain change, while preserving the main benefit of exact OSMesa replay
for causal decode-time interventions.
