# METHOD.md — Complete Technical Specification (current as of July 12, 2026)

*The single authoritative description of how everything works. History to July 8 in
PROJECT.md; numbers in RESULTS.md; landscape/protocol in BENCHMARK_DESIGN.md.*

## 1. Architecture

Base: SmolVLA (450M, LeRobot 0.5.2). The flow-matching action expert denoises a
`(B, chunk_size, 32)` token block; `chunk_size` is pure sequence length. We set
`chunk_size = n_ctrl` **spline tokens**. Token channel layout (first `_N_OUT=8` of 32
supervised; loss sliced to 8):

```
token i = [ 6 pose-path control-point dims | 1 gripper control point | 1 duration channel ]
```

Promoted config **C-n8**: `n_ctrl=8, predict_duration=true, horizon_max=24, min_seg=8,
spline_degree=3`, stats `spline_stats_libero_v2_h24_n8.json`. Baseline B (fixed-time):
`predict_duration=false, horizon=20`. Baseline A: stock SmolVLA (50 waypoints).

`action_delta_indices = range(horizon_max if predict_duration else horizon)` — the
dataloader fetches the RAW delta-action window; targets are built on the fly in
`forward()` (no derived dataset).

## 2. Target construction (v2, event-segmented)

Per sample, on the fetched raw window `a ∈ R^{Hm×7}` (pose deltas 6D + gripper ±1):

1. **Padding**: steps past episode end get pose deltas zeroed (path stops); gripper
   is left as delivered (dataloader repeats last value = hold).
2. **Event segmentation** `_first_event`: first k ∈ [min_seg, horizon_max) where
   (a) gripper sign flips at k, or (b) two consecutive speeds < `pause_frac(0.15) ×
   torch.quantile(speed, 0.5)` (torch.quantile, NOT torch.median — numpy semantics),
   or (c) episode end (pad); else T = horizon_max ("cap" chunk).
3. **Path**: p₀=0; p_k = Σ_{i<k} a_i (cumulative COMMANDED path, 6D). Time-uniform
   parameter u_k = k/T. (Arc-length parametrization was tried and REFUTED — v3 study:
   time-uniform + control-point spacing encodes the speed profile at full LSQ
   resolution; explicit per-span time maps are coarser.)
4. **Pinned LSQ fit** (clamped uniform cubic, both endpoints pinned): c₀=0,
   c_{n-1}=p_T; interior c = pinv(B[:,1:n-1]) @ (p − B[:,n-1:]·p_T). Implemented as
   precomputed zero-padded per-length operator BANKS (`_pm_bank/_bl_bank/_pg_bank`,
   one slice per T ∈ [min_seg, horizon_max]; zero-padding is exact) gathered per
   sample, single batched einsum. Torch Cox–de Boor basis == scipy to 2.8e-16.
5. **Gripper**: fit raw ±1 signal as 7th spline channel at u_k = k/(T−1); decode
   sign(); toggle timing p95 = 0.26 env steps offline.
6. **Duration channel**: log T replicated across tokens (predict via mean over
   tokens at decode). Optional `fit_end_weight` (contact-weighted WLS — implemented;
   success-neutral, kept as ablation) and `speed_aug` (per-episode duration scaling
   s·T for the factorization study; shape untouched; s<1 allowed for v2 only).
7. **Normalization**: ACTION → IDENTITY in the processor (LOAD-BEARING: per-step
   MEAN_STD breaks Hz-retargeting telescoping); targets normalized internally with
   per-token stats from the JSON (regenerate via `scripts/data/gen_v2_stats_param.py`
   whenever n_ctrl/min_seg/horizon_max/weighting change).

Optional auxiliary loss `decode_consistency_weight` (piloting): the model forward is
replicated to expose the SIGNED flow error e = v_t − u_t; since x̂₀ − x₀ = t·(u−v) and
decode is linear, `aux = t²·‖B_T·(e_pose·σ)‖²/(6(T+1))` is exactly the x₀-space PATH
error — aligning the loss metric with step-space geometry (targets encode terminal
deceleration at ratio 0.78; the learned policy executes at 0.88 — this loss targets
that gap).

## 3. Decode (the time-authority surface)

`_decode_tokens(tokens, h_exec)`: unnormalize; force c₀=0; sample the basis on a
u-grid; emit per-step deltas (clamped ±1) + sign(gripper). All mechanisms are
u-grid/time-map operations on the SAME fitted geometry:

| knob (config) | operation | status |
|---|---|---|
| `exec_rate_ratio` / `exec_horizon` | h_exec = round(T̂·ratio) — Hz transfer | core |
| `feasibility_stretch` (+`actuator_bound`) | if max|Δ| > bound, h_exec ×= worst/bound (re-decode) | +19 @ half rate |
| `speedup_alpha` + `speedup_T_threshold` | chunk-level selective: α only when T̂>thr | ladder rung 2 |
| **`profile_alpha` + `profile_speed_threshold` (+`profile_soft`)** | **interval-level: intervals with |Δ|>θ·mean get dt×α; u-grid rebuilt by inverting the piecewise-linear time map; endpoint exact** | **flagship: 92.0 @ 1.30×, n=300 crossed** |
| `ease_out` | last 3 path steps re-timed over 3+e cosine-spaced steps (terminal velocity ×0.14) | scoped: self-paced/hardware |
| `replan_frac` / `replan_margin` | duration-scheduled replanning (execute to T̂−margin) | margin law; 2.7× fewer calls |
| `duration_snap_threshold` | snap smeared cap predictions to cap | mostly obsolete post-h24 |
| `chain_velocity` | c₁ = v_prev·h/9 (s'(0)=9(c₁−c₀)) — C¹ chaining | hardware story |
| T̂ trace (free) | stall detector: 8-step trend ≥0 while T̂<cap | 86% precision detection |

Decode T̂ = exp(mean logT̂) clamped [min_seg, horizon_max] — **set the clamp to the
LABEL max, not the fetch cap, when labels exceed the cap** (speed-aug lesson;
censoring otherwise).

## 4. Design laws (all causally established)

1. **Capacity**: control-point density per step is the master knob — h40→h24: +21
   long-horizon; n6→n8: +2 avg (both spline arms). Knot redistribution BACKFIRES at
   this budget (K2: tail err +82%). Density ≥ ~0.33 ctrl/step at LIBERO 20fps.
2. **Timing lives in ctrl-pt spacing** (v3 refutation): time-uniform fits reproduce
   demo speed profiles to ~0.02 rel err incl. terminal deceleration (0.784/0.782).
3. **Retiming granularity — the effect AND its ordering are data-regime
   coupled** (final form, July 15). LIBERO (scripted, n=300 crossed): retiming
   costs success, finer gating costs least (interval −2.0@1.30× > chunk −3.7 >
   uniform −5.3); protect-slow is load-bearing (θ=0.5 breaks it). CALVIN
   (human teleop, 3 seeds × n=1000): retiming GAINS success, coarser gating
   gains most (uniform +0.38@1.42×, t=12.1, all seeds > interval +0.27@1.33×,
   t=9.1; U−P t=3.37 all seeds) and θ=0.5 is the best crossed cell (θ=0.3
   regresses — only the extreme-slow tail needs protection). Unified law:
   **scripted demos leave nothing to compress, so gating protects scarce
   margin; human demos carry dead time everywhere, so compression should be
   broad. Time authority appreciates as data gets more realistic, and its
   optimal allocation is a decode-time knob, not an architecture decision.**
   Seeded consequence: uniform-retimed spline matches-or-beats the waypoint
   baseline (pooled +0.127, t=3.89; 2/3 seeds ahead) at 1.42× less wall-clock.
4. **Replan before predicted events, never at them** (margin 0/2/4 → 84/90/93).
5. **Speed ≠ rushability on precision suites** (goal/spatial collapse under ANY
   speedup) — event/speed signals gate safely only where transport dominates.
6. **Eval protocol variance** (LIBERO, n=100/cell): same-checkpoint spread up to
   9 pts across seed sets × harnesses (spatial worst). No suite-ordering claims
   without protocol crossing; capability claims = within-protocol paired contrasts,
   pooled across ≥3 seed bases (n=300) for anything load-bearing.
7. **The time budget is one budget — decode knobs don't compose** (July 12): full
   stack (half-rate + stretch + interval retiming) = 64–69 vs stretch-alone 75,
   interval-alone 92. Each knob spends the same timing slack; savings aren't
   additive. Pick the single knob matched to the deployment constraint. (Ordering
   still matters *within* a stack: feasibility must be re-checked AFTER any
   compressive warp — post-warp per-interval dilation is implemented in decode —
   but ordering wasn't the binding failure here.)

## 5. Baselines implemented (all in policy/smolvla_spline/)

`smolvla_interp` (waypoint + RAW-space path resampling — must unnormalize first);
`smolvla_tempo` (speed-as-input: VSTA retiming per sample + scalar speed in state
padding slot 31; `exec_speed` at inference); `smolvla_dsel` (data-level selectivity:
event-aware retiming baked into training); `smolvla_speedaug` (per-episode slowdown
for the factorization study). Factory registration: `patch_factory.py` (idempotent,
both factory.py and policies/__init__.py — draccus needs the config import).

## 6. CALVIN port (in flight)

Dataset `fywang/calvin-task-ABCD-D-lerobot` (24,053 eps, 10 fps, LeRobot v2.1 →
converting to v3.0 with the official script, revision pin patched "v2.1"→"main").
Actions verified = scaled rel EEF deltas (×50 pos) + ±1 gripper; LIBERO-like.
Offline gate: event CoV 0.36, gripper events 41–54%, pauses 0%; fit error sits BELOW
the demo jitter floor (0.149 vs LIBERO 0.024 — 6× noisier demos; the spline is a
built-in denoiser). Config chosen: **n8/min_seg8/cap16** (duration signal logT σ=0.169
beats n10's 0.107; both fits below noise floor). Stats:
`spline_stats_calvin_v2_n8h16.json`. Eval: calvin_env installed (dedicated conda env
"calvin"; py3.10 shims: collections.Mapping et al., fractions.gcd, numpy==1.23.5,
opencv-headless 4.8, setuptools<81).

**Replay gate PASSED (July 12): the converted actions are STATE-RECOMPUTED 10 fps
deltas** (Δpos×50, Δrot×20, gripper ±1), not naive subsamples. In-sim tracking RMSE:
recomputed@repeat-3 = 0.0065 m (beats native 30 Hz replay 0.0100 — recomputed deltas
self-correct); naive subsample@repeat-3 = 0.118 m (fails). **Execution convention
locked: policy acts at 10 fps, each action held 3 sim steps at native 30 Hz.**

**Eval harness = two processes** (calvin_env's numpy-1.23 pins conflict with
lerobot): `calvin_policy_server.py` (lerobot env, GPU — loads any LeRobot ckpt with
its saved processor pipeline via `make_policy(cfg, ds_meta=local calvin_v30)`) ↔
`calvin_eval_client.py` (calvin env — official protocol: multistep_sequences loaded
from the calvin repo with a stub for its lightning-heavy utils, vendored FNV-1
initial-condition seeding for bit-identical official eval states, calvin_env task
oracle, EP_LEN=360). Wire = length-prefixed pickles carrying bytes/lists only
(numpy-2 pickles don't load in numpy-1.23). Gripper binarized to ±1 at the client.
Cameras: static+gripper only — the tactile (tacto) camera hangs headless; the
obs mapping is rgb_static→observation.images.top, rgb_gripper→wrist,
robot_obs(15)→observation.state, scene_obs(24)→environment_state.
`cluster/eval_calvin.sbatch <ckpt> [n_seq] [tag]` runs both. Pair-smoke passed
(calvA@5k solved 1/2 first subtasks; ~1.5 min/sequence).

## 7. Cluster operations (hard-won)

Misha primary (torch cu128 — cu130 CPU-crawls on driver 570; every GPU job hard-aborts
if !cuda). Helper scripts on SHARED fs (~/vla_bspline/cluster/), never /tmp (node-local
— killed one chain). `sbatch --export` values MUST NOT contain spaces (silent
truncation — killed two recovery runs; hardcode instead). Rollout logs pipe through
`tail -1` in batteries — per-episode data lives in the npz, not the .out. lerobot
resume: `--config_path=<ckpt>/train_config.json --resume=true`. Eval variants =
config-edited symlink dirs in `~/scratch/vla_bspline/outputs/hz_variants/`.
