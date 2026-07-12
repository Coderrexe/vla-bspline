# PROJECT.md — Technical Record: B-Spline + Time-Allocation Action Heads for SmolVLA

*Written 2026-07-08. This is the full technical history of the project: motivation, exact
architecture, every experiment, every result, every correction. Line numbers refer to the
repo state at time of writing. Code excerpts are copied verbatim from source, not
reconstructed from memory.*

---

## 1. Motivation and problem statement

### 1.1 The fixed-frequency problem

Every mainstream VLA/behavior-cloning policy (ACT, Diffusion Policy, π0, OpenVLA, SmolVLA)
outputs a **fixed-length chunk of N waypoints at a fixed timestep dt**, tied to whatever
control frequency the training data was collected at. Three consequences:

1. **Deployment must match training Hz**, or you interpolate/extrapolate as a bolt-on hack.
2. **Skill duration is implicit** in chunk length — the model cannot express "this motion
   should take longer" as a first-class output.
3. Every re-plan happens on a fixed wall-clock or fixed-step cadence, not when the *motion*
   actually calls for a new plan (e.g., right after a grasp).

### 1.2 Proposed fix (from the original project brief)

Replace the waypoint chunk with **B-spline control points + an explicit time-allocation
output**. A downstream decode samples the continuous trajectory `x(t)` at *any* rate. This
is inspired by quadrotor trajectory planning (MINCO, EGO-Planner, AllocNet), where control
points + segment-duration allocation is the standard representation for smooth,
time-optimal trajectories — but manipulation, unlike quadrotor racing, can fix the number of
segments per chunk, so **the time-allocation head is a plain MLP regression, not an LSTM**
(AllocNet needs an LSTM specifically because quadrotor segment count is variable/geometry-
dependent; ours is a fixed-shape output).

### 1.3 Prior work and our positioning

- **Cubic B-Spline VLA / velocity-feedforward paper** (arXiv 2603.16218): closest prior work.
  Uses a fixed-duration cubic B-spline action space for C² position/velocity profiles on
  industrial arms under impedance control. Reports jerk artifacts at chunk boundaries. **No
  time allocation** — this is exactly our v1 (Baseline B) equivalent.
- **MPD** (arXiv 2412.19948): B-spline control points as diffusion-model output for motion
  planning; not a VLA (no vision/language), fixed knot spacing.
- **FAST** (arXiv 2501.09747): DCT-based action tokenization; addresses the same "dense
  per-timestep binning is wasteful" problem via frequency-domain compression rather than a
  geometric spline; no duration head, no continuity guarantee.
- **AllocNet / MINCO / EGO-Planner / FASTER**: quadrotor trajectory optimization with time
  allocation; inspiration for the method, not applicable directly to manipulation (no
  vision/language conditioning, and their variable-segment-count problem doesn't exist here).
- **ISR / Trajectory Standardization** (arXiv 2606.22907, found via literature search
  2026-07-07): the closest work to our *speed-heterogeneity* study (§7.7). Offline
  **preprocessing** that resamples demonstrations to information-equidistant spacing,
  removing speed non-uniformity, +25% success. Confirms the phenomenon (operator-speed
  variability degrades IL) is real and current. **Distinction:** ISR treats timing as noise
  to delete before training; we show it is a *representation* question — a duration head
  factorizes timing out at training time, with no preprocessing, and the timing information
  is *retained* (replayable at any speed) rather than discarded.
- **Variable-Frequency IL** (arXiv 2411.12310): conditions the policy on a *commanded* speed
  as an input. Adjacent but orthogonal — that is speed-as-conditioning, not speed-as-dataset-
  heterogeneity-to-be-robust-to.

---

## 2. Backbone and benchmark

### 2.1 SmolVLA

- ~450M parameter VLA, native to the LeRobot library (`lerobot==0.5.2` lineage, our fork
  patches on top of the upstream `lerobot/policies/smolvla/`).
- The action expert is a **flow-matching** model: it denoises a tensor of shape
  `(batch, chunk_size, max_action_dim=32)` conditioned on VLM features, language tokens, and
  robot state. `chunk_size` in stock SmolVLA is pure sequence length (number of waypoint
  tokens); there is nothing in the architecture that requires those tokens to *be* waypoints.
  **This is the load-bearing fact that makes our method a zero-surgery reuse**: we keep the
  transformer, the flow-matching objective, and the sampling procedure completely unchanged,
  and simply redefine what the `chunk_size` tokens mean and how the loss is sliced.
- `action_delta_indices` (a `PreTrainedConfig` property) controls exactly which timesteps of
  the dataset's action array the dataloader fetches as the training target window. Overriding
  this property is how we make the standard LeRobot data pipeline hand us a *raw* window
  which we then convert to spline targets ourselves inside `forward()` — no custom dataset,
  no offline spline-fitting pass over the whole dataset at train time.

### 2.2 LIBERO benchmark

- Dataset: `HuggingFaceVLA/libero`, LeRobot v3.0 format. fps=10 in the stored dataset
  metadata (a mismatch with the simulator's native 20Hz control loop — itself a small Hz
  inconsistency baked into the benchmark, separate from anything we introduce).
- 4 standard eval suites: `libero_object`, `libero_spatial`, `libero_goal`, `libero_10`
  (a.k.a. "long", the long-horizon suite — hardest, most temporally extended tasks). 10
  tasks/suite × 10 episodes/task = 100 episodes per suite in our standard eval protocol
  (occasionally 30 for cheap pilots, we call this out every time).
- `observation.state` is 8-dim: eef xyz (0:3) + axis-angle orientation (3:6) + 2 gripper
  finger joint positions (6:8).
- `action` is 7-dim: **6D delta** end-effector command (0:6) + 1D gripper (6), gripper is
  bang-bang ±1. Actions live in `Box(-1, 1, shape=(7,))` — this `[-1,1]` box is the
  "actuator bound" referenced throughout the decode-time feasibility-stretch mechanism.
- Env stack: robosuite OSC_POSE controller, Franka Panda, native 20Hz control frequency
  (`control_freq` patched into the env config, see §6.3).

### 2.3 Core design decision: fit splines to the *absolute* path, not the raw deltas

Early exploration (`libero/explore_and_fit.py`, findings pocketed in memory
`libero-fit-findings`) established:

- The **absolute** eef trajectory (reconstructed by cumulative-summing the delta actions) is
  smooth, low-frequency, and cleanly splineable.
- The **raw delta-action signal** is high-frequency, bursty, with flat-zero pauses — directly
  fitting a spline to it would be like splining a velocity signal; errors integrate and the
  "C² smooth trajectory" story becomes meaningless.
- Gripper is bang-bang (±1 steps) and is never splined as a continuous position channel — it
  gets its own dedicated spline channel fit directly to the raw ±1 signal, then decoded via
  `sign()` (see §4.2).
- A single global spline over an entire (multi-stage) episode oversmooths fast sub-motions —
  this is why the method is **chunked with replanning**, not one spline per episode.

So: **the policy fits a B-spline to the cumulative sum of the commanded delta-action window**
(pose dims only), not to the raw deltas, and not to the dataset's `observation.state` absolute
pose directly (using the commanded-delta cumsum keeps everything in "action space" so the
decode step — differencing the spline back into per-step deltas — round-trips exactly to
what the environment consumes).

---

## 3. Architecture: the SmolVLA-Spline head

Source of truth: `libero/smolvla_spline_pkg/` (rsynced verbatim to both compute clusters as
`lerobot/policies/smolvla_spline/`). Two files matter:
`configuration_smolvla_spline.py` (dataclass config, `SmolVLASplineConfig`) and
`modeling_smolvla_spline.py` (`SmolVLASplinePolicy(SmolVLAPolicy)`).

### 3.1 Token layout

Each of the `n_ctrl` expert output tokens (default `n_ctrl=6`) has this channel layout in its
first 8 of 32 (`max_action_dim`) dims — the remaining 24 dims are unused padding, consistent
with how SmolVLA already pads all action heads to 32:

```
token[:, :6]  = pose spline control-point coordinates (one control point PER TOKEN, 6 pose dims each)
token[:, 6]   = gripper spline control-point value
token[:, 7]   = duration channel (v2 only; ignored/zero in v1)
```

`_N_OUT = 8` (`modeling_smolvla_spline.py:83`) is the number of real output channels; the
training loss is sliced to these 8 dims (`modeling_smolvla_spline.py:285`,
`losses = losses[:, :, :self._N_OUT]`).

`chunk_size` is force-set to `n_ctrl` in `__post_init__`
(`configuration_smolvla_spline.py:112`, `self.chunk_size = self.n_ctrl`) — the expert's
sequence length literally *is* the number of spline control points, e.g. 6 tokens instead of
SmolVLA's default 50 waypoint tokens. This is the source of the "10× / 5.8× compression"
claims in the results (36 real floats — 6 tokens × 6 pose dims — vs 350 for 50 waypoints ×
7 dims).

### 3.2 The B-spline basis (Cox–de Boor, float64)

`bspline_basis(u, n_ctrl, degree=3)` (`modeling_smolvla_spline.py:37-72`) computes a clamped
uniform cubic B-spline basis matrix via the Cox–de Boor recursion, entirely in `float64` for
numerical exactness, then is cast back to `float32` by callers. Knot vector:

```python
kn = torch.cat([
    torch.zeros(degree),                                   # 3 repeated knots at 0 (clamped)
    torch.linspace(0, 1, n_ctrl - degree + 1),              # interior knots
    torch.ones(degree),                                     # 3 repeated knots at 1 (clamped)
])
```

This basis is validated against `scipy.interpolate.BSpline` in the module self-test
(`_self_test()`, `modeling_smolvla_spline.py:414-432`) and in the earlier offline validation
script `libero/validate_spline_head_math.py` — **max |torch − scipy| = 2.8e-16** across
multiple query grids (`_self_test()` prints this and asserts `< 1e-9`).

### 3.3 v1: fixed-horizon pinned least-squares fit (Baseline "B")

`predict_duration=False`. `_init_spline_operators` (`modeling_smolvla_spline.py:98-109`)
precomputes, once at policy construction, the LSQ solve operators for a FIXED window length
`H = cfg.horizon` (default 20 env steps):

```python
H = cfg.horizon
u_path = torch.arange(H + 1, dtype=torch.float64) / H          # (H+1,) e.g. 21 points for H=20
B_path = bspline_basis(u_path, n, deg)                          # (H+1, n) e.g. (21, 6)
pinv_mid = torch.linalg.pinv(B_path[:, 1 : n - 1])               # (n-2, H+1) e.g. (4, 21)
u_grip = torch.arange(H, dtype=torch.float64) / (H - 1)
pinv_grip = torch.linalg.pinv(bspline_basis(u_grip, n, deg))      # (n, H) e.g. (6, 20)
```

**Both endpoints of the pose path are pinned**: control point 0 is forced to 0 (the chunk
starts exactly at the robot's current pose — no drift), and the last control point is forced
to the exact chunk-end displacement. Only the `n_ctrl − 2` interior control points are solved
via least squares. This is why `pinv_mid` operates only on `B_path[:, 1:n-1]` — the pinv of the
*interior* basis columns, applied to the *residual* after subtracting the pinned endpoint's
contribution. Concretely, in `_build_spline_targets` (`modeling_smolvla_spline.py:214-238`):

```python
pose = a[..., :6]                              # (B, H, 6) raw commanded pose deltas
path = torch.cat([torch.zeros_like(pose[:, :1]), torch.cumsum(pose, dim=1)], dim=1)  # (B, H+1, 6), path[0]=0
p_end = path[:, -1:, :]                          # (B, 1, 6) — pinned last control point
resid = path - self._b_last.unsqueeze(0) * p_end # (B, H+1, 6) — subtract endpoint's basis contribution
c_mid = torch.einsum("mh,bhd->bmd", self._pinv_mid, resid)   # (B, n-2, 6) — the LSQ solve, ONE matmul
c_pose = torch.cat([torch.zeros_like(p_end), c_mid, p_end], dim=1)  # (B, n, 6): [0, interior..., p_end]
```

**Fitting is a single precomputed matmul per batch** — no per-sample optimization loop, no
scipy call at train time. The gripper channel is fit the same way but *unpinned*
(`c_grip = einsum("nh,bh->bn", self._pinv_grip, grip)`), directly on the raw ±1 signal.

Endpoint exactness (measured, `validate_spline_head_math.py` T2 output and the module
self-test): endpoint error is **exactly 0** at all decode rates by construction (the pinned
LSQ guarantees this algebraically, not approximately) — verified to `<1e-4` in float32
(`_self_test()` asserts `e < 1e-4`, `modeling_smolvla_spline.py:478`, tighter in float64
offline validation).

### 3.4 v2: event-segmented time allocation (the "C" / "ours" arm)

`predict_duration=True`. Two things change: (1) chunk boundaries are no longer fixed at `H`
env steps but are found by scanning for the first "motion event"; (2) the duration itself
becomes a learned output channel.

**Event detection**, `_first_event` (`modeling_smolvla_spline.py:149-172`):

```python
def _first_event(self, a: Tensor, pad: Tensor | None) -> Tensor:
    # a: (B, Hm, 7) raw window -> T: (B,) int64 in [min_seg, horizon_max]
    ev = torch.zeros(B, Hm, dtype=torch.bool)
    grip = a[..., 6]
    ev[:, 1:] |= grip[:, 1:] != grip[:, :-1]                     # gripper toggle event
    speed = a[..., :6].norm(dim=-1)                              # (B, Hm)
    med = torch.quantile(speed, 0.5, dim=1, keepdim=True) + 1e-9  # NOT torch.median!
    low = speed < cfg.pause_frac * med
    ev[:, 1:] |= low[:, 1:] & low[:, :-1]                        # pause event (2 consecutive slow steps)
    if pad is not None:
        ev |= pad                                                # episode-end event
    ev[:, :cfg.min_seg] = False                                  # respect minimum chunk length
    T = torch.where(ev.any(dim=1), torch.argmax(ev.int(), dim=1), horizon_max)  # first True, else cap
    return T.clamp(min_seg, horizon_max)
```

**Correctness subtlety (a fixed bug, worth recording):** `torch.quantile(speed, 0.5, ...)`
is used, *not* `torch.median`. `torch.median` on an even-length tensor returns the
lower-middle element (numpy's `median` interpolates); using it under-thresholds windows that
are exactly half slow, half fast, causing systematic pause-detection misses.

Event types: **gripper toggle**, **pause** (2 consecutive steps below `pause_frac` (default
0.15) × the window's median speed), or **episode end** (padding flag). If none occur before
`horizon_max`, the chunk is **capped** at `horizon_max`.

**Zero-padded operator banks** (`_init_spline_operators`, v2 branch,
`modeling_smolvla_spline.py:110-129`): because different samples in a batch have different
event-determined chunk lengths `T`, but the flow-matching loss needs a rectangular tensor, the
policy precomputes ("banks") the pinned-LSQ operators for **every possible T** in
`[min_seg, horizon_max]` at construction time, zero-padded to a common shape:

```python
Hm, lo = cfg.horizon_max, cfg.min_seg                # e.g. 40, 6  (or 24, 6 for the h24 fix)
n_len = Hm - lo + 1                                   # number of distinct T values
pm_bank = torch.zeros(n_len, n - 2, Hm + 1)           # (n_len, n-2, Hm+1)
bl_bank = torch.zeros(n_len, Hm + 1, 1)
pg_bank = torch.zeros(n_len, n, Hm)
for i, T in enumerate(range(lo, Hm + 1)):
    u_path = torch.arange(T + 1) / T
    B_path = bspline_basis(u_path, n, deg)
    pm_bank[i, :, :T + 1] = pinv(B_path[:, 1:n-1])    # only the first T+1 columns are non-zero
    bl_bank[i, :T + 1, :] = B_path[:, n-1:n]
    ...
```

At training time, each sample's own `T` selects its row of the bank via `torch.gather` (no
Python loop over the batch):

```python
bank_i = T - cfg.min_seg                               # (B,) index into the bank
pm = self._pm_bank[bank_i]                              # (B, n-2, Hm+1)  — per-sample operator
bl = self._bl_bank[bank_i]                               # (B, Hm+1, 1)
pg = self._pg_bank[bank_i]                               # (B, n, Hm)
```

Because zero-padded columns multiply path entries beyond the segment (which are themselves
masked to the segment via `seg_mask`), the padding contributes exactly zero to the sum — this
is an *exact* batched implementation of "solve a different-length LSQ problem per sample,"
not an approximation.

**Duration channel**: `dur = torch.log(T.to(dtype)).unsqueeze(-1).expand(-1, n_ctrl)`
(`modeling_smolvla_spline.py:267`) — every token in the chunk carries the same scalar
log-duration target, broadcast across the `n_ctrl` dimension (so the loss sees it `n_ctrl`
times per sample, same as the other channels).

### 3.5 Normalization — IDENTITY on ACTION is load-bearing

```python
normalization_mapping: dict = {
    "VISUAL": NormalizationMode.IDENTITY,
    "STATE": NormalizationMode.MEAN_STD,
    "ACTION": NormalizationMode.IDENTITY,   # <-- load-bearing
}
```

SmolVLA's standard per-step `MEAN_STD` action normalization does not commute through the
decode's resampling/differencing operations (the per-step mean does not "telescope" through a
`cumsum` → resample → `diff` pipeline at an arbitrary execution rate). Instead, normalization
is applied **internally, per spline-token-channel**, using precomputed stats loaded from a
JSON file (`spline_stats_libero.json` for v1, `spline_stats_libero_v2*.json` for v2):

```python
mean = torch.zeros(n, self._N_OUT); std = torch.ones(n, self._N_OUT)
mean[:, :6] = pose_ctrl_mean; std[:, :6] = pose_ctrl_std      # per-token, per-pose-dim
mean[:, 6]  = grip_ctrl_mean; std[:, 6]  = grip_ctrl_std      # per-token
mean[:, 7]  = logT_mean;      std[:, 7]  = logT_std           # scalar, broadcast (v2 only)
```

Stats are generated offline by fitting the same pinned-LSQ operators over a large sample of
real dataset windows (`libero/validate_spline_head_math.py` T4 for v1,
`libero/v2_event_segmentation_study.py` for v2) — this is the ONE place actual offline
spline-fitting of the dataset happens; it is only for computing normalization statistics, not
for building training targets (those are built on-the-fly in `forward()`).

### 3.6 Training forward pass

```python
def forward(self, batch, noise=None, time=None, reduction="mean"):
    images, img_masks = self.prepare_images(batch)
    state = self.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]; lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    target = self._build_spline_targets(batch)                  # (B, n_ctrl, 8) normalized
    actions = pad_vector(target, self.config.max_action_dim)     # (B, n_ctrl, 32) — pad to 32
    losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
    losses = losses[:, :, :self._N_OUT]                          # slice back to 8 real channels
    return losses.mean(), {...}
```

Nothing else in SmolVLA's flow-matching training loop changes — `self.model` is the
*unmodified* SmolVLA expert; only the target tensor's semantics and the loss slicing differ.

### 3.7 Decode: Hz-decoupled sampling

`_decode_tokens(tokens, h_exec)` (`modeling_smolvla_spline.py:315-367`) turns normalized
expert tokens into `(B, h_exec, 7)` environment-ready delta actions, at **any** `h_exec` the
caller requests:

```python
t = tokens[..., :8] * self._tgt_std + self._tgt_mean            # unnormalize
c_pose = t[..., :6].clone(); c_pose[:, 0, :] = 0.0               # hard guarantee: chunk starts at current pose
c_grip = t[..., 6]

def _path_deltas(h):
    u = torch.arange(h + 1) / h                                  # sample grid for THIS execution rate
    Bp = bspline_basis(u, n_ctrl, degree)                         # (h+1, n_ctrl)
    path = torch.einsum("hn,bnd->bhd", Bp, c_pose)                # (B, h+1, 6) sampled path
    return path[:, 1:] - path[:, :-1]                             # (B, h, 6) consecutive deltas

deltas = _path_deltas(h_exec).clamp(-1.0, 1.0)                   # emitted env deltas
u_grip = torch.arange(h_exec) / max(h_exec - 1, 1)
g = einsum("hn,bn->bh", bspline_basis(u_grip, ...), c_grip)
grip = torch.where(g >= 0, 1.0, -1.0).unsqueeze(-1)               # sign() decode
return torch.cat([deltas, grip], dim=-1)                          # (B, h_exec, 7)
```

Because `h_exec` is a free parameter of the decode (not baked into the trained weights),
**the SAME trained checkpoint can be sampled at any control rate at inference time** — this
is the core "Hz decoupling" claim. `select_action`'s existing queue mechanism (unmodified
from stock SmolVLA — `_queues[ACTION].extend(actions.transpose(0,1)[:n_action_steps])`) works
unchanged regardless of `h_exec`.

### 3.8 Decode-time capability flags (all zero-retraining, config-only)

These are all implemented purely in `_decode_tokens` / `_get_action_chunk`
(`modeling_smolvla_spline.py:315-411`) and can be toggled on an already-trained checkpoint
via a config edit + symlink (the "hz_variants" pattern used throughout, see §6).

**(a) Rate retargeting** — `exec_horizon` (v1) / `exec_rate_ratio` (v2): decode at a
different `h_exec` than the training horizon. This alone is the base Hz-decoupling
capability.

**(b) Feasibility-aware time stretching** (`feasibility_stretch`,
`modeling_smolvla_spline.py:338-348`): if any decoded per-step delta would exceed the
actuator bound `[-1,1]` (which happens when compressing the same path into too few execution
steps), *lengthen* `h_exec` proportionally and re-decode, rather than clipping/distorting:

```python
deltas = _path_deltas(h_exec)
if feasibility_stretch:
    worst = deltas.abs().max().item()
    if worst > actuator_bound:
        h_exec = min(int(h_exec * worst / actuator_bound) + 1, 8 * h_exec)  # proportional + margin, capped 8x
        deltas = _path_deltas(h_exec)                                        # re-decode at the new rate
```

This treats the chunk as a continuous path and re-samples it more densely rather than
clipping — a capability that only makes sense because the representation IS a continuous
path, not a discrete waypoint list.

**(c) Duration-aware selective speedup** (`speedup_alpha`, `speedup_T_threshold`,
`_get_action_chunk`, `modeling_smolvla_spline.py:383-394`):

```python
T = self._predicted_T(t_un)                     # scalar predicted duration (median over batch=1 eval)
alpha = 1.0
if speedup_alpha < 1.0 and T > speedup_T_threshold:
    alpha = speedup_alpha                        # only speed up chunks predicted to be LONG
h_exec = max(2, round(T * exec_rate_ratio * alpha))
```

`alpha < 1` compresses execution into fewer steps (faster); gating it on `T > threshold`
means only chunks the model itself predicts are long (interpreted as "transport," not
"careful grasp approach") get sped up. `speedup_T_threshold=0` recovers **uniform** (blind)
speedup — the ablation against which selectivity is measured.

**(d) Self-paced replanning** — two variants, both requiring `predict_duration=true`:

```python
# replan_frac: execute a FRACTION of the chunk's own predicted duration, then replan
if predict_duration and replan_frac is not None:
    n_exec = max(2, min(round(chunk.shape[1] * replan_frac), chunk.shape[1]))
    chunk = chunk[:, :n_exec]
# replan_margin: replan a FIXED number of steps BEFORE the predicted chunk end
elif predict_duration and replan_margin is not None:
    n_exec = max(2, chunk.shape[1] - replan_margin)
    chunk = chunk[:, :n_exec]
```

`replan_frac=1.0` = execute the entire predicted chunk before replanning ("full self-paced").
`replan_margin=k` = always stop `k` steps short of the predicted event. The config validator
(`configuration_smolvla_spline.py:120-141`) requires `n_action_steps` (the LeRobot
`select_action` queue's per-invocation take) to be set ≥ `horizon_max` in this mode, since the
*actual* consumed length varies per-invocation and is controlled entirely by the slicing above,
not by the queue's usual fixed `n_action_steps` truncation.

Both variants increment `self.n_chunks_generated` (`modeling_smolvla_spline.py:410`), an
analysis hook read by `record_rollouts.py` to count policy invocations per episode — the
metric used to quantify "compute cost" (forward passes of the 450M model) independent of
success rate.

**(e) Velocity-continuous chunk chaining** (`chain_velocity`,
`modeling_smolvla_spline.py:320-326, 355-365`): pins the *second* control point (not just the
first, which is always pinned to 0) so that the new chunk's initial velocity matches the
previous chunk's velocity at the point where the previous chunk was actually truncated by
replanning (not necessarily its nominal endpoint). Uses the exact clamped-cubic derivative
identity for this basis/knot configuration: `s'(0) = 9·(c₁ − c₀)` in `u`-parameter units
(verified in `_self_test`, `modeling_smolvla_spline.py:568-574`, asserted to `<1e-3`). Given a
tracked previous per-step velocity `v_prev` (computed via a central finite difference of the
basis at the point of replanning, `du=1e-5`, `modeling_smolvla_spline.py:355-365`):

```python
c_pose[:, 1, :] = v_prev * h_exec / 9.0     # sets c1 so the new chunk's v(0) == v_prev
```

Measured effect (from the C-gap investigation, §5): boundary velocity discontinuity ratio
4.0 → 0.97 (near-perfect continuity). Success rate on LIBERO was **unchanged** by this fix —
interpreted as the simulator's OSC controller absorbing command-level jerk — but it is kept
as the hardware-relevant contribution (a real robot's impedance/velocity-feedforward
controller would care about this even if the LIBERO sim doesn't).

**(f) Duration mode-snap** (`duration_snap_threshold`, `_predicted_T_batch`,
`modeling_smolvla_spline.py:297-308`): a decode-only fix for a *measured* smearing pathology
(see §5) where flow matching regresses a bimodal duration target (short "event" chunks vs.
capped chunks) and generates intermediate values between the modes:

```python
T = torch.exp(logT).round().clamp(min_seg, horizon_max)
if duration_snap_threshold is not None:
    T = torch.where(T >= threshold, horizon_max, T)     # snap smeared-high predictions back to the cap
```

### 3.9 Config validation (fail-fast, `__post_init__`)

`SmolVLASplineConfig.__post_init__` (`configuration_smolvla_spline.py:105-147`) deliberately
skips the parent `SmolVLAConfig.__post_init__`, because that parent validates
`n_action_steps <= chunk_size` against the *token count* — meaningless here, since our tokens
are control points, not waypoints. Instead it validates against the *decoded* chunk length
(`horizon_max` if `predict_duration` else `exec_horizon or horizon`), and additionally
enforces: `speed_aug` requires ≥2 factors all `≥1.0` (§4.2 explains why `<1.0` is disallowed);
`replan_frac` XOR `replan_margin` (not both); `n_ctrl ≥ degree+1 = 4` (cubic spline minimum);
`horizon+1 ≥ n_ctrl`.

---

## 4. Baselines

### 4.1 Baseline "A" — plain waypoint SmolVLA

Stock `SmolVLAPolicy`, unmodified. The one experimental knob that matters a great deal is
`n_action_steps` (the replanning cadence): **nas=50 (SmolVLA's own default) gives only 60%**
on libero_object at 20k steps, while **nas=10 gives 93%** — a large, confound-worthy
sensitivity to replan cadence that every A-vs-{B,C} comparison in this project controls for
by fixing A at nas=10 throughout.

### 4.2 `smolvla_interp` — the strong resampling baseline (red-team baseline)

File: `libero/smolvla_spline_pkg/smolvla_interp.py`. Built specifically to red-team the
"our spline decouples Hz" claim against the baseline any competent practitioner would actually
build: take a *stock trained waypoint checkpoint* (same weights, unmodified) and, at decode
time, linearly resample its cumulative commanded path to the desired execution rate:

```python
def _get_action_chunk(self, batch, noise=None, **kwargs):
    chunk_n = super()._get_action_chunk(batch, noise, **kwargs)   # (B, chunk_size, 7) NORMALIZED
    h_exec = self.config.exec_horizon or chunk_n.shape[1]
    ...
    mu, sigma = self._action_stats(dev)          # loaded from the CHECKPOINT's own unnormalizer
    raw = chunk_n * sigma + mu                    # unnormalize to raw deltas
    path = cumsum(raw pose deltas, with a leading zero)
    # linear interpolation of the cumulative path onto h_exec+1 points, then diff -> new deltas
    # gripper: nearest-original-step sign snap
    return (raw_out - mu) / sigma                 # re-normalize
```

**Correctness trap this code exists to avoid**: SmolVLA normalizes actions per-step with
`MEAN_STD`; emitting `h_exec != chunk_size` steps naively would bias the total path by
`(chunk_size − h_exec) · mean` because the postprocessor adds the per-step mean back onto
*every* emitted action regardless of how many there are. The fix is to load the checkpoint's
own unnormalizer stats (`safetensors` file matching `*unnormalizer*.safetensors` in the
checkpoint dir), do all resampling arithmetic in **raw** action space, and re-normalize only
at the very end. `smolvla_interp` also supports the same `feasibility_stretch` flag as the
spline policy, for an apples-to-apples stretch comparison.

### 4.3 Baseline "B" — spline, fixed-T

Exactly the `smolvla_spline` policy of §3 with `predict_duration=False` (v1 mode). This
isolates the effect of the *B-spline representation itself* (compression, continuity) from
the effect of *time allocation* — B has the compression and continuity, C additionally has
variable-duration event-aligned chunks and the duration head.

---

## 5. The C-gap investigation (July 5–6): diagnosis → fix

Early on, C (time-allocation) trailed B by ~9 points average at native rate, worst on
libero_10 (long-horizon): ~47–50 vs B's 70. Full hypothesis ledger in `C_FIX_DESIGN.md`. Four
hypotheses were tested via a disciplined "one variable per pilot, eval object+long only for
fast signal" protocol:

- **H1 — cap coarseness**: 49% of v2 training chunks hit the `horizon_max=40` cap; fitting
  only 6 control points over a 40-step window halves the effective control-point density
  vs. B's 20-step chunks. Long-horizon (transport-dominated) tasks would be hit hardest —
  matches the observed 47 vs 70 gap.
- **H2 — duration-mode smearing**: the duration target is genuinely bimodal (short
  "event" chunks ~22±10 steps vs. capped chunks at 40). Calibration measurement (n=1280,
  built by mirroring the exact training data path — `resolve_delta_timestamps` + checkpoint
  preprocessor + `sample_actions` — in `libero/calibrate_duration.py`) found: **grasp-event
  chunks predicted with MAE 3.7, r=0.84** (the head genuinely learned "time to grasp" — this
  became the flagship calibration figure), but **capped/transport chunks smear toward ~30.5**
  (measured bias −9.5) — flow matching regresses toward the shorter mode, so cap chunks
  execute ~33% overspeed relative to their fitted shape's intent. The decode-only
  `duration_snap_threshold` fix (§3.8f) recovered object success (91→94) but not long-horizon.
- **H3 — replan-boundary velocity discontinuity**: measured boundary velocity ratio ≈4 across
  methods (position-only continuity from `c₀`-pinning leaves velocity discontinuous at every
  replan). Built the fix (velocity-continuous chaining, §3.8e): boundary ratio 4.0 → 0.97,
  **success unchanged** — eliminated as the driver of the object-level gap (kept as a
  hardware-relevant contribution regardless).
- **H4 — event-boundary toggles**: partially addressed by tighter replan cadence (nas5:
  object 85→91) but not by long-horizon (50→47) — secondary.

**The binding constraint, found by direct measurement (the density discriminator)**: cap
chunks fit the executed window **4.15× worse** than B's fixed-20-step chunks (0.051 vs 0.012
RMSE), and capped chunks are **60% of all long-episode training anchors**. This is a
supervision-quality ceiling that no decode-time fix (snapping, chaining) can repair — the
*training target itself* is systematically worse for the majority of long-horizon anchors.

**The fix: `horizon_max` 40 → 24** ("h24"). This directly restores control-point density on
capped chunks to roughly B's level (24 steps / 6 control points ≈ B's 20/6), at the
acknowledged cost of weakening the duration signal somewhat (cap fraction 49%→72%,
duration coefficient-of-variation 0.38→0.24 — measured via
`libero/v2_event_segmentation_study.py`, parameterized for `min_seg=8, horizon_max=24` in the
generalized n8 variant later). At a matched 20k-step budget, **C now beats B**:
object 93 vs 91, **long-horizon 57 vs 52** (spatial 65 vs 63, goal 73 vs 84 — a genuine
trade, discussed further in §7). A controlled single-variable comparison (only `horizon_max`
changed, everything else held fixed) put the C-cap40 long-horizon control at 36 vs the h24
fix's 57 — **+21 points, attributable to this one design change alone**.

**Resulting design law** (stated explicitly in `C_FIX_DESIGN.md` and generalized further in
§7.5): *chunk support (control points per env step) must scale with control-point budget;
duration prediction should not be entangled with fit support.* A cleaner decoupled design
(F1″: fit shape over `min(T, cap)` steps while the duration *label* remains uncapped
time-to-event) was designed and is documented but not implemented — pocketed as a
camera-ready refinement, since the simpler h24 fix already closed the gap sufficiently.

`Cfinal` throughout the results = the h24-fixed, 100k-step-trained, `n_ctrl=6` time-allocation
policy — the primary "C" arm of the main results table before the n8 generalization (§7.5).

---

## 6. Training and evaluation infrastructure

### 6.1 Clusters

Yale HPC: **Misha** (primary — cu128 torch, driver 570; short-walltime + untyped `--gpus=1`
backfill trick keeps queue wait low) and **Bouchet** (secondary — `gpu_devel` partition has a
1-concurrent-job cap per user, which blocks dependency chains; used opportunistically, not for
the long training runs).

### 6.2 Registration

`patch_factory.py` idempotently inserts `elif name == "smolvla_spline": ...` /
`elif policy_type == "smolvla_spline": ...` branches into lerobot's
`policies/factory.py`, **and** a corresponding import line into `policies/__init__.py` (the
latter is necessary because `draccus` — the CLI config library — builds its
`--policy.type` choice list from configs imported at CLI startup; factory registration alone
is insufficient). The same pattern registers `smolvla_interp` and `smolvla_speedaug`.

### 6.3 Environment patch

`patch_control_freq.py` threads a `control_freq: int = 20` field through
`envs/configs.py::LiberoEnv` and `envs/libero.py`, plumbing it into robosuite's
`OffScreenRenderEnv(..., control_freq=...)` constructor call — needed for the (now-retracted,
see §8) Hz-mismatch deployment experiments.

### 6.4 Training command shape

```
lerobot-train \
  --policy.type=smolvla_spline \
  [--policy.predict_duration=true --policy.horizon_max=24 \
   --policy.spline_stats_file_v2=spline_stats_libero_v2_h24.json] \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --steps=100000 --batch_size=32 --num_workers=24 [--seed=1001]
```

100k steps ≈ 4–7h on one GPU (L40S-class). Resume uses lerobot's
`--config_path=<ckpt>/pretrained_model/train_config.json --resume=true` API (NOT re-passing
the full CLI args with a resume flag — an early source of `FileExistsError`s).

### 6.5 Eval-variant pattern ("hz_variants")

Rather than retraining for every decode-flag combination, a completed checkpoint directory is
symlinked into `~/scratch/vla_bspline/outputs/hz_variants/<name>/` with every file
symlinked except `config.json`, which is a real edited copy (e.g., changing
`n_action_steps`, `speedup_alpha`, `replan_margin`, `n_ctrl` for a from-scratch-trained
variant, etc.). This is how every decode-time capability in §3.8 is evaluated cheaply against
an already-trained checkpoint.

### 6.6 Eval / rollout recording

`libero/record_rollouts.py` is a from-scratch rollout recorder (LeRobot's dataset recorder is
broken for LIBERO's `/`-containing feature names). Per-episode it logs commanded actions,
`observation.state`, the per-step `policy.last_predicted_T` (duration head's prediction, −1
if unsupported), success (`reward ≥ 1.0`, robust to gymnasium's info-packing differences), and
`policy.n_chunks_generated` (policy invocation count). Success/steps/policy-calls are
aggregated per config across all 10 task-ids of a suite for the summary tables throughout
§7. A hard CUDA-availability gate (`exit 97` if unavailable) is baked into every sbatch script
after an early incident where a torch/driver CUDA mismatch silently fell back to (23.8s/step)
CPU training.

---

## 7. Experimental results, in full

*All success numbers are percentages; all evals default to 100 episodes/suite (10 tasks × 10
episodes) unless explicitly marked otherwise (30-episode pilots are always flagged as such).*

### 7.1 20k-step pilot on libero_object (first end-to-end validation)

| policy | success |
|---|---|
| A waypoint, nas=50 (SmolVLA default) | 60% |
| A waypoint, nas=10 | **93%** |
| B spline, fixed-T | **91%** |
| C spline + time-allocation (pre-h24) | 85% |

This was the first proof the spline decode works end-to-end at all (91% at 1/5 the training
budget of the published SmolVLA reference), and the discovery of the nas=50→10 sensitivity
that fixed the A-baseline protocol for the rest of the project.

### 7.2 Main table: 100k steps, all four suites, 3 seeds per arm (n_ctrl=6, "n6")

| policy | obj | spa | goal | long | avg over suites |
|---|---|---|---|---|---|
| A waypoint (seeds 1000/1001/1002) | 93/93/98 | 83/86/84 | 84/88/86 | 60/65/65 | **82.1 ± 1.8** |
| B spline fixed-T (3 seeds) | 95/89/91 | 80/80/70 | 86/85/78 | 70/59/62 | **78.8 ± 3.8** |
| C spline + time-alloc, h24 (3 seeds) | 90/95/94 | 72/76/75 | 84/82/87 | 64/66/57 | **78.5 ± 1.1** |
| *published SmolVLA reference (1 seed)* | *96* | *90* | *92* | *71* | *87.3* |

**Reads:**
1. **B ≡ C on average (78.8 vs 78.5)** — time allocation costs nothing relative to fixed-time,
   and **C is the most seed-stable arm in the entire study** (±1.1 vs B's ±3.8), despite
   having a strictly more complex training target (variable-length event segmentation +
   duration regression vs. a fixed 20-step window).
2. A leads the spline arms by ~3.5 points on average, and this gap is **almost entirely
   localized to libero_spatial** (A 84.3 vs B 76.7 / C 74.3 as per-suite seed means; object,
   goal, and long are within seed noise). This suite-specific gap is investigated and
   partially resolved in §7.5.
3. C's long-horizon seed-mean (62.3) is statistically indistinguishable from B's (63.7) — the
   h24 density fix (§5) closed what was previously a ~20-point gap.

**⚠️ The 2-seed table (before the 3rd seed) taught the project's central methodological
lesson.** B's two initial seeds averaged 82.75 and 78.25 (spread 4.5, driven by long-horizon
alone: 70 vs 59) — large enough that a naive "B beats A" or "C trails B" reading from a
single seed is not reliable. **Standing rule adopted from this point on: never present a
single-seed suite-level difference smaller than ~6 points as a real effect.** This rule
subsequently reversed two other claims (§8).

### 7.3 Control-rate robustness (Hz mismatch) — retracted, see §8.1

An early Hz-mismatch sweep produced a striking-looking table (naive waypoint 0% at half/double
rate; spline 50-56%; feasibility-stretch +16-18pts) that was **later invalidated** by an
artifact in the eval budget and is not being carried forward as a claim. Full account in §8.1.
The one piece of that investigation that IS validated and kept: **feasibility-aware time
stretching** (§3.8b) is representation-agnostic (implemented for both `smolvla_interp` and the
spline policy) and its algorithmic idea — treat the chunk as a continuous path, lengthen
execution when actuator bounds would clip rather than distorting the path — is sound
regardless of the retracted numeric table.

### 7.4 Duration-aware selective speedup — the flagship duration-head capability

**First confirmation, n=100, libero_object, C-final (h24, n_ctrl=6), replan every 5 steps,
"selective" = speed up (×α) only chunks with predicted duration T̂ > 20 steps, "uniform" =
speed up every chunk:**

| α | selective success @ steps-to-done | uniform success @ steps | selectivity gap |
|---|---|---|---|
| 1.0 (no speedup, baseline) | 93% @ 145.3 | (anchor) | — |
| 0.8 | 94% @ 124.1 | 90% @ 116.5 | +4 |
| 0.7 | 85% @ 119.4 | 85% @ 110.4 | 0 (wobble) |
| **0.6** | **95% @ 122.7** | 73% @ 109.6 | **+22** |
| 0.5 | 76% @ 113.2 | 57% @ 97.5 | +19 |

Free-lunch headline: **selective α=0.6 = 95% success, 16% faster than no-speedup (93% @
145.3), while blind speedup at the same α loses 20 points.** Control: B (fixed-T, no
duration head, so uniform speedup is its *only* option) at α=0.6 scores 77% ≈ C-uniform
(73%) — establishing the gap is due to *selectivity*, not the spline representation per se.

**Suite generalization (n=100/cell) — the flagship is object-suite-specific:**

| suite | no speedup | selective α=0.6 | uniform α=0.6 |
|---|---|---|---|
| object | 93 @ 145.3 | **95 @ 122.7** | 73 @ 109.6 |
| goal | 90 @ 111.1 | 62 @ 87.3 | 53 @ 78.9 |
| spatial | 77 @ 109.1 | 47 @ 88.4 | 47 @ 81.8 |
| long | 65 @ 270.1 | 49 @ 216.6 | 48 @ 174.7 |

Mechanistic explanation: "long predicted duration ⇒ safe-to-rush transport" holds for
libero_object (pick-and-place over open space) but not for spatial/goal, where long-duration
chunks are frequently slow **precision** operations (careful placement, articulated-object
interaction), which selectivity mis-identifies as rushable. Duration alone predicts
rushability only when a task family cleanly separates transport from precision by duration.

**Mild-α (0.8) sweep on the other suites**: nearly free on object (94 vs 93) and long (66 vs
65, 13% faster), costs 6–12 points on spatial/goal, selectivity gap vanishes at mild α
everywhere. Full scoping of the capability: the **decode-time speed knob** (retime the same
path with zero retraining) is the robust general capability; **selectivity** specifically
earns its keep in the aggressive-speedup regime on transport-dominated suites.

### 7.5 Capacity ablation: n_ctrl=8 ("n8") and the generalized density law

Motivated by §7.2's spatial-suite deficit, a single-variable intervention (`n_ctrl` 6→8,
33% more control points per chunk, everything else unchanged) was tested at 20k steps:
**B-n8 spatial 63→74 (+11), object 91→94.** Combined with the h24 long-horizon fix, this
suggested a general design law: **control-point density per environment step is the master
capacity knob for spline action heads** — too sparse breaks long-horizon fits (fixed by
h24), mildly sparse dents spatial precision (fixed by n8).

**At 100k (single seed each): B-n8 = 98/87/84/66 (avg 83.75) — at or above the 3-seed A
mean (82.1) on 3 of 4 suites, spatial "fully fixed" (87 vs A's 84.3).** C-n8 (min_seg
correspondingly raised 6→8, new stats file `spline_stats_libero_v2_h24_n8.json` generated
with `MIN_SEG,H_MAX=8,24`, `N_CTRL,DEGREE=8,3`; resulting duration distribution: 74.7% capped,
CoV 0.20) scored **94/76/95/63 (avg 82.0) ≡ A's 3-seed mean**, with **goal=95 the best score
of any arm on that suite in the whole project** (published SmolVLA reference is 92).

**⚠️ Correction on a second seed** — critical, and documented in full in §8.2: B-n8's
apparent "spatial gap erased" result did **not** replicate. Seed 1001 gave B-n8 spatial=71
(not 87). Two-seed mean = 79, essentially indistinguishable from n6's 3-seed mean of 76.7.
**Honest status as of this writing: n8 robustly lifts object (97–98%, the best in the study)
and (via h24-style density) long-horizon; its effect on spatial precision specifically is
real at the 20k pilot scale but did not survive a second 100k seed and is provisional pending
a 3rd seed.** The density law's long-horizon arm (h24) is bedrock (confirmed across
C-final's full 3-seed table); its spatial arm (n8) is not yet.

**The selective-speedup flagship was re-tested on n8 and survived, relocated** — this is the
most important methodological result of the capacity ablation. On n8 (denser control-point
representation), blind α=0.6 speedup no longer collapses (88% vs n6's 73%); the selectivity
gap **moves** to a harsher α:

| α | selective @ steps | uniform @ steps | selectivity gap |
|---|---|---|---|
| 1.0 | 90 @ 143.4 | anchor | — |
| 0.6 | 90 @ 114.6 | 88 @ 102.9 | +2 |
| 0.5 | 84 @ 112.5 | 73 @ 105.0 | **+11** |
| 0.4 | 63 @ 116.3 | 42 @ 89.6 | **+21** |

Interpretation: the mechanism (selectivity protects against blind speedup breaking grasps) is
unchanged and now confirmed *twice*, at two different representation capacities; only the
specific α at which the effect is largest shifts with capacity. The originally reported "+22
at α=0.6" was real but partly an artifact of n6's coarser shape representation being more
fragile under blind speedup — the *capability* generalizes, the *exact operating point* does
not.

*Status at time of writing: two more seeds each of B-n8 and C-n8 are training/queued on
Misha (jobs 2074627–2074630 and their descendants) to decide whether n8 gets promoted to
the paper's main table.*

### 7.6 Self-paced replanning and the margin design law

The original project brief's design-decision list for replanning cadence named "predict a
re-plan horizon as a third output" as the *most principled but most complex* option — this
falls out of the existing duration head for free (§3.8d), no retraining needed.

**n=100, C-final (n6):**

| replan policy | object | calls/ep | long | calls/ep |
|---|---|---|---|---|
| fixed every 5 (default) | 93% | 31.3 | 61% | 73.8 |
| fixed every 12 | **94%** | 13.4 | **66%** | 32.9 |
| self-paced, full T̂ (`replan_frac=1.0`) | 84% | **9.2** | 55% | **23.2** |
| B, full fixed chunk (matched compute) | 91% | 8.3 | 56% | 19.2 |

**Honest verdict**: self-paced replanning *works* (84/55% at 3.2–3.4× fewer policy
invocations than the default cadence) but does **not** beat a well-tuned fixed cadence
(nas=12 wins both suites at comparable compute). **A 30-episode pilot had suggested
self-paced *won* on libero_10 (73.3%!)** — the n=100 confirmation reversed this to 55% — the
same "pilot deltas under ~15 points are noise" lesson as the seed-variance discovery,
independently re-derived.

**The negative result was converted into an understood, positive design law via a targeted
intervention.** Hypothesis: executing exactly *to* the predicted event places gripper
toggles precisely at the chunk boundary, where duration-prediction error clips them (the
model's own duration error becomes an execution error at exactly the wrong moment). Tested
via `replan_margin` (§3.8d) — replan a fixed number of steps *before* the predicted event:

| replan at | object | calls/ep | long | calls/ep |
|---|---|---|---|---|
| T̂ (margin 0) | 84 | 9.2 | 55 | 23.2 |
| T̂ − 2 | 90 | 9.6 | 57 | 27.9 |
| **T̂ − 4** | **93** | **11.7** | **62** | 33.0 |
| fixed nas=5 reference | 93 | 31.3 | 61 | 73.8 |

**Monotone, confirmed causal.** Design law: *replan before the predicted event, never
exactly at it.* With a 4-step margin, duration-scheduled replanning **matches the default
fixed-cadence success rate at 2.7× fewer policy invocations** — for a 450M-parameter VLA,
this is a meaningful deployment-side compute reduction with no success cost, and the
mechanism (not just the number) is understood and generalizable to any event-terminated
action-chunk policy.

### 7.7 Speed-heterogeneous demonstrations — the training-time factorization study

Full design in `SPEED_AUG_DESIGN.md` (§9 below summarizes the novelty positioning). This is
the study that turns "time allocation" from a purely decode-time trick into a
*representation-level training-time claim*.

**Hypothesis**: real demonstration data has execution-speed variability (different
operators, teleop rigs, control rates). An action representation either factorizes motion
*shape* from motion *timing*, or entangles them:

- **A (waypoint)**: target = "position at t+k·dt" — the same observation maps to different
  per-step deltas at different demo speeds; speed ambiguity contaminates shape supervision.
- **B (spline, fixed-T)**: chunk = "path covered in the next 20 steps" — a speed-dependent
  *prefix* of the underlying path; same contamination, in control-point space.
- **C (spline + time allocation)**: chunks end at events; the spatial segment between events
  is speed-invariant; only the duration *label* varies with speed. All speed ambiguity is
  isolated into the one channel designed to carry it.

**Synthetic speed heterogeneity**: each episode gets a deterministic factor
`s = speed_aug[episode_index % 3]`, `s ∈ {1.0, 1.5, 2.0}` — the same spatial path executed
`s×` slower. Only `s ≥ 1` is used because `s<1` (speeding up the path) would clip commanded
deltas past the `[-1,1]` actuator bound, corrupting the synthetic demo (this constraint is
enforced in the config validator, §3.9). Implementation, per arm:

- **A**: new policy `smolvla_speedaug` (`smolvla_speedaug.py`) wraps stock `SmolVLAPolicy`;
  its `forward()` unnormalizes `batch[ACTION]` using dataset-level stats loaded from
  `waypoint_action_stats_libero.json`, builds the cumulative path, resamples it at
  `x = arange(H+1) / s` via linear interpolation, re-diffs, re-normalizes, and defers to the
  parent's unmodified `forward()`. Deployment/inference is completely unmodified — this
  transformation only exists in training.
- **B**: `speed_aug` config flag in `smolvla_spline`; in `_build_spline_targets`
  (`modeling_smolvla_spline.py:220-232`), the *cumulative path itself* is resampled at
  `j/s` via the shared `_lerp_path` helper before the LSQ fit — so the fitted control points
  literally describe a different (slower) spatial extent, i.e. shape contamination by
  construction.
- **C**: the event-segmentation and shape fit run on the RAW (unscaled) window exactly as
  without speed_aug; **only the duration label is multiplied by `s`**:
  `T_lab = T.to(dtype) * self._speed_factors(...)` then `dur = log(T_lab)`
  (`modeling_smolvla_spline.py:260-267`). This is a **one-line** implementation of the entire
  factorization hypothesis.

**Machine-verified factorization** (module self-test, `modeling_smolvla_spline.py:516-565`):
for constant-velocity synthetic actions, B's fitted endpoint control point at `s=2` is
*exactly* half of its `s=1` value (err 0.00e+00) — proving B's shape targets are genuinely
speed-contaminated by construction. For C, the shape control points are **bit-identical**
across `s=1` vs `s=2` (max diff 0.00e+00) while the recovered durations are exactly 12 and 24
respectively for a toggle at raw step 12 — proving C's shape targets are genuinely
speed-invariant by construction, before any training happens.

**20k pilot results (vs. clean 20k twins, object/long, 100 eps):**

| arm | clean | speed-aug | Δ |
|---|---|---|---|
| A waypoint | 93 / 51 | 95 / 58 | **+2 / +7** |
| B spline fixed-T | 91 / 52 | 90 / **39** | −1 / **−13** |
| C spline + time-alloc | 93 / 57 | 87 / 53 | −6 / −4 |

Two findings, one revised (sharpened) thesis:

1. **Waypoint policies absorb pure speed heterogeneity** (a genuine, honest null against the
   original prediction). Mechanistic explanation: per-step deltas across different demo
   speeds are *colinear* (same path direction, different magnitude), so averaging over
   speeds still points in roughly the right direction at close to mean speed, and closed-loop
   replanning re-synchronizes progress every cadence step. The naive "fixed-dt breaks under
   mixed demo speeds" intuition is **false** for a closed-loop deployment with a fair time
   budget.
2. **Fixed-time trajectory *chunks* are the vulnerable representation, not waypoints per se.**
   B's chunk boundary is time-defined, so the same observation is asked to produce different
   *shapes* at different demo speeds — genuine shape-supervision blur, −13 points on
   long-horizon. C's event-terminated chunks isolate this into the duration channel and hold
   at −4 — **C beats B by +14 points on long-horizon specifically under this stress.**

**100k confirmation (1 seed each, vs. clean 3-seed means) — the contrast attenuates with
training budget**, Δ(object/long): A −3.7/−5.3, B −3.7/−6.7, **C −2.0/−1.3**. With 5× more
gradient steps, even B's blurred targets mostly converge; the *ordering* is preserved (C
least degraded at both budgets; B worst on long-horizon at both budgets) but the absolute
100k gap is within single-seed noise.

**Honest final framing** (not a standalone headline, but a real, mechanistically-verified
supporting result): *speed heterogeneity is a sample-efficiency tax on fixed-time trajectory
chunks (−13 points on long-horizon at 20k) that time allocation largely waives (−4 at 20k,
−1.3 at 100k — the only arm statistically indistinguishable from its clean-data twin at
full budget).* Waypoint policies avoid the tax entirely via closed-loop slack, at any budget.

**Mechanism confirmed end-to-end, with a caught implementation bug.** A direct measurement
of the duration head's predicted-T̂ distribution on speed-augmented C, vs. clean C:
clean C predicts mean T̂=20.8 (correctly cap-pinned near `horizon_max=24`); speed-aug C
predicts mean T̂=**27.0, p90=40, max=48** — a ~1.3× shift, close to the theoretical
prediction of `e^{E[log s]} ≈ 1.44×` (a geometric-mean prediction is the correct Bayes-optimal
response to *unresolvable* per-episode speed ambiguity — the model cannot tell from a single
observation which of the 3 synthetic speeds this episode belongs to, so it should predict
something like the geometric mean of the possible durations, which is what was measured).
Crucially, this happened while shape channels remained bit-identical (machine-verified above)
— **the speed variation went exactly where the representation design sends it: the duration
channel, and nowhere else.**

*Caught mid-investigation*: the first attempt at this measurement was **right-censored** — the
decode's `_predicted_T_batch` clamps `T̂` to `cfg.horizon_max`, which for the speed-aug
checkpoint was still set to the *raw fetch cap* (24) rather than the *label's actual max*
(48, since `s=2 × T=24` can reach 48). Both the speed-aug and clean checkpoints' T̂
distributions were pinned at the same ceiling (24), making the comparison uninformative (and,
as an unplanned side effect, meaning the speed-aug checkpoint's earlier success-rate eval had
been executing *faster* than its own duration head intended). Fixed by evaluating a config
variant with `horizon_max` raised to 48 for decode purposes only; the corrected measurement is
the one reported above. Documented as a reusable implementation lesson: **for
speed-augmented training, the decode-time duration clamp must be set to the label distribution's
max, not the raw window-fetch cap.**

---

## 8. Honest corrections and negative results (a first-class part of this record)

The project adopted a discipline of promoting no claim without an n=100 (or 3-seed)
confirmation, specifically because early pilot-scale numbers were repeatedly overturned.
Recording every reversal here, not just the surviving claims:

### 8.1 Retracted: the Hz-mismatch deployment table

An early sweep (deploy at f=10/20/40 Hz vs. 20Hz training) produced a dramatic-looking
"naive waypoint baseline collapses to 0% success" result. This was **later attributed to a
weak/undertrained checkpoint artifact combined with an unfair fixed step-budget across
different control rates** (a fixed 280-step cap punishes slower-Hz deployment
disproportionately; a fair comparison needs the episode step budget to scale with the
control rate). Re-run with a properly trained A-baseline and fair per-rate budgets gave very
different (non-catastrophic) naive-waypoint numbers. **The entire Hz-mismatch table and its
associated figure (`fig_hz.png`) are retracted from the results** and are not being carried
into any claim; the user separately deprioritized ("held off") the broader Hz
deployment-stack redesign this would have supported. What *is* kept: the feasibility-stretch
mechanism (§3.8b) as an algorithmic contribution, independent of that retracted table.

### 8.2 Reversed: self-paced replanning's 30-episode pilot result

Covered in full in §7.6. 30-episode pilot: self-paced replanning appeared to *win* on
libero_10 (73.3%, best of all tested cadences). 100-episode confirmation: 55%, worse than
several fixed cadences. This, plus the B seed-variance discovery (§7.2), is what established
the project's standing rule: **pilot-scale (≤30 episode, or single-seed) deltas smaller than
roughly 6–15 points are not to be reported as real effects** until confirmed at n=100 or
across seeds.

### 8.3 Reversed: B-n8's "spatial gap erased" claim

Covered in full in §7.5. Single-seed 100k result (B-n8 spatial = 87, apparently matching or
beating A's 84.3) was reported as a capacity-law confirmation. A second seed gave spatial=71
— two-seed mean 79, statistically indistinguishable from n6's 3-seed mean of 76.7. The
*correct* current statement is: n8's long-horizon and object-suite gains are robust (density
law confirmed via h24 across C-final's full 3 seeds); its spatial-suite gain is real at 20k
pilot scale but not yet confirmed at 100k with proper seeding — a third seed is
training/queued to resolve this before any promotion of n8 to the paper's main table.

### 8.4 Censored measurement, caught and fixed: speed-aug T̂ distribution

Covered in full at the end of §7.7. First attempt at measuring the duration head's response
to speed-augmented training was silently right-censored by a decode-time clamp mismatched to
the label distribution; caught because the "no shift at all" result contradicted the strong
theoretical prediction (geometric-mean absorption), prompting a re-check rather than
acceptance of a convenient null.

---

## 9. Novelty positioning (literature check, 2026-07-07)

Web search for the closest recent work on speed variability in imitation learning surfaced:

- **ISR / "Improving Robotic Imitation Learning via Trajectory Standardization"**
  (arXiv 2606.22907, June 2026): offline preprocessing (Information-Standardized Trajectory
  Resampling) that maps demonstrations onto an information-modulated Riemannian manifold and
  resamples them to geodesic-equidistant spacing, removing speed/pause non-uniformity;
  reports +25% success vs. naive uniform-time downsampling. **This independently confirms
  the underlying phenomenon (operator-speed variability degrades IL) is real, current, and
  worth ~25 points of success when handled well** — but their handling is to *delete* the
  timing information via preprocessing before training ever sees it.
- **Variable-Frequency Imitation Learning** (arXiv 2411.12310): treats *commanded* speed as a
  conditioning input to the policy — adjacent (also about speed) but addressing a different
  problem (controllable speed at inference) rather than robustness to *demonstration-side*
  speed heterogeneity during training.

**Our distinct claim**: demonstration-speed heterogeneity is a *representation* question, not
just a data-cleaning question. A duration head factorizes timing out of the shape-learning
problem at training time, requires no preprocessing pass over the dataset, and — unlike
ISR's deletion of timing information — the duration signal is *retained* as a first-class,
calibrated, reusable model output (replayable at any commanded speed via `speedup_alpha`,
usable for replanning cadence via `replan_margin`, etc.), not thrown away.

---

## 10. Reproducibility map

- **Code**: `libero/smolvla_spline_pkg/` (source of truth) → rsynced verbatim to
  `lerobot/policies/smolvla_spline/` on both clusters, plus the `patch_factory.py` /
  `patch_control_freq.py` idempotent source patches.
- **Stats files** (normalization; regenerate if `n_ctrl`/`horizon`/`horizon_max`/`min_seg`
  change): `spline_stats_libero.json` (v1, H=20/n=6), `spline_stats_libero_v2.json` (v2,
  cap=40/n=6, CoV 0.38), `spline_stats_libero_v2_h24.json` (v2, cap=24/n=6, CoV 0.24, cap
  fraction 72%), `spline_stats_libero_v2_h24_n8.json` (v2, cap=24/n=8, min_seg=8, CoV 0.20,
  cap fraction 74.7%), `waypoint_action_stats_libero.json` (dataset per-dim mean/std for the
  speed-aug wrapper), `spline_stats_libero_saug.json` / `_v2_h24_saug.json` (speed-augmented
  variants; the v2 logT stats are shifted *analytically*: `logT_mean += E[log s]`,
  `logT_std = sqrt(logT_std² + Var[log s])`, since the fit itself is proven
  speed-invariant).
- **Offline validation / analysis scripts**: `libero/validate_spline_head_math.py` (v1
  math vs. scipy, endpoint exactness, gripper toggle timing, stats gen T4),
  `libero/v2_event_segmentation_study.py` (event-duration distribution, v2 stats gen),
  `libero/gen_saug_stats.py`, `libero/gen_n8_stats.py` (speed-aug and n8 stats variants),
  `libero/calibrate_duration.py` (duration-head calibration, mirrors the exact training data
  path), `libero/record_rollouts.py` (eval/rollout recorder), `libero/failure_signatures.py`,
  `libero/measure_smoothness.py` (jerk metrics), `libero/plot_paper_figures.py` /
  `plot_duration_figures.py` (`fig_pareto`, `fig_calibration`, `fig_timeline` — the T̂
  countdown-to-grasp figure).
- **Training/eval sbatch scripts**: `cluster/train_*.sbatch` (per-arm/seed/variant training
  jobs, each with a 100-step smoke-test phase gating a full pilot/run), `cluster/eval_spline_
  misha.sbatch` (single config × task suite eval), `cluster/sp_battery*.sbatch`,
  `cluster/pareto_job.sbatch`.
- **Design docs**: `C_FIX_DESIGN.md` (the C-gap hypothesis ledger, §5), `SPEED_AUG_DESIGN.md`
  (§7.7 design + fairness rules + decision gates), `ROADMAP.md` (tiered plan, largely
  superseded by the actual investigation path recorded here), `RESULTS.md` (the living
  numbers digest this document is built from).

---

## 11. Open items / experiments in flight at time of writing

1. **n8 third seeds** (B-n8, C-n8) — training/queued on Misha; will decide whether the
   `n_ctrl=8` configuration is promoted to the paper's main table, and finally resolve
   whether the spatial-suite gain is real (§7.5, §8.3).
2. **n8 Pareto/frontier confirmation** — the α=0.4/0.5 selective-speedup re-emergence on n8
   (§7.5) is currently single-run; would benefit from a second confirmation once seeds land.
3. Speed-aug 100k results are single-seed; the C-vs-B ~3.5-point gap at 100k (§7.7) is not
   promotable to a standalone claim without speed-aug seeds (currently deprioritized as a
   supporting rather than headline result).
4. F1″ (decoupled shape-support/duration-label design, `C_FIX_DESIGN.md`) remains a pocketed
   camera-ready refinement, not implemented.
5. Arc-length (v3) reparametrization remains a pocketed contingency, not pursued.
6. The Hz deployment-stack redesign (absolute-setpoint + tracking-controller evaluation) is
   explicitly held per user direction, separate from and not to be confused with the
   retracted Hz-mismatch *table* (§8.1) — the retraction is about a specific bad measurement,
   the "redesign" is a separate, larger, deliberately-deferred piece of work.
