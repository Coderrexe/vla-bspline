# Speed-Heterogeneous Demonstrations: the training-time case for time allocation

*Design doc, July 6 2026. Status: pilots. This is the experiment that turns "time
allocation" from a decode-time capability into a **representation-level training claim**.*

## The claim under test

Real demonstration data varies in execution speed (different operators, different
teleop rigs, cross-dataset pooling at different control rates). A policy's action
representation either **factorizes** motion shape from motion timing, or it
**entangles** them:

- **A (waypoint chunk)**: the target is "position at t+1·dt, t+2·dt, …". Two demos of
  the same skill at different speeds give the *same observation* different per-step
  delta targets → speed ambiguity contaminates the **shape** supervision.
- **B (spline, fixed T)**: the chunk is "path covered in the next 20 steps" — a
  *speed-dependent* prefix of the path. Same contamination, in control-point space.
- **C (spline + time allocation)**: chunks end at *motion events*. The spatial segment
  between events is **speed-invariant**; only the duration label varies. All speed
  ambiguity is isolated into the one channel built to express it. Shape supervision
  stays exact.

Prediction: under controlled speed heterogeneity, success drops
Δ(A), Δ(B) >> Δ(C) ≈ 0 (each arm vs its clean-data twin, which we already trained).

## Synthetic speed heterogeneity (controlled, causal)

Each episode gets a deterministic speed factor **s ∈ {1.0, 1.5, 2.0}** by
`episode_index % 3` (s>1 = slower demo; we avoid s<1 because speeding up the commanded
path 2× would clip ~8.6% of deltas at the [-1,1] actuator bound → corrupted synthetic
demos). The synthetic demo is *the same spatial path executed s× slower*. Only the
**action window** is transformed (SmolVLA conditions on the current frame only, so no
image resampling is needed — this is what makes the experiment feasible).

Per-arm target construction (all arms see the SAME synthetic demos):

| arm | transformation | where implemented |
|---|---|---|
| A | cumulative path → linear resample at s× steps → diffs → first `chunk_size` deltas (gripper: nearest-sample). Fetch window = chunk_size × 2. | new `smolvla_speedaug` wrapper policy, `forward()` transforms `batch[ACTION]` then defers to SmolVLA |
| B | same path resample; fit the 6 control points on the first 21 synthetic-rate path points | `speed_aug` flag in spline pkg, `_build_spline_targets` |
| C | **event segmentation and shape fit on the RAW trajectory (unchanged!); duration label T ← s·T** | one line in the duration-label path |

The C row is the whole thesis in implementation form: speed augmentation touches one
channel. It also preserves the h24 density law (fit support stays ≤24 raw steps) —
a slower demo does not smear its shape supervision.

Note C's decoded h_exec = T̂ may now exceed horizon_max (up to 48): the decode already
supports arbitrary h_exec; duration stats (logT mean/std) must be regenerated with the
s-scaled labels → `spline_stats_libero_v2_h24_saug.json`.

## Fairness rules

1. Same synthetic demos for every arm (same episode→s map), same training budget.
2. Eval at native 20Hz, standard suites, but **episode step budget ×1.75 for ALL arms**
   (policies legitimately learn slower average behavior; a fixed budget would punish
   slowness, which is not the quantity under test). Report steps-to-done alongside.
3. A's per-step MEAN_STD action stats are computed from raw-rate data; synthetic
   slow deltas are smaller → slightly non-unit variance. Acceptable for pilots; noted.
4. Clean twins = the existing 20k pilots (A 93 / B 91 / C-h24 93 on libero_object).

## Decision gates

- **Pilot (20k, libero_object + libero_10 evals, ~80 min/arm):** if Δ(C) < Δ(A) and
  Δ(B) by more than ~6 pts (the pilot noise floor we calibrated), scale to 100k.
- If C ALSO drops: the factorization story is wrong or the duration head can't absorb
  bimodal timing — either way that's informative; investigate calibration spread first
  (the duration head should show the s-multimodality).
- Follow-on if it works: decode-time speed restoration (speedup_alpha on a
  slow-demo-trained C restores native-speed execution — "learn from slow demos,
  execute fast" — a capability neither A nor B can express).

## Threats / honest caveats

- LIBERO demos already contain natural speed variation; the augmentation makes it
  severe and controlled. Both facts go in the writeup.
- Prior-work check (July 7 2026, web): the closest works are (a) **ISR / Trajectory
  Standardization** (arXiv 2606.22907, June 2026) — offline *preprocessing* that
  resamples demos to information-equidistant spacing, +25% success from removing
  speed non-uniformity → confirms the phenomenon is real and current, and treats
  timing variability as noise to delete; (b) **Variable-Frequency IL** (arXiv
  2411.12310) — speed as a *command input* (conditioning), not dataset heterogeneity.
  Our positioning is distinct from both: demo-speed variability as a *representation*
  question — fixed-time trajectory chunks entangle it with shape (measured
  sample-efficiency tax), a duration head factorizes it into an explicit, calibrated,
  reusable output (measured T-hat absorption). Unlike ISR, no preprocessing and the
  timing information is *retained* (replayable at any speed) rather than discarded.
- s assignment by episode parity means visually similar contexts (not identical
  frames) carry conflicting speed labels — matches the realistic ambiguity.
