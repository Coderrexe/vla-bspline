# Project summary

---

## 1. The idea, in one example

Every standard robot policy (SmolVLA included) predicts a chunk of, say, 50 future
waypoints, always spaced at a fixed timestep. We replace those 50 waypoints with **6
numbers that define a smooth curve** (a B-spline) through the same motion — a decode step
then samples that curve at however many points you actually need. Call this **B**. The
second idea, **C** ("time allocation"), is about *where the chunk ends*, and contrast makes
it clearest:

**Without time allocation (B):** every chunk is exactly 20 robot-steps long. Always. No
matter what happens in those 20 steps — maybe the robot grasps an object at step 4, maybe
at step 17 — you always fit your 6-point curve to that arbitrary 20-step slice. The curve
sometimes has to represent an ugly mix of "the interesting part" (a grasp) and "whatever
random thing came after," because the chunk boundary doesn't care what's happening.

**With time allocation (C):** instead of always stopping at 20, stop **right when
something meaningful just happened** — right after the gripper opens or closes (a
grasp/release), or after the arm pauses. One chunk might be 12 steps (a quick grasp), the
next might be 24 (the cap, if nothing eventful happened yet). Since chunk length now
varies meaningfully, the model also **predicts that length** — this prediction ("this
motion will take about 12 steps") is the "time allocation," and it becomes a usable signal
at deployment (§5).

That's the whole idea. Fitting a curve is unchanged; the only new things are (1) a smarter
rule for where a chunk ends, and (2) one extra number the model learns to output.

## 2. One training example, concretely

Robot transporting toward an object, grasps at raw step 12:

```
step:     0    1    2   ...  11   12   13  ...
gripper: -1   -1   -1  ...  -1   +1   +1  ...
```

1. **Find where it ends** — walk the gripper column comparing consecutive steps; first flip
   (`-1→+1`) is step 12. (No flip/pause anywhere → cap at 24, use the whole window.)
2. **Build the path** — cumulative-sum the first 12 pose-deltas into a 13-point trajectory.
3. **Fit 6 control points** to those 13 points (least-squares), first point pinned to
   `(0,...,0)`, last pinned to the exact endpoint. Now: **36 numbers** (6 points × 6 pose
   dims) describing the *shape* of this 12-step motion.
4. **Gripper gets its own small curve fit** the same way → 1 more number per control point.
5. **Duration label = `log(12)`** — the T from step 1 — written into a 7th channel,
   repeated across all 6 tokens (one number for the whole chunk, broadcast to fit the shape).

Target tensor, shape `(6, 8)` (6 control points × [6 pose + 1 gripper + 1 duration]):

```
token 0: [ 0, 0, 0, 0, 0, 0,        | grip_0, | log(12) ]   <- pinned to 0
token 1: [ c1_x, ..., c1_rz,        | grip_1, | log(12) ]
  ...
token 5: [ p12_x, ..., p12_rz,      | grip_5, | log(12) ]   <- pinned to exact endpoint
```

Batched: `(B, 6, 8)`. **Training** just regresses this tensor — same flow-matching loss
SmolVLA already uses, nothing new about *how* it learns, only *what* the target means.
**Inference**: the model outputs its own guess for channel 7; `exp()` it → **T̂**, the
model's own claim of "I think this motion takes ~12 steps," *before* the motion happens.

## 3. Why SmolVLA needed zero architecture surgery

SmolVLA's action expert denoises a tensor `(batch, chunk_size, 32)` via flow matching;
`chunk_size` is just a sequence length, nothing in the transformer cares what the tokens
mean. We set `chunk_size = 6` and slice the loss to the first 8 of 32 padded dims
(`configuration_smolvla_spline.py:112`, `_N_OUT = 8` in `modeling_smolvla_spline.py:83`).
Transformer, training objective, sampling procedure: **all completely unmodified**.

The B-spline math is the standard Cox–de Boor recursion (`bspline_basis()`,
`modeling_smolvla_spline.py:37-72`), validated against `scipy` to machine precision (max
error 2.8e-16). Fitting control points is **one precomputed matrix multiply per batch**
(`_pinv_mid`, `_pinv_grip` buffers) — no iterative solve, no scipy call at train time.

**Decode** samples the fitted curve at *any* number of points on request
(`_decode_tokens()`, `modeling_smolvla_spline.py:315-367`) — this is what lets one trained
checkpoint run at half-rate, double-rate, or any custom cadence with zero retraining, and
it's the hook every capability in §5 plugs into.

## 4. Three policies compared throughout

| name | what it is |
|---|---|
| **A** | plain, unmodified SmolVLA — 50 fixed waypoints |
| **B** | our spline head, chunks still fixed at 20 steps (no time allocation) |
| **C** | our spline head **+** time allocation — variable-length, event-aligned chunks + predicted duration |

B isolates "does the compact curve cost anything?" C isolates "does time allocation on top
of that cost or gain anything?" A strong resampling baseline (`smolvla_interp` — take A's
trained weights, linearly resample its output path at decode time) exists specifically to
red-team any claim that sounds like "only a continuous representation can retarget rate."

## 5. What the predicted duration (T̂) is used for

All decode-time only — same checkpoint, a config flag, zero retraining:

- **Rate retargeting** — sample the curve at a different rate than trained. Base capability.
- **Feasibility stretch** — if compressing into too few steps would demand an impossible
  single step, stretch the execution window instead of clipping (also implemented for the
  waypoint baseline, for a fair comparison).
- **Selective speedup** — if T̂ says "long motion" (probably transport, not a delicate
  grasp), execute it faster; leave short/careful chunks alone.
- **Self-paced replanning** — replan whenever the *predicted* motion is about to end,
  instead of every fixed N steps.
- **Velocity-continuous chaining** — match the new chunk's starting velocity to the old
  chunk's velocity at the moment it was cut off, removing a jerk discontinuity at replans.

---

## 6. Results — full tables

### 6.1 First proof it works (20k-step pilot, libero_object)

| policy | success |
|---|---|
| A waypoint, nas=50 (SmolVLA's own default) | 60% |
| A waypoint, nas=10 | **93%** |
| B spline, fixed-T | **91%** |
| C spline + time-allocation (pre-fix) | 85% |

(`nas` = replan cadence; discovering A's huge sensitivity to it is why every A-run since
fixes nas=10.)

### 6.2 Main table — 100k steps, 3 seeds/arm, all 4 suites

| policy | obj | spa | goal | long | avg |
|---|---|---|---|---|---|
| A waypoint (seeds 1000/1001/1002) | 93/93/98 | 83/86/84 | 84/88/86 | 60/65/65 | **82.1 ± 1.8** |
| B spline fixed-T | 95/89/91 | 80/80/70 | 86/85/78 | 70/59/62 | **78.8 ± 3.8** |
| C spline + time-alloc (h24) | 90/95/94 | 72/76/75 | 84/82/87 | 64/66/57 | **78.5 ± 1.1** |
| *published SmolVLA ref (1 seed)* | *96* | *90* | *92* | *71* | *87.3* |

**Reads:** B ≈ C on average (time allocation costs nothing over fixed-T), and **C is the
most seed-stable arm in the study** (±1.1 vs B's ±3.8) despite a strictly harder training
target. A's ~3.5-point lead over the spline arms is almost entirely libero_spatial (A 84.3
vs B 76.7 / C 74.3 as per-suite seed-means) — object/goal/long are parity within noise.

⚠️ The *2-seed* version of this table (before the 3rd seed landed) showed B swinging
82.75 → 78.25 seed-to-seed (long-horizon 70 vs 59) — large enough that single-seed
"B beats A" or "C trails B" readings are not trustworthy. **Standing rule since: never
report a single-seed suite difference under ~6 points as real.**

### 6.3 The capacity law — two single-variable fixes

Every deficit the spline arms showed traced back to one knob: *control points per env-step
of motion*.

**Fix 1 — long-horizon.** Capped chunks (up to 40 raw steps) were fit with only 6 control
points — too sparse. Cutting the cap to 24 steps ("h24"), controlled 20k comparison, single
variable changed:

| config | libero_10 (long) success |
|---|---|
| cap=40 (original) | 36 |
| cap=24 (h24 fix) | **57** (+21 pts) |

This is bedrock — confirmed again across C-final's full 3-seed table above (long seed-mean
62.3 ≈ B's 63.7).

**Fix 2 — spatial.** Control points 6 → 8 ("n8"), 20k pilot:

| config | spatial | object |
|---|---|---|
| B, n_ctrl=6 | 63 | 91 |
| B, n_ctrl=8 | **74** (+11) | **94** (+3) |

At 100k (1 seed each): **B-n8 = 98/87/84/66 (avg 83.75)** — at/above A's 3-seed mean on 3 of
4 suites; **C-n8 = 94/76/95/63 (avg 82.0) ≡ A's 82.1**, with goal=95 the best score any arm
has gotten on that suite in the project (published ref is 92).

⚠️ **This second fix did not replicate on a 2nd 100k seed:** B-n8 seed 1001 gave
spatial=**71**, not 87. Two-seed mean = 79 ≈ n6's 76.7 — the 87 was a favorable draw, not a
robust effect. **Current honest status: n8 robustly helps object and long-horizon; its
spatial-specific gain is real at 20k-pilot scale but unconfirmed at 100k.** A 3rd seed is
what decides whether n8 replaces n6 as the paper's headline config.

### 6.4 Duration-aware selective speedup — the flagship capability

n=100, libero_object, C-final (n_ctrl=6, h24), "selective" = speed up (×α) only chunks with
predicted duration T̂ > 20 steps; "uniform" = speed up every chunk:

| α | selective success @ steps | uniform success @ steps | selectivity gap |
|---|---|---|---|
| 1.0 (no speedup) | 93% @ 145.3 | anchor | — |
| 0.8 | 94% @ 124.1 | 90% @ 116.5 | +4 |
| 0.7 | 85% @ 119.4 | 85% @ 110.4 | 0 |
| **0.6** | **95% @ 122.7** | 73% @ 109.6 | **+22** |
| 0.5 | 76% @ 113.2 | 57% @ 97.5 | +19 |

**Free lunch: selective α=0.6 = 95% success, 16% faster** than no speedup at all (93% @
145.3 steps). Blind speedup at the same α loses 20 points. Control: B (no duration head, so
uniform is its only option) at α=0.6 scores 77% ≈ C-uniform's 73% — the gap is from
*selectivity*, not the spline representation.

**Suite generalization (n=100/cell) — this free lunch is libero_object-specific:**

| suite | no speedup | selective α=0.6 | uniform α=0.6 |
|---|---|---|---|
| object | 93 @ 145.3 | **95 @ 122.7** | 73 @ 109.6 |
| goal | 90 @ 111.1 | 62 @ 87.3 | 53 @ 78.9 |
| spatial | 77 @ 109.1 | 47 @ 88.4 | 47 @ 81.8 |
| long | 65 @ 270.1 | 49 @ 216.6 | 48 @ 174.7 |

Why: "long predicted duration ⇒ rushable transport" is true on object, false on
spatial/goal, where long chunks are often slow *precision* moves, not transports. Duration
alone predicts rushability only when a task family separates transport from precision by
length. Mild α=0.8: nearly free on object/long, costs 6-12 pts on spatial/goal.

**Re-tested on the denser n8 config — survives, relocated:**

| α | selective @ steps | uniform @ steps | selectivity gap |
|---|---|---|---|
| 1.0 | 90 @ 143.4 | anchor | — |
| 0.6 | 90 @ 114.6 | 88 @ 102.9 | +2 (was +22 on n6!) |
| 0.5 | 84 @ 112.5 | 73 @ 105.0 | **+11** |
| 0.4 | 63 @ 116.3 | 42 @ 89.6 | **+21** |

Denser shapes tolerate blind α=0.6 with almost no penalty; the selectivity gap doesn't
vanish, it **moves** to a harsher α (0.4-0.5). Mechanism confirmed twice, at two different
representation capacities: selectivity pays exactly where blind speedup starts breaking
grasps — that frontier just shifts with capacity.

### 6.5 Self-paced replanning + the margin design law

n=100, C-final:

| replan policy | object | calls/ep | long | calls/ep |
|---|---|---|---|---|
| fixed every 5 (default) | 93% | 31.3 | 61% | 73.8 |
| fixed every 12 | **94%** | 13.4 | **66%** | 32.9 |
| self-paced, full T̂ | 84% | **9.2** | 55% | **23.2** |
| B, full fixed chunk (matched compute) | 91% | 8.3 | 56% | 19.2 |

Self-paced *works* (84/55% at 3.2-3.4× fewer model calls than default) but does **not**
beat a well-tuned fixed cadence. ⚠️ A 30-episode pilot had suggested self-paced *won* on
long-horizon (73.3%!) — n=100 reversed it to 55%. Same "pilot deltas under ~15 points are
noise" lesson, independently re-discovered.

**The negative result, converted into an understood design law.** Hypothesis: replanning
exactly *at* the predicted event clips gripper toggles right at the chunk boundary, where
duration-prediction error hurts most. Test: replan a fixed margin *before* the predicted end:

| replan at | object | calls/ep | long | calls/ep |
|---|---|---|---|---|
| T̂ (margin 0) | 84 | 9.2 | 55 | 23.2 |
| T̂ − 2 | 90 | 9.6 | 57 | 27.9 |
| **T̂ − 4** | **93** | **11.7** | **62** | 33.0 |
| fixed nas=5 (reference) | 93 | 31.3 | 61 | 73.8 |

Monotone, confirmed causal. **Design law: replan before the predicted event, never at it.**
With a 4-step margin, duration-scheduled replanning matches default-cadence success at
**2.7× fewer model calls** — meaningful compute savings for a 450M-parameter policy, with a
mechanism (not just a number) that generalizes to any event-terminated action-chunk policy.

### 6.6 Speed-heterogeneous demonstrations — the training-time argument

Real demo data mixes operator/teleop speeds. We simulated this: every training episode
replayed at 1×, 1.5×, or 2× slower (deterministic per-episode factor), same synthetic demos
shown to all three arms. Prediction: waypoints (A) and fixed-T spline (B) should suffer
because the same observation now maps to different targets at different speeds; time
allocation (C) should be immune because — by construction — only the duration *label*
changes with speed, the fitted *shape* target is untouched (machine-verified: control
points bit-identical across simulated speeds, endpoint scales exactly with 1/s for B).

**20k pilot vs. clean 20k twins, object/long, 100 eps:**

| arm | clean | speed-aug | Δ |
|---|---|---|---|
| A waypoint | 93 / 51 | 95 / 58 | **+2 / +7** |
| B spline fixed-T | 91 / 52 | 90 / **39** | −1 / **−13** |
| C spline + time-alloc | 93 / 57 | 87 / 53 | −6 / −4 |

**Two findings.** (1) Waypoints *absorb* speed heterogeneity — an honest null against the
original prediction. Per-step deltas across speeds are colinear (same path, different
magnitude), so averaging still points the right way, and closed-loop replanning re-syncs
progress. "Fixed-dt breaks under mixed speeds" is false for closed-loop deployment with a
fair time budget. (2) **Fixed-time chunks (B) are the vulnerable representation**: the
chunk boundary is time-defined, so the same observation is asked to produce different
*shapes* at different speeds — real shape-blur, −13 on long-horizon. C isolates this into
the duration channel and holds at −4 — **C beats B by +14 points on long-horizon under this
stress.**

**100k confirmation (1 seed each) — the contrast attenuates with budget.** Δ(obj/long):
A −3.7/−5.3, B −3.7/−6.7, **C −2.0/−1.3**. With 5× more steps even B's blurred targets
mostly converge; ordering survives (C least degraded at both budgets) but the 100k gap is
within single-seed noise — **honest framing: a sample-efficiency tax on fixed-time chunks
that time allocation largely waives, not a standalone permanent headline.**

**Mechanism confirmed end-to-end.** Measuring the duration head's T̂ distribution directly:
clean C predicts mean T̂=20.8 (cap-pinned); speed-aug C predicts mean **T̂=27.0, p90=40,
max=48** — a ~1.3× shift, matching the theoretical prediction (`e^E[log s] ≈ 1.44×` — the
Bayes-optimal response to *unresolvable* per-episode speed ambiguity is roughly the
geometric mean of possible durations). The speed variation went exactly where the design
sends it — the duration channel — while shape channels stayed provably clean.

---

## 7. Honest caveats (see `PROJECT.md` §8 for full detail)

- An early "policy fails catastrophically at the wrong control rate" result was traced to a
  weak checkpoint + an unfair episode-budget artifact — **retracted**, not used anywhere.
- Self-paced replanning's 30-episode pilot win reversed at n=100 (§6.5) — the origin of the
  "don't trust small-pilot deltas" rule.
- n8's "spatial gap erased" claim did not survive a 2nd seed (§6.3) — provisional.
- A T̂-distribution measurement was silently truncated by a decode clamp bug (clamp was set
  to the raw fetch cap instead of the speed-augmented label's true max); caught because the
  first (null) result contradicted theory, then fixed and re-measured (§6.6).

---

## 8. Where the code lives

| file | what's in it |
|---|---|
| `libero/smolvla_spline_pkg/configuration_smolvla_spline.py` | all config flags (n_ctrl, horizon_max, speedup_alpha, replan_margin, speed_aug, ...) |
| `libero/smolvla_spline_pkg/modeling_smolvla_spline.py` | spline fit, event detection (`_first_event`), decode, every capability flag |
| `libero/smolvla_spline_pkg/smolvla_interp.py` | strong resampling baseline (red-teams the Hz-decoupling claim) |
| `libero/smolvla_spline_pkg/smolvla_speedaug.py` | waypoint arm of the speed-heterogeneity study |
| `RESULTS.md` | living numbers digest |
| `PROJECT.md` | full technical writeup (this doc's source) |
| `SPEED_AUG_DESIGN.md`, `C_FIX_DESIGN.md` | design docs for the two biggest sub-investigations |
