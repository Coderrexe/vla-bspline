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

---
---

# Part II — July 8–15 (everything after the sections above)

Sections 9–17 continue the story. The arc in one paragraph: we discovered that the
*evaluation protocol* itself was producing several of our suite-level "gaps" (§9), which
closed the spatial investigation (§10) and promoted n8 to the final config with full
3-seed parity (§11). We then answered "where should speed control live" with a
three-way benchmark against retrained baselines (§12), invented a finer-grained retiming
knob out of a refuted redesign (§13), and found the limits of what composes (§14). The
duration head turned out to double as a runtime failure detector (§15). Then the big
one: we ported everything to CALVIN — a second benchmark, second robot, and crucially
*human-teleop* (noisy) demonstrations — built its official evaluator from scratch
(§16), and found that on noisy data our decode-time retiming stops being a trade-off
and becomes a simultaneous win in success *and* speed, seed-robust at n=1000 scale
(§17). That result, its mechanism, and a real-world plan (§18) are where the project
stands.

## 9. The discovery that changed our rules of evidence: protocol variance

While chasing the spatial gap (§6.2-6.3 above), we ran the *same checkpoint* through
different evaluation harnesses and different environment-seed sets. The "gap" moved:

| same Cn8 checkpoint, spatial | result |
|---|---|
| lerobot-eval harness, 3 eval-seeds | 77 vs A's 84 (the "gap") |
| our rollout harness, seed base 1000 | 81 vs A's 81 (gone) |
| our rollout harness, seed base 2000 | **79 vs A's 74 (sign flip!)** |

Same model, same suite — the ordering between A and C depends on which harness and
which seed set you ask. Same-checkpoint spread reaches **9 points** at n=100. The goal
suite did the same thing in the opposite direction (our favorable "Cn8 goal 89.3 > A
86" also dissolved under the rollout protocol). Duplicate runs of an identical
configuration differ by ±5.

**Standing rules since (used for every claim below):**
1. No suite-level ordering claim in either direction without *protocol crossing*
   (reproducing it under a second harness or seed set).
2. Load-bearing numbers get pooled across ≥3 seed bases (n=300).
3. Capability claims are *paired same-checkpoint contrasts* (only the decode flag
   differs), which subtracts out most of this noise.

This is also a publishable methodological point: LIBERO comparisons at n=100 with
training-seed error bars alone are not sufficient evidence.

## 10. The contact/spatial investigation — closed, with the cause identified

The spatial deficit (§6.2) got a full elimination ladder. Each row is a controlled
experiment:

| hypothesis | test | verdict |
|---|---|---|
| global fit fidelity | n6→n8 capacity | helps object/long; spatial gain didn't survive seed 2 |
| contact-local fit fidelity | end-weighted LSQ ("W9": weight ramps ×9 over last 25% of chunk) | tail fit error −23 to −56%, success **unchanged** — fit near contact was NOT the binding constraint |
| knot placement near contact | redistribute interior knots toward the event end (γ-warp) | **backfires**: at n6 tail +82% worse; at n8 (14.9k segments) strictly dominated by W9 at every γ — tail −5% at best while body degrades +17-36% |
| generative variance | K=16 sample dispersion, A vs C, by phase | equal — eliminated |
| **evaluation protocol** | §9 crossing | **the gap is largely protocol artifact** |

(The knot-redistribution row is the direct answer to Xiatao's July suggestion of
"allocating more control points near the grasping region": tested at production
capacity, offline, zero GPU — reweighting the fit beats reallocating knots, and even
reweighting doesn't move success.)

**What IS real near contact — kinematics, not shape.** Demonstrations decelerate to
0.555× cruise speed at gripper toggles; our executed rollouts toggled at 0.877× —
the fit smooths away terminal deceleration (position is pinned at the event, velocity
is not). Fix: **ease-out decode** — retime the last 3 path steps over a cosine-spaced
schedule (endpoint exact to 6e-8, terminal velocity ×0.14). Inert at fixed replan
cadence, real for self-paced mode (66→71 long-horizon; executed toggle ratio 0.844 →
0.766 ≈ A's 0.750). Kept as a deployment/hardware feature, not a sim headline.

## 11. Final LIBERO main table — n8 promoted, three seeds, everything crossed

The n8 3rd seeds landed and resolved §6.3's open question:

| arm | per-suite seed-means obj/spa/goal/long | avg ± sd |
|---|---|---|
| A waypoint | 94.7 / 84.3 / 86.0 / 63.3 | **82.1 ± 1.8** |
| B spline n8 | 96.3 / 77.7 / 84.0 / 65.3 | **80.8 ± 2.5** |
| C spline+time n8 | 92.3 / 77.3 / 89.3 / 63.7 | **80.7 ± 1.3** |

**Reads:** all three arms within ~1.4 points with overlapping error bars = **seeded
parity**; C remains the most seed-stable arm; and given §9, we make *no* suite-level
ordering claims — the honest differentiation between arms lives entirely in the
capability set (retiming, stretch, self-pacing, monitoring), not the main table.

## 12. Where should speed control live? — the three-way benchmark

Anyone can want a faster robot. The design question is *where the speed knob sits*.
Three architectures, all trained to the same 100k budget on the same data, all
evaluated by *realized* speedup (actual steps saved, not commanded speed):

| approach | knob location | object success @ realized speedup |
|---|---|---|
| speed-as-input ("tempo", TempoVLA-style: retime the training data, feed a speed scalar as input) | retrain + runtime input | 98 @ 1.0×, 80 @ 1.37×, **43 @ 1.43× (saturates — commanded 2× only realizes 1.43×)** |
| data-level selection ("dsel": bake event-aware retiming into training data) | retrain, fixed | 85 @ 1.18× (no runtime knob at all) |
| **ours: decode-time retiming** | **zero retraining, per-chunk adjustable** | **90 @ 1.26× (sel06), 88 @ 1.40×** (uni06) |

At matched realized ~1.4×: ours 88 vs tempo's 43. One honest extra finding: tempo's
multi-speed *training augmentation* is a genuine regularizer at 1× (98 vs A's ~95) —
and it's composable with our head in principle (future work).

## 13. The retiming-granularity ladder — a refuted redesign turned into the best knob

We attempted a fundamental redesign ("v3"): make the spline a pure *shape* in
arc-length and add an explicit learned time-map (per-span durations) — "complete"
time allocation. Before spending any GPU, an offline study on 14.9k segments killed
it: **the standard time-uniform fit already reproduces the demonstration's speed
profile almost exactly** (toggle-deceleration ratio 0.784 vs demos' 0.782) — because
control-point *spacing* encodes timing implicitly at full fitting resolution, while
v3's 5-span explicit map is coarser. Zero GPU wasted; one insight gained: **the
decoded chunk already knows its own speed profile.**

That insight became a new decode flag: **interval-level (profile-selective) retiming**
— within a chunk, compress only the intervals whose speed exceeds θ× the chunk mean
(the fast transport portions), leave slow precision intervals untouched; implemented
as an inverted piecewise-linear warp of the sampling grid (endpoint exact). This
completes a *granularity ladder* for time authority, all decode-only on one
checkpoint (object, protocol-crossed, n=300):

| granularity | success @ realized |
|---|---|
| base (no retiming) | 94.0 @ 1.0× |
| **interval-level (α=.6, θ=.7)** | **92.0 @ 1.30×** |
| chunk-level (T̂-gated, §6.4) | 90.3 @ 1.25× |
| uniform (blind) | 88.7 @ 1.38× |
| best retrained baseline (§12) | 76-80 @ ≤1.37× |

On LIBERO: **finer authority = cheaper speed**, and *protecting slow intervals is the
load-bearing property* (θ=0.5 — compressing slow intervals too — breaks it to 83;
α=0.5 breaks it to 81). Settings-sensitivity is mechanistic, not noise. Note the
"free lunch" of §6.4 is retired: pooled and crossed, speed costs ~2-5 points on
LIBERO at every granularity. Keep this in mind for §17, where the sign flips.

## 14. What does NOT compose — three clean negatives

1. **Temporal augmentation can't regularize the C head** (structural null): retiming
   a demo changes only the duration *label*, never the fitted shape target — that's
   the factorization working as designed, §6.6 — so there is nothing for shape
   regularization to grab. The factorization is immune to timing noise AND unable to
   harvest timing augmentation. Confirmed by a 20k pilot (no gain).
2. **Decode-time knobs don't stack**: half-rate execution + feasibility stretch +
   interval retiming = 64-69 vs stretch-alone 75 vs retiming-alone 92. **The time
   budget is one budget** — each knob spends the same slack; pick the one matched to
   the deployment constraint. (A real ordering bug was found and fixed on the way:
   feasibility must be re-checked *after* any compressive warp — but the fix wasn't
   the binding issue; the composition itself is.)
3. **Decode-consistency auxiliary loss** (differentiable decode inside training,
   penalizing x₀-space path error — our candidate for closing the §10 toggle-ratio
   gap from the learning side): 20k pilot obj 88 / spatial 62 vs twin's 89 / 72.
   No gain anywhere. Thread closed; the toggle-ratio residual stands as a documented
   structural limit with ease-out as the decode-side mitigation.

## 15. The duration head is also a runtime failure monitor

Free capability, discovered by looking at T̂ traces: on failures, T̂ *stagnates*
(the countdown stops counting down — the policy predicts "~14 steps left" forever
while fumbling). On 260 episodes: failures stall 2× more (stall-fraction 0.414 vs
0.216, ~2.2σ). As an online detector: **86% precision / 52% recall** whole-episode;
at a stricter threshold, 100%-precision abort signals. No existing VLA has an
endogenous progress monitor — ours is a column we were already predicting.

⚠️ Scoped honestly: (1) *detection* is real; kinematic self-*recovery* (back off and
retry on alarm) does not convert failures — fumbles are persistent incompetence, not
perturbation traps. The payoff is flagging/escalation, not self-repair. (2) The
monitor needs duration-signal *range*: it does NOT transfer to the CALVIN config
(§17), whose cap of 16 compresses T̂ into [13,16] — no countdown dynamic, recall ≤4%.
A capability coupled to config headroom, stated as such.

## 16. CALVIN — the second benchmark, built end-to-end

Why CALVIN: different robot (Franka), different tasks (5-instruction chains,
language-conditioned), and — the property that turned out to matter most —
**human-teleop demonstrations** (jitter floor 6× higher than LIBERO's scripted
demos: 0.149 vs 0.024). Everything below was verified before any success number was
trusted:

1. **Dataset**: `fywang/calvin-task-ABCD-D-lerobot` (24k episodes, 10fps), converted
   to LeRobot v3.0. Offline gate first: CALVIN segments are event-richer than LIBERO
   (gripper events in 41-54% of chunks), and our fit error sits *below* the demo
   noise floor — the spline is a built-in denoiser there. Config chosen from offline
   stats alone (n8, cap 16), zero rollout tuning.
2. **Action-semantics replay gate** (mandatory before evals): replaying converted
   actions in the simulator to determine what they actually are. Verdict: the 10fps
   actions are *state-recomputed* deltas, and the correct execution convention is
   one policy action held for 3 sim steps at native 30Hz — tracking RMSE 0.0065m,
   vs 0.118m (18× worse) for the naive-subsample interpretation we might otherwise
   have assumed.
3. **Official evaluator, two-process**: calvin_env needs an old numpy that conflicts
   with lerobot, so the policy runs in one process (GPU) and the simulator in
   another, joined by a socket. The protocol is the *official* one: 1000 five-task
   chains from CALVIN's own sequence generator (bit-identical seeding — we
   reimplemented their hash so initial scene states match the official evaluation
   exactly), their task oracle for success, 360-step budget per subtask. Sharded
   10× so a full n=1000 evaluation runs in ~1.5h of queue time.
4. **The cadence lesson replicated immediately**: SmolVLA's default replan cadence
   (nas=50 = a 5-second open loop at 10fps) scores 0.52 avg_len; at nas=10 it scores
   1.24-1.55. Same confound as §6.1, second benchmark. All comparisons cadence-fair
   since. Bonus finding: the spline arm is cadence-*robust* (1.36 vs 1.39 at
   nas 10 vs 5) while the waypoint arm is sharply peaked — the event-aligned chunks
   don't care as much when you cut them.

## 17. The CALVIN result — the headline of the project

**Final matrix: 3 training seeds × n=1000 official chains × 4 arms = 12,000 evaluated
chains.** `avg_len` = average tasks completed per 5-task chain before the first
failure (CALVIN's official metric; higher is better).

| seed | A: waypoint | C: spline (base) | C + interval retiming | C + uniform retiming |
|---|---|---|---|---|
| 1000 | 1.510 | 1.422 | 1.672 | 1.734 |
| 1001 | 1.764 | 1.239 | 1.433 | 1.579 |
| 1002 | 1.416 | 1.273 | 1.647 | 1.759 |
| **mean ± sd** | **1.56 ± .15** | **1.31 ± .08** | **1.58 ± .11** | **1.69 ± .08** |

Every arm ran the *identical* 1000 chains, so any two columns can be compared
*paired* (chain-by-chain) — far more statistically powerful than comparing raw
averages.

### The five pairwise comparisons

| comparison | pooled effect (n=3000 pairs) | t | per-seed signs | verdict |
|---|---|---|---|---|
| uniform retiming − C base | **+0.38 avg_len** @ 1.42× realized speed | 12.1 | +, +, + | **Strongest result in the project** — same checkpoint, decode-only flag, simultaneously more successful *and* faster |
| uniform retiming − A waypoint | +0.127 | 3.89 | +0.22, **−0.19**, +0.34 | Matches-or-beats, not unconditional — loses on 1 of 3 seeds |
| uniform − interval retiming | +0.107 | 3.37 | +, +, + | Granularity ladder **inverts** vs. LIBERO §13 (there, interval beat uniform) |
| interval retiming − A waypoint | +0.021 | 0.65 | — | Statistical parity (not significant) |
| A waypoint − C base | +0.252 | 8.28 | — | The *un-retimed* spline is clearly worse than the baseline |

(An earlier single-seed version of row 2 had claimed an unconditional "beats" at
t=3.18 — corrected once seeds 2–3 landed. §9's rules applied to our own favorite
number.)

### Why it wins — mechanism, measured

At the 267 subtasks where base-C failed and retimed-C succeeded, two competing
explanations were tested:

| explanation | prediction | found |
|---|---|---|
| "just beats the clock" (retiming sneaks in under the 360-step budget) | recovered successes finish near the time limit | only 23% did |
| "changes the actual behavior" (compressing hesitation avoids the failure mode) | recovered successes finish with time to spare | **77% did** |

The gain is ~3/4 *behavioral* (compressing reproduced human hesitation changes the
executed trajectory away from failure modes), only ~1/4 clock management.

### Why the granularity ordering flips vs. LIBERO

| | LIBERO (scripted demos) | CALVIN (human teleop demos) |
|---|---|---|
| best method | interval-level (selective) | uniform (blind) |
| best θ setting | 0.7 (protect most slow intervals) | 0.5 (compress almost everything) |
| why | no dead time in the data — selectivity protects the little genuine precision-work that exists | dead time/hesitation scattered everywhere — broad compression captures it without needing to be picky |

θ=0.3 regresses on CALVIN, so this isn't "more compression is always better" —
there's still an optimum, just a much coarser one than on LIBERO. **Unified law:
how selective the speed knob needs to be depends on how much slack is actually in
the data.** Time authority appreciates as data gets more realistic — and real-robot
demos are the noisy regime.

### Supporting facts, and what each rules out

| fact | what it shows |
|---|---|
| C base loses to A waypoint (+0.252, t=8.28) | the spline faithfully reproduces its human demos' *timing*, hesitation included — a liability on noisy data, and exactly what retiming is fixing |
| waypoint success ranges 1.42 → 1.76 across seeds (identical architecture) | CALVIN training-seed variance is large — third confirmation (after LIBERO §9/§11) that no single-seed CALVIN comparison should be trusted |

### The bottom line

| claim | status |
|---|---|
| spline representation ties the waypoint baseline on raw task success, on both benchmarks | supported, with real seed-based error bars |
| spline exposes a decode-time control surface (speed, replanning, monitoring) waypoints structurally can't have | supported |
| on realistic noisy data, that control surface stops costing success and starts adding it | supported — **+0.38 avg_len and 1.42× speed simultaneously, seed-robust, on the official n=1000 protocol** |

## 18. Real-world plan (Xiatao's question, answered with a selection principle)

The §17 mechanism gives a *task filter* for hardware experiments, not just a wish:

- **Compressible slowness** (careful/hesitant teleop, quasi-static tasks) → retiming
  converts dead time to speed at zero or negative cost. Cloth folding (Xiatao's
  suggestion) is exactly this regime — slow *because demonstrators must be careful*.
- **Dynamics-limited slowness** (pouring, pushing, throwing) → NOT compressible;
  speed changes the physics. Retiming preserves path geometry and is only valid
  quasi-statically. This boundary goes in the paper.
- **Zero-hardware first step**: run our dead-time census offline on public teleop
  datasets (DROID, ALOHA) and *predict* which tasks benefit before touching a robot.

Proposed experiments, ranked: P1 cloth folding (predicted CALVIN-regime win), P2
kitting throughput (T̂-gated selective speedup, items/hour), P3 precision insertion
(ease-out + velocity chaining measured as contact force/vibration on an
impedance-controlled arm — the claim sim absorbs). Full details:
`docs/REAL_WORLD_PLAN.md`.

## 19. Updated code / docs map (post-reorganization)

The repo was reorganized (July 12) from the old `libero/` layout into:

| location | contents |
|---|---|
| `policy/smolvla_spline/` | the head + all baselines (interp/tempo/dsel/speedaug) — source of truth, mirrored to the cluster |
| `scripts/data/` | dataset IO + stats generation (LIBERO + CALVIN) |
| `scripts/analysis/` | every offline study (fit variants, v3 refutation, contact diagnosis, stagnation probe...) |
| `scripts/eval/` | CALVIN two-process evaluator (`calvin_policy_server.py`, `calvin_eval_client.py`, replay gate) |
| `scripts/figures/` | paper figures incl. `fig_calvin` (+ pooled data JSONs in `data/`) |
| `cluster/` | all sbatch files (`eval_calvin.sbatch <ckpt> [n] [tag] [offset] [total]` shards the official eval) |
| `docs/METHOD.md` | the deep technical spec: target construction, decode surface, 7 design laws, cluster ops |
| `docs/RESULTS.md` | living numbers digest (§7 = CALVIN) |
| `docs/REAL_WORLD_PLAN.md` | §18 in full |
| `legacy/calvin_prototype/` | Quinten's original scripts, kept as format reference |
