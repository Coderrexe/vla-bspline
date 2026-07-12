# B-Spline + Time-Allocation Action Head for SmolVLA — Results Digest

*LIBERO benchmark, LeRobot pipeline. All evals: 100 episodes (10 tasks × 10), replan every 10 env steps (nas=10) unless noted. Updated July 12, 2026.*

## Method in one line
SmolVLA's flow-matching expert generates **6–8 B-spline control-point tokens** (+ gripper channel + duration channel) instead of 50 waypoints; a decode samples the continuous trajectory at **any** execution rate. v1 = fixed chunk time (2s). v2 = **time allocation**: chunks end at motion events (gripper toggles/pauses), duration is a learned output. Promoted config: **n_ctrl=8, horizon_max=24 ("C-n8")**. Framing note: `TIME_AUTHORITY.md` — the executor keeps authority over the time map after training; the duration channel is the interface for exercising it safely.

## 0. State of the project (executive summary, July 12)
1. **Native-rate parity, fully seeded** (§1): A 82.1±1.8 / B-n8 80.8±2.5 / C-n8 80.7±1.3;
   no suite-level ordering survives protocol-crossed evaluation in either direction
   (§4c) — differentiation lives entirely in the capability set.
2. **The capability set, each claim red-teamed** (all on the promoted config):
   feasibility stretch **+19 at half rate** (§2); decode-time speed knob —
   protocol-crossed at n=300: 20–28% time reduction for ~4–5 pts at α=0.6,
   selectivity paying at the capacity-dependent frontier (+21 at α=0.4) (§3);
   duration-scheduled replanning at **2.7× fewer policy calls** (§3c);
   **endogenous failure detection** — 86% precision, live-validated; kinematic
   self-recovery scoped as a negative (§3e); soft contact landing via ease-out
   decode (terminal velocity ×0.14, exact endpoint) (§4b); shape/timing
   factorization absorbs demonstration-speed heterogeneity (§3d); calibration
   MAE 2.8 on the promoted config.
3. **Benchmark head-to-head, budget-matched at 100k** (BENCHMARK_DESIGN.md): the
   speed-control triad — speed-as-input saturates at realized 1.43× and collapses
   to 43% where our decode retiming holds 88%; data-level selectivity is safe but
   fixed; ours is the only zero-retraining, per-chunk-adjustable mechanism.
   Multi-speed augmentation is a genuine 1×-regularizer for waypoint heads (98%);
   duration-only augmentation is not (composition pilot null, July 12).
4. **Methodological finding** (§4c): LIBERO suite comparisons at n=100 carry
   ±5–9 pt protocol/seed-set variance — training-seed error bars are insufficient.
5. Figures banked on the promoted config: countdown timeline, calibration,
   dose-response Pareto, realized-speed head-to-head.

## 1. Native-rate success (the "does it cost anything?" table)

**20k-step pilots, libero_object:**
| policy | success |
|---|---|
| A waypoint SmolVLA (nas=50, its default) | 60% |
| A waypoint SmolVLA (nas=10) | **93%** |
| B spline fixed-T | **91%** |
| C spline + time-allocation | 85% |

**100k full budget, all four suites, 3 SEEDS PER ARM (n6 July 7; n8 complete July 8):**
| policy | obj | spa | goal | long | avg over suites |
|---|---|---|---|---|---|
| A waypoint (3 seeds) | 93/93/98 | 83/86/84 | 84/88/86 | 60/65/65 | **82.1 ± 1.8** |
| B spline fixed-T n6 (3 seeds) | 95/89/91 | 80/80/70 | 86/85/78 | 70/59/62 | **78.8 ± 3.8** |
| C spline + time-alloc n6 (3 seeds) | 90/95/94 | 72/76/75 | 84/82/87 | 64/66/57 | **78.5 ± 1.1** |
| **B spline n8 (3 seeds)** | 98/97/94 | 87/71/75 | 84/82/86 | 66/67/63 | **80.8 ± 2.5** |
| **C time-alloc n8 (3 seeds)** | 94/91/92 | 76/79/77 | **95/89/84** | 63/63/65 | **80.7 ± 1.3** |
| *published SmolVLA ref (1 seed)* | *96* | *90* | *92* | *71* | *87.3* |

**Headline (fully seeded):** with the capacity knob set (n8), both spline arms sit
within 1.3 pts of the waypoint baseline (80.8/80.7 vs 82.1, overlapping CIs) —
**parity, seeded, on the promoted config**. C remains the most stable arm at both
capacities (±1.1/±1.3). *(C-n8's goal seed-mean 89.3 vs A's 86.0 does NOT survive
protocol crossing — rollout harness gives A 90–93 vs C-n8 84–87 across two seed
bases, the mirror of the spatial flip. We red-teamed our own favorable number with
the §4c methodology and it dissolved like the unfavorable one: at these sample
sizes, NO suite-level ordering claims are defensible in either direction. The
table's claim is parity; the differentiation is the capability set.)*

**⚠️→✅ The "spatial gap" DISSOLVED under protocol replication (July 9).** The
apparent ~7-pt spatial deficit (lerobot-eval, 3 training seeds) does not survive
crossing the *evaluation* protocol: under the rollout harness the same checkpoints
give A 81 / C-n8 81 (seed base 1000) and **A 74 / C-n8 79 (seed base 2000 — sign
flipped)**. Per-protocol spatial eval noise (±5–7 per 100 episodes) exceeds the
claimed gap; training-seed replication had been replicating a fixed protocol, not a
truth. Full elimination ledger that got here: capacity ✗ (n8, seed-fragile),
contact-weighted fit ✗ (W9, clean negative), generative variance ✗ (equal dispersion),
protocol ✓ (the actual cause). **Resolution of the contact-rich direction: no robust
success deficit remains at n8; the one real kinematic difference (under-deceleration
at gripper toggles, 0.88× vs demos' 0.55×) is fixable decode-side (ease-out: terminal
velocity ×0.14, exact endpoint) and matters for self-paced execution (66→71 spatial,
toggle ratio →0.766 ≈ A's 0.750) and plausibly for real hardware where the controller
is less forgiving than sim OSC. Methodological takeaway for the field: LIBERO
suite-level comparisons at ~100 episodes need protocol-crossed evaluation (seed sets ×
harnesses), not just training-seed error bars.**

**Honest 3-seed reads:**
1. **B ≡ C (78.8 vs 78.5): time allocation costs nothing** relative to fixed-time —
   and C is the most seed-stable arm in the study (±1.1 vs B's ±3.8).
2. **A leads the spline arms by ~3.5, localized almost entirely in libero_spatial**
   (per-suite seed-means: A 84.3 vs B 76.7 / C 74.3; object/goal/long are parity
   within noise). Spatial = precision placement: the compression (6 control points)
   costs fine spatial detail exactly where precision dominates.
   **CONFIRMED CAUSAL (July 7): B with n_ctrl=8 at 20k lifts spatial 63 → 74 (+11,
   past the ±6 pilot floor; object 91 → 94).** Together with h24 (+21 on long), the
   general design law: *control-point density per env step is the master capacity
   knob for spline action heads* — 6/40 breaks long-horizon, 6/20 dents spatial
   precision, 8/20 recovers it.
   **At 100k: B-n8 seed1000 = 98 / 87 / 84 / 66 (83.75); C-n8 seed1000 = 94 / 76 / 95
   / 63 (82.0 ≡ A's 82.1).** C-n8's goal 95 is the project's best on that suite.
   ⚠️ **BUT the "spatial gap erased" claim did NOT survive a second seed:** B-n8
   seed1001 = 97 / **71** / 82 / 67 (79.25). B-n8 spatial two-seed = 87/71 (mean 79,
   range 16) ≈ B-n6's 3-seed 76.7 — the single-seed 87 was a favorable draw, not a
   robust capacity effect. **Honest status: at 100k, n8 lifts object (97–98, best in
   study) and long (66–67) but its spatial gain over n6 is within seed noise.** The
   20k n8 spatial pilot (+11) reflected pilot-scale luck too. Density robustly fixes
   *long-horizon* (h24, confirmed across C-final's 3 seeds); its effect on *spatial
   precision* is real at 20k but seed-fragile at 100k. Third seeds (training) decide
   whether n8 is promoted to the main table at all. **The capacity law's long-horizon
   arm is bedrock; its spatial arm is provisional.**
3. C's long-horizon seed-mean (62.3) ≈ B's (63.7): the density fix closed the C-gap
   (controlled 20k comparison: cap-40 → h24 = long 36 → 57, **+21 pts**, single-variable).

→ The compact spline representation (36 floats vs 350) matches waypoints at native rate.
→ **C's gap resolved (July 5): evaluating C with replan-every-5 (nas=5) → 91% on object, parity with B — zero retraining.** Mechanism: event-terminated chunks interact with replan cadence (the executed window must reach predicted events). C rows below being re-run at nas=5.

## 2. Control-rate robustness — HONEST version (with the strong baseline)

Deploy at a different control frequency than training (fair wall-clock budgets, libero_object).
We red-teamed our own claim by building the baseline any practitioner would use: **linear
path-resampling of the waypoint chunk** (`smolvla_interp`, same trained weights):

| policy | f=10 (half) | f=20 (native) | f=40 (double) |
|---|---|---|---|
| A waypoint, naive | **0%** | 93% | **0%** |
| A waypoint + interp resample (20k) | 67% | 93% | 59% |
| **A interp + our stretch (20k)** | **85%** | — | — |
| B spline (100k) | 56% | **95%** | 50% |
| B spline (100k) + stretch | 72% | — | — |
| C time-alloc n6 (100k) | 50% | 91% (nas5) | 52% |
| C time-alloc n6 (100k) + stretch | 66% | — | — |
| **C time-alloc n8 (promoted, July 9)** | 56% | 92–94% | **62%** |
| **C n8 + stretch** | **75% (+19)** | — | — |

Takeaways (reshaped, more defensible):
1. **Naive deployment fails catastrophically (0%)** — rate-adaptation must be an explicit mechanism.
2. **Feasibility-aware time stretching is our key algorithmic contribution and is representation-agnostic**: +18pts on interp-waypoints, +16pts on splines. The idea — treat the chunk as a continuous path; when actuator bounds would clip, execute the *same* path over a longer horizon — originates from the continuous-trajectory view.
3. Path-resampled waypoints retarget rate as well as (currently better than) the spline head at 100 eps/1 seed — so the spline's case rests on its **irreducible** advantages: lower jerk (2.3× vs naive waypoints at native rate; ~1.2× vs the strong interp baseline at half rate, *with* higher success — see below), 5.8× compression at no native cost, analytic C¹/C² derivatives (interp velocity is piecewise-constant, acceleration is a train of impulses — mathematically untenable for impedance/velocity-feedforward control regardless of measured jerk), and the **duration head** (unpredictable by any resampler; §3).

**Half-rate (f=10) executed smoothness, interp vs spline (5 eps, task 0):**
| policy | cmd jerk | exe jerk |
|---|---|---|
| A interp + stretch | 0.063 | 1.6e-3 |
| **B spline + stretch** | **0.053** | **1.4e-3** |
→ with stretch, the spline is ~17% smoother than resampled waypoints *and* higher success
on this slice; the analytic-derivative argument is the deeper, benchmark-independent point.

## 3. Duration-aware selective speedup — the time head's payoff (CONFIRMED AT n=100)

**Full dose-response sweep, C-final (h24, 100k), libero_object, 100 episodes/point,
replan every 5 steps (July 6).** "Selective" = accelerate a chunk (×alpha) ONLY when
its predicted duration T-hat > 20 (long transport, not grasp-approach); "uniform" =
accelerate every chunk. Steps = mean env steps to completion, on successes.

| alpha | selective success @ steps | uniform success @ steps | selectivity gap |
|---|---|---|---|
| 1.0 (baseline) | 93% @ 145.3 | (same anchor) | — |
| 0.8 | 94% @ 124.1 | 90% @ 116.5 | +4 |
| 0.7 | 85% @ 119.4 | 85% @ 110.4 | 0 |
| **0.6** | **95% @ 122.7** | 73% @ 109.6 | **+22** |
| 0.5 | 76% @ 113.2 | 57% @ 97.5 | +19 |

B fixed-T at alpha=0.6 (uniform is its ONLY option): 77% @ 116.5 ≈ C-uniform (73%) —
the gap comes from *selectivity*, not the representation. 95% binomial CI ≈ ±5–9 pts.

**Headline reads (n=100, single checkpoint, only the *where* of the speedup differs):**
1. ~~Free lunch~~ **FINAL (protocol-crossed, n=300 × 3 seed bases, C-n8, July 12):
   base 94.0 @ 143.8; selective α.6 90.3 @ 114.7 (realized 1.25×); uniform α.6
   88.7 @ 103.9 (1.38×). The single-cell "equal success" reading does not survive
   pooling: the honest claim is a favorable TRADE — 20–28% execution-time
   reduction for ~4–5 pts of success — adjustable post-hoc, with selectivity's
   protection growing at aggressive α (+21 at α=0.4, §frontier).**
2. Aggressive blind speedup collapses monotonically (90→85→73→57); selective stays
   ≥76% everywhere. The duration head is what tells the policy which motions are safe
   to rush — inexpressible for fixed-time or waypoint heads.
3. Honest nuance: at very aggressive throughput targets (~110 steps), *mild uniform*
   (alpha=0.7: 85% @ 110.4) beats *harsh selective* (alpha=0.5: 76% @ 113.2) —
   concentrating the entire speed budget on transports eventually breaks the
   transports themselves. Selectivity buys success at moderate speed, not magic.
   (The alpha=0.7 selective point (85%) sits oddly below both neighbors (94, 95) —
   treat as sampling wobble, CIs overlap.)

Fresh calibration on C-final: overall MAE 6.78→**2.99**, cap bias −9.5→**−2.9** —
the h24 redesign fixed most of the duration smearing at the source. → `fig_pareto.png`
(two-panel dose-response, 100 eps/point, 95% CI).

**⚠️ Suite generalization (n=100/cell, July 6 evening): the α=0.6 free lunch is
libero_object-specific.** Aggressive speedup collapses success on the other suites and
selectivity only partially protects:

| suite | no speedup | selective α.6 | uniform α.6 |
|---|---|---|---|
| object | 93 @ 145.3 | **95 @ 122.7** | 73 @ 109.6 |
| goal | 90 @ 111.1 | 62 @ 87.3 | 53 @ 78.9 |
| spatial | 77 @ 109.1 | 47 @ 88.4 | 47 @ 81.8 |
| long | 65 @ 270.1 | 49 @ 216.6 | 48 @ 174.7 |

Mechanistic reading: "long predicted duration ⇒ transport ⇒ safe to rush" is TRUE on
object (pick-and-place over distance) but false on spatial/goal, where long chunks are
often slow *precision* moves (careful placement, articulated-object motion). Duration
alone identifies rushable motion only when the task family separates transport from
precision by length.

**α=0.8 (mild) suite sweep (n=100): mild speedup is free on object (94 vs 93) and
long (66 vs 65, 13% faster), costs 6–12 pts on spatial/goal, and the selectivity gap
vanishes at mild α everywhere (±4).** Full scoping of the capability: the duration
head gives a *decode-time speed knob* (retime the same path, zero retraining — B can
only do this uniformly); *selectivity* pays specifically in the aggressive regime on
transport-dominated tasks (+22 on object at α=0.6), and mild uniform speedup is the
safe default elsewhere.

**⭐ Capacity moves the aggression frontier, and selectivity re-emerges AT it (July 7–8,
we red-teamed our own flagship and it survived, relocated). n8 config, object, n=100:**

| α | selective @steps | uniform @steps | selectivity gap |
|---|---|---|---|
| 1.0 | 90 @ 143.4 | (anchor) | — |
| 0.6 | 90 @ 114.6 | 88 @ 102.9 | +2 |
| 0.5 | 84 @ 112.5 | 73 @ 105.0 | **+11** |
| 0.4 | 63 @ 116.3 | 42 @ 89.6 | **+21** |

At n6 the selectivity gap peaked at α=0.6 (+22); at n8 it has *moved* to α=0.4–0.5
(+21/+11) while α=0.6 is now nearly free even blind (+2). **The mechanism is unchanged
and confirmed twice over: duration-aware selectivity pays exactly where blind speedup
starts breaking grasps — that frontier just sits at a more aggressive α when the shape
representation is denser.** Revised (stronger) claim: *decode-time retiming is the
robust general capability; representation capacity sets how far you can push it before
grasps break; duration-aware selectivity buys ~15–20 pts of success at the frontier,
wherever capacity places it.* The original +22 was real but capacity-specific; the
capability generalizes, the exact α does not.

## 3c. Self-paced replanning — the policy schedules its own replans (July 6)

The original proposal's "most principled" replanning option — *predict a re-plan
horizon as a third output* — falls out of the duration head for free: execute each
chunk for its own predicted duration T-hat, then replan (decode-only, `replan_frac`
config; replans land at predicted motion events instead of arbitrary fixed offsets).

**n=100 per cell, C-final checkpoint:**
| replan policy | object | calls/ep | libero_10 | calls/ep |
|---|---|---|---|---|
| fixed every 5 (default) | 93% | 31.3 | 61% | 73.8 |
| fixed every 12 | **94%** | 13.4 | **66%** | 32.9 |
| **self-paced (full T-hat)** | 84% | **9.2** | 55% | **23.2** |
| B, full fixed chunk (matched compute) | 91% | 8.3 | 56% | 19.2 |

**Honest verdict:** self-paced replanning *works* (84%/55% at 3.2–3.4× fewer policy
calls than default) but does **not** beat a well-chosen fixed cadence — nas=12 is
better on both suites at moderate compute. A 30-episode pilot had suggested self-paced
won on libero_10 (73%!); n=100 reversed it (55%) — same lesson as the seed variance:
**pilot-scale deltas < ~15 pts are noise.** The chunk-boundary hypothesis for why
full-T-hat execution underperforms: executing exactly *to* the predicted event places
gripper toggles at chunk boundaries where T-hat error clips them; fixed mid-chunk
replans avoid this. The compute story ("fewer calls is nearly free") holds for ALL
representations (B full-chunk: 91% object at 8.3 calls/ep).

**Mechanism CONFIRMED by intervention (n=100, monotone dose-response):** replanning a
fixed *margin* before the predicted event (`replan_margin`) recovers success
monotonically:

| replan at | object | calls/ep | long | calls/ep |
|---|---|---|---|---|
| T-hat (margin 0) | 84 | 9.2 | 55 | 23.2 |
| T-hat − 2 | 90 | 9.6 | 57 | 27.9 |
| **T-hat − 4** | **93** | **11.7** | **62** | 33.0 |
| fixed nas=5 (ref) | 93 | 31.3 | 61 | 73.8 |

→ **Design law: replan *before* the predicted event, never at it.** With a 4-step
margin, duration-scheduled replanning matches the default cadence's success at
**2.7× fewer policy calls** (and ties tuned nas=12 within noise). The negative result
became an understood, reusable rule for any event-terminated action-chunk policy.

## 3e. The duration head as a runtime failure monitor (July 9, NEW capability)

T-hat predicts time-to-next-event; in successful approaches it counts down (the
timeline figure). Measured on 260–400 episodes of existing rollouts (no new
training, no new model machinery):

- **Failures stall ~2×**: fraction of steps with non-decreasing 8-step T-hat trend
  (while T-hat < cap) = 0.416 ± 0.109 in failures vs 0.203 ± 0.096 in successes
  (replicated on both n6 and n8 configs).
- **Post-hoc identification: 86% precision at 52% recall** (thr 0.4); 100%
  precision at 0.5.
- **Matched-horizon early warning (rigor check — first 150 steps only, episodes
  alive at 150): 42% of eventual failures flaggable at 11% false alarms** — the
  full separation partly accumulates late (fumble loops intensify), so early
  detection is moderate, not magic. Honest scoping.
- **Behavioral intervention (A/B, n=200/arm): detection transfers live (fired in
  46/47 failures) and the back-off is benign (fired in ~50% of successes at only
  −2…−4 pts / +10 steps) — but it does NOT convert failures** (object 91 vs 95,
  long 62 vs 65). Flagged episodes retreat, re-approach, and fumble again: LIBERO
  failures are persistent-incompetence, not perturbation traps, and since each
  back-off resamples the chunk, noise-resampling recovery is covered by this
  negative. **Scoped claim: the duration head gives free, live failure DETECTION
  (act on it via early abort at the 100%-precision threshold, or operator/planner
  escalation); kinematic self-recovery is not the payoff.** (Reporting bug that
  masked firing in earlier runs: per-episode logs piped through `tail -1`.)

No existing VLA carries an endogenous execution-progress signal; entropy/OOD
monitors require ensembles or external models. This one is free wherever a
duration head exists.

## 3b. Why time allocation (supporting evidence)
- Event-segmented chunk durations vary strongly (CoV 0.38 over 53k anchors); 47% of chunks end at a gripper event with T = 22±10 steps → duration is a *predictable, semantically meaningful* output.
- The time-allocation variant **trains better** than fixed-T at every budget (event-aligned chunks are more homogeneous).
- Rate-feasibility (the stretch result) is exactly the capability duration-awareness formalizes.

## 3d. Speed-heterogeneous demonstrations — the training-time case (July 6–7, pilots)

Design: `SPEED_AUG_DESIGN.md`. Every training episode presented as the same spatial
path executed s× slower, s ∈ {1, 1.5, 2} by episode; all arms see identical synthetic
demos; eval budgets ×1.75 for all arms (slower learned behavior is legitimate).
20k pilots vs clean 20k twins (object / long, 100 eps):

| arm | clean | speed-aug | Δ |
|---|---|---|---|
| A waypoint | 93 / 51 | 95 / 58 | **+2 / +7** |
| B spline fixed-T | 91 / 52 | 90 / **39** | −1 / **−13** |
| C spline + time-alloc | 93 / 57 | 87 / 53 | −6 / −4 |

**Two mechanisms, one sharpened thesis:**
1. *Waypoint policies absorb pure speed heterogeneity* (honest null, prediction
   revised): per-step deltas across speeds are colinear (same path, scaled magnitude),
   so averaging gives the right direction at ~mean speed, and closed-loop replanning
   re-syncs progress. The naive "fixed-dt breaks under mixed speeds" story is false
   under closed-loop deployment with fair time budgets.
2. *Fixed-time trajectory chunks are the vulnerable representation*: B's chunk is a
   time-defined path prefix, so the same observation maps to different SHAPES across
   speeds — shape supervision blurs, −13 on long-horizon. **C's event-terminated
   chunks isolate speed into the duration channel (implementation: duration ×= s,
   shape targets bit-identical — machine-verified) and hold at −4; C > B by +14 on
   long under heterogeneity.**

**100k confirmation (1 seed each, vs clean 3-seed means): the contrast ATTENUATES
with budget.** Δ(obj/long): A −3.7/−5.3, B −3.7/−6.7, C **−2.0/−1.3**. Given 5× more
steps, even B's speed-blurred targets mostly converge; the ordering survives (C least
degraded at both budgets, B worst on long) but the 100k separation is within
single-seed noise.

→ **Honest final claim: speed heterogeneity is a sample-efficiency tax on fixed-time
trajectory chunks (−13 on long at 20k) that time allocation largely waives (−4 at
20k; −1.3 at 100k, the only arm statistically indistinguishable from its clean twin).**
Waypoint policies absorb speed variation through closed-loop slack at all budgets.
A supporting result for the factorization thesis, not a standalone headline —
promoting the 100k C-vs-B gap (~3.5 pts) would need speed-aug seeds.

**Mechanism confirmed end-to-end (T-hat distribution, decode clamp lifted):** clean C
predicts T-hat mean 20.8 (cap-pinned at 24); speed-aug C predicts mean **27.0, p90 40,
max 48** — shifted ~1.3× (theory: geometric-mean prediction under unresolvable
per-episode speed ambiguity = e^{E[log s]} ≈ 1.44×) and expressing the full synthetic
duration range, while shape channels stayed clean (bit-identical targets, success
−1.3). The speed variation went exactly where the representation sends it: the
duration channel. *(Implementation note: decode clamps T-hat to horizon_max — for
speed-aug training the clamp must be set to the label max (48), not the raw fetch cap
(24), or deployment silently re-speeds the policy; first mechanism measurement was
right-censored by this.)*

## 4. The C-gap investigation → C-final (July 5–6)

C initially trailed B by ~9pts avg (worst on long-horizon: ~49 vs 70). A
hypothesis-driven investigation closed it:

- **Eliminated by direct experiment:** replan-boundary velocity jerk (built the fix —
  velocity-continuous chaining, boundary discontinuity 4.0→0.97, exact math s'(0)=9(c₁−c₀) —
  success unchanged ⇒ sim's OSC absorbs command jerk; kept as the hardware-relevant
  contribution); late grasps (failures grasp *early* then fumble, 9.5 vs 4 grasp cycles).
- **Confirmed with measurements:**
  1. *Duration-mode smearing* — calibration (n=1280): grasp-event chunks **MAE 3.7,
     r=0.84** (the head genuinely learned time-to-grasp → `fig_calibration.png`,
     `fig_timeline.png`: closed-loop T̂ counts down to each grasp); capped transport
     chunks smear 40→30.5 → transports execute ~33% overspeed. Decode-only mode-snap
     fixed object (91→94) but not long-horizon.
  2. *Supervision-density ceiling (the binding constraint)* — cap chunks fit the
     executed window **4.15× worse** than B's fixed-20 chunks (0.051 vs 0.012 RMSE),
     and caps are 60% of long-episode anchors. A target-quality ceiling no decode fixes.
- **The fix (h24):** chunk support capped at 24 steps (≈B's control-point density),
  event search/duration unchanged in spirit. **At matched 20k budget C now beats B:
  object 93 v 91, long-horizon 57 v 52** (spatial 65 v 63, goal 73 v 84).
  → design law for spline action heads: *chunk support must scale with control-point
  budget; duration prediction should not be entangled with fit support.*
- **C-final (100k, h24) training overnight** → full-row battery + Pareto rerun.

## 4c. Measurement-noise anatomy of LIBERO evaluation (July 9)

Same-checkpoint success across evaluation conditions (100 episodes each unless noted):

| checkpoint × suite | lerobot-eval | rollout sb1000 | rollout sb2000 | spread |
|---|---|---|---|---|
| A (s1000) × spatial | 83 | 81 | **74** | **9** |
| C-n8 (s1000) × spatial | 76 | 81 | 79 | 5 |
| C-n8 (s1000) × object | 94 | 93 | 95* | 2 |
| C-n8 (s1000) × long | 63 | 65* | — | 2 |

*duplicate-run cells from other batteries; also two literal duplicate lerobot-eval
runs of Bn8-s1001 spatial gave 71 vs 76 (±5 on identical everything).

**Findings:** (1) same-weights spatial measurements spread up to 9 pts — 2–4× the
binomial SE at n=100 — i.e., the evaluation protocol (initial-state seed set,
harness) contributes *systematic* variance beyond sampling noise, concentrated in
the precision-placement suite; (2) object/long are comparatively tight; (3) any
suite-level claim under ~8–10 pts at n=100 on spatial is unsupportable without
protocol-crossed evaluation. This is how the "spline spatial gap" survived 3
training seeds (§1) — training-seed error bars replicate the protocol's bias, not
the truth. Recommendation adopted for all remaining comparisons: cross seed-sets ×
harness, or use within-protocol paired contrasts only.

## 5. Reproducibility
- Code: `lerobot/policies/smolvla_spline/` (+ factory & env `control_freq` patches), deployed on Misha & Bouchet.
- Train: `--policy.type=smolvla_spline [--policy.predict_duration=true]` on `HuggingFaceVLA/libero`; 100k steps ≈ 4–7h on 1 GPU.
- Eval variants (Hz/stretch/nas) are config-edited checkpoint dirs under `~/scratch/vla_bspline/outputs/hz_variants/` (Misha).
- Offline validation: `libero/validate_spline_head_math.py`, `libero/v2_event_segmentation_study.py`, `libero/horizon_study.py`.

## 6. Open items (July 12)
1. Real-robot validation of the deployment capabilities (stretch, ease-out,
   selective retiming, failure flagging) — the sim OSC absorbs command-level
   kinematics; hardware is where continuity/deceleration claims bite (Xiatao).
2. n8 spatial protocol-crossed at higher n if any suite claim is ever needed.
3. Future-work pocket: shape-level temporal augmentation for spline heads
   (composition pilot showed duration-only aug ≠ regularizer); decode-consistency
   aux loss (coded, gated — no motivating evidence after protocol resolution);
   F1″ decoupled shape-support/duration; CALVIN unification with Quinten.
