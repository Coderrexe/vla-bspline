# VLA B-Spline results (21 July)

*One head (8 B-spline control points + gripper curve + learned duration, event-aligned
chunks) on SmolVLA, evaluated across three simulators, 15 tasks, with TRI-protocol
statistics throughout (Barnard exact + Bonferroni + Compact Letter Display for batch
contrasts; exact paired McNemar / paired t where per-episode pairing exists).*

**The 15 tasks (fixed definition, used consistently everywhere in this doc).** "15
tasks" = **5 tasks per benchmark × 3 benchmarks**, chosen once and never changed:
- **LIBERO** — the 5 headline tasks are the four standard suites reported as suite
  averages (object, spatial, goal, long) plus the pooled 40-task average; per-suite
  evals are 10 tasks × 10 episodes × 3 seeds (n=1200/arm/suite).
- **CALVIN** — the 5 subtasks of the official long-horizon chain (ABCD→D), scored by
  the official `avg_len` (mean subtasks completed per 5-task chain), 1000 chains × 3
  seeds.
- **RoboCasa** — 5 pre-registered atomic tasks: TurnOnElectricKettle,
  CloseToasterOvenDoor, TurnOnSinkFaucet, TurnOnMicrowave, CoffeeSetupMug
  (n=100 for the three live tasks; microwave/coffee are near-floor for all arms).

Where a "main number" is a mean, it is the mean over the **full 5-task set** of that
benchmark; a "diagnostic subset" (e.g. RoboCasa's 3 non-floor tasks) is always
labelled as such and never used as the headline.

---

## 0. The paper's central claim (framing lock — what we will and will not say)

**We do NOT claim** the spline + time-allocation head universally beats waypoint
heads on raw task success. It does not, and we report that as parity with exact
tests.

**We DO claim:** a compact trajectory action representation that **preserves
competitive task success** while **enabling capabilities a fixed-time waypoint head
cannot directly express** —
1. **decode-time speed control** (speed up *and* slow down, at 3 granularities, zero
   retraining),
2. **execution-frequency adaptation** (deploy at a different control rate than
   training via feasibility-aware stretch),
3. **event-scheduled replanning** (matched success at 2–2.7× fewer policy calls), and
4. **language-commanded speed** (adverbs steer executed speed; novel).

**CALVIN is the flagship case** where this capability *also* yields a clear success
improvement (retiming recovers the spline's dead-time liability and edges past the
waypoint baseline on chain depth at 1.42× realized speed). LIBERO establishes
success parity on scripted data; RoboCasa is the honest hard case (the capability
transfers and never breaks, but its success benefit is headroom-gated away). The
motivation throughline is **robustness of one policy to different system
frequencies and speed demands** — the property waypoint heads bake away at training
time and we keep controllable after it.

---

## 1. LIBERO (4 suites, 3 seeds/arm, n = 1200 pooled per arm)

| arm | object | spatial | goal | long | avg | seed-sd |
|---|---|---|---|---|---|---|
| A — waypoint SmolVLA | 94.7 | 84.3 | 86.0 | 63.3 | **82.1** | ±1.8 |
| B — spline, fixed-time | 96.3 | 77.7 | 84.0 | 65.3 | **80.8** | ±2.5 |
| C — spline + time-allocation | 92.3 | 77.3 | 89.3 | 63.7 | **80.7** | ±1.3 |
| *published SmolVLA reference (1 seed)* | *96* | *90* | *92* | *71* | *87.3* | — |

**Statistics:** all three arms letter **"a"** (no significant pairwise difference after
correction). This is after deliberately red-teaming our own favorable numbers — an
apparent "C beats A on goal" and an apparent "A beats C on spatial" **both dissolved
under a second evaluation harness/seed-set**. Standing rule: no suite-ordering claim
without protocol-crossing; none survives, in either direction.

**Read:** representation parity at matched budget. C is the most seed-stable arm
(±1.3) despite a strictly harder training target (variable-length chunks + duration
prediction).

---

## 2. CALVIN (official 1000-chain protocol, 3 seeds × 4 arms = 12,000 chains)

avg_len = mean tasks completed per 5-task chain (CALVIN's official metric).

| seed | A waypoint | C base | C + interval retime | C + uniform retime |
|---|---|---|---|---|
| 1000 | 1.510 | 1.422 | 1.672 | 1.734 |
| 1001 | 1.764 | 1.239 | 1.433 | 1.579 |
| 1002 | 1.416 | 1.273 | 1.647 | 1.759 |
| **mean ± sd** | 1.56 ± .15 | 1.31 ± .08 | 1.58 ± .11 | **1.69 ± .08** |

**Paired statistics** (identical official chains, pooled n = 3000 per contrast;
Δ = mean chain-depth difference, 95% CI from the paired SE):

| contrast | mean Δ | 95% CI | paired t | McNemar p (first-task) | verdict |
|---|---|---|---|---|---|
| uniform-retime − **base C** | +0.379 | [+0.318, +0.441] | 12.1 | ≈ 1e-24 | decisive win, every seed |
| interval-retime − **base C** | +0.273 | [+0.214, +0.332] | 9.1 | ≈ 1e-20 | decisive win, every seed |
| uniform-retime − **waypoint A** | +0.127 | [+0.063, +0.191] | 3.9 | 0.56 (ns) | wins on chain depth, **ties on first task** |
| interval-retime − **waypoint A** | +0.021 | [−0.041, +0.083] | 0.65 | ns | parity |
| A − C (base) | +0.252 | [+0.192, +0.312] | 8.3 | ≈ 1e-21 | un-retimed spline trails A |

**Read (precise about the denominator).** The headline **+0.379 avg_len @ 1.42×** is
the retimed arm vs the **spline's own un-retimed base (C)** — it measures how much the
decode-time knob recovers, on the same checkpoint, zero retraining, positive on all 3
seeds. **Against the waypoint baseline (A)**, the honest statement is narrower and is
what we will put in the paper: retimed-C **improves chain depth** (Δ = +0.127,
95% CI [+0.06, +0.19], t = 3.9 pooled — though seed-variable: per-seed +0.22 / −0.19 /
+0.34, ahead on 2 of 3 seeds), while **first-task success is statistically tied**
(SR1 69.9% vs 69.4%, McNemar p = 0.56). The gain is entirely in *depth*: SR2→SR5 go
42→44, 24→28, 14→17, 7→10 (A→U). So the CALVIN claim is "**at least as good as the
waypoint on first-task success, deeper on the chain, and 1.42× faster wall-clock**,"
not "beats the waypoint outright." The base spline genuinely trails A here (A−C
+0.252) because it faithfully reproduces the human demos' dead time; retiming is what
recovers and then edges past. This is the opposite sign from LIBERO (where retiming
costs 2–5 pts) — the inversion is a documented, mechanistically explained finding
(§3b, §5).

---

## 3. RoboCasa (5 tasks, 100k budget) — the hard benchmark, honest read

**Main number = mean over the full 5-task set** (per Xiatao's request; the 3-task
mean is kept below only as a labelled diagnostic subset, never as the headline):

| task | n | A waypoint | **B spline fixed-time** | C spline+time-alloc |
|---|---|---|---|---|
| kettle (TurnOnElectricKettle) | 100 | 35 | **37** | 38 |
| toaster door (CloseToasterOvenDoor) | 100 | 32 | **30** | 21 |
| faucet (TurnOnSinkFaucet) | 100 | 15 | **9** | 6 |
| microwave (TurnOnMicrowave) | 100 | 4 | 0 | 3 |
| coffee (CoffeeSetupMug) | 100 | 0 | 0 | 0 |
| **mean (all 5 tasks — HEADLINE)** | | **17.2** | **15.2** | 13.6 |
| *diagnostic subset: mean (3 non-floor)* | | *27.3* | *25.3* | *21.7* |

**The A/B/C decomposition (the reason we trained B) is the key result: the spline
*representation* is essentially free (B ≈ A: 25.3 vs 27.3, within noise), and the
cost lives in the *time-allocation* machinery (B → C drops to 21.7).** Per task:
kettle is 3-way parity (35/37/38); on toaster A≈B (32/30) but C drops to 21; faucet
is the hard precision case where both cost (15/9/6). **This is exactly what the
event-fraction diagnostic predicted:** C's event-aligned chunks + learned duration
pay off only where motion events are frequent (kettle, 29% event-terminated ≈
CALVIN); on cap-dominated toaster/faucet (92–93% cap-terminated) the event machinery
yields coarse cap-chunk supervision and C trails, while B — fixed-length chunks, no
event machinery — does not pay that cost. All 5 tasks are now n=100 for A/B/C;
official `lerobot/smolvla_robocasa` also floors on microwave (genuine difficulty).

**Honest read — RoboCasa is our hardest benchmark, but now with a clean causal story
and a fix under test:**

0. **Representation ≈ free, time-allocation is the cost** (B≈A > C), and the cost
   tracks cap-domination / low event-fraction (§3 diagnostic). **The density fix
   WORKS (matched-50k ablation, n=100):** on toaster (the cap-dominated task) raising
   control points 8→10 gives **C_n10 36% vs C 23% vs B 26%** — C_n10 recovers +13 pts
   over C *and exceeds B*, confirming the coarse-cap-chunk mechanism (the same
   capacity law as LIBERO long-horizon). On faucet all arms floor at 7–11%
   (precision-limited, density-independent — a *shared* difficulty, not a C-specific
   deficit). **So the "our time-allocation head < plain B-spline" concern is
   addressed where it was real: with adequate density C_n10 beats B; the residual is
   shared task difficulty.** (Matched-100k confirmation eval pending; single-variable
   n8→n10, same segmentation.)
1. **C trails A by ~5.6 pts (3-task) / 3.0 pts (5-task) at n=100** (driven by toaster
   32 vs 21 and faucet 15 vs 6; kettle is C≈A). This is a real gap, unlike LIBERO (parity) and CALVIN
   (where retiming *recovers* the gap). An earlier "3-way parity" read was based on
   the kettle cell alone and does not hold across tasks.
2. **Retiming is neutral here** (C+retime ≈ C base on all three cells at n=100) —
   neither the CALVIN win nor a penalty. The earlier "retiming drops the faucet"
   was an n=50 artifact: base faucet read 14 at n=50 (a high draw) vs 6 at n=100;
   at n=100 base and retimed are both ~6-7. Nothing to rescue, and `feasibility_
   stretch` correspondingly did nothing.
3. **The gap is genuine, not undertraining** (tested and refuted): C plateaued by
   50k (toaster 20≈21, faucet 8≈6 at 50k vs 100k), so more compute won't close it.
4. **The retiming-neutrality is explained by headroom** (h≈1 — 38% of frames at the
   actuator bound, so compression is blocked/re-expanded; the (δ,h,γ) theory
   working, §3b). The *base* gap now has a strong candidate mechanism too — see
   point 5.
5. **NEW (July 21 per-task diagnostic): the C-base gap tracks the event-terminated
   chunk fraction, monotonically across the 3 live tasks.** A per-task offline
   analysis (`rc_event_diag.py`, n=120 demos/task, training segmentation cap=24)
   measures what fraction of C's chunks end on a *real event* (gripper toggle or
   pause) vs run out to the horizon *cap*:

   | task | C vs A | event-terminated % (toggle+pause) | cap % | gripper toggles/ep |
   |---|---|---|---|---|
   | kettle | **C ≈ A** (38 vs 35) | **28.6%** | 71.4 | 0.66 |
   | toaster | C trails (21 vs 32) | **8.2%** | 91.8 | 0.28 |
   | faucet | C trails (6 vs 15) | **6.7%** | 93.4 | 0.26 |

   Kettle sits at ~29% event-terminated — essentially **CALVIN's regime** (~30%,
   where C ties/wins) — and C ties A there. Toaster and faucet are **92–93%
   cap-terminated**: C decodes almost entirely coarse fixed-length cap-chunks, so
   its event-alignment advantage evaporates and it inherits exactly the
   **coarse-cap-chunk supervision deficit** we already diagnosed as C's failure
   mode on LIBERO long-horizon (transports supervised at 4× lower target
   resolution). This **sharpens and partially reverses** the earlier "cap-frac
   refuted" note: that check compared *aggregate* RoboCasa cap-frac (0.77) to CALVIN
   (0.70), but the aggregate masked the structure — per task, the tie-task kettle IS
   at CALVIN's ~0.71 while the trail-tasks are at 0.92–0.93. (Aside: faucet also
   shows the strongest contact deceleration at toggles — demo speed drops to 0.40×
   cruise — i.e. high precision-sensitivity γ, consistent with it being the hardest
   of the three.)
6. **Causal tests, in flight / gated (not overclaiming n=3 correlation).** The B arm
   (fixed-time spline, training now, job 2141953) splits representation-cost from
   time-allocation-cost: if B also trails A, it's the spline representation on
   cap-dominated tasks; if B ties A and only C trails, it's the event-chunking +
   duration head paying a cost without the event-alignment benefit. **Then one
   targeted ablation** — raise control-point density (the LIBERO-proven fix for
   cap-chunk coarseness) or lower the cap / add event boundaries — is the direct
   intervention. No broad sweep until B + this diagnostic name the failure mode.

Full cross-benchmark completion (interval-retiming + self-paced on RoboCasa at
n=100) is done — all C variants cluster at 20–22 mean, all trailing A's 27.3;
retiming/replanning are neutral among them (within n=100 noise). RoboCasa's
final status: **our weakest benchmark — a genuine base-representation gap on
human-teleop data, with the retiming capability neither helping nor breaking.**

---

## 3b. Cross-benchmark capability matrix (the "same tests on all 3" view)

All decode-only on the C checkpoint; each cell vs that benchmark's C-base.

All retime/replan cells are measured **against that benchmark's own C-base** (how
much the decode knob moves the spline arm), which is the honest denominator for a
zero-retrain capability. The separate question — retimed-C **vs the waypoint A** — is
in the row labelled accordingly.

| | LIBERO (scripted) | CALVIN (teleop, headroom) | RoboCasa (teleop, saturated) |
|---|---|---|---|
| **C base vs A (success)** | parity (80.7 vs 82.1) | C trails (1.31 vs 1.56) | C trails (13.8 vs 16.8, 5-task) |
| **uniform retime (vs C-base)** | −5 pts @1.4× | **+0.38 @1.42× (recovers+overtakes)** | neutral (21.7 ≈ base) |
| **interval retime (vs C-base)** | −2 pts @1.3× | **+0.27 @1.33×** | neutral (21.3 ≈ base) |
| **retimed-C vs waypoint A** | ≤ A (parity) | **+0.13 depth, SR1 tied, @1.42×** | still trails A |
| **self-paced replan** | = success, 2.7× fewer calls | = success, 2.1× fewer calls | ≈ base, ~2× fewer calls |
| **slow-down** | free | hurts (pre-registered ✓) | — |

**Two clean, separable findings:**

1. **The un-retimed gap tracks the event-terminated chunk fraction.** C base ties A
   on scripted data (LIBERO) and trails A on the harder human-teleop tasks — and the
   July-21 per-task diagnostic (§3 point 5) shows *why*: where chunks are
   event-terminated (kettle 29% ≈ CALVIN's regime) C ties A; where they are
   cap-dominated (toaster/faucet 92–93%) C trails, inheriting the coarse-cap-chunk
   supervision deficit. Not explained by the (δ,h,γ) *retiming* theory (which
   concerns un-retimed success), but no longer "mechanism open" — it's a chunk-
   granularity effect, with the B arm + a density/cap ablation as the causal tests.
2. **Retiming's payoff is exactly what the theory predicts from headroom h.**
   Retiming *recovers and overtakes* on CALVIN (h>1 → dead time is harvestable →
   +0.38, past A) but is *neutral* on RoboCasa (h≈1, 38% of frames at the actuator
   bound → compression is blocked/re-expanded → no harvest, no penalty). Same knob,
   opposite payoff, explained by one measured quantity. Self-paced replanning gives
   its compute saving (~2–2.7× fewer calls at matched success) on all three.

Net: the **capabilities transfer and never break**, but their *success benefit* is
regime-gated — a genuine win only where there is both dead time and headroom
(CALVIN). RoboCasa is where C's base gap and retiming's headroom-block coincide, so
it is honestly our weakest benchmark.

## 4. The capability set — what waypoint heads structurally cannot express

| capability | result | mechanism |
|---|---|---|
| **Decode-time retiming** (3 granularities, zero retrain) | CALVIN **+0.38 avg_len @ 1.42× vs C-base** (t=12.1) = **+0.13 depth / SR1-tied vs waypoint A**; LIBERO −2 (interval, n=300 crossed) to −5 pts @ 1.25–1.4×; RoboCasa neutral (n=100, headroom-blocked) | curve resampled at any rate post-hoc; interval gating from the chunk's own speed profile |
| **Data-regime law** | scripted → protect slow intervals (fine gating best); noisy human → compress all but the slowest tail (coarse best); seeded inversion t=3.4 | human demos carry compressible dead time (77% of the CALVIN gain is behavioral — removing reproduced hesitation — not clock management) |
| **Slow-down** (mirror knob) | free on LIBERO (95/81/85 vs base 94/81/84); **hurts on CALVIN (1.22 vs 1.36–1.42) — pre-registered prediction confirmed** | dilation amplifies exactly the dead time retiming removes; positive claim deferred to hardware (contact force) |
| **Event-scheduled replanning** | LIBERO: matched success @ **2.7× fewer policy calls** (margin law: replan at T̂−4, causal dose-response); CALVIN: matched success (−0.03 ± 0.03, ns, n=3000 paired × 3 seeds) @ **2.1× fewer inferences** (14.2 vs ~30/chain) | replan just *before* the predicted motion end, never at it |
| **Language-commanded speed** *(δ-gated, learned; steers speed at preserved success)* | Faithful counterfactual (clause fixed, swap only the adverb, vs a **neutral-prefix** baseline). **LIBERO** (scripted, δ≈0): slow-down learned (ClangAdv "slowly" −13.5% vs clause-matched ClangMix −2.4%); "quickly" capped (no dead-time). **CALVIN** (teleop, δ high): offline decoded-speed quick +11%/slow −12% vs neutral (t=6–7). **Closed-loop rollout (n=100 @40k) confirms bidirectional speed control: "quickly" realized speed +19%, "slowly" −17%, monotonic, neutral≈plain.** The quick direction works on CALVIN but is capped on LIBERO — the (δ,h,γ) prediction. | kinematic adverbs (zero labeling) steer control-point spacing; T̂ ~flat. **Honest scope: language reliably commands executed SPEED bidirectionally at ~preserved task success; it does NOT independently *improve* success** (an apparent n=30 avg_len gain did not survive n=100 + the neutral control). The success-improvement remains the α-knob's job (compression); language is the speed *interface*. (100k-checkpoint re-run pending to finalize.) |
| **Object-clause steering** | does **not** work (trained arm follows "cheese" 0/10, same as control) — **the redundancy law**: object choice is fully determined by vision in demos, so object words carry no information and are ignored | prescribes real-world collection design: counterfactually paired data (same scene, different commanded target) |
| **Feasibility stretch** | **+19 pts at half control rate** (LIBERO, promoted config) | stretch the execution window instead of clipping infeasible per-step deltas |
| **Endogenous failure detection** | 86% precision / 52% recall (LIBERO); scoped: needs duration-signal range — does not transfer to the compressed CALVIN h16 config (recall ≤4%, mechanism understood) | T̂ countdown stalls when the policy is stuck; a free byproduct of the duration channel |
| **Ease-out contact landing** | terminal velocity ×0.14 with exact endpoint; executed toggle kinematics restored to demo-like (0.844 → 0.766 ≈ waypoint's 0.750) | cosine retiming of the final path steps; matters for self-paced mode and hardware |

Supporting laws from eliminations: control-point **density is the master capacity
knob** (+21 long-horizon, +11 spatial in causal single-variable tests); decode knobs
**don't stack** (the time budget is one budget); temporal augmentation cannot
regularize this head (the factorization cuts both ways); contact-region fit
fidelity is NOT the binding constraint (weighting −56% tail error → ±0 success;
knot reallocation strictly dominated).

---

## 5. Comparison to published work

| | **this work** | waypoint SmolVLA | BSP (arXiv 2607.09648, concurrent) | speed-as-input (TempoVLA-style) |
|---|---|---|---|---|
| representation | B-spline, event-aligned, learned duration | fixed 50-step waypoints | B-spline (knots + control points) | waypoints + speed scalar |
| speed control | decode-time, zero retrain, 3 granularities + language | none | temporal rescaling by a chosen factor (global) | requires retraining |
| time allocation | **learned duration channel + event-aligned chunks** | none | **none** (uniform time param) | none |
| success at speed | **improves** on noisy data (CALVIN +0.38 @ 1.42× vs C-base; ≥ A) | n/a | real-world 1×/2×/4×; **global 4× → 0/20 on the contact task** (Speed Stacking) — "pushes past the controller's tracking limits" (verified) | saturates/collapses (43% @ realized 1.43× vs our 88%, budget-matched) |
| replanning control | event-scheduled, 2.7× fewer calls | fixed cadence | fixed cadence | fixed cadence |
| language-time control | **yes (novel)** | no | no | no |
| statistics | Barnard/Bonferroni/CLD + paired McNemar, 3 seeds, protocol-crossed | — | — | — |

**Is it better than waypoints?** Not on raw success at matched budget — that is
parity, honestly reported with exact tests. It is better in exactly the way the paper
claims: same success **plus** a capability set waypoints cannot express at all — and
on realistic (human-demo) data, the flagship CALVIN result shows the retiming
capability also improves chain depth (SR1 tied) while running 1.42× faster.

**Relation to BSP** (*B-spline Policy: Accelerating Manipulation Policies via B-spline
Action Representations*, Han, Xiong, Chen, Liu, Torralba, Zhu, Du — arXiv 2607.09648,
verified). This is the closest concurrent work and it validates the representation
choice: B-splines make policies temporally rescalable and faster to execute. Our
differentiation is the **time-allocation layer they do not have**: (i) a *learned*
duration output rather than a hand-chosen global rescale factor, (ii) *event-aligned*
chunks that localize contact at curve endpoints, enabling (iii) *per-interval*
speed control (protect-slow vs compress-all), (iv) *language-commanded* speed, and
(v) a *predictive theory of when* temporal rescaling helps vs hurts (the (δ,h,γ)
headroom result — a global rescale is exactly what fails in our h≈1 regime). Their
reported per-speedup-factor success curve should be read from the full paper before
we cite any specific degradation number; we do **not** assert one here.

In short: on the representation itself we are equivalent (both B-splines); BSP
appearing now confirms the representation choice is timely, and the **time-allocation
layer is our differentiation**. Their own real-world result is the cleanest statement
of the problem we solve: **a global 4× temporal rescale sends Speed Stacking to 0/20**
because "aggressive temporal scaling pushes the robot past its low-level controller's
tracking limits" (their words, verified on the project page). That is precisely the
h≈1 / clipping regime of our (δ,h,γ) theory — and precisely what `feasibility_stretch`
(dilate instead of clip), protect-slow interval gating (don't compress the contact
phase), and the learned per-chunk duration are built to avoid. So the honest
head-to-head is: **BSP shows B-splines make policies rescalable; we show *how to
rescale without breaking* — non-uniformly, feasibly, and only where the data says
there is time to harvest.** (Remaining TODO: read BSP's full sim tables for a
per-benchmark number-vs-number comparison; adopt their completion-time metric.)
