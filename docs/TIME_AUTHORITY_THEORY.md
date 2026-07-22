# A predictive theory of time authority
*(synthesis memo, July 20 — unifies every timing result in the project into one
model with three measurable dataset parameters and falsifiable predictions)*

## 1. The object of study

An imitation policy inherits not just the demonstrator's **path** but its
**time map** — how execution time is distributed along that path. Waypoint
heads freeze the inherited time map into the representation (fixed Δt between
waypoints); our head factorizes it (shape = control points; timing = duration
channel + control-point spacing) and therefore exposes the time map to
*post-hoc control*. The question this project answered empirically, and this
memo makes predictive: **when does exercising that control help, and how much?**

## 2. Three dataset parameters

For any demonstration set, three statistics — all computable offline in
minutes, no training — determine the value and the correct use of time
authority:

| symbol | name | definition (operational) | LIBERO | CALVIN | RoboCasa | DROID | ALOHA-static |
|---|---|---|---|---|---|---|---|
| **δ** | dead-time fraction | frac. of steps in sustained near-zero-velocity runs (speed < 0.15×median, ≥2 steps) | ≈0 | high (jitter floor 6×, hesitation-rich)* | 0.16–0.20 (pause events) | 0.095 | 0.015–0.03 |
| **h** | actuator headroom | bound B / p95(per-step delta) | ≫1 (deltas 0.1–0.3, B=1) | >1 (scaled rel-actions) | **≈1** (p99 = 1.0; 38% of frames >0.6B) | — | — |
| **γ** | precision sensitivity | success loss per unit terminal-tracking error (proxy: contact-task fraction; demo deceleration ratio at events, e.g. 0.55× cruise) | high on spatial/goal | moderate | task-dependent (faucet high) | — | — |

*CALVIN's pause-flag reads 0% because hesitation there manifests as low-speed
jitter rather than full stops; the jitter floor (6× LIBERO) is its δ-signature.
A refined δ̂ should integrate "compressible slowness" = time spent below the
speed needed by the task's kinematic optimum, not only full stops.

## 3. The model

Write a demonstrated segment as motion + dead time: T_demo = T_motion + T_dead,
with T_dead = δ·T_demo. Retiming with factor α (execute in α·T of the original
time) has three first-order effects on success:

1. **Dead-time harvest** (helps): executed dead time shrinks by (1−α)·δ·T.
   Beyond saving wall-clock, reproduced hesitation is *behaviorally harmful* —
   dithering visits states off the demonstration manifold. (Measured: 77% of
   CALVIN's retiming gain was behavioral, not clock; the policy that dithers
   less fails less.)
2. **Clipping cost** (hurts): compression multiplies required per-step deltas
   by 1/α on compressed intervals. Where 1/α > h the controller clips,
   distorting the path. Cost rises sharply as h→1: expected distortion ∝
   E[(|d|/α − B)₊], which is negligible for LIBERO (h≫1) and immediate for
   RoboCasa (h≈1).
3. **Precision cost** (hurts): compressing contact-adjacent segments raises
   terminal tracking error; cost ∝ γ × (fraction of compressed time that is
   contact-critical). Gating (chunk-level via T̂, interval-level via the speed
   profile) exists precisely to steer compression away from these segments.

Net effect of a retiming policy π (which intervals get compressed how much):

  ΔSuccess(π) ≈ A·δ_π − B·clip(α, h)_π − C·γ_π

where the subscript π denotes restriction to the intervals π compresses. The
knob design question is: choose π to capture δ while avoiding h and γ terms.

## 4. The three regimes — our benchmarks as the parameter corners

**Regime I — margin-scarce (δ≈0, h≫1): LIBERO.** There is no dead time to
harvest, so every compressed second spends real margin; the only available
strategy is to *minimize* the γ term → fine gating (protect slow = protect
contact) is best, and all retiming is a mild trade. Measured: interval −2.0 <
chunk −3.7 < uniform −5.3 at ~1.3×; protect-slow threshold θ is load-bearing
(θ=0.5 breaks it: −11). Slow-down is free (nothing to amplify).

**Regime II — dead-time-rich (δ high, h>1): CALVIN, and human teleop
generally.** The δ term dominates and is *everywhere*, including mid-speed
intervals → coarse compression captures more of it than cautious gating:
uniform (+0.38, t=12.1) > interval (+0.27) > chunk (+0.16); the θ
dose-response peaks at θ=0.5 (compress all but the extreme-slow tail — the
tail is where γ lives) and regresses by θ=0.3. Slow-down *hurts* (it
manufactures more dead time — pre-registered and confirmed, 1.22 vs 1.42).
Retiming is a Pareto win: success AND speed.

**Regime III — bound-saturated (h≈1): RoboCasa.** ⚠️ *Partially falsified — see
below.* The clipping term should gate compression: with h≈1, any α<1 pushes
deltas over B and either clips or (with feasibility on) re-expands to base, so
the predicted signature is **retiming ≈ neutral**. This half holds: at n=100,
C+retime ≈ C base on all three live tasks (kettle 38→37, toaster 21→21, faucet
6→7). But the sharper pre-registered prediction — that feasibility would
*rescue a retiming penalty* — **failed, because the premise was a noise
artifact**: the apparent penalty (n=50 base faucet 14 → retimed 6) vanished at
n=100 (base faucet is 6). There was no penalty to rescue; feasibility
correctly did nothing. **Separately and unexplained by this model, C base
trails A on RoboCasa (mean 21.7 vs 27.3 at n=100)** — a representation/budget
gap the (δ, h, γ) decomposition does not address, since it concerns un-retimed
success. Two candidate causes were tested and **both refuted**: undertraining
(C plateaued by 50k — toaster 20≈21, faucet 8≈6 at 50k vs 100k) and
cap-domination (RoboCasa cap-frac 0.77 ≈ CALVIN 0.70, where C wins). Mechanism
remains open; the one clear structural difference is 4× fewer gripper-event
chunk boundaries (toggle-frac 0.07 vs CALVIN 0.30), thinning C's event-alignment
advantage — plausible, unproven. Reported as a genuine, not-yet-explained
limitation, not spun. Regime III's *hardware warning* stands regardless: real velocity/
torque bounds put deployment in the h≈1 regime, so feasibility dilation must be
on by default — but as a safety layer, not a demonstrated success-booster.

## 5. What this predicts (falsifiable, some pre-registered)

1. ~~**In-flight (pre-registered):** RoboCasa faucet under uni06+feasibility
   recovers toward base (plain uni06: 6 vs base 14).~~ **RESOLVED — FALSIFIED
   (July 20).** The premise was noise: at n=100 base faucet is 6 (not 14), so
   there was no penalty to recover; feasibility left it unchanged (7). Lesson
   re-applied: never build a prediction on a single n=50 cell. The honest
   surviving statement is the weaker one — retiming is *neutral* at h≈1, not
   penalized-then-rescued.
2. **Any new dataset** can be triaged before training: compute (δ, h, γ-proxy);
   the best knob is fine-gated mild retiming if δ≈0, coarse retiming (+
   feasibility if h≲1.5) if δ high. DROID-style data (δ=0.095) sits in Regime
   II → predicted uniform-retiming win; expert ALOHA (δ≈0.02) sits near Regime
   I → predicted mild trade. This is testable with two 100k trainings on any
   of those datasets.
3. **Hardware cloth folding** (novice teleop; quasi-static so γ low, h set by
   the arm's velocity limits): Regime II with a hard h — predicted outcome is
   a large realized speedup at unchanged fold quality **provided** feasibility
   dilation is enabled; without it, prediction flips to failures at cloth-grasp
   moments. Running both arms is a one-day hardware experiment that tests the
   theory, not just the feature.
4. **Language-time interacts with regime**: adverb-commanded slowdown
   ("carefully") should *help* only in Regime I/III contexts (precision, near
   bounds) and hurt in Regime II — i.e., the right speed *direction* is
   context-dependent and the language interface lets a planner choose it at
   runtime. (Untested; natural real-world demo.)
5. **A waypoint head retrained on dead-time-filtered data** (offline δ
   removal) should recover part of CALVIN's retiming gain (the δ term) but
   none of the runtime adjustability (no α knob, no per-scene adaptation) —
   the ablation that separates "data cleaning" from "time authority."
   (Buildable in sim now; predicted partial recovery.)

## 6. Why the head design follows from the theory

- The **duration channel** makes T̂ (and thus dead-time-aware scheduling)
  available at decode; fixed-Δt heads have no handle for any term in §3.
- **Event-aligned chunks** put contact (γ) at chunk *boundaries* (u=1), which
  is what makes gating implementable — γ-critical time is localized, not
  smeared mid-chunk.
- **Control-point spacing** carries the demo speed profile at full fitting
  resolution (the v3 refutation), so interval gating needs no extra learned
  machinery — the profile is already in the representation.
- **Feasibility stretch** is the h-term controller; **ease-out** is a targeted
  γ-term controller; **adverb conditioning** is a language interface to α.
  Every knob in the stack maps to exactly one term of §3's decomposition.

## 7. One-line summary

*Imitation inherits a time map; success under retiming is a three-term budget
(dead-time harvested − clipping incurred − precision spent), the three terms
are measurable offline as (δ, h, γ), our three benchmarks are the three
corners of that parameter space, and every decode knob in this head is the
controller for exactly one term.*
