# ICRA 2026 paper plan — composition, ablations, sim scenarios, real-world, and the frequency experiment
*Written 21 July 2026, responding to Xiatao's directive. Companion to `TEAM_UPDATE.md`
(results), `TIME_AUTHORITY_THEORY.md` (the (δ,h,γ) theory), `EVAL_DESIGN.md`
(statistics), `REAL_WORLD_PLAN.md`, `ROBOCASA_PLAN.md`, `POSITIONING_BSP.md`.*

---

## 0. Strategic context — the space is crowding, positioning speed is the top risk

Continuous / retimeable action representations for policies became a **crowded 2026
topic**. Neighbours to be aware of and cite:
- **BSP** (B-spline Policy, arXiv 2607.09648) — B-splines to accelerate manipulation;
  real-world 1×/2×/4×. Their verified failure: global 4× → **0/20** on the contact
  task ("pushes past the controller's tracking limits"). This is *our solved problem*.
- **Spline Policy** (arXiv 2606.07386) — splines for policies, framed capability-not-
  SOTA (same rhetorical strategy as ours).
- **BEAST** (B-spline action tokenizer), **OmniSAT** (B-spline + VQ-VAE action tokens,
  ICLR 2026), **PACE** (phase-aware chunk execution), **SEAM** (smooth execution of
  action-chunked motion for VLAs), **"VLA Knows Its Limits: Adaptive Execution
  Horizons,"** **Neural Implicit Action Fields**.

**Consequence for us.** The paper cannot be sold as "splines for VLAs" (taken) or
"faster execution" (BSP owns it). It must be sold as **the learned time-allocation
layer and the control surface it exposes** — non-uniform/feasible retiming, Hz
adaptation, event-scheduled replanning, language-commanded speed, endogenous failure
detection — with the **CALVIN Pareto win as proof the layer is worth having** and the
**(δ,h,γ) theory as the "how to rescale without breaking" that BSP lacks.**
**Action: get a strong arXiv v1 out early to stake priority.**

---

## 1. Central claim and abstract skeleton (framing lock)

**One-sentence contribution (name the artifact — decide the name this week):**
> We give VLA policies a **learned time-allocation head** that factorizes trajectory
> *shape* from *timing*, turning execution speed, control frequency, replan schedule,
> and language-commanded pace into **decode-time control knobs on a single trained
> checkpoint** — capabilities a fixed-time waypoint head cannot express.

**Abstract, 5-move template (award-paper pattern):**
1. *Tension:* VLA action heads emit fixed-rate waypoints, freezing the demonstrator's
   time map into the weights; execution speed/rate/replanning are baked in at training.
2. *Contribution (named):* a B-spline control-point head with a learned duration
   channel and event-aligned chunks.
3. *Property unlocked:* the time map stays with the executor — speed, Hz, replan
   cadence, and language pace become post-hoc knobs on one checkpoint.
4. *Evidence (capability-at-parity, not victory):* success parity on scripted data
   (LIBERO), a **Pareto win on realistic human-teleop data** (CALVIN: deeper chains
   at 1.42× realized speed, first-task tied), and a **predictive theory of when**
   retiming helps (the (δ,h,γ) headroom law) validated across three simulators.
5. *Breadth/real-world:* hardware demonstration of commanded-speed + Hz adaptation.

**Say / don't-say (already locked into `TEAM_UPDATE.md §0`):** we do NOT claim a
universal raw-success win; we DO claim competitive success + a control surface
waypoint heads cannot express. CALVIN is the flagship where the capability also wins.

---

## 2. Figure plan (build in this order)

1. **Fig. 1 teaser — "one checkpoint, a control surface."** Same policy, same scene,
   rows = {language pace / decode speed knob, deploy Hz, replan schedule}; each row is
   the *same task executed differently*. Center: fitted spline held fixed while the
   time-map warps (transport compressed, contact phase protected). Punchline caption:
   *"shape is trained once; timing is a decode-time choice."* (Alt: shape-vs-timing
   schematic + the CALVIN Pareto dot.)
2. **CALVIN success-vs-wall-clock Pareto** (main body): our retimed point up-and-left
   of waypoint-A and BSP-style uniform. The flagship in one glance.
3. **Data-regime inversion** (the novel *scientific* finding): granularity ladder,
   LIBERO protect-slow vs CALVIN compress-all, with beta-posterior violins.
4. **Time-allocation ablation** (thesis-critical): B fixed-time vs C, isolating the
   duration head — including the **RoboCasa B-vs-C row now training**.
5. **Capability panels** (one multi-part figure): commanded-speed sweep;
   Hz-adaptation (naive-fails vs feasibility-stretch); language-pace probe (t≈−8 vs
   flat control); T̂ countdown / failure-detection timeline.
6. **Method/system diagram** (after the teaser, never before it).
7. **Real-world filmstrips + project video** (start the capture rig now).

Assets already banked: countdown timeline, calibration, dose-response Pareto,
granularity ladder, realized-speed head-to-head. Figure economy is in good shape.

---

## 3. Ablation study design (what actually showcases the method)

Ablations are grouped by the claim they defend. Thesis-critical ones go in the body;
sweeps go to the appendix.

**A. The time-allocation layer is worth having (thesis-critical, main body).**
- **A vs B vs C** at matched budget on all 3 benchmarks. A = waypoint; B = spline
  fixed-time (representation only); C = spline + learned time allocation. This is the
  single cleanest decomposition: A→B isolates the *representation* cost; B→C isolates
  the *time-allocation* cost. **RoboCasa B is the missing cell, training now** — it
  tells us whether RoboCasa's C-trails-A is a representation problem (B also trails)
  or a time-allocation problem (B ties A, C trails).
- **Uniform-time decode on C vs learned duration** (does the learned T̂ matter, or is
  a fixed clock enough?) — already have the calibration evidence; formalize as ablation.

**B. Retiming granularity (the data-regime law, main body).**
- uniform vs interval-selective (protect-slow) vs chunk-selective, at matched realized
  speed, on LIBERO (scripted) and CALVIN (teleop). The **sign inversion** is the
  finding (LIBERO: fine gating best; CALVIN: coarse best). Beta-posterior CIs.

**C. Feasibility / headroom (the "rescale without breaking" claim).**
- feasibility-stretch ON vs OFF under compression, as a function of measured headroom
  h. Directly answers BSP's 4×→0/20 failure. Best shown where h≈1 (RoboCasa) and at
  low deploy frequency (LIBERO half-rate).

**D. Language-commanded speed (novel capability).**
- the counterfactual eval of §5 below (trained adverbs, unseen synonyms, control
  checkpoint). Ablate: adverb-augmented training vs standard training; ClangAdv vs Cn8.

**E. Event-boundary construction (candidate RoboCasa fix — run only after the
diagnostic, §4).** gripper-toggle+pause boundaries vs fixed-window vs denser
boundaries; and control-point density n∈{6,8,10}. Single targeted run, not a sweep.

**Negatives we will report (credibility, feeds the mandatory Limitations section):**
knobs don't compose (one time budget); self-paced replanning ties but does not beat a
tuned fixed cadence; failure-monitor gives detection, not self-recovery; the spatial
"gap" dissolved under protocol-crossing; temporal augmentation cannot regularize this
head. Reporting these is a differentiator vs the 20-rollout-no-test norm.

---

## 4. RoboCasa diagnostic (Xiatao's direct-measurement request) — spec

Goal: directly measure *why* C trails A on RoboCasa, on **toaster + faucet**, A vs B
vs C. Offline where possible (no rollouts needed for open-loop reconstruction).

Metrics, per arm, on held-out demo windows:
1. **Open-loop action reconstruction error** — predict the demonstrated chunk from the
   demo observation; RMSE of predicted vs demonstrated actions. (A predicts waypoints;
   B/C predict control points → decode to steps → compare in action space.)
2. **Rotation error** — same, restricted to the orientation channels (RoboCasa has
   6-DoF pose; rotation is where contact tasks are precise).
3. **Gripper-transition timing** — predicted vs demonstrated gripper-toggle step;
   error in steps around each toggle. Tests whether C mis-times grasps.
4. **Error around contact / gripper events** — reconstruction error in a ±k-step
   window around each gripper event vs away from events. Tests the leading hypothesis
   (RoboCasa has 4× fewer gripper-event boundaries → thinner event-alignment signal →
   worse contact-local supervision for C).

Deliverable: a table (A/B/C × 4 metrics × {toaster, faucet}) that either confirms the
event-boundary hypothesis (C worse specifically at/around events, and B better than C
there) or points elsewhere. **Only after this** do we run **one** targeted E-ablation
(event-boundary construction *or* control-point density) — not a broad search.

---

## 5. Language-commanded-speed counterfactual (Xiatao's requested eval) — READY

Scripts written and staged (`lang_counterfactual.py`, `lang_cf_analyze.py`,
`cluster/lang_cf.sbatch`); launches on ClangAdv + Cn8 control the moment the cluster
is reachable. Design (matches the ask exactly):
- Same seed ⇒ identical initial condition and matched flow-noise stream up to the
  first behavioural divergence; **swap only the adverb prefix** on the instruction.
- Conditions: plain, "quickly", "slowly and carefully" (trained), **"rapidly",
  "gently, slowly" (unseen synonyms)** → learned speed concept vs token shortcut.
- Per matched episode: **realized speed** (mean ‖pose-delta‖), **steps-to-done**,
  **success**, **T̂ mean** (expected flat), **endpoint + path-length deviation vs
  plain**. Paired t across conditions. Control checkpoint must be flat.
- Generalization: task_ids the adverb labels didn't emphasize (unseen scenes).

Existing offline evidence to be upgraded by this rollout: "quickly" +8.4% / "slowly
and carefully" −10.1% executed speed, t=7.98 (n=128 paired), control flat, standard
prompts preserved 87–88%.

---

## 6. The frequency experiment (0.5× / 1× / 2×, A vs B vs C) — honest design

Xiatao: "add one clean train-frequency vs deployment-frequency experiment … robustness
to different system frequencies is the central motivation."

**Critical caveat we must respect (from our own rigor log).** On robosuite/OSC sim,
changing the controller `control_freq` for a *delta-action* policy is largely
invariant — the OSC controller reaches each commanded delta regardless of rate — so a
naive `control_freq` sweep does NOT expose the zero-order-hold jerk that Hz-decoupling
actually targets. An earlier "naive waypoint = 0% at half rate" number was a
**weak-checkpoint artifact**; a strong 100k waypoint gets ~45%/77% at f10/f40, and a
path-resampling ("interp") waypoint baseline is competitive. **We will not present a
dramatic sim "waypoint cliff."**

**The defensible sim experiment (supporting result):**
- Fix training rate. At deployment, execute the chunk at **0.5× / 1× / 2×** the
  training execution rate by resampling the continuous curve (spline: native; waypoint:
  requires resampling), with **episode-length budget scaled ∝ 1/rate** (fair walltime).
- Arms: **A-naive** (hold/repeat waypoints), **A-interp** (strong path-resampling
  baseline — the honest competitor), **B** (spline fixed-time), **C + feasibility-
  stretch**. All at matched 100k budget.
- Metrics: success, realized speed, and **commanded/executed jerk** (the mechanism
  variable). Expectation, stated up front: graceful degradation for B/C and A-interp;
  the spline's edge in sim is **smoothness (jerk), not success** — analytic C¹/C²
  vs waypoint impulse-train acceleration.
- Benchmarks: LIBERO (clean) + CALVIN (teleop). One figure: success + jerk vs rate.

**The decisive experiment is hardware (see §7).** On an impedance/admittance-controlled
arm the zero-order-hold jerk is real, so the frequency-robustness claim that sim papers
over becomes measurable: deploy the same checkpoint at half/native/double rate, show
naive waypoint tracking degrades / vibrates while the spline (with feasibility-stretch)
holds. This is the experiment that earns the "central motivation" framing.

---

## 7. Real-world experiment setup

**Bar for a top manipulation paper:** ~3–5 tasks × ~20 rollouts on one robot, with
video. For a *capability* paper the bar shifts: show the capabilities behave **on
hardware where sim cannot validate them** (terminal deceleration/ease-out, C¹ chaining,
feasibility-stretch under real actuator limits, jerk under impedance control). Framing:
*sim proves it's free; hardware proves it bites.*

**Task selection principle** (from `REAL_WORLD_PLAN.md`): pick tasks with a
**compressible dead-time / transport phase** (where retiming pays) **and** a
**contact/precision phase** (where protect-slow + ease-out bite). Candidates:
- **P1 cloth folding** (novice-teleop, quasi-static: γ low, h set by arm velocity
  limits; Regime II with a hard h — predicted large realized speedup at unchanged fold
  quality *iff* feasibility-stretch on).
- **P2 kitting / pick-place with a fast transport + precise insertion.**
- **P3 a contact task analogous to BSP's Speed Stacking** — head-to-head on the exact
  failure they report (4×→0/20): show our per-interval + feasibility mechanism keeps it
  alive where global rescale kills it. **Highest-value single demo.**

**What to record (capability-first, not success-first):**
- commanded-speed sweep 1.0/1.2/1.4×: completion time + tracking RMSE + **measured
  jerk** + success (STEP-analyzed for the ≤20-rollout regime — statistically legit
  small-n, a differentiator vs BSP's untested cells);
- Hz adaptation half/native/double: naive fails vs stretch holds;
- language pace "quickly/slowly" → measured EEF speed change;
- event-scheduled replanning: fewer policy calls at matched success;
- overlay of executed vs commanded velocity profile (proves the decode-time time-warp
  survives the real controller).

**De-risking:** a strong 3-benchmark sim story with the rigorous protocol can carry the
paper, but expect a reviewer to demand hardware. A *small* real demo of even one
capability (commanded speed, or the Speed-Stacking head-to-head) massively de-risks
acceptance — schedule it as the top hardware task. Build the capture rig + filmstrip
template now, even if data lands late.

---

## 8. Sim scenarios required (mapping ablations → what must be run)

| ablation / claim | benchmark(s) | status |
|---|---|---|
| A/B/C representation vs time-allocation | LIBERO ✓, CALVIN ✓, **RoboCasa B training now** | B row pending |
| RoboCasa 5-task aggregate at consistent n | RoboCasa (micro/coffee n=100 top-ups running) | in flight |
| RoboCasa contact/gripper-event diagnostic | RoboCasa demos (toaster, faucet) | spec'd, §4 |
| retiming granularity + data-regime inversion | LIBERO ✓, CALVIN ✓ | done |
| feasibility / headroom (h-sweep) | RoboCasa (h≈1) + LIBERO half-rate | partial; formalize |
| language-commanded speed counterfactual | LIBERO (ClangAdv + control) | scripts ready, §5 |
| frequency 0.5/1/2× (A-naive/A-interp/B/C+stretch) | LIBERO + CALVIN | design'd, §6 |
| slow-down (mirror knob) | LIBERO (free), CALVIN (hurts, pre-reg ✓) | done |

Adopt **BSP's completion-time metric and RoboCasa task set** everywhere for direct
cross-paper comparison.

---

## 9. Prioritized actions

**Now / this week:**
1. Land RoboCasa **B** (training) → fill the A/B/C decomposition; then B on all 5 tasks.
2. Run the **RoboCasa contact/gripper-event diagnostic** (§4) → identify the failure
   mode before any ablation.
3. Launch the **language counterfactual** (§5) the moment the cluster is reachable.
4. **Name the artifact**; lock the one-sentence contribution and abstract.
5. Draft the **BSP / concurrent-work positioning paragraph** (load-bearing; §0).

**Next:**
6. Frequency sim experiment (§6) + design the hardware version (§7).
7. Build Fig. 1 teaser + CALVIN Pareto + data-regime-inversion figures.
8. One targeted event-boundary/density ablation (§3E), only after the diagnostic.
9. Scope and schedule the first hardware capability demo (Speed-Stacking head-to-head
   or commanded-speed) + project video.

**Deadline logic:** priority/positioning speed is the top risk (§0) → target an early
arXiv v1 on the sim story + theory to stake the wedge, hardware section following.
