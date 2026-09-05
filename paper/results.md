# Results & Methodology — full context (working document)

*Working source of truth for the method, completed results, active experiments, and
limitations. Numbers include seeds, sample sizes, pairing, and confidence intervals
where available. Last updated: 3 September 2026.*

---

## Current results at a glance

| capability | completed evidence | result |
|---|---|---:|
| Standard VLA success | LIBERO, four suites, three seeds | waypoint 82.1±1.8; fixed-time spline 80.8±2.5; spline+duration 80.7±1.3 |
| Decode-time retiming | CALVIN, 3 seeds × 1,000 chains | **+0.379** average chain length, 95% CI [0.318, 0.441], at **1.42×** realized speed vs unretimed spline |
| Retimed spline vs waypoint | CALVIN, same paired chains | **+0.127** average chain length, 95% CI [0.063, 0.191]; first-task success tied |
| Language-commanded timing | CALVIN, n=600/condition | “quickly”/“slowly” changes decoded action magnitude by approximately +16%/−16%; quick−plain **+0.172** average chain length (`p=.018`, Bonferroni `p=.035`) |
| Decode-time control rate | LIBERO Spatial t5 | **38%→68%** and **34%→72%** under two horizon protocols; matched waypoint **0/50** |
| Executable language clauses | LIBERO-Long t0/t4, three seeds/head | waypoint **+34.3 pp**; spline **+43.7 pp** |
| Spline×language interaction | matched A/C fixed-clock factorial | **+9.33 pp**, hierarchical 95% CI **[+0.67,+18.0]** |
| Clause-order steering | LIBERO-Long t4, exact paired replay | requested first-object flip **10/50** for both heads; zero anti-direction flips (`p=.00195`) |
| Target-scene composite execution | RoboCasa365 | official phases: seed 1000 **4/100→16/100**; independent seed 1001 **3/40→9/40**; Rinse replication **1/20→8/20** (`p=.0197`) |

The three-seed clause effects and seed-1000 fixed-clock RoboCasa comparison are
summarized in [Figure 1](figures/headline_results.pdf); the two-seed stage results
appear in the linked RoboCasa figures below.

Historical LIBERO top-up pools are retained only as exploratory analyses because they
reused initial states. Claim-bearing LIBERO language results use immutable artifacts
and fresh-process exact replay; RoboCasa uses frozen source hashes, target-split seed
grids, complete videos, and independent training-seed replication.

### Executable language programs on LIBERO-Long

The experiment uses exactly 71 demonstrations / 19,378 frames from LIBERO-Long t0
and t4. Original and clause conditions have identical observations, actions, and
episode schedules; only the language labels differ. A frozen two-clause clock tests
whether the policy can execute the learned clauses sequentially. Static compound
prompts test ordinary task competence without the program.

| seed | waypoint A: original→clauses | spline C: original→clauses | C−A interaction |
|---:|---:|---:|---:|
| 1000 | 11/100→42/100 (**+31 pp**) | 13/100→54/100 (**+41 pp**) | **+10 pp** |
| 1001 | 7/100→37/100 (**+30 pp**) | 14/100→56/100 (**+42 pp**) | **+12 pp** |
| 1002 | 3/100→45/100 (**+42 pp**) | 8/100→56/100 (**+48 pp**) | **+6 pp** |
| **mean** | **+34.33 pp** | **+43.67 pp** | **+9.33 pp** |

The interaction is positive on every training seed. Its training-seed t 95% CI is
[+1.74,+16.92] pp and its hierarchical seed/state bootstrap 95% CI is
[+0.67,+18.0] pp. The within-head training-seed CIs are [+17.79,+50.87] pp for A
and [+34.26,+53.07] pp for C. A separate same-L40S sequential C replication gives
+36 pp, supporting robustness to the training allocation without entering the
primary matched-H100 factorial.

Controls isolate the mechanism:

| evaluation interface | waypoint A mean effect | spline C mean effect | interpretation |
|---|---:|---:|---|
| fixed two-clause clock | **+34.33 pp** | **+43.67 pp** | primary matched-head estimand |
| predicted action-event clock | — | **+40.33 pp** | gain persists when the policy's close-to-open gripper event advances the clause |
| static compound prompt | **−9.33 pp** | **−24.0 pp** | fine-tuning alone does not explain the gain |

All 30 evaluation gates passed independent fresh-process replay: 6,000 rollouts with
identical initial/processed observations, prompts, switch times, action traces,
termination steps, and outcomes. Clause-conditioned spline successes also finish
faster descriptively (seed-1000 medians: t0 321→270 steps; t4 292→236), so the
success gain is not purchased with a longer rollout budget.

#### Clause-order steering

A separate exact-replay probe reverses the two executable clauses and scores which
official BDDL goal predicate is completed first. This uses a privileged predicate
transition oracle to advance the program, so it measures language-conditioned order
control rather than a deployable scheduler.

| head | requested first-goal flips, t0 | requested first-goal flips, t4 | anti-direction flips | final conjunction success, normal→reversed |
|---|---:|---:|---:|---:|
| waypoint A | 3/50 | **10/50** (`p=.00195`) | 0 | 49/100→16/100 |
| spline C | 0/50 | **10/50** (`p=.00195`) | 0 | 55/100→21/100 |

For t4, 10 of the 12 states where both programs complete a unique first subgoal flip
in the requested direction for each head (83.3% conditional flip rate). The absence
of anti-direction flips and exact fresh-process replay establish real first-subgoal
steering. The large final-success drop shows that the dataset still encodes a strong
canonical-order bias: language can redirect the first object, but the policy often
fails to retain it and finish the reversed conjunction.

#### Held-out compositional recombination (smoke complete; full test pending)

The next locked test asks whether piecewise language recombines learned subgoals,
rather than merely fitting the original two tasks more easily.

| role | LIBERO-Long task | demonstrations | frames |
|---|---|---:|---:|
| train | t0: soup + tomato → basket | 33 | 9,571 |
| train | t1: cream cheese + butter → basket | 49 | 12,702 |
| held out | t7: soup + cream cheese → basket | 43 | 11,494 |

All four waypoint/spline × whole-caption/piecewise-clause cells start from public
SmolVLM weights with a new action expert; no LIBERO-finetuned checkpoint or t7 frame
is used. Action/state normalization and spline statistics are recomputed from exactly
the 82 training demonstrations / 22,273 frames. The primary evaluation changes t0's
goal conjunction to soup+cream cheese inside the same Scene-2 simulator, so scene,
fixtures, all object initial regions, and basket placement remain fixed. Both language
conditions receive the same completion-predicate queue reset. Normal order is primary;
reversed order and official cross-scene t7 are secondary. A two-state exact-replay
smoke gate precedes the 50-state run, and t0/t1 competence is checked before any
held-out result is interpreted.

All four 20k-step training cells and their held-out smoke evaluations completed
successfully (Slurm build/train/eval jobs 2347690--2347698). Each evaluation was
repeated in a fresh process and all four exact-replay gates passed. The two-state
smoke result is:

| head / language | requested first goal, normal | requested first goal, reversed | full conjunctions (four rollouts) |
|---|---:|---:|---:|
| waypoint / whole caption | 0/2 | 1/2 | 0/4 |
| waypoint / clauses | **2/2** | 0/2 | 0/4 |
| spline / whole caption | 0/2 | 0/2 | 0/4 |
| spline / clauses | **2/2** | 1/2 | 0/4 |

Across heads, clauses select the requested first goal in 4/4 normal-order smoke
rollouts versus 0/4 for whole captions. This is a promising pipeline-level signal,
not yet a generalization estimate: only two states were used, no cell completed the
held-out conjunction, and reversed-order selection is tied at 1/4. The next locked
decision is the pre-specified t0/t1 in-distribution competence gate. Only competent
checkpoints advance to the 50-state held-out evaluation; held-out outcomes are not
used for checkpoint selection.

### RoboCasa365 target-scene execution and language steering

All current spline checkpoints use the corrected production event-target and embedded
normalization contract. Task-level and mixed-granular arms share the same atomic
initialization and 2,029 physical demonstrations / 843,869 frames; only the language
overlay differs.

| target-split evaluation | full success | diagnostic progress |
|---|---:|---|
| task-caption spline, RinseSinkBasin | **7/50** | initial 1/10 plus 6/40 disjoint target scenes; success video verifies faucet actuation and basin sweep |
| task-caption spline, KettleBoiling diagnostic | 0/10 | grasp 6/10; best lift 36.2 cm; stove contact and burner actuation occur but fail to stay in the correct order |
| mixed-granular spline, KettleBoiling | 0/10 | grasp 7/10; best lift 13.0 cm; generic mixed clauses do not improve completion |
| original-label Kettle, scheduled clauses | **2/50 fixed; 0/10 oracle** | matched label-control checkpoint; fixed result combines disjoint 10+40-scene shards |
| official-phase Kettle | **6/50 fixed; 4/10 oracle** | phase supervision triples fixed-clock completion; oracle is diagnostic |
| official-phase Kettle, slower placement | **3/10 fixed; 5/10 oracle** | best diagnostic result; fixed clock remains the deployable number |

Kettle video localized the failure: a strong rollout transports and places the kettle,
moves toward the knob bank, then returns to and disrupts the completed placement.
RoboCasa’s official annotations provide pick→place→burner phases for all 501 Kettle
demonstrations / 228,349 frames. Holding the physical demonstrations, initialization,
and training budget fixed, those phase labels raise fixed-clock success from 2/50 to
6/50 across disjoint seed ranges 1000–1049. The pilot completion-gated scheduler reaches
4/10, and a slower placement decode reaches 5/10 under that diagnostic oracle. A goal-consistent
front-left/front-right relabeling reaches only 1–2/10 and does not beat the official
phase program quantitatively. It does provide a paired causal steering probe:
left/right commands have identical initial observations and common pick-clause action
prefixes on all 10 seeds, then all traces diverge after the language switch. A matched
successful pair places and finishes on the two different requested burners.

[Figure 2](figures/kettle_phase_storyboard.pdf) shows a matched target scene: the
original-caption checkpoint picks up and places the kettle but never completes burner
actuation in 1,500 steps, while the official-phase checkpoint completes the full task
in 516 steps. The two videos share the exact initial-observation hash; their predicate
traces reach grasp at steps 151/140 and stove contact at 203/201, isolating the final
phase transition as the outcome difference.

For Rinse, the corresponding 509-episode / 211,036-frame official phase split raises
fixed-clock target success from 2/50 to **10/50** (`p=.0277`, two-sided Fisher exact),
including an independent 40-scene shard of 2/40→8/40. Its pilot completion-gated
result is 3/10, but it scores 0/10 under an unscheduled compound prompt; the
original-caption checkpoint reaches 3/10 with that environment prompt. Thus phase
labels specifically improve scheduled short-clause execution rather than universally
improving the checkpoint.

The deployable fixed-clock result now has two training seeds:

| training seed | Kettle original→phases | Rinse original→phases | two-task aggregate |
|---:|---:|---:|---:|
| 1000 | 2/50→6/50 (+8 pp) | **2/50→10/50 (+16 pp)** | **4/100→16/100 (+12 pp, 4×; `p=.00808`)** |
| 1001 | 2/20→1/20 (−5 pp) | **1/20→8/20 (+35 pp, 8×; `p=.0197`)** | **3/40→9/40 (+15 pp, 3×)** |

The aggregate gain is positive on both training seeds, with an unweighted mean of
**+13.5 pp**. Pooling only as a descriptive count gives 7/140→25/140 (3.57×), but
the per-seed rows are primary because seed 1000 has more evaluation scenes. The
effect is task-heterogeneous: Rinse repeats strongly, while Kettle does not repeat
on seed 1001. The earlier diagnostic pilot completion-gated comparison remains
**0/20→7/20** and is not mixed into the deployable rows.

Failure-analysis stage visitation explains where the first-seed gain comes from.
These are post-hoc mechanism diagnostics computed from each rollout's recorded
extrema; a stage may be reached out of order, so these counts complement rather than
replace full success.

| task | recorded stage | original labels | official phases | Fisher `p` |
|---|---|---:|---:|---:|
| Kettle | ever grasped | 24/50 (48%) | **39/50 (78%)** | **.00346** |
| Kettle | lifted at least 5 cm | 14/50 (28%) | 18/50 (36%) | .521 |
| Kettle | ever near a burner | 12/50 (24%) | 19/50 (38%) | .194 |
| Kettle | ever stove contact | 13/50 (26%) | 20/50 (40%) | .202 |
| Kettle | any burner ever on | 6/50 (12%) | 14/50 (28%) | .0784 |
| Rinse | water ever on / at least one region washed | 7/50 (14%) | **22/50 (44%)** | **.00175** |
| Rinse | at least two regions washed | 6/50 (12%) | **19/50 (38%)** | **.00495** |

Thus phase labels substantially improve acquisition on Kettle and faucet/wash
execution on Rinse even before the stricter ordered conjunction is satisfied. The
independent seed replicates the Rinse mechanism: water-on / at least one washed
region rises **3/20→12/20** (`p=.00791`) and at least two regions rises
**3/20→10/20** (`p=.0407`). Its Kettle stages are mixed—grasp 14/20→16/20 but
lift/placement stages decline—matching the full-success non-replication. Complete
stage visitation for both seeds is shown in
[seed 1000](figures/robocasa_phase_progress_seed1000.pdf) and
[seed 1001](figures/robocasa_phase_progress_seed1001.pdf); no diagnostic is dropped.

Existing atomic checkpoints also show matched task-level prompt sensitivity:

| intervention | waypoint A | spline C | interpretation |
|---|---:|---:|---|
| OpenDrawer left↔right | 7/50→0/50 (`p=.0156`) | 6/50→0/50 (`p=.0313`) | direction prompt suppresses the original goal |
| dishwasher rack in↔out | 18/50→0/50 (`p=7.6e-6`) | 15/50→0/50 (`p=6.1e-5`) | strong prompt-conditioned task avoidance |

These probes show that spline compression preserves instruction sensitivity relative
to the waypoint head. Because reward still scores the original goal, they do not by
themselves demonstrate completion of the requested counterfactual goal.

Quinten’s complementary counterfactual-object experiment is currently team-reported
preliminary evidence:

| training data | canonical 0.1 slot | counterfactual 0.3 slot | slot gap |
|---|---:|---:|---:|
| original confounded | 86.7% | 6.7% | 80.0 pp |
| 600 decorrelated episodes | 91.3% | 23.5% | 67.8 pp |
| 1,500 decorrelated, 35k | 63.2% | 23.8% | 39.4 pp |
| 1,500 decorrelated, 70k | 76.9% | 35.7% | 41.2 pp |

The disadvantaged slot improves from 6.7% to 23–36%, showing that decorrelated
language data changes target selection. Paper inclusion requires the raw artifacts,
a larger grasp-and-lift evaluation, and a matched waypoint/spline comparison.

### Duration-head mechanism

| test | result | conclusion |
|---|---:|---|
| Held-out event-duration calibration | MAE 1.61 steps; Pearson r=.819; event/cap balanced accuracy=.892 | learned duration predicts action-event timescale well |
| Three-seed clause execution with predicted action-event clock | mean **+40.33 pp** (per seed +44/+40/+37) | the policy's predicted close-to-open gripper event is a usable clause switch |
| Action-event clock versus fixed clock on clause-trained C | 54% vs 58% (seed 1000) and 56% vs 56% (seed 1001 matched run); differences non-significant | the action-event switch is competitive, but does not replace semantic completion detection |

The head comparison uses the same fixed language clock; predicted action-event
scheduling is a separate deployment ablation. Its large clause gain and parity with
the fixed clock show that the policy's gripper curve can trigger a useful clause
switch. The scalar duration channel is validated separately by calibration and
selective-replanning experiments, while RoboCasa still benefits from semantic phase
labels and, diagnostically, completion-gated switching.
## 0. Method in one paragraph

SmolVLA's flow-matching action expert emits **8 B-spline control points + a gripper
curve + a learned duration (T̂) channel** on **event-aligned chunks** (a chunk ends at
the next motion event — gripper toggle or pause — else at a horizon cap), instead of
50 fixed-rate waypoints. This **factorizes trajectory shape (control points) from
timing (duration + control-point spacing)**, so the demonstrator's *time map* stays
**controllable at decode time**. Fitting is one precomputed pinned-LSQ matmul (Cox–de
Boor basis; c₀ pinned to 0, endpoint pinned to the segment end). Zero transformer
surgery. Three arms compared throughout:
- **A** = waypoint SmolVLA (baseline).
- **B** = spline, **fixed-time** chunks, no duration head (≈ the "plain B-spline policy").
- **C** = spline **+ time-allocation** (event-aligned chunks + learned duration) — *ours*.

**Central claim (framing lock).** We do **not** claim a universal raw-success win over
waypoints. We claim a compact trajectory representation that **preserves competitive
success while exposing a decode-time control surface waypoint heads cannot express** —
speed control, execution-frequency adaptation, event-scheduled replanning, and
language-commanded speed — with a **predictive theory of when each knob helps**, and a
**Pareto success win on realistic (human-teleop) data** as the flagship.

---

## 1. LIBERO (scripted data) — success parity at matched budget

| arm | object | spatial | goal | long | avg | seed-sd |
|---|---|---|---|---|---|---|
| A waypoint | 94.7 | 84.3 | 86.0 | 63.3 | **82.1** | ±1.8 |
| B spline (fixed-time) | 96.3 | 77.7 | 84.0 | 65.3 | **80.8** | ±2.5 |
| C spline + time-alloc | 92.3 | 77.3 | 89.3 | 63.7 | **80.7** | ±1.3 |
| *published SmolVLA ref (1 seed)* | *96* | *90* | *92* | *71* | *87.3* | — |

- 3 training seeds, n=1,200/arm across four suites (300/suite), **overlapping CIs → statistical parity** (all
  arms letter "a" under Barnard+Bonferroni). No suite-level ordering survives
  protocol-crossing (an apparent spatial gap dissolved under a second eval harness).
- **The spline + time-allocation representation costs nothing on scripted data**; C is
  the most seed-stable arm despite a strictly harder training target.
- Methodological finding: LIBERO n=100 suite comparisons carry ±5–9 pt protocol/seed
  variance — single-seed suite differences <~6 pts are noise.

### 1b. LIBERO hard-cell mechanism studies (July campaign)

The July hard-cell campaign remains useful, but its top-up pool is not treated as
independent sample growth because several jobs restarted from the same initial states.
The checkpoints, videos, and canonical per-run cells support the following traceable
findings:

| intervention | canonical result | evidence | conclusion |
|---|---:|---:|---|
| control-point density, Long t7 | C-n8 42%→C-n12 **78%** | one seed; paired p=.000912 | extra shape capacity can repair a capacity-bound long task |
| execution cadence, Long t8 | same n16/h48 checkpoint 30%→**52%** | one seed; paired p=.0347; spatial 81.8%→77.4% | longer execution can reduce long-horizon compounding, with a precision tradeoff |
| decode rate, Spatial t5 | 19/50→**34/50** at 2× | matched 600/1,200-step protocol; paired p=.00408 | finer curve sampling repairs a tracking/fidelity failure |
| decode-rate replication, Spatial t5 | 17/50→**36/50** at 2× | separate 1,000/2,000-step protocol; paired p=.000311 | direction reproduces under a second horizon protocol |
| waypoint at 2×, Spatial t5 | **0/50** | same task/rate intervention | waypoint deltas do not provide the spline’s rate-adaptation behavior |

These cells reveal distinct capacity-, cadence-, and fidelity-limited failure modes.
They do not support the previous wording that every member of one pre-registered
Hard-5 set was swept: the later table substituted t8 for t6, several top-ups repeated
states, and the h24=26% source in the 26→42→52 sequence is not traceable. Those
historical runs remain useful for selecting mechanisms and checkpoints. A final
multi-cell table should rerun the frozen cells once per explicit state under the
locked evaluator rather than discard the underlying work.
---

## 2. CALVIN (human teleop, dead-time-rich) — the flagship

Metric = official `avg_len` (mean subtasks completed per 5-task chain). 3 seeds ×
1000 official chains = **n=3000 paired** per contrast.

*Config note (team FAQ):* the CALVIN checkpoint is **n8/h16** (n_ctrl=8, min_seg=8,
horizon_max=16 — not the LIBERO h24). The winning uniform arm is decode-only
(`speedup_alpha=0.6`, threshold 0) with `feasibility_stretch=False`; compression
safety comes from the post-retime clamp to `actuator_bound=1.0` plus CALVIN's
intrinsic headroom (α=0.6 clips only ~2% of steps), not from stretch.

**Base + decode-time retiming:**

| contrast | mean Δ avg_len | 95% CI | paired t | note |
|---|---|---|---|---|
| **uniform retime − C-base** | **+0.379** | [0.318, 0.441] | 12.1 | decisive, every seed, @ **1.42×** realized speed |
| interval retime − C-base | +0.273 | [0.214, 0.332] | 9.1 | @ 1.33× |
| **uniform retime − waypoint A** | +0.127 | [0.063, 0.191] | 3.9 | wins chain depth (seed-variable +.22/−.19/+.34); **first-task tied** (SR1 69.9 vs 69.4, McNemar p=0.56) |
| interval retime − A | +0.021 | [−0.041, 0.083] | 0.65 | parity |
| A − C-base | +0.252 | [0.192, 0.312] | 8.3 | un-retimed spline trails A on teleop data |

**Read.** Base C trails A because it faithfully reproduces the demos' dead-time
(hesitation). **Decode-time retiming recovers and overtakes** — *more success AND ~1.4×
faster simultaneously*, on all 3 seeds. Precise framing: the +0.38/1.42× is vs the
spline's own base; **vs the waypoint, retimed-C improves chain depth while first-task
success is tied.** This is the opposite sign from LIBERO (§4) and is the paper's
strongest single result. Mechanism: 77% of the gain is *behavioral* (removing
reproduced dithering that visits off-manifold states), only 23% clock.

**Cadence robustness.** C is replan-cadence-robust (avg_len flat across nas=5/10),
whereas the waypoint has a narrow sweet spot (nas=50→0.52, nas=10→1.55).

**Language-commanded speed on CALVIN — FINAL, n=600/condition** (6 paired shards,
100k checkpoint; full thread in §6). Speed control is established (+16% "quickly" /
−16% "slowly", adverb-specific). The success question, finalized:

| contrast | Δ avg_len | paired t | p | p (Bonferroni ×2) |
|---|---|---|---|---|
| **quick − plain** | **+0.172** | 2.38 | **0.018** | **0.035** |
| quick − neutral (adverb-specific) | +0.147 | 2.04 | 0.042 | 0.084 |
| neutral − plain (specificity control) | +0.025 | 0.37 | 0.71 | — |

**Headline claim (Pareto form): one word makes the policy 16% faster at
significantly-or-at-least-equal success.** We do not headline "language improves
success" standalone: the effect cleared conventional significance but sat below
our stricter pre-registered t>2.5 bar (per-shard heterogeneity +0.46…−0.05).
Evidence arc: n=30 +0.40 (artifact) → n=100 ns → n=600 significant with clean
controls.

---

## 3. RoboCasa atomic tasks — representation and timing diagnosis

RoboCasa is messy human-teleop kitchen manipulation on a mobile base, with the robot
frequently near its actuator limits (38% of frames > 60% of the bound; p99 = bound).
This historical A/B/C campaign predates the corrected embedded target-statistics
contract. It remains a useful diagnosis of the deployed checkpoints, but it is not
used as the corrected-contract long-horizon comparison reported above.

**A/B/C decomposition** (n=100; the reason we trained the B arm):

| task | A waypoint | B spline (fixed-time) | C spline + time-alloc |
|---|---|---|---|
| kettle (TurnOnElectricKettle) | 35 | 37 | 38 |
| toaster (CloseToasterOvenDoor) | 32 | 30 | **21** |
| faucet (TurnOnSinkFaucet) | 15 | 9 | **6** |
| microwave / coffee | 4 / 0 | 0 / 0 | 3 / 0 |
| **mean (5 tasks — headline)** | **17.2** | **15.2** | **13.6** |
| mean (3 non-floor, diagnostic) | 27.3 | 25.3 | 21.7 |

- **A → B isolates representation cost; B → C isolates time-allocation cost.** Result:
  **B ≈ A** (25.3 vs 27.3 — *the spline representation is free*), **C < B** (21.7 — *the
  cost is the time-allocation machinery*), and only on specific tasks.

**Mechanism — cap-domination.** Event-aligned chunks end on a real event (gripper
toggle/pause) or run out to the 24-step cap. On **event-rich** tasks (kettle: pauses,
29% of chunks event-terminated) contact lands on the pinned chunk endpoint → clean
supervision → **C ties A**. On **event-sparse** tasks (toaster/faucet: push motions,
only 7% event-terminated, 92–93% cap-terminated) the chunks are arbitrary 24-step
windows and the 8 control points are spread thin (density 0.33) → **coarse transport
supervision** → C trails. Direct measurement: the **event-terminated chunk fraction
tracks the C-vs-A gap monotonically** (29% → tie, 7% → trail).

**Density ablation** (single-variable, n_ctrl 8→10) — tested at 50k AND matched 100k:

| task | C@50k | C_n10@50k | B@50k | | C@100k | C_n10@100k | B@100k |
|---|---|---|---|---|---|---|---|
| toaster | 23 | **36** | 26 | | 21 | 22 | **30** |
| faucet | 9 | 7 | 11 | | 6 | 8 | 9 |

- **The density fix does NOT survive matched full budget (honest negative).** At 50k it
  looked like a fix (C_n10 36 > B 26 > C 23 on toaster); but at **matched 100k, C_n10
  (22) ≈ C (21), both below B (30)** — the 50k signal was a favorable/undertrained draw
  (B rose 26→30 with training; C_n10 fell 36→22). So raising control-point density does
  **not** reliably close the RoboCasa gap. **The vulnerability stands: the fixed-time
  spline B beats the time-allocation arm C on the cap-dominated task, and we do not yet
  have a fix that survives matched budget.** Faucet: all arms floor at 6–11%
  (precision-limited, shared difficulty). *Lesson: confirm at matched full budget before
  claiming a fix.*
- **Event-boundary ablation (offline elimination — the other candidate fix).** If the
  deficit is that cap-chunks give the duration head *uninformative* targets (always the
  cap), the natural fix is to subdivide long cap-chunks at **motion events** (velocity
  minima / direction changes), not to add control points. An offline necessary-condition
  gate (`rc_motion_events.py`, n=120 demos/task) asks whether such events even exist to
  subdivide the cap-chunks:

  | task | cap% (gripper only) | cap% + motion events (principled thr) | motion events/ep |
  |---|---|---|---|
  | kettle | 71 | 53 | 5.6 |
  | toaster | 92 | **79** | 2.8 |
  | faucet | 93 | **81** | 3.4 |

  At a **principled** threshold the transports are genuinely event-sparse (~3 events/ep;
  cap-fraction still ~80%), so boundaries would give the duration head informative
  targets on only ~20% of chunks (up from ~8%) — a smaller intervention than density,
  which already failed. Forcing cap-fraction down to ~50% needs such an aggressive
  detector (>10 events/ep, firing on any mild velocity dip) that the boundaries are no
  longer *events* — it is just chopping monotonic motion into shorter fixed windows,
  i.e. re-deriving fixed-time B. **So event-boundary construction cannot cleanly close
  the gap either: too weak at a real threshold, or it collapses to B if forced.** We
  therefore did not spend training budget on it (evidence before compute).
- **Reconstruction diagnostic** (Xiatao's ask; A/B/C open-loop `predict_action_chunk`
  vs demo, n=40/task): trans-RMSE toaster A0.44/B0.54/C0.54, faucet A0.24/B0.29/C0.27.
  **B ≈ C** → C's *closed-loop* deficit is an **execution/chunking** effect, **not**
  open-loop prediction accuracy. A (waypoint) has lowest open-loop error yet only ties
  B in closed-loop → fidelity ≠ success. (Gripper-event-region metric only populated on
  kettle — toaster/faucet too event-sparse, re-confirming the event-fraction finding.)

- **Retiming arms at n=100** (kettle/toaster/faucet): C-interval 29/28/7, C-uni06+
  feasibility-stretch 37/21/7, C-self-paced 33/22/4 — all cluster with C-base
  (21–22 three-task mean): **retiming is success-neutral on RoboCasa**, as the h≈1
  theory predicts (July 20 full matrix; the earlier n=50 "retiming hurts" read was a
  small-n draw — A itself moved 52→35 kettle at n=100).
- **Realized wall-clock on kettle** (mean steps on *successful* episodes; measured
  from rendered eval videos — the recording API rejects RoboCasa's `pixels/` keys —
  so n=1–5 per cell, indicative not headline):

  | arm | A | B | C-base | C-interval | **C-uni06+stretch** | C-self-paced |
  |---|---|---|---|---|---|---|
  | steps to success | 545 | 242 | 230 | 103 | **97** | 128 |
  | success (n=100) | 35% | 37% | 38% | 29% | 37% | 33% |

  Two reads: (i) success-neutral retiming still buys ~**2.3×** faster completion
  vs C-base (97 vs 230) at unchanged success — the wall-clock Pareto axis survives
  on the h≈1 benchmark even though the success axis is flat; (ii) the spline arms
  complete ~2.3× faster than the waypoint *at base speed* (230–242 vs 545) —
  event-chunking does not reproduce per-step dither.

**Bottom line.** In the legacy atomic campaign the spline geometry is nearly free
(B≈A), while the added time-allocation target hurts on cap-dominated, event-sparse
tasks. Raising control-point density did not survive matched 100k training, and an
offline motion-boundary detector found too few principled events to cleanly repair the
targets. This supports event density as a useful predictor of when the timing head is
informative. It does not establish an immutable law: corrected normalization and
official semantic-phase supervision are separate interventions now being tested on
the composite tasks.

---

## 4. The (δ, h, γ) theory — why the three benchmarks behave oppositely

**Core idea.** An imitation policy inherits the demonstrator's **time map** (how time
is distributed along the path). Waypoint heads freeze it into the weights; our head
factorizes it → controllable post-hoc. The theory predicts *when* exercising that
control helps.

**Three offline-measurable dataset statistics** (computable in minutes, no training):
- **δ — dead-time fraction:** fraction of the demo in near-zero-velocity dithering.
- **h — actuator headroom:** bound ÷ p95(per-step Δ). Room to compress before clipping.
- **γ — precision sensitivity:** success loss per unit terminal-tracking error (proxy:
  contact fraction; demo deceleration at events).

**Model.** Retiming (execute in α·time) has three first-order effects on success:

> **ΔSuccess ≈ A·δ − B·clip(α, h) − C·γ**

1. **Dead-time harvest (+, scales with δ):** compression removes hesitation. Not just
   wall-clock — dithering visits off-manifold states, so removing it is behaviorally
   beneficial (77% of CALVIN's gain was behavioral).
2. **Clipping cost (−, explodes as h→1):** compression ×1/α on per-step deltas; where
   1/α > h the actuator clips → path distortion. Negligible for h≫1, immediate for h≈1.
3. **Precision cost (−, scales with γ):** compressing contact segments raises terminal
   error; selective gating (protect-slow) exists to avoid this.

**The three benchmarks are the three regimes:**
- **LIBERO — Regime I (δ≈0, h≫1):** no dead-time → retiming a mild trade (−2 to −5),
  fine gating best (protect slow = protect contact); slow-down free.
- **CALVIN — Regime II (δ high, h>1):** dead-time everywhere → coarse (uniform)
  compression best → **+0.38 Pareto win**; uniform > interval (granularity inversion);
  slow-down *hurts* (manufactures dead-time — pre-registered ✓, 1.22 vs 1.42).
- **RoboCasa — Regime III (h≈1, 38% frames at bound):** compression blocked/re-expanded
  → retiming **neutral**.

**Why it matters:** (a) triages any new dataset before training (compute δ,h,γ → pick
the knob); (b) unifies every timing result, including the **language** asymmetry
("quickly" works on CALVIN, capped on LIBERO — same δ-gating); (c) every decode knob
maps to exactly one term (feasibility-stretch = h-controller, ease-out/protect-slow =
γ-controller, adverbs = a language interface to α).

---

## 5. The capability suite (decode-time knobs on one checkpoint, zero retrain)

| capability | mechanism | result |
|---|---|---|
| **Decode-time retiming** (3 granularities) | resample the continuous curve on a warped time-grid: uniform / chunk-selective (T̂>θ) / interval-selective (protect fast-vs-slow from the chunk's own speed profile) | CALVIN **+0.38 @1.42× (win)**; LIBERO −2 to −5 @1.25–1.4× (trade); RoboCasa neutral (h≈1). Winning granularity is data-dependent (LIBERO fine, CALVIN coarse — the inversion) |
| **Language-commanded speed** *(novel)* | kinematic adverbs (speed terciles, zero labeling) prepended to the clause → learned adverb→control-point-spacing coupling; **T̂ stays flat** (steers speed, not length) | bidirectional ±16% (CALVIN rollout), adverb-specific (neutral-controlled); δ-gated (capped on LIBERO); **Pareto at n=600: +16% speed AND +0.17 avg_len (p=.018/.035 corrected)** |
| **Decode-rate adaptation** *(novel, §5b)* | resample the same curve at k× control rate (`exec_rate_ratio`) — per-step commands scale 1/k (velocity-preserving), verified in closed loop | Spatial t5 **38→68%** and **34→72%** under two horizon protocols; same-state repeats 68/72/74%; matched A waypoint **0/50**. Full-suite effects are task-dependent. |
| **Slow-down** (mirror knob) | dilate the time-grid (α>1) | free on LIBERO; **hurts on CALVIN** (dead-time amplification — pre-registered ✓); positive use → hardware |
| **Event-scheduled replanning** | T̂ = time-to-event; replan at T̂−margin (before the event, never at it) | matched success at **2.1–2.7× fewer policy calls** |
| **Feasibility stretch** (h-controller) | at decode, lengthen the execution window until per-step deltas fit the actuator bound, instead of clipping | **+19 pts at half control rate** (LIBERO) — the mechanism BSP lacks |
| **Endogenous failure detection** | T̂ countdown stalls (trend ≥0 below cap) ⇒ stuck | 86% precision / 52% recall (LIBERO); scoped: needs duration dynamic-range (fails on compressed CALVIN h16) |
| **Ease-out contact landing** (γ-controller) | cosine-retime the final path steps | terminal velocity ×0.14, exact endpoint; restores demo-like toggle kinematics; matters for self-paced + hardware |
| **Object-clause steering** | — | **fails** (redundancy law, §6) |

**Supporting laws** (each from a single-variable elimination):

| law | evidence |
|---|---|
| **Density can repair capacity-bound cells** | Long t7 C-n8 42%→C-n12 78% in one matched seed (p=.000912); intermediate n10 effects are heterogeneous. It did not fix RoboCasa (§3) or the fidelity-bound Spatial t5 cell. |
| **Capacity and rate are orthogonal** — disjoint cell classes | density fixes t7 & is immune on sp5; rate fixes sp5 & costs on long-horizon (§5b) |
| **Time knobs don't stack** (one time budget) | full stack 64–69 vs best single knob 92 |
| **Rate × compression don't compose** | compression claws back the per-step fidelity rate buys: t5 success tracks delta magnitude — 0.75→34%, 0.47→54%, 0.375→72% |
| Temporal augmentation can't regularize this head | the factorization cuts both ways (duration-only aug touches no shape target) |
| Contact-region *fit* fidelity is not binding | −56% tail error → ±0 success |

**Compute** (BEAST-standard reporting, A40 = most conservative GPU):

| | A waypoint | C spline+duration |
|---|---|---|
| full policy call | 209.1 ms | 221.7 ms (**+6%**, VLM-dominated; spline decode = fixed matmul ≪1 ms) |
| env-steps per call at 2× decode | — (0% success) | ×2, zero added latency |
| policy calls per episode (self-paced) | baseline | **2.1–2.7× fewer** |

---

## 5b. Execution-frequency adaptation (the project's stated central motivation)

**The representational premise, proven offline** (decode the *same* spline tokens at
0.5×/1×/2× — fixed flow-matching noise, so only the sampling resolution changes; n=40):

| deploy rate | endpoint dev vs fine ref | path dev vs fine ref | commanded jerk |
|---|---|---|---|
| 0.5× | 3.32 | 1.65 | 0.210 |
| 1× | 0.055 | 0.023 | 0.040 |
| **2×** | **0.000** | **0.002** | **0.0055** |

- **A B-spline samples at any frequency: at 1× and 2× the decoded trajectory is
  *identical* (endpoint/path deviation ≈ 0), and commanded jerk falls monotonically
  with rate (0.040 → 0.0055).** Deploying *faster* is free and strictly smoother —
  something a fixed-Δt waypoint head cannot do (it must interpolate, and its
  piecewise-linear path has impulsive jerk). This is the representational property, with
  zero sim/controller confound.
- **The 0.5× degradation is the (δ,h,γ) headroom term, not a flaw:** coarse sampling
  doubles per-step deltas, which hit the actuator bound and get clamped (endpoint dev
  3.3). Deploying *slower* needs more headroom h; **feasibility-stretch is the designed
  fix** (dilate the execution window until deltas are feasible). So frequency-robustness
  is *derived* from the theory: **up is free (2× identical + smoother), down needs
  feasibility-stretch.**
- **Closed-loop sim sweep (supporting, directionally consistent).** libero_object,
  generous 30 s budget (`max_steps = 30·F`), matched `episode_length`, n = 30; success %
  (hit-cap %) / commanded jerk:

  | arm | 0.5× (10 Hz) | 1× (20 Hz) | 2× (40 Hz) |
  |---|---|---|---|
  | A waypoint (replay) | 0% (cap100) j0.148 | 70% (cap30) j0.111 | **0%** (cap0) j0.157 |
  | A + linear interp | 17% (cap83) j0.356 | 70% (cap30) j0.115 | **0%** (cap0) j0.073 |
  | B spline (fixed-time) | 53% (cap47) j0.122 | 90% (cap10) j0.087 | 73% (cap0) j0.113 |
  | **C spline + time-alloc** | 60% (cap40) j0.424 | 90% (cap10) j0.089 | **93%** (cap0) **j0.033** |

  Three things line up with the theory and with the offline proof: (i) at the native 1×
  the spline arms already lead the waypoint baselines (90 vs 70); (ii) **off the
  native rate the waypoint arms collapse while the spline arms hold** — at 2× the
  waypoint heads go to **0%** (deltas calibrated for 20 Hz command 2× the velocity at
  40 Hz → overshoot), whereas both spline arms survive (**C 93% / B 73%**) because
  `exec_rate_ratio = 2` resamples the *same* curve at double density → correct
  velocities. Linear interpolation of the waypoints (A-interp) does **not** rescue 2×
  — the issue is per-step velocity, not just path shape. And (iii) **the
  time-allocation head *helps* here: C > B at 2× (93 vs 73) at a third the jerk (0.033
  vs 0.113)** — the same machinery that *costs* on cap-dominated RoboCasa (§3) *pays*
  on frequency adaptation, because the learned duration gives a cleaner curve to
  resample at high rate. At 0.5× every arm loses headroom (actuator clamp) but the
  spline arms degrade gracefully (C 60 / B 53) where waypoint-replay floors (A 0).
### 5b-final. Matched-budget closed-loop rate study

The clean result is retained at the level of each canonical protocol; repeated
same-state top-ups are stability checks, not independent sample growth.

| protocol | C @1× | C @2× | paired result | waypoint @2× |
|---|---:|---:|---:|---:|
| 600/1,200-step matched-duration | 19/50 (38%) | **34/50 (68%)** | p=.00408 | **0/50** |
| 1,000/2,000-step horizon | 17/50 (34%) | **36/50 (72%)** | p=.000311 | — |

Additional repeats on the same 50 initialization cases produced 68%, 72%, and 74%
at 2×. They demonstrate numerical stability but are not pooled as n=150. The closed-
loop diagnostic explains the effect: velocity-preserving curve resampling halves the
typical per-step command magnitude (approximately 0.75→0.375), whereas the waypoint
head remains calibrated to native-rate deltas and fails when replayed at 2×.

The capability is task-selective. Historical full-spatial runs are approximately
flat overall (77.8%→76.2%), while the selected LIBERO-Long subset falls from 59.5%
to 29.5%. These suite figures are descriptive because the legacy wrapper did not
retain exact state pairing. Together with the clean t5 cells, they support the scoped
claim: decode rate is a powerful fidelity/tracking knob, not a universal accuracy
improvement. Hardware remains the decisive venue for testing whether the smoother
high-rate spline also reduces real controller jerk.

---

## 6. Language / Molmo thread (the full picture)

**Origin.** Xiatao's directive: use **Molmo 2** (AllenAI VLM) to label each
event-segment with granular language, then test granular language *steering*. LIBERO
ships one coarse instruction per task, so Molmo *generates* per-segment descriptions.

**Pipeline.** Two-stage (extract frames / label in the Molmo env) with an
anti-hallucination video-level context prompt (naming interaction/objects/colors) →
**5,028 verified granular clauses** on libero_object. Variants: **Clang** (granular
only → *specializes*, standard-prompt success crashes to 34%), **ClangMix** (mixed →
restores 88%), **ClangAdv** (mixed + speed adverbs).

**Two independent axes were bundled:** object/interaction steering (from the **Molmo
clauses**) and speed steering (from **kinematic adverbs** — *not* Molmo; Molmo supplies
the clause the adverb attaches to).

**Object steering — fails (the redundancy law).** Two-object scene (soup + cheese),
command "grasp the cheese": the policy grasps the habitual object regardless —
**Δ = P(cheese|cmd cheese) − P(cheese|cmd soup) = 0.00**, tested *in-distribution*
(libero_object task 4) across all three models including granular-only Clang. Mechanism:
which object is grasped is fully determined by the scene + fixed grasp order in the
demos, so the object *word* carries no vision-independent information → ignored. Clean,
un-confounded negative. Prescription: **counterfactually-paired demonstrations** (same
scene, different commanded target).

**Speed steering — works, δ-gated.** Two controls: a **neutral prefix** ("please, …" —
rules out a generic-prefix effect) and a **clause-matched control** (ClangMix — no
adverb training). Findings: **LIBERO** (δ≈0) slow-down is *learned* (ClangAdv "slowly"
−13.5% vs clause-matched control −2.4%), "quickly" *capped*; **CALVIN** (δ high)
**bidirectional** in rollout (+16%/−16%, adverb-specific). Quick working on CALVIN but
capped on LIBERO is the *same δ-gating* as decode-time retiming. **Success (final,
n=600/condition, §2): "quickly" +0.172 avg_len, paired t=2.38, p=0.018 (Bonferroni×2
0.035), specificity-controlled** — significant at conventional levels, below our
stricter pre-registered t>2.5, so the headline claim is the **Pareto form**: one word
buys +16% speed at significantly-or-equal success.

**Long-horizon clause discovery.** On LIBERO-Long, the Molmo/ClangMix checkpoint
moved t0 from 24/50 to **38/50** and t4 from 24/50 to 29/50 in one training seed.
The full suite was flat (308/500→300/500) and t5/t6 regressed, so this is a focused
t0 discovery rather than a broad suite gain. It motivated the exact 71-demonstration
clause intervention reported at the start of this document, which subsequently
produced +30/+31/+42 pp waypoint gains and +41/+42/+48 pp spline gains across
three matched training seeds.

**The unifying principle — the redundancy law:** *language conditioning is learned
exactly where the words carry vision-independent information.* Object words → redundant
with the scene → not learnable from natural data. Speed adverbs → non-redundant (same
scene had fast AND slow continuations; VLM has speed priors) → learnable. One principle
predicts both outcomes.

---

## 7. Positioning vs prior work

- **BSP** (*B-spline Policy…*, arXiv 2607.09648 — full-PDF calibration in
  docs/RELATED_WORK_CALIBRATION.md): the closest work — B-spline segments in DP/ACT
  backbones, accelerated by uniform temporal rescaling. **Phrasing discipline: BSP
  DOES retime at decode time** (that is their core claim) — what they lack is
  *learned/adaptive* timing: their speedup factor is a manually chosen open-loop
  constant, their knots reproduce demonstration timing, and they have **no language
  at all** (no text encoder). Their verified real-world failure — **global 4×
  rescale → 0/20** ("exceeds the controller's tracking limits") — is precisely the
  unlearned retiming our method replaces; and where their success monotonically
  *degrades* with speed, our rate adaptation *increases* success on the hardest
  cells (34→72% @2×, §5b) while their single-seed 20-rollout, test-free evaluation
  contrasts with our 3-seed, paired-test, n≤600 protocol. Their hardware results
  are *favorable evidence for us*: spline actions survive hardware; uniform
  unlearned retiming is what breaks.
- **Speed-as-input** (TempoVLA-style): saturates/collapses (43% @ realized 1.43× vs our
  88%, budget-matched) and requires retraining; ours is decode-time, per-chunk, zero
  retrain.
- The continuous/retimeable-action-representation space is crowding (Spline Policy,
  BEAST, OmniSAT, PACE, SEAM, adaptive-horizon, implicit action fields) → the wedge is
  the **learned time-allocation layer + control surface + theory**, not "splines" or
  "faster execution." **Priority: early arXiv.**

---

## 8. Honest limitations (reviewer-facing)

1. Base spline **trails the waypoint on human-teleop data** before retiming (CALVIN,
   RoboCasa) — mechanistically the event-fraction effect on RoboCasa (NOT fixable by
   density there — both fixes eliminated; §3); faucet: shared precision-floor.
2. Retiming is **neutral on RoboCasa** (h≈1) — no success benefit there.
3. **Language-commanded speed's success gain is moderate**: p=0.018/0.035 at n=600 but below our
   pre-registered t>2.5, with per-shard heterogeneity — hence the Pareto framing,
   not a standalone "language improves success" headline.
4. **Rate-up is scoped, not universal**: historical Long evaluations fall sharply
   at 2×, while the clean positive cell is a short precision/tracking task; its 3×
   rung is also substep-limited in robosuite.
   Hardware (impedance control) remains the decisive frequency venue.
5. **Object steering fails** on redundant natural data — needs counterfactual collection.
6. **No real-robot results yet** — all three closest works (BSP, TRI LBM, BEAST)
   have them; ours are planned for August (cloth-fold + BSP Speed-Stacking
   head-to-head). Mitigation until then: BSP's own hardware results validate the
   representation class; our exposure is the timing layer, which the (δ,h,γ)
   theory + feasibility-stretch are designed to protect.

---

## 9. Evidence and execution ledger

| status | result | use in paper |
|---|---|---|
| **completed, primary** | three-seed LIBERO suite parity | compact-representation baseline |
| **completed, primary** | CALVIN retiming: +0.379 chain length at 1.42× vs C-base; +0.127 vs waypoint | shape–timing factorization headline |
| **completed, primary** | CALVIN kinematic-adverb control: approximately ±16%; quick−plain +0.172 | language-steerable timing |
| **completed, primary** | exact LIBERO-Long clause program: A +34.3 pp, C +43.7 pp across three seeds; interaction +9.33 pp, CI [+0.67,+18.0] | executable-language and spline-specific head effect |
| **completed, scoped** | reversed clause order changes t4's first completed object in 10/50 states for both heads, with zero anti-direction flips | direct language steering of execution order; final retention remains weak |
| **completed, scoped** | Spatial t5 at 2×: 38→68% and 34→72% in two horizon protocols; waypoint 0/50 | decode-rate adaptation |
| **completed, scoped** | RoboCasa target Rinse 7/50 | replicated target-scene composite success |
| **completed, scoped** | RoboCasa official phases: seed 1000 4/100→16/100; independent seed 1001 3/40→9/40; Rinse 1/20→8/20 (`p=.0197`) | replicated phase supervision on Rinse; Kettle remains seed-heterogeneous |
| **supporting mechanism** | Long t7 density 42→78%; Long t8 cadence 30→52%; spatial 81.8% short-cadence run | single-seed capacity/cadence evidence, not one pooled sweep |
| **supporting mechanism** | RoboCasa rack/drawer wrong-prompt success falls to zero for both heads | prompt sensitivity, not alternate-goal completion |
| **preliminary team result** | Quinten’s counterfactual slot: disadvantaged slot 6.7→23–36% | motivates matched A/C grasp-and-lift steering study |
| **smoke complete; full test pending** | leakage-free LIBERO t0+t1→held-out t7 recombination, waypoint/spline × whole/piecewise language; normal-order requested-first signal 4/4 clauses vs 0/4 whole, but 0 full conjunctions | run the frozen seen-task competence gate before the 50-state held-out test |
| **designed, awaiting hardware details** | three-task real-robot compositional study; P1 is held-out two-object kitting | transfers recombination/alignment and decode-time control to hardware |

Historical ClangMix, Hard-5 top-ups, co-scaling runs, and composite fleets are
retained as exploratory evidence and failure-directed discovery. Their checkpoints,
videos, and canonical cells remain useful; only repeated-state pooling, invalid
pairing, or mislabeled replication claims are excluded.

**Current limitations:** no real-robot evaluation; the spline-specific clause effect
has been established only in the SmolVLA backbone and two LIBERO-Long tasks; Quinten’s
raw counterfactual artifacts are not in this workspace; RoboCasa fixed-clock phase
support is unequal across training seeds (50 versus 20 target scenes/task) and Kettle
does not repeat on seed 1001; the completion-oracle pilot is smaller; and
decode-rate gains are task-selective rather than universal.
