# Team update — language steering and long-horizon execution (25 Aug 2026)

Figures: [headline comparisons](../paper/figures/headline_results.pdf) and
[matched Kettle rollout](../paper/figures/kettle_phase_storyboard.pdf), plus RoboCasa
stage visitation for [seed 1000](../paper/figures/robocasa_phase_progress_seed1000.pdf)
and [seed 1001](../paper/figures/robocasa_phase_progress_seed1001.pdf).

## Results summary

| question | result | takeaway |
|---|---:|---|
| Do executable clauses improve long-horizon execution? | mean gain across 3 seeds: waypoint **+34.3 pp**, spline **+43.7 pp** | yes, on both heads, every seed, and exact replay |
| Does the spline head benefit more? | spline−waypoint interaction **+9.33 pp**, hierarchical 95% CI **[+0.67,+18.0]** | yes: compact event-aligned actions improve clause execution beyond the language-data effect |
| Can language steer subgoal order? | on t4, reversing the program flips the first completed object in **10/50** states for both heads, with zero anti-direction flips (`p=.00195`) | yes, although reversed full-task retention remains weak |
| Did the Molmo/clause direction produce useful task gains? | ClangMix t0 **24/50→38/50**; t4 24/50→29/50 | strong one-seed t0 discovery, followed by the focused replicated clause experiment |
| Can representation/decode choices repair hard tasks? | Long t7 density **42%→78%**; Long t8 cadence **30%→52%**; Spatial t5 rate **38%→68%** and **34%→72%** | capacity, replan cadence, and control rate address different failure modes |
| Does RoboCasa respond to language? | rack: A 18/50→0, C 15/50→0; drawer: A 7/50→0, C 6/50→0 under changed directions | both waypoint and spline heads are strongly prompt-sensitive |
| Can our corrected model complete target composites? | RinseSinkBasin **7/50** on held-out target scenes | the initial 1/10 success replicated on 40 disjoint scenes |
| Do semantic phases repair RoboCasa composites? | seed 1000 **4/100→16/100**; independent seed 1001 **3/40→9/40**; Rinse replication **1/20→8/20** (`p=.0197`) | yes on Rinse and both two-task aggregates; Kettle is heterogeneous across seeds |

## LIBERO language programs and hard-task execution

We use two difficult two-subgoal tasks: putting two objects into a basket and placing
two mugs on their respective plates. The exact clause experiment contains 71
demonstrations / 19,378 frames. Original and clause datasets contain identical
observations, actions, and episode schedules; only the language labels differ. At
evaluation, the model executes a frozen two-clause program. A static compound-prompt
control separates sequential-language execution from ordinary fine-tuning.

| seed | waypoint: original→clauses | spline: original→clauses | spline−waypoint gain |
|---:|---:|---:|---:|
| 1000 | 11/100→42/100 (**+31 pp**) | 13/100→54/100 (**+41 pp**) | **+10 pp** |
| 1001 | 7/100→37/100 (**+30 pp**) | 14/100→56/100 (**+42 pp**) | **+12 pp** |
| 1002 | 3/100→45/100 (**+42 pp**) | 8/100→56/100 (**+48 pp**) | **+6 pp** |
| **mean** | **+34.3 pp** | **+43.7 pp** | **+9.33 pp** |

The spline×language interaction is positive on every seed; its training-seed t 95%
CI is [+1.74,+16.92] pp and its hierarchical seed/state bootstrap CI is
[+0.67,+18.0] pp. Spline event-clock gains are +44/+40/+37 pp, while static
compound prompts regress for both heads (A mean −9.3 pp; C mean −24 pp). The effect
therefore requires executable clause sequencing and is not generic fine-tuning or a
clock artifact. All 30 evaluation gates passed exact fresh-process replay: 6,000
rollouts with identical observations, prompts, actions, switches, and outcomes.

A separate exact-replay order probe reverses the two clauses. On the mug-placement
task, the requested first object flips in **10/50 states for waypoint A and 10/50 for
spline C**, with zero anti-direction flips (`p=.00195` for each). Among states where
both programs complete a unique first subgoal, the requested flip rate is 10/12
(83.3%). Normal→reversed final conjunction success falls from 49%→16% for A and
55%→21% for C, exposing a remaining canonical-order/retention bias rather than
invalidating the first-subgoal steering result.

The same research program produced the following complementary hard-task results:

| intervention | result | scope |
|---|---:|---|
| Molmo/ClangMix supervision | t0 24/50→**38/50**; t4 24/50→29/50; suite 308/500→300/500 | one training seed; t0 motivated the exact-clause experiment, while suite/t5/t6 results show the intervention was not uniformly beneficial |
| control-point density | Long t7 C-n8 42%→C-n12 **78%** (p=.000912) | one-seed capacity mechanism |
| execution cadence | same n16/h48 checkpoint, Long t8 30%→**52%** at 5→12 actions/replan (p=.0347) | longer execution reduces compounding on this task; spatial changes 81.8%→77.4% |
| decode-rate adaptation | Spatial t5 **38%→68%** and **34%→72%** under two horizon protocols; waypoint at 2× **0/50** | strong task-specific fidelity/control-rate effect, not a universal suite gain |
| duration prediction | held-out MAE 1.61 steps, Pearson r=.819, balanced accuracy=.892 | predicts motion-event timescale well, but not semantic subgoal completion by itself |

These are individual mechanism results rather than one combined sweep: each uses the
checkpoint and protocol shown in its row. Existing checkpoints and videos remain
useful for locked multi-seed follow-up.

## RoboCasa365 language steering and composite execution

### Instruction sensitivity

| intervention | waypoint A | spline C | interpretation |
|---|---:|---:|---|
| dishwasher rack in↔out | 18/50→0/50 | 15/50→0/50 | strong direction-conditioned suppression of the original goal |
| OpenDrawer left↔right | 7/50→0/50 | 6/50→0/50 | spline compression preserves prompt sensitivity |
| Kettle→faucet prompt | historical Kettle 35/100; changed prompt 0/50 | historical Kettle 38/100; changed prompt 0/50 | qualitative cross-task sensitivity; rates are not paired |

These probes complement rather than repeat Quinten’s target-slot experiment. Our
reward still scores the original goal, while his experiment tests whether a different
language-named object overrides a memorized physical position. We have not rerun his
data; the natural combined follow-up is a matched waypoint/spline grasp-and-lift
evaluation once the raw artifacts are shared.

### Composite execution

We corrected the mismatch between production spline targets and offline normalization
statistics, retrained matched task/granular checkpoints from the same atomic
initialization and 2,029 physical demonstrations, and moved evaluation to the held-out
target split with progress predicates and videos.

| evaluation | full success | observed progress |
|---|---:|---|
| task-caption RinseSinkBasin | **7/50** | initial 1/10 plus 6/40 disjoint-scene replication; success video verified |
| task-caption KettleBoiling | 0/10 in the initial target diagnostic | grasp 6/10; best lift 36.2 cm; video exposed the phase-ordering failure |
| mixed-granular KettleBoiling | 0/10 | grasp 7/10; generic sentence splitting does not improve completion |
| original-label Kettle, scheduled clauses | **2/50 fixed; 0/10 oracle** | matched control; fixed result combines disjoint 10+40-scene shards |
| official-phase Kettle | **6/50 fixed; 4/10 oracle** | pick→place→burner labels triple fixed-clock completion |
| official-phase Kettle + slower placement | **3/10 fixed; 5/10 oracle** | best result; oracle is diagnostic, fixed clock is deployable |
| broad composite fleets | scratch arms approximately zero; atomic-warm-start C-granular **0/450** | broad low-exposure training is insufficient; focused phase supervision is needed |

The 50-scene deployable rows combine a 10-scene pilot and a preregistered 40-scene
disjoint confirmation (target seeds 1000–1049) for training seed 1000. The completed
independent training seed adds 20 new target scenes per task (1050–1069) and is
reported below. The oracle rows use simulator completion predicates and are mechanism
diagnostics rather than deployable policy results.

Kettle video shows the robot place the kettle, approach the knob bank, then return to
and disrupt the completed placement. We therefore created label-only interventions
over the same physical demonstrations:

- official pick→place→burner phases for 501 Kettle episodes / 228,349 frames;
- goal-consistent labels naming the same burner during placement and actuation;
- official turn-water-on→wash phases for 509 Rinse episodes / 211,036 frames.

The matched Kettle seed-1011 videos make the mechanism concrete: both checkpoints
grasp and place the kettle from the same initial observation, but the original-label
model never completes burner actuation in 1,500 steps, whereas the official-phase
model completes the task in 516 steps.

The goal-consistent front-left/front-right follow-up reached only 1–2/10 and did not
beat official phases quantitatively. It nevertheless gives a paired steering probe:
left/right commands share identical initial observations and pick-clause action
prefixes on all 10 seeds, then every trace diverges after the language switch; a
matched successful pair finishes on the two different requested burners. On Rinse,
official phases raise fixed-clock success from **2/50 to 10/50** (`p=.0277`),
including a disjoint 40-scene replication of 2/40→8/40. Original-caption environment
prompting reached 3/10; official-phase training reached 3/10 with diagnostic
completion-gated switching but 0/10 with the unscheduled compound prompt. This
separates better phase execution from generic checkpoint competence. Across Kettle
and Rinse, fixed-clock success rises from **4/100 to 16/100** (+12 pp, 4×,
`p=.00808`).

The independent seed-1001 run used the same physical demonstrations and initialization,
changed only the language schedule, and evaluated 20 disjoint target scenes per task.

| training seed | Kettle original→phases | Rinse original→phases | aggregate |
|---:|---:|---:|---:|
| 1000 | 2/50→6/50 | **2/50→10/50** | **4/100→16/100 (+12 pp)** |
| 1001 | 2/20→1/20 | **1/20→8/20** (`p=.0197`) | **3/40→9/40 (+15 pp)** |

The two aggregate gains are +12 and +15 pp (unweighted mean **+13.5 pp**). Rinse
therefore replicates strongly across training seeds and unseen scenes; Kettle does not,
which keeps its result scoped rather than averaging away the heterogeneity.

The raw progress extrema add a useful mechanism result rather than treating sparse
full-task success as the only signal. On Kettle, official phases increase grasp from
24/50 to **39/50** (`p=.00346`), lift ≥5 cm 14/50→18/50, near-burner 12/50→19/50,
stove contact 13/50→20/50, and any-burner-on 6/50→14/50. On Rinse, faucet-on / at
least one washed region rises 7/50→**22/50** (`p=.00175`) and at least two washed
regions 6/50→**19/50** (`p=.00495`). These are post-hoc stage-visitation diagnostics
and may occur out of order; the 4/100→16/100 ordered task-success result remains the
primary outcome. Seed 1001 independently repeats the Rinse mechanism: water-on / at
least one washed region **3/20→12/20** (`p=.00791`) and at least two regions
**3/20→10/20** (`p=.0407`). Its Kettle trace is mixed (grasp 14/20→16/20 but lower
lift/placement rates), matching the final-success non-replication.

## Current next experiments

| experiment | decision it resolves |
|---|---|
| held-out LIBERO recombination | train only on soup+tomato and cream-cheese+butter, then test the unseen soup+cream-cheese composition in the identical Scene-2 layout; compare waypoint/spline × whole/piecewise language |
| real-robot P1 kitting | train three seen object/zone programs and hold out one composition plus reversed order; use paired reset templates and a later-clause intervention |

The compositional dataset contains 82 training demonstrations / 22,273 frames. All
43 official t7 episodes and their normalization statistics are excluded, and every
cell starts from public SmolVLM weights with a new action expert. A two-state
exact-replay smoke test and seen-task competence gate precede the full held-out grid.
The hardware plan starts with a 90-demonstration three-object/two-zone kitting setup,
then adds a place+button pairing and a drawer close-versus-leave-open intervention
only after the first pipeline is stable.
