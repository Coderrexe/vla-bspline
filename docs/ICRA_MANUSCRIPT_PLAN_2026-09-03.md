# ICRA 2027 manuscript and closeout plan

**Current revision:** [5 September focused plan](ICRA_REVISION_PLAN_2026-09-05.md).
It implements the team's two-claim feedback and replaces this document's
capability-heavy main-paper layout. The underlying results remain in the ledger.
In particular, the +34.3/+43.7 pp language effects are changes in training labels
under the same clause-scheduled inference interface, not gains over native
whole-caption inference. Read the complete factorial in `paper/results.md`.

*Decision draft for team review, 3 September 2026. The paper deadline is
15 September 2026, 11:59 PM Pacific.*

## Recommended paper identity

**Recommended title**

> **Language-Addressable Spline Action Programs for Long-Horizon
> Vision-Language-Action Policies**

Strong alternatives:

- **Event-Aligned Spline Action Programs for Language-Steerable Robot Policies**
- **Executable Language Programs with Event-Aligned Spline Actions**

The first title is one phrase, has no colon, and is the most differentiated. It does
not compete with BSP, BEAST, or Spline Policy for the generic claim “splines are a
useful action representation.” It names the new object in this paper: a language
clause paired with a compact trajectory and its learned physical duration.

**One-sentence thesis**

> Fixed waypoint chunks entangle geometry, timing, and task semantics; an
> event-aligned spline head separates trajectory shape from physical duration, making
> each action chunk a compact executable unit that can be scheduled and addressed by
> language.

**What the paper proves**

1. The representation preserves broad VLA competence: across four LIBERO suites and
   three seeds, waypoint, fixed-time spline, and spline-plus-duration policies are
   statistically comparable (82.1, 80.8, and 80.7 average success).
2. Executable clauses improve long-horizon task completion for both heads, and the
   gain is larger for the event-aligned spline head: **+34.33 pp** for waypoints,
   **+43.67 pp** for splines, and a **+9.33 pp** interaction with hierarchical 95% CI
   **[+0.67,+18.0]** across three seeds.
3. Language changes behavior, not only the reported reward: reversing clauses changes
   the requested first completed object on LIBERO-Long t4 in **10/50** states for each
   head with zero anti-direction flips; semantic phases also improve official
   RoboCasa execution on two training seeds.
4. The same shape/time factorization exposes useful execution control: on CALVIN,
   decode-time retiming improves mean chain length by **+0.379 at 1.42x realized
   speed** over the unretimed spline and by **+0.127** over the waypoint policy;
   language controls pace by about +/-16%; spline resampling repairs a LIBERO tracking
   cell from 38% to 68% and from 34% to 72% under two protocols.

These are complementary parts of one story. The earlier timing results should not be
discarded or described as a superseded project: they establish that the proposed
language-addressable action unit carries a controllable physical clock.

## Three contributions for the introduction

Use exactly three bullets.

1. **Representation.** An event-aligned VLA action head that predicts eight B-spline
   control points, a gripper curve, and physical duration. It factorizes path geometry
   from time without changing the VLM or transformer.
2. **Executable language evidence.** A controlled waypoint/spline by
   whole-instruction/clause factorial on LIBERO-Long, with identical physical data and
   exact replay, showing a three-seed spline-specific clause benefit and causal
   first-subgoal steering. RoboCasa provides a harder-domain semantic-phase
   replication.
3. **Time control.** A single learned head supports event scheduling, language pace,
   temporal rescaling, and control-rate adaptation at decode time, including a CALVIN
   success-speed Pareto improvement.

Do not say that clauses only work with splines: the waypoint gain is large. Say that
splines **amplify** the gain while uniquely supplying learned duration and continuous
decode-rate control. Do not say that we invented spline action heads or arbitrary-rate
sampling. Those are established by BSP, BEAST, and Spline Policy.

## Abstract blueprint

The abstract should be about 165--185 words and follow five moves:

1. **Problem:** conventional VLA heads emit fixed-rate waypoint chunks that do not
   expose a compact correspondence between language subgoals, trajectory shape, and
   execution time.
2. **Method:** introduce event-aligned spline action programs: eight control points,
   a gripper curve, learned duration, and executable clause scheduling.
3. **Primary result:** report the three-seed LIBERO-Long A/C factorial and the +9.33 pp
   interaction with its confidence interval.
4. **Behavior and breadth:** mention clause-order steering and replicated RoboCasa
   semantic-phase gains.
5. **Second capability:** report CALVIN +0.379 at 1.42x and state that the same head
   supports language pace and control-rate adaptation. End with hardware only after
   the hardware number exists.

Never place a preliminary compositional-smoke number, an oracle-only result, or a
team-reported number in the abstract.

## Eight-page structure

The official limit includes references. Target **6.5 pages of paper content and 1.5
pages of references**. The budget below sums to eight pages and includes title and
abstract space.

| section | budget | job |
|---|---:|---|
| Abstract + I. Introduction | 0.95 p | problem, thesis, headline results, three contributions, Figure 1 teaser |
| II. Related Work | 0.45 p | spline actions; long-horizon/language grounding; adaptive chunk execution |
| III. Event-Aligned Spline Action Programs | 1.35 p | representation, event targets, training, clause/event scheduling, decode-time controls |
| IV-A. Experimental Setup | 0.35 p | A/B/C arms, benchmarks, pairing, metrics, concise reproducibility |
| IV-B/C. Executable Clauses and Behavioral Alignment | 1.25 p | three-seed factorial, controls, learned clock, order intervention, composition if successful |
| IV-D. Timing and Rate Control | 0.75 p | CALVIN Pareto, adverbs, rate adaptation, compact mechanism readout |
| IV-E. RoboCasa | 0.45 p | two-seed semantic phases, target-scene success, scoped cross-domain lesson |
| IV-F. Hardware | 0.65 p | two focused tasks, paired interventions, one table/filmstrip |
| V. Discussion and Conclusion | 0.35 p | synthesis, limits, next scale step |
| References | 1.45 p | approximately 35--45 carefully selected citations |

If the references reach 1.6 pages, remove prose before reducing figure readability.
Do not use `\tiny` tables or illegible multi-panel plots to force eight pages.

## Section-by-section content

### I. Introduction

Open with the mismatch: long-horizon instructions have subgoal structure, while
ordinary action heads expose a uniform array of fixed-rate deltas. Then show the
factorization:

`instruction -> executable clauses -> event-aligned {curve geometry, duration}`.

The second paragraph should explain why compactness alone is insufficient. A useful
unit needs a boundary, a clock, and a behavioral test. Our event targets give the
boundary, predicted duration gives the clock, and clause/order interventions test
language dependence. The third paragraph gives the headline numbers. End with the
three contribution bullets above.

### II. Related Work

Use three dense paragraphs, not a catalog.

1. **Action representations:** ACT/Diffusion Policy/FAST, then BEAST/BSP/Spline
   Policy. Credit continuous splines and resampling fully; distinguish event duration
   and the language-head interaction.
2. **Long-horizon and language grounding:** pi0.5, Long-VLA, VLAs as Tools, CAST,
   LIBERO-CF/CAG. Distinguish an end-to-end language-addressable action unit from a
   planner/tool family or counterfactual inference guidance.
3. **Execution horizons and steering:** PACE and ITPS. Distinguish supervised physical
   duration and language clause scheduling from action-derived phase detection or
   human steering.

The detailed source calibration is in
`docs/ICRA_LITERATURE_REVIEW_2026-09-03.md`.

### III. Event-Aligned Spline Action Programs

Use four subsections and one notation table at most.

**A. Shape-time factorization.** Define a degree-p spline

`a(u) = sum_i B_i,p(u)c_i`, with normalized phase `u=t/T`.

The policy predicts eight control points, the gripper trajectory, and duration
`T_hat`. Explain pinned start/end fitting and the precomputed least-squares basis in
two sentences. State the waypoint, fixed-time spline, and duration-spline arms.

**B. Event-aligned targets.** A training chunk terminates at the next gripper toggle
or sustained pause, otherwise at the cap. Separate continuous pose and discrete
gripper handling. Be explicit that the standard action loss is applied after the
spline target transform and that the duration loss is an added channel.

**C. Executable clauses.** A program is an ordered list of clauses. At execution,
the active clause conditions the same policy until a schedule advances it. The main
head comparison uses a fixed clock to keep A and C identical; the learned-duration
clock is an additional spline-only deployment result. The order probe uses a
privileged predicate transition and must be labeled diagnostic.

**D. Decode-time controls.** In one compact equation/diagram, show temporal scaling
and curve resampling. Language pace modulates the time map; execution-rate changes
sampling density; predicted duration chooses replanning time. Avoid reproducing the
entire delta/headroom theory in the method section.

### IV. Experiments

Frame the section around four questions:

- **Q1:** Does the representation preserve ordinary task competence?
- **Q2:** Do executable clauses improve long-horizon execution, and does the spline
  head benefit more?
- **Q3:** Does changing language change the appropriate part/order of behavior?
- **Q4:** Does shape-time factorization yield useful timing and rate control?

RoboCasa and hardware answer Q3 in new domains; CALVIN and the rate study answer Q4.

## Figure and table plan

### Figure 1 — paper teaser and method (full width, page 1)

The visual target is the information density of PACE's page-one overview and the
behavioral clarity of ACG/ITPS, rendered in our own visual language. Use three
horizontally connected panels:

1. a whole instruction split into two executable clauses;
2. each clause mapped to eight spline control points plus duration, compared with a
   dense fixed-rate waypoint array;
3. two outcomes from one checkpoint: changed clause/order changes the subgoal, while
   changed timing/rate changes execution without changing geometry.

Place two headline callouts directly in the figure: **+9.33 pp spline-specific clause
gain** and **+0.379 chain length at 1.42x**. Use a colorblind-safe blue/orange pair,
direct labels rather than a detached legend, and vector text at final print size.
Figure 1 should make the entire paper understandable before the method equations.

### Table I — broad competence and primary clause factorial

Use two blocks in one double-column table:

- LIBERO four-suite three-seed A/B/C averages;
- the three LIBERO-Long clause seeds and mean interaction.

This simultaneously answers “does the representation work?” and “what is the main
win?” Put the hierarchical CI in the caption or final row.

### Figure 2 — executable-language mechanism

Left: interaction plot for whole versus clauses, A and C, with seed dots. Middle:
normal/reversed t4 filmstrip with the first completed goal highlighted. Right: held-out
composition if and only if the full test succeeds; otherwise show learned-clock versus
fixed-clock or the faster successful spline trajectories. Do not devote a main panel
to a null unfinished experiment.

### Table II — cross-domain control

Keep this compact:

| domain | intervention | primary outcome |
|---|---|---|
| CALVIN | spline retiming | +0.379 vs C-base at 1.42x; +0.127 vs waypoint |
| CALVIN | “quickly” | +16% pace; +0.172 average chain length |
| LIBERO Spatial | 2x curve sampling | 38->68 and 34->72; waypoint 0/50 |
| RoboCasa | semantic phases | 4/100->16/100 and 3/40->9/40 |

Use denominators and seeds in the caption. Do not combine percentages into a single
average.

### Figure 3 — hardware (approximately two-thirds page)

One wide filmstrip plus one small quantitative plot. Preferred content:

- held-out order/composition on a two-object 3D-printed task;
- a matched later-clause intervention with identical early behavior and divergent
  late behavior;
- optionally, native versus changed control rate or pace on the same checkpoint.

If only two physical tasks are completed, present two strong tasks. Accepted ICRA
2026 examples hPGA-DP and ACG both support a compact two-task hardware section.

## Result inclusion map

No verified result is being discarded. The eight-page paper ranks results by how
directly they support the thesis.

| result family | placement | role |
|---|---|---|
| LIBERO A/B/C three-seed suite comparison | Table I | competence/control |
| three-seed exact clause factorial | Table I + Figure 2 | primary headline |
| predicted action-event clock and static-prompt controls | text/Figure 2 caption | mechanism |
| clause-order intervention | Figure 2 | behavioral causality |
| held-out composition | Figure 2 only if full competent evaluation is positive | strongest possible extension |
| RoboCasa target and semantic phases | Table II + short paragraph | hard-domain replication |
| CALVIN retiming and language pace | Table II + small Pareto inset | second headline/capability |
| LIBERO rate adaptation | Table II | continuous-rate capability |
| duration calibration | method/evaluation sentence | validates learned clock |
| t7 density and t8 cadence | compact discussion sentence or table footnote | supporting mechanism |
| Quinten's slot decorrelation | related/diagnostic text only until raw artifacts are locally verified | team complement, not our main estimand |
| historical Molmo/ClangMix | motivation sentence | discovery that led to exact clauses |

The complete traceable ledger remains `paper/results.md`; page pressure determines
presentation density, not whether an experiment is considered useful.

## Essential experiment closeout

### Priority 0 — compositional holdout, one gated decision

The four 20k checkpoints and exact-replay smoke tests are complete. Normal-order
requested-first selection is 4/4 for clauses versus 0/4 for whole captions, but no
two-state smoke rollout completes the conjunction. The next step is already locked:

1. Evaluate t0/t1 in-distribution competence at the 20k checkpoints.
2. If competence is still improving, resume all four matched cells to 30k and select
   using t0/t1 only.
3. Run the 50-state same-scene held-out grid only after checkpoint selection is
   frozen.
4. Replicate a positive interaction before making it a headline.

This is the only new simulation direction that can materially improve the main story
within the deadline. Do not start broad new language ablations first.

### Priority 1 — hardware aligned with the paper claim

Use the existing `docs/REAL_ROBOT_COMPOSITIONAL_PLAN.md` as the starting protocol,
adapted to the photographed Yale platform: two 7-DoF xArm7 arms mounted on 1-DoF
linear actuators, with a handheld one-arm teleoperation interface and existing
3D-printed non-cuboid stack/drawer fixtures. Confirm which arm carries the camera and
which carries the gripper before collection. The rail can remain fixed unless a reset
or transport is unreachable without it; avoiding unnecessary base motion simplifies
the action space and evaluation.

This is the same physical platform and fixture family used by the accepted hPGA-DP
ICRA 2026 paper, which reported two tasks with 200 demonstrations each in approximately
one page. Two other accepted steering/VLA papers used two hardware tasks with 40--60
demonstrations and about 10 trials per task. We should reuse the established platform
but test a new language/action-program hypothesis. The minimum strong package is:

1. **Drawer program (primary):** `open drawer -> insert red object -> close drawer`,
   paired with a later-clause intervention such as `leave the drawer open` or a second
   compatible target. This is the strongest visually obvious long-horizon/alignment
   task because the first two clauses and initial scene can be identical while only
   the requested terminal state changes.
2. **Non-cuboid stack/program composition (secondary):** use the existing printed
   blocks for two demonstrated primitives and a held-out order or composition that is
   physically valid. Score requested first primitive and final construction. If the
   printed geometry admits only one stable order, use it as a timing/control-rate task
   rather than forcing an invalid compositional split.

Collect each physical trajectory once and derive whole-caption and clause-labeled
views from identical frames/actions. The preferred comparison is the full 2x2
waypoint/spline by whole/clause factorial. If four hardware checkpoints cannot all
reach competence, preserve the simulation factorial and use the competent spline
whole/clause pair for the hardware capability demonstration; never compare treatments
trained on different physical episodes.

Start with approximately 40--60 demonstrations per training program, balanced across
pose bins; add data in balanced blocks only if seen-program validation is inadequate.
Use at least 10 and preferably 20 frozen paired initial configurations per condition,
interleave treatments, and report first-subgoal, final success, requested order, and
prefix/late trajectory divergence. Add a timing/control-rate condition only after the
language result works. Two carefully controlled tasks are better aligned with the
manuscript than three unrelated demonstrations.

### Priority 2 — paper assets, not another benchmark

- Generate Figure 1 and the main tables directly from immutable JSON/CSV artifacts.
- Re-render the most diagnostic successful and failed rollouts for Figure 2/video.
- Measure model/head parameter count, training time, and inference/decode latency once.
- Run an independent number-to-artifact audit before any value enters TeX.

### Explicitly deferred unless already nearly operational

- A fresh pi0.5 or GR00T port. It would be useful later, but integration risk is too
  high relative to the established three-seed causal head comparison and twelve-day
  deadline.
- A broad RoboCasa composite sweep. Current phase/stage evidence is more informative
  than another large set of near-zero full-task cells.
- Additional Molmo prompt variants. The exact clause intervention already isolates the
  useful signal more cleanly.
- An exhaustive comparison to every concurrent spline repository. Cite and distinguish
  them; implement a direct baseline only if it can use the same data/evaluator without
  a new engineering stack.

Deferring these does not weaken the paper. It protects the experiments that uniquely
differentiate it.

## Hardware page standard

The hardware section need not occupy more than about 0.65--0.9 page. The accepted
hPGA-DP paper on this exact platform uses roughly one page for two tasks; ACG and ITPS
do the same on other platforms. Our section must contain:

- robot/camera/control mode and number of demonstrations;
- the train/test program split;
- fixed paired reset configurations and trial count;
- a table with denominators, not only percentages;
- one visibly obvious language intervention in the video;
- basic safety/stopping behavior.

It does not need a large task suite. It does need to test the same composition or
alignment claim as simulation. The page becomes much stronger if it also illustrates
the spline clock—different pace or rate on the same learned geometry—without making
that a separate data collection campaign.

## Twelve-day schedule

| date | hard deliverable |
|---|---|
| **Sep 3** | approve title/thesis/outline; freeze result ledger and literature map |
| **Sep 4** | run compositional competence gate; draft Introduction, Related Work, and Figure 1 wireframe |
| **Sep 5** | select/extend compositional checkpoints; draft Method and primary Table I |
| **Sep 6** | complete compositional full test if gated in; freeze all simulation experiments |
| **Sep 7** | hardware calibration and pilot; complete first full manuscript draft |
| **Sep 8** | hardware data collection/training; finish all simulation figures and Table II |
| **Sep 9** | hardware smoke evaluation; upload an initial video if sufficiently complete before the first video window closes |
| **Sep 10** | paired hardware evaluation; manuscript compression to <=8 pages |
| **Sep 11** | team technical review; reconcile every comment and every number |
| **Sep 12** | hardware contingency/final filming; freeze results and figures |
| **Sep 13** | argument, related-work, and anonymity pass; compile clean PDF |
| **Sep 14** | independent artifact audit, PDF compliance, references, metadata, AI-use disclosure |
| **Sep 15** | final coauthor approval and submit several hours before the deadline |

The official second video window is Sep 17--22, but an initial Sep 9 upload is safer
if the system permits replacement in the second window. Confirm the PaperPlaza video
association before relying on the later window.

## Number and claim audit

Before drafting prose around a result, record:

- artifact path and SHA256 or immutable job/output ID;
- checkpoint/data/normalization hashes;
- number of training seeds, evaluation states, and repeats;
- paired versus unpaired design;
- exact numerator/denominator and uncertainty/test;
- whether any oracle, privileged predicate, or reused initial state was involved.

Every manuscript table should be regenerated by a script where possible. A second
person should cross-check the generated table against `paper/results.md` and the raw
artifact. This is not pessimism: it lets the paper state its unusually strong results
confidently and prevents a last-day correction from weakening the story.

## Decision requested from the team

Approve or revise four choices before prose drafting:

1. the recommended one-phrase title and language-addressable action-program framing;
2. executable clauses as the primary headline, CALVIN timing as the second;
3. the 6.5-page content / 1.5-page reference budget;
4. one gated compositional simulation closeout plus two focused hardware tasks, with no
   new broad benchmark or backbone port before submission.
