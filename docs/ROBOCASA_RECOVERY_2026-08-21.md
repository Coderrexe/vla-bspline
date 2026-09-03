# RoboCasa365 Recovery Log — 2026-08-21

This is the immutable experiment/provenance handoff for the focused recovery of
the four atomic-adjacent composite tasks.  Numbers produced before the v7 scene
lock are explicitly **exploratory** and must not be used as paired paper results.

## Data and exposure

- Selected exact source ranges: KettleBoiling `1010–1510` (501 episodes),
  RinseSinkBasin `3528–4036` (509), ScrubCuttingBoard `4037–4540` (504), and
  StackBowlsCabinet `5546–6060` (515): 2,029 episodes / 843,869 frames.
- Task/granular physical transitions are byte-identical over all selected
  frames (states, actions, rewards, timestamps, indices).
- The granular intervention is an episode-level 50/50 mixture, not a complete
  relabel: Kettle has 251 pure-clause episodes and 250 full-caption controls.
  Scrub and Stack have the analogous mixture; Rinse has no language change.
- At 30k updates, the easy4 curriculum has seen about 1.14 effective epochs and
  the original cosine schedule has reached its low-LR floor.  Current 100k
  runs continue as low-LR controls; task-specific branches reset the optimizer
  and schedule from immutable checkpoints.

Language audit artifact:
`outputs/robocasa_language_audit/easy4_2321865.json`, SHA
`a66291b97e76fed25ed614dcc98258d46c7b4a4aab29e256ab80e09844340ae8`.

## Baseline checkpoint diagnosis

The previously unevaluated warm task-level checkpoint (`rcwarmC_100k`) is 0/10
binary success on each of Kettle, Scrub, Rinse, and Stack.  Failure-directed
n=5 traces reveal useful but incomplete skills:

- Kettle: grasp 2/5, maximum lift 9.8 mm; never transports to the stove.
- Rinse: one seed turns on water and washes 2/3 zones; closest baseline.
- Scrub: sponge grasp 3/5; never transitions to board contact/sweep.
- Stack: first-bowl grasp 2/5; never reaches cabinet or second bowl.

Decode cadence alone is not the fix.  Old-checkpoint NAS2/NAS5 Kettle did not
improve transport, and NAS5 regressed the positive Rinse seed.  At gran30,
NAS5 again degraded acquisition relative to NAS10 (one grasp in each n=5 arm,
no transport/contact).

## Scheduler mechanism

Across all 251 pure-clause Kettle demonstrations, the first→second clause
boundary is exactly a positive→negative gripper command transition.  Boundary
step median is 232 (mean 241; 5–95% = 147–365).  Demonstrators hold the close
command for at least 26 steps before release (median 126), and observed finger
separation at release has median 0.0192 and p1 0.00965.

The old release-only self-paced scheduler is invalid: it switches on the first
spurious release (98–231 steps) despite dozens of later sign oscillations.  A
deployable proprioceptive gate (close streak ≥25 and finger separation ≥0.009)
suppresses most chatter, but still has false fixture/object contacts and is not
sufficient on the underexposed checkpoint.  FixedK=227 remains the clean
outcome-independent clock; cycleK=227 is an exploratory language program.

Exploratory cycleK on gran30 is 0/5, but separately elicits both subskills:
one seed turns the burner on, another lifts 13.5 cm and reaches a burner at
step 1469.  They are not yet coordinated in the same trajectory.

## Easy4 30k progress (exploratory, pre-v7)

Gran30 model SHA:
`7d95363ed5e4ce6964864482f8294f1576736cbe332ef873df05600440b6ca43`.

- Full instruction: 0/5; one grasp, one out-of-order burner actuation.
- FixedK227: 0/5; grasp 2/5, best lift 1.57 cm; no transport.
- Proprio gate: 0/5; grasp 3/5; no accepted semantic transition.
- Oracle/event scheduler: 0/5.  One run grasps at step 150, lifts 11.2 cm,
  reaches 5.5 cm XY from a burner at step 733, but never contacts it.
- Another constructor-locked diagnostic reaches actual stove contact at step
  1006 and switches to the burner clause at 1018, but does not turn it on.
- Gran30 Rinse: 0/5; one seed turns on water and washes center only (1/3),
  below the warm-task diagnostic.

These runs establish reachable intermediate skills and exact bottlenecks, but
not a causal scheduler comparison because the original evaluator did not lock
RoboCasa's one-time constructor randomness.

Under the v7 constructor lock, task-curriculum30 Kettle remains `0/5`; only one
episode grasps, with 1.06 cm maximum lift and no transport or stove contact
(job `2322302`, artifact SHA `0cd616344031c25cc7a20240eaa63f68e6e8768cbe744a65bc7d3bca628243b8`).
Pure-clause focus10 with a fixed step-227 instruction change has zero grasps in
five episodes (job `2322298`, SHA
`07dc36451e9dae2ae63f474b8fbddf9a2c1e579dfabf14176efec5741c09f2b4`).
Together with the oracle arm, this locks the pure-focus branch as a negative
result rather than a scheduler miss.

Task-curriculum30 Rinse and task-focused10 Rinse are also `0/5` with no water
actuation or washed zone in any locked episode (jobs `2322303` and `2322313`,
artifact SHAs `86ae3b33...` and `e4879096...`).  Additional updates under the
same stale target contract are therefore not a sound recovery lever.

## Reproducibility correction

Root cause: LeRobot calls `RoboCasaGymEnv(...)` with `seed=None`; RoboCasa
samples layout/style/object instance during construction and performs an
initial reset before the later episode `reset(seed=...)`.  Thus nominally
identical cross-job episode seeds did **not** start from the same scene.

Evaluator v7 supplies a fixed process-local constructor seed without editing
the installed package, then applies episode reset seeds, locks A100 hardware,
sets deterministic CUDA/TF32 flags, and records reset-observation/action hashes.
Frozen v7 SHA:
`81f0de7bfa8041bf36d82d59ca1571a2b94eb9853003e2345f38a348581f2b75`.

The constructor fix succeeds: duplicate reset-observation hashes and first
actions were byte-identical in the first v7 replicate, but this is not a stable
property.  Closed-loop traces diverged later across processes.  V8 additionally
pins OMP/MKL/OpenBLAS/NumExpr to one thread and records cumulative action hashes.
Its A100 repeats (`2322306/7`) still had different reset-observation hashes and
first actions, so the first stored prefix already differed at step 1.  Therefore:

- n=5 runs are diagnostics only;
- paper results require locked initial scenes, n≥30, replicate seeds, and CIs;
- exact cross-job trajectory identity is not assumed unless the v8 gate passes.

V8 did not pass.  The next gate uses OSMesa because the same renderer change
already recovered exact replay in LIBERO after EGL camera drift was localized.
Legacy checkpoints require the legacy installed policy while corrected
checkpoints require the isolated new source; mixing those contracts correctly
fails fast.  Legacy-compatible OSMesa repeats are jobs `2322435/36`.

## Focus branches

1. **Pure-clause Kettle** from gran30: refuted/stopped after preserving 10k and
   20k checkpoints.  At 10k the training loss collapsed to ~0.20 but locked
   diagnostics had 0/5 grasps, consistent with catastrophic forgetting from
   reduced scene/language diversity.
2. **Mixed Kettle** (`2322293`): all 501 episodes (250 full-caption + 251
   clause-labeled), same gran30 base/seed, fresh 30k schedule.  This is the
   language-density/diversity control and current primary Kettle recovery.
3. **Rinse task focus** (`2322064`): 509 episodes from immutable task10k, fresh
   30k schedule.  Its 10k checkpoint is 1.53 focused epochs, loss 0.236, SHA
   `599811086b0c4ca1d28e88f208a5d6eba22d55f8735cb9111386697d7ab012d3`.
   The locked 10k evaluation had no progress in 5/5 scenes, so this stale-
   contract branch is refuted at its first checkpoint-selection gate.
4. Easy4 task/gran 100k controls continue.  Their immutable 30k model SHAs are
   task `66df95eda589979a649b82bf58edc9e97d33708cb48cccb5ca5b2902b6600783`
   and gran `7d95363e...` above.

## Corrected-contract pivot

A late provenance audit found that every legacy/current RoboCasa C checkpoint
above still uses the stale sampled statistics file
`spline_stats_robocasa_comp_v2_n8h24.json` (SHA `bbc65e99...`) and the old event
target implementation.  The active 30k configs have no embedded stats payload.
This is not a cosmetic mismatch: the corrected all-composite artifact has SHA
`287be415...`; representative normalized-target parameters change materially
(gripper-control mean changes sign, about `+0.11` to `-0.10`; passthrough
control-mode mean shifts about `-0.41` to `-0.72`; pose scales also move).
Continuing the active curricula is therefore a legacy exposure diagnostic, not
the claim-bearing recovery training promised by the project audit.

The repaired path is isolated from the installed policy and all active jobs:

- Frozen source:
  `scratch/vla_bspline/code_snapshots/robocasa_corrected_contract_v1_20260821`.
- Production event target SHA: `9335cae5...`.
- Corrected all-composite stats SHA: `287be415...`.
- Updated checkpoint contract embeds the canonical stats in `config.json` and
  persists the target mean/std tensors in `model.safetensors`.
- Matched task/gran arms start from the same atomic `rcC_100k` model, use the
  same 2,029 physical episodes, seed, updates, optimizer recipe, and source;
  only the audited language overlay differs.
- Launcher: `cluster/train_robocasa_corrected_curriculum.sbatch` (SHA
  `d207c0dd...`).  A CPU import/serialization gate is job `2322330`; GPU jobs
  are launched only after that gate passes.

The CPU gate passed.  Corrected task job `2322332` is running; its launch log
binds the frozen source hashes above, exact dataset size 843,869 frames / 2,029
episodes, and selected-physical-transition digest `e0ec48da...`.  Corrected
granular job `2322333` is queued under normal QOS.  Both use seed 1000 and the
same atomic model SHA; the granular manifest must reproduce the task arm's
physical-transition digest before its result is admissible.

The shared atomic initialization was itself learned in legacy target
coordinates, so these arms are recovery/coordinate-adaptation experiments.
They are substantially stronger and more interpretable than further tuning of
the stale branches, but a final causal paper run should rebuild the atomic
initialization under the same corrected contract if the gate is positive.

### Corrected task-100k video diagnosis (22 August)

The task arm completed 100k and passed the embedded-contract/tensor gate. A
renderer-locked Kettle rollout at seed 1002 shows the policy immediately moving
to the stove controls and turning a burner on while the kettle remains
unacquired; it then wanders around the stove for the rest of the horizon. The
raw progress trace agrees: burner-on is reached, but no grasp, lift, near-burner,
or stove contact. A Rinse rollout similarly moves around the basin without first
turning on water. These are prerequisite-order failures, not inactive policies.

Artifacts are under `outputs/robocasa_forensics/`; the source-video SHAs are
Kettle `1c69f5b...` and Rinse `b7258c0f...`. This diagnosis pre-registers the
corrected granular-100k Kettle crossing: full-prompt control, source-demo median
`fixedk:227` two-clause execution, and simulator-predicate oracle switching as a
mechanism upper bound, all n=10 on identical seed support. The oracle is never a
deployable headline result.

The task-100k triage subsequently exposed two late-stage trajectories despite
zero full completions so far: Kettle seed 1005 grasped and lifted 26.7 cm and
approached within 11.7 cm of a burner but never made stove contact; Rinse seed
1006 turned on water and covered the center/right basin regions but missed the
left region. Renderer-controlled videos (SHAs `dd555f...` and `7170e6...`)
confirm terminal placement and spatial-coverage failures, respectively. These
are progress diagnostics, not successes.

The Rinse trace localizes its phase times: water-on and center coverage first
occur at step 220, right coverage at 893, and left is absent through step 1,350.
Before further Rinse outcomes, this pre-registers a four-clause oracle mechanism
probe: turn on water, then explicitly wash left, center, and right; predicates
advance the clause pointer and already-complete regions are skipped. It uses the
same task-100k checkpoint and ten seeds and records stride-10 videos for every
failure (`2327452`). This cannot be reported as a deployable scheduler because
it reads simulator predicates. A positive result would instead establish that
the missing long-horizon behavior is accessible through language decomposition
and would gate a separate learned-event/fixed-clock scheduler; a null result
would reject prompt scheduling as the immediate Rinse fix.

## Official semantic-phase branch (22 August)

The first held-out `target`-split Kettle rollout from corrected task-100k reached
every individual prerequisite within one episode—grasp, 36.2 cm lift, stove
contact, near-burner, burner-on, and gripper-far—but never simultaneously. This
is the clearest failure localization so far: physical acquisition and transport
are available, while phase order and placement persistence prevent reward. The
exact target seed is queued for stride-5 video replay (`2327636`).

RoboCasa's official July-2026 Kettle archive supplies the missing supervision.
It contains 501 episodes / 228,349 frames and exactly four stages in every
episode: pick, place, actuate the demonstrated burner, done. Across all semantic
boundaries, the nearest raw gripper toggle is 20 steps away at the median and an
exact match only 0.665% of the time. Release-only scheduling is therefore not a
valid approximation of the official subtask program.

Every official episode was matched to combined-data episodes 1010–1510 by its
ordered frame/action SHA256. All 501 signatures and all 228,349 frames match.
The immutable overlay built by `2327584` changes only `task_index` and task
vocabulary; manifest SHA is `b1a7e572...`. Matched H100 fine-tunes `2327585/86`
use original compound versus official per-frame labels from the same corrected
task-100k initialization, seed, physical examples, and 30k-update recipe.

The dependency-armed target-split video matrix (`2327608–13`) evaluates both
arms with the static environment prompt, a deployable demonstration-median
three-phase clock (steps 128 and 242), and a predicate-gated oracle reported
only as a mechanism upper bound. The identical current task-100k checkpoint is
also being crossed with the fixed/oracle programs (`2327660/61`) so any gain
from scheduling alone is separated from the gain from official phase labels.

The official RinseSinkBasin archive was audited independently rather than
assuming the Kettle program generalizes. It contains 509 episodes / 211,036
frames and one consistent three-stage sequence: turn on sink, move the spout to
wash all basin locations, done. Median stage lengths are 162, 230, and 3 frames.
These semantic boundaries are even less compatible with release scheduling:
the median nearest gripper-toggle offset is 50.5 steps and only 0.196% match
exactly. The archive does not provide separate left/center/right language, so it
can test prerequisite ordering (water before washing) but cannot by itself fix
the observed missing-left-region failure. Kettle remains the first GPU priority;
Rinse is the pre-specified second-task replication if Kettle warrants it.

The target-split task-100k Rinse cell subsequently completed at 1/10: seed
1005 achieved benchmark success at step 385. The environment requires all three
regions; the pre-action progress trace records water+center at step 173, left at
318, and reward on the final right-region action. Artifact SHA256 is
`728c7a34aa21e09f9480cc7dac9350888dc4766bfc9bc94efaee6f8393167f9a`.
This is the first nonzero held-out-target composite result of the recovery and
motivates the already physically matched official Rinse label pair (`2327781/82`).

Before the granular-100k outcomes are observed, the Kettle placement diagnosis
also pre-registers one config-only timing crossing. Rate 1 is the full prompt /
`fixedk:227` / oracle matrix above. Rate 2 decodes the identical spline shape at
twice the duration for gentler terminal placement; its fixed clock is co-scaled
to `fixedk:454`, while the oracle predicate is unchanged. Both use the same ten
seeds, 1,500-step benchmark horizon, OSMesa evaluator, immutable weights, and
only `exec_rate_ratio` differs in a job-ID-keyed config variant. Jobs are
`2327441` (oracle) and `2327442` (fixed clock), strictly after the corrected
granular final gate. Rate 2 is only useful if it increases stove contact or full
success without merely suppressing progress; the unchanged horizon is retained
so a slower arm receives no extra interaction budget.

The final task trace shows why slowing alone is unlikely to suffice: the sole
near-burner episode first reaches that region at step 1,485/1,500. This
pre-registers the complementary duration-selective arm already established in
the project: `speedup_alpha=0.6` only when predicted `T>20`, preserving the
learned duration of short event/contact chunks. It is evaluated on the task
full-prompt checkpoint (`2327454`) and on the granular checkpoint with oracle
clause gating (`2327455`), again on the same ten seeds and fixed 1,500-step
horizon. The hypothesis is specifically earlier transport plus preserved
terminal precision—not an unconstrained hyperparameter sweep.

### Official per-frame supervision branch (22 August)

The RoboCasa authors updated the target composite datasets on 7 July 2026 with
per-frame subtask index, atomic-skill, stage, and natural-language annotations
for hierarchical policy learning. The cluster's existing converted
`robocasa_composite_seen` copy predates that schema: its Parquets contain only
the episode task description plus actions/state/reward metadata. This makes the
official annotations a strictly better next language source than further Molmo
or gripper-release heuristics.

CPU jobs `2327499` (Kettle) and `2327500` (Rinse) download only the official
500-demo target archives into new immutable scratch roots. They never overwrite
the current datasets and fail unless all 500 episodes and at least the promised
subtask/skill/stage columns are present in both the first and last Parquet
schemas. No training is authorized until the annotation vocabulary, physical
data identity, boundary distribution, and compatibility with the corrected
event target contract are audited. If the gate passes, the highest-value next
training is a matched task-level versus official per-frame-language arm—not a
larger heuristic-caption run.

The same documentation audit found that the earlier progress evaluator left
RoboCasa's scene `split=None`; those OSMesa cells are valuable failure
diagnostics but are not the official held-out-target-scene protocol. A frozen
v11 evaluator now binds `split=target` in both the environment config and result
artifact. The registered n=10 triage matrix is task Kettle/Rinse (`2327520/21`)
and granular Kettle/Rinse/full-prompt plus Kettle `fixedk:227`/oracle
(`2327522–25`). Only this target-split matrix can gate a RoboCasa365 benchmark
claim; any prior “all” result must remain labeled diagnostic.

## Decision rule

- First require a nonzero full-success task under the v7+ locked scene protocol.
- If mixed Kettle succeeds, cross full instruction, fixed/event language, and
  task-level language on the identical model/scene support.
- If Rinse succeeds first, use it to validate the exposure curriculum, then
  transfer that schedule to Kettle mixed rather than scaling the failed pure
  overlay.
- Final paper claims require an exact task-vs-gran comparison from the same
  atomic initialization, seed, physical episodes, stats contract, and training
  budget.  Corrected jobs `2322332/3` provide that matched recovery factorial,
  but their atomic initialization was learned in legacy coordinates.  A
  positive result gates the final fully corrected atomic-to-composite rerun.
