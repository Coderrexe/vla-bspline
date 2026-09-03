# Overnight execution — 2026-08-21

This file locks the experiment design before the new results are observed.

## Decision being tested

Does mixed whole-task/granular language supervision improve long-horizon
execution, and is any improvement specific to the event-segmented B-spline
head rather than ordinary language augmentation?

## Factorial design

- Data: the exact existing `libero_l10gran` dataset for every new training.
- Heads: stock SmolVLA waypoint (A) and event-segmented n=8 B-spline (C).
- Seeds: 1000, 1001, and 1002 for both heads.
- Evaluation: LIBERO-Long, explicitly assigned initial states 0–49 for every
  task/checkpoint, 50 episodes per task and 500 per checkpoint. The evaluator
  overrides and logs the state id before every episode, because the deployed
  environment's internal terminal reset otherwise makes the visited sequence
  depend on earlier successes. Python, NumPy, Torch, and CUDA RNGs are also reset
  from the task/state key before every episode so stochastic flow samples are
  paired across checkpoints.
- Primary matched cadence: both A and C execute 10 actions per inference. This
  is the architecture-by-language interaction used for the primary comparison.
- Deployment secondary: all three C baseline/language seed pairs are also
  evaluated at five actions per inference. This preserves the promoted C
  deployment setting without confounding the A-versus-C head comparison.
- Primary estimand: within-head paired language-minus-standard change, followed
  by the A-vs-C difference in that change across independent training seeds.
- Secondary estimands: paired per-task McNemar effects, family-wise corrected;
  suite-level change; regressions as well as gains.

## Interpretation guardrails

- The previous seed-1001 language checkpoint is excluded: its launcher passed
  both seed 1001 and seed 1000, and the saved config confirms seed 1000.
- Repeated top-up evaluations are not pooled. Several historical jobs restart
  LIBERO from initial state 0, so concatenating them inflates the effective n.
- The earlier RoboCasa swapped-instruction reward probe demonstrates instruction
  sensitivity, not successful execution of the counterfactual goal. Quinten's
  slot-decorrelated target-object experiment is the stronger causal steering
  evidence and is complementary rather than redundant.
- No new RoboCasa spline training is launched until its offline statistics are
  regenerated from the same event-boundary and speed-scale definition used by
  the production policy.
- A successful generic language augmentation effect is useful, but it is not
  evidence that B-splines uniquely enable language steering. The interaction
  between head and supervision must support that stronger claim.
- Historical standard-data checkpoints lack the complete job-start training
  manifests now required for a paper-grade causal factorial. In particular,
  the historical C source changed after those trainings and the language/base
  checkpoints were trained on different GPU classes with nondeterministic CUDA
  kernels. The overnight factorial is therefore a **diagnostic gate**. If its
  interaction is promising, the claim-bearing follow-up is a fresh
  source-snapshotted, hardware-matched standard-data baseline round; negative or
  flat results stop that expense.

## Cluster constraints

Jobs use the production GPU partition, standard QOS, explicit resource requests,
fail-fast shell behavior, and exclusion of the node that caused the prior RoboCasa
fleet crashes. Outputs are compact and new dataset copies are avoided because the
group project and scratch allocations are close to their file-count quotas.

## Submitted jobs

Training jobs:

| Job | Cell |
|---:|---|
| 2319090 | A + language mix, seed 1000 |
| 2319091 | A + language mix, seed 1001 |
| 2319092 | A + language mix, seed 1002 |
| 2319093 | C + language mix, genuine seed 1001 |
| 2319094 | C + language mix, genuine seed 1002 |

The first state-lock smoke test was job 2319095 and completed successfully on
all ten tasks. Comparing it to historical state-0 outcomes exposed an additional
uncontrolled policy-RNG variable. Jobs 2319103–2319109 were cancelled after four
minutes (2319110–2319111 had not started) rather than accept non-paired evidence.
The evaluator was then strengthened with per-task/state RNG locking. Independent
smoke jobs 2319121 and 2319122 produced exactly identical success, step-count,
and policy-call traces for every task. The final single-evaluation-hardware-class
(L40S) full baseline/cadence jobs are 2319146–2319154. Final language-checkpoint jobs
2319155–2319159 are gated by `afterok` dependencies on their exact training jobs.
The superseded dependency jobs 2319112–2319116 were cancelled before running.

Red-team review before any language outcomes were available found that A@10
versus C@5 would still confound the architecture interaction with cadence.
Jobs 2319604–2319605 add baseline C seeds 1001/1002 at NAS=10, and dependency
jobs 2319606–2319607 add language C seeds 1001/1002 at NAS=10. Together with
the already submitted seed-1000 crossing, the primary comparison is now fully
matched at NAS=10 across all three seeds; C@5 is explicitly secondary.

Corrected RoboCasa statistics jobs 2319171 (all atomic episodes) and 2319172
(all composite-seen episodes) run on the CPU `day` partition. They write only to
an isolated repair directory; they do not install the new statistics into a
policy or alter any existing checkpoint.

Both completed successfully: job 2319171 in 28 seconds over 9,126 episodes and
job 2319172 in 59 seconds over 8,077 episodes. The manifest-recorded SHA-256s are
`7dec255e...ecac` for the generator, `9335cae5...e9f6` for the shared event-target
source, `d07661a0...50e0` for corrected atomic statistics, and
`287be415...8007` for corrected composite statistics. Full hashes live in
the generated manifests under `outputs/rc_stats_repair_20260821`.

Dataset audit job 2319212 scanned all 273,465 frames of `libero_l10gran` in
eight seconds. It found 86,148 changed frames (31.5%) in 831/1,693 episodes,
covering every Long and Goal task but no Object or Spatial task. Of 1,385
appended captions, 975 are used and 481 appear in only one contiguous run. The
dataset is therefore a substantial mixed language intervention, but it was made
with the legacy heuristic boundary extractor; tonight's factorial tests generic
language augmentation under that artifact, not exact production-event alignment.
The immutable audit is
`outputs/language_dataset_audit/l10gran_2319212.json` on Misha.

The old-vs-production boundary audit is now quantitative: on the 3,599
surviving captioned intervals, 71.66% require recaptioning and only 4.61% of
episode partitions are exact. The current factorial is therefore explicitly a
legacy-language augmentation study.

Semantic negative-control gates also completed. Job 2319578 found a
same-original-task-support caption derangement covering 92.83% of appended
frames and 93.36% of runs. Exact policy-tokenizer audit 2319582 found 0/966
token collisions and no truncation at 48 tokens. The compact overlay build
2319588 completed at only 413 KiB apparent size, with frame data and episode
metadata symlinked to the immutable source. No shuffle-control GPU training is
launched until the primary factorial justifies that follow-up.

The first locked baseline results to land are A seeds 1000/1001/1002 at
284/500, 328/500, and 319/500 (56.8%, 65.6%, 63.8%; mean 62.1%, training-seed
SD 4.65 points). These are calibration cells, not language results. The large
seed range validates the three-seed design.

C baseline seed 1000 at the promoted NAS=5 deployment cadence is 321/500
(64.2%), with per-task successes
`[28, 43, 41, 36, 21, 43, 33, 33, 15, 28]`. Its immutable result SHA-256 is
`20c450b44a4e...c7298`. This is also calibration, not a language effect.

The three A trainings project close to their nine-hour limits. Dormant
`afternotok` resumes 2319590/2319592/2319594 and after-resume evaluations
2319591/2319593/2319595 are queued. They consume no resources if the original
jobs succeed and prevent a timeout after the 50k checkpoint from losing the
factorial.

Because the active LeRobot worktree contains untracked spline code, its complete
`src/lerobot` tree, launcher, dataset metadata, Git state, and Slurm job records
were frozen during the still-running trainings as one compact read-only archive:
`scratch/vla_bspline/code_snapshots/language_training_active_20260820_v2`.
The `SHA256SUMS` digest is `5dfaf99b49d0...61883e7`. This snapshot was taken
before any installed policy change; the installed source is intentionally left
untouched until the diagnostic and any matched follow-up training finish.

Duration-mechanism work remains gated. A ten-episode no-intervention smoke job
2319599 matched 8/10 outcomes but diverged on two tasks from the concurrent
canonical C evaluation, despite identical model/config/state/seed and the same
L40S node. The stronger single-GPU sequential job 2319617 then ran two canonical
arms on one exact GPU UUID and already diverged on task 2 (240 versus 251 steps)
before invoking the hook. Thus v1 is not trajectory-deterministic and the active
factorial is diagnostic rather than exact common-random-number evidence. The
superseded independent repeats 2319608–2319609 were cancelled before running.
Deterministic-v2 smoke 2319625 seeds before construction and after reset,
requires deterministic CUDA/cuDNN, disables TF32/flash/memory-efficient SDPA,
pins CPU threads, and hashes every executed action. It must replay the known
sensitive task 2 twice on one GPU with exact action-trace equality. **It did not
pass:** the two processes succeeded at 244 versus 241 steps with different
executed-action hashes. The next gate must reuse one constructed environment
and one loaded policy in one process, repeat the same reset twice, and hash the
initial observation plus action prefixes. No full duration intervention is
admissible until that stronger in-process gate passes. CPU
job 2319621 uses an action-only Parquet scanner for the training-target prior;
its pending request was right-sized to two CPUs/8 GB. The earlier generic
LeRobot scan was cancelled because it unnecessarily decoded images, and the
first 8-CPU/24-GB action-only submission was superseded before running.

The right-sized job completed in 13 seconds over all 1,693 episodes and 273,465
training frames. The global discrete median event duration is the horizon cap,
`T=24`; most original-task medians are also 24. The fixed-duration arm therefore
has a clean interpretation as an always-full-horizon controller. The immutable
prior SHA-256 is `c75f6f7e730d...1ccf28`, with action-schedule digest
`fcaba294fbfd...d97f17` and pinned event-kernel digest
`9335cae53e27...16e9f6`.

Direct-vs-LeRobot parity job 2319637 then matched all eight explicit
start/interior/near-end/final action windows, padding masks, and event targets
bit-for-bit without video decoding. Its artifact SHA-256 is
`dd02378e2020...cccddb`. The fixed `T=24` prior has passed both exhaustive
direct scanning and real-loader parity.

The stronger one-process duration gate was then run as job 2319737 on the
previously unstable task 2/state 0. It also failed, despite matching raw and
processed initial-observation hashes and matching every executed action through
step 119. The first action difference was step 120; both rollouts nevertheless
succeeded at 239 versus 244 steps, and their predicted-duration sequences first
differed at chunk 29. The immutable diagnostic artifact SHA-256 is
`d88db01b8e02...d268ba`. No fixed or shuffled intervention was executed. This
localizes the residual variability to closed-loop simulator/observation drift
after an identical prefix, rather than policy construction, initial-state
selection, or the duration hook. Claim-scale exact-pair inference is therefore
not available in this stack. The next admissible design is a repeated,
counterbalanced stochastic block with an explicit repeatability/noise estimate;
it must be labeled stochastic and analyzed across states and training seeds.

The first complete cadence-matched language diagnostic is now available for
the historical C seed-1000 checkpoints at NAS=10. Standard data scored
321/500 (64.2%), while the legacy granular-language mix scored 303/500 (60.6%),
a suite change of -3.6 points. Per-task changes were
`[+4, -8, -16, -14, +20, +2, -22, 0, -14, +12]` points. Task 4 is the one
large positive cell (29/50 versus 19/50; uncorrected paired McNemar p=0.0309),
but task 2 and task 6 regress significantly before multiplicity correction and
the old task-0 headline does not reproduce. This is one discovery seed with
historical provenance, so it is neither a confirmation nor a basis for a
task-selected claim. The two genuinely new training seeds remain the decision
gate. Immutable result hashes are `b7da070f...2073e` (base) and
`36fc577e...105be` (language).

Stochastic duration engineering job 2320368 completed all six balanced orders
on task 2/state 0 (30 rollouts) with every initial seed, raw/processed
observation, compiled-model, and MuJoCo integration-state check matching. The
three predicted positions scored 3/6, 2/6, and 3/6; donor versus closure outcome
agreement was 6/6, although only 4/6 exact action traces replayed. Fixed `T=24`
and shuffled duration each scored 5/6, versus 2/6 for the randomized-position
efficacy predicted arm (+50 points in this deliberately selected engineering
cell; paired two-sided p=0.25 at n=6, not an efficacy claim). The shuffle was
strong (77.6% calls changed; mean absolute change 5.92 steps). The operational
and intervention gates passed, while the deliberately strict six-block pilot
repeatability screen failed because one predicted-position outcome flipped.
This justifies the predeclared all-task 12-state stochastic pilot, not a paper
claim. Artifact SHA-256: `86ab7346...a1502`.

The hidden-state traces localize the residual noise: all six repeats began from
identical complete integration states. Observation hashes diverged before the
first action and physics divergence (median raw/processed step 101.5, action
step 120, integration-state step 121 for donor versus efficacy predicted). In
one donor/closure pair, the integration state and exact actions remained equal
for the whole episode while observations differed from step 107. This points
to the rendered observation path rather than the duration head or simulator
reset. A first OSMesa diagnostic failed immediately because the conda runtime
did not find `libOSMesa`; a retry loads Yale's provided Mesa module rather than
installing or modifying the environment. The Mesa-backed diagnostic job
2320402 then produced three exactly identical predicted trajectories: every raw
and processed component, integration state, duration, and action matched for
all 245 steps (artifact SHA-256 `e3c48f56...f4f80a`). This recovers exact paired
evaluation by changing only the renderer backend. The superseded pending EGL
pilot 2320398 was cancelled before it ran. A six-block OSMesa exact gate is job
2320438; the 600-rollout all-task OSMesa pilot 2320439 is dependency-gated on
it. Both use immutable source snapshot
`code_snapshots/duration_v3_osmesa_release_20260821` (SHA256SUMS digest
`4bf0d07c...a9911d`) and fail after publishing diagnostics if any predicted
pair diverges.

The same seed-1000 language comparison is complete at the native NAS=5 cadence:
standard 321/500 (64.2%) versus language 316/500 (63.2%), or -1.0 suite point.
Task changes are `[-2, -18, +4, +14, +20, -14, -24, +6, 0, +4]` points. Task 4
again improves by +20 (31/50 versus 21/50; uncorrected p=0.0414), but task 1
and task 6 regress; none of the ten task tests survives Holm correction. Thus
the only repeatable-looking positive is task 4 across two deployment cadences
of the same discovery checkpoint, while aggregate performance is flat/worse
and task 0 does not reproduce. Language result SHA-256:
`ba813f20...e33f8`; its paired base is `20c450b4...c7298`.

Additional completed C baseline calibration: seed 1001 scores 300/500 at NAS=5
and 313/500 at NAS=10; seed 1002 scores 310/500 at NAS=5. These remain
schema-v1 diagnostic cells, not claim-bearing comparisons.
