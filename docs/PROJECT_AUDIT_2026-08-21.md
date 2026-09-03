# Project audit and execution state — 2026-08-21

This is the current source of truth after reviewing every local Markdown/code
artifact, the project-owned code and raw outputs on Misha, the supplied
Simba/Xiatao/Quinten discussion, and current related work. It intentionally
distinguishes completed evidence from interpretation and from work in flight.

## Executive conclusion

The project has a real method and several strong phenomena, but the broad pivot
claims were ahead of the evidence. The defensible current position is:

> An event-delimited VLA action head that predicts geometric B-spline shape and
> explicit physical duration, enabling a direct temporal contract between
> language, event structure, and continuous execution. The submission must test
> that contract causally under matched data, compute, states, stochastic policy
> samples, and output budgets.

What is already credible:

- Strong decode-time rate effects on a specific LIBERO-Spatial task, with a
  matched same-checkpoint effect from 34% to 68% or 72% depending the locked
  protocol. The effect is task-dependent and loses substantially on parts of
  LIBERO-Long.
- Strong language control of decoded action magnitude on CALVIN: quick versus
  plain is +16.6%, slow versus block-matched plain is -16.7%, and quick versus
  slow is +38.3%. This is not yet a measurement of physical end-effector speed.
- Strong RoboCasa prompt sensitivity on drawer and dishwasher-rack opposites for
  both waypoint and spline heads.
- A large single-seed granular-language gain on LIBERO-Long t0: 48% to 76%,
  paired exact p=0.00661 under the historical evaluator.

What is not established yet:

- Successful completion of an alternate RoboCasa goal under a counterfactual
  instruction.
- A spline-specific language-steering advantage over waypoints.
- A broad long-horizon gain: the same granular checkpoint is flat overall on
  LIBERO-Long and regresses on t5/t6; RoboCasa composite success is zero.
- A replicated ClangMixL10 result: the intended seed-1001 job actually used
  seed 1000.
- The claimed Hard-5 sweep or 26→42→52 monotonic co-scaling story under a locked,
  traceable protocol.

## Team context and overlap

Xiatao's requested validations are the right strategic questions:

1. Does language steering work on RoboCasa365?
2. Does granular language improve long-horizon execution?

The current answers are **partially** and **on one strong LIBERO task**, not yet
the earlier “maximally achieved” / “yes decisively” summary.

Quinten's target-slot experiment and the Misha steering battery overlap under the
same theme but are not duplicate experiments:

- Quinten tests whether a language-named object can override a memorized physical
  slot in `PickPlaceCounterToCabinet`. This is the stronger causal grounding test.
- Our battery changes fixture/direction/task instructions (drawer side, rack
  direction, cross-task prompts). Its reward still scores the original task, so
  it establishes sensitivity/avoidance rather than alternate-goal completion.

Quinten's supplied sequence is:

| Training condition | Slot 0.1 | Slot 0.3 | Absolute gap |
|---|---:|---:|---:|
| Original, confounded | 86.7% | 6.7% | 80.0 pp |
| 600 decorrelated episodes | 91.3% | 23.5% | 67.8 pp |
| 1,500 episodes, 35k steps | 63.2% | 23.8% | 39.4 pp |
| 1,500 episodes, 70k steps | 76.9% | 35.7% | 41.2 pp |

The disadvantaged slot improves substantially, but the latest gap remains 41.2
points and evaluation is roughly 20 trials per slot. More training improved both
slots without closing the gap. The next useful step is a balanced 100–200 trials
per slot with correct-target touch/grasp and distractor touch/grasp, plus a
waypoint head trained on the exact same decorrelated data—not another combined
data-volume-plus-scene-diversity run.

## Method state

The implementation is based on SmolVLA/LeRobot. The principal arms are:

- A: stock waypoint action chunks.
- B: fixed-time cubic B-spline control points.
- C: event-segmented cubic B-spline control points plus learned log duration.

For C, the decoder uses pinned cubic splines: the initial pose control point is
zero and the final point is the integrated segment endpoint. A predicted event
duration selects the segment; control points represent shape and duration
represents timing. The promoted LIBERO configuration is n=8 control points,
`horizon_max=24`, `min_seg=8`; canonical evaluation executes five actions per
replan. A canonically executes ten.

An important precision correction: exact endpoint reconstruction applies to the
unclipped spline target/reconstruction. The production decoder clamps pose
deltas, so executed trajectories need not preserve that endpoint when clipping
activates.

## Audited result tables

### Standard benchmark parity

Historical three-training-seed LIBERO suite means:

| Head | Mean success |
|---|---:|
| A waypoint | 82.1 ± 1.8% |
| B fixed-time spline | 80.8 ± 2.5% |
| C event+duration | 80.7 ± 1.3% |

CALVIN, three training seeds and 1,000 official chains per seed:

| Arm | Mean solved-chain length |
|---|---:|
| A | 1.56 ± 0.15 |
| C base segmentation | 1.31 ± 0.08 |
| C interval variant | 1.58 ± 0.11 |
| C uniform variant | 1.69 ± 0.08 |

### RoboCasa task-level instruction sensitivity

| Probe | A original→changed | C original→changed | Interpretation |
|---|---:|---:|---|
| OpenDrawer left↔right | 7/50→0/50 | 6/50→0/50 | prompt sensitivity |
| Rack in↔out | 18/50→0/50 | 15/50→0/50 | strong prompt sensitivity |
| Stove burner swap | 0/50→1/50 | 4/50→2/50 | inconclusive |
| Microwave→faucet | 2/50→0/50 | 0/50→0/50 | floor |
| Coffee→cabinet | 0/50→0/50 | 0/50→0/50 | floor |

Paired exact McNemar p-values are 7.6e-6 (A rack), 6.1e-5 (C rack),
0.0156 (A drawer), and 0.0313 (C drawer). Both heads respond similarly. The
kettle comparison is not paired: historical original rates use n=100 while the
changed-prompt run uses n=50.

### RoboCasa composite tasks

| Arm | Evaluated tasks | Success |
|---|---:|---:|
| A granular, scratch | 14 | 0/420 |
| A task-level, scratch | 6 | 0/180 |
| C granular, scratch | 15 | 1/450 |
| C task-level, scratch | 12 | 0/360 |
| C granular, atomic warm-start | 15 | **0/450** |
| C task-level, atomic warm-start | 0 | not evaluated |

The warm-granular result is complete, not pending. It failed. The 8,077-episode
dataset has 6,002,265 frames; 100k steps × batch 32 gives only about 0.53 dataset
epoch. Only 2,524 episodes (31.25%) were granularized. Several nominal composite
instructions remain one sentence, so the evaluator cannot stream meaningful
subtasks on those classes.

This result rejects the current recipe, not the possibility of transfer. A
16-task blind rerun is low value. The useful diagnostic is two best-progress
tasks with normalized subtask progress and a fully crossed A/C × compound ×
fixed-K × release-switch × simulator-oracle scheduler.

### LIBERO-Long granular language, historical evaluator

| Model | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cn8 baseline | 24 | 37 | 38 | 37 | 24 | 47 | 35 | 21 | 18 | 27 | 308/500 |
| ClangMixL10 | 38 | 35 | 46 | 37 | 29 | 37 | 21 | 22 | 11 | 24 | 300/500 |

- t0: +28 pp, paired p=0.00661.
- t4: +10 pp, p=0.458.
- t5: -20 pp, p=0.0129.
- t6: -28 pp, p=0.00936.
- Full suite: -1.6 pp, p=0.634.

The t0 effect is a compelling mechanism case study. The broad result is not yet
positive. In addition, the historical comparison used C-language at ten actions
per replan and the advertised C baseline at five, so language was confounded with
cadence. The new run explicitly crosses both cadences.

### Capacity and cadence

Cn8 LIBERO-Long seed totals are 61.6%, 58.8%, and 61.2%; Cn10 totals are 59.8%,
61.4%, and 63.0%; Cn12 currently has one seed at 60.6%. Long t7 rises from n8
seed-1000 42% to n12 seed-1000 78% (paired p=0.000912), but the intermediate n10
seed effects are heterogeneous and n12 is unreplicated.

For n16/h48, changing only evaluation cadence from five to twelve actions per
replan moves long t8 from 30% to 52% (paired p=0.0347, one seed), while spatial
drops from 81.8% to 77.4%. The old h24=26% source could not be traced, so the
claimed 26→42→52 capacity dose-response is not currently reproducible.

### Decode-time rate adaptation

The cleanest historical t5 spatial comparisons for Cn8 seed 1000 are:

- 19/50 at 1× versus 34/50 at 2× under the 600/1,200-step matched-duration
  budget, paired p=0.00408.
- 17/50 versus 36/50 under a 1,000/2,000-step budget, p=0.000311.
- A at 2× is 0/50.

The often-quoted n=150 pool repeats the same 50 initialization cases and includes
different horizon protocols; it is pseudoreplication. Full-suite spatial success
is approximately unchanged (77.8%→76.2%), but the selected long-horizon subset
drops from 59.5% to 29.5%. This is a strong task-dependent control capability,
not a universal performance improvement.

## Validity defects found in the audit

### Repaired for the new run

- The supposed Clang seed-1001 launcher passed both 1001 and 1000; saved config
  confirms seed 1000. Two genuinely new C seeds and three A-language seeds are
  training now.
- Historical LIBERO state order was outcome-dependent: `step()` internally
  resets on success and increments the state pointer, then the rollout driver
  resets again. New evaluation explicitly assigns and logs every state id.
- Policy flow-sampling RNG was not reseeded per episode. New evaluation keys
  Python/NumPy/Torch/CUDA seeds to task and state.
- Two independent smoke jobs produced exactly identical per-task success, step,
  and policy-call traces before the full fleet was launched.
- Historical language C@10 versus baseline C@5 cadence confounding is now a
  pre-registered 2×2 cadence control.
- New launchers use `set -euo pipefail`, refuse output overwrite, verify final
  checkpoints/results, and embed code/model/config/stat/data hashes.
- New result files are immutable/job-ID keyed and contain no video trees.

### Repaired locally for the next RoboCasa training round

- Production used event index `T=k`, raw gripper-value changes, and a full-window
  interpolated speed median; offline RoboCasa statistics used `k+1`, sign changes,
  and an episode-global nonzero-speed median.
- `event_targets.py` is now a shared source of truth used by production and the
  statistics generator. Six deterministic parity/boundary/padding tests pass.
- Statistics sampling no longer takes the first task-ordered episodes. Corrected
  atomic and composite artifacts were generated from all 9,126 and 8,077
  episodes, respectively, in an isolated directory. The mismatch is material:
  atomic log-duration standard deviation changes 0.2645→0.3674 and the median
  passthrough-control standard deviation is 2.47× the old value. Composite
  log-duration changes 2.8222±0.4482→2.8638±0.4631.

Existing RoboCasa checkpoints remain stale and are not repaired retroactively.
The new shared target module and statistics must be installed, hashed into each
checkpoint, and the affected policies retrained.

- New checkpoints now embed the validated statistics contract in `config.json`
  and save target mean/std as persistent model buffers. Loading validates the
  event convention, horizon, minimum segment, pause threshold, action layout,
  spline degree/control count, endpoint weighting, augmentation, shapes, and
  positive scales. Legacy checkpoints retain an explicitly warned fallback to
  their configured JSON; a partial or inconsistent embedded contract is a hard
  error.
- Molmo segmentation now delegates to the same production event function and
  version-stamps every new manifest. Existing `libero_l10gran` remains an
  immutable **legacy heuristic language-augmentation** artifact: exact
  event-aligned-language claims require re-extraction, re-captioning, rebuilding,
  and retraining under a new dataset identity.

### Realized `libero_l10gran` intervention

CPU audit job 2319212 scanned every frame and found:

| Quantity | Audited value |
|---|---:|
| Episodes | 1,693 |
| Frames | 273,465 |
| Episodes with any changed language frame | 831 (49.1%) |
| Frames with an appended granular label | 86,148 (31.5%) |
| Appended caption strings in task table | 1,385 |
| Appended captions actually used | 975 |
| Used captions occurring in only one contiguous run | 481 |

All ten LIBERO-Long tasks (IDs 0–9) and all ten LIBERO-Goal tasks (20–29)
receive changed frames; Object and Spatial remain untouched. Every Long episode
is affected, with 43.2–55.5% of its frames relabeled. The derived episode
metadata was not recomputed: all 831 affected episodes still report an
original-only task-index maximum. Training consumes frame-level `task_index`, so
this does not nullify the intervention, but consumers using episode statistics
must not treat that metadata as authoritative.

The boundary migration audit further shows that this intervention is not exact
production-event alignment. Across the 3,599 surviving captioned intervals,
only 14/304 episode partitions (4.61%) match exactly; 2,579/3,599 intervals
(71.66%) require recaptioning under the corrected boundary contract. The median
nearest-boundary drift is one action step and p95 is nine. This does not
invalidate the generic aligned-vs-task-level language factorial, but it requires
the paper to call the current artifact legacy heuristic segmentation.

A timing-matched semantic negative control is now CPU-audited: a same-task-
support caption derangement can change 79,971/86,148 appended frames (92.83%)
without changing any frame, task-index schedule, switch time, segment length,
caption vocabulary, or frequency multiset. The exact policy tokenizer preserves
all 966/966 mapped distinctions with no 48-token truncation. A zero-copy overlay
build is therefore safe; policy training remains conditional on the primary
factorial rather than consuming GPUs speculatively.

### Still open

- Existing checkpoints still have non-persistent target buffers and depend on a
  mutable source-tree JSON; only checkpoints saved after the local persistence
  repair are self-contained.
- `rc_data_quality.py` still uses a legacy event approximation. Existing Molmo
  labels also retain legacy boundaries even though the extractor is repaired for
  future artifacts.
- The granular RoboCasa sentence-to-release alignment is heuristic; release does
  not prove subgoal completion.
- The A/C RoboCasa scheduler comparison was not crossed fairly; self-paced logic
  works for any head but was primarily assigned to C.
- Current RoboCasa evaluator does not expose the official unseen-composition
  split.
- Warm A launchers point to a C checkpoint and are invalid as an A control.
- Historical result poolers concatenate duplicate initial states and top-ups.
  Hard-5 n≈80–210 is not independent n.
- The original pre-registered Hard-5 set contained t6; the later pivot table
  substituted t8. “All five pre-registered tasks swept” is incorrect and also
  selects post hoc best levers per task.

## In-flight causal experiment

The missing experiment is the head × language interaction:

`(C_granular - C_standard) - (A_granular - A_standard)`.

The overnight diagnostic design is:

| Head | Standard data | Identical language-mix data |
|---|---|---|
| A waypoint | existing seeds 1000/1001/1002 | new seeds 1000/1001/1002 |
| C event spline | existing seeds 1000/1001/1002 | existing 1000 + new 1001/1002 |

Every checkpoint is evaluated on LIBERO-Long task IDs 0–9, explicit initial
states 0–49, task/state policy RNG, and one evaluator hardware class. The
architecture interaction is cadence-matched at A@10/C@10; C@5 is a deployment
secondary. The primary result is the three-seed interaction; t0/t4 are pre-specified
mechanism tasks, and suite noninferiority prevents cherry-picking gains bought by
broad regressions. Per-task p-values receive Holm correction. Repeated top-ups
are not pooled.

This round is a go/no-go diagnostic, not yet the claim-bearing factorial.
Historical baselines lack job-start manifests; historical C was trained before
later untracked source/config changes, and base/language training GPU classes
differ while cuDNN determinism was disabled. If the diagnostic interaction is
positive, seven fresh, source-snapshotted/hardware-matched trainings are needed:
three A bases on H100, three C bases on L40S, and C-language seed 1000 on L40S.
The exact active source tree and language dataset metadata were frozen during
training under `code_snapshots/language_training_active_20260820_v2` before any
installed policy changes.

Stop/go decisions:

- Positive replicated interaction with suite noninferiority: language/event
  alignment can be a headline.
- Similar language gains for A and C: retain language augmentation as a generic
  training result and drop the spline-specific language claim.
- No replicated language gain: retain t0 as exploratory failure-directed evidence
  and make explicit duration/event timing the core causal story.
- Fixed or shuffled duration matching learned duration in the next ablation:
  remove learned duration from the headline.

Exact job IDs and the locked pre-registration are in
`docs/OVERNIGHT_EXECUTION_2026-08-21.md`.

## Novelty-safe positioning

The broad components now have close prior art: B-spline action tokens, continuous
resampling, decode-time retiming, event/keyframe abstraction, and hierarchical
language steering. Relevant collisions include [BEAST](https://arxiv.org/abs/2506.06072),
[Spline Policy](https://arxiv.org/abs/2606.07386),
[B-spline Policy](https://arxiv.org/abs/2607.09648),
[ABPolicy](https://arxiv.org/abs/2602.23901),
[StaKe](https://arxiv.org/abs/2606.26801),
[SparkVLA](https://arxiv.org/abs/2608.16172),
[RT-H](https://arxiv.org/abs/2403.01823), and
[π0.7](https://arxiv.org/abs/2604.15483).

Do not claim first B-spline VLA, first continuous-rate spline controller, first
decode-time speed control, first language-steerable VLA, or first counterfactual
language evaluation. A credible title direction is:

> **Event-Delimited Spline-Duration Action Primitives for Counterfactually
> Steerable Vision-Language-Action Policies**

The next architecture-specific experiment after the factorial is a duration
mechanism ablation on one checkpoint: predicted duration, a fixed training-target
median, and within-episode shuffled predicted durations. A simulator-oracle arm
is rejected because the production event target depends on the demonstrator's
future actions and would leak unavailable future information. This experiment
is currently blocked: a supposedly no-op duration hook diverged on 2/10 smoke
trajectories from the canonical evaluator. It must pass exact deterministic
identity before any full run is interpretable.

## Misha operating rules and current storage risk

Official guidance used for the overnight run:

- Do not compute or run substantive transfers on login nodes; submit through
  Slurm. Use `gpu` for production and `gpu_devel` only for short debugging.
- Misha permits at most 200 submissions/hour. Use accurate CPU, memory, GPU, and
  walltime requests; short jobs help backfill and fairshare.
- Idle GPU jobs can be cancelled automatically. Inspect jobs with `jobstats` and
  verify outputs/checkpoints rather than trusting Slurm `COMPLETED` alone.
- Home is backed up; project and scratch are not. Scratch files inactive for 60
  days may be purged, and artificial timestamp extension is prohibited.

Official sources: [Misha cluster](https://docs.ycrc.yale.edu/clusters/misha/),
[accounts and login-node rules](https://docs.ycrc.yale.edu/clusters-at-yale/access/accounts/),
[storage](https://docs.ycrc.yale.edu/data/hpc-storage/),
[job scheduling](https://docs.ycrc.yale.edu/clusters-at-yale/job-scheduling/),
[resource requests](https://docs.ycrc.yale.edu/clusters-at-yale/job-scheduling/resource-requests/),
and [cluster policies](https://docs.ycrc.yale.edu/clusters-at-yale/policies/).

Quota snapshot from Aug. 20:

| Fileset | Space | File count |
|---|---:|---:|
| User home | 48/125 GiB | 306k/500k |
| Group project | 3,529/4,096 GiB | 4,959,463/5,000,000 |
| Group scratch | 6,770/10,240 GiB | 14,970,415/15,000,000 |

Project and scratch are critically close to file-count limits. The new evaluator
writes one compact JSON per checkpoint and no videos. Do not clone another large
dataset tree, generate a video-per-episode fleet, or build a new hardlink mirror
without first coordinating cleanup with the group. No existing data is deleted
as part of this audit.

## Immediate roadmap after the factorial

1. Analyze the locked diagnostic factorial and make the spline-specific language
   hypothesis pass or fail on the cadence-matched interaction—not a selected
   task. If promising, run the fresh provenance/hardware-matched baseline round.
2. Treat LIBERO rollouts as stochastic: even the single-process identity gate
   diverged after 120 identical actions. Run predicted/fixed/shuffled only in
   repeated, counterbalanced within-state blocks with a measured repeatability
   baseline and hierarchical inference. There is no online oracle.
3. Strengthen Quinten's target-grounding study with balanced high-n target and
   distractor contact/grasp metrics plus a matched A head.
4. Evaluate matched checkpoints on LIBERO-CF / perturbation suites if storage and
   environment integration can be done without a large clone.
5. Replace RoboCasa binary-only composite evaluation with subtask progress and
   oracle scheduling on two failure-directed tasks before another broad fleet.
6. Make normalization checkpoint-contained and rebuild Molmo/granular boundaries
   from the shared production event function.
7. Only then port the validated recipe to π0.5/GR00T. A backbone port alone does
   not restore novelty because spline π0.5 prior work already exists.

The ICRA 2027 submission deadline is September 15, 2026. The remaining time
should be spent on one clean causal story, not additional disconnected sweeps.
