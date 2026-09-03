# Duration Mechanism Ablation v3: Stochastic Paired Blocks

Status: **local-only proposal; not synced or submitted.**

## Why exact replay is no longer an assumption

Same-process v2 smoke job 2319737 reset task 2/state 0 to identical raw and
processed initial observations and produced identical actions through step 119.
The first action difference appeared at step 120 and the first duration-trace
difference at chunk 29; both rollouts succeeded, in 239 versus 244 steps. Thus
strict CUDA settings, matched seeds, one loaded policy, and one environment do
not make this stack bitwise replay-deterministic.

V3 treats that variability as part of the experiment. Every output is visibly
watermarked `STOCHASTIC_PAIRED_BLOCK_RESULT`, explicitly says bitwise replay is
not assumed, and forbids episode-level IID inference.

## Block design

Each task/state/repeat block runs five episodes in one process with the same
loaded policy, environment, explicit initial state, and derived block seed:

1. `predicted_donor`, first, trace-only;
2. all three efficacy arms—`predicted_eval`, `fixed`, and `shuffled`—in one of
   their six possible orders;
3. `predicted_closure`, last, trace-only.

Within every task, a preregistered randomization seed assigns exact quotas over
all six efficacy permutations, then shuffles those assignments across
state/repeat blocks. Six or 12 blocks per task give exactly equal counts; 50
differs by at most one. The complete task-local order manifest is materialized
and hashed before any rollout. It does not depend on checkpoint seed, so the
same manifest is shared across independently trained models. Donor and closure
are never efficacy observations; the primary predicted result is
`predicted_eval`, and donor supplies the in-process shuffle trace.

Block seeds are SHA-256-derived from seed base, task, state, and repeat. Each
block receives a valid unsigned-32-bit simulator seed and an independent
unsigned-64-bit policy seed; all five episodes share that pair. The evaluator
rejects collisions in either seed space before rollout.

## Intervention and provenance

The isolated v3 controller saves the policy's bound original `_predicted_T` and
calls it once, with the current tokens, in every arm and rollout. This preserves
the deployed recomputation exactly while giving every arm identical wrapper
overhead. Only after recording that canonical scalar does the wrapper return
predicted, fixed, or shuffled duration. All controller state is cleared between
rollouts. `fixed` uses the config-bound complete training prior (`T=24` in the
current artifact); `shuffled` uses a Sattolo permutation of the donor trace.

Every shuffle-source manifest binds checkpoint path, model/config hashes,
suite, task/state/repeat, simulator and policy seeds, control frequency, donor
duration trace, and donor action digest. It is canonically hashed and embedded
in the block. Each shuffled episode reports changed-call fraction, mean absolute
duration change, values unused because the counterfactual produced fewer
chunks, values reused because it produced more chunks, and whether the executed
shuffle retained the donor's exact duration multiset. The predeclared shuffle
strength target is at least 50% effectively changed calls.

Optional duration-consuming systems—selective speed changes, retiming,
self-paced replanning, velocity chaining, feasibility stretch, and rate
retargeting—remain fail-closed so the intervention is not bundled with another
decode controller.

## Divergence telemetry

For every episode, v3 records hashes of:

- raw and fully processed observations at every environment step;
- MuJoCo `mjSTATE_INTEGRATION` at every step (including warm-start, controls,
  applied forces, mocap/history, and plugin state under the installed binding),
  plus compiled model XML once per reset;
- raw, processed, and MuJoCo-state inputs at every policy replan/chunk
  generation, joined to canonical predicted/executed duration scalars and given
  a record hash;
- each exact action array and every cumulative action prefix; and
- the final exact action trace.

It also stores every predicted/executed duration by chunk, arm order, calls,
steps, success, initial-state pairing, and all source/code/checkpoint hashes.
Every atomic block also binds the complete process-runtime manifest that
produced it (determinism settings, interpreter, arguments, Slurm job, and
node). A resumed aggregate retains all distinct runtime manifests rather than
incorrectly attributing earlier blocks to the final allocation.
All three predicted rollouts are compared pairwise (donor/eval,
donor/closure, eval/closure). Diagnostics report outcome agreement, the four
outcome pairs, and first differing MuJoCo state, raw observation, processed
observation, replan input, action, action prefix, and predicted-duration chunk.
These are drift diagnostics, not extra efficacy samples. Telemetry is named
`mujoco_integration_state` rather than “full simulator state”; the exact helper
protocol and compiled-model hash are recorded without claiming unexposed Python
controller internals.

## Atomic block resume and preregistered gates

Five rollouts are indivisible. A block is published with a no-clobber atomic
write only after donor, all efficacy arms, closure, and diagnostics complete. A
preemption inside a block leaves no block artifact, so resume restarts that
whole block. Completed blocks may be reused only when checkpoint, model/config,
fixed prior, task/order manifest, seeds, thresholds, and all source hashes match.

The pilot records three eligibility targets established before outcomes:

- every task's six-permutation quota counts differ by at most one;
- shuffled calls change duration at least 50% of the time overall and within
  every task; and
- the three predicted-position marginal success rates span at most 10
  percentage points both overall and within every task for the descriptive
  pilot.

The last is an observed pilot screen, not proof of equivalence. Confirmatory
repeatability requires 95% hierarchical paired confidence intervals for every
predicted-position contrast to fall within ±5 percentage points.

## Pilot and inference

The default pilot is 12 states per task and one repeat: 120 blocks and 600
episodes across LIBERO-Long. Task subsets, state ranges, and repeat counts are
supported. The proposed launcher is:

```bash
sbatch cluster/eval_duration_ablation_v3.sbatch \
  /path/to/pretrained_model TAG /path/to/complete_duration_prior.json
```

After a preemption or wall-time exit, resubmit with the exact same tag and
arguments. The stable tag-specific block directory resumes only complete,
identity-matched blocks; the interrupted five-rollout block runs again from its
donor. Never run two allocations with the same tag concurrently.

For a small integration smoke after review:

```bash
TASK_IDS=2 STATES_PER_TASK=1 REPEATS=6 sbatch \
  cluster/eval_duration_ablation_v3.sbatch \
  /path/to/pretrained_model t2_v3_engineering /path/to/complete_duration_prior.json
```

The scientific comparison uses only `predicted_eval`, `fixed`, and `shuffled`.
Report per-task block differences and fit an arm effect that includes efficacy
period/order while respecting repeats nested within state and states within
task. A task/state/repeat hierarchical bootstrap or a preregistered mixed model
is appropriate; treating the five episodes in a block or all simulator episodes
as independent is not. All three predicted pairwise agreement and
first-divergence summaries must accompany the result as reliability diagnostics.

A paper-level mechanism claim requires independent checkpoint seeds and a
consistent predicted advantage over both controls under hierarchical
state/repeat inference. The 12-state, one-repeat run is a pilot for effect size
and variance, not confirmatory evidence. The engineering gate uses one state
with six repeats (30 rollouts), exercising every efficacy order exactly once.

## Engineering result (job 2320368)

The six-block task-2/state-0 integration run completed all 30 rollouts. All 24
donor-to-other-arm initial comparisons matched seeds, raw and processed
observations, compiled model XML, and MuJoCo integration state. Shuffle changed
77.6% of duration calls by 5.92 steps on average. Outcomes were donor 3/6,
efficacy-predicted 2/6, fixed 5/6, shuffled 5/6, and closure 3/6. The apparent
+50-point fixed/shuffled effects are useful pilot signal but not inference: this
state was selected for instability and n=6 gives paired two-sided p=0.25.

Donor and closure agreed on success in 6/6 blocks and replayed exact actions in
4/6. Across donor versus efficacy-predicted, observation hashes diverged at a
median step 101.5, actions at 120, and MuJoCo integration state at 121. One
donor/closure pair retained identical integration state and exact actions for
the full episode even though observations first differed at step 107. Thus the
remaining stochasticity is upstream in the rendered observation path, not an
initial-state or duration-controller mismatch. The immutable combined artifact
SHA-256 is `86ab7346...a1502`.

The first software-renderer attempt failed before environment creation because
the conda runtime did not expose `libOSMesa`. Loading Yale's provided
Mesa/22.2.4 module fixed that without changing the environment. Job 2320402 then
made all three predicted positions exactly identical for every raw/processed
component, integration state, duration, action, and all 245 steps. Its artifact
SHA-256 is `e3c48f56...f4f80a`. This strongly attributes the EGL variation to
the rendering path and restores exact paired evaluation by changing only the
render backend.

The pending EGL pilot was cancelled before running. OSMesa six-block exact gate
2320438 now precedes two dependency-gated 300-rollout task shards, 2320468
(tasks 0--4) and 2320469 (tasks 5--9), which together form the unchanged
600-rollout all-task pilot while halving wall time. They use the immutable source snapshot
`duration_v3_osmesa_release_20260821` and are configured to publish diagnostics
then fail if any predicted pair diverges. The full pilot remains exploratory;
the selected engineering-cell effect cannot be quoted as general performance.

## Claim-safe analysis

Analyze only completed combined v3 JSON artifacts, never loose block files:

```bash
python3 scripts/analysis/duration_v3_analysis.py \
  /path/to/duration_v3_combined.json \
  --out-prefix outputs/audits/duration_v3_pilot
```

The analyzer fails closed on schema/protocol drift, inconsistent source,
checkpoint/config/prior identities, incomplete or duplicate block grids,
task/order-manifest mismatches, failed initial pairing, weak shuffles, and a
failed exact-replay requirement. It recomputes the primary paired contrasts
(`predicted_eval` versus fixed and shuffled), exact two-sided McNemar tests with
Holm correction, per-task effects, efficacy-period/order sensitivity, and all
donor/eval/closure repeatability diagnostics. For one checkpoint it reports a
block-preserving task/state bootstrap strictly as exploratory uncertainty. It
does not produce a training-seed confidence interval until at least three
distinct, identity-matched model hashes are supplied. JSON and Markdown are
published as an atomic, no-overwrite pair.
