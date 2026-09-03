# Duration Mechanism Ablation v2: Single-Process Matched Experiment

Status: **local-only proposal; not synced or submitted.**

## Why v2 is a single process

Deterministic-kernel evaluator job 2319625 still produced different task-2
trajectories in two processes on the same L40S (244 versus 241 steps, with
different exact action digests). A causal comparison made from separate jobs is
therefore inadmissible even when checkpoint, state IDs, RNG seeds, GPU, CUDA
settings, and deterministic Torch flags match.

The v2 experiment loads one policy and one LIBERO environment set once. It then
executes four complete passes in this order:

1. `predicted`: production learned duration;
2. `predicted_repeat`: the same scientific mode, used only as an exact replay
   gate;
3. `fixed`: global median training-target duration from a complete manifest;
4. `shuffled`: an in-process Sattolo permutation of each matched predicted
   episode's duration trace.

Fixed and shuffled do not run unless the two predicted passes match exactly for
every task/state episode. A failed gate still publishes an immutable diagnostic
artifact and exits nonzero.

Before admitting each intervention arm, a second gate requires its task/state
seeds plus raw and processed initial-observation hashes to exactly match the
first predicted pass. If fixed fails this gate, shuffled is not run; if either
intervention fails it, the combined artifact is diagnostic-only and exits
nonzero.

The evaluator also fails closed when optional duration-consuming decode systems
are active, including selective speed/slowdown, ease-out or profile retiming,
self-paced replanning, velocity chaining, feasibility stretching, or rate
retargeting. Duration snapping remains inside the common learned prediction
mapping and is therefore shared by all arms before intervention.

## Intervention invariant

All three scientific modes patch only `_predicted_T`. The patched function does
not call `_predicted_T_batch`. Instead it consumes the batch-1
`policy.last_predicted_T_batch` that production `_get_action_chunk` computed
immediately beforehand, takes its median, and records it. Only after that common
operation does it return the predicted, fixed, or shuffled scalar. Reusing the
same prediction object twice or evaluating a batch other than one fails closed.

This matters because v1 redundantly called `_predicted_T_batch` inside the hook.
Although that operation should be algebraically deterministic, it was an
unnecessary path difference in a mechanism experiment.

## Exact replay gate and localization

Every episode records:

- hashes of the raw reset observation and the first fully processed policy
  input;
- the exact action-array SHA-256 at every environment step;
- the cumulative action-prefix SHA-256 after every step;
- the final exact action digest, success, steps, select-action calls, and chunk
  generations; and
- predicted and executed duration for every generated chunk.

The gate compares all of these fields. If it fails, the result identifies the
first different action step and prefix and says whether divergence was already
present in the raw or processed initial observation. This distinguishes reset,
preprocessing, and policy/execution divergence. The complete four-pass artifact
is larger than prior scalar-only JSON, but remains one file and contains no
images or videos.

## Fixed and shuffled source binding

`fixed` requires a manifest with `complete_training_scan=true`, equal positive
expected/scanned frame counts, the exact checkpoint-config hash, action-schedule
hash, event-target-source hash, and integer global duration. The current
complete prior is `T=24`; its reported artifact hash is
`c75f6f7e730d...1ccf28`, but the launcher never trusts a filename or hard-coded
value and revalidates the full manifest.

The shuffle source is constructed only after the in-process gate passes. Its
embedded manifest binds the resolved checkpoint, model and config hashes,
suite, task IDs, state IDs, environment and policy seeds, control frequency,
and every predicted chunk trace. The manifest receives a canonical JSON hash
and is stored in the final artifact. There is no external shuffle-source path
that could accidentally mix experiments.

Changing durations changes subsequent observations, so fixed and shuffled are
closed-loop total-effect interventions. The shuffled sequence cycles only if a
counterfactual rollout generates more chunks than its matched predicted source;
reuse count, changed-call fraction, and mean absolute intervention are reported.

## Proposed launch

The isolated launcher is `cluster/eval_duration_ablation_v2.sbatch`:

```bash
sbatch cluster/eval_duration_ablation_v2.sbatch \
  /path/to/pretrained_model TAG /path/to/complete_duration_prior.json
```

`TASK_IDS`, `STATE_START`, `STATES_PER_TASK`, and `SEED_BASE` allow a scoped
smoke or full manifest. The default is all suite tasks and 50 states per task;
the 16-hour request covers all four passes on one L40S without fragmenting the
matched block across jobs. The launcher stages immutable copies of the v2
evaluator, controller, canonical deterministic evaluator, and hashing helper,
then records their SHA-256 values. It is explicitly marked local-only and must
not be synced or submitted until code review.

After review, the first launch should target the previously divergent cell:

```bash
TASK_IDS=2 STATES_PER_TASK=1 sbatch \
  cluster/eval_duration_ablation_v2.sbatch \
  /path/to/pretrained_model t2_s0_gate /path/to/complete_duration_prior.json
```

Only if that same-process gate passes should the unchanged sources be promoted
to all ten tasks and 50 states. Passing the smoke establishes the evaluator
path, not the scientific result.

## Decision rule

Only an artifact with `status=complete` and an exact replay gate is analyzed.
For each checkpoint seed, compare predicted with fixed and shuffled using the
same task/state pairs; report discordant counts, exact McNemar tests, task-wise
deltas, and hierarchical seed/task/state intervals. The mechanism claim needs
the same direction against both controls on at least two of three independently
trained C seeds. A weak shuffled first stage (low changed-call fraction or high
source reuse) cannot support a null mechanistic conclusion.

## Empirical verdict (job 2319737)

The required task-2/state-0 smoke did **not** pass. In a single process with one
loaded policy and environment set, the two predicted rollouts had identical raw
and processed initial observations and exactly identical actions for steps
0--119. They then diverged at action step 120 and completed successfully at 239
and 244 steps; duration predictions first differed later, at chunk 29. The
artifact SHA-256 is `d88db01b8e02...d268ba`.

Accordingly, v2 stops here and no full v2 intervention may be launched. This is
evidence of residual closed-loop simulator/observation variability after an
identical prefix, not a license to ignore the gate. A successor must estimate
that stochastic variation directly with repeated, counterbalanced within-state
blocks and use hierarchical inference rather than exact paired-trajectory
language.
