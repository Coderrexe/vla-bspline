# Decode-Time Duration Mechanism Ablation

Status: **implemented and unit-tested; identity validation is not yet passed.**
The code was staged in isolation on Misha without changing the evaluator used
by the running language jobs. A 10-episode predicted/no-op smoke (job 2319599)
matched the canonical run on eight tasks but diverged on two. Full ablations are
blocked until independent canonical repeats determine and remove the cause.

The stronger same-allocation test identified the cause class: in job 2319617,
two canonical v1 processes on the same L40S UUID produced task-2 completion at
240 versus 251 steps. The no-op hook is therefore not uniquely responsible;
v1 itself lacks trajectory replay. Deterministic-v2 job 2319625 is the new gate.
It uses task 2, seeds before environment/policy construction and again after
each reset, requires deterministic kernels, disables TF32 and fused SDPA paths,
pins CPU thread counts, and compares exact executed-action SHA-256 traces across
two sequential reloads on one physical GPU.

That gate failed: deterministic-v2 job 2319625 produced task-2 success at 244
versus 241 steps, and its full action traces differ. This rules out simple
Torch/cuDNN flag drift but not constructor-time environment/model randomness.
The next implementation therefore runs the predicted repeat and all duration
arms in one process with one loaded policy/environment, explicitly resets and
reseeds each state, and records initial-observation and per-step action hashes.

## Question and estimand

For a fixed trained C checkpoint, does aligning its decoded execution horizon
with its learned, observation-conditioned duration prediction improve
closed-loop task success?

The primary estimand is the paired success-rate difference on the same LIBERO
task/state manifest between:

1. `predicted`: the production learned duration;
2. `fixed`: every primitive is decoded at the **global median training-target
   duration**, computed before evaluation using the checkpoint's exact event
   rule; and
3. `shuffled`: the predicted run's duration multiset for the same task and
   initial state, deterministically permuted across policy calls.

`fixed` asks whether a constant time scale is enough. `shuffled` is the sharper
mechanism control: it preserves the per-episode marginal duration distribution
while destroying alignment between duration and the current action primitive.
It uses a seeded Sattolo cycle, so no call index remains fixed when the donor
trace has length greater than one. Repeated duration values can still weaken the
effective intervention, so every result reports the changed-call fraction and
mean absolute duration change; a null result is not accepted as mechanistic
evidence when that first-stage intervention is weak.
The intervention patches only `_predicted_T`; the sampled pose and gripper
control-point tokens are untouched. After the first altered action, the robot's
state can diverge, so the reported contrast is correctly interpreted as the
closed-loop **total effect**, not a per-step open-loop effect.

## Why there is no online oracle arm

The training target is the first gripper/pause/episode-end event in the
demonstrator's *future action window*. During a rollout, the corresponding
future event depends on actions the evaluated policy has not generated yet.
Simulator object state does not reveal that counterfactual time-to-event. An
"oracle" duration would therefore leak future actions or require a different
controller, invalidating the same-policy causal comparison. Oracle-duration
analysis belongs only in an explicitly labeled open-loop reconstruction study.

## Locked protocol

- Same model and config hashes in every arm; no checkpoint config edit.
- Same 50 explicit initial states per LIBERO-Long task (500 episodes/arm).
- Same simulator and Torch/NumPy/Python seeds per task/state.
- Native `n_action_steps` from the checkpoint (C uses 5 in the current matched
  factorial); 20 Hz control rate.
- Optional duration-consuming decode controllers must be neutral:
  speedup/slowdown, ease-out, profile retiming, self-paced replanning,
  feasibility stretch, and rate retargeting. The evaluator fails closed if any
  is active. Duration snap, if present in the checkpoint, remains part of the
  learned-duration mapping and is applied before all interventions.
- The fixed value must come from a complete training-split scan using
  `scripts/analysis/duration_training_prior.py`. Smoke/partial manifests are
  rejected.
- Run `predicted` first. `shuffled` accepts only a predicted result with exactly
  matching model/config hashes, suite, state list, and seed base.
- Every result records raw predicted and actually executed duration traces,
  intervention-source hashes, policy/environment/evaluator/controller hashes,
  and counterfactual trace reuse.

The shuffle uses the same state's predicted duration list and a stable seeded
permutation. If an intervened rollout needs more policy calls than its source,
the list cycles and every reuse is counted. Report reuse frequency; a material
rate is a limitation and should trigger a longer preregistered donor trace.

## Analysis and decision rule

For each checkpoint seed, report the arm totals and paired discordant counts.
Use exact two-sided McNemar tests for `predicted` versus `fixed` and `predicted`
versus `shuffled`, with Holm correction across the two primary contrasts.
Report task-wise paired deltas and a task-stratified bootstrap confidence
interval; do not pool repeated states as independent evidence for a task-level
claim.

Repeat all three arms on the three C seeds. The architecture-mechanism claim is
supported only if predicted duration beats both controls in the same direction
on at least two seeds and the hierarchical seed/task/state interval excludes
zero. One seed is a pilot, not paper evidence. Also report episode steps and
policy calls, but success is primary; conditional completion time is vulnerable
to survivorship bias.

Interpretation is intentionally falsifiable:

- `predicted > fixed` and `predicted > shuffled`: observation-aligned duration
  is causally useful.
- `predicted ≈ fixed`, `predicted > shuffled`: duration's task/episode scale is
  useful, but fine call-level allocation is not established.
- `predicted ≈ shuffled`: remove learned duration as a success mechanism claim;
  retain only representation/calibration claims supported elsewhere.
- `fixed > predicted`: the learned head is miscalibrated; prioritize calibrated
  decoding or a duration loss before additional capability experiments.

## Files and proposed launch order

- `scripts/eval/duration_intervention.py`: isolated, traceable policy hook.
- `scripts/eval/libero_duration_ablation.py`: locked evaluator; does not import
  or alter any running cluster job.
- `scripts/analysis/duration_training_prior.py`: exact training-target prior.
- `scripts/analysis/duration_prior_parity.py`: explicit-anchor parity smoke
  against LeRobot's real action-query helpers, with video decoding bypassed.
- `cluster/eval_duration_ablation.sbatch`: reviewed proposal, marked not to run
  yet.
- `tests/test_duration_intervention.py`: intervention contract tests.

The action-only CPU prior scan is job 2319621. It reads only Parquet action/task
columns and recreates every future-action training window without image decode.
It completed all 273,465 frames in 13 seconds. The global discrete median is
`T=24`, the checkpoint's horizon cap; the complete artifact SHA-256 is
`c75f6f7e730d...1ccf28`.

The direct scanner now rejects any episode whose `frame_index` does not start at
zero and advance contiguously. This is a correctness precondition: otherwise a
row offset would not identify the same anchor as LeRobot's frame index. Before
promoting the prior artifact, run `duration_prior_parity.py` on a small explicit
set that includes the first, an interior, and the final/H-near-final frames of
at least two episodes. The pinned LeRobot revision has no public action-only
window API: `dataset[idx]` also decodes every camera. The smoke therefore calls
the exact internal `_get_query_indices` and `_query_hf_dataset` helpers used by
`__getitem__`, verifies the resolved offsets are `0..H-1`, and fails closed if
those APIs change. It compares action tensors and `action_is_pad` bit-for-bit,
then compares the pinned event target. No full-image decode fallback is allowed.
CPU job 2319637 passed this gate on eight explicit anchors spanning
episode starts, interiors, near-ends, and final frames. Actions, padding masks,
and event targets matched bit-for-bit; video decode was never called. The
parity artifact SHA-256 is `dd02378e2020...cccddb`, bound to action schedule
`fcaba294...`, event kernel `9335cae5...`, checkpoint config `0f83c900...`, and
LeRobot revision `192a0b92...`.
After exact identity and the language factorial are completely harvested,
launch `predicted` and `fixed` concurrently per checkpoint; launch
`shuffled` with `afterok` on that checkpoint's predicted job and its exact
output path. Start with one seed as a 50-state pilot only if queue pressure is
high, then promote unchanged code and state manifest to all three seeds.

## Limitations

This ablation isolates the value of the duration signal at decode time, not the
training effect of supervising duration or event-aligned chunk construction.
All modes use a model trained with both. That separate estimand requires a
matched training ablation (event-aligned shape with the duration loss masked),
which is substantially more expensive. The shuffled arm also preserves the
baseline episode's duration multiset, not necessarily the multiset that the
policy would predict along the counterfactual trajectory; that is deliberate
for a replay-based alignment control and must be described as such.
