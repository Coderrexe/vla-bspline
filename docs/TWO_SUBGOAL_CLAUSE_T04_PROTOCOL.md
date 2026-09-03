# Exact-task two-subgoal clause protocol (locked before outcomes)

This is the replacement for the rejected Molmo 12-phase artifact. Job 2321848
failed its unchanged temporal gate (17.2% adjacent inversions and 16.45% gross
inversions); that artifact and every dependent build remain excluded.

## Task identity correction

LIBERO dataset `task_index` is not the LIBERO suite evaluator task ID. The
dataset is therefore resolved only by exact evaluator task description:

| Suite task | Exact evaluator description | Source dataset task | Episodes | Frames |
|---|---|---:|---:|---:|
| t0 | put both the alphabet soup and the tomato sauce in the basket | 5 | 33 | 9,571 |
| t4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 0 | 38 | 9,807 |

The earlier numeric selection `dataset task_index in {0,4}` contained the mug
task and the different alphabet-soup/cream-cheese task. Its 81 episodes are not
the evaluator t0/t4 pair and cannot enter the replacement experiment. The exact
task-text selection contains 71 episodes and 19,378 frames. A canonical mapping
hash is required in every dataset, training, and evaluation manifest.

## Clause construction and switch

The reviewed deterministic decompositions are:

- t0: “put the alphabet soup in the basket”; then “put the tomato sauce in the basket”.
- t4: “put the white mug on the left plate”; then “put the yellow and white mug on the right plate”.

For a source demonstration, a release candidate is an action index `r` with
`grip[r-1] > 0` and `grip[r] <= 0`. A valid first-subgoal release must follow at
least 48 consecutive closed commands (two 24-action training horizons), which
rejects short failed-grasp retries. The release action remains under clause 1;
the sole switch is at `r+1`. Every episode must have exactly two nonempty,
gap-free intervals, exactly one language-ID transition, and at least eight
frames on each side.

The builder creates a paired-original root and a clause root from one source
snapshot. It changes only language metadata and `task_index`; all selected
non-language Arrow columns must compare equal and have identical deterministic
IPC SHA-256 digests after the clause Parquet round trip. Outputs refuse
overwrite and carry the builder, source files, labels, mapping, and verification
hashes in immutable provenance.

## Matched warm factorial

Train A and C crossed with paired-original and clause conditioning: four cells,
all seed 1000, batch 32, 7,500 optimizer steps, the same 71 episode manifest,
the same frozen `language_source_20260820_v2` source tar, and one H100 per job.
A starts from `baselineA_100k`; C starts from `Cn8_100k`. C retains n=8,
`min_seg=8`, `horizon_max=24`, and its frozen stats artifact. No cell may launch
unless the mapping, coverage, one-switch, and physical-column gates pass.

## Locked evaluation

Each of the four checkpoints is evaluated with `n_action_steps=10` on suite t0
and t4, initial states 0..49, seed base 100000, control frequency 20 Hz, and
OSMesa. The three predeclared modes are:

- Static: original compound benchmark instruction for the full rollout.
- Fixed: the two clauses, switching before control step 139 for t0 and 107 for
  t4. These are the half-up-rounded median causal switch frames from the source
  labels (t0 median 139.0; t4 median 106.5).
- Event: the two clauses, switching immediately after the executed trajectory's
  first close-to-open gripper command and clearing queued actions from clause 1.

Primary comparisons are clause-minus-original within head and mode. The A/C
interaction is secondary. Exact paired state/seed outcomes are reported with
paired McNemar intervals; no engineering smoke or mislabeled 81-episode result
can be pooled into this table.
