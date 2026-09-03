# LIBERO-Long compositional holdout protocol

## Question

Does representing a long-horizon instruction as executable language clauses
help a policy recombine individually learned subgoals into a task composition
that was never present in its robot demonstrations? The experiment is a
focused extension of the completed t0/t4 clause and clause-order results.

## Exact train/holdout split

The split uses three official LIBERO-Long task specifications with the same
receptacle and an overlapping object vocabulary:

| role | suite task | instruction | demonstrations | frames |
|---|---:|---|---:|---:|
| train | t0 | put both the alphabet soup and the tomato sauce in the basket | 33 | 9,571 |
| train | t1 | put both the cream cheese box and the butter in the basket | 49 | 12,702 |
| held out | t7 | put both the alphabet soup and the cream cheese box in the basket | 43 | 11,494 |

The model therefore sees soup-to-basket and cream-cheese-to-basket physical
behavior during training, but never sees them composed in one episode. The
official t0 and t1 BDDL files define the same scene, fixtures, seven movable
objects, object initial regions, and basket region; they differ only in
language, objects of interest, and the two goal predicates. In particular,
both soup and cream cheese are present at the same predefined regions in both
training environments. This permits a stronger primary test than running t7
directly: evaluate the held-out soup+cream-cheese goal in the t0 Scene-2
environment, holding scene and object layout constant and changing only the
composition requested through language.

All 43 official t7 episode IDs are excluded from dataset selection and from
spline normalization statistics. No checkpoint previously fine-tuned on
LIBERO may be used: every cell starts from the same public SmolVLM
initialization with a new action expert. This avoids the t7 leakage that would
occur if the existing full-suite `baselineA_100k` or `Cn8_100k` checkpoints
were reused.

The immutable data specification is
`configs/compositional_holdout_t01_to_t7.json`. The paired roots contain the
same selected 82 episodes and 22,273 physical frames. The only changed training
field is language/task ID:

- whole-task control: the original t0 or t1 caption at every frame;
- piecewise treatment: clause 1 through the first sustained-grasp release,
  then clause 2 from the next observation/action row.

Every selected non-language Arrow column must be byte-equal after the Parquet
round trip. Action/state normalization metadata for both heads and the separate
n=8 spline normalization artifact are recomputed from exactly the same 82
episode IDs; none can include held-out t7 frames. Image statistics remain in
metadata but are unused because visual normalization is identity.

## Factorial

Train the four cells below with batch 32, seed 1000, identical optimizer
schedule, 30,000-step cap, and checkpoints every 5,000 steps. Use one hardware
class for all cells where queue availability permits; otherwise record GPU
UUID/class and repeat the paired treatments on the same class before treating
small differences as meaningful.

| head | whole-task captions | piecewise clauses |
|---|---|---|
| waypoint SmolVLA (A) | A-whole | A-clause |
| n=8 spline + duration (C) | C-whole | C-clause |

Training is staged rather than blindly taking the last checkpoint. At 10k and
20k, run a small fixed state grid on the two training tasks. Advance to 30k
only if in-distribution competence is still increasing. Checkpoint selection is
based only on t0/t1 validation; t7 remains untouched until the selection rule
is frozen.

## Matched held-out evaluation

Primary evaluation uses t0 Scene-2 states 0–49 but replaces the evaluation goal
with the held-out conjunction `In(alphabet_soup, basket)` and
`In(cream_cheese, basket)`. OSMesa, 20 Hz, fixed seeds, and exact fresh-process
replay gates remain locked. Before rollout, the evaluator asserts that the
training t0/t1 BDDL initial predicates for soup, cream cheese, and the basket
match and that neither held-out predicate is initially true. Both language
treatments receive the same two-phase control structure and the same queue
clear at the same transition:

- whole-task model: repeat the full t7 caption in phases 1 and 2;
- piecewise model: `put the alphabet soup in the basket`, then
  `put the cream cheese box in the basket`.

The phase transition occurs when the requested first official LIBERO goal
predicate becomes true. This is a privileged simulator oracle applied equally
to all four cells; it isolates representational recombination and is not a
claim about learned boundary detection. A secondary deployment result uses the
same causal gripper-release switch already validated in the clause experiment.

The normal order (soup then cream cheese) is primary. The reversed order (cream
cheese then soup) is a prespecified secondary test of order recombination.
The official t7 state grid is a secondary cross-scene transfer result, not the
primary compositional estimate.

## Outcomes and decision rule

Report, without dropping failed states:

1. final two-goal success on the same-scene held-out composition;
2. requested-first subgoal completion and full requested-order completion;
3. retention after the first placement (whether the first predicate remains
   true when the second is attempted);
4. in-distribution t0/t1 success at the checkpoint selected without t7 and,
   secondarily, success on official cross-scene t7;
5. paired state-level discordances and exact McNemar intervals.

Primary effects are clause-minus-whole within A and C. The spline-specific
estimand is `(C_clause − C_whole) − (A_clause − A_whole)`. A result supports the
paper's compositional claim only if clauses improve the held-out composition without merely
reflecting a collapse of the whole-caption cell on the training tasks. One
seed is a screen; any positive interaction must be replicated before becoming
a headline number.

## Optional matched intervention

Only after the holdout result is known, test a later-clause intervention from
the same initial scene: preserve the first clause and change only the second
object/target. Compare action-prefix similarity before the switch and
trajectory divergence after it. This remains a follow-up, not part of the
current launch matrix.
