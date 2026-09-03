# LIBERO t0/t4 clause-order steering protocol

## Question

Does changing the requested order of two executable language clauses change
which official LIBERO subgoal the policy completes first, from the same scene?
This is stronger than comparing final success under two prompts: the outcome is
the identity and time of the first simulator-verified placement.

## Valid counterfactual orders

| task | official conjunctive BDDL goals | normal order | reversed order |
|---|---|---|---|
| t0 | `In(alphabet_soup, basket)` and `In(tomato_sauce, basket)` | soup → sauce | sauce → soup |
| t4 | `On(white_mug, left_plate)` and `On(yellow_white_mug, right_plate)` | white → yellow-white | yellow-white → white |

The two predicates in each task act on distinct movable objects. Neither goal
is a precondition of the other, and the official success test is their logical
conjunction, so both orders preserve task semantics and final success criteria.

## Paired evaluator

For every task/state coordinate, normal and reversed programs receive the same
explicit LIBERO initial state, environment seed, policy seed, OSMesa renderer,
control frequency, checkpoint, and action cadence. The evaluator asserts equal
raw-observation, MuJoCo integration-state, and compiled-XML hashes after reset.
Only the order of the two reviewed clauses changes.

The program advances after LIBERO's official predicate for the requested first
subgoal becomes true, then clears actions queued under clause 1. This transition
is a **privileged simulator-predicate oracle**. It isolates language-conditioned
subgoal selection; it is not evidence for autonomous boundary detection.

Per episode, the immutable result records:

- the official predicate first completed and its control step;
- whether the requested first predicate was uniquely completed first;
- predicate changes, switch step, final official conjunction success, and
  completion length;
- explicit-state identity and the complete executed-action trace hash.

Two fresh-process replicas run sequentially on one L40S allocation. The gate
requires exact equality of all initial-condition, predicate, action-trace, and
outcome fields. A failed gate produces diagnostics but no claim-eligible result.

## Preregistered analysis

The primary descriptive estimand is the fraction of paired states with a full
requested order flip: goal 0 is uniquely first under the normal program and
goal 1 is uniquely first under the reversed program. It is reported over all
states and over states with a unique first completion in both conditions.

The paired directional test conditions on states whose unique first-goal
identity changes. The exact two-sided McNemar/binomial test compares requested
direction changes (normal 0 → reversed 1) against anti-direction changes
(normal 1 → reversed 0) under a 0.5 directional null. Ties and no-completion
states remain in the descriptive denominator but are excluded from this
conditional test. Results are reported per task and pooled.

Final success is a separate retention metric. Normal and reversed success,
paired discordances, and reversed-minus-normal success are reported. A −10 pp
margin is shown only as a descriptive noninferiority reference; no formal
noninferiority claim or test is made from one checkpoint seed.

## Execution rule

Run a one-state-per-task smoke first. Expand only after it confirms the deployed
wrapper path, exact BDDL identifiers, terminal autoreset handling, and replay
gate. The first smoke is Slurm job `2328259`.
