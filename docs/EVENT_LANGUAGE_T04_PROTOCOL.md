# Corrected event-language t0/t4 protocol (pre-results)

This is the locked rescue experiment for the two discriminating multi-object
LIBERO-Long cells. No result may be claimed from the engineering smoke jobs or
from a single head/protocol cell.

## Training intervention

- Tasks: LIBERO-Long t0 and t4 only; every other LIBERO frame is unchanged.
- Boundaries: the shared production `first_event` kernel, `horizon_max=24`,
  `min_seg=8`, `pause_frac=0.15`, half-open `k`-exclusive convention.
- Label input: three chronological time points from both the external and wrist
  cameras.
- Label output: Molmo2-8B selects exactly one reviewed, task-grounded atomic
  phase ID; open-ended multi-action captions are not accepted.
- Mixing: task-stratified 50/50 at the episode level. A demonstration is either
  task-conditioned throughout or event-clause-conditioned throughout.
- Heads: matched seed-1000 A waypoint and C n=8 spline trainings from the exact
  frozen source snapshot used by the legacy-language factorial.

The 1,011-label artifact must pass the independent temporal gate before the
derived dataset can build: adjacent phase inversions <=15%, gross drops <=5%,
at least 20 episodes/task, exact phase/label identity, and one boundary/prompt
version.

## Locked closed-loop protocols

All cells use OSMesa, LIBERO initial states 0..49, seed base 100000, and tasks
t0/t4 (100 paired episodes/cell).

**Static control:** the benchmark's original compound instruction for the full
rollout.

**Fixed clock:** issue the first object-placement clause, then switch to the
second at a predeclared step and drain actions generated under the old clause.
The clock is calibrated only from source demonstrations: the median first-object
release is step 105 for t0 (38 demos) and step 118 for t4 (43 demos). The
respective 10/25/50/75/90 percentiles are t0
`94.7/99.25/105/116.5/126.9` and t4 `102.2/108/118/128.5/134.8`.

**Event clock:** issue the same clauses, but switch when the executed decoded
gripper trajectory first closes and then releases. Drain the residual queue at
that causal boundary. This is adaptive to policy execution rather than eval
episode index or success outcome.

**Order swap:** reverse the two clauses while preserving the scene and success
predicate. Normal and reversed success together test counterfactual language
control rather than memorized demonstration order.

## Required cells and estimands

The minimum interpretable table is:

| Head | Training | Static compound | Fixed clock | Event clock |
|---|---|---:|---:|---:|
| A | standard | required | required | clock ablation |
| A | corrected event language | required | required | clock ablation |
| C | standard | required | clock ablation | required |
| C | corrected event language | required | clock ablation | required |

The primary language-training effects are event-minus-standard within A/fixed
and within C/event. The head-by-language interaction is their difference. The
cross-clock cells determine whether a gain belongs to the action head, the
scheduler, or their composition. Order-swap cells are secondary
counterfactual-steering tests and require the normal-order partner.

Use paired McNemar tests per task, Holm correction over the two predeclared
tasks for each family, and paired bootstrap intervals for pooled differences.
The claim gate also requires all static controls, clock partners, immutable
checkpoint/dataset manifests, and the Molmo temporal-quality report.

Job 2321868 is a 2-state/task C-baseline engineering smoke of the earlier
release-trigger implementation. It is not a statistical cell and cannot enter
the result table.
