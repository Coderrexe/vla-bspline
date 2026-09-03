# Two-subgoal t0/t4 v2 replay protocol

This protocol supersedes only the *evaluation provenance* of the exact-task
clause experiment; it does not reinterpret the seed-1000 screen as a result.

For every claim-bearing evaluation, `eval_two_subgoal_clause_t04_v2.sbatch`
creates an isolated source snapshot, an isolated `n_action_steps=10` checkpoint
view, and two sequential OSMesa rollouts on one L40S allocation.  Both rollouts
use suite tasks 0 and 4, states 0..49, seed `100000 + 1000*task + state`, and
20 Hz.  The launcher rejects task text other than the exact soup/tomato and
two-mug evaluator descriptions.

Each episode records hashes of the raw reset observation, policy-processed
reset observation, MuJoCo integration state, compiled XML, first action, and
the framed full sequence of actions passed to `env.step`; camera tensors are
retained as compressed NPZ sidecars.  The replay gate fails if *any* t0/t4
coordinate differs in any initial-condition, program, action trace, outcome,
or termination length field.  It publishes the two raw traces and a gate JSON
with `passed: false` for diagnosis, but exits nonzero and declares no
claim-eligible result.

The only accepted replication table is 3 seeds (1000–1002) for these eight
cells per seed: A original/clause × compound/fixed-clock, and C
original/clause × compound/event-clock.  `two_subgoal_clause_stats.py` accepts
only passed gates, checks the locked 71-episode/19,378-frame H100 training
manifests, and refuses incomplete tables.  Its comparisons retain every paired
task/state outcome and report discordant pairs plus exact McNemar p-values;
they are not generated for a partial seed set.

The seed-1000 discovery checkpoints were re-registered under this same v2
protocol on 22 August rather than mixing their older evaluator artifacts into
the final table. The immutable Slurm job mapping is:

| arm | treatment | compound | programmed execution |
|---|---|---:|---:|
| A | original | 2326249 | 2326250 (fixed clock) |
| A | clause | 2326251 | 2326252 (fixed clock) |
| C | original | 2326253 | 2326254 (event clock) |
| C | clause | 2326255 | 2326256 (event clock) |

These jobs use the same frozen v3 evaluation source as the seed-1001/1002
replicas. No legacy seed-1000 result is admissible to the locked analyzer.

To remove the head-versus-scheduler confound, the final matrix also evaluates
C under the same fixed clock used by A. The C original/clause fixed-clock jobs
are 2326309/2326310 (seed 1000), 2326311/2326312 (seed 1001), and
2326313/2326314 (seed 1002). The primary spline-specific estimand is the
training-seed-level difference-in-differences
`(C_clause - C_original) - (A_clause - A_original)` at this matched clock.
The analyzer reports its n=3, df=2 t interval as primary and a hierarchical
seed/state bootstrap as secondary; release/event-clock results are reported
separately as a deployment mechanism.
