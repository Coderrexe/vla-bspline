# Duration Mechanism Ablation v4: Frozen Compiled-Model Pairing

Status: **local-only engineering gate; do not use for a scientific result until
the OSMesa smoke artifact passes review.** V3 remains immutable and is not
changed by this protocol.

## Motivation

The OSMesa v3 paired evaluator matched raw and processed observations and the
MuJoCo integration state at step zero, but task IDs 3, 8, and 9 exposed a
separate fault: each arm's ordinary LIBERO reset could produce a different
compiled-model XML. Matching integration coordinates in two distinct compiled
models is not enough to establish a same-world paired rollout.

V4 removes that reset boundary within a block. For each task/state/repeat it:

1. creates the selected LIBERO environment once and does the sole explicit
   initial-state reset;
2. captures the raw observation, compiled-model XML digest, full MuJoCo
   `mjSTATE_INTEGRATION`, and safe mutable wrapper counters/RNG state;
3. runs donor, counterbalanced efficacy arms, and closure by restoring the
   captured state into that same live model with `mj_setState` and `mj_forward`;
   and
4. resets/reseeds policy and intervention-controller episode state for every
   arm, but never calls `env.reset` after capture.

The installed LeRobot wrapper has an important additional reset path:
`LiberoEnv.step()` calls `self.reset()` after a terminal/success transition, and
Gymnasium's default vector wrapper marks that member for reset on its next
step. That was the first v4-smoke failure: donor completed, its discarded
terminal reset changed placement-bearing model XML, and the next efficacy arm
correctly refused the changed model. V4 now suppresses only this discarded
terminal reset and snapshots/restores the vector next-step-autoreset flag. The
terminal transition itself (pre-reset observation, reward, termination, and
info) is unchanged; the evaluator stops immediately afterwards as before.

The five-arm v3 structure is retained: predicted donor; one of the six
preregistered efficacy permutations of predicted-eval/fixed/shuffled; predicted
closure. Donor and closure remain trace-only. Atomic no-clobber blocks, source
hashes, fixed-prior binding, task-local order manifests, seed collision checks,
shuffle-strength gate, and predicted-position repeatability diagnostics remain
in force.

## Frozen initial-pairing gate

Every restored arm must agree with donor on the same environment and policy
seeds, initial raw and processed observation digests, initial integration-state
digest, and compiled-model XML digest. Each arm also has an immediate
`frozen_restore_verified` assertion before policy inference. The result reports
the four donor-to-other-arm comparisons per block; a required gate fails after
publishing the diagnostic artifact if any of these fields disagrees.

The first allowed execution is only:

```bash
sbatch cluster/smoke_duration_ablation_v4_frozen_osmesa.sbatch \
  /path/to/checkpoint t38_v4_frozen_smoke /path/to/fixed_duration_prior.json
```

It is fixed to OSMesa, task IDs 3 and 8, state 0, and six repeats. That gives
12 balanced blocks / 60 rollouts, with every efficacy order once per task. The
launcher checks that all 48 donor-to-other-arm comparisons pass the frozen
initial-pairing gate. It intentionally does not offer a full-pilot mode.

## Caveats

- `mjSTATE_INTEGRATION` is the complete MuJoCo integration API state, but it is
  not a claim that every arbitrary Python object in robosuite is serializable.
  V4 explicitly snapshots scalar, array, container, counter, and NumPy RNG
  wrapper fields while preserving live simulator/model/renderer objects. Any
  changed wrapper chain, vanished field, model XML drift, integration-state
  round-trip mismatch, or captured-observation mutation fails closed.

- LeRobot's public `make_env(...)[suite]` factory may allocate suite siblings
  while selecting a task. V4 immediately closes all of those siblings and uses
  exactly one selected live environment/model for each block; no arm rebuilds
  or resets that selected model. This is a factory-allocation caveat, not a
  second model in a paired block.

- This gate establishes same-model initial pairing, not automatically exact
  full-trajectory replay. The optional v3 exact predicted replay diagnostic is
  still available. Scientific inference remains state/repeat hierarchical and
  requires independent checkpoint seeds.

- Do not combine v3 and v4 blocks or run the smoke concurrently with the same
  tag. Their resume identities and protocol schemas intentionally differ.
