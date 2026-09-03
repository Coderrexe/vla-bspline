# Deterministic LIBERO Evaluator v2

Status: implemented and runtime-tested on Misha, but the separate-process replay
gate **failed**. It is not approved for claim-bearing evaluation.

The existing explicit-state evaluator fixes the major outcome-dependent reset
confound, but matching initial states is not enough to prove bitwise replay of a
stochastic flow policy. `scripts/eval/libero_locked_eval_v2.py` adds a stricter,
auditable protocol without changing the deployed evaluator:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before Torch import; deterministic
  algorithms are required (not warn-only), cuDNN benchmarking and TF32 are off.
- Every episode seeds simulator-related Python/NumPy/Torch streams before its
  explicit task/state reset. Immediately after `env.reset`, Torch and every CUDA
  generator are reseeded for the policy, so simulator reset cannot advance the
  policy's stochastic sampling stream.
- Every NumPy action array actually passed to `env.step` is hashed in order with
  dtype/shape framing. Each episode records this SHA-256 together with success,
  steps, policy calls, task/state, and both seeds.
- The result records Python, NumPy, Torch, CUDA, cuDNN, GPU, determinism flags,
  checkpoint/model/source hashes, and scheduler identity.
- Final JSON is published atomically with a no-clobber hard link. A concurrent
  process or rerun cannot replace an existing result.

Job 2319625 ran the known-sensitive LIBERO-Long task 2 twice sequentially on one
L40S allocation. Both episodes succeeded, but they completed at 244 versus 241
steps and had different exact action SHA-256 traces (`c30c0490...` versus
`cf9b512e...`). Deterministic Torch/cuDNN flags and construction-time seeding are
therefore insufficient when the environment and model are reconstructed in
separate processes. The next gate reuses one loaded model and environment in one
process, verifies identical initial-observation hashes, and compares action
prefixes across explicit repeat resets. Historical v1 results do not
retroactively acquire action-level determinism.

Known limits: MuJoCo/GPU-driver nondeterminism outside Torch is not guaranteed
away merely by Torch flags. Exact trace equality is therefore an empirical gate
for the full stack and should be checked per hardware/software environment. A
deterministic-kernel error is a protocol failure to investigate, not a reason to
fall back to `warn_only=True`.
