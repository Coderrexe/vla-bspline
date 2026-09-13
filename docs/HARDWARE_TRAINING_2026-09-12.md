# Cabinet and drawer hardware training

## Checkpoint status and offline results

**September 12, 2026:** all four models finished 20,000 updates and passed strict
checkpoint reload and prediction. Cabinet waypoint, the last training job,
finished at approximately 04:23 EDT. No training jobs remain in this batch.

Every completed model was checked on the same 50 predetermined observations per
task: ten frames spread across each of five held-out demonstrations. These are
offline command-prediction checks, **not physical task completion rates**.

| Task / model | Steps | Per-step translation RMSE (mm) | Eight-step cumulative translation RMSE (mm) | Gripper MAE | Reload / predict |
|---|---:|---:|---:|---:|---|
| Cabinet, zero motion / hold gripper | — | 1.268 | 9.290 | 0.0854 | Reference |
| Cabinet, spline | 20,000 | 0.892 | 5.486 | 0.0572 | PASS |
| Cabinet, waypoint | 20,000 | 0.819 | 4.211 | 0.0612 | PASS |
| Drawer, zero motion / hold gripper | — | 1.565 | 11.453 | 0.1097 | Reference |
| Drawer, spline | 20,000 | 1.198 | 7.105 | 0.0408 | PASS |
| Drawer, waypoint | 20,000 | 1.286 | 7.686 | 0.0401 | PASS |

Translation RMSE is averaged over xyz components and sampled observations. The
eight-step metric compares the sums of eight predicted and recorded delta
commands (nominally 0.32 s), **not a measured endpoint on the physical robot**.
Gripper MAE uses the [0,1] opening range. One training seed and five held-out
episodes are insufficient to claim a general hardware advantage for either head.

All four checkpoints improve translation and gripper prediction over the
zero-motion / hold-gripper reference. On these drawer observations, spline
translation agreement is slightly better than waypoint, while gripper agreement
is essentially similar. On cabinet, waypoint translation agreement is better,
while spline gripper agreement is slightly better. This supports trying both
models in the lab, not calling either physical task solved.

Raw reports: `outputs/hardware_training_20260912/<run>/offline_validation.json`
and `.npz` locally; the same files are beside each run's `checkpoints` directory
on Misha. The deployment interface and remaining robot-side checks are below.

## Objective and run plan

Prepare separate task policies for the two native Apollo recordings in `bc_demo`:
50 cabinet demonstrations and 51 drawer demonstrations. Start with SmolVLA and
the project's event spline head, plus a waypoint control using the same episodes,
cameras, pretrained initialization, and training budget. Checkpoints will remain
separate for each task. Separate training limits the variables for first deployment;
joint training does not inherently cause catastrophic forgetting.

1. Audit every episode's numeric data and decode both video streams. Inspect contact
   sheets and inspect insertion/gripper transitions more closely where necessary.
2. Export immutable LeRobot datasets, retaining the original episode IDs and all
   observations/actions. Hold out five complete episodes per task for offline
   validation; never split adjacent frames across training and validation.
3. Fine-tune the pretrained `lerobot/smolvla_base` checkpoint. First run a short
   end-to-end training/load/predict check, then run the per-task jobs with saved
   intermediate checkpoints and offline validation.
4. Adapt the spline head to the actual analog gripper and action layout. Validate
   fit/decode round trips and use normalization estimated from these demonstrations.
5. Package model weights, processors, pinned source, sample inputs, action schema,
   and a load/predict example for the Apollo integration. Test locally on a GPU
   allocation without sending commands to any robot.

## Data audit completed

| Dataset | Episodes | Retained frames | Training / validation episodes |
|---|---:|---:|---:|
| Cabinet assembling (raw) | 50 | 55,359 | 45 / 5 before telemetry exclusion |
| Cabinet assembling (training export v2) | 49 | 54,087 | 44 / 5, same held-out IDs |
| Drawer assembling | 51 | 49,259 | 46 / 5 |

All numeric rows were checked for finite values. All frames in all 202 videos
were decoded and matched to the expected counts. Time-sampled contact sheets
from **every episode and both cameras** were visually reviewed; this is not a
claim that every video frame was manually inspected. The export preserves every
numeric column exactly and records original episode IDs and Parquet hashes.

A subsequent command/state consistency check identified one corrupt EE pose
observation at frame 1,236 of cabinet episode `20260911T005454.988Z-2ad672`.
The EE position jumps about 47 cm for one frame despite continuous joint states
and millimetre-scale commands. The raw data remain intact; the v2 export
quarantines this episode rather than inventing a corrected observation. All
other episodes had substantially stronger command/state agreement (per-episode
median lag-one translation cosine: cabinet 0.9775; drawer 0.982, rounded).
The detailed exclusion and launch provenance are in
`cluster/launch_records/hardware_20260912.md`.

Cabinet demonstrations place two blue components into separate positions on a
red fixture. Drawer demonstrations insert two red drawers into a blue housing,
lower drawer followed by upper drawer in the reviewed sequence. Both contain
grasp, transport, insertion, release, and a return to the other component.
The moving wrist view is complemented by the stationary view-arm camera.
There are no annotated language phase boundaries or alternative orders yet.

Across every recorded action, both rails and the view arm are inactive, with the
view gripper open. The Apollo spline adapter learns the leading six delta-pose
commands and **continuous** gripper opening, then restores the native 16-channel
output with inactive commands held fixed. A 0.5 gripper threshold is used only
to detect event boundaries, not to binarize training targets or predictions.

## Training specification

| Item | Setting |
|---|---|
| Initialization | `lerobot/smolvla_base`, revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205` |
| Models | Separate cabinet and drawer policies; spline and waypoint for each |
| Raw action window | 24 retained frames, nominally 0.96 s |
| Spline | 8 cubic control points; event duration 8–24 frames; learned duration |
| Execution prefix | At most 8 actions before a fresh observation, nominally 0.32 s |
| Inputs | Both RGB cameras; native 32-value state; recorded whole-task caption |
| Normalization | Training-episode statistics; spline target statistics embedded in checkpoint |
| Completed budget | 20,000 updates per model, batch 32, seed 1000, initial learning rate 1e-4 |
| Saved checkpoints | Every 5,000 updates plus final model; offline validation during training |
| Offline handoff check | Strict checkpoint reload; both cameras; 50 observations spread across five held-out episodes |

These are training specifications, not hardware success results. Offline reports
separate translation error (metres), rotation error (radians), and gripper opening
error, and include a zero-motion / hold-gripper reference. They do not compare
flow losses across representations as if those losses had the same units.

The first GPU smoke allocation failed CUDA initialization before training. A
second spline smoke exposed a concurrent Hugging Face Arrow-cache race; subsequent
jobs use their own derived dataset cache. Raw demonstrations and exports were not
modified by either failure. Smoke jobs are distinct from the final training runs.

Both 100-update smoke trainings now pass save/reload/predict checks. The spline
test checked 50 held-out observations, and the waypoint interface test checked
10. Both camera tensors are present and reach the image encoder in [-1, 1].
All 500 tensors from the pretrained model were checked exactly before the spline
smoke's first update; the same check is required for each full run. These short
smokes establish the training/inference path, **not task competence**.

Final-checkpoint prediction times below exclude the first call and include
preprocessing, GPU inference, and postprocessing. They exclude camera capture,
network communication, and robot command application. Benchmark the complete
loop again on the lab GPU; these are integration timings, not a controlled speed
comparison across GPU types.

| Task / model | Recorded GPU | Median prediction (ms) | p95 prediction (ms) |
|---|---|---:|---:|
| Cabinet spline | H200 | 139.9 | 141.7 |
| Cabinet waypoint | L40S | 137.6 | 139.3 |
| Drawer spline | H200 | 143.4 | 145.4 |
| Drawer waypoint | H200 | 134.7 | 135.9 |

The cabinet waypoint allocation is recorded as **L40S**, correcting an earlier
monitoring note that called it A100. An eight-action chunk spans 320 ms at 25 Hz;
a blocking inference call must not be mistaken for a control tick or filled by
repeatedly applying a stale delta command.

| Full run | Slurm job | Checkpoint directory below cluster workspace |
|---|---:|---|
| Cabinet waypoint | 2456089 | `outputs/cabinet_assembling_waypoint_v2_2456089/checkpoints/last/pretrained_model` |
| Cabinet spline | 2456090 | `outputs/cabinet_assembling_spline_v2_2456090/checkpoints/last/pretrained_model` |
| Drawer waypoint | 2456084 | `outputs/drawer_assembling_waypoint_v1_2456084/checkpoints/last/pretrained_model` |
| Drawer spline | 2456086 | `outputs/drawer_assembling_spline_v1_2456086/checkpoints/last/pretrained_model` |

Every checkpoint above has completed training and post-training verification.
Each run has `offline_validation.json` beside its `checkpoints` folder. See the
launch record for failures, replacements, and gates.
The final packaging job **2456098** completed in 15 seconds and wrote
`~/scratch/vla_bspline/hardware_20260912/handoff_2456098/`.
Its `handoff_manifest.json` lists all four models as `OFFLINE_VERIFIED`, with
no incomplete runs, and gives model hashes. A complete copy is downloaded to
`/Users/simbashi/Downloads/VLA_Hardware_20260912/`. Start with its `README.md`;
`models/` contains `cabinet_assembling_spline`, `cabinet_assembling_waypoint`,
`drawer_assembling_spline`, and `drawer_assembling_waypoint`. The source archive,
configuration/tokenizer cache, processors, native interface, initial-profile
metadata, and recorded prediction fixtures accompany the weights.
All four downloaded model SHA-256 hashes match the cluster manifest. The source
archive hash, cached revision, fixture shapes, and saved reference outputs were
also checked locally. This checks transfer and fixture consistency; GPU prediction
was verified on Misha, not rerun on this Mac.
The `portable_gate_*` folders contain minimally
trained smoke-test models and are **not** the models to take to the robot.

The portable-bundle gate (2456102) also passed prediction with
`HF_HUB_OFFLINE=1`, using a copied
checkpoint, the archived policy source, and only the included configuration and
tokenizer cache. Thus deployment does not need access to the training dataset
or mutable package-level spline statistics. It still needs a compatible Python/
GPU environment and the mentor's robot-side integration.

## Recorded interface

| Item | Recorded value |
|---|---|
| Platform | Apollo `xarm7_2arm_rail` |
| Camera inputs | `observation.images.grip_wrist`, `observation.images.view_wrist`; RGB 640 × 480 |
| State | 32 values, names and ordering in each task's `manifest.json` |
| Action | 16 values: grip EE delta pose (6), absolute gripper opening (1), rail delta (1), then the equivalent 8 fields for view arm |
| Frames | `arm_base:grip`, `arm_base:view` |
| Nominal stored rate | 25 Hz |
| Recorder | Apollo 0.1.0; metadata reports LeRobot 0.6.1 |
| Cluster checkout | LeRobot reports 0.5.2; snapshot used for reproducible training |
| Task language | Whole-task names; no recorded executable-clause annotations |
| Success metadata | Unset in inspected episodes; completion needs video review and later actual robot trials |

The recorder removes low-action frames and reindexes retained frames at 25 Hz.
Wall-clock timestamps and filter-gap metadata are retained. Thus stored duration is
an active-motion clock and does not equal elapsed human demonstration time. Do not
report removal of human idle time as a learned timing improvement. The training
export must not scale each retained action by its wall-clock gap.

The simulator spline implementation uses a binary ±1 gripper. Apollo records an
analog opening in [0, 1], so using the simulator gripper decode directly is invalid.
The nonmanipulating arm and rails must be checked for inactivity across all rows
before deciding which outputs to learn or hold fixed.

## Files and cluster workspace

- Raw local recordings: `bc_demo/{cabinet_assembling,drawer_assembling}`.
- Audit script: `scripts/data/audit_apollo_demos.py`.
- Audit results and contact sheets: `outputs/hardware_audit_20260912/`.
- Isolated cluster workspace: `~/scratch/vla_bspline/hardware_20260912/` on `ssh misha`.
- Training source snapshot: `source/src/lerobot/` in that workspace.
- Job logs, datasets, model outputs, and manifests have separate subdirectories.
- Prediction-only loader: `scripts/hardware/apollo_predictor.py`.
- Offline checkpoint test: `scripts/hardware/verify_apollo_checkpoint.py`.
- Native command/state diagnostic: `scripts/data/check_apollo_action_alignment.py`.

Compute and video processing on Misha run under Slurm allocations, following the
[YCRC job guidance](https://docs.ycrc.yale.edu/clusters-at-yale/job-scheduling/).
The base checkpoint is intended for task-specific fine-tuning; see the
[official SmolVLA model card](https://huggingface.co/lerobot/smolvla_base).

## Deployment handoff

**September 12 evening update:** the lab integration, corrected live-pose adapter,
tested Dora contract, and supervised rollout procedure are now documented in
`docs/HARDWARE_INFERENCE_2026-09-12.md`. The drawer spline has produced valid
predictions from live camera/state streams and registered with the runtime in
shadow mode. A subsequently authorized shadow hardware session hit an input
compatibility refusal and a controller-not-ready startup fault; it was closed
without return-home, with zero learned actions published. See the inference
document's startup diagnosis before retrying. The older
rotation diagnostic below is retained as provenance; the source audit identified
the observation-format mismatch rather than changing the native action targets.

The mentor's inference stack is now available. It did not block
training on recorded command targets. Before the first physical run, confirm the
rotation composition, command application timing, camera/color conventions, and
policy reset behavior in Apollo. Preserve Apollo's existing collision, workspace,
and operator-stop mechanisms. Offline losses cannot establish robot success.

**Rotation interpretation is not established by the feature names alone.** A
read-only diagnostic compared raw angular commands with changes in the recorded
wxyz orientation, testing spatial/body rotation vectors and Euler increments.
No tested interpretation gave a consistent match across both datasets. This
does not change the learned native command targets, but it means the inference
stack should reuse the collection-time Apollo action-application function and
confirm its rotation/controller mode. Do not invent a sign flip or frame
conversion from these correlations. The diagnostic and raw reports are
`scripts/data/check_apollo_rotation_convention.py` and
`outputs/hardware_audit_20260912/*_rotation_convention_extended.json`.

Both tasks share the recorded initial profile
`e2d42f3418b74418b6d0db89e5add5ed` ("2026-09-09 FurnitureBench"). Its grip rail
is 0.636 m and view rail 0 m. The handoff includes the first recorded profile
and camera/arm references for each task as metadata, **not commands to execute**.
Each packaged model also includes `prediction_fixture.npz`: a recorded 32-value
state, both RGB images, the random seed, and the previously verified model output.
Use this offline fixture to check the mentor's inference environment and input
packing before connecting to the robot; it is not a demonstration of task success.

The prediction helper accepts `state: float32[32]` and **RGB**, not BGR,
`view_rgb, grip_rgb: uint8[480,640,3]`. It returns a raw `float32[8,16]`
action prefix. It does not move the robot, change safety settings, or rescale
recorded delta commands by wall-clock gaps. Call `reset()` after an intervention
or episode restart, and discard any queued actions. Do not replay a stale chunk
repeatedly if inference or communications stalls.

```python
from apollo_predictor import ApolloPredictor

predictor = ApolloPredictor("/path/to/pretrained_model", device="cuda")
predictor.reset()
actions = predictor.predict_chunk(state, view_rgb, grip_rgb)
# Prediction only: mentor's Apollo stack validates and executes commands.
```

Use the pinned training source in a **separate inference environment/process**
if the mentor's Apollo dependencies differ. Do not replace the working robot
environment with the cluster environment. Keep the camera arm, rail positions,
camera crops, fixture placement, and state ordering consistent with collection.
The supervisor must confirm rotation composition and physical command limits;
neither a model output clamp nor the maximum demonstrated motion certifies safety.
The telemetry outlier also motivates rejecting invalid/stale pose readings in
the inference stack rather than feeding a fallback pose to the policy.

The useful first paper experiment is task completion and completion of each of the
two insertions for the matched policies. Clause or order interventions require
verified phase labels and a physically feasible alternative order; they are not
established by training on one fixed caption and one fixed demonstrated order.
