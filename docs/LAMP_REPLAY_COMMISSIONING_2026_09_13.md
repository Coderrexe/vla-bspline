# Lamp replay commissioning — September 13 evening

## Current state

Both selected lamp models are available on the lab GPU. No learned grasp or
assembly success has been recorded. The robot session is closed. Trial 007's
waiting replay process expired without an authorization or hardware session;
its 400 ms absolute-action setting has **not** been physically tested.

Xiatao identified the reference launcher at
`/home/mavis-v2/apollo-mavis-v2-ws/scripts/dev/run_hw_replay.sh`.
We read it, `replay_dryrun.py`, `align_props.py`, and the complete §16.8 of
`docs/design/14-dora-interface.md` in that workspace on September 13 evening.
We have not executed the reference launcher. Its documented successful
trajectory tracking is not a successful learned-policy evaluation.

## Measurements from our client

All three trials below restored the same recorded initial configuration after
controller startup, using the existing planned `goto_initial` API. They used
hardware speed scale 0.1 and 200 ms command rows, and drove only the manipulation
arm during the command portion. The reset itself can move both arms and rails.
No reported fault flag appeared in their command telemetry. That does not imply
table clearance: the front task table is absent from the collision model.

| Trial | Command source / transport | Rows published | Requested / measured net x (mm) | Net rotation tracking error | Outcome |
|---|---|---:|---:|---:|---|
| 004 | Selected spline / delta EE | 24 | −18.425 / −16.252 | 3.188° | Restored start permits substantial approach motion; rotation does not follow the requested change |
| 005 | Recorded prefix / delta EE | 24 | −29.639 / −27.382 | 3.843° | Orientation discrepancy also occurs without a learned predictor |
| 006 | Same recorded trajectory / absolute EE | 8 | −12.241 / −5.493 | 0.0646° | Better angular tracking over this shorter interval; translation lag makes the next prefix exceed the unchanged 6 mm step bound, so it is refused |

Rotation error compares the composed requested net rotation with measured net
rotation, not an RMS over a full episode. Trial 006 is shorter than 005: this
table is a diagnostic, not a matched aggregate performance comparison. The
trial-006 rail also changed by −1 mm, despite a nominal hold target.

Raw reports, telemetry and source hashes are under
`outputs/hardware_lamp_20260913/lamp_trial_004_tracking/`,
`lamp_trial_005_tracking/` and `lamp_trial_006_tracking/`.
Trial 004's video capture ended too early to cover the complete policy interval;
its after-trial images are not synchronized full-rollout evidence. Trial 005/006
camera captures remain in the corresponding isolated remote report folders.

## What the reference implementation adds

| Finding in the lab documentation | Consequence for our next test |
|---|---|
| §16.8 records an open hardware delta-EE orientation-drift defect and recommends absolute EE | Test absolute execution before attributing the approach failure solely to training |
| Replay publication rate must be consistent with the reduced hardware speed | At speed scale 0.1, use 2.5 Hz for a 25 Hz recorded path; our prepared 400 ms rows provide that rate |
| The reference absolute replay used a different 425-frame lamp episode and speed scale 0.6 | Do not transfer its reported errors or settings to our 521-frame control as if they were our measurements |
| Even that replay closed on air when the props differed from the recorded layout | Verify prop alignment before using grasp replay to assess infrastructure |
| Xiatao confirms no barrier for the front task table; arm/rail setup must match black markers | Obtain the physical marker check, maintain on-site e-stop supervision, and do not regard fault-free telemetry as table protection |

The read-only `align_props.py` check against our selected native episode
`20260911T204950.391Z-538dc8` found shade centroid displacement
(+9.3, −2.1) pixels and lamp displacement (+1.3, −5.9) pixels in the observing
camera. The script's approximate constant pixel scale converts these to
7.8 and 4.9 mm. These are image-based alignment estimates, not calibrated 3-D
measurements or proof that placement caused our learned-policy behavior.
Reference and overlay images are saved in
`outputs/hardware_lamp_20260913/lamp_prop_alignment_001/`.

## Prepared changes, not yet physically validated

1. Absolute EE transport retains the model's native delta predictions, converts
   each accepted chunk from the actual measured TCP pose, and checks the
   resulting absolute commands against the existing motion bounds. Recorded
   prefixes instead retain one fixed trajectory anchor. No learned output is
   replaced with a demonstration.
2. Slow the recorded control to 400 ms/row at speed scale 0.1. The 200 ms run
   accumulated translation lag; slower timing is a hypothesis to test, not a
   demonstrated repair.
3. Preserve the stationary rail at the wire boundary. The live configuration
   has `rail_flip=false`; the SDK truncates meters to integer millimeters.
   Float32(0.635) therefore became 634 mm. The isolated adapter now encodes the
   native millimeter hold value without that downward roundoff and monitors
   unexpected rail motion. Shared driver/runtime code is unchanged.

The local adapter/contact/replay/selection tests pass **287 tests**;
`outputs/hardware_lamp_20260913/lamp_absolute_slow_tests.xml` is the test receipt.
This does not validate physical execution. Trial 007 sent zero action rows.

## Next supervised sequence

1. Confirm marked arm/rail setup and operator readiness; retain the existing
   high-clearance approach envelope and e-stop supervision.
2. Run one finite absolute recorded-prefix test at 400 ms/row, with synchronized
   video and measured orientation/translation/rail tracking. Stop on a tracking
   fault or unexpected movement; do not automatically repeat it.
3. If tracking is satisfactory, test the selected learned lamp checkpoint via
   the same absolute transport and initial-pose restoration.
4. Before a grasp/full replay, separately review the actual fingertip/table
   clearance and align the props for the replay control. The current approach
   guard deliberately cannot grasp: its TCP floor is +100 mm and gripper must
   remain at least 85% open. In the selected demonstration, the grasp TCP is
   approximately −17 mm in the arm-base frame, **117 mm below that floor**.
   Completing this task requires a new, physically grounded envelope, not just
   increasing the chunk count or removing the guard.
5. Once full execution works, freeze settings and record matched spline and
   waypoint attempts with explicit success criteria, videos and all failures.
   Demonstration replay remains a separate control, never a learned success.

The physical assembly/marker check cannot be performed from telemetry alone.
The current lamp models use whole-task captions, so successful deployment would
support task execution, not by itself a hardware clause-generalization claim.
