# Apollo hardware inference integration

## Current status

**Latest (trial 025, September 13): operator-reported table contact; learned
motion is on hold.** The operator subsequently reset the arm. No confirmed grasp,
insertion or task completion has occurred. Our policy, helper and recorder are
stopped. Read-only API inspection finds the operator's healthy teleoperation
reset session; it has been left untouched. No production runtime, driver,
collision setting or model checkpoint was changed.

| 400 ms/row trial | Requested / measured net descent | Gripper opening | Outcome |
|---|---:|---:|---|
| 024: 16 chunks, 128 rows | 237.920 / 233.417 mm | 81→72 mm | Manually interrupted as open fingers approached the table; no grasp. Operator later estimated a 1 cm gap at the stopped pose. |
| 025: 19 chunks, 152 rows | 262.181 / 257.269 mm | 81→73 mm | Next prefix refused at the recorded workspace floor. **Operator reports table contact during this trial.** No grasp. |

Trial 025's final recorded corrected TCP is [0.704377, −0.047514, −0.094046] m
in the manipulation-arm base frame. Its unpublished next prefix requests another
8.011 mm descent. The existing −0.100 m TCP floor checks demonstration support,
not the lowest fingertip or actual table geometry. Thus a later workspace
refusal did **not** prevent physical contact. There are zero reported fault frames
in 1,805 running telemetry samples, but these telemetry values cannot override the
operator's observation. Contact time and contact force are not established.
The attempted manual stop found API 404 after the helper had already ended
trial 025; it was not the terminating intervention. Trial 024, by contrast, was
manually interrupted and did not hit a policy guard.

### Post-contact scene coverage diagnosis

The selected production hardware scene is `mavis_v2` with safety enabled. A fresh
**offline** build of that scene and replay of all 1,805 archived joint samples
finds the final TCP projection **215.777 mm outside the modeled table footprint**.
The modeled table spans world y = [−0.310, +0.310] m, while the final TCP has
world y = +0.525777 m. Forward kinematics and the transformed recorded TCP agree
within 0.001263 mm over the audit. This verifies the coordinate calculation,
not physical calibration: the work surface under the task is not represented by
that table box. The model table plane is at world z = 0.735 m; the final TCP
is 13.142 mm above that plane, but the TCP is not the lowest fingertip. Neither
this height nor the contact report establishes a safe floor value.

`audit_apollo_table_coverage.py` imports only the simulation package, builds a
separate model, and performs kinematics without stepping physics, connecting to
hardware, or modifying the production scene. Evidence:
`drawer_table_coverage_046.json`, `drawer_post_contact_reset_045.json`,
`drawer_trial_025_analysis/`, and `drawer_task_slow_slew_042/` under
`outputs/hardware_inference_20260912/` (raw camera recordings also retained on the
Apollo workstation). Both sampled heads still predict open-gripper descent on
the refused input in the separate offline diagnostic
`drawer_refused_head_comparison_043/`; switching heads alone is not a recovery.

Our deployed launchers now check `apollo_contact_review_required.json` before
any motion authorization or hardware-session creation, including controller-enabled
shadow tests. Read-only prediction remains available. This is a **local workflow
hold, not a hardware safety device**; it does not change the lab's teleoperation
or stop another application. The original 206 isolated tests plus 11 hold tests
pass (**217 total**, `adapter_tests_v28.xml`). There is no CLI bypass.

Before releasing the hold: obtain on-site inspection of the contact and hardware
condition; have the lab owner validate the actual work surface and installed
gripper geometry, including startup motion; verify the revised protection offline
and through the lab's commissioning procedure; then review a revised first-grasp
trial. Do not lower the existing floor, disable collision protection, replay the
refused prefix, or treat a reset alone as resolving either failure.

### Execution profile and preceding trials

**Trial 023 (September 13, 00:33 EDT):** the operator reset both arms
to the collection pose; fresh idle telemetry and both RGB views confirmed the
starting setup and an 81 mm gripper opening. The new slow/slew profile below was
loaded, but startup produced a manipulation-arm mode/state warning and a recovery
event before any policy prediction or action. The helper closed only its own
session, without return-home. The policy and read-only recorder were then stopped.
The production runtime was not restarted or modified. Three Studio application
instances still had established connections to the manipulation controller on
port 18333. The warning alone does not prove what issued the controller change;
these connections must be eliminated before another supervised attempt.
Evidence: `drawer_reset_preflight_037.json`, `drawer_trial_023/session_check.json`,
`drawer_trial_023_summary.json`, `drawer_task_slow_slew_037/node_report.json`.
**Zero learned actions in trial 023; no new grasp or assembly result.**

The opt-in execution revision passed **206 isolated CPU tests**
(`adapter_tests_v27.xml`), including the original profiles, a full mocked 150-chunk
horizon, the exact trial-022 refusal, workspace stops, and non-response checks:

| Change | Scope |
|---|---|
| 400 ms rows, 2.5 Hz wire rate | Same predicted pose deltas; twice the row duration of trial 022. Previous chunk finishes before the next inference starts. |
| Gripper reference slew | At most 0.06 normalized opening change per row, including the chunk boundary. Only gripper references change; raw and transmitted outputs are both logged. This is not a physical velocity/force limit. |
| Response timing | At 400 ms, compare measurements with the expected active row, not the future final target. Same-host publication time approximates receipt; this is not a hardware acknowledgement. Persistent non-response still stops after 3 s. |
| Finite grant | Distinct `SUPERVISED_DRAWER_TASK_SLEW` grant; at most 150 chunks and 550 s at the slower clock. Original 200 ms profile remains capped at 300 s. |
| Preserved limits | Same pose, path, workspace, rotation, stale-input and reset guards; same driver speed and force settings. Gripper-step tolerance of 1e-7 only covers float32 rounding (<0.00001 mm). |

Trial 023 is a startup interruption, not a scored policy failure or a successful
timing test. After Studio exited, trials 024–025 executed the revised clock/filter,
as reported above. Improved net tracking does not establish task success or
physical safety. The archived trial-024 executor window has zero joint-step cap
events in 6,212 logged ticks, versus 1,922 in 3,958 for trial 022; trajectories
and termination points differ, so this is not a controlled timing ablation.

**Prior 200 ms policy (trial 022, September 13 shortly after midnight):** after the operator
restored both arms to the collection pose and opened the gripper, the existing
teleoperation reset session was handed over to inference using one targeted
DELETE without return-home. No recording or engaged teleoperation was present.
The earlier waiting client 032 had expired without any motion grant. Fresh
client 033 then ran 22 eight-row chunks continuously at 200 ms/row. Measured
net xyz was [16.372, −43.981, −236.456] mm, versus requested
[27.525, −54.103, −313.993] mm. Gripper response occurred in both directions
throughout the run; the response-window endpoints were 81→67 mm. The view arm
and rails stayed still during execution, with no reported fault in 991 running
telemetry frames. There is still no established grasp or insertion.

The next prefix was refused because an adjacent gripper target changed by
0.065143 (5.472 mm) against the 0.06 (5.040 mm) per-row allowance. Its total
gripper variation, initial target difference, and pose budgets were not the
trigger. The operator then reported the fingers very close to the tabletop.
The session is closed and our client/capture are stopped; **the gripper limit
has not been increased and the refused motion has not been replayed**.
Fresh physical clearance must be established before further motion.
Evidence: `drawer_trial_022_analysis/`, `drawer_task_033/`,
`drawer_reset_handoff_033.json`.

The reset reduced the view-arm discrepancy to 5.241 mm downward and 0.679 degrees
after controller startup (versus 33.893 mm and 4.424 degrees in trial 021).
The stopped grip TCP z is −74.412 mm. Across the demonstrations, the median
first target below half-open is [+18.526, +28.159, −3.195] mm relative to that
pose; this is a descriptive comparison, **not a correction command or clearance
estimate**. Offline checks on the same stopped input at three inference-noise
seeds predict 3.15–5.12 mm descent for spline and 8.23–16.17 mm for waypoint.
Neither head establishes a safe grasp from this input merely by being finite.
Evidence: `drawer_trial_022_pose_support.json`, `drawer_stopped_pose_comparison_034/`.

The operator confirmed a small visible gap below both fingers. No further
controller enable or learned motion is authorized from that near-table pose
without clearance/recovery review: the next predicted descent is several
millimetres, and controller startup itself has caused downward movement.

### First-grasp and execution diagnostics after trial 022

Read-only runtime logs record 1,922 joint-step cap events over 3,958 controller
ticks in the saved trial window, with zero reported IK slips/divergences.
Combined with the command-versus-motion discrepancy above, this supports testing
a slower command clock to reduce saturation; it does not prove that saturation
is the only reason the grasp was missed. A pure-source replay verifies unit
integrated delta bookkeeping for the candidate 400 ms rows / 2.5 Hz wire rate.
The clock was initially offline-only; the reviewed opt-in implementation,
interrupted trial 023, and executed trials 024–025 are documented above.
No driver speed, force, workspace or substantive gripper
step limit was raised. Evidence: `drawer_trial_022_executor_limits.json`,
`drawer_trial_022_runtime_window.log`, `runtime_delta_cadence_400ms_offline_036.json`.

`diagnose_apollo_first_grasp.py` separately checks six fixed offsets around the
first recorded gripper target below 0.5, in all five held-out episodes and the
first five training episodes in saved provenance: 60 observations, three paired
inference-noise seeds, both frozen heads. Original legacy state/action files are
hash-checked against checkpoint provenance; both original RGB streams are decoded
at exact frame indices. No live robot interface is imported or action published.
Training and held-out episodes are reported separately; these are not additional
training seeds or physical successes.

| Offset from first half-opening target | Held-out gripper MAE, spline (mm) | Held-out gripper MAE, waypoint (mm) |
|---|---:|---:|
| −32 frames (−1.28 s) | 3.97 | 2.08 |
| −16 frames (−0.64 s) | 11.81 | 8.68 |
| −8 frames (−0.32 s) | 13.77 | 14.50 |
| Boundary | 25.05 | 27.89 |
| +8 frames (+0.32 s) | 4.14 | 4.24 |
| +24 frames (+0.96 s) | 1.83 | 0.85 |

Each row averages five observations and three inference-noise samples over the
eight predicted targets. Both heads predict strong closure on recorded inputs
after the transition, but the transition itself is uncertain. In particular,
similar **mean** target openings at the boundary conceal substantial per-example
error and must not be advertised as accurate grasp timing. These checks separate
command-prediction behavior on recorded states from the unresolved closed-loop
approach. Raw per-example predictions, targets, splits and video hashes are in
`first_grasp_inputs_035/` and `first_grasp_diagnostic_035/`.

**Previous continuous attempt (trial 021):** a separate continuous drawer-task profile replaced the
short commissioning horizons. Seven eight-row chunks executed before the next
proposed descent crossed the recorded-workspace envelope. Requested net xyz was
[24.668, 5.254, −34.748] mm; measured [22.411, 4.362, −33.144] mm. Gripper
readout changed from 67 to 68 mm; no grasp or insertion was established. There
were 388 running telemetry frames with no reported fault. The view arm and rails
were unchanged during execution. Our session is closed (API 404), our client
and camera capture are stopped, and the production runtime remains running.
Evidence: `drawer_trial_021_analysis/`, `drawer_task_030/`.

The recorded-view comparison identifies an important setup mismatch: the view
arm TCP is 33.893 mm lower, 9.034 mm displaced in x and 5.574 mm in y, with a
4.424-degree orientation difference from the collection pose. Across trials
016–021 its reported height fell from 288.460 to 261.011 mm in approximately
5.5 mm increments between trials, although it stayed still during each action
window. Session-local stationarity calibration does not check absolute agreement
with the collection camera pose. Therefore its passing result must not be
interpreted as visual-input equivalence. This is a plausible contributor to
the missed approach, not an established sole cause. The next physical attempt
should restore both arms to the collection setup and use one continuous session,
with camera/rail agreement checked before publication. Do not increase the floor
allowance to continue the current missed approach. See the pose audit below.

**Operator clarification:** the earlier reported gripper fix was resetting the
arm and bringing it back, not a specified hardware or software repair. Subsequent
checks through our same Dora inference path confirm response in both directions:
trial 016 closed from 81 to 74 mm (target 74.28 mm); trial 017 reopened from
74 to 80 mm (target 80.72 mm). Both used a fixed target repeated in three
eight-row chunks, with zero pose/rail increments and no measured arm/rail motion
during the response window. All sessions were closed without return-home. The
runtime PID/epoch and audited gripper/driver/executor source hashes are unchanged.
These checks allowed commissioning to resume, but later discrepancies mean the
gripper response issue should not be described as permanently resolved.

The subsequent learned approach (trial 018) executed eight eight-row chunks at
200 ms per row. Requested net xyz was [−1.335, −26.361, −125.709] mm; measured
net xyz was [−1.759, −25.746, −121.200] mm. Measured gripper opening decreased
from 80 to 73 mm, following a final target of 73.332 mm. The view arm and rails
were unchanged during the response window; 421 running telemetry frames contained
no reported fault. The ninth prefix was refused by the cumulative 0.1 gripper
change allowance, before publication. All sessions and our clients are stopped.
The saved camera views show an approach, not an established grasp or insertion.
Evidence: `drawer_trial_018_analysis/report.json`, `drawer_postfix_approach_025/`.

After the operator confirmed roughly 8–10 cm of clearance from the knob, trial
019 continued with the separate grasp-stage profile. Seven eight-row chunks
produced measured net xyz [16.371, −7.687, −66.301] mm, against requested
[17.387, −7.745, −67.889] mm. The gripper initially closed from 73 to 67 mm, then
its readout remained at 67 mm despite later targets down to 59.668 mm. No contact
or grasp is established from this discrepancy. The eighth prefix was refused
because its 0.038715 rad rotation path would take the accumulated 0.147331 rad
above the 0.15 rad stage limit. Translation and gripper-stage budgets were not
the stopping condition. There were 380 running frames with no reported fault;
view arm and rails were unchanged during execution. The session is closed and
our client stopped. Camera sequence and raw predictions are retained under
`drawer_trial_019_analysis/` and `drawer_grasp_026/`.

After the operator confirmed a visible gap between the right finger and right
knob, trial 020 executed one eight-row slow alignment prefix. Requested net xyz
was [13.230, 1.165, −0.873] mm; measured net xyz was [12.289, 1.092, −0.571] mm.
All eight rows completed and the helper recorded a three-second response window
before closing its own session without return-home. There were 172 running
telemetry frames, no reported fault, and no measured view-arm or rail motion.
Measured gripper opening remained 67 mm throughout, despite targets from
63.532 mm initially to 60.471 mm finally. The independent idle monitor also read
67 mm afterward, with no reported arm error. This is not a grasp result; command
delivery, gripper feedback, and possible mechanical obstruction still need to be
distinguished before further closure. Our client and video capture have ended.
Evidence: `drawer_trial_020_analysis/`, `drawer_align_029/`.

September 12, evening EDT: all four trained checkpoints are on the lab workstation.
The drawer spline has executed a sustained physical approach using full action
chunks. Slowing row execution from 40 to 200 ms preserves commanded increments
while reducing their requested speed; this addresses the live executor's restrictive
motion caps without changing those caps. Before the reset, trial 013 executed six eight-row
chunks and descended 126.2 mm toward the drawer, with no reported fault and no
motion of the view arm or rails during policy execution. It stopped when the
predicted gripper closure exceeded the approach-only allowance. **No grasp,
insertion, or complete task has yet been demonstrated.** All completed sessions
were closed without return-home. The operator confirmed that the slightly rotated
gripper remains away from the knobs. A subsequent deterministic gripper-only
check (trial 014; not a learned-policy result) requested a 6.72 mm reduction in
opening, with zero pose/rail increments. Dora accepted the action, but the measured
opening did not change during the three-second response window. Trial 015 repeated
the same fixed target in three eight-row chunks, again with no measured closure
and zero pose/rail movement during the response window. Both sessions are closed.
Trials 016–018 subsequently demonstrated some gripper response, as described above. The production
runtime and drivers remain unchanged.

Initial commissioning history: at 20:26 EDT, one
explicitly authorized **shadow hardware session** was opened at 10% speed with
`keep_current`. It encountered a parked-state input refusal and a manipulation-arm
servo startup fault before any learned action was published. That session was
closed without a return-home request, and our client stopped. No learned action
was executed in that initial session. Controller enabling
did change the reported joint posture; zero policy actions must not be described
as proof of zero physical movement. Further enabling or motion requires renewed
operator review and approval. Subsequent supervised motion was explicitly approved
by the operator at the table. See the startup diagnosis below.

Later the same evening, the user closed UFACTORY Studio and a read-only socket
check confirmed its connection was gone. The runtime and cameras stayed live.
The input mismatch was traced to zero-variance state features; a read-only
constant-feature diagnostic produced five fresh finite predictions. Two further
explicitly approved 15-second shadow hardware sessions then completed without a
reported controller fault, generating 45 predictions each and publishing none.
Both sessions are closed and both clients stopped. The operator observed a click
and small startup jerk, and subsequently confirmed this is expected on this setup.
The original adapter and no-publication diagnostic remain unchanged. Session-local
parked-input calibration subsequently passed a live hold (34 predictions, no
actions). A quaternion-sign discontinuity was then identified and corrected in
an opt-in adapter path; saved-input tests confirm the corrected behavior. That
orientation correction was subsequently used in the successful one-row and
three-step learned-motion tests described below.

| Check | Observed result | Evidence under `outputs/hardware_inference_20260912/` |
|---|---|---|
| Four checkpoint transfers / strict loads | All weight hashes match; all models produce finite 8×16 action prefixes | `gpu_bundle_check_cu128_v3.json` |
| Observation adapter round trip | All four fixture predictions are bit-identical before/after the stabilized adapter on the lab GPU | same report |
| Isolated adapter / publication tests | 191 passed, including native/slow wire timing, session grants, cumulative budgets, approach/grasp/task profiles, workspace and response checks, and deterministic gripper checks; mocked transport only | `adapter_tests_v25.xml` |
| Real Dora inputs → drawer spline prediction | 5/5 finite prefixes; 202–236 ms inference, median 211 ms | `drawer_observer_002/report.json` |
| Real cameras in policy client | 1,171 RGB frames over a 20 s shadow attachment; no frame errors | `drawer_shadow_001/node_report.json` |
| Runtime recognizes our client | Telemetry reported `policy_attached: true`, drawer spline ID, `policy_arms: [grip]` | read-only `/ws/telemetry` check; shadow log/spec counters |
| Initial passive checks: action messages / runtime API writes | 0 / 0 | observer and initial shadow reports |
| Supervised shadow hardware session | 1 create + 1 delete; 0 learned action messages; stopped after startup fault | `drawer_hardware_session_001/`, `drawer_session_shadow_003/node_report.json` |
| Numerical equality with H200 fixtures | Cabinet models pass original tight criterion; drawer models need the qualification below | `gpu_bundle_check_cu128_v3.json` |
| Constant-feature diagnostic on fresh inputs | 5 finite predictions; 196–217 ms; no actions or API writes; not execution-ready | `drawer_constant_observer_005/report.json` |
| Controller-enabled shadow checks after Studio closure | Two 15 s checks; 45 predictions each; no reported faults or learned actions | `drawer_hardware_session_002/`, `drawer_hardware_session_003/`; client reports `drawer_session_shadow_006/`, `drawer_session_shadow_007/` |
| Offline parked-calibration replay | Three recorded holds establish a stationary reference in 3.26–3.33 s; 35 later inputs checked per hold | `parked_calibration_replay_v2.json` |
| Live session-local calibration | Session 005: 34 predictions after calibration; median 208 ms; no reported faults or action messages | `drawer_hardware_session_005/`, `drawer_session_calibrated_009/` |
| Quaternion-sign ablation | Equivalent sign removes a >120-standard-deviation component discontinuity; both drawer fixtures unchanged | `drawer_quaternion_diagnostic_001/`; spline guarded integration `drawer_quaternion_integration_002/` |
| Corrected single learned row | Requested 0.772 mm translation norm; measured 0.565 mm; gripper target 0.85883, measured 0.85714; no reported fault | `drawer_trial_008/`, `drawer_one_row_012/` |
| Three-step closed-loop commissioning | Three two-row prefixes; 4.738 mm total requested path; 2.146 mm measured net displacement; gripper 0.96429→0.97619; view/rails unchanged; no reported faults | `drawer_trial_010_analysis/`, `drawer_bounded_014/`; timestamp-matched camera frames inspected |
| Short slow-clock approach | Three eight-row chunks at 200 ms/row; requested net xyz [−1.313, −42.524, −7.531] mm; measured [−0.657, −37.926, −7.006] mm; no reported faults | `drawer_trial_012_analysis/`, `drawer_slow_approach_016/` |
| Sustained slow-clock approach | Six eight-row chunks; requested path 148.388 mm; measured net xyz [3.662, −3.024, −126.215] mm; no reported faults; stopped at requested gripper closure | `drawer_trial_013_analysis/`, `drawer_extended_approach_017/`; synchronized camera sequence inspected |
| Deterministic gripper-only diagnostic | One row: opening 0.97619→target 0.89619, zero pose/rail increments; measured opening unchanged, no reported arm fault | `drawer_trial_014_analysis/`, `drawer_gripper_check_018/node_report.json`; not a learned-policy or grasp result |

The camera images were inspected: the two red drawers and blue housing are visible
in both real wrist streams. The first five predicted drawer actions have translation
norms 0.94–1.66 mm and command an open gripper. Their complete eight-row prefixes
span 7.9–30.1 mm of summed translation. These are **proposed commands, not measured
robot motion or evidence that the policy can complete an insertion**. The later
motion measurements above are distinct from those original predictions.

## Learned-motion commissioning and native action timing

| Trial | Scope and result | Qualification |
|---|---|---|
| 006 | One learned row; measured about 4.29 mm translation | Revealed an incorrect wire-rate declaration; retain as a protocol diagnostic, not native tracking validation |
| 007 | Preflight refused because the waiting policy had expired | No session created and no action sent |
| 008 | One learned row at the corrected native wire rate; small arm and matching gripper response | No task completion |
| 009 | Full eight-row prediction refused before publication; offline replay requested 19.38 mm against the 10 mm commissioning budget | Correctly bounded refusal, not a hardware fault; preserve saved input/replay |
| 010 | Three successive two-row prefixes executed at the corrected native wire rate | Net grip TCP delta [0.409, −2.066, 0.411] mm; requested summed delta [0.012, −4.063, 0.445] mm; no task completion |
| 011 | Two native eight-row chunks; third refused because cumulative proposed path would exceed 80 mm | Requested path 46.183 mm; measured net displacement 8.087 mm before cleanup; no reported fault; shortened observation due to guard stop |
| 012 | Three full chunks at 200 ms/row, wire 5 Hz, model predictions at most 0.625 Hz | Measured net displacement 38.573 mm; requested net displacement about 43.21 mm; all three chunks completed and response recorded |
| 013 | Extended slow approach; six full chunks executed, seventh refused on gripper allowance | Seventh predicted opening 0.828–0.868 versus measured 0.976; translation and rotation were within limits; no grasp command was authorized beyond the approach allowance |
| 014 | Deterministic gripper response check, not a learned policy; one native row, zero arm/rail increments | Target opening 0.89619; measured opening stayed 0.97619 over three seconds; both arms/rails unchanged during the response window; session closed |
| 015 | Deterministic repeated-target gripper check; three eight-row chunks, all pose/rail increments zero | Exactly the same 0.89619 target throughout, not cumulative closure; measured opening stayed 0.97619; 166 running frames, no reported arm fault; session and client closed |
| 016 | After operator reset: fixed closing target through Dora; three eight-row chunks | Requested 0.884286; measured 0.964286→0.880952 (81→74 mm); no measured arm/rail motion during response; 168 running frames, no reported fault |
| 017 | Small reopening of the empty gripper; fixed opening target through Dora, three eight-row chunks | Requested 0.960952; measured 0.880952→0.952381 (74→80 mm); no measured arm/rail motion during response; 169 running frames, no reported fault |
| 018 | Learned approach after operator reset; eight eight-row chunks, ninth refused at the approach-only closure limit | Requested path 131.696 mm; measured net descent 121.200 mm; gripper 80→73 mm; 421 running frames, no reported fault; view arm and rails unchanged during response; no grasp established |
| 019 | First bounded grasp-stage continuation; seven eight-row chunks, eighth refused at cumulative rotation limit | Requested path 76.467 mm; measured net xyz [16.371, −7.687, −66.301] mm; gripper initially 73→67 mm, then unchanged despite smaller targets; no reported fault; view arm/rails unchanged; no grasp established |
| 020 | One eight-row slow alignment prefix after operator-confirmed visible gap | Requested net xyz [13.230, 1.165, −0.873] mm; measured [12.289, 1.092, −0.571] mm; no reported fault; gripper remained 67 mm despite final target 60.471 mm; session closed; no grasp established |
| 021 | Continuous task profile; seven eight-row chunks, eighth refused at workspace floor | Requested path 85.157 mm; measured net xyz [22.411, 4.362, −33.144] mm; gripper 67→68 mm; 388 running frames, no reported fault; view arm/rails unchanged during action window; no grasp established |
| 022 | Continuous attempt from restored collection setup; 22 eight-row chunks, next prefix refused at per-row gripper step | Requested path 507.924 mm; measured net xyz [16.372, −43.981, −236.456] mm; sustained gripper response, endpoints 81→67 mm; 991 running frames, no reported fault; view arm/rails unchanged during action window; no grasp established |

The production source scales external deltas by `period / chunk_dt_s`, then its
`ActionAnchor.row_step` consumes one full row budget. Declaring the model-call rate
(3.125 Hz) as the wire action rate while using 0.04 s rows multiplies integrated
delta bookkeeping by eight. Our client now declares **25 Hz native row timing**
and separately limits model calls to **3.125 Hz**. Pure-method replay of the exact
runtime source confirms 8× before and 1× after for one- and eight-row chunks;
the runtime itself was not edited. Evidence: `runtime_delta_cadence_v1.json` and
`scripts/hardware/check_apollo_runtime_delta_cadence.py`.

Physical tracking is not identical to commanded delta sums: the unchanged executor
also applies IK, slew, leash, and joint/Cartesian step limits and drops rejected
intent. The current evidence does not isolate each one's contribution. The full
hardware bring-up path applies `speed_scale=0.1` to host and driver motion caps,
and passes the capped control configuration into the policy executor. It also
uses the scale for goto planning. This corrects the earlier incomplete description
based only on the control loop's stored `speed_scale` field. **Scaled motion caps
are not uniform temporal rescaling of the learned trajectory**: saturated commands
can lose displacement. Finite delta/rotation/gripper budgets and the existing
executor protections bound these supervised tests. No configured driver limits
or safety gates were changed.

The distinct `--supervised-approach` profile retains fresh session/epoch grants,
native timing, parked-state monitoring, and automatic disarming. It permits at most
eight chunks and ten seconds, with a total 80 mm translation path, at most 40 mm
per prefix and 6 mm per row, total rotation at most 0.15 rad, and gripper change
at most 0.1, with an initial measured opening of at least 0.9. It is an approach
test, not full-task authorization. The default
commissioning profile remains 10 mm total. Trials 011 and 012 used the short
approach profile.

The separately selected `--extended-approach` profile requires 200 ms rows and
permits at most 12 chunks, 30 seconds, and 250 mm total translation. The 40 mm
prefix, 6 mm row, 0.15 rad total rotation, initially open gripper, and 0.1 gripper
change checks remain. Trial 013 used this profile and stopped on the gripper
check, before consuming its translation allowance. Parked-state inputs continue
to be checked at the original callback cadence between slow model predictions.

The separately selected `--supervised-grasp` profile was mock-tested before its
first supervised execution in trial 019. It cannot be combined with either approach flag.
It retains the extended approach's 200 ms rows, maximum 12 chunks/30 s, 250 mm
total translation, 40 mm per prefix, 6 mm per row, 0.15 rad total rotation,
session-local input calibration, fresh images/state, fixed view arm/rails, and
unchanged runtime/driver limits. It requires a new grant with the distinct purpose
`SUPERVISED_GRASP_STAGE`.

Only the gripper envelope changes for this stage: an arbitrary valid initial
opening is permitted, allowing continuation without a gratuitous reopen/reset.
The first target of each prefix must be within 0.1 of both measured opening and
the preceding published target; adjacent targets within a prefix may differ by
at most 0.06. Gripper-target variation is limited to 0.45 per prefix and 1.0
over the stage, including reversals and chunk boundaries. Targets remain in [0,1].
An oversized prediction is saved and refused, never clipped or retried. This
permits gradual learned closure but does not establish clearance or contact-force
safety; physical alignment and clearance require operator review between bounded
stages. Trial 019 was automatically closed at its rotation allowance, without
return-home or increasing any limit.

The separate `--supervised-drawer-task` profile permits one finite continuous
attempt without the approach/grasp profiles' small cumulative allowances. It
requires a fresh `SUPERVISED_DRAWER_TASK` grant, the drawer checkpoint, eight
200 ms rows, at most 150 chunks and 300 seconds, and the unchanged hardware
speed scale of 0.1. It cannot be combined with the other profiles.

| Continuous-task check | Limit |
|---|---|
| Requested translation | 2 m total path; 40 mm/prefix; 6 mm/row |
| Requested rotation | 1.5 rad total path; 0.125 rad/prefix; 0.025 rad/row |
| Corrected grip TCP box, metres | x [0.53, 0.77], y [−0.16, 0.15], z [−0.10, 0.23] |
| Orientation excursion from initial task observation | 0.30 rad |
| Gripper-target variation | 0.1 initial difference from measured opening; 0.2 across chunk boundaries; 0.06/row; 0.62/prefix; 4.5 total |
| Gripper response | Stop after 3 s of sustained target <0.4 while measured >0.6, or target >0.85 while measured <0.65 |
| Motion response | Stop if >30 mm requested path over a 5 s window yields <3 mm measured position excursion |

The box is a demonstration-support envelope, **not a collision model**. Every
proposed intermediate pose is checked. The response checks permit normal partial
opening around a grasped knob; they do not certify contact or force safety.
Predictions are refused rather than clipped or retried. Fresh state/images,
session/epoch identity, parked-state monitoring, and disarming on reset, fault,
intervention or lost connection remain active. Production code/settings were not
changed. The 51 demonstrations have median translation path 1.572 m, maximum
1.822 m, maximum rotation path 1.270 rad, and maximum gripper-target variation
3.936; these describe the data, not hardware safety ratings. The full recording
lasts at most 43.88 s, about 219 s at the slow execution clock.

The runtime log specifically reports 0.04 m/s TCP and 0.06 rad/s joint caps,
with capped ticks during native and some slow motion. Fivefold time stretching
does not guarantee all caps are inactive. Trial 012 nevertheless tracks its
requested trajectory much more closely than the native commissioning segment.
These trials began at different poses and used different predictions; **they are
not a matched hardware timing ablation or an architecture comparison**. Pure-method
replay confirms unit integrated delta bookkeeping at both clocks:
`runtime_delta_cadence_slow_v2.json`.

An issue identified before grasp validation: trial 013's gripper readout remained
0.976 throughout, despite published targets as low as 0.884 within allowed
prefixes. Earlier single-row and slow tests did show gripper response. The current
evidence does not establish whether this is command delivery, sensor freshness,
or a physical restriction. Trial 014 reproduced the missing response without any
learned pose/rotation command. The operator reports clearance from the knobs;
this does not by itself identify the electrical/controller cause. No gripper
force/speed settings or driver code were changed.

### Gripper diagnosis and shared-runtime checks

Read-only checks after trial 014 established the following:

- The G2 path uses `set_gripper_g2_position(..., wait=False, wait_motion=False)`.
  The driver does not publish that call's return code or the age of its last
  successful gripper poll in the telemetry captured here. Its monitor loop also
  catches exceptions without logging them. Therefore arm error code zero and
  fresh joint telemetry do not prove a successful gripper command or fresh
  gripper position.
- `arm_cmd` carries joint/rail targets, not the commanded gripper fraction. Its
  trace cannot establish that the gripper target reached the SDK. A recorded
  Dora action receipt establishes acceptance upstream, not mechanical execution.
- Two older Xiatao replay clients (PIDs 2734239 and 3672254) were still alive.
  Their current logs show repeated broken-pipe failures sending their specs;
  there is no evidence here that they supplied competing actions during our
  trials. They were not stopped or modified. The active session API reports no
  session after our cleanup.
- SDK messages go to `var/dora/node-stdout.log`, separately from
  `var/logs/runtime.log`. The former reports `set_tcp_load -> code=9` during
  bring-up, including trial 014. The installed SDK defines this as
  `STATE_NOT_READY`. The lab's `backstops.py` documents this stopped-controller
  status echo. Read-back before trial 014 and a subsequent two-second idle audit
  both match the configured loads: grip 0.95 kg, CoG [0,0,60] mm; view 0.55 kg,
  CoG [0,0,90] mm, with `backstops_match=true` for both. Thus this warning is not
  evidence of a wrong retained load or the cause of the gripper symptom. No
  payload or force setting was changed. The idle audit also still reads an
  82 mm gripper opening with no monitor error (`gripper_idle_readback_020.json`).

The production process, runtime sources, drivers, services, and other users'
processes remain untouched by our integration. The operator subsequently reset
the arm and returned it, and trials 016–018 confirmed measured response through
our Dora action path. No hardware/software repair is established. Any future runtime
instrumentation/restart requires coordination with its owner.

### Approach pose compared with the recorded demonstrations

A read-only audit of all 51 drawer recordings (49,259 frames) compared trial
018's refused observation with recorded poses. The five closest recorded positions
are 10.9–16.2 mm away, at frames 42–77. Their measured openings are about 82 mm,
versus 73 mm in the live observation: the policy has begun partial closure earlier.
Across the recordings, the median position at the first target below half-open is
[+43.7, +14.3, −108.5] mm relative to this stopped pose. This supports interpreting
the current pose as an early approach; it is **not a commanded correction**, an
object-localization estimate, or proof of contact clearance.

The median demonstration is 38.04 s with 1.572 m of summed translation commands.
The 99.9th percentiles of per-row translation norm, rotation norm, and successive
gripper-target change are 5.999 mm, 0.01619 rad, and 0.0480, respectively. These
describe the data, not hardware safety limits. A full 5×-slowed task would require
a substantially longer horizon than the commissioning profiles. Evidence:
`drawer_grasp_stage_data_audit.json`. No complete-task trial is represented by
these statistics.

At trial 019's stopped pose, the three closest recorded positions are 3.05–5.15 mm
away, with orientation differences of 4.27–4.79 degrees. The median first
half-open target in the recordings is still [+33.6, +22.1, −31.6] mm away.
These remain descriptive comparisons, not localization or commands; nearest
position alone can also match another phase of an episode. Saved-input checks
of both frozen heads across three inference-noise seeds propose predominantly
positive-x motion: spline +13.13–13.67 mm and waypoint +8.49–12.95 mm over eight
rows. Spline proposes 1.84–5.16 mm downward movement; waypoint 0.29–4.43 mm.
Both propose negative-y rotation. None of these offline predictions was executed.
Evidence: `drawer_trial_019_pose_audit.json`, `drawer_stopped_pose_comparison_027/`.

At trial 021's refused observation, the current TCP z is −98.446 mm. The
unpublished eight-row prefix requests another 9.022 mm downward, with target
opening 70.875–71.230 mm; its fourth proposed position crosses z=−100 mm.
The nearest recorded TCP position is 8.742 mm away with 1.300-degree orientation
difference and measured gripper opening 82 mm, versus 68 mm live. The median
first recorded target below half-open is [+12.294, +19.088, +20.839] mm relative
to the live pose—higher, not lower. These comparisons are not instructions to
move by those offsets, since position matching does not identify the object or
episode phase. The fixed-view pose mismatch is reported at the top of this
document. Images at the nearest recording frame and the refused live observation
were inspected in both streams. Evidence: `drawer_trial_021_pose_support.json`,
`drawer_pose_reference_031/`; script `audit_apollo_pose_support.py` is read-only.

The current gripper discrepancy is more specific than a complete failure to
respond: its first command moved the fingers, but later changed targets did not
produce a changed readout. Existing traces do not expose the SDK command return
code or last successful gripper-poll timestamp. The next diagnosis should separate
physical contact, repeated-command delivery within a session, and polling before
assuming a learned-policy or hardware cause. No production instrumentation,
settings, force/speed change, or runtime restart has been performed.

### Saved-pose comparison of both drawer heads

`drawer_stopped_pose_comparison_019/` applies both frozen checkpoints to the same
saved state and images from trial 014's calibration. The following ranges are
across three inference-noise seeds, not physical trials or training replications;
they use the deployed training-hemisphere adapter. No actions were published.

| Head | First requested opening | Net downward increment across eight rows |
|---|---:|---:|
| Spline | 0.820–0.850 | 18.6–22.7 mm |
| Waypoint | 0.793–0.825 | 14.2–21.1 mm |

Both heads propose continued descent and partial closure, so changing heads alone
does not resolve the missing gripper response. Neither result establishes a grasp
or authorizes execution. Keep the frozen checkpoints and original trial logs.

## Isolated installation

SSH: `ssh mavis-v2@10.66.241.33` using the lab-provided authentication. No credentials
are stored in this project.

Deployment root: `/home/mavis-v2/simba/vla_bspline_hardware_20260912/`.

| Directory | Contents |
|---|---|
| `bundle/models/` | `drawer_assembling_spline`, `drawer_assembling_waypoint`, `cabinet_assembling_spline`, `cabinet_assembling_waypoint` |
| `bundle/hf_cache/` | Offline tokenizer and model configuration cache |
| `source/src/` | Archived training-time LeRobot source |
| `vendor/policy-node/` | Isolated clone of the mentor's reference Dora client, revision `f4763bec9259d225eb119453a9908b40e8b5d0af` |
| `.venv/` | Independent inference environment; physical GPU 1 is used |
| `code/` | Project predictor, observation adapter, guarded plugin, launcher, and tests |
| `reports/` | Raw checks, camera snapshots, predictions, logs, dependency manifests |

The training source archive SHA256 is
`39cbb16ea1116dfb617b8cf43163f6d56a8b08213cb68fc1cd897d0d5821c9b1`.
The source models, processors, original datasets, and weights were not changed.
The lab's runtime, environment, dataflow, ports, UI, and other repositories were
not modified or restarted. Do not run `uv run`, package installation, runtime
tests, `dora up/down/destroy`, or broad process-killing commands in the live stack.

The isolated environment uses Torch 2.11.0+cu128, torchvision 0.26.0+cu128,
transformers 5.5.4, and dora-rs 1.0.1. `uv pip check` passes for all 113 installed
packages. Exact packages are in `reports/inference_environment_cu128.txt`.

## Supervised shadow startup diagnosis

Session `1e6c4e8ebcbc4d0ab4d4b86ebc6d9a3a` used the drawer spline, both arms,
external inference, base-frame actions, `keep_current`, and speed scale 0.1.
The shadow policy had no execution authorization and published **zero** actions.
The helper created this session once and deleted only that same session on exit.
It did not send `return_home`, a recovery request, or a direct SDK command.

| Time (EDT) | Observation |
|---|---|
| 20:26:17.012 | Runtime started hardware bring-up |
| 20:26:20.805 | Runtime reported bring-up complete |
| 20:26:20.860 | First policy observation refused: constant parked-arm features differed from training |
| 20:26:20.944 | Manipulation-arm driver reported `source servo, code 9`; controller mode 1, state 4 |
| 20:26:20.965–20.978 | Existing runtime briefly auto-recovered; the policy remained disarmed |
| After fault detection | Helper closed its session without return-home; later read-only checks returned no active session |

Code 9 here is the SDK **STATE_NOT_READY return code**, not controller error C9:
the sampled controller error register was zero. The runtime's own held-posture
servo stream encountered the error, not a learned command. Its warning suggests
checking UFACTORY Studio; the warning alone is not a diagnosis. A subsequent
read-only socket check found an `ufactory-studio` process connected to the
manipulation controller on port 18333, while only the runtime was connected to the
perception controller. This is a specific lead, not proof Live Control caused
the fault. Ask the operator/mentor to confirm Studio has released control.
Do not kill Studio, clear controller faults, or change the production driver
automatically. The lab already documents a historical servo-entry race and a
bounded 0.3 s readiness grace; this attempt does not establish whether that race,
Studio control, or another cause produced the stopped state.

Reported posture changed from pre-enable idle readings to the first active sample:

| Quantity | Manipulation arm | Perception arm |
|---|---:|---:|
| Largest joint-coordinate change | 0.011984 rad (0.687°) | 0.005324 rad (0.305°) |
| Rail position before / active | 0.635 / 0.635 m | 0 / 0 m |
| Gripper opening before / active | 0.988095 / 0.988095 | no physical gripper |

Post-teardown **idle** readings agree with the active readings to about 2e-5 rad.
Thus the difference is not explained simply by switching idle/active telemetry
APIs. The source does use plain `get_servo_angle` for idle monitoring and
`is_real=True` to seed an active driver; encoder/report semantics or physical
settling still need operator/mentor confirmation. Do not infer a cause or command
repositioning from this table alone. The perception arm's training state was
exactly constant, so its changed readings exceed the 1e-3 input compatibility
band. That band was **not widened**. A follow-up shadow observation must pass
before considering any learned motion.

A read-only Dora observer after teardown independently reproduced the input
refusal, with no session creation or action output. Its perception-arm position
differs from the training constant by −2.022 mm in y and −4.214 mm in z; six joint
features and four legacy quaternion components also exceed the compatibility
band. Raw state, both camera snapshots, and named residuals are preserved in
`drawer_post_session_observer_004/`. This confirms a persistent input mismatch,
not merely an exception during one policy call.

Our client now records a refused observation's state and camera images and names
the failing fields. Three new isolated tests verify the diagnostic, zero
publication on refusal, and disarming even if saving the diagnostic fails. The
shadow-session helper now returns a nonzero exit status for an observed fault;
the original attempt's JSON correctly recorded `SESSION_NOT_RUNNING` despite
its old zero shell exit code. Original reports remain intact.

The handheld tracker also reported no tracked device / `dongle_present: false`.
Do not rely on its buttons as a stop control until the operator verifies them.
The physical E-stop and normal lab controls remain the operator's responsibility.
The production runtime and Dora processes were not restarted or edited.

### Constant-feature diagnosis after Studio was closed

Both drawer normalizers have exactly 17 zero-variance state fields: the grip
rail and all 16 perception-arm fields. Their saved minima, maxima, and means
agree. The archived LeRobot processor uses `(state - mean)/(std + 1e-8)`.
A 1 mm rail difference alone becomes about 100,000 in normalized coordinates;
the post-startup snapshot's largest normalized constant feature is 596,213.375.
These are input-conditioning diagnostics, not physical distances or robot errors.

An offline ablation restores only those 17 fields to their saved training values,
leaving all varying manipulation-arm features and both RGB images unchanged.
For both drawer heads, the archived fixture input and repeated prediction remain
**bit-identical**. On saved pre/post-startup inputs, three fixed noise seeds per
head produce finite outputs. The post-startup spline's first-row translation
norms are 2.424 / 4.089 / 4.706 mm and its gripper command stays fully open; these
are proposed actions, not measured motions or a success result. Without this
projection the same inputs yield gripper changes of 0.231–0.269, which would
exceed the first-trial gripper-change budget. The original guarded adapter had
already protected near-reference inputs; this ablation is **not** a comparison
against actions that were ever deployed.

Evidence: `drawer_constant_features_001/{report.json,predictions.npz}`; script
`diagnose_apollo_constant_features.py`. No weights or normalizers were edited.

The passive observer then used the candidate projection on fresh inputs and
produced five finite spline predictions in 196–217 ms (median 204 ms). First-row
translation norms were 0.353–3.052 mm; all five gripper commands were 1.0.
Its label is `CONSTANT_DIAGNOSTIC_COMPLETE_NOT_EXECUTION_READY`, with zero action
messages and zero runtime API writes. The observer logged local input-queue
drops while computing; its state/image freshness checks were retained, and this
test does not establish loss-free streaming. Camera snapshots and raw inputs
are retained in `drawer_constant_observer_005/`.

`run_apollo_dora.py --shadow-constant-diagnostic` now permits this hypothesis test
inside a future explicitly approved shadow session. CLI and constructor reject
combining it with motion authorization, and `_may_publish` independently refuses
publication in this mode. Default execution behavior is unchanged. Three added
tests cover forbidden execution configuration, retaining raw mismatched inputs
without publishing, and invalid-quaternion refusal. Total: 50 passing isolated
tests. This is preparation, not another hardware session.

Before moving, independently establish an operator-approved parked posture and
validate its stationary telemetry. Do not turn the diagnostic into a way to hide
an arbitrary camera/rail change. The two subsequent startup/holding checks below
used this no-publication diagnostic; actual motion still requires a separate
calibrated input check and a new explicit go-ahead.

### Two successful shadow hold checks and observed startup jerk

| Check | Session 002 (20:56 EDT) | Session 003 (20:59 EDT) |
|---|---:|---:|
| Running telemetry duration | 15.003 s | 15.003 s |
| Running telemetry samples | 367 | 368 |
| Predictions / action messages | 45 / 0 | 45 / 0 |
| Prediction latency min / median / max | 194.7 / 212.1 / 239.2 ms | 192.6 / 213.9 / 233.7 ms |
| Reported faults, stale arms, collision blocks during hold | 0 | 0 |
| Largest startup joint-coordinate change, manipulation arm | 0.006654 rad | 0.006648 rad |
| Largest startup joint-coordinate change, perception arm | 0.006437 rad | 0.006395 rad |

The largest startup changes are about 0.38°. Within each running interval the
reported joint coordinates remained exactly unchanged; this describes telemetry,
not independent metrology of zero physical movement. Rail positions remained
0.635 m / 0 m, and the physical gripper opening remained 0.988095. Both helpers
created one session, deleted that same session after the requested interval, and
verified there was no active session afterward. No return-home, learned action,
fault clearing, production-code edit, or runtime restart was requested.

The user watched the repeat and reported a click and small jerk at startup,
without a significant movement elsewhere. Thus the startup transition is real,
not just a model-input exception. It happened with zero learned actions. The
relevant existing driver path enables the motors, enters servo mode, reads
`get_servo_angle(is_real=True)`, and seeds both target and last-sent hold position
from that value. SDK 1.18.5 routes this read to `get_joint_states(num=1)`, whereas
the ordinary idle read uses `get_joint_pos()`. This is a specific place to review
with the mentor, **not a demonstrated root cause** or authorization to edit the
driver. Do not repeatedly enable/disable to investigate without approval.

The official [UFACTORY servo guide](https://docs.supportarticle.ufactory.cc/support_articles/developer/ufactory-servo-mode-guide.html)
states that servo-mode speed/acceleration parameters do not control the motion;
the host must supply smoothly spaced targets. The runtime's 10% setting scales
its target-step limits, but does not itself certify a smooth enable transition.
No servo settings or safety thresholds were altered here.

Repeat startup video: `drawer_startup_video_003/`, 898 view frames and 897 grip
frames. Original JPEGs and timestamp/sequence indices remain on the lab host.
`startup_transition.mp4` is a side-by-side preview (view left, manipulation wrist
right), covering startup and roughly the first five seconds of the hold; it is
not the full 15-second hold. Its nominal 30 fps is for viewing, while the raw
indices retain actual timestamps. Selected before/after frames were inspected;
the fixture and both drawers remain visible. Neither video nor a clear runtime
gate establishes collision safety for later insertion.

Client reports confirm 0 `act_errors`, 0 `actions_sent`, 0 `arm_actions_sent`, and
0 returned chunks in both checks. Input-queue drops occurred during prediction;
state/image freshness checks remained active. Maximum observation age after
inference was 0.245 s in these reports. Do not call the stream loss-free or these
checks task-completion results.

## Observation compatibility

### Session-local parked-input calibration

The operator confirmed that the click and small enable jerk are normal startup
behavior on this setup. This resolves the need to investigate that behavior
before another approved check; it does not establish task success or approve
movement. The remaining zero-variance input problem is addressed by separating
the **observed physical reference** from the **constant model-input values**.

`--confirm-parked-setup` is a new opt-in mode, requiring operator approval of the
current parked camera arm and rail positions. It does not create a session or
enable motion by itself. After startup, `SessionParkedStateAdapter` requires at
least 11 fresh observations spanning 3 s, with 31 raw state fields stationary
within the 1e-3 native-unit band. The active manipulation gripper opening is not
a parked field: it remains a measured policy input, validated in [0,1], with the
original separate action-change limit. Equivalent quaternion signs are compared
on the same hemisphere when checking stationarity. The reference is held only in memory and bound
to that session and runtime epoch; saved reports cannot be reloaded as approval.
Subsequently, the 17 parked fields must remain within the same band of that live
reference. The model receives their saved training constants, while varying
manipulation-arm state and both images remain unchanged. Actions are not modified.
This guards stationarity, not physical clearance or camera-view equivalence.

Movement during calibration, changed parked hardware, invalid poses, interrupted
observations, or a changed session/epoch invalidate the reference without an
automatic retry. All original action/freshness guards remain. For commissioning,
this mode permits only **one predicted 40 ms row**, and only with the separate
explicit session and supervised-motion flags. Calibration itself cannot publish,
and does not extend the 10 s authorization deadline. Missing audit output or a
failed audit write also prevents publication. The shadow diagnostic cannot be
combined with this mode.

Offline replay of the two saved holds establishes references after 3.280 s and
3.261 s; each then checks 35 sampled inputs with no parked drift. Telemetry agrees
with the saved Dora snapshots within 4.62e-7 per field. On those snapshots, the
resulting model input is bit-identical to the earlier no-action diagnostic input.
These are archived-data checks, **not a new live calibration or motion trial**.
The isolated suite also checks the actual mocked Dora publication path: no
messages during calibration, exactly one `action_grip` row afterward, no second
row. No production stack, driver, weight, or normalizer was changed.

### Live calibration and orientation-boundary diagnosis

| Session | Outcome | Learned actions |
|---|---|---:|
| 004, `91e24990bd8d4fa2bfcb4dba8e7645c2` | Initial calibration refused an active-gripper readout change, 83/84→1.0; no joint/pose/rail difference in the refused snapshot | 0 |
| 005, `5e785dde375743f58c7a97e7340b7080` | Corrected parked-feature calibration passed; 34 predictions, median 208.4 ms, max post-inference observation age 0.242 s | 0 |

Both were approved prediction-only holds. Each ran for approximately 15 s after
bring-up and was closed without return-home. Telemetry contains 370 running
samples per hold, with no reported faults, stale arms, recovery, collision blocks,
or action receipt. All clients exited. Session 004's old calibration incorrectly
treated the **active** gripper opening as a parked feature. Its 0.011905 readout
change corresponds to 1 mm in the G2's 84 mm opening convention; this observation
does not distinguish sensor quantization from small physical gripper settling.
The correction excludes only that active field from the stationarity test; it
does not overwrite its measured input or relax the predicted gripper-change limit.

Session 005 exposed a separate input-representation problem. The legacy grip
quaternion crossed the positive-w boundary after the startup posture changes:
qx/qy changed sign, moving their normalized values from about +0.2/+0.2 to
−121.4/+121.3. This is a representational discontinuity in addition to the small
physical posture change, not a 120-standard-deviation physical rotation. Across
all 51 original drawer recordings (49,259 frames), only three grip quaternions
lie opposite the dominant hemisphere; these are frame 243 of
`20260910T080731.605Z-2162c2` and frames 213–214 of
`20260910T081658.413Z-11c914`. No recordings or weights were changed.

An archived-input ablation holds RGB, every non-quaternion field, and prediction
seed fixed, changing only q to its physically equivalent −q when closer to the
training mean. Both drawer fixtures and the earlier valid shadow input remain
bit-identical in input and prediction. On the new snapshot, three fixed seeds give:

| Drawer head | Original first-row gripper opening | Equivalent-hemisphere opening | Corrected first-row translation norm |
|---|---|---|---|
| Spline | 0.663 / 0.681 / 0.629 | 0.952 / 0.953 / 0.936 | 2.893 / 4.307 / 4.569 mm |
| Waypoint | 0.721 / 0.823 / 0.744 | 0.969 / 0.970 / 0.957 | 2.316 / 4.500 / 4.749 mm |

Openings are absolute fractions, where 1 is fully open; these are predictions,
not executed commands. The original live session's first-row openings ranged
0.553–0.700 and would fail the commissioning gripper-change bound. The corrected
saved-input proposals fall within the original bound; no bound was widened.

`--align-to-training-hemisphere` enables this observation-only correction, requires
`--confirm-parked-setup`, and retains all original publication guards. Raw snapshots
remain unmodified. Geometric tests verify that rotations are unchanged, and the
guarded adapter matches the ablation's model input exactly on the saved snapshots.
This is not another live check or a hardware task-success result. Native controller
and action conventions, the model, and all normalizer statistics are unchanged.

### Legacy pose features

The local demonstrations predate a robot-stack pose correction. Their state
contains **SDK flange positions** and quaternions formed using the old
`Rx(roll) Ry(pitch) Rz(yaw)` interpretation. Current Dora publishes **true TCP
poses from joint-based forward kinematics**, with the corrected xArm convention.
Feeding those current features directly to the old checkpoints is incorrect.

`apollo_legacy_state.py` reconstructs only the trained observation features:

1. Select the 32 named fields in training order: grip arm, then view arm. Never
   assume the stream's arm order; the live workcell can put view first.
2. Undo the grip tool transform: flange-to-TCP is +0.172 m along tool z and Rz(π).
   The gripperless view arm's TCP is its flange.
3. Recover physical extrinsic xyz Euler angles, then reproduce the old intrinsic
   XYZ quaternion feature. Quaternions use canonical wxyz.
4. Restore exactly constant training features to their saved normalizer values,
   but only after verifying they differ by at most 1e-3 in their native units.
   The grip rail and all view-arm state fields have zero saved standard deviation;
   even tiny roundoff otherwise becomes a large normalized input. A genuinely
   moved parked arm/rail is refused, not silently hidden. This compatibility band
   is **not a physical safety tolerance**.

This transformation must **never** be applied to the predicted actions. The
recorded commands remain TCP translation increments and space-frame rotation
vectors in each arm's base frame, plus absolute gripper openings and rail deltas.
The production executor uses these same conventions. Code evidence:
Apollo core commit `4259524` and its parent (`se3.py`), current hardware `units.py`,
runtime `dora_bridge/publishers.py`, `dagger/step.py`, and `dagger/policy_runner.py`.

An exhaustive numeric comparison against the lab's backfilled copies covers all
100 episodes used by the current training/validation exports (103,346 frames).
Action targets are identical. The 99th percentile maximum state-field residual
is 8.94e-7 for cabinet and 8.79e-7 for drawer. Rare larger discrepancies remain
from SDK/joint sampling differences and old fallback telemetry. The largest
retained discrepancy is cabinet episode `20260910T190441.302Z-a6dd92`, frame 749:
old grip x=0.206 m versus reconstructed x=0.631786 m. It was **not** caught by the
earlier quarantine and remains in the trained cabinet data. No data or weights
were silently replaced. See `recording_state_compatibility.json`; reproduce with
`audit_apollo_state_compatibility.py`. Drawer-first testing does not use that episode.

## Dora contract and motion precautions

| Interface | Exact contract |
|---|---|
| Real RGB inputs | `cam_view_wrist`, `cam_grip_wrist`; RGB uint8, 480×640×3 |
| Inputs not used | Synthetic `*_wrist_cam` / front / top streams and all depth streams |
| Policy state | Named float32[32], poses in `arm_base:grip` / `arm_base:view`, wxyz |
| Read-only idle probe | `observer` placeholder, no declared outputs; derives the same fields from `arm_state` |
| Policy output | Only `action_grip`, float32[K,8]; no view-arm action stream |
| Row layout | dx, dy, dz [m]; drx, dry, drz [rad]; absolute gripper opening [0,1]; rail delta [m], fixed zero |
| Row timing | 0.04 s (25 Hz), independently of the 3.125 Hz policy-call cap |
| Original predicted prefix | Eight rows; first commissioning test returns **only the first row** |
| Required metadata | Current session ID/epoch, monotonic sequence, observation ID, policy version, chunk length, dimension, row dt, image sequence IDs |

The plugin defaults to **shadow**: it returns `None`, not a zero action. Zero
gripper command would mean CLOSE, not HOLD. Execution requires both an explicit
current session ID and `--confirm-supervised-trial`. Defaults are one published
chunk containing one row, a 10 s authorization window, and a 30 s client lifetime.
The existing session must be hardware/external/inference, include both arms, use
base frames, `keep_current`, and at most 10% configured speed.

The commissioning guard refuses nonfinite actions, changed parked channels,
translation path sum >10 mm, rotation path sum >0.15 rad, or gripper change >0.1
within the selected prefix. It does not clip the model into a different policy.
These extra bounds are not collision certification. It also refuses stale states
or cameras, stale session announcements, expired permissions, and repeat
observations. Runtime resets, operator gate events, collision/anomaly events,
connection loss, and session changes disarm it; there is no automatic motion
restart. No fallback to the whole-cell action stream is allowed.

The production runtime retains its gate, workspace checks, delta-row consumption
budget, and command limits. Its digital twin has known blind spots and alignment
error. A trained operator, accessible physical E-stop, clear workspace, and a
review of the proposed movement remain necessary. Stopping this client cannot
instantaneously retract a row already accepted by the runtime; physical stopping
must use the lab's established controls.

## Numerical portability qualification

The original comparison uses `np.allclose(atol=1e-5, rtol=0.005)` on unnormalized
actions; it is a tight reproducibility screen, not a safety criterion.

| Model | Max translation-component difference from H200 (mm/row) | Max rotation-component difference (rad/row) | Max gripper-fraction difference | Original tight screen |
|---|---:|---:|---:|---|
| Cabinet waypoint | 0.00217 | 0.0000113 | 0.000715 | Pass |
| Cabinet spline | 0.00150 | 0.0000327 | 0.000494 | Pass |
| Drawer waypoint | 0.48076 | 0.0000230 | 0.001327 | Review |
| Drawer spline | 0.01090 | 0.0000152 | 0.000904 | Review |

On drawer spline, summing eight rows gives xyz differences of −0.0049, −0.0244,
and +0.0607 mm. On drawer waypoint, y accumulates to 0.9131 mm. CUDA 13.0→12.8
matching did not change these outputs; disabling cuDNN TF32 also did not fix the
waypoint discrepancy. The model has mixed float32/BF16 parameters. The exact
source of the remaining H200/4090 difference has not been isolated. Do not label
the tight comparison all-pass or overwrite its original expected outputs.
PyTorch itself does not guarantee bitwise equivalence across platforms; this is
context, not proof of the cause here. [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html)

The drawer spline is the proposed first commissioning candidate because it has
the smaller portability discrepancy and better offline drawer translation error.
This is not a claim of better physical success. Investigate the waypoint difference
before treating a later hardware comparison as a finalized matched experiment.

## Next supervised steps

1. The operator has approved supervised policy trials and is at the table. Await
   confirmation of the requested demonstration-start reset with the gripper open.
   Do not command a home/reset automatically. Keep the physical E-stop accessible.
2. Use the separate, tested approach profile with a new output directory, fresh
   exact-session grant, and `--confirm-parked-setup --align-to-training-hemisphere`.
   Never reuse an old grant/calibration or replace the original 10 mm default.
3. Record real RGB, commands, and measured state. Confirm idle healthy hardware
   and our grip-only policy before creating one `keep_current` session. Preserve
   the current runtime/driver settings and other lab projects.
4. Inspect the bounded full-chunk response before extending execution toward a
   grasp. Closed-loop approach, contact handling, complete insertion, and scored
   hardware trials still require validation; the short tests above do not establish
   task success. Close only our session, without return-home.

Shadow launcher on the lab workstation (prediction only; unique output name):

```bash
cd /home/mavis-v2/simba/vla_bspline_hardware_20260912
bash code/launch_apollo.sh \
  --checkpoint bundle/models/drawer_assembling_spline \
  --output reports/drawer_shadow_next --run-seconds 120
```

Motion flags require the selected supervised profile and its fresh session grant.
The live runtime under `/home/mavis-v2/apollo-mavis-v2-ws` must remain unchanged.
