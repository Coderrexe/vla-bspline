# Apollo hardware inference integration

## Current status

September 12, evening EDT: all four trained checkpoints are on the lab workstation.
The drawer spline has executed a sustained physical approach using full action
chunks. Slowing row execution from 40 to 200 ms preserves commanded increments
while reducing their requested speed; this addresses the live executor's restrictive
motion caps without changing those caps. The latest run executed six eight-row
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
The next step is to establish reliable gripper response before approaching contact. The production
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
| Isolated adapter / publication tests | 128 passed, including native/slow wire timing, session grants, cumulative budgets, approach profiles, and deterministic gripper checks; repeat-hold cannot accumulate closure; mocked transport only | `adapter_tests_v21.xml` |
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

The runtime log specifically reports 0.04 m/s TCP and 0.06 rad/s joint caps,
with capped ticks during native and some slow motion. Fivefold time stretching
does not guarantee all caps are inactive. Trial 012 nevertheless tracks its
requested trajectory much more closely than the native commissioning segment.
These trials began at different poses and used different predictions; **they are
not a matched hardware timing ablation or an architecture comparison**. Pure-method
replay confirms unit integrated delta bookkeeping at both clocks:
`runtime_delta_cadence_slow_v2.json`.

One issue to resolve before grasp validation: trial 013's gripper readout remained
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
processes remain untouched. Confirm whether the existing lab teleoperation path
can operate the gripper; if it cannot, resolve that shared hardware path with the
runtime owner before another grasp attempt. Any runtime instrumentation/restart
requires coordination with that owner.

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
