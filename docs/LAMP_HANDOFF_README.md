# Lamp policy handoff

Latest commissioning status: see
`docs/LAMP_REPLAY_COMMISSIONING_2026_09_13.md`. Recorded-start restoration was
tested in trials 004–006. A recorded delta-action control reproduced the
orientation discrepancy; the lab's reference documentation independently
records an open delta-EE rotation defect and recommends absolute EE. The first
eight-row absolute control improved angular tracking but lagged in translation
and stopped before its next chunk. The slower absolute control and rail-hold
encoding correction are prepared, **not physically validated**. No learned
grasp/assembly success has been recorded. Xiatao confirms the front table is
not protected by the runtime collision model; marked setup and physical
clearance checks remain necessary.

September 13 evening status: both 40k trainings and all eight checkpoint
evaluations completed. The fixed held-out rule selected 10k for both heads.
The complete cluster bundle was downloaded to
`/Users/simbashi/Downloads/VLA_Hardware_Lamp_20260913/`; both model SHA-256
hashes match the selection and handoff manifests. The bundle was copied
into the new, isolated lab directory
`/home/mavis-v2/simba/vla_bspline_hardware_lamp_20260913/bundle/`.
On the lab RTX 4090, both selected models load strictly and produce valid
prefixes; the observation-adapter round trip leaves fixture inputs and
predictions bit-identical. Both miss the original tight cross-GPU numerical
screen, so its status remains `REVIEW_REQUIRED`, not pass. Maximum per-axis
translation differences from cluster fixtures are 0.00940/0.00945 mm per row
(spline/waypoint); maximum rotation-component differences are
4.154e-5/3.140e-5 rad, and gripper-fraction differences are
0.0005928/0.0006859. These are numerical reproducibility differences, not
measured hardware tracking errors or a physical clearance test. No tolerances
were changed. Physical commissioning remains pending; the existing September-12
deployment and shared runtime are unchanged.

Raw lab report/arrays: `outputs/hardware_lamp_20260913/lab_portability_20260913/`
locally, and `reports/lamp_gpu_bundle_check_v1*` inside the isolated lab
directory. Median prediction latency (excluding first call) is 222.8 ms spline
and 187.6 ms waypoint on this fixture; no robot or Dora API was connected.

At approximately 20:08 EDT, both models also passed three live-input predictions
each through the existing **observer** placeholder, which declares no outputs.
No hardware session was created and no actions were published. Both real wrist
cameras were fresh, and the measured 81 mm gripper opening reached the model
unchanged (fraction 0.9642857). The spline's first proposed eight-row chunk has
net xyz translation [-12.752, -0.426, +0.260] mm; the selected demonstration's
first eight recorded commands sum to [-12.241, -0.002, +0.017] mm. This is an
initial-prediction comparison, not a trajectory rollout or grasp result.

Saved live frames show the lamp fixtures. The reported starting TCP differs
from the selected native replay episode's initial TCP by approximately 0.014 mm
in position norm, with a 0.0010-degree reported orientation difference. This
checks recorded/live feature alignment, not independent physical calibration.
The scene YAML hash remains identical to the configuration used for the earlier
table-coverage audit, and the contact-review hold is unchanged. Runtime session
queries before and after the probes report no active session.

Live reports, images and arrays are in
`outputs/hardware_lamp_20260913/lamp_live_observer_001/` and
`outputs/hardware_lamp_20260913/lamp_live_waypoint_observer_002/`. These live-input
passes do not supersede the separate cross-GPU numerical review above.

At 20:31 EDT, supervised lamp trial 003 executed three eight-row chunks at
200 ms/row. The learned net TCP request was [-3.209, -0.468, +3.245] mm;
measured response was [-2.479, -0.489, +3.341] mm. The gripper changed from
81 to 78 mm, and the view arm and both rails remained stationary during
learned execution. None of 252 running telemetry frames reported a hardware
fault or collision block. The short approach stopped at its planned budget
and the session was deleted without return-home. This is a functional
command-response result, **not a grasp or assembly success**. Video and raw
commands are saved under `outputs/hardware_lamp_20260913/lamp_trial_003*`
and `lamp_approach_003/`.

The two preceding starts published no learned actions: the initial 5 mm
starting-point check refused the controller-startup pose shift. Each start
changed reported joints and lowered the reported TCP by about 8 mm. Before
trial 003's first learned command, the total shift from the original idle
probe was [-8.857, -7.147, -23.176] mm. The revised lamp-specific guard accepts
a small initial region but retains the +100 mm TCP floor, open gripper,
80 mm cumulative translation budget, and three-chunk limit. All 262 local
adapter/contact/replay/selection tests passed. The original drawer contact
hold remains; the operator's lamp clearance only admits this bounded approach.

An archived-input comparison isolates sensitivity to this changed state.
With identical inference seeds (three seeds), the original state produces
12.15–12.85 mm translation path in its first chunk; the enabled state gives
1.42–1.99 mm. Swapping only the saved images preserves this large difference:
original state with enabled images gives 12.18–12.89 mm; enabled state with
original images gives 1.56–1.99 mm. The crossed inputs are offline diagnostics
only, never a proposed substitute for actual measured state in deployment.
Raw report: `lamp_startup_sensitivity_spline_001/` under the same outputs root.
Trials 004–006 subsequently restored the recorded initial configuration
**after** controller startup and retained that session. Trial 004 produced
16.43 mm measured TCP displacement over 24 learned rows, but orientation
tracking remained problematic. See the commissioning report above.

The matched waypoint checkpoint shows the same qualitative state sensitivity.
Mean first-chunk translation path across the same three inference seeds:

| Saved state | Saved images | Spline (mm) | Waypoint (mm) |
|---|---|---:|---:|
| Original idle | Original idle | 12.590 | 12.699 |
| After startup | After startup | 1.737 | 2.679 |
| Original idle | After startup | 12.582 | 12.298 |
| After startup | Original idle | 1.718 | 3.030 |

The last two rows are crossed-input diagnostics, not real observation pairs.
Waypoint provenance is saved in `lamp_startup_sensitivity_waypoint_001/`.
These comparisons establish state sensitivity, not the physical cause of
startup movement or proof that initialization alone will complete the task.

The existing `goto_initial` playback API can restore the first recorded frame
through its planned, gated reset path; do not confuse it with the generic
default home profile, which is different. The audited lamp episode is
`bc_demo/lamp_assembling`, `20260911T204950.391Z-538dc8`. Its reset targets
both arms and rails (recorded manipulation rail ~0.636 m, current ~0.635 m),
so the operator must have clearance for that setup motion too. The three
approved resets were completed; only finite approach prefixes, not a full
replay, were sent by this commissioning code.

This bundle has passed a short supervised command-response check, not full-task
physical validation.
`handoff_manifest.json` names the actual chosen checkpoints and their hashes.
`checkpoint_selection.json` preserves every candidate's held-out transition
errors and the fixed selection rule. Inspect it before selecting a deployment.

Important differences from the September-12 drawer bundle:

- The lamp models expect **corrected TCP observations**, not legacy SDK flange
  features. Preserve `apollo_observation_contract.json`. Do not run them through
  the unmodified September-12 live-to-legacy adapter.
- The physically parked view arm and rails retain the same monitoring band.
  One recorded view-quaternion component has rounding-level variation between
  native/backfilled recordings (std ~1.7e-8). The updated input adapter projects
  the 17 parked channels to checkpoint reference values after validating their
  recorded ranges. Session calibration still monitors the real hardware.
  Active-arm position/orientation/gripper measurements are not replaced.
- Prediction fixtures and final offline checks use this recorded-input
  stabilization. Each fixture also retains its raw recorded state. Compare
  GPU outputs with the saved fixture before connecting the policy to Dora.
- Both RGB streams are 640x480 HWC uint8: `view_wrist`, then `grip_wrist`.
  Actions remain native base-frame delta translation/spatial rotation and an
  absolute opening fraction. The task caption is `Lamp Assembling`.
- The predictor returns up to eight command rows. Recorded dense time is
  40 ms/row; actual recordings contain pauses omitted from that timeline.
  Prediction frequency, command-row duration and wire metadata are different
  quantities. Match the runtime's delta scaling exactly; never repeat a stale
  increment or scale an absolute gripper opening as a velocity.

Use a new isolated deployment directory. Preserve the existing contact-review
hold and shared lab runtime. No checkpoint download, successful offline test,
or demonstration replay automatically authorizes hardware motion. On-site
work-surface/finger clearance review and supervised commissioning remain
necessary. The drawer-specific full-task envelope is not a lamp-task envelope.

The recorded-data packet is separate at
`reports/replay_packet_v3/` in the training workspace. It contains the first
non-backfilled training example, both command forms, both timelines and hashes. Start with the
mentor's recorded-demonstration replay support, with identical setup and explicit
timing, before evaluating learned behavior. This packet contains no executable
grant and does not move to its initial pose automatically.
