# Hardware closeout plan — September 13

## Objective and present constraints

Prepare a reliable supervised hardware evaluation within the user's remaining
two-day submission window. No robot session creation, controller enabling, policy
execution, demonstration replay, or automatic recovery while the operator is away.
The post-contact launcher hold remains in place. Shared lab software and existing
checkpoints are not to be overwritten.

## Priorities and decision gates

| Priority | Work now, without robot motion | Gate before the next step |
|---|---|---|
| 1 | Audit all 52 lamp episodes: numeric integrity, video synchronization, observed task completion, pose/action convention, stationary channels and initial setup | Resolve any mixed legacy/corrected TCP features before export; freeze an episode-level train/validation split |
| 2 | Train matched lamp spline and waypoint policies from the same pretrained initialization; retain intermediate checkpoints and evaluate grasp/placement transitions as well as broad prediction error | Select checkpoints using recorded-data validation, not a lucky physical trial; package input/output contracts and hashes |
| 3 | Compare existing drawer checkpoints at 5k/10k/15k/20k on identical grasp-transition observations, with training and held-out errors separate | Continue training or change transition sampling only when the diagnostic identifies a specific lever |
| 3b | Give cabinet its own replay and stage diagnosis using its already verified spline and waypoint checkpoints | Do not assume drawer's failure mechanism applies to cabinet; localize the first failed stage before changing its training |
| 4 | Prepare immutable demonstration replay inputs and expected trajectories; validate delta integration and absolute-label conventions offline | On-site inspection and verified work-surface/tool clearance, then operator-approved replay through the same execution path intended for the policy |
| 5 | Draft the hardware methodology and evaluation sheet, preserving the manuscript's two central claims | Fill outcome counts and rollout figures only from completed, scored physical trials |

The new training workspace is `~/vla_hardware_20260913` on
`ssh misha`. Use Slurm for video processing, dataset construction and training;
do not compute on login nodes. Reuse a copy of the pinned September-12 policy
source and pretrained initialization. Keep raw lamp data, exports, source,
checkpoints, logs and reports in separate directories. Initially allow at most
two full training jobs plus one short checkpoint diagnostic; expand only in
response to evidence, not a broad parameter sweep.

Storage note: the initial scratch transfer hit the group's 15,000,000-file limit.
Only the newly created partial September-13 workspace was relocated to the user's
home allocation (48/125 GiB and 306,817/500,000 files at the check). All earlier
data/checkpoints remain untouched. Bound new checkpoint storage to fit the remaining
quota and download selected deployment artifacts promptly.

## What Xiatao's replay test can establish

A successful replay is an important control. Match the recording's initial arm,
camera, rail, gripper and object configuration; replay recorded commands through
our exact Dora action path, clock and command interpretation; compare measured
pose/gripper traces with the recording and inspect the actual grasp/placement.
Keep delta replay separate from absolute-pose replay: they test different command
paths. The filtered recording's encoded 25 Hz time is not necessarily the original
wall-clock time. Record the chosen timing and do not silently multiply deltas by
timestamp gaps.

Passing that control supports execution fidelity for that trajectory and setup.
It does not by itself validate live camera/state preprocessing, model feedback
timing, off-trajectory recovery, all workspace geometry or task generalization.
Replay success is not learned-policy success. The current table coverage defect
must be resolved before either replay or learned execution near the surface.

## Focused paper contribution

The existing lamp recordings have a whole-task caption. A lamp success would
support real-robot deployment of the representation, not clause steering or
compositional generalization. Prepare all three task policies for supervised
commissioning. Lamp is the shortest route to establishing the complete deployment
pipeline; drawer and cabinet have separate recovery tracks below. The target is
complete assembly on all three, with matched head comparisons. Allocate repeated
evaluation to tasks that can be commissioned safely within the time budget, while
retaining an explicit record of incomplete tasks and their observed failures.

Once basic completion works, the most contained representation-specific extension
is a paired execution-clock comparison on the same frozen spline checkpoint,
with a matched waypoint timing control. If there is insufficient time for that
control, do not claim a hardware timing advantage over waypoints. Use only lab-approved
rates within controller limits. Clause experiments require actual stage labels
and a trained clause-conditioned model; do not imply the existing task-caption
models already implement that intervention.

Aim for approximately 0.75–1 page: one clear hardware/setup and rollout figure,
one compact table of full-task counts, and a short explanation of any focused
intervention. Define success before evaluation: correct placement, released
object remains seated, and no human correction. Also record grasp, placement,
interventions, contact stops and elapsed time separately. Keep development trials
distinct from a frozen evaluation protocol. Use matched initial configurations,
report x/n rather than percentages alone, and disclose any training-set changes.

## Pending lab handoff

Evening update, verified September 13 at 19:53 EDT: both lamp trainings,
all eight transition evaluations and packaging completed successfully. The
fixed rule selects 10k checkpoints for both heads. See `paper/results.md`
Section 10.2 and the launch ledger for all candidate results. The models are
offline prediction-verified; no learned lamp execution has yet been evaluated.
The user is now in the lab with the lamp fixture set up. Model transfer and
prediction-only portability checks do not release the post-contact motion hold.

Completed offline work:

- All 52 lamp episodes / 23,401 frames passed full video decoding, numeric and
  FK pose-convention checks. All use corrected TCP observations, including five
  backfilled recordings. All 13 two-camera contact sheets were reviewed (six
  sampled times per episode); no visual exclusion was identified. This is not
  frame-by-frame manual video scoring or an independent physical success audit.
- The task is grasping the tapered shade's rim, lifting, placing over the post,
  releasing and withdrawing. No 180-degree object flip occurs in these demos.
- Export is numerically lossless; 47 training / five held-out episodes. Spline
  reconstruction and inactive-output checks passed. Models 2466234/2466235 were
  training at the last successful cluster check (approximately 11:50 EDT,
  September 13); drawer checkpoint audit 2466201 completed. A subsequent
  read-only SSH status request was denied authentication, so newer lamp status
  is not yet verified. Full launch records live
  in `cluster/launch_records/hardware_20260913.md`.
- All 52 episodes pass delta/absolute command consistency. Maximum consecutive
  target residual is 0.000061 mm translation and 1.44e-8 rad orientation.
  Five absolute tracks were backfilled from deltas, so their agreement is not
  independent evidence. Across all episodes, maximum command-to-next-retained
  measured TCP discrepancy is 5.79 mm. These are recorded-data checks only.
- Original wall-clock pauses reach 3.761 s; dense training time omits them.
  The selected replay episode (first non-backfilled training example) is
  `20260911T204950.391Z-538dc8`: 521 rows, 20.84 s dense duration and 35.133 s
  recorded wall-clock span. It closes at frame 202 and releases at frame 455.
  `outputs/hardware_lamp_20260913/replay_packet_v3/` preserves both timelines,
  original delta/absolute arrays, observed state, and source hashes.
- Corrected-TCP input routing and the unchanged legacy drawer route pass the
  local isolated suite: 239 tests including replay-label and selection checks. No motion
  limits were loosened. The new contract must accompany every lamp checkpoint.
- Automatic transition validation is job 2466263; selection/packaging is
  2466270. The fixed selection rule is recorded in the launch ledger. The
  manuscript-ready setup/protocol draft is `ICRA/hardware_draft_2026_09_13.tex`,
  not yet included in the main paper and containing no invented outcomes.

On-site sequence: inspect hardware/work surface; establish a physically clear
initial pose and object layout; run a recorded demonstration through the existing
replay support with the operator present; check tracking and actual completion;
run new-model shadow input checks; then supervised policy development. Record
whether replay uses dense or wall-clock timing. Original-timing replay alone does
not validate a differently timed dense-policy path. Do not multiply delta rows
by omitted-pause duration. No automatic session startup or reset is permitted.

## Evaluation to freeze after commissioning

### Completed drawer checkpoint comparison

Job 2466201 finished in 8m46s on an A40. All eight checkpoints used identical
images, original state/action arrays, five held-out episodes, six offsets
around first closure, and three inference-noise seeds. Training observations
were evaluated separately, not pooled into the table below. Errors are in mm.
"Mean xyz" averages the per-observation eight-step cumulative xyz-component
RMSE across the six offsets; it is not physical endpoint tracking error.

| Head | Updates | Gripper MAE, all offsets | Gripper MAE, boundary | Mean xyz RMSE |
|---|---:|---:|---:|---:|
| Spline | 5k | 10.38 | 21.48 | 1.560 |
| Spline | 10k | 10.25 | 22.88 | 1.357 |
| Spline | 15k | 10.28 | 23.72 | 1.196 |
| Spline | 20k | 10.09 | 25.39 | 1.046 |
| Waypoint | 5k | 10.41 | 24.40 | 1.715 |
| Waypoint | 10k | 9.71 | 26.21 | 1.045 |
| Waypoint | 15k | 9.83 | 27.91 | 0.987 |
| Waypoint | 20k | 9.67 | 27.89 | 1.062 |

There is no earlier checkpoint that uniformly improves pose and gripper
prediction. More training improves approach prediction and training-set grip
errors, but the held-out closure-boundary error does not improve. This does not
prove that collecting more data is necessary, or predict physical success from
five validation episodes. It does argue against simply swapping to an earlier
checkpoint as a complete solution.

Original frames for a late-closing and early-closing case were inspected in
`drawer_late_closure_hydrated.jpg` and `drawer_early_closure_hydrated.jpg` under
`outputs/hardware_lamp_20260913/`. Near grasp, the wrist view looks mainly into
the drawers and the arm partly occludes the knob in the second view. In one
case the recorded opening is 79 mm and the eight-row target averages 37.3 mm,
but the 20k spline predicts 75.3 mm. After the recorded gripper has already
closed to 50 mm, its prediction falls to 25.4 mm. Another episode closes earlier
than the expert target, so the error is not uniformly a delay. These observations
motivate one bounded offline input-sensitivity probe (2466282), holding images
and arm pose fixed while varying only the input gripper reading. Perturbed
readings are never to be used for live control. The probe tests model feature
dependence, not physically realizable counterfactual success.

Probe 2466282 completed in 1m31s. Across the five held-out grasp-onset images,
changing only the gripper input from 1.0 to 0.8 lowers mean predicted opening
from 73.00 to 13.42 mm (spline) and 72.81 to 11.25 mm (waypoint). At input 0.5,
predictions average 6.66/5.54 mm. This establishes strong dependence on that
feature, not that the dependence is wholly undesirable or the sole physical
failure cause. In particular, altered readings conflict with the unchanged
images. Do not spoof telemetry, force closure, or infer that a dropout/retraining
change is already validated. Keep the current 20k models as references while
the replay control and lamp candidates are evaluated.

All 13 drawer and both lamp recording-session sidecars specify speed scale 1.0.
Our prior policy commissioning used speed scale 0.1 and, in later trials,
200/400 ms command rows. Those conditions are not identical. Ask Xiatao for his
successful replay source, timing mode and speed settings. Reproduce an approved
working recipe and compare measured response; do not autonomously raise the
robot speed or assume that slower playback preserves every feedback interaction.

If the operator arrives before the 40k sweep finishes, an already completed
common intermediate checkpoint can be copied into an immutable, correctly
annotated view and prediction-validated for development. A final evaluation
must still freeze its checkpoint first; later validation choices must not be
retroactively mixed into that trial set.

### Physical evaluation protocol

1. Start with lamp assembly. Use 10 matched starting configurations per head
   as an initial evaluation block (20 trials); expand to 20 per head if lab time
   permits. Alternate head order across pairs and keep the same allowed start
   variation for both. Freeze model hashes and the execution rate first.
2. Count complete success only when the shade is correctly seated, released,
   remains seated for three seconds, and no person corrected the trajectory.
   Report grasp, placement, operator interventions and stops separately. A stop
   or intervention is a failure for autonomous full-task success, not an omitted
   trial. Paired layouts do not make stochastic rollouts identical experiments.
3. Commission drawer and cabinet separately using the recovery tracks below.
   Drawer success requires both drawers seated in their intended slots and
   released; cabinet success requires both components seated in their intended
   fixture positions and released. Confirm these criteria against the fixture
   with the operator before freezing trials. Keep task counts separate. Record
   uncompleted commissioning explicitly; do not imply an omitted task succeeded.
4. If there is time after successful baseline evaluation, compare one additional
   lab-approved clock on the same spline checkpoint and the matched waypoint
   timing control. Do not change clipping, replanning, limits and rate together
   and attribute the result solely to clock control.
5. Save every counted trial's checkpoint, setup image, prompts, intervention log,
   actions, measured states, video, duration and result. Do not mix commissioning
   attempts or validation-selected checkpoints into a frozen-test denominator.

## Remaining motion prerequisites

The operator has forwarded the scene-coverage finding to Xiatao. Required before
motion: physical inspection following trial 025, verified geometry for the actual
task work surface and installed fingers, clearance validation including startup
motion, and an explicitly reviewed trial/replay plan. No automatic release of
the contact hold is authorized by this document.

## Two-day recovery and submission plan

This is a proposed execution schedule using the user's remaining two-day budget,
not a claim that new repairs or physical successes have already been obtained.
The official submission cutoff and timezone must be checked before fixing the
final freeze times. No task count or success rate guarantees acceptance.

### Task-specific recovery

| Task | Verified starting point | First decisive test | Bounded repair if the test identifies a policy error |
|---|---|---|---|
| Drawer | Both 20k models are packaged. Eight checkpoint comparisons do not identify a uniformly better earlier model. Grasp closure can be early or late, and predictions depend strongly on the measured gripper input. | After the contact review, replay a drawer demonstration and compare execution with the recording. Then record the first learned failure stage using synchronized camera, command, measured-pose and gripper traces. | If execution matches but grasp initiation/recovery fails, collect a small targeted batch around the observed starts and failed transitions, retain the original training data, and fine-tune both heads under the same protocol. Compare a stage-balanced sampling candidate with the existing model offline before deployment. |
| Cabinet | Both 20k models are packaged from 49 usable demonstrations. Offline translation errors favor waypoint; this is not a physical success comparison. No scored physical outcome is established. | Verify cabinet's own initial setup and replay, then test approach, first grasp, insertion/release, return, and second assembly in order. Do not assume its failure is the drawer failure. | Correct the first demonstrated bottleneck. If a learned insertion or second-object transition fails despite replay success, collect targeted examples for that stage and repeat the same bounded repair protocol for both heads. |
| Lamp | All 52 episodes are audited; matched training and automatic checkpoint validation/packaging are launched. | Verify the selected deployment bundle and a lamp-specific execution profile, replay the prepared native recording, then test full assembly. | Use the simplest complete task to expose any remaining shared input/action/timing issue. Repair a shared issue only when the recorded traces identify it, then recheck the other tasks. |

Specific implementation decisions:

- **Replay and live-input checks are complementary.** A replay bypasses the
  learned observation pipeline. Validate camera identity/order, preprocessing,
  gripper units, state conversion, timestamps and action buffering separately.
  Drawer/cabinet use the legacy observation convention; lamp uses corrected TCP
  observations. Their adapters must remain task/checkpoint specific.
- **Measure command timing rather than guessing.** Record requested and measured
  motion, command acceptance and camera/state age. Use an operator/mentor-approved
  replay recipe and compare its clock with the intended policy clock. The
  previous 0.1 speed scale and 200/400 ms rows differ from recording settings;
  this is a testable mismatch, not an established root cause or authorization
  to increase speed.
- **Localize before collecting.** One clear first-failure diagnosis per task is
  more useful than repeated complete rollouts with the same failure. Partial
  stage starts require their own reviewed safe setup and are diagnostics, not
  autonomous full-task successes.
- **Targeted data is conditional.** An initial 10–15 corrective demonstrations
  per affected task is a planning estimate, not a proven sufficient data volume.
  Include recovery from the actual observed deviations and complete transitions,
  not just extra repetitions of easy approach frames. Keep existing held-out
  episodes out of training; record the added training data and changed sampling.
  If views hide the grasp, first assess whether the existing second view can
  resolve it. Changing a camera requires checking the resulting training/live
  observation distribution and new calibration/data; do not silently move it.
- **One bounded repair cycle first.** Preserve the current checkpoints and use
  separate exports/runs. Choose update budget and intermediate checkpoints from
  the diagnosed issue and available GPU time. Match data, initialization and
  tuning opportunity across heads. Neither transition balancing nor targeted
  fine-tuning has yet demonstrated an improvement here.
- **No forced grasp shortcut.** The offline gripper-input probe does not justify
  spoofing telemetry, removing measured state at inference, or forcing closure.
  If a scripted component is ultimately tested, label it as a hybrid method and
  give the baseline the same component; do not report it as the original policy.

### Schedule and decision points

| Window | Hardware / model work | Paper work and deliverable |
|---|---|---|
| Now, before the operator returns | Harvest lamp jobs when SSH authentication is restored; verify selected models and bundles. Prepare drawer/cabinet stage review and replay references without robot motion. | Freeze the two-claim narrative and reconcile every main simulation number against its report. Keep hardware outcomes blank. |
| First supervised lab block today | Complete contact/work-surface review; establish replay and live-input checks; identify the first failure or complete behavior on each task. Save synchronized traces rather than repeating uninformative failures. | Select the hardware figure layout and trial ledger fields. Write setup, data and model details from verified records. |
| Remaining lab time today / overnight | If evidence supports a repair, collect the targeted batch and run one bounded matched fine-tuning cycle. Validate intermediate candidates and preserve baseline models. No unattended robot execution. | Finish prose, simulation figures/tables, bibliography, claim boundaries and page layout apart from hardware outcomes. |
| Tomorrow, before the experiment freeze | Freeze models/settings; run matched starting configurations with alternating head order. Initial target is 10 trials per head per commissioned task, with 20 per head on the primary task if capacity permits. Record all counted attempts. | Populate exact counts, durations and representative learned-policy rollout panels from the trial ledger. Separate development failures, diagnostic stage tests and frozen evaluation. |
| After basic completion, only if time remains | Run one additional approved execution clock with both heads, preserving checkpoints and other settings. Do not add clause training or a new backbone unless the core evaluation is already complete and the additional comparison can be finished fairly. | Add the physical-clock comparison only if completed; otherwise the timing evidence remains in simulation and hardware establishes assembly deployment. |
| Target: last 12 hours before submission | Stop launching new experiments. Finish harvesting and check artifacts; retain a buffer for failed transfers or rendering. | Reconcile paper/PDF/results ledger, verify denominators and source videos, audit captions and citations, and remove all placeholders. Reserve the final six hours for visual review, coauthor corrections and submission checks. |

The 10/20-trial targets are scheduling choices, not conference requirements and
not a reason to ignore uncertainty. If all three tasks can be evaluated, report
all three. If one remains unreliable, retain its actual status and concentrate
the finite repeat budget on interpretable comparisons rather than selecting a
single attractive rollout. Any changed evaluation protocol starts a new named
block; do not merge it retrospectively with a previous denominator.

### Manuscript endpoint

The central language evidence remains the controlled simulation experiments:
matched whole-task versus clause training/execution, replication across seeds,
and behavioral interventions. The clock claim retains the existing CALVIN and
LIBERO evidence with the matched waypoint retiming baseline. Hardware adds a
compact assembly comparison and, if completed, a physical-clock intervention.
Whole-task hardware captions do not establish clause alignment or compositional
generalization. Broader useful results remain fully preserved in
`paper/results.md`; the eight-page manuscript should select the evidence that
directly supports these claims rather than become a list of capabilities.

Deliver a versioned manuscript/PDF, all referenced figure assets, a complete
results ledger with report/trial provenance, and an Overleaf upload manifest.
Use roughly one page for hardware, including a readable setup/rollout figure
and a compact task-by-head table, while respecting the total page budget
including references. Replace the current placeholder task description with
the actual evaluated assembly tasks. Do not insert unmeasured hardware wins,
reuse teleoperation frames as policy results, or claim broad superiority from
a selected successful video.
