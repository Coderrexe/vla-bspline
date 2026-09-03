# Duration-head calibration v2: locked train-only diagnostic

## Why this is the next high-value duration experiment

The paired duration intervention now suggests that fixed or shuffled durations
can outperform the learned prediction on at least the selected task-2 pilot.
Before spending another large closed-loop fleet, the cheapest discriminating
question is whether the duration head contains useful ordering information but
maps it to the wrong clock. If so, a scalar decode-time calibration can recover
the mechanism without retraining and without changing the sampled B-spline
shape. If not, the project should stop treating the learned duration as an
execution advantage and redesign its supervision/head.

This protocol supersedes `scripts/analysis/calibrate_duration.py` for new
claims. The older analysis was an unsplit on-train sample, maintained its own
event rule, omitted the production episode-end pad event, and did not bind the
dataset language metadata or model/source hashes. Its historical numbers are
useful hypotheses, not sufficient evidence for the current checkpoint. A v2
rerun is needed before reusing the claimed `r=0.84` or MAE values.

## Leakage boundary

All calibration examples come from the original training demonstrations. This
is post-hoc calibration, not held-out policy generalization.

- Whole episodes are assigned within task to deterministic `fit`, `select`, and
  `audit` folds. No episode crosses folds.
- Frame sampling is uniform by salted coordinate rank. Duration/event labels
  do not influence selection.
- Candidate maps are fit on `fit`, selected once on `select`, and evaluated once
  on `audit`. The deployment map refits the already-selected family on
  `fit+select`; `audit` is never used for either selection or refitting.
- Simulator initial states are not opened. The manifest precommits a later
  full LIBERO-Long comparison on tasks 0–9 and states 0–49 before a calibrated
  closed-loop result exists.
- Calibration is global. Task labels are used only for diagnostics, not for a
  task-specific lookup that could memorize the training set.

The prediction dataset must use the checkpoint's original task strings. The
`libero_l10gran` overlay is action-identical and therefore valid for the
action-only `T=24` prior, but it is invalid for this inference: the policy sees
language, so changing captions changes the quantity being calibrated. The
launcher requires the official `tasks.parquet` hash and fails on the overlay.

## Exact targets and measurements

`prepare` scans every action Parquet, reproduces LeRobot future windows and pad
masks, and calls the pinned production `event_targets.py`. This stage does not
load an image or video. The selected manifest contains 32 frames per task per
fold: 3,840 frames across 40 tasks.

`infer` re-queries each selected frame through LeRobot and verifies every action
window, pad mask, event type, and target against the manifest before inference.
The checkpoint is sampled four times per frame. It saves the raw continuous
`exp(mean(logT token))` and the production rounded/clamped result. Images are
unavoidable here because duration is vision-language conditioned, but no image
is written to the result.

`analyze` reports:

- draw-level and frame-mean MAE, RMSE, bias, Pearson/Spearman correlation, and
  true/predicted duration histograms;
- cap-vs-event confusion, balanced accuracy, cap recall/precision, and a
  separately fitted raw cap-discrimination threshold;
- calibration by production event type and all 40 tasks;
- within-frame sampling variance;
- a frame-clustered paired bootstrap for the selected-map MAE improvement.

The predeclared scalar candidates are identity, nonnegative log-affine,
cap-snap, isotonic, and event-isotonic plus cap-snap. A non-identity candidate
must improve selection MAE by at least both 0.25 steps and 5%, while worsening
neither event MAE by more than 0.25 nor cap balanced accuracy by more than 0.02.
The serialized map consumes only raw duration and modifies no pose or gripper
control point.

## Cost

- Action-only preparation: about 15–60 seconds CPU plus one sequential model
  hash; no video decode.
- Inference: 3,840 frames × 4 draws = 15,360 duration samples. Expected cost is
  roughly 0.5–2 L40S GPU-hours depending on random-access video throughput; the
  launcher reserves four hours and eight CPU workers.
- Analysis: under one CPU minute.
- Output: compact JSON, expected under 5 MB; no rendered frames or rollouts.

This is materially cheaper than another 500–2,000-episode simulator fleet and
directly tells us whether such a fleet has a plausible calibrated treatment.

## Safe decision tree

1. **Identity wins selection, or the audit confidence interval includes no
   useful gain.** Do not tune on audit and do not launch a calibrated simulator
   arm. The problem is representation/prediction, not a scalar clock map.
   Prioritize cap-aware supervision (separate cap probability plus conditional
   event time) or a duration-loss ablation/retrain.
2. **Snap wins; audit improves with stable event MAE.** Freeze the threshold in
   the artifact. After active source-frozen trainings end, express it through
   the existing `duration_snap_threshold`, verify that shape/action hashes are
   unchanged, then run raw-vs-calibrated paired closed-loop evaluation with no
   further tuning.
3. **Log-affine or isotonic wins and the paired audit interval is positive.**
   Add a config-bound duration-only map after the current source freeze, with
   serialization and monotonicity tests. Task 2 may be used as an implementation
   gate, but it was selected after pilot evidence and cannot support a claim.
   The claim-scale run is all ten LIBERO-Long tasks, states 0–49; repeat on three
   checkpoint seeds before a paper capability statement.
4. **Cap/event balanced accuracy is poor (roughly <0.65) or cap and event
   distributions overlap heavily.** Do not force a snap. A scalar calibrator
   cannot create missing separability; train a two-part head: cap/event
   classification and conditional event-time regression.
5. **Global calibration helps overall but harms multiple tasks/event types.**
   Reject it as a deployment mechanism. Diagnose task-balanced supervision or
   duration label noise; do not introduce per-task calibration from this same
   training set.

## Files and proposed launch

- `scripts/analysis/duration_calibration_core.py`: pure monotone maps, metrics,
  selection rule, and leakage-safe splitting.
- `scripts/analysis/duration_head_calibration_v2.py`: prepare/infer/analyze
  pipeline and immutable provenance.
- `tests/test_duration_calibration_core.py`: seven deterministic contract tests.
- `cluster/calibrate_duration_head_v2.sbatch`: reviewed proposal; not submitted
  by the implementation agent.

Example after review and sync, using the exact official snapshot rather than a
language overlay:

```bash
sbatch cluster/calibrate_duration_head_v2.sbatch \
  /path/to/Cbase_s1000/pretrained_model Cbase_s1000 \
  "$HOME/scratch/vla_bspline/hf_cache/lerobot/hub/\
datasets--HuggingFaceVLA--libero/snapshots/86958911c0f959db2bbbdb107eb3e17c5f9c798e" \
  c01ec32de3ba5f5cd7eb251ce1e6279a60fbe1c80d12df1eeca746927f78df6c
```

