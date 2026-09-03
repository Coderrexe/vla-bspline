# Molmo language-boundary migration

`scripts/data/molmo_segment_labels.py` now uses the same event detector as the
production spline head in `policy/smolvla_spline/event_targets.py`.

The corrected convention is:

- an event at action index `k` ends the current half-open segment at `k`; the
  event action starts the next language segment;
- every raw gripper-value change is a toggle (not only a sign change);
- pause thresholds use the interpolated median of the current fixed-length
  production window (not an episode-global median over nonzero speeds);
- short tail windows reproduce LeRobot edge padding and the padding mask, while
  the language interval itself is clipped to the real episode end.

The extractor repeatedly applies that kernel at non-overlapping segment starts.
This partition is intentionally not identical to the set of normalization or
training windows: the latter are overlapping windows starting at every dataset
index. New manifest and label records carry
`boundary_version=event_targets_k_exclusive_window_quantile_v1` so artifacts can
be audited without relying on filenames; they also store the horizon, minimum
segment, pause fraction, and action indices. The label stage rejects legacy or
mixed-version manifests before loading Molmo.

## Existing artifact status

- **`libero_l10gran` must be rebuilt and its affected checkpoints retrained**
  before making an exact event-aligned-language claim. Its existing Molmo labels
  were extracted with the legacy boundaries. Re-run extraction, re-caption the
  corrected segments, rebuild the derived dataset, and train into new immutable
  output paths. Do not overwrite the historical dataset/checkpoints; they remain
  valid evidence for the broader language-augmentation factorial as long as they
  are described as legacy heuristic segmentation.
- **The existing RoboCasa granular dataset is not repaired or invalidated by
  this Molmo-specific change**, because its sentence-to-release alignment does
  not consume this script. It has separate open issues: heuristic subgoal
  alignment, and checkpoints/statistics made with the former offline spline
  target definition. Those require their own dataset/statistics repair and
  retraining before an exact event-alignment claim.

## Measured migration impact on LIBERO (2026-08-21)

`scripts/analysis/legacy_molmo_boundary_audit.py` freezes the old extractor and
compares it with the corrected shared kernel on action data only; it neither
decodes images nor mutates a dataset. With the historical extraction settings
(`horizon_max=24`, `min_seg=8`, `pause_frac=0.15`), the complete 1,693-episode
LIBERO dataset has exact legacy/corrected partitions in only 166 episodes
(9.81%). Only 4,708/11,833 legacy internal boundaries match exactly (39.79%);
the nearest-boundary drift has median 1 step and p95 9 steps. Conservatively
requiring an unchanged source interval before reusing a caption, 8,702/13,526
legacy intervals (64.34%) require re-captioning, and 6,714 legacy intervals
cross a corrected boundary. The maximum-overlap assignment leaves 12,355/273,465
frames (4.52%) outside their legacy label's dominant corrected interval.

The surviving Molmo label directories contain 3,599 segments. Their counts map
exactly to source task IDs `0,1,2,3,6,8,9,29` (task 29 is also caught by the
"chocolate" filter). On precisely those 304 episodes, only 14 partitions (4.61%)
are exact, 1,026/3,295 legacy boundaries match (31.14%), 2,579/3,599 captions
(71.66%) lack an exactly reusable corrected interval, 2,177 intervals cross a
corrected boundary, and the frame-level maximum-overlap residual is 6.06%.
These measurements quantify why the old derived dataset is evidence for
language augmentation, not for exact production-event-aligned language.

The local, no-cluster reproduction was:

```bash
PYTORCH_PY=/Users/simbashi/Developer/miniforge3/envs/pytorch/bin/python
LIBERO_ROOT="$HOME/.cache/huggingface/hub/datasets--HuggingFaceVLA--libero/snapshots/86958911c0f959db2bbbdb107eb3e17c5f9c798e"

"$PYTORCH_PY" scripts/analysis/legacy_molmo_boundary_audit.py \
  --root "$LIBERO_ROOT" \
  --out outputs/audits/libero_legacy_molmo_boundaries_h24_m8_20260821.json

"$PYTORCH_PY" scripts/analysis/legacy_molmo_boundary_audit.py \
  --root "$LIBERO_ROOT" \
  --task-id 0 --task-id 1 --task-id 2 --task-id 3 \
  --task-id 6 --task-id 8 --task-id 9 --task-id 29 \
  --out outputs/audits/libero_legacy_molmo_labeled_tasks_h24_m8_20260821.json
```

The reports are local ignored artifacts and the writer refuses to overwrite
them. Full-report SHA256 is
`61b5ef3396fbda5834c28aaf36c49e5c0b4a8195722044f01d488f776dc3cb99`;
the 3,599-label subset report is
`1fff8cefc1e356a14c799480f43665cf2beeceed7ec871ea70a7d6c732ee5b0b`.
Their canonical semantic action/task digests are respectively
`5e721e32a3839817ea1ec170ce5f878d854f43e7b0dbc7b74036ac4f81fa08f5`
and `9a154199dace7631c47b026dc5318c240ffde840c3cf31b583f4250588e85327`.
Both reports record the exact metadata hashes and source hashes; notably the
shared event kernel is
`9335cae53e277aa76975d59f43c266b0a1beee17d12d7737fd2d09db3016e9f6`.
