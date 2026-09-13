# Focused manuscript revision

Revision plan recorded before manuscript edits, 5 September 2026.

## Decision

Build the paper around **piecewise language execution with event-aligned spline
actions**, followed by **control of the same representation's physical clock**.
These are two tested properties of one interface, not a catalog of capabilities.
Use the title **Executable Language Programs with Event-Aligned Spline Actions**.

The central language estimand is the effect of clause-aligned **training labels
under a shared clause-scheduled inference interface**. The complete train-label ×
test-interface table must be visible. The large gains are not a comparison against
whole-caption policies executing their native whole caption. The matched head
interaction is real, positive in all three seeds, and worth leading with; event
alignment, compactness, and learned duration are a bundled head intervention, not
separately identified causes.

## What the two reference papers teach

[Camera Conditioning](https://arxiv.org/pdf/2510.02268), especially Figs. 1–3,
Table I, and Sec. IV: the page-one, upper-right teaser shows an actual robot and
camera geometry, without trying to teach the architecture. A separate small
method diagram follows. Six simulation tasks, three policy families, and targeted
ablations all address one question. Importantly, the authors replay converted
actions to check that an action-space comparison does not accidentally compare
controllers. Its hardware section uses three UR5 tasks, 200 demonstrations per
task, and 21 trials per setting; partial success is defined separately from final
success. The section occupies roughly a page including photographs and plots.
This supports a concise hardware section, not a universal minimum trial count or
an acceptance guarantee. We should borrow the separation of teaser, method, and
evidence, and the explicit baseline-equivalence checks.

[MAC-VO](https://arxiv.org/pdf/2409.09479), especially Figs. 1–2 and 6–7,
Tables I–V: the teaser links reconstructed geometry to actual camera observations.
The method diagram is short and the major qualitative figure shows localized
failures alongside measured trajectories. Main tables answer the same odometry
question on different datasets; the ablation table removes specific parts of the
proposed uncertainty model. It does not claim to win every metric. The linked
version has an appendix beyond the main paper, so its total figure/table inventory
is not an eight-page, no-appendix target for us. We should adopt its evidence-led
visuals and component-specific ablations rather than its dense table typography.

Both papers were read in PDF text and key pages inspected visually. Neither
supports the proposition that adding one page of hardware guarantees acceptance.

## Evidence and main-paper placement

| Question | Evidence to retain | Presentation |
|---|---|---|
| Does the head help execute clauses? | Three-seed LIBERO t0/t4; gains A +34.33 pp, C +43.67 pp under the same fixed clause clock; interaction +9.33 pp | Primary table: both training-label treatments under both static and scheduled inference; compact seed plot |
| Does clause order affect behavior? | t4 requested first-object flip 10/50 for both heads, no opposite flips; final conjunction decreases under reversal | Small behavioral table; distinguish steering from held-out composition |
| Does semantic labeling transfer to longer tasks? | RoboCasa Kettle/Rinse aggregate 4/100→16/100 and 3/40→9/40; identical phase schedules | Task-level counts, stage-progress result, actual paired rollout |
| Can the trained action program be retimed? | CALVIN C 1.311→1.691, A 1.563; three training seeds and 1,000 chains per seed | Absolute outcomes and chain-depth plot; native waypoint identified as native, not interpolated |
| Is retiming competitive with waypoint interpolation? | New matched-cadence CALVIN comparison below | Separate four-arm table, never pool its fresh seeded runs with the historical evaluator |
| Does the interface retain general competence? | Four LIBERO suites, A/B-n8/C-n8 82.1/80.8/80.7 | Compact secondary table; descriptive closeness, not formal equivalence |

The duration calibration diagnostic uses episodes from the policy's training data
split for calibration/audit. It is not held-out policy-generalization evidence.
Keep at most one correctly qualified sentence. A predicted gripper-release clause
switch is not the learned scalar-duration clock. A clause spans multiple predicted
chunks; eight control points describe a chunk, not a whole long-horizon task.

## One focused new experiment

**Question:** with an equally retimeable waypoint output and matched physical
query cadence, does the timing advantage persist?

- Four arms: native waypoint, linearly retimed waypoint, native event spline,
  uniformly retimed event spline. No retraining; use the existing 100k CALVIN
  checkpoints, seeds 1000–1002.
- Freeze alpha at 0.6 from the existing result, not from this new test.
- Resample the waypoint **cumulative raw commanded path**, then difference and
  renormalize using its own checkpoint statistics. Never interpolate normalized
  deltas. Match sign-discrete gripper handling and pose clipping; no feasibility
  stretch in either arm. No claim that linear interpolation is the strongest
  conceivable spline-fitting baseline.
- Five issued actions per neural query for all arms: the spline's shortest retimed
  segment is round(8 × 0.6)=5, so every arm can obey this cadence. Hold each action
  for three native simulator steps, as in the existing replay-validated CALVIN
  controller. Keep the 360-simulator-step budget per subtask.
- First run identity/normalization/endpoint checks and a two-chain smoke. Then
  evaluate the same first 50 of the official 1,000 chains for all four arms and
  all three checkpoints. Seed policy noise by chain/subtask, so prior rollout
  lengths cannot shift the RNG stream of later chains.
- Record initialization, source/config hashes, chain outcomes, subtask completion
  steps, and neural-call counts. Report seed-level contrasts and paired-state
  uncertainty. Treat 50-chain cells as a focused screen, not a replacement for
  the existing 1,000-chain result. Do not expand/tune the screen according to which
  head wins; any larger confirmation requires a fixed protocol before launch.
- Slurm only; one GPU/job, at most four concurrent jobs, bounded wall time,
  isolated source/config snapshots, no modification of trained checkpoints.

The existing Spatial rate experiments retain useful within-spline improvements,
but direct replay at a changed rate is not the fair baseline for a general
interpolation claim. Remove that comparison from the main paper for this revision.

## Figures and tables

1. **Page-one teaser:** actual Kettle rollout frames and short phase text. Keep the
   shape/clock explanation in the separate method figure. Prefer a full-width strip below the title; if this requires
   invasive template changes, use the top of the right column, as both linked
   papers do. Do not put result bars, architecture boxes, and capability labels
   into the teaser. Clearly label any inset as a schematic.
2. **Method:** one plain left-to-right diagram: active clause + observation →
   unchanged VLA → control points and duration → decoded actions. Explicitly show
   that program switching and short motion segments have different time scales.
3. **Language evidence:** two readable panels, absolute scheduled success and
   seed-level label effect. Put the complete interface factorial in a table.
4. **Timing:** CALVIN chain-depth survival curves and seed-wise native/retimed
   spline outcomes; use actual JSONs, no illustrative result curves. A dedicated
   small four-arm baseline table is reserved for the new experiment.
5. **Hardware:** reserve approximately one page including actual robot images,
   paired late-clause trials, completion counts, and a clear scoring definition.
   Hardware collection is planned, not reported as already underway.

Use one font family, 8–9 pt labels at final size, consistent blue/orange head
colors, black text on white, 1–1.3 pt strokes, no rounded-card motif, and legends
outside data. Do not fabricate rollouts or use generated imagery as evidence.
Render and inspect every page after compilation.

## Cut from the manuscript, preserve in the ledger

The old secondary-capabilities Table III; language pace; failure detection;
contact ease-out; feasibility stretch results; capacity/cadence hard-cell sweeps;
speed-as-input training; the full retiming-regime theory; exploratory Molmo pilot;
wrong-prompt reward suppression. Selective replanning can be one implementation
sentence if necessary, not a third results storyline. These cuts are an editorial
selection, not deletion or invalidation of their source artifacts.

## Writing and layout

Use concrete nouns and verbs: what is predicted, what changes, what is held fixed,
and what succeeds. Avoid “unified control surface,” “airtight,” “maximally,”
“free capability,” and unsupported “preserves”/“establishes” language. Explain
results once, rather than repeating all numbers in abstract, introduction,
captions, results, and conclusion. Use absolute values before deltas.

The eight-page budget including references is approximately: title/teaser/
introduction 1.35; related work 0.45; method 1.35; setup and language 1.6;
timing/general competence 1.15; hardware 0.9; conclusion 0.2; references 1.0.
This is a layout target, not permission to shrink fonts or insert filler.

## Completion checklist

- Save the reviewed draft before editing; keep old plots reproducible.
- Generate manuscript tables/plots from identified JSONs where available.
- Preserve numerical findings while correcting their comparison labels.
- Build with the supplied IEEE class; keep acknowledgments removed.
- Check page count, embedded fonts, overflow, figure readability, and citations.
- Record the new baseline's job IDs and status without inserting unfinished data.

## Execution status

The reviewed draft is archived and the focused revision is implemented in `ICRA/`.
All four figures have been rebuilt; the teaser uses the supplied class's title-area
hook, with no change to page margins or type size. The first compiled revision has
seven pages, including references and a short hardware placeholder; the completed
hardware study and timing table must fit the eight-page budget.

The four-arm two-chain smoke array **2396996** completed, with exact first-subtask
observation agreement and five-action neural-query cadence verified in every arm.
The complete frozen **12-cell, 50-chain matrix is job 2397013**, with at most four
concurrent GPUs. Smoke outcomes are engineering checks, not added to efficacy
statistics. The local smoke audit is
`outputs/calvin_matched_timing_20260905/smoke_gate_2396996.json`.

Analysis job **2397037** depends on successful completion of all matrix cells. It
validates the full matrix before writing a summary, including all four frozen
contrasts: waypoint retiming, spline retiming, retimed-spline minus retimed-waypoint,
and their retiming interaction. Intervals include a three-seed t interval and
paired-chain bootstraps (shared chain indices across checkpoint seeds), with and
without resampling training seeds. Analysis uses a separate immutable snapshot;
it cannot overwrite checkpoints, rollout artifacts, or the manuscript.

Validation completed locally: six synthetic analysis-contract tests; Python and
shell syntax checks; complete seven-page PDF visual inspection; no overflow or
unresolved citations. The bibliography now uses the supplied class's native
layout, and all PDF fonts are embedded. The old acknowledgment remains removed.
