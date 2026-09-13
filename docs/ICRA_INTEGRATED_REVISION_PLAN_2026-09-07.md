# Integrated manuscript revision

Plan recorded before revising the 7 September manuscript. The 5 September paper
and source bundle are preserved in `paper/manuscript_versions/`.

## Central argument

The paper should not read as either a capability inventory or a two-experiment
paper that hides the rest of the project. Its organizing idea is a single action
interface with two levels of control:

1. **Semantic control:** an ordered language program selects the active subgoal.
2. **Physical control:** each short event-aligned spline segment represents a
   command path together with a duration, which controls sampling and replanning.

The main scientific claim is that these levels become more useful when action
segments follow manipulation events. The strongest causal evidence is the action
head by language-label interaction on LIBERO. Timing, rate adaptation, language
pace, and duration-scheduled replanning are not separate headline claims; they are
tests of the physical clock exposed by the same representation.

The paper will continue to report broad LIBERO performance, the clause-order
intervention, and RoboCasa semantic phases. They answer general competence,
behavioral response, and cross-domain transfer, respectively. They are supporting
evidence for the same argument rather than side projects.

## What the additional ICRA paper review changes

In addition to the two team-supplied award papers, the following complete papers
were read and their key pages inspected visually:

- [ACG](https://arxiv.org/pdf/2510.22201), accepted at ICRA 2026;
- [Discrete Policy](https://arxiv.org/pdf/2409.18707), ICRA 2025;
- [No Plan but Everything Under Control](https://arxiv.org/pdf/2503.01732),
  ICRA 2025 Best Paper in Planning and Control;
- [Robo-DM](https://arxiv.org/pdf/2505.15558), ICRA 2025 Best Paper in Robot
  Learning;
- [Camera Conditioning](https://arxiv.org/pdf/2510.02268), supplied by the team;
- [MAC-VO](https://arxiv.org/pdf/2409.09479), an ICRA 2025 Best Conference Paper.

The strongest introductions begin with a field-level technical tension, explain
the resulting failure, and only then introduce the method. They do not begin with
a toy instruction. ACG is especially relevant: its first page gives the problem,
mechanism, and cross-domain effect in separate visuals, while its evaluation is
organized by explicit questions. No Plan uses a real behavior montage in the
teaser and carries one argument through simulation, hardware, and ablations.
Discrete Policy includes many results, but groups them by questions about the
representation rather than presenting a chronological experiment log. Robo-DM
shows that a dense evidence package can remain legible when each figure has one
job and each section has a clear question.

The revision therefore adopts four style rules:

- Open with the problem of coordinating semantic progress and physical timing in
  long tasks, followed immediately by the representation and evidence.
- Use ordinary technical prose. Avoid unnatural compounds such as
  “within-checkpoint,” slogans, rhetorical commands, and excessive qualifications.
  Standard terms such as “B-spline,” “long-horizon,” and
  “vision-language-action” remain hyphenated.
- State absolute outcomes before effects and keep one main number per sentence.
- Preserve the clean rollout teaser and method diagram. Do not return to the old
  multi-box capability poster.

## Evidence hierarchy and manuscript placement

| Role | Evidence | Placement |
|---|---|---|
| Main language result | Under the same clause schedule, label changes improve A by 34.3 points and C by 43.7 points; interaction +9.33 points, positive for all three seeds | Abstract, Fig. 3, complete factorial table |
| Behavioral intervention | Reversing t4 clauses changes the requested first object in 10/50 states for both heads, with no opposite flips | Compact table and one paragraph |
| Harder-domain transfer | RoboCasa aggregate 4/100 to 16/100 and 3/40 to 9/40; Rinse stage progress repeats | Compact table, teaser rollout |
| Broad competence | LIBERO A/B/C 82.1/80.8/80.7 across three seeds | One context table |
| Main timing result | Historical CALVIN C 1.311 to 1.691 on 3,000 official chains at 1.42 times realized pace | Timing figure and table |
| Fair timing control | Fresh four-decoder screen compares native and retimed A/C at common K=5 and the same 50 chains per seed | Same timing section; label as a focused screen, separate from historical K=10 |
| Duration is operational | On LIBERO Object and Long, replanning at predicted duration minus four matches the K=5 reference while reducing calls 31.3 to 11.7 and 73.8 to 33.0 | Physical-clock evidence table |
| Language controls pace | CALVIN “quickly” and “slowly” change issued pose-command magnitude by about +16% and -16%; quick improves mean chain length 1.382 to 1.553 at n=600 | Physical-clock evidence table and short paragraph |
| Control-rate adaptation | C improves Spatial t5 from 38% to 68% and 34% to 72% in two paired protocols at twice the control rate | Physical-clock evidence table; explicitly task-specific |

The three final rows restore valuable earlier work without restoring the old Table
III. They are targeted tests of the learned clock: when to query, how fast to move,
and how densely to sample. Failure detection, contact ease-out, feasibility stretch,
control-point density, and isolated cadence cells remain in `paper/results.md`.
Those results are preserved and useful for future analysis, but adding them here
would introduce new mechanisms that do not strengthen either central claim enough
to justify their explanation cost.

## Interpretation of the new waypoint control

All twelve 50-chain timing cells completed. Native/retimed waypoint scores are
1.52/2.02, while native/retimed spline scores are 1.21/1.77. The retiming gains are
+0.50 and +0.56; their interaction is +0.06 with a hierarchical 95% interval of
[-0.50,+0.62]. Generic temporal resampling therefore benefits both formats.

The first automated audit stopped because a combined pickled-request hash differs
across the original independent jobs, although chain identities, controller budgets,
query cadence, and checkpoint hashes match. Two progressively stricter four-run
reset probes now reproduce all 50 robot states, scene states, camera observations,
and the exact packed policy requests bit for bit. The audited screen is in the paper
as a focused control, separate from the 3,000-chain historical experiment. A grouped
same-node rerun, Slurm job 2423956, attempted to run all four decoders sequentially
for each seed. All three jobs stopped before evaluation because the shared SmolVLM
Hugging Face cache was empty and compute nodes run offline. No partial outcomes were
produced; a repeat remains optional after intentional cache restoration.

The result does not invalidate the spline timing study. It establishes that generic
retiming is not exclusive to splines and focuses the representation claim on learned
event duration, executable clauses, duration-gated compression, and predicted query
times.

## Planned manuscript changes

1. Replace the current introduction opening with a field-level problem statement.
2. Describe the contribution as one event-aligned action interface with semantic
   and physical clocks. Retain two contributions, but connect every experiment to
   one of them.
3. Organize experiments around two questions, not a list of capabilities.
4. Add a compact table with the three targeted physical-clock tests above.
5. Update the CALVIN figure and table after the reset probe and statistical audit.
6. Remove unnatural hyphen compounds throughout while retaining standard technical
   compounds.
7. Keep the current full-width rollout teaser, clean method diagram, and language
   plot. Do not increase label density.
8. Preserve the hardware placeholder. The final physical section should test a
   late-clause intervention and one timing or rate condition using the same metrics
   as simulation.
9. Compile, inspect every page, verify references and fonts, and save each material
   revision separately. The current draft is
   `paper/manuscript_versions/2026-09-07_integrated_revision_audited_v2.pdf`, with a
   matching source bundle. The immediately preceding screen version is also
   preserved. `ICRA/root.pdf` remains the current working copy.

## Revision status

The integrated screen version is complete and eight pages in US Letter format. It
has no overfull boxes, missing references, or acknowledgments. Figure 1 is an actual
RoboCasa rollout; the method, language, and timing figures have one purpose each and
were inspected at rendered page size. The introduction now begins with the technical
problem of semantic progress and continuous timing. The experiments are organized
around the two questions above, and the physical-clock evidence appears in a
three-column table rather than the previous capability matrix. Hardware remains a
clearly marked placeholder to be replaced by the focused physical study.

## Wording constraints

- Never describe the 50-chain screen as replacing the 3,000-chain historical
  evaluation.
- Use “same frozen policy” or “for each frozen policy,” not “within-checkpoint.”
- Use “effect for each training seed,” not “seed-level effect.”
- Use “fixed clause schedule,” not “fixed-clock,” unless the physical clock is
  literally being discussed.
- “Speed” is reserved for measured execution time. The adverb experiment measures
  issued pose-command magnitude and chain completion; state that precisely.
- Retimed waypoint interpolation is a strong baseline, not a strawman.
