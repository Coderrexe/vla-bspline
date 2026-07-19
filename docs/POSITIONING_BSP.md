# Positioning vs "B-spline Policy" (BSP, arXiv 2607.09648, July 10 2026)

Han, Xiong et al. (Harvard/MIT/UT Austin). Read in full July 17. This doc is the
threat assessment + how we position. TL;DR: **BSP validates our representation
choice and leaves the entire time-allocation layer — where our contribution
lives — untouched. Their stated limitation is our solved problem.**

## What BSP is

- B-spline action representation for **Diffusion Policy and ACT** backbones
  (NOT VLAs — no language, no VLM). Policy outputs knots + control points
  (adaptive knot insertion at fitting time, FITPACK-style, bounded error).
- Acceleration = **uniform temporal rescaling** with a **human-chosen global
  speedup factor n** (1×/2×/4×): a_exec(t) = a(nt). No learned duration, no
  event segmentation, no per-chunk or per-interval gating, no slow-down.
- **Inference-time segment alignment**: at replan, search the new segment for
  the argmin-MSE point vs the last executed action and start there — their
  answer to chunk-boundary discontinuity (cf. our velocity chaining/ease-out;
  theirs re-anchors *time*, ours pins *velocity*).
- Real-world: 3 tasks (ARX5 arms), n=20 rollouts/cell. Result: completion time
  ~halved at preserved success (e.g., Table Cleaning 23.6s → 11.8s at "4×").
- Sim: Push-T, RoboMimic, **RoboCasa** (4 tasks: sink faucet 79→85, coffee
  button 93→94, microwave 77→89, close door 27→46, Diff vs Diff+BSP at 1×).
- Their Finding 2 + Limitations: **"aggressive speedup can exceed controller
  limits"** — Speed Stacking drops to 0/20 at 4×. They have no mechanism to
  decide *where* speed is safe; they name better low-level controllers as
  future work.

## Overlap (what we can no longer claim as novel)

- "Continuous B-spline actions instead of discrete chunks" as an idea (BSP +
  the BEAST tokenizer both exist — cite both).
- "Temporal rescaling without retraining" *in its uniform form*.
- The smoothness argument in generic form.

## Differentiation (all of it already measured)

1. **Time allocation is the contribution, not the spline.** Event-segmented
   variable-length chunks + a *learned* duration T̂. BSP's n is a global
   constant a human picks; our time authority is endogenous and per-chunk.
2. **The granularity ladder + its data-regime law.** BSP implements exactly
   one rung (uniform) of our four (uniform / chunk-gated / interval-level /
   bidirectional incl. slow-down). We show *where* each rung pays: fine gating
   on scripted data (LIBERO), broad compression on noisy teleop (CALVIN,
   +0.38 avg_len AND 1.42× speed, t=12.1, 3 seeds × n=1000). BSP has no
   analysis of when uniform rescaling is safe — our answer to their Finding 2.
3. **Safety machinery for speed**: duration-gated selectivity, protect-slow
   thresholds, feasibility stretch (actuator-bound-aware), slow-down for
   precision phases. Their 4×-failure is the regime our knobs are built for.
4. **VLA + language**: we live in a VLA (SmolVLA), with granular per-segment
   language (Molmo 2 pipeline) upcoming — BSP has no language axis at all.
5. **Duration byproducts BSP cannot express**: self-paced replanning (2.7×
   fewer policy calls at matched success), endogenous failure detection (86%
   precision), calibration analysis (MAE ~3 steps).
6. **Evaluation rigor**: BSP: 20 rollouts/cell, no significance tests. Ours:
   paired identical-condition protocols, 3 seeds, n=1000 chains, exact tests
   (EVAL_DESIGN.md).

## Actions

- Cite BSP prominently as concurrent work; frame: *"BSP shows continuous
  spline actions accelerate policies under a manually chosen global speed
  factor; we contribute the time-allocation layer — learned duration,
  event-aligned chunks, and granularity-selective retiming — that decides
  where and how much time authority is safe to exercise, and show the answer
  is data-regime-dependent."*
- RoboCasa (Xiatao T1) now doubly strategic: direct numeric comparison with
  BSP's published RoboCasa cells; include their 4 tasks in our 5.
- Adopt their completion-time metric (avg time on successes) everywhere ours
  is reported — makes cross-paper comparison trivial.
- Consider their segment-alignment as an additional baseline row for the
  chaining ablation (time-re-anchor vs velocity-pin).
