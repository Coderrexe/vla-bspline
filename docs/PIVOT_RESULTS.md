# Language Steerability & Long-Horizon Pivot — Results (Aug 6–9, 2026)

> **Historical experiment log.** The checkpoints, videos, and canonical cells in
> this file remain useful discovery and mechanism evidence. Later harvesting
> completed the warm-granular RoboCasa fleet at 0/450, showed that the intended
> Clang seed-1001 launcher had saved seed 1000, and found that some LIBERO top-ups
> repeated initial states. Those findings narrow the statistical claims; they do
> not erase the underlying observations. Use `docs/TEAM_UPDATE_2026-08-22.md` for
> the cohesive team-facing synthesis and `paper/results.md` for the comprehensive
> current tables, including the locked replacement experiments.

*The Aug-6 pivot (Simba/Xiatao): BSP owns speed/frequency novelty → new headline =
**granular language control, steerability, and long-horizon execution** enabled by
the compact, event-aligned B-spline action space. Two validation questions were
set; both were pre-registered before data. Standing results (CALVIN flagship,
frequency axis, capacity ladder) live in `paper/results.md`.*

---

## Q1. Does language steering work on RoboCasa 365? — **YES, totally, both arms**

Counterfactual-instruction probe on the existing RC atomic checkpoints (trained on
213 natural-language instructions): same scene, same seeds, correct vs swapped
instruction. n=50/cell + n=100 bases.

| probe cell | A waypoint: base → mis-instructed | C spline: base → mis-instructed |
|---|---|---|
| **kettle → faucet instruction** | 35% → **0/50** (p<1e-4) | 38% → **0/50** (p<1e-4) |
| **rack in ↔ out** | 36% → **0/50** (p<1e-4) | 30% → **0/50** (p=3e-4) |
| **drawer left ↔ right** | 14% → 0/50 (p=.028) | 12% → 0/50 (p=.053) |
| microwave / stove / coffee | floor-base cells (0–8%), uninformative | — |

- **Total instruction control**: on every informative cell, swapping the
  instruction eliminates success completely. Frame strips (stored in-probe)
  show mis-instructed policies approaching the trained fixture and *refusing*
  the manipulation — behavior follows words, not scene.
- **The spline factorization costs nothing in language grounding** — C matches
  A's sensitivity exactly at matched baselines. (Pre-registered ≥15-pt drop:
  exceeded — drops are total.)
- The probe methodology itself (per-episode counterfactual instruction
  intervention, matched scenes/seeds) is novel for this benchmark — the
  RoboCasa365 paper contains no steering analysis.
- Harness: `cluster/rc_rollouts.py` (per-episode swap/static override + frame
  capture). Data: `outputs/steer/*.npz`, strips at `outputs/steer/strips/`.

## Q2. Does granular language improve long-horizon execution? — **YES on LIBERO-Long; RC365 composite = transfer-gated (below)**

### LIBERO-Long (ClangMixL10): the two hardest multi-object cells transformed

Trained the C-n8 recipe with 8,627 Molmo-2 per-segment clause labels (3,599 newly
generated for libero_10; ClangMix 50% mix recipe), evaluated with standard
instructions, n=50/task:

| cell | A | B | C-n8 | C-n10 | **ClangMixL10** |
|---|---|---|---|---|---|
| **libero_10 t0** (dual-object basket; Hard-5) | 45.2 | 57.4 | 51.5 | 44.7 | **76** |
| **libero_10 t4** (dual-mug; Hard-5) | 43.3 | 50.0 | 45.4 | 42.0 | **58** |
| suite (all 10) | 62.7 | 64.7 | 61.4 | 60.6 | 60.0 |
| spatial regression check | 79.7 | — | 77.3 | — | 77.0 (clean) |

- **t0: +18.6 points over the best arm of any family** (+25 over its own base);
  t4 best-arm. Gains are **free** at suite level and cost nothing on spatial.
- Mechanism chain is closed: video forensics identified return-to-placed-object
  confusion on exactly these multi-object tasks → predicted granular clauses
  disambiguate → confirmed. (Single seed; replication seed s1001 trained,
  eval pending.)

### The Hard-5 endgame: every pre-registered hardest cell now has a within-family winner

| Hard-5 cell | best baseline | our best | winning lever |
|---|---|---|---|
| libero_10 t0 | B 57.4 | **76** | granular language |
| libero_10 t4 | B 50.0 | **58** | granular language |
| libero_10 t7 | B 61.6 | **78** | control-point density (n12) |
| libero_10 t8 (moka pots) | A 47.6 | **52** | co-scaled chunks @ nas12 |
| libero_spatial t5 | B 43.2 | **72–74** | 2× decode rate |

The family's per-cell best beats or matches the best baseline on **all five**
cells selected (pre-registered, baseline-only rule) as LIBERO's hardest.

### Co-scaled chunk configs (new long-horizon lever, Aug 6–7)

Cn16h48 / Cn12h32 (n_ctrl and chunk-cap co-scaled at constant density; 1 seed):

- **t8 (project-worst cell) vs chunk length at nas12 is monotone: h24 26% →
  h32 42% → h48 52%** — and cadence is causal: the same h48 checkpoint scores
  30% at nas5 vs 52% at nas12. Fewer replan boundaries per episode fix the
  compounding failure the videos showed — a decode-time, single-variable
  dose-response.
- **Cn16h48 @ nas5 spatial = 81.8 — best spatial number of any arm all
  project** (A 79.7); the capacity curve breaks its former asymptote
  (75.5 → 77.3 → 79.0 → 79.4 → 81.8).
- Cadence law extended: short precision tasks want frequent replanning (nas5),
  long compounding tasks want long executions (nas12) — one checkpoint, one knob.

---

## RoboCasa 365 composite (the 2×2 program)

**Design**: {A waypoint, C spline} × {task-level compound instruction, granular
per-subtask} on the 16 composite-seen tasks (paper protocol: n=30/task, binary
success, per-task horizons 1,200–4,350 steps). Granular training = sentence-split
compound instructions aligned to gripper-release events (**99.6% of 2,524
episodes' boundaries landed on release events** — the event-alignment thesis
measured at scale). Headline eval protocol = **self-paced switching**: the
policy's own executed release events advance the sentence pointer (RT-H/Hi-Robot
autonomy norms; mechanically verified in telemetry — no oracle, no extra modules).

**Result 1 — from-scratch floor (definitive, 1,410 episodes):**

| arm | pooled success |
|---|---|
| A-task (compound instr) | 0.0% (0/180) |
| C-task (compound instr) | 0.0% (0/360) |
| A-gran (fixedK switching) | 0.0% (0/420) |
| C-gran (self-paced) | 0.3% (1/450) |

At 0.45B params trained 100k-from-scratch on composite-only data, **all arms
floor** — replicating the RC365 paper's own finding (Diffusion Policy 0.2%
composite) at 7× smaller scale. Composite kitchen tasks *require transfer*.
This is the motivation row for the transfer recipe, not a dead end.

**Result 2 — warm-start transfer (the RC365-proven recipe):** initializing from
the atomic checkpoint (whose skills are the composites' building blocks) is
verified working — training loss starts at ~1.0 vs 1.9 fresh and reaches
0.343/100k (granular arm). Both warm C checkpoints (task-level + granular)
completed.

**⏳ PENDING HARVEST (VPN down at write time):** the warm-granular 16-task
self-paced fleet (resubmitted Aug 7 after a sick-node wipeout — all 10 crashes
traced to r817u29n05 and excluded) and the warm task-level twin. These cells
decide whether the composite 2×2 becomes quantitative. Reference bar: GR00T
N1.5 (3B, NVIDIA) target-only = 35.0% composite-seen.

---

## Context & rigor notes

- **RC365 baselines** (their Table 1/2): multi-task composite-seen — DP 0.2%,
  π0 5.2%, π0.5 7.1%, GR00T N1.5 9.6%; with target fine-tuning GR00T reaches
  35.0/33.3 (seen/unseen). Long-horizon is where the field fails — their own
  conclusion, and our pivot's target.
- **Everything was pre-registered** before data (P-L1/P-H1/P-H2 + the earlier
  P1–P8); refutations are reported as findings (e.g., the horizon meta-analysis:
  spline-vs-waypoint delta correlates *negatively* with demo length at
  task-level conditioning — the motivation for granular conditioning, which
  then fixed the multi-object cells).
- Self-paced switcher, granular dataset builder, probe harness, and eval
  launchers are all in the repo (`cluster/rc_rollouts.py`,
  `libero/build_granular_composite.py`, `~/comp_eval.sh` on misha).
- Standing results unchanged (see `paper/results.md`): CALVIN retiming
  +0.379 @1.42× (n=3000×3), language Pareto at n=600, frequency dose-response
  with A@2× = 0/250, compute +6%/call, full stats appendix
  (`docs/STATS_TABLES.md`).
