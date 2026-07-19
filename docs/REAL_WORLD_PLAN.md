# Real-world experiment plan (draft for team discussion, July 14)

Purpose: answer Xiatao's question — which real-world tasks benefit from
decode-time speedup — using the selection principles our sim evidence already
established, so hardware time goes to experiments with predicted wins.

## The selection principle (from the CALVIN mechanism probe)

Retiming gains come from **compressing demonstration dead time** (77% of the
CALVIN gain was behavioral — removing reproduced hesitation — not clock
management). This gives a sharp task filter:

1. **Compressible slowness** (dead time, careful teleop, dithering) → retiming
   converts it to speed at no cost, sometimes a success GAIN (CALVIN: +0.27
   avg_len AND 1.34× speed, t=9.1, 3 seeds).
2. **Dynamics-limited slowness** (pouring liquids, pushing, throwing) → NOT
   compressible: speed changes the physics. Retiming preserves path geometry,
   so it is only valid in the quasi-static regime. This boundary must be
   stated in the paper and respected in task selection.
3. **Where to compress is data-regime-dependent** (LIBERO vs CALVIN): scripted
   demos → protect slow segments; human teleop → compress everything except
   the extreme-slow tail. Human real-robot demos are the CALVIN regime.

## Zero-hardware first step (can start now): dead-time census of public teleop data

Before any robot time, run the offline speed-profile analysis (the CALVIN
jitter/dead-time gate) on public real-robot teleop datasets — DROID, ALOHA
(incl. cloth tasks), BridgeData — and rank tasks by *predicted* retiming gain
(dead-time fraction × quasi-static fraction). Deliverable: a table "task →
predicted speedup at zero success cost." This makes the hardware proposal
evidence-driven and is a paper subsection by itself ("which tasks benefit and
why — predicted, then confirmed").

## Proposed hardware experiments (ranked)

**P1 — Cloth folding (Xiatao's scenario; primary).**
Demos are slow *because teleoperators must be careful*, and the task is
quasi-static — exactly the compressible regime. Hypothesis: θ-tuned retiming
executes at 1.3–1.5× demo speed at equal-or-better fold success (the CALVIN
sign-flip transferred to hardware). Metrics: fold-quality score, success rate,
wall-clock per fold, jerk/motor current (smoothness). Bonus claim from the
duration head: predicted-time-remaining as a progress display during folding.

**P2 — Tabletop kitting / bin-to-bin transport (throughput framing).**
Transport-dominated with precise grasps: the LIBERO selectivity regime.
Hypothesis: duration-gated selective speedup raises items/hour at matched
success where uniform speedup breaks grasps. Metrics: items/hour, grasp
success, collisions. This is the industrial-relevance experiment.

**P3 — Precision insertion under speed (peg-in-hole or connector).**
The protect-slow boundary on hardware + the smoothness argument: ease-out
decode (terminal velocity ×0.14) and velocity-continuous chaining should show
up as lower contact forces/vibration on an impedance-controlled arm — the
claim sim absorbs (LIBERO OSC hid it) becomes measurable. Metrics: insertion
success vs speed, peak contact force, motor current spectra.

## What each experiment leans on (already-validated components)

| component | validated where |
|---|---|
| retiming knobs (uniform/θ/selective) | LIBERO n=300 crossed; CALVIN 3×n=1000 |
| feasibility stretch (rate adaptation) | LIBERO +19 pts at half rate |
| ease-out / velocity chaining | LIBERO kinematics (toggle ratio → demo-like) |
| duration head calibration | MAE 2.8–3.0 both configs |
| quasi-static validity boundary | stated; enforced by task selection |

## Open questions for the meeting

1. Which arm + teleop rig does the lab have ready, and who demos? (P1 needs
   ~50–100 folding demos.)
2. Is there an impedance-controlled setup for P3's force measurements?
3. Paper scope: is one hardware task (P1) + the dead-time census enough, with
   P2/P3 as stretch?
