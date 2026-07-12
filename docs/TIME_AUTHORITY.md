# Execution-Time Authority: the unifying principle (framing note, July 9)

Every capability this project has demonstrated is an instance of ONE structural
property, worth stating once and precisely:

> **A time-aware trajectory representation factorizes the action into a geometric
> path p(u) and an explicit time map t(u). The policy learns both; the executor
> retains authority over the time map after training.** Waypoint chunks hard-code
> the time map into the representation (u ≡ t/dt·N), surrendering that authority
> at training time.

Each result is a choice of time-map transformation, applied at decode with zero
retraining:

| capability | time-map operation | measured result |
|---|---|---|
| rate transfer (Hz) | global rescale t → t·(f_train/f_exec) | §2 Hz table |
| feasibility stretch | rescale until ‖dp/dt‖ ≤ actuator bound | +16–18 pts at half rate |
| speed knob / selective speedup | per-chunk rescale t → α·t, gated by T̂ | 95% @ −16% time (n6, α.6); frontier moves with capacity, selectivity pays at it (+21 @ n8, α.4) |
| **ease-out (soft contact landing)** | local reparametrization: du/dt → 0 at event | terminal velocity ×0.14, endpoint exact (battery running) |
| self-paced replanning | replan schedule = t(1) − margin | fixed-cadence success at 2.7× fewer calls |
| **T̂-stagnation monitor / recovery** | the time map as a *runtime observable*: d T̂/dt ≥ 0 near events ⇒ no progress | failures stall 2×; 86% precision detection; recovery A/B running |
| demo-speed robustness | heterogeneous t(u) in data absorbed by the T̂ channel | B −13 @20k, C −4 (factorization measured end-to-end) |

Two corollaries that make this a thesis rather than a feature list:

1. **The knob map has structure.** Global rescaling is nearly free where dynamics
   allow (object/long at mild α); local operations near contact are where naive
   retiming breaks and where the learned time signal (T̂) supplies exactly the
   gating information needed (selectivity, easing, monitoring). The duration
   channel is not a bonus output — it is the *interface* through which execution
   authority is exercised safely.

2. **The waypoint baseline can only imitate this with external machinery** —
   resampling interpolators (adds the continuity problems we measured), external
   schedulers (TempoVLA's GPT-4o), or retraining per operating point (VSTA,
   DemoSpeedup, SpeedAug). Within-stack head-to-heads quantify the cost of each.

Positioning in one line: *prior work makes policies faster by changing the data,
the weights, or the prompt; we change where the clock lives.*
