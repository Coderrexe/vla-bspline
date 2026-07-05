# ICRA Roadmap — B-Spline + Time-Allocation VLA (2 months, target ~Sept 2026)

*Principle: every headline claim gets (i) the strongest possible baseline, (ii) seeds/error bars, (iii) an ablation isolating the design choice. Red-team first, then run.*

## Tier 1 — Defensibility-critical (next 2 weeks)
| # | item | why | cost |
|---|---|---|---|
| 1 | **Interpolated-waypoint baseline** (resample waypoint chunk to any rate; same cumsum/telescoping care as spline decode) | kills the "strawman Hz baseline" attack; our claim must survive the baseline everyone would actually use | impl 0.5d + evals |
| 2 | **Duration-aware selective speedup** (rule: α<1 only when T̂ > threshold → success-vs-completion-time Pareto, C vs uniform-α B) | the experiment that makes the time head *uniquely* valuable, not just "free" | impl 0.5d + evals |
| 3 | **nas sweep** {5,10,20,50} × {A,B,C} native | replan-cadence confound is real (A: 60→93); pre-empt "you tuned it" | 12 evals |
| 4 | **n_ctrl ablation** {4,6,8} policy-level (20k pilots) + **unpinned-c₀ ablation** | representation-choice defense + boundary-continuity design justification | 4×80min train + evals |
| 5 | **Duration-head calibration**: T̂ vs event-truth on held-out demos (R², by event type) + per-episode T̂ timeline figure | proves the head *learned* duration; flagship figure | offline analysis |
| 6 | Smoothness/jerk table (running) + analytic-velocity continuity figure | C² claim quantified; velocity-FF story | ~done + plotting |
| 7 | Pin down true control dt / dataset rate (robosuite config + demo stats) | physical-units correctness | 1h investigation |

## Tier 2 — Strength & scale (weeks 3–6)
- **3 training seeds** × {A, B, C} × 100k + 500-episode evals on headline tables (error bars everywhere).
- **Hz curve densification**: f ∈ {5, 10, 15, 20, 30, 40} for the money plot; stretch on/off.
- **Event-interior training fix** for C (fit past events) → removes nas-sensitivity; re-run C row.
- **CALVIN unification with Quinten's track** — same head, second benchmark, shared results section (discuss w/ Xiatao).
- Failure-mode taxonomy from eval videos (grasp-timing vs tracking vs perception).

## Tier 3 — Reach (weeks 6–8, pick by payoff)
- **DCT-coefficient head** (FAST-style) in the same flow-matching framework → spline vs DCT vs waypoints, one table. High-impact ablation if time.
- Real-robot demo (lab hardware? ask Xiatao) — even a single task at two control rates would be a showstopper for the velocity-FF/impedance story.
- Contact-rich / variable-speed stress task where time allocation should shine on *reconstruction* too.
- Predict replan-horizon as third output (the doc's "most principled" option).

## Paper skeleton mapping
- Main table: A/B/C × 4 suites, 3 seeds (T2).
- Money plot: success vs deploy-rate, 5 curves (A naive, A interp, B, B+stretch, C+stretch) (T1.1 + T2).
- Time-allocation showcase: Pareto frontier (T1.2) + calibration + timeline (T1.5).
- Design ablations: n_ctrl, pinning, nas (T1.3–4).
- Smoothness: jerk + boundary table (T1.6).

## Standing infra notes
- Misha = primary (backfill trick: short walltime + untyped GPU). Bouchet = redundancy.
- Scratch purges at 60 days — final models mirrored to `~/vla_bspline/final_models/`.
- All evals 100 eps for iteration; 500 eps + seeds for camera-ready numbers.
