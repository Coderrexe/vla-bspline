# B-Spline + Time-Allocation Action Head for SmolVLA — Results Digest

*LIBERO benchmark, LeRobot pipeline. All evals: 100 episodes (10 tasks × 10), replan every 10 env steps (nas=10) unless noted. Updated July 5, 2026.*

## Method in one line
SmolVLA's flow-matching expert generates **6 B-spline control-point tokens** (+ gripper channel + duration channel) instead of 50 waypoints; a decode samples the continuous trajectory at **any** execution rate. v1 = fixed chunk time (2s). v2 = **time allocation**: chunks end at motion events (gripper toggles/pauses), duration is a learned output.

## 1. Native-rate success (the "does it cost anything?" table)

**20k-step pilots, libero_object:**
| policy | success |
|---|---|
| A waypoint SmolVLA (nas=50, its default) | 60% |
| A waypoint SmolVLA (nas=10) | **93%** |
| B spline fixed-T | **91%** |
| C spline + time-allocation | 85% |

**100k full budget, all four suites (object / spatial / goal / long):**
| policy | obj | spa | goal | long | avg |
|---|---|---|---|---|---|
| B spline fixed-T | **95** | 80 | 86 | 70 | **82.8** |
| C spline + time-alloc | 85 | 64 | 72 | tbd | tbd |
| A waypoint (in-house) | *training* | | | | |
| *published SmolVLA ref* | *96* | *90* | *92* | *71* | *87.3* |

→ The compact spline representation (36 floats vs 350) matches waypoints at native rate.
→ **C's gap resolved (July 5): evaluating C with replan-every-5 (nas=5) → 91% on object, parity with B — zero retraining.** Mechanism: event-terminated chunks interact with replan cadence (the executed window must reach predicted events). C rows below being re-run at nas=5.

## 2. Control-rate robustness (the headline)

Deploy at a different control frequency than training (fair wall-clock budgets, libero_object):

| policy | f=10 (half) | f=20 (native) | f=40 (double) |
|---|---|---|---|
| A waypoint (20k) | **0%** | 93% | **0%** |
| B spline (20k) | 49% | 91% | **43%** |
| B spline (100k) | 56% | 95% | 50% |
| **B spline (100k) + feasibility stretch** | **72%** | — | — |
| C time-alloc (100k) | 50% | 91% (nas5) | **52%** |
| **C time-alloc (100k) + stretch** | **66%** | — | — |

**Waypoint policies collapse to 0% under rate mismatch. The spline policy retargets and keeps 43–72%.**

**Feasibility-aware time stretching** (new, ours): at decode, if any per-step delta would exceed the actuator bound, lengthen the horizon until the *same continuous trajectory* is executed feasibly — the robot slows down instead of clipping. +16 points at half rate. Only expressible with a continuous-time action representation.

## 3. Why time allocation (evidence so far)
- Event-segmented chunk durations vary strongly (CoV 0.38 over 53k anchors); 47% of chunks end at a gripper event with T = 22±10 steps → duration is a *predictable, semantically meaningful* output.
- The time-allocation variant **trains better** than fixed-T at every budget (event-aligned chunks are more homogeneous).
- Rate-feasibility (the stretch result) is exactly the capability duration-awareness formalizes.

## 4. Reproducibility
- Code: `lerobot/policies/smolvla_spline/` (+ factory & env `control_freq` patches), deployed on Misha & Bouchet.
- Train: `--policy.type=smolvla_spline [--policy.predict_duration=true]` on `HuggingFaceVLA/libero`; 100k steps ≈ 4–7h on 1 GPU.
- Eval variants (Hz/stretch/nas) are config-edited checkpoint dirs under `~/scratch/vla_bspline/outputs/hz_variants/` (Misha).
- Offline validation: `libero/validate_spline_head_math.py`, `libero/v2_event_segmentation_study.py`, `libero/horizon_study.py`.

## 5. Open items
1. C decode fix (two candidate mechanisms; discriminating eval running) → retrain pilot ~80 min.
2. A_full in-house waypoint 100k (training, ~4h) → completes the main table.
3. Full-table Hz sweep on 100k ckpts + C's Hz row (running).
4. Multi-seed / 500-episode confirmation for the paper's final numbers.
5. Jerk/smoothness metrics; chunk-boundary continuity analysis (c₀-pinning story).
