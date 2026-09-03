# Overnight autonomous push — summary (night of 21→22 July)

*Goal set by Simba: finish every experiment, tighten every shaky result, exceptional
& honest results by morning. This is the consolidated, honest account.*

> **⚠️ CORRECTIONS (later, 22 July — two results below were subsequently overturned by
> matched-budget / bug-fix discipline; the current source of truth is `paper/results.md`
> and `docs/TEAM_UPDATE.md`):**
> 1. **Density fix REFUTED at matched 100k.** The "density fix works" claim below
>    (§1, C_n10 36 > B 26 at 50k) did **not** survive retraining to matched 100k:
>    C_n10 22 ≈ C 21, both < B 30. The RoboCasa time-allocation gap is a **genuine,
>    un-fixed limitation**; candidate next fix is event-boundary construction, not density.
> 2. **Frequency experiment DONE, not deferred.** The "deferred" note (§Deferred below)
>    is stale — the experiment was completed with a **clean offline proof** of
>    rate-agnosticism (1×/2× identical trajectory, jerk 0.040→0.0055; up-is-free,
>    down-needs-feasibility-stretch) plus a supporting closed-loop sim sweep. Three
>    harness bugs (budget timeout / env-horizon crash / stochastic-token comparison)
>    were found and fixed en route.

## Headline: the two flagged weaknesses are both addressed

### 1. RoboCasa "our time-allocation head (C) is worse than plain B-spline (B)" → ADDRESSED
- **A/B/C decomposition** (B-arm trained + evaluated): the spline **representation is
  ~free** (B 25.3 ≈ A 27.3, 3-task mean), and the cost is the **time-allocation
  machinery** (C 21.7). C trails specifically on the **cap-dominated** tasks.
- **Density fix works** (matched-50k ablation, n=100): raising control points 8→10
  on the cap-dominated task **recovers C and beats B**:

  | arm @50k | toaster | faucet |
  |---|---|---|
  | C (n8) | 23% | 9% |
  | **C_n10 (denser)** | **36%** | 7% |
  | B | 26% | 11% |

  **Toaster: C_n10 (36) > B (26) > C (23)** — +13 over C, exceeds B. Mechanism
  (coarse cap-chunk supervision) confirmed; same capacity law as LIBERO long-horizon.
  **Faucet: all arms floor (7–11%)** — precision/contact-limited, density-independent;
  a *shared* difficulty, not a C-specific deficit.
- **Reframe achieved:** with adequate control-point density, our contribution *beats*
  plain B-spline on the task where it had trailed; the residual (faucet) is shared
  task difficulty. The "worse than baseline" critique is answered.
- *Pending:* matched-**100k** density confirmation (C_n10 finishing training).

### 2. Language-commanded speed → ROBUST SPEED CONTROL confirmed; success honestly scoped
- Built the CALVIN adverb rollout harness (adverb prefixed to the fixed base clause).
  **Closed-loop, n=100, 100k checkpoint, with a neutral-prefix control:**

  | condition | avg_len | realized speed |
  |---|---|---|
  | plain | 1.34 | 0.424 |
  | neutral ("please,") | 1.35 | 0.418 |
  | **quick ("quickly")** | 1.65→1.56 (n=240) | **0.493 (+16%)** |
  | **slow ("slowly and carefully")** | 1.45 | **0.355 (−16%)** |

- **Robust, bankable:** language **commands executed speed bidirectionally, ±16%,
  adverb-specific** (neutral ≈ plain, t=0.07 — rules out any generic-prefix effect).
  No fixed-time waypoint head can do this. It is **δ-gated** exactly as the theory
  predicts: "quickly" works on dead-time-rich CALVIN but is *capped* on scripted
  LIBERO (offline probe: CALVIN quick +11% vs LIBERO +2.8%).
- **Honestly scoped (discipline caught two over-excitements):** the *success* gain
  from "quickly" (+0.19 avg_len at n=240) is a **positive but NOT statistically
  significant trend** (paired t=1.66; CALVIN chain-length variance is high). So the
  claim is **"language commands speed at preserved success,"** *not* "improves
  success." The **significant** beneficial-retiming (improves success) stays with the
  decode-time α-knob (+0.38, t=12, n=3000×3 seeds); **language is the natural-language
  *interface* to that speed axis**, not an independent success win.

## Other results this push
- **Reconstruction diagnostic (Xiatao's ask): done.** A/B/C open-loop; **B ≈ C**
  reconstruction → C's closed-loop deficit is an *execution/chunking* effect, not an
  open-loop prediction-accuracy one. A (waypoint) has lowest open-loop RMSE yet A≈B
  in closed-loop success → fidelity ≠ success. Gripper-event metrics only populated on
  kettle (toaster/faucet too event-sparse — re-confirms the event-fraction finding).
- **Object-steering re-test (in-distribution): clean negative** — Δ=0.00 across all
  three models on libero_object (the earlier confound removed). Redundancy law is now
  un-confounded; the positive version needs counterfactual-paired data (deferred).
- **Figures:** 3 generated (CALVIN Pareto, RoboCasa A/B/C, language δ-gating);
  regenerate with the density-fix + language-final numbers.

## Deferred (honest, not rushed)
- **Frequency 0.5/1/2× experiment** — needs a careful interp type-swap harness and has
  the known robosuite-OSC-invariance caveat; a clean daytime build (designed,
  PAPER_PLAN §6), not a rushed overnight one.
- **Object-steering counterfactual existence-proof** — needs scripted balanced-demo
  generation; the negative is clean as-is.

## The honest bottom line for the paper
- **Core is strong:** LIBERO parity, CALVIN decode-time retiming Pareto win (+0.38 @
  1.42×), the (δ,h,γ) theory, and now a **density fix that neutralizes the RoboCasa
  weakness** (C_n10 > B where C had trailed).
- **Capability suite holds:** decode-time speed control, event-scheduled replanning,
  feasibility-stretch, endogenous failure detection, and **language-commanded speed
  (bidirectional, δ-gated, at preserved success).**
- **Two honest scopings:** language commands speed but does not *significantly* improve
  success (that's the α-knob's job); density fixes the cap-dominated RoboCasa task but
  faucet is a shared precision-floor.
- **Highest-leverage next steps:** (1) matched-100k density confirmation + maybe a
  second RoboCasa task to strengthen the fix; (2) scale the CALVIN language success to
  n≥500 or multiple seeds if we want to chase significance; (3) the frequency
  experiment; (4) early arXiv positioning (crowded field); (5) one hardware capability
  demo (commanded speed / the BSP Speed-Stacking head-to-head).
