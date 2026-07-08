# C (time-allocation) native-gap: diagnosis → intervention decision tree

*The gap: C@nas10 avg 67.8 / C@nas5 74.0 vs B 82.8; worst on libero_10 (47–50 vs 70).
C trains BETTER than B (loss 0.263 vs 0.307) ⇒ the problem is target/decode design,
not optimization. Diagnosis job 2064021 produces: duration calibration (on-train),
libero_10 rollout signatures (C vs B), T-hat closed-loop traces.*

## Hypotheses and their signatures

**H1 — cap-coarseness.** 49% of training chunks hit the 40-step cap; 6 control
points over 40 steps ⇒ early-trajectory resolution ~halved vs B's 20-step chunks.
Long-horizon tasks are transport-dominated ⇒ hits libero_10 hardest (matches 47 vs 70).
*Signature:* C failures show imprecise transport — path_len inflated, stalls,
n_grasp_cycles elevated because approaches arrive misaligned; calibration itself fine
(T̂≈40 when true=40).
*Fix F1 (plain):* `horizon_max` 40→24 (retrain pilot). **Measured trade-off (July 6):
cap fraction 49%→72%, duration CoV 0.38→0.24** — weakens the duration signal materially.
Stats generated (`spline_stats_libero_v2_h24.json`) but F1″ below is preferred.

*Fix F1″ (decoupled, preferred):* fit the SHAPE over min(T, 24) steps (restores B-level
control-point density on cap chunks) while the DURATION label remains time-to-event
within 40 (keeps CoV 0.38; still drives selective speedup). Decode: execute the shape
over its own support; T̂ is the speedup/replan signal. Slightly more code (shape-support
channel decoupled from duration channel target), conceptually the right factorization.

**H2 — duration-mode smearing.** The duration target is bimodal (events ~22 ± 10 vs
cap 40). Flow matching regresses log-T per token and we decode exp(mean-of-logs):
if generation smears between modes (T̂≈30 for a should-be-40 chunk), the decode
timing warps every cap chunk.
*Signature:* calibration scatter shows mass between modes; `hat_at_cap_frac` ≪
`true_at_cap_frac`; high within-frame sample std.
*Fix F2 (decode-only, FREE):* snap T̂ to nearest mode / use per-token median /
asymmetric rounding toward cap. Evaluate on the existing C_full immediately.

**H3 — replan-boundary velocity discontinuity.** boundary_ratio ≈ 4 for all methods
(c₀-pinning gives position continuity only); C replans 2× as often at nas5.
*Signature:* C failures show oscillation right after replans (elevated n_grasp_cycles
near approach, high cmd accel at boundary steps).
*Fix F3 (decode-only, FREE to try):* velocity-continuous chaining. Clamped cubic
initial velocity: s'(0) = (n_knots_factor)·(c₁−c₀); with our clamped-uniform knots
(degree 3, n_ctrl 6, interior span 1/3): s'(0) = 3·(c₁−c₀)/(1/3)·(1/h) — set
c₁ ← c₀ + v_prev·h/9 so the new chunk starts at the previous chunk's replan-point
velocity (v_prev = derivative of the previous spline at the consumed-u point, tracked
as policy state). Projection of the model's intent — test empirically; if decode-side
helps, the training-side version (condition targets on v_prev) is the principled follow-up.

**H4 — event-boundary toggles.** Toggles sit at chunk end; partially mitigated by
nas5 (object 85→91) but NOT on libero_10 (50→47) ⇒ secondary for the long-horizon gap.
*Fix F4:* fit `post_event=4` steps past the event (toggle becomes interior), duration
target stays T_event. Retrain pilot. Only if H1/H2/H3 fixes leave a grasp-timing residue.

## Decision rule (when 2064021 lands)
1. Calibration bad → apply F2 (free), re-eval C_full object+long; keep if +.
2. Signatures show transport imprecision → F1 pilot (20k, ~80 min) is primary retrain.
3. Test F3 (free) regardless — orthogonal, measured boundary problem exists.
4. F4 only on evidence of grasp-timing residue after 1–3.
5. Best combination → 100k retrain → full row. Success bar: C ≥ B − 1pt native,
   libero_10 ≥ 65, while keeping the Pareto capability.

## Guardrails
- One variable per pilot; eval object+long only (fast signal) before full suites.
- Keep B untouched as the control arm.
- Arc-length reparametrization (shape over path-length + separate time law) stays
  pocketed as v3 if F1–F4 can't close the gap.
