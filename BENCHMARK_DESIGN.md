# Benchmark, Baselines & Ablations — design doc (July 8, for Xiatao's direction A)

## 1. The landscape (lit review, July 2026)

A 2025–26 wave has converged on our question — *execute manipulation faster than
demonstrated while protecting contact phases* — but every entry solves it outside
the action representation:

| work | speed mechanism | selectivity signal | retraining? | notes |
|---|---|---|---|---|
| **TempoVLA** (2606.06491) | speed as *command input*: VSTA retiming aug + text-prefix / scalar-embed conditioning, π0.5 on LIBERO | none per-chunk; **GPT-4o schedules speed per task** (96% real-world w/ scheduling) | yes (30k it × batch 512 × 32 H20) | high-speed failures = controller tracking mismatch; realized speedup saturates ~1.6× |
| **DemoSpeedup** (2506.05064) | retime the *demos*, retrain | action-entropy (needs distributional policy) | yes | contact-rich named as the limitation |
| **ESPADA** (2512.07371) | semantic downsampling of demos | semantics + 3D spatial cues (external) | yes | preserves contact-critical phases by construction |
| **SpeedAug** (2512.00062) | speed-augmented prior + **RL fine-tuning** | RL reward | yes (+online) | 1.8× throughput, 16 min interaction |
| **AutoSpeed** (2607.01051) | picks among re-timed candidate futures during training | endogenous (prediction-error trade-off) | yes | closest in spirit to "endogenous timing signal" |
| **SAIL** (PMLR v305) | formalizes faster-than-demo execution | — | yes | problem formalization reference |
| **Ours (C: spline + time-alloc)** | **decode-time retiming of a continuous trajectory** | **the duration head itself** (calibrated: event MAE 2.99, r=0.84) | **no** — α is a decode knob | + Hz transfer, + feasibility stretch, + 2.7× fewer policy calls, + C² preserved under retiming |

Orthogonal axis (also "speeding up VLAs"): inference-latency work — token pruning
(SpecPrune-VLA), caching (EfficientVLA, DepthCache), pipelining (ActionFlow, 2.55×
FPS), real-time chunking. None touch motion time. **Measured (July 8): per-call
latency is FLAT across heads** (A 116.7 ms, B/C 119.9, C-n8 121.4; VLM prefix encode
dominates, expert token count immaterial — honest null). The compute win is
episode-level: margin-scheduled replanning cuts policy calls 2.7× at equal success →
~2.7× less inference compute per episode, composable with all latency work above.

**Positioning sentence:** everyone retrains to add a speed capability driven by an
external signal; a time-aware action representation gets the capability *and* the
signal for free, adjustable after training.

## 2. Benchmark comparison protocol

Absolute LIBERO numbers are incomparable across stacks (TempoVLA: π0.5, 500
demos/task, 32 H20s → 96.7% base; us: SmolVLA-450M, 1 GPU → ~82%). The honest
protocol is **within-stack controlled comparison** + **normalized cross-work curves**:

1. **Primary plot: success vs realized speedup** (relative to that stack's 1× —
   x = realized_steps(1×)/realized_steps(s), y = success retention Δ from 1×).
   Report *realized*, not commanded, speed (TempoVLA's own saturation finding).
2. **Within-stack head-to-head (the experiment nobody has):** on the same SmolVLA +
   LIBERO + budget — (a) speed-as-INPUT: "TempoVLA-lite" = VSTA-style retimed
   training + scalar speed embedding; (b) speed-as-OUTPUT: our C with decode-α;
   (c) data-level selectivity: retrain A on event-segmented selectively-accelerated
   demos (transit merged, contact preserved — ESPADA/DemoSpeedup mechanism, our
   segmentation); (d) naive: waypoint linear resampling at exec time (have).
   Axes: success @ each speed × retraining cost × per-chunk adjustability
   (input-conditioned policies get ONE speed per episode unless re-prompted;
   ours retimes per chunk).
   **First 20k numbers (July 9, object, n=100): TempoVLA-lite v=0.5/1/1.5/2 →
   90/93/78/40.** Conditioning is free at 1× (93 = A-20k twin) but success falls
   steeply with commanded speed (−15 @1.5×, −53 @2×) — the tracking-mismatch
   failure TempoVLA itself reports. Our decode-retimed C loses −2…−6 at comparable
   commanded speedups with zero retraining (100k α-curves; budget-matched rerun +
   realized-speed measurement queued before any cross-claim is promoted).
3. **Cost axes no prior work reports:** policy calls/episode, per-call latency
   (6 vs 50 denoised tokens), env steps to completion, all at matched success.
4. **Contact protection curve:** success vs speedup separately on transport-dominated
   (object) and precision-dominated (spatial/goal) suites — our suite-scoping result
   becomes the benchmark's hardness axis.
5. Seeds: ≥3 for any promoted number (project rule); n=100 episodes/point.

## 3. Ablation matrix for our method

| ablation | isolates | status |
|---|---|---|
| A vs B vs C at matched budget, 3 seeds | representation & time head cost | ✅ done (82.1/78.8/78.5) |
| horizon_max 40→24 (controlled 20k) | chunk-support density | ✅ done (+21 long) |
| n_ctrl 6→8 | control-point capacity | ✅ 20k; 100k seeds landing (spatial arm seed-fragile — honest) |
| α × {selective, uniform} × suite, n=100 | duration head's decode payoff + frontier | ✅ done (frontier moves w/ capacity; +21 at α=.4) |
| speedup threshold sweep (thr 15/20/25) | selectivity-rule sensitivity | ❌ cheap, queue |
| replan cadence: nas × margin × self-paced | duration-scheduled replanning | ✅ done (margin law, 2.7× calls) |
| duration-target design: event-seg vs fixed-T (B) | why time allocation trains better | ✅ (loss + speed-aug study) |
| speed-aug: A/B/C under demo-speed heterogeneity | shape/timing factorization | ✅ (sample-efficiency tax + T̂ absorption) |
| c0/c_last pinning on/off | endpoint constraints | ❌ cheap pilot, low risk |
| gripper-as-spline vs argmax-token | gripper channel design | ❌ optional |
| per-call latency A vs B/C | compute efficiency of 6-token expert | ❌ micro-benchmark, queue tonight |
| knot placement uniform vs event-densified | contact-region granularity (§4) | 🔶 designing now |

## 4. Contact-rich direction (Xiatao direction B) — plan

Evidence base: spatial gap is 3-seed robust at n6 (A 84.3 vs B 76.7/C 74.3); n8 fix
is 20k-real but 100k-seed-fragile; object-suite failures fumble AT the grasp (9.5
cycles); speedup collapses on precision suites. So the deficit lives in
*contact/precision phases*, and capacity alone is not a reliable fix.

**Step 1 — phase-resolved diagnosis (offline, running first):** decompose spline fit
residuals on the dataset by phase — distance-to-next-gripper-event — and compare
demo action spectral content near contact vs transport. If representation residuals
concentrate near contact, it's a smoothing/capacity ceiling (fix = D2/D3); if not,
the gap is behavioral (fix = cadence/decode).

**Step 2 — candidate fixes, representation-compatible by construction:**
- **D2 (primary): event-anchored non-uniform knots.** v2 chunks *end at contact
  events*, so "near contact" = "near u=1" in chunk parameter. Pack knot density
  toward the chunk end (geometric spacing): contact region gets fine basis
  resolution, transport stays coarse — same token count, same decode interface,
  still one B-spline (mentor's "more granular segmentation in the contact area",
  done with the spline's own mechanism instead of switching representations).
- **D3 (follow-up design): additive micro-correction tokens.** k extra tokens =
  per-step residual deltas for the last k steps before the predicted event, added
  on top of the spline and tapered to keep continuity. Waypoint-level freedom
  exactly in the contact window without a representation switch. (This answers
  "switch to waypoints during contact" elegantly: *add*, don't switch.)
- **D4 (cheap, decode-only): duration-adaptive replan cadence** — replan every
  clamp(T̂/2, 2, 10) steps so contact-imminent chunks get fresh observations more
  often. Uses existing machinery.

Gate: D2 pilot at 20k on spatial+object vs n6/n8 twins; promote only on >6-pt
spatial gain that survives a second seed (we've been burned twice).

**Status after the July 8 overnight (elimination ledger for the spatial gap):**
1. ✗ global fit fidelity (n8: residual −37% → seed-fragile ±2)
2. ✗ contact-local fit fidelity (W9: tail −23/−31% → success ±0; clean negative)
3. ✗ generative variance (K-sample dispersion equal at matched horizon:
   A 0.62/0.90 vs C 0.60/0.94 at steps 3/5, libero_spatial)

**Open hypothesis H-bias (next probe):** systematic bias of the *learned mean* —
the flow loss in ctrl-pt space allocates its mass over 6 tokens × 8 channels where
terminal fine detail is a tiny fraction, vs the waypoint loss supervising 50 steps
at per-step-equalized scale. Probe: prediction bias vs held-out demo continuations,
by phase (near/far from contact), A vs C (calibrate_duration-style pipeline).
**Implied fix if confirmed: decode-consistency auxiliary loss** — decode predicted
ctrl pts to per-step deltas inside training and penalize against demo steps
directly (differentiable: decode = fixed matmul). Unifies the supervision geometry
of both representations while keeping every spline capability. This is the top
candidate for the contact-region fix and a clean method-level contribution.

## 5. Follow-up work threads (for the record)
- TempoVLA-lite head-to-head (speed-as-input vs time-as-output) — §2.2.
- Deployment-stack Hz experiment (absolute setpoints + tracking controller) — held.
- F1″ decoupled shape-support/duration design — pocketed.
- Real-robot validation of stretch + selective speedup (needs hardware — Xiatao).
