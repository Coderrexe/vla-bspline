# Related-work + evaluation-standards calibration (July 30, 2026)

> **ICRA acceptance calibration added Aug 6 — see §E at bottom** (4 verified
> ICRA-2025/2026-accepted PDFs read; verdict: sim-only 35–50%, with hardware
> section 65–75%; risk concentrated in headline framing, backbone count,
> hardware — NOT rigor).

*Source: full-PDF reads of BSP (arXiv 2607.09648), TRI LBM (2507.05331), BEAST
(2506.06072, NeurIPS 2025). Purpose: calibrate our evaluation package against
the field's standard and lock the BSP positioning before writing.*

## A. Evaluation-standards table

| Paper | Benchmarks | n per cell | Seeds | Stats | Real robot? |
|---|---|---|---|---|---|
| BSP (Jul 2026) | Push-T, RoboMimic, RoboCasa (4), 3 real tasks | real 20; sim 50 × best-of-3 ckpts | 1 | none (raw fractions) | yes (3 tasks) |
| TRI LBM (2025) | 8 real tasks × 3 conds (1,800 trials), Drake sim | real 50, sim 200 | 1 ckpt (caveated) | Beta posteriors, sequential paired tests, Bonferroni+CLD, blind A/B | yes, extensively |
| BEAST (NeurIPS'25) | CALVIN 1000-chain, LIBERO, ALOHA, 8 real tasks | LIBERO 50/task; real 10–30 | ~1 | mean SR only; reports 617 Hz / 19 ms latency | yes (3 setups) |
| **Ours** | LIBERO 4 suites, CALVIN 1000-chain, RoboCasa 5 | 50–210; 3,000 paired | **3** | paired t / Barnard / Bonferroni / CLD | not yet (Aug plan) |

## B. BSP deep-dive (verified from the PDF)

- Cubic B-splines, adaptive knot fitting (ε tolerance), predicts fixed segments
  [16 knots + control points] in Diffusion-Policy/ACT backbones; 10 Hz policy,
  100 Hz sampling; **speedup = manually chosen constant** (Alg. 2 input);
  inference-time segment alignment = boundary-MSE repair (disabled in sim).
- Numbers: table cleaning 2× time-cut at preserved success; speed stacking
  8/20→16/20 (1×) but **0/20 at 4×** (controller tracking limit, their §6);
  RoboCasa close-door 27→46 / 40→60; alignment ablation −20 pts at 4×.
- **What BSP lacks (verified):** (1) learned duration/time-allocation — speed
  is open-loop operator-chosen; (2) event-aligned chunking/replanning;
  (3) language anything (no text encoder at all); (4) multi-seed/CIs/tests
  (and sim uses best-of-3 checkpoint selection).
- **⚠️ Positioning correction:** BSP DOES retime at decode time (that's their
  core claim). Never write "BSP lacks decode-time rate adaptation." Our
  differentiators: timing is **learned** (not operator-set), event-aligned,
  language-addressable — and our rate adaptation **increases** success
  (34→72% at 2×) where theirs monotonically degrades (0/20 at 4×).
- Their hardware results are *favorable* to us: splines survive hardware and
  deliver 2× preserved-success; what fails there is exactly the uniform,
  unlearned retiming our method replaces.

## C. Honest gaps in our package (action items)

1. **Real robot** — biggest exposure; all three have it. August plan; BSP shows
   the binding constraint is controller tracking, which sim under-surfaces.
2. **Compute/latency reporting** — BEAST sets the standard (617 Hz, 19 ms,
   training cost). Action: report spline-head decode latency vs waypoint,
   policy-call rate, realized (not commanded) speedups, training compute.
   (bench_lat.sbatch ran earlier — harvest/refresh those numbers.)
3. Add Wilson/Beta CIs per cell + multiplicity note in final tables (mostly in
   paper_stats.py already); state the fixed checkpoint rule ("always last" —
   contrast BSP's best-of-3); cite TRI's ~50%-difficulty targeting to justify
   Hard-5 cell selection.
4. **Where we exceed the field (say explicitly):** 3 training seeds (TRI
   explicitly caveats 1-ckpt; BSP/BEAST effectively single-seed, test-free);
   n up to 210/cell + 3,000 paired chains vs their 10–50; paired significance
   testing throughout; pre-registered predictions.

## D. Locked positioning sentences vs BSP

1. Closest to our work, BSP replaces discrete chunks with B-spline segments
   and accelerates by uniform temporal rescaling; the speedup factor is a
   manually chosen open-loop constant, and with no model of how long motion
   should take, aggressive uniform speedups collapse (0/20 at 4×).
2. We factorize shape from timing and *learn* time allocation, making speed a
   task-informed decode-time decision — where BSP degrades monotonically with
   speed, our rate adaptation *increases* success on the hardest cells
   (34→72% at 2×, p=1e-4) while the waypoint baseline collapses to 0%.
3. BSP and BEAST show spline actions are smoother and cheaper to decode, but
   neither predicts duration: BEAST uses a fixed normalized time base; BSP's
   knots reproduce demonstration timing.
4. BSP builds on language-free DP/ACT; we instantiate spline+duration in a
   VLA and show timing is language-addressable ("quickly" → +16% speed).
5. BSP's segment alignment repairs boundary mismatch post hoc; our
   event-scheduled replanning aligns boundaries with task events by
   construction — validated with 3 seeds and paired tests vs their
   single-seed 20-rollout comparisons.

## E. ICRA acceptance calibration (Aug 6, 2026 — four verified accepted PDFs)

Papers read in full (acceptance verified via arXiv comments): Discrete Policy
(ICRA'25, 2409.18707), ITPS (ICRA'25, 2411.16627, MIT/NVIDIA), ACG (ICRA'26,
2510.22201), FPO (ICRA'26, 2510.09976 — **sim-only LIBERO paper, accepted**).

**Key facts about the bar:** none of the four has a single hypothesis test; per-cell
n = 10–24 rollouts; one is 100% simulation; one has zero external baselines; what
carried each was a big headline delta or a striking capability figure.

**Where we exceed the bar:** stats rigor (by a wide margin — pre-registration,
Barnard/Bonferroni, 3 seeds, n≤600 paired), per-cell n (50–210), benchmark breadth
(3 suites incl. official CALVIN-1000), mechanism evidence (dose-responses,
command-magnitude curve), and a baseline-structurally-impossible capability
(A@2× = 0/250 vs our 72%).

**Where we fall short:** (1) headline table is parity, not a big delta — the wins
live in hard cells and capability columns → the paper MUST lead with the
capability asymmetry, not the parity table; (2) single small backbone (ACG showed
3 backbones incl. SmolVLA — that's the local norm for head-level claims);
(3) internal baselines only — need vanilla SmolVLA/ACT-DP-chunking/BSP on one
shared table; (4) no real robot (FPO proves not disqualifying, but it's the top
reviewer-objection risk, ~65%).

**Verdict (ICRA 2027, submit Sept 2026; base accept rate ~40–46%):**
- **Sim-only, as-is: 35–50%** — coin flip, hinges on framing + reviewer draw.
- **With a competent 2–3-task hardware section (cloth-fold + BSP Speed-Stacking
  head-to-head, timing claims demonstrated on hardware): 65–75%** — above the
  measured bar on every axis.

**What August should buy, in priority order:** (1) hardware timing demos,
(2) headline reframing around the capability asymmetry, (3) one more backbone
OR a direct BSP comparison row. Rigor is already surplus — it's what gets the
paper *cited*, not what gets it *accepted*.

## F. Language steering and long-horizon positioning (Aug 22, 2026)

| closest direction | what it adds | distinction to preserve |
|---|---|---|
| [VLAs-as-Tools](https://arxiv.org/abs/2605.13119) | a high-level VLM selects specialized VLA tools and receives progress feedback; +4.8 points on LIBERO-Long | our clause program remains inside one end-to-end VLA/action head rather than a tool family |
| [Long-VLA](https://arxiv.org/abs/2508.19958) | phase-aware input masking for long-horizon manipulation | our intervention makes phases language-addressable and executable at decode time |
| [CAST](https://arxiv.org/abs/2508.13446) | counterfactual labels/actions improve fine-grained instruction following | complements Quinten's decorrelated-object data; our primary test concerns sequential clause execution and head representation |
| [LIBERO-CF / CAG](https://arxiv.org/abs/2602.17659) | benchmark and inference guidance for vision-over-language failures | motivates alternate-goal scoring; wrong-prompt suppression alone is not enough |
| [CofactVLA](https://arxiv.org/abs/2608.04396) | counterfactual flow guidance to remove visual confounding | makes a direct matched A/C decorrelated-target comparison essential before claiming spline-specific grounding |

**Positioning lock.** Do not claim that clauses or phase annotations are new by
themselves. The paper-specific claim is that an event-aligned spline action head
turns a compact trajectory into a language-addressable execution unit: the same
head supports clause programs, learned event timing, decode-rate adaptation, and
retiming without a separate high-level tool policy. The clean differentiating
experiment is the matched waypoint/spline clause factorial plus normal-versus-
reversed clause order; the RoboCasa complement is alternate-goal completion, not
merely suppression of the original reward.
