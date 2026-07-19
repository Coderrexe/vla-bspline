# Evaluation design (Xiatao T3): benchmarks, baselines, ablations, metrics, statistics

Drafted July 18 for team review. Sources: our protocol-variance findings
(RESULTS §4c/§9), the TRI statistical-evaluation article Xiatao assigned, and
the BSP paper's protocol (POSITIONING_BSP.md).

## 1. Benchmarks — the 15-task table (Xiatao T1)

| benchmark | tasks (5 each) | data regime | why it's in |
|---|---|---|---|
| LIBERO | 5 of the 40 (pick spanning object/spatial/goal/long) | scripted demos | capability ladder home turf; protect-slow regime |
| CALVIN ABCD→D | 5 most frequent chain subtasks (or full official chains as the headline + 5-task breakdown) | human teleop, noisy | sign-flip regime; seeded n=1000 machinery exists |
| RoboCasa | 5 incl. BSP's 4 (sink faucet, coffee button, microwave, close door) + 1 pick-place | MimicGen + human | 3rd morphology/sim; direct BSP comparison |

Task-selection rule (pre-registered, not post-hoc): for LIBERO/CALVIN pick by
demo-count ranking, not by our results; for RoboCasa pick BSP's four + the
highest-demo-count pick-place task.

## 2. Arms

**Main table (per benchmark):** A waypoint SmolVLA (best cadence) | C spline+
time (base decode) | C + promoted retiming (regime-appropriate: interval on
scripted, uniform on noisy).

**Baselines beyond A** (already implemented): smolvla_interp (path resampling
— kills "only splines can retarget" claims); smolvla_tempo (speed-as-input,
TempoVLA-style); smolvla_dsel (data-level retiming); BSP-style = our uniform
rung (cite their numbers on RoboCasa directly as external reference).

## 3. Ablations (each = one decode flag or one training toggle)

1. Granularity ladder: uniform / chunk-gated / interval / bidirectional —
   on one scripted + one noisy benchmark (the inversion is a headline figure).
2. Time-allocation ablation: B (fixed-T spline, no duration) vs C — isolates
   the duration head.
3. Capacity: n_ctrl 6/8, cap 16/24 (the density law).
4. θ dose-response (protect-slow boundary) on both regimes.
5. Cadence dose-response (nas sweep) — cadence-robustness claim.
6. Replan policy: fixed nas / self-paced / self-paced+margin (the margin law).
7. Chaining: none / velocity-pin / BSP-style time-re-anchor (new, cheap).
8. Slow-down: chunk-level and interval-level dilation on precision suites
   (in flight July 17-18).
9. Language granularity: episode-level vs Molmo segment-level prompts (T2).

## 4. Metrics

- Success rate (binary, per official benchmark criterion — pre-registered).
- **Avg completion time on successes** (BSP's metric — adopt everywhere).
- Realized speedup (steps ratio on matched solved cases, paired).
- Policy calls per episode (self-paced claims).
- Commanded jerk / executed jerk (smoothness; hardware-relevant).
- CALVIN: avg_len + SR1-5 (official).

## 5. Statistics (the TRI-article-compliant protocol)

**Design principle: paired everything.** Both arms see identical initial
conditions (CALVIN: identical official chains; LIBERO/RoboCasa: identical
env-seed sets). This is stronger than the TRI unpaired setting and shrinks
required n substantially.

- **Binary paired comparisons → McNemar's exact test** on discordant pairs
  (the paired analogue of their Barnard recommendation; report discordant
  counts n01/n10 alongside p).
- **Unpaired settings (only if pairing impossible) → Barnard's exact test**
  (scipy.stats.barnard_exact), per TRI.
- **Sequential/expensive settings (real robot, n≤50) → STEP** (TRI+Princeton
  sequential test): decide continue/stop after each rollout with controlled
  false-positive rate — exactly matches the 20-rollout real-world regime
  where BSP ran without any test.
- **Never** incrementally peek at a fixed-n exact test (TRI: p-hacking);
  either pre-commit n or use STEP.
- **Report per-seed values + pooled paired effect ± SE** (our standing
  practice); flag any claim whose per-seed signs disagree (as we did for
  uniform-vs-waypoint: "matches-or-beats, 2/3 seeds").
- **Figures: Bayesian beta-posterior violins** for success rates instead of
  overlapping CIs (TRI recommendation; trivial: Beta(1+s, 1+f) per cell).
- Protocol hygiene (TRI checklist): success criteria pre-registered; policies
  interleaved within eval sessions on hardware, evaluator-blind where
  possible; full conditions reported (n, seeds, env versions, budgets).

## 6. Statistical power reality check (from our own measured variance)

LIBERO/CALVIN same-checkpoint protocol noise: ±5-9 pts at n=100 → suite-level
claims need n≥300 pooled across ≥3 seed bases (our standing rule). CALVIN
training-seed sd on avg_len ≈ 0.08-0.15 → 3 training seeds is the floor for
any cross-arm claim; decode-only (same-checkpoint paired) claims are exempt
from training-seed variance and were significant at n=200 already.

## 7. What gets crossed before entering the paper

Every table cell that is (a) favorable to us AND (b) single-protocol gets a
second protocol/slice/seed pass before promotion. This caught: prof06's 96,
the uniform slice-1 dip, the s1000 "beats waypoint" t=3.18. The rule stays.

## 8. C-lang evaluation (Xiatao T2 — granular language)

Two questions, two protocols:

**8a. Robustness (does granular training cost standard use?).** Standard
libero_object battery on Clang_100k with the env's ORIGINAL suite-level
prompts — strings the model never saw paired with object episodes during
training (its object frames carried Molmo clauses). Twin: Cn8 (92–94 object).
A small drop is acceptable and interpretable (instruction-distribution shift);
a large drop bounds how the granular arm must be deployed (with a
sub-instruction provider).

**8b. Steering (the capability claim).** Same scene, same checkpoint,
different sub-instruction → measurably different motion. Protocol: for K
object-suite scenes with ≥2 candidate objects, roll T=30 steps under
sub-instruction variants ("move toward the {X}"), where X ∈ {target,
distractor}; metric = EEF displacement toward commanded object minus toward
the alternative (paired per scene), plus grasp-attempt choice at longer
horizons. Controls: (i) Cn8 given the same sub-instructions (should NOT
steer — it never learned clause-level language); (ii) Clang given the
original full-task prompt (baseline drift). Success = Clang steers
significantly above both controls. Implementation: record_rollouts-style
harness with a `--task_override` flag; ~K=10 scenes × 3 prompts × 5 samples.

Statistic: paired per-scene displacement differences, sign test + mean ± SE
(n = K×samples); this is a *mechanism* demonstration, not a benchmark score —
absolute steering magnitude matters less than the Clang-vs-Cn8 contrast.
