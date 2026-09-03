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

## 9. FINAL paper experiment matrix (July 20, per Xiatao's prioritization)

### 9a. Scenario selection (his LIBERO-saturation concern, addressed)

| block | what | why |
|---|---|---|
| **LIBERO-Hard-5** | the 5 lowest per-task cells for the WAYPOINT baseline (selection rule fixed BEFORE looking at our arms' numbers — no cherry-picking in our favor; computed from A's own 3-seed per-task data) | answers "LIBERO is saturated": we evaluate exactly where it is NOT; selection by baseline difficulty is reviewer-proof |

**COMPUTED (July 21, canonical f20 battery, A = Afull/As1001/As1002_nas10 pooled
n=30/task; C shown for context only — never consulted for selection):**

| rank | cell | A (waypoint) | C (ours, base) |
|---|---|---|---|
| 1 | libero_10 task 0 | 33% (10/30) | 50% (20/40) |
| 2 | libero_10 task 4 | 47% (14/30) | 28% (11/40) |
| 3 | libero_spatial task 5 | 47% (14/30) | 32% (13/40) |
| 4 | libero_10 task 7 | 53% (16/30) | 58% (23/40) |
| 5 | **libero_10 task 6** (tie-break winner) | 60.0% (36/60) | 50% (20/40) |

Suite-level A cross-check: object 94.7 / goal 86.0 / spatial 84.3 / 10 63.3 —
matches the canonical battery, and confirms Xiatao's instinct: the unsaturated
frontier is LIBERO-Long. **Rank-5 tie-break (pre-registered, then executed July
29):** the initial ranking had a 4-way tie at 63% (19/30) among libero_10 tasks
{5, 6, 8, 9}; protocol = add A-arm episodes only (3 canonical seeds × libero_10
+10/task, jobs 2182178/2182179/2182230 → n=60/task), lowest pooled A% wins,
task_id breaks residual ties, C never consulted. Outcome: task 6 = 60.0%,
task 8 = 61.7%, tasks 5 & 9 = 68.3% → **task 6 takes rank 5**. FINAL
Hard-5 = libero_10 {0, 4, 6, 7} + libero_spatial {5}. Full per-task table:
`~/libero_pertask.tsv` on misha. **Top-up campaign COMPLETE (July 29, 20/20
cells, n=180–210/cell): final table + reads in paper/results.md §1b** (B≥A on
4/5 hard cells, t0 +12.2 p=.016 uncorr; C parity; sel06 −23 on spatial = γ-law;
frozen-set transparency note — pooled re-rank would swap t7→t8, full data
reported). Regenerate: `~/hard5_pool.py`.
| **CALVIN-5** | official 5-instruction chains (n=1000 protocol) | the field's accepted long-horizon language benchmark; already our flagship |
| **RoboCasa-5** | kettle / toaster-door / faucet / microwave / coffee-mug (pre-registered) | third morphology + human-teleop regime; reference-model-validated floors |
| **Real-world (2–3 tasks)** | cloth fold (primary), kitting throughput, careful-place with glassware | §9d |

### 9b. Main comparisons (arms per benchmark)

A waypoint · B spline fixed-T (LIBERO only — isolates time-allocation) ·
C spline+time · C+retime (regime-appropriate gating: interval on scripted,
uniform on human data — the data-regime law IS the setting rule) ·
C+self-paced · ClangAdv (LIBERO). External: BSP-style global-speedup arm =
our uniform rung at their single factor (their method is a special case of
our decode surface — one row shows it); speed-as-input (tempo) and
data-level (dsel) triad rows already measured at matched budget.

### 9c. Ablations (each isolates one design choice; all already measured)

1. capacity (n6 vs n8; h40 vs h24) — the density law
2. event segmentation (B vs C at matched capacity) — time-allocation's cost: zero
3. retiming granularity (uniform/chunk/interval × 2 data regimes) — the inversion
4. duration head usage (chunk-selective needs T̂; uniform doesn't) — what T̂ buys
5. replan schedule (fixed nas grid vs T̂−margin) — the margin law
6. conditioning mix (granular-only 34 vs mixed 88) + adverbs (87, steering t=7.98)
7. normalization/stats construction (window-sampled vs episode-contiguous; the
   RoboCasa loss-33 lesson) — appendix material

### 9d. Real-world tasks + metrics (speed + language showcase)

1. **Cloth fold** (novice-teleop demos = the compressible regime the census
   predicts): metrics = fold-success, wall-clock/fold at 1× vs retimed, fold
   quality rubric; claim = CALVIN sign-flip on hardware.
2. **Kitting throughput**: items/hour at matched success, T̂-gated selective
   vs uniform vs base — the industrial framing.
3. **Careful-place (glass)**: "quickly move the cup" vs "slowly and carefully
   place the glass" — language-commanded speed live on hardware; metrics =
   executed speed ratio, placement force/success. STEP sequential testing for
   all hardware comparisons (pre-registered thresholds; batch tests stay
   frozen at the sim sample sizes already run).

### 9e. Metrics (uniform across benchmarks)

success / avg_len · realized speedup on matched solved episodes · policy
calls per episode · beta-posterior intervals + Barnard/McNemar per protocol ·
commanded-vs-realized speed curves · (hardware) jerk + contact force.

### 9f. Language-backbone note (Xiatao pt 3)

SmolVLA's language tower is SmolLM2-class (~135–360M) — plausibly a ceiling
for clause-level steering, and pi0.5-scale backbones are the right test —
BUT the redundancy law says backbone size is not the binding constraint for
object steering in-domain: object words carry zero conditional information in
demo data regardless of encoder capacity. Adverb steering (t=7.98) proves the
small tower reads and uses language when language is informative. Position in
the paper: limitation + future work (pi0.5 port), with the redundancy law as
the data-design prescription that any backbone will need.
