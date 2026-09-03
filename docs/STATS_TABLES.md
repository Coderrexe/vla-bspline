# Statistical appendix (TRI-protocol: Barnard exact + Bonferroni + CLD + beta intervals)

Batch tests were run ONCE on the final pre-committed sample sizes (no incremental re-testing; sequential designs reserved for hardware STEP).

### LIBERO main table (3 seeds pooled, n=1200/arm)

Pairwise Barnard exact, Bonferroni alpha = 0.05/3 = 0.0167

| arm | success (beta 94% interval) | CLD |
|---|---|---|
| A waypoint | 82.1% [79.9, 84.1] | a |
| B spline | 80.8% [78.6, 82.9] | a |
| C spline+time | 80.7% [78.4, 82.7] | a |

No pair significant after correction.

*Aggregate over suites; per-suite ordering claims additionally require protocol crossing (RESULTS 4c).*

### RoboCasa kettle (n=50/cell)

Pairwise Barnard exact, Bonferroni alpha = 0.05/3 = 0.0167

| arm | success (beta 94% interval) | CLD |
|---|---|---|
| A waypoint | 52.0% [39.0, 64.7] | a |
| C spline+time | 34.0% [22.9, 47.4] | a |
| C uniform-retimed | 34.0% [22.9, 47.4] | a |

No pair significant after correction.

### RoboCasa toaster (n=50/cell)

Pairwise Barnard exact, Bonferroni alpha = 0.05/3 = 0.0167

| arm | success (beta 94% interval) | CLD |
|---|---|---|
| A waypoint | 20.0% [11.6, 32.6] | a |
| C spline+time | 16.0% [8.7, 28.1] | a |
| C uniform-retimed | 16.0% [8.7, 28.1] | a |

No pair significant after correction.

### RoboCasa faucet (n=50/cell)

Pairwise Barnard exact, Bonferroni alpha = 0.05/3 = 0.0167

| arm | success (beta 94% interval) | CLD |
|---|---|---|
| A waypoint | 14.0% [7.3, 25.7] | a |
| C spline+time | 14.0% [7.3, 25.7] | a |
| C uniform-retimed | 6.0% [2.3, 15.8] | a |

No pair significant after correction.

### RoboCasa microwave (n=50/cell)

Pairwise Barnard exact, Bonferroni alpha = 0.05/3 = 0.0167

| arm | success (beta 94% interval) | CLD |
|---|---|---|
| C spline+time | 4.0% [1.3, 13.0] | a |
| A waypoint | 2.0% [0.5, 10.1] | a |
| C uniform-retimed | 2.0% [0.5, 10.1] | a |

No pair significant after correction.

### RoboCasa coffee (n=50/cell)

Pairwise Barnard exact, Bonferroni alpha = 0.05/3 = 0.0167

| arm | success (beta 94% interval) | CLD |
|---|---|---|
| A waypoint | 0.0% [0.1, 6.6] | a |
| C spline+time | 0.0% [0.1, 6.6] | a |
| C uniform-retimed | 0.0% [0.1, 6.6] | a |

No pair significant after correction.

### C-lang arms, libero_object (n=100)

Pairwise Barnard exact, Bonferroni alpha = 0.05/6 = 0.0083

| arm | success (beta 94% interval) | CLD |
|---|---|---|
| Cn8 twin | 89.0% [81.7, 93.5] | a |
| ClangMix | 88.0% [80.5, 92.8] | a |
| ClangAdv | 87.0% [79.3, 92.0] | a |
| Clang granular-only | 34.0% [25.8, 43.4] | b |

Significant pairs: Cn8 twin vs Clang granular-only (p=3.9e-16); Clang granular-only vs ClangMix (p=1.3e-15); Clang granular-only vs ClangAdv (p=6.7e-15)

### RoboCasa kettle FINAL (n=100/cell)

Pairwise Barnard exact, Bonferroni alpha = 0.05/15 = 0.0033

| arm | success (beta 94% interval) | CLD |
|---|---|---|
| C spline+time | 38.0% [29.4, 47.4] | a |
| B spline | 37.0% [28.5, 46.4] | a |
| C uniform+stretch | 37.0% [28.5, 46.4] | a |
| A waypoint | 35.0% [26.7, 44.4] | a |
| C self-paced | 33.0% [24.9, 42.3] | a |
| C interval | 29.0% [21.3, 38.2] | a |

No pair significant after correction.

*Event-density law: C-variants cluster; kettle (event-rich) parity, toaster/faucet (cap-dominated) C trails B.*

### RoboCasa toaster FINAL (n=100/cell)

Pairwise Barnard exact, Bonferroni alpha = 0.05/15 = 0.0033

| arm | success (beta 94% interval) | CLD |
|---|---|---|
| A waypoint | 32.0% [24.0, 41.3] | a |
| B spline | 30.0% [22.2, 39.2] | a |
| C interval | 28.0% [20.4, 37.1] | a |
| C self-paced | 22.0% [15.3, 30.7] | a |
| C spline+time | 21.0% [14.4, 29.6] | a |
| C uniform+stretch | 21.0% [14.4, 29.6] | a |

No pair significant after correction.

*Event-density law: C-variants cluster; kettle (event-rich) parity, toaster/faucet (cap-dominated) C trails B.*

### RoboCasa faucet FINAL (n=100/cell)

Pairwise Barnard exact, Bonferroni alpha = 0.05/15 = 0.0033

| arm | success (beta 94% interval) | CLD |
|---|---|---|
| A waypoint | 15.0% [9.5, 23.0] | a |
| B spline | 9.0% [5.0, 15.9] | a |
| C interval | 7.0% [3.6, 13.5] | a |
| C uniform+stretch | 7.0% [3.6, 13.5] | a |
| C spline+time | 6.0% [2.9, 12.2] | a |
| C self-paced | 4.0% [1.7, 9.6] | a |

No pair significant after correction.

*Event-density law: C-variants cluster; kettle (event-rich) parity, toaster/faucet (cap-dominated) C trails B.*

### Hard-5 cell: libero_10 t0 (pooled, selection pre-registered A-only)

Pairwise Barnard exact, Bonferroni alpha = 0.05/6 = 0.0083

| arm | success (beta 94% interval) | CLD |
|---|---|---|
| B spline | 57.4% [50.5, 63.9] | a |
| C n8 | 51.5% [43.4, 59.6] | a |
| A waypoint | 45.2% [38.9, 51.7] | a |
| C n10 | 44.7% [37.2, 52.3] | a |

No pair significant after correction.

### Hard-5 cell: libero_10 t4 (pooled, selection pre-registered A-only)

Pairwise Barnard exact, Bonferroni alpha = 0.05/6 = 0.0083

| arm | success (beta 94% interval) | CLD |
|---|---|---|
| B spline | 50.0% [43.2, 56.8] | a |
| C n8 | 45.4% [37.4, 53.6] | a |
| A waypoint | 43.3% [37.1, 49.8] | a |
| C n10 | 42.0% [34.7, 49.7] | a |

No pair significant after correction.

### Hard-5 cell: libero_spatial t5 (pooled, selection pre-registered A-only)

Pairwise Barnard exact, Bonferroni alpha = 0.05/6 = 0.0083

| arm | success (beta 94% interval) | CLD |
|---|---|---|
| B spline | 43.2% [36.6, 50.0] | a |
| A waypoint | 38.9% [32.3, 45.9] | a |
| C n8 | 32.8% [26.6, 39.7] | a |
| C n10 | 32.7% [26.0, 40.2] | a |

No pair significant after correction.

### Hard-5 cell: libero_10 t7 (pooled, selection pre-registered A-only)

Pairwise Barnard exact, Bonferroni alpha = 0.05/6 = 0.0083

| arm | success (beta 94% interval) | CLD |
|---|---|---|
| C n10 | 64.0% [56.4, 71.0] | a |
| B spline | 61.6% [54.8, 67.9] | a |
| A waypoint | 61.0% [54.5, 67.0] | a |
| C n8 | 49.2% [41.1, 57.4] | a |

No pair significant after correction.

### Hard-5 cell: libero_10 t6 (pooled, selection pre-registered A-only)

Pairwise Barnard exact, Bonferroni alpha = 0.05/6 = 0.0083

| arm | success (beta 94% interval) | CLD |
|---|---|---|
| C n8 | 60.8% [52.5, 68.4] | a |
| C n10 | 60.0% [52.3, 67.2] | a |
| A waypoint | 55.7% [49.2, 62.0] | a |
| B spline | 53.2% [46.3, 59.8] | a |

No pair significant after correction.

### Frequency dose-response, spatial t5 (matched 50s budgets, in-harness)

Pairwise Barnard exact, Bonferroni alpha = 0.05/10 = 0.0050

| arm | success (beta 94% interval) | CLD |
|---|---|---|
| C n8 @2x (3 reps pooled) | 71.3% [63.9, 77.7] | a |
| C n8 @3x | 62.0% [48.6, 73.7] | a |
| C n8 @2x + uniform a0.8 | 54.0% [40.9, 66.6] | a |
| C n8 @1x | 34.0% [22.9, 47.4] | b |
| A waypoint @2x | 0.0% [0.1, 6.6] | c |

Significant pairs: C n8 @1x vs C n8 @2x (3 reps pooled) (p=2e-06); C n8 @1x vs A waypoint @2x (p=7.6e-05); C n8 @2x (3 reps pooled) vs A waypoint @2x (p=1.3e-19); C n8 @3x vs A waypoint @2x (p=1.4e-11); C n8 @2x + uniform a0.8 vs A waypoint @2x (p=2.9e-09)

*Success tracks per-step command magnitude monotonically (0.75/0.47/0.375/0.25 for 1x / 2x+comp / 2x / 3x). A@2x = 0 (velocity-doubling overshoot; same collapse on all libero_10 hard tasks, 0/250 total).*

### Full spatial suite, 1x vs 2x (n=500/arm, paired)

Pairwise Barnard exact, Bonferroni alpha = 0.05/1 = 0.0500

| arm | success (beta 94% interval) | CLD |
|---|---|---|
| C n8 @1x (suite) | 77.8% [74.1, 81.1] | a |
| C n8 @2x (suite) | 76.2% [72.4, 79.6] | a |

No pair significant after correction.

*Suite-level parity (p=0.61 uncorrected Barnard) with task-level redistribution: t5 +40, t8 +12, t2 -36, t9 -20 (losers = longest tasks).*

### Capacity ladder, spatial suite (seeds pooled)

Pairwise Barnard exact, Bonferroni alpha = 0.05/10 = 0.0050

| arm | success (beta 94% interval) | CLD |
|---|---|---|
| A waypoint | 79.7% [77.8, 81.4] | a |
| C n12 (1 seed) | 79.4% [75.8, 82.6] | a |
| C n10 | 79.0% [77.0, 80.9] | a |
| C n8 | 77.1% [75.1, 78.9] | a |
| C n6 | 75.5% [73.6, 77.3] | b |

Significant pairs: A waypoint vs C n6 (p=0.0024)

*Monotone saturation toward the waypoint: 75.5 -> 77.3 -> 79.0 -> 79.4.*

### Capacity dose-response, libero_10 t7 (the capacity-bound cell)

Pairwise Barnard exact, Bonferroni alpha = 0.05/3 = 0.0167

| arm | success (beta 94% interval) | CLD |
|---|---|---|
| C n12 (1 seed) | 78.0% [65.2, 86.9] | a |
| C n10 | 64.0% [56.4, 71.0] | a |
| C n8 | 49.2% [41.1, 57.4] | b |

Significant pairs: C n8 vs C n10 (p=0.013); C n8 vs C n12 (1 seed) (p=0.00047)

*49 -> 64 -> 78 with control-point density; sp5 is capacity-immune across all four configs (rate-knob cell).*

### CALVIN — paired on identical official chains (stronger than Barnard)

Per-chain paired analysis, 3 seeds x n=1000; McNemar on SR1 + paired t on solved-count.

| contrast | paired t (solved) | McNemar SR1 p | sig @ 0.0083 |
|---|---|---|---|
| A waypoint vs C base | +8.28 | 4.3e-21 | YES |
| A waypoint vs C interval | -0.65 | 0.7 | no |
| A waypoint vs C uniform | -3.89 | 0.56 | t-only |
| C base vs C interval | -9.10 | 2e-20 | YES |
| C base vs C uniform | -12.14 | 3.5e-24 | YES |
| C interval vs C uniform | -3.37 | 0.3 | t-only |

### CALVIN language, n=600/condition — paired within-shard (final)

n = 600 rollouts/condition; avg_len plain 1.382 / neutral 1.407 / quick 1.553

| contrast | mean diff | paired t | p | p (Bonferroni x2) |
|---|---|---|---|---|
| quick - plain | +0.172 | 2.38 | 0.0175 | 0.0350 |
| quick - neutral | +0.147 | 2.04 | 0.0422 | 0.0844 |
| neutral - plain (specificity control) | +0.025 | 0.37 | 0.7141 | - |
