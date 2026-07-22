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
