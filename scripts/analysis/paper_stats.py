"""Paper-grade statistics per the TRI article (EVAL_DESIGN §5):
  - Barnard's exact test for unpaired batch contrasts (scipy.stats.barnard_exact)
  - exact paired tests (McNemar / paired t on per-chain solved counts) where
    per-episode pairing exists (CALVIN identical official chains)
  - beta(1+s, 1+f) posteriors -> mean and 94% HDI-ish quantile intervals
  - Bonferroni correction within each table + Compact Letter Display
Writes docs-ready markdown to stdout. Run on misha (needs the CALVIN records).

  python paper_stats.py --calvin_dir ~/scratch/vla_bspline/outputs
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os

import numpy as np
from scipy import stats

# ---------- registries of final aggregate cells (n successes / n trials) ----------
LIBERO_MAIN = {  # 3-seed pooled per arm, native rate, all suites (n=1200 = 4x100x3)
    "A waypoint": (985, 1200),      # 82.1% * 1200
    "B spline": (970, 1200),        # 80.8%
    "C spline+time": (968, 1200),   # 80.7%
}
ROBOCASA = {  # 5 tasks x 50
    "A waypoint": ({"kettle": (26, 50), "toaster": (10, 50), "faucet": (7, 50),
                    "microwave": (1, 50), "coffee": (0, 50)}),
    "C spline+time": ({"kettle": (17, 50), "toaster": (8, 50), "faucet": (7, 50),
                       "microwave": (2, 50), "coffee": (0, 50)}),
    "C uniform-retimed": ({"kettle": (17, 50), "toaster": (8, 50), "faucet": (3, 50),
                           "microwave": (1, 50), "coffee": (0, 50)}),
}
CLANG = {
    "Cn8 twin": (89, 100), "Clang granular-only": (34, 100),
    "ClangMix": (88, 100), "ClangAdv": (87, 100),
}


def beta_summary(s, n):
    a, b = 1 + s, 1 + (n - s)
    lo, hi = stats.beta.ppf([0.03, 0.97], a, b)
    return f"{100*s/n:.1f}% [{100*lo:.1f}, {100*hi:.1f}]"


def barnard(s1, n1, s2, n2):
    tbl = [[s1, n1 - s1], [s2, n2 - s2]]
    try:
        r = stats.barnard_exact(tbl, alternative="two-sided")
        return r.pvalue
    except Exception:
        return stats.fisher_exact(tbl)[1]


def cld(names, pmat, alpha):
    """Compact letter display: greedy letter assignment; arms sharing a letter
    are NOT significantly different at (Bonferroni-corrected) alpha."""
    order = sorted(range(len(names)), key=lambda i: -RATES[names[i]])
    letters = {n: "" for n in names}
    letter = ord("a")
    for i in order:
        if letters[names[i]]:
            continue
        group = [i]
        for j in order:
            if j == i or letters[names[j]]:
                continue
            if all(pmat[min(k, j), max(k, j)] >= alpha for k in group):
                group.append(j)
        for k in group:
            letters[names[k]] += chr(letter)
        letter += 1
    return letters


RATES = {}


def table(title, cells, note=""):
    global RATES
    names = list(cells)
    m = len(names)
    n_tests = m * (m - 1) // 2
    alpha = 0.05 / max(n_tests, 1)
    RATES = {k: cells[k][0] / cells[k][1] for k in names}
    pmat = np.ones((m, m))
    print(f"\n### {title}\n")
    print(f"Pairwise Barnard exact, Bonferroni alpha = 0.05/{n_tests} = {alpha:.4f}\n")
    print("| arm | success (beta 94% interval) | CLD |")
    print("|---|---|---|")
    for i, j in itertools.combinations(range(m), 2):
        s1, n1 = cells[names[i]]
        s2, n2 = cells[names[j]]
        pmat[i, j] = pmat[j, i] = barnard(s1, n1, s2, n2)
    letters = cld(names, pmat, alpha)
    for k in sorted(names, key=lambda x: -RATES[x]):
        print(f"| {k} | {beta_summary(*cells[k])} | {letters[k]} |")
    sig = [(names[i], names[j], pmat[i, j]) for i, j in itertools.combinations(range(m), 2)
           if pmat[i, j] < alpha]
    if sig:
        print("\nSignificant pairs: " + "; ".join(f"{a} vs {b} (p={p:.2g})" for a, b, p in sig))
    else:
        print("\nNo pair significant after correction.")
    if note:
        print(f"\n*{note}*")


def calvin_paired(base_dir):
    print("\n### CALVIN — paired on identical official chains (stronger than Barnard)\n")
    print("Per-chain paired analysis, 3 seeds x n=1000; McNemar on SR1 + paired t on solved-count.\n")

    def pooled(pats):
        recs = []
        for p in pats:
            for f in sorted(glob.glob(os.path.join(base_dir, "calvin_eval_" + p + ".json"))):
                recs += json.load(open(f))["records"]
        return np.array([r["solved"] for r in recs], float)

    arms = {
        "A waypoint": ["calv_fA_*", "calv_s1001_A_*", "calv_s1002_A_*"],
        "C base": ["calv_fC_*", "calv_s1001_C_*", "calv_s1002_C_*"],
        "C interval": ["calv_fP_*", "calv_s1001_P_*", "calv_s1002_P_*"],
        "C uniform": ["calv_s1000_U_*", "calv_s1001_U_*", "calv_s1002_U_*"],
    }
    sv = {k: pooled(v) for k, v in arms.items()}
    n_tests = 6
    alpha = 0.05 / n_tests
    print(f"| contrast | paired t (solved) | McNemar SR1 p | sig @ {alpha:.4f} |")
    print("|---|---|---|---|")
    for a, b in itertools.combinations(arms, 2):
        d = sv[a] - sv[b]
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        x, y = sv[a] >= 1, sv[b] >= 1
        b01, b10 = int((~x & y).sum()), int((x & ~y).sum())
        p_mc = stats.binomtest(min(b01, b10), b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
        star = "YES" if (p_mc < alpha and abs(t) > 3) else ("t-only" if abs(t) > 3 else "no")
        print(f"| {a} vs {b} | {t:+.2f} | {p_mc:.2g} | {star} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calvin_dir", default=None)
    args = ap.parse_args()

    print("# Statistical appendix (TRI-protocol: Barnard exact + Bonferroni + CLD + beta intervals)")
    print("\nBatch tests were run ONCE on the final pre-committed sample sizes "
          "(no incremental re-testing; sequential designs reserved for hardware STEP).")

    table("LIBERO main table (3 seeds pooled, n=1200/arm)", LIBERO_MAIN,
          "Aggregate over suites; per-suite ordering claims additionally require "
          "protocol crossing (RESULTS 4c).")
    for task in ("kettle", "toaster", "faucet", "microwave", "coffee"):
        cells = {arm: ROBOCASA[arm][task] for arm in ROBOCASA}
        table(f"RoboCasa {task} (n=50/cell)", cells)
    table("C-lang arms, libero_object (n=100)", CLANG)
    if args.calvin_dir:
        calvin_paired(os.path.expanduser(args.calvin_dir))


if __name__ == "__main__":
    main()
