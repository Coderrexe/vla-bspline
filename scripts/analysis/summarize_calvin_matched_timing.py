"""Complete-matrix statistics; no partial-job or smoke efficacy reporting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from check_calvin_matched_timing import ARMS, check, check_results


SEEDS = [1000, 1001, 1002]


def interval(values):
    return np.quantile(values, [.025, .975]).tolist()


def summarize(audit, bootstraps=20000):
    arrays = {}
    for arm in ARMS:
        arrays[arm] = np.array([
            [r["solved"] for r in json.loads(Path(
                audit["cells"][f"{arm}_s{seed}"]["result"]).read_text())["records"]]
            for seed in SEEDS], dtype=float)
    nseeds, nchains = arrays[ARMS[0]].shape
    assert nseeds == 3 and nchains == 50
    rng = np.random.default_rng(20260905)
    # Draw the same chain indices across checkpoint seeds, retaining the paired
    # evaluation-state structure. Hierarchical intervals also resample seeds.
    chain_indices = rng.integers(nchains, size=(bootstraps, nchains))
    seed_indices = rng.integers(nseeds, size=(bootstraps, nseeds))
    contrasts = {
        "waypoint_retiming": arrays["A_linear"] - arrays["A_native"],
        "spline_retiming": arrays["C_retime"] - arrays["C_native"],
        "retimed_spline_minus_retimed_waypoint": arrays["C_retime"] - arrays["A_linear"],
        "retiming_interaction": (arrays["C_retime"] - arrays["C_native"]
                                 - arrays["A_linear"] + arrays["A_native"]),
    }
    summary = {"audit": audit, "bootstrap_draws": bootstraps, "bootstrap_seed": 20260905,
               "note": "Fresh matched-K5 screen; never pool with smoke or historical K10 results.",
               "arms": {}, "contrasts": {}}
    for arm, values in arrays.items():
        summary["arms"][arm] = {
            "per_seed_mean": values.mean(axis=1).tolist(), "mean": float(values.mean()),
            "chains": int(values.size),
            "survival": [float((values >= depth).mean()) for depth in range(1, 6)],
        }
    for name, values in contrasts.items():
        seed_effects = values.mean(axis=1)
        mean = float(seed_effects.mean())
        halfwidth = 4.302652729911275 * float(seed_effects.std(ddof=1)) / np.sqrt(3)
        chain_boot = values[:, chain_indices].mean(axis=(0, 2))
        hierarchical = values[seed_indices[:, :, None], chain_indices[:, None, :]].mean(axis=(1, 2))
        summary["contrasts"][name] = {
            "per_seed": seed_effects.tolist(), "mean": mean,
            "seed_t95_df2": [mean - halfwidth, mean + halfwidth],
            "paired_chain_bootstrap95_conditional_on_checkpoints": interval(chain_boot),
            "hierarchical_seed_and_paired_chain95": interval(hierarchical),
        }
    summary["analysis_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__), Path(__file__).with_name("check_calvin_matched_timing.py")]
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--input-type", choices=("logs", "results"), default="logs")
    parser.add_argument("--reset-audit", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.input_type == "logs":
        assert args.reset_audit is None, "reset certificate is only needed for result inputs"
        audit = check(args.inputs, 50, SEEDS)
    else:
        assert args.reset_audit is not None
        audit = check_results(args.inputs, 50, SEEDS, args.reset_audit)
    summary = summarize(audit)
    with args.out.open("x") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps({"output": str(args.out), "arms": summary["arms"],
                      "contrasts": summary["contrasts"]}, indent=2))


if __name__ == "__main__":
    main()
