"""Read-only translation command/state lag diagnostic for native Apollo data.

This estimates correlations, not an action convention or safety certification.
Filtering removes idle frames, so exclude gaps outside 20--80 ms from this test.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    grouped = {lag: [[], []] for lag in range(-2, 6)}
    intervals = []
    for path in sorted((args.source / "episodes").glob("*/frames.parquet")):
        table = pq.read_table(path, columns=["action", "observation.state", "wallclock_ns"])
        action = np.asarray(table["action"].to_pylist())[:, :3]
        position = np.asarray(table["observation.state"].to_pylist())[:, 9:12]
        gap = np.diff(np.asarray(table["wallclock_ns"].to_pylist(), dtype=np.int64)) / 1e9
        intervals.append(gap)
        delta = np.diff(position, axis=0)
        for lag, (xs, ys) in grouped.items():
            lo, hi = max(0, -lag), min(len(delta)-lag, len(action))
            ix = np.arange(lo, hi)
            good = np.ones(len(ix), dtype=bool)
            for offset in range(min(0, lag), max(0, lag)+1):
                check = np.clip(ix+offset, 0, len(gap)-1)
                good &= (gap[check] >= .02) & (gap[check] <= .08)
            xs.append(action[ix[good]])
            ys.append(delta[(ix+lag)[good]])
    rows = []
    for lag, (xs, ys) in grouped.items():
        x, y = np.concatenate(xs), np.concatenate(ys)
        scale = np.sum(x*y) / max(np.sum(x*x), 1e-20)
        rows.append({"state_delta_lag_frames": lag, "samples": len(x),
                     "fitted_scalar_state_delta_per_command": float(scale),
                     "uncentered_cosine": float(np.sum(x*y)/np.sqrt(max(np.sum(x*x)*np.sum(y*y),1e-20))),
                     "translation_residual_rmse_m": float(np.sqrt(np.mean((y-scale*x)**2)))})
    dt = np.concatenate(intervals)
    result = {"source": str(args.source), "tested_frame_gaps_seconds": [.02, .08],
              "retained_frame_gap_seconds_quantiles": np.quantile(dt, [0,.5,.9,.99,1]).tolist(),
              "fraction_gaps_over_80ms": float(np.mean(dt>.08)), "translation_lag_tests": rows,
              "note": "Do not change training targets based on this diagnostic alone; confirm runtime command semantics"}
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
