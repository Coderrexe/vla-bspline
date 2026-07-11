"""Phase-resolved diagnosis of the spline representation near contact.

Questions (design doc BENCHMARK_DESIGN.md §4 step 1):
  Q1  Do B-spline fit residuals concentrate near gripper events (contact)?
      -> per-step residual binned by steps-to-next-toggle, absolute AND relative
         to local motion scale; n_ctrl 6 vs 8 vs 10 by phase.
  Q2  Do demos carry high-frequency action content near contact that a cubic
      low-passes? -> delta-action spectra near-contact vs transport windows.

Run on cluster:  python contact_diagnosis.py   (writes outputs/contact_diagnosis.json)
"""
from __future__ import annotations

import json

import numpy as np

from lerobot_io import episode_arrays, episode_indices
from validate_spline_head_math import basis_matrix

DEGREE, H = 3, 20
STRIDE = 4
MAX_EPS = 300
PHASE_BINS = [(0, 2), (3, 5), (6, 10), (11, 20), (21, 10_000)]  # steps to next toggle


def fit_ops(n_ctrl):
    u = np.arange(H + 1) / H
    B = basis_matrix(u, n_ctrl)
    return B, np.linalg.pinv(B[:, 1:n_ctrl - 1]), B[:, n_ctrl - 1:n_ctrl]


def main():
    ops = {n: fit_ops(n) for n in (6, 8, 10)}

    # residual accumulators: [n_ctrl][bin] -> list of per-step residuals
    res_abs = {n: [[] for _ in PHASE_BINS] for n in ops}
    res_rel = {n: [[] for _ in PHASE_BINS] for n in ops}
    # spectra accumulators
    spec_contact, spec_transport = [], []
    n_windows = 0

    for ep in episode_indices()[:MAX_EPS]:
        _, _, action = episode_arrays(int(ep))
        L = len(action)
        grip = action[:, 6]
        toggles = np.where(np.diff(grip) != 0)[0] + 1          # indices of toggle steps
        # steps-to-next-toggle for every t (large if none ahead)
        t2e = np.full(L, 10_000, dtype=int)
        for tog in toggles[::-1]:
            t2e[:tog] = np.minimum(t2e[:tog], tog - np.arange(tog))

        for a0 in range(0, L - H, STRIDE):
            w = action[a0:a0 + H, :6]
            path = np.concatenate([np.zeros((1, 6)), np.cumsum(w, axis=0)], axis=0)
            scale = max(np.linalg.norm(w, axis=1).mean(), 1e-6)  # local motion scale
            phases = t2e[a0:a0 + H]
            for n, (B, pinv_mid, b_last) in ops.items():
                p_end = path[-1:, :]
                c_mid = pinv_mid @ (path - b_last @ p_end)
                c = np.concatenate([np.zeros((1, 6)), c_mid, p_end], axis=0)
                rec = B @ c
                # per-STEP residual: error in the emitted delta at each step
                err = np.linalg.norm(np.diff(rec, axis=0) - w, axis=1)  # (H,)
                for bi, (lo, hi) in enumerate(PHASE_BINS):
                    m = (phases >= lo) & (phases <= hi)
                    if m.any():
                        res_abs[n][bi].extend(err[m].tolist())
                        res_rel[n][bi].extend((err[m] / scale).tolist())
            n_windows += 1

        # Q2 spectra: windows fully near contact vs fully transport
        for a0 in range(0, L - H, STRIDE):
            ph = t2e[a0:a0 + H]
            w = action[a0:a0 + H, :6]
            mag = np.abs(np.fft.rfft(w, axis=0)) ** 2            # (H/2+1, 6)
            tot = mag.sum()
            if tot < 1e-12:
                continue
            hf = mag[H // 4:].sum() / tot                        # top-half band energy
            if (ph <= 10).all():
                spec_contact.append(hf)
            elif (ph >= 21).all():
                spec_transport.append(hf)

    out = {"n_windows": n_windows, "phase_bins": PHASE_BINS,
           "residual_abs_mean": {}, "residual_rel_mean": {},
           "hf_energy_frac": {"near_contact": float(np.mean(spec_contact)),
                              "transport": float(np.mean(spec_transport)),
                              "n_contact_windows": len(spec_contact),
                              "n_transport_windows": len(spec_transport)}}
    for n in ops:
        out["residual_abs_mean"][f"n{n}"] = [float(np.mean(b)) if b else None for b in res_abs[n]]
        out["residual_rel_mean"][f"n{n}"] = [float(np.mean(b)) if b else None for b in res_rel[n]]

    with open("outputs/contact_diagnosis.json", "w") as f:
        json.dump(out, f, indent=1)

    print(f"windows: {n_windows}")
    print(f"\nHF energy fraction (demo deltas): near-contact {out['hf_energy_frac']['near_contact']:.3f} "
          f"vs transport {out['hf_energy_frac']['transport']:.3f} "
          f"(n={len(spec_contact)}/{len(spec_transport)})")
    hdr = "  ".join(f"{lo}-{hi if hi < 9999 else '+'}" for lo, hi in PHASE_BINS)
    print(f"\nper-step |residual| by steps-to-toggle   [{hdr}]")
    for n in ops:
        print(f"  n_ctrl={n} abs: " + "  ".join(f"{v:.4f}" for v in out["residual_abs_mean"][f"n{n}"]))
    print(f"\nper-step residual RELATIVE to local motion scale")
    for n in ops:
        print(f"  n_ctrl={n} rel: " + "  ".join(f"{v:.3f}" for v in out["residual_rel_mean"][f"n{n}"]))


if __name__ == "__main__":
    main()
