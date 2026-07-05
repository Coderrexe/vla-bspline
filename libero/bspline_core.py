"""
Core cubic B-spline fitting / reconstruction utilities, shared across scripts.

Design decisions (see project memory):
  - We fit the ABSOLUTE eef trajectory (observation.state dims 0:6 = xyz + axis-angle),
    NOT the delta actions.
  - Gripper (state 6:8 / action 6) is bang-bang -> kept raw, never splined.
  - Clamped uniform knot vector on normalized time [0, 1]; a cubic needs n_ctrl >= 4.
  - Time is carried EXPLICITLY as the chunk duration (seconds) -> this scalar (and,
    later, a per-segment vector) is the supervision target for the time-allocation head,
    which is our novelty over the fixed-time B-spline VLA baseline.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import make_lsq_spline, BSpline

DEGREE = 3
POS_DIMS = (0, 1, 2)          # xyz
ROT_DIMS = (3, 4, 5)          # axis-angle
FIT_DIMS = POS_DIMS + ROT_DIMS
GRIP_STATE_DIMS = (6, 7)


def make_clamped_uniform_knots(n_ctrl: int, degree: int = DEGREE) -> np.ndarray:
    """Clamped uniform knot vector on [0, 1], length n_ctrl + degree + 1."""
    if n_ctrl < degree + 1:
        raise ValueError(f"cubic B-spline needs n_ctrl >= degree+1 = {degree + 1}, got {n_ctrl}")
    return np.concatenate([
        np.zeros(degree),
        np.linspace(0.0, 1.0, n_ctrl - degree + 1),
        np.ones(degree),
    ])


def fit_chunk(state_chunk: np.ndarray, n_ctrl: int, fit_dims=FIT_DIMS):
    """Least-squares fit a cubic B-spline to each fitted dim of one chunk.

    state_chunk: (L, D) absolute state window.
    Returns dict with control_points (len(fit_dims), n_ctrl), knots, and the
    per-dim reconstruction used for QC.
    """
    L = state_chunk.shape[0]
    if L < n_ctrl:
        raise ValueError(f"chunk length {L} < n_ctrl {n_ctrl}")
    u = np.linspace(0.0, 1.0, L)
    knots = make_clamped_uniform_knots(n_ctrl)
    cps = np.zeros((len(fit_dims), n_ctrl), dtype=np.float64)
    recon = np.zeros((L, len(fit_dims)), dtype=np.float64)
    for i, d in enumerate(fit_dims):
        spl = make_lsq_spline(u, state_chunk[:, d], knots, k=DEGREE)
        cps[i] = spl.c
        recon[:, i] = BSpline(knots, spl.c, DEGREE)(u)
    return {"control_points": cps, "knots": knots, "recon": recon, "n_ctrl": n_ctrl,
            "fit_dims": tuple(fit_dims)}


def eval_spline(control_points: np.ndarray, knots: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Evaluate a fitted spline at normalized times u in [0,1]. Returns (len(u), n_dims)."""
    n_dims = control_points.shape[0]
    out = np.zeros((len(u), n_dims))
    for i in range(n_dims):
        out[:, i] = BSpline(knots, control_points[i], DEGREE)(u)
    return out


def sample_at_hz(control_points: np.ndarray, knots: np.ndarray,
                 duration_s: float, hz: float) -> tuple[np.ndarray, np.ndarray]:
    """THE decoupling primitive: evaluate x(t) at an ARBITRARY control rate.

    Given a spline fit over a chunk of real duration `duration_s`, sample it at `hz`.
    Returns (t_seconds, values). This is what makes execution-Hz independent of
    training-Hz: the same spline can be sampled at 10, 20, 50, 1000 Hz.
    """
    n_steps = max(2, int(round(duration_s * hz)) + 1)
    t = np.linspace(0.0, duration_s, n_steps)
    u = t / duration_s
    return t, eval_spline(control_points, knots, u)


def chunk_rmse(recon: np.ndarray, state_chunk: np.ndarray, fit_dims=FIT_DIMS):
    """Per-group RMSE (pos in metres, rot in rad)."""
    err = {}
    pos_idx = [i for i, d in enumerate(fit_dims) if d in POS_DIMS]
    rot_idx = [i for i, d in enumerate(fit_dims) if d in ROT_DIMS]
    orig = state_chunk[:, list(fit_dims)]
    if pos_idx:
        err["pos_rmse"] = float(np.sqrt(np.mean((recon[:, pos_idx] - orig[:, pos_idx]) ** 2)))
    if rot_idx:
        err["rot_rmse"] = float(np.sqrt(np.mean((recon[:, rot_idx] - orig[:, rot_idx]) ** 2)))
    return err
