"""Shared event segmentation and raw v2 B-spline target construction.

This module is the single source of truth for the variable-duration head's
training boundaries.  Dataset-statistics generators import these functions
instead of maintaining a NumPy approximation of the policy logic.

Boundary convention is deliberately explicit: if an event is observed at
action index ``k``, the duration target is ``T = k`` and the fitted action
segment is ``actions[:k]``.  The event action itself belongs to the next chunk.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
from torch import Tensor


class EventMasks(NamedTuple):
    """Per-step event masks after enforcing ``min_seg``."""

    toggle: Tensor
    pause: Tensor
    episode_end: Tensor


class EventSplineTargets(NamedTuple):
    """Unnormalized raw targets produced by the v2 action head."""

    pose_ctrl: Tensor
    grip_ctrl: Tensor
    pass_ctrl: Tensor | None
    duration: Tensor


def bspline_basis(u: Tensor, n_ctrl: int, degree: int = 3) -> Tensor:
    """Cubic B-spline basis on clamped uniform knots over ``[0, 1]``."""

    device = u.device
    dtype = torch.float64
    u = u.to(dtype).clamp(0.0, 1.0)
    knots = torch.cat(
        [
            torch.zeros(degree, dtype=dtype, device=device),
            torch.linspace(0, 1, n_ctrl - degree + 1, dtype=dtype, device=device),
            torch.ones(degree, dtype=dtype, device=device),
        ]
    )
    n_spans = len(knots) - 1
    basis = torch.zeros(len(u), n_spans, dtype=dtype, device=device)
    for j in range(n_spans):
        left, right = knots[j], knots[j + 1]
        if right > left:
            covered = (u >= left) & ((u < right) | ((right >= 1.0) & (u <= 1.0)))
            basis[:, j] = covered.to(dtype)
    for order in range(1, degree + 1):
        next_basis = torch.zeros(len(u), n_spans - order, dtype=dtype, device=device)
        for j in range(n_spans - order):
            den1 = (knots[j + order] - knots[j]).item()
            den2 = (knots[j + order + 1] - knots[j + 1]).item()
            term = torch.zeros(len(u), dtype=dtype, device=device)
            if den1 > 0:
                term = term + (u - knots[j]) / den1 * basis[:, j]
            if den2 > 0:
                term = term + (knots[j + order + 1] - u) / den2 * basis[:, j + 1]
            next_basis[:, j] = term
        basis = next_basis
    return basis


def make_event_operator_banks(
    *,
    n_ctrl: int,
    degree: int,
    min_seg: int,
    horizon_max: int,
    fit_end_weight: float | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build the exact float32 operator banks used by production target fitting."""

    n_lengths = horizon_max - min_seg + 1
    pm_bank = torch.zeros(n_lengths, n_ctrl - 2, horizon_max + 1, dtype=torch.float64, device=device)
    bl_bank = torch.zeros(n_lengths, horizon_max + 1, 1, dtype=torch.float64, device=device)
    pg_bank = torch.zeros(n_lengths, n_ctrl, horizon_max, dtype=torch.float64, device=device)
    bp_bank = torch.zeros(n_lengths, horizon_max + 1, n_ctrl, dtype=torch.float64, device=device)
    for i, duration in enumerate(range(min_seg, horizon_max + 1)):
        u_path = torch.arange(duration + 1, dtype=torch.float64, device=device) / duration
        path_basis = bspline_basis(u_path, n_ctrl, degree)
        if fit_end_weight is not None:
            weight = float(fit_end_weight)
            sqrt_weight = torch.sqrt(
                1.0 + (weight - 1.0) * ((u_path - 0.75) / 0.25).clamp(0, 1)
            )
            pinv_mid = (
                torch.linalg.pinv(sqrt_weight.unsqueeze(1) * path_basis[:, 1 : n_ctrl - 1])
                * sqrt_weight.unsqueeze(0)
            )
        else:
            pinv_mid = torch.linalg.pinv(path_basis[:, 1 : n_ctrl - 1])
        pm_bank[i, :, : duration + 1] = pinv_mid
        bl_bank[i, : duration + 1, :] = path_basis[:, n_ctrl - 1 : n_ctrl]
        bp_bank[i, : duration + 1, :] = path_basis

        u_grip = torch.arange(duration, dtype=torch.float64, device=device) / max(duration - 1, 1)
        pg_bank[i, :, :duration] = torch.linalg.pinv(bspline_basis(u_grip, n_ctrl, degree))

    # Production registers float32 buffers, so offline statistics must fit in
    # float32 as well instead of silently using higher-precision SciPy targets.
    return pm_bank.float(), bl_bank.float(), pg_bank.float(), bp_bank.float()


def event_masks(
    actions: Tensor,
    pad: Tensor | None,
    *,
    pose_lo: int,
    grip_idx: int,
    min_seg: int,
    pause_frac: float,
) -> EventMasks:
    """Return toggle, pause, and episode-end masks using production semantics."""

    if actions.ndim != 3:
        raise ValueError(f"actions must have shape (B, H, D), got {tuple(actions.shape)}")
    batch_size, horizon, _ = actions.shape
    if not 0 <= pose_lo <= actions.shape[-1] - 6:
        raise ValueError(f"invalid six-dimensional pose slice starting at {pose_lo}")
    if not 0 <= grip_idx < actions.shape[-1]:
        raise ValueError(f"invalid gripper index {grip_idx}")
    if not 1 <= min_seg <= horizon:
        raise ValueError(f"min_seg={min_seg} must lie in [1, {horizon}]")

    grip = actions[..., grip_idx]
    toggle = torch.zeros(batch_size, horizon, dtype=torch.bool, device=actions.device)
    toggle[:, 1:] = grip[:, 1:] != grip[:, :-1]

    speed = actions[..., pose_lo : pose_lo + 6].norm(dim=-1)
    # torch.quantile(0.5) matches NumPy's interpolated median for even windows;
    # torch.median would select the lower-middle element.
    median_speed = torch.quantile(speed, 0.5, dim=1, keepdim=True) + 1e-9
    low = speed < pause_frac * median_speed
    pause = torch.zeros_like(toggle)
    pause[:, 1:] = low[:, 1:] & low[:, :-1]

    if pad is None:
        episode_end = torch.zeros_like(toggle)
    else:
        if pad.shape != (batch_size, horizon):
            raise ValueError(
                f"pad must have shape {(batch_size, horizon)}, got {tuple(pad.shape)}"
            )
        episode_end = pad.to(device=actions.device, dtype=torch.bool).clone()

    toggle[:, :min_seg] = False
    pause[:, :min_seg] = False
    episode_end[:, :min_seg] = False
    return EventMasks(toggle, pause, episode_end)


def first_event_indices(
    actions: Tensor,
    pad: Tensor | None,
    *,
    pose_lo: int,
    grip_idx: int,
    min_seg: int,
    horizon_max: int,
    pause_frac: float,
) -> Tensor:
    """Return ``T`` for each window, with the event action excluded from the segment."""

    if actions.shape[1] != horizon_max:
        raise ValueError(
            f"expected windows of horizon_max={horizon_max}, got {actions.shape[1]}"
        )
    masks = event_masks(
        actions,
        pad,
        pose_lo=pose_lo,
        grip_idx=grip_idx,
        min_seg=min_seg,
        pause_frac=pause_frac,
    )
    any_event_mask = masks.toggle | masks.pause | masks.episode_end
    has_event = any_event_mask.any(dim=1)
    first = torch.argmax(any_event_mask.int(), dim=1)
    duration = torch.where(has_event, first, torch.full_like(first, horizon_max))
    return duration.clamp(min_seg, horizon_max)


def first_event_types(
    actions: Tensor,
    pad: Tensor | None,
    *,
    pose_lo: int,
    grip_idx: int,
    min_seg: int,
    horizon_max: int,
    pause_frac: float,
) -> list[str]:
    """Classify first events for offline diagnostics (toggle > pause > end > cap)."""

    masks = event_masks(
        actions,
        pad,
        pose_lo=pose_lo,
        grip_idx=grip_idx,
        min_seg=min_seg,
        pause_frac=pause_frac,
    )
    duration = first_event_indices(
        actions,
        pad,
        pose_lo=pose_lo,
        grip_idx=grip_idx,
        min_seg=min_seg,
        horizon_max=horizon_max,
        pause_frac=pause_frac,
    )
    kinds: list[str] = []
    for row, index in enumerate(duration.tolist()):
        if index < horizon_max and masks.toggle[row, index]:
            kinds.append("toggle")
        elif index < horizon_max and masks.pause[row, index]:
            kinds.append("pause")
        elif index < horizon_max and masks.episode_end[row, index]:
            kinds.append("episode_end")
        else:
            kinds.append("cap")
    return kinds


def build_event_spline_targets(
    actions: Tensor,
    pad: Tensor | None,
    *,
    pm_bank: Tensor,
    bl_bank: Tensor,
    pg_bank: Tensor,
    pose_lo: int,
    grip_idx: int,
    pass_dims: Sequence[int] = (),
    min_seg: int,
    horizon_max: int,
    pause_frac: float,
) -> EventSplineTargets:
    """Fit the production v2 raw control-point and duration targets."""

    duration = first_event_indices(
        actions,
        pad,
        pose_lo=pose_lo,
        grip_idx=grip_idx,
        min_seg=min_seg,
        horizon_max=horizon_max,
        pause_frac=pause_frac,
    )
    pose = actions[..., pose_lo : pose_lo + 6]
    grip = actions[..., grip_idx]
    passthrough = actions[..., list(pass_dims)] if pass_dims else None
    if pad is not None:
        pose = pose * (~pad.to(dtype=torch.bool, device=actions.device)).unsqueeze(-1).to(pose.dtype)

    index = torch.arange(horizon_max, device=actions.device)
    segment_mask = (index.unsqueeze(0) < duration.unsqueeze(1)).to(pose.dtype)
    pose_segment = pose * segment_mask.unsqueeze(-1)
    path = torch.cat(
        [torch.zeros_like(pose_segment[:, :1]), torch.cumsum(pose_segment, dim=1)], dim=1
    )

    bank_index = duration - min_seg
    pinv_mid = pm_bank[bank_index]
    basis_last = bl_bank[bank_index]
    pinv_grip = pg_bank[bank_index]
    endpoint = path.gather(1, duration.view(-1, 1, 1).expand(-1, 1, path.shape[-1]))
    residual = path - basis_last * endpoint
    middle = torch.einsum("bmh,bhd->bmd", pinv_mid, residual)
    pose_ctrl = torch.cat([torch.zeros_like(endpoint), middle, endpoint], dim=1)
    grip_ctrl = torch.einsum("bnh,bh->bn", pinv_grip, grip)
    pass_ctrl = (
        torch.einsum("bnh,bhk->bnk", pinv_grip, passthrough)
        if passthrough is not None
        else None
    )
    return EventSplineTargets(pose_ctrl, grip_ctrl, pass_ctrl, duration)
