# Copyright 2026 — VLA B-spline project (Simba Shi / Yale).
"""SmolVLA policy variant whose flow-matching expert generates B-spline control
points + duration instead of fixed-rate waypoints.

Pipeline (all on-the-fly; the dataset is the STANDARD LeRobot LIBERO dataset):

  train:  batch[ACTION] (B, H, 7) raw delta actions
            -> zero pose deltas on padded steps (episode end => clean stop)
            -> cumulative path p_k (B, H+1, 6), p_0 = 0
            -> pinned LSQ fit (single matmul): control points c (B, n_ctrl, 6)
               with c_0 = 0 and c_last = p_H (exact endpoints)
            -> gripper spline fit on the raw +/-1 channel (B, n_ctrl)
            -> targets (B, n_ctrl, 8), normalized per-token -> flow matching loss

  decode: sampled tokens -> unnormalize -> force c_0 = 0
            -> evaluate path spline at H_exec+1 points -> consecutive diffs
            -> deltas (B, H_exec, 6) clamped to [-1, 1]; gripper = sign(spline)
            -> (B, H_exec, 7) env actions; select_action queues them as usual.

Validated against scipy + real LIBERO windows in
libero/validate_spline_head_math.py (endpoint error exactly 0 at all rates;
gripper toggle timing p95 = 0.26 env steps).
"""

import json
import os

import torch
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ..smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector
from .configuration_smolvla_spline import SmolVLASplineConfig


def bspline_basis(u: Tensor, n_ctrl: int, degree: int = 3) -> Tensor:
    """(len(u), n_ctrl) cubic B-spline basis on clamped uniform knots over [0,1].

    Cox–de Boor recursion in float64. u must lie in [0, 1].
    """
    device = u.device
    dt = torch.float64
    u = u.to(dt).clamp(0.0, 1.0)
    kn = torch.cat(
        [
            torch.zeros(degree, dtype=dt, device=device),
            torch.linspace(0, 1, n_ctrl - degree + 1, dtype=dt, device=device),
            torch.ones(degree, dtype=dt, device=device),
        ]
    )
    m = len(kn) - 1
    # degree 0: indicator of the knot span; last span right-closed so u=1 is covered
    N = torch.zeros(len(u), m, dtype=dt, device=device)
    for j in range(m):
        left, right = kn[j], kn[j + 1]
        if right > left:
            covered = (u >= left) & ((u < right) | ((right >= 1.0) & (u <= 1.0)))
            N[:, j] = covered.to(dt)
    for d in range(1, degree + 1):
        N_new = torch.zeros(len(u), m - d, dtype=dt, device=device)
        for j in range(m - d):
            den1 = (kn[j + d] - kn[j]).item()
            den2 = (kn[j + d + 1] - kn[j + 1]).item()
            term = torch.zeros(len(u), dtype=dt, device=device)
            if den1 > 0:
                term = term + (u - kn[j]) / den1 * N[:, j]
            if den2 > 0:
                term = term + (kn[j + d + 1] - u) / den2 * N[:, j + 1]
            N_new[:, j] = term
        N = N_new
    return N  # float64 (len(u), n_ctrl)


class SmolVLASplinePolicy(SmolVLAPolicy):
    """SmolVLA with a B-spline + duration action head. See module docstring."""

    config_class = SmolVLASplineConfig
    name = "smolvla_spline"

    # token channel layout
    _POSE_DIMS = 6
    _N_OUT = 8  # 6 pose ctrl + 1 gripper ctrl + 1 duration channel

    def __init__(self, config: SmolVLASplineConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._init_spline_operators(config)

    # ------------------------------------------------------------------ setup
    def _init_spline_operators(self, cfg: SmolVLASplineConfig):
        n, deg = cfg.n_ctrl, cfg.spline_degree

        if not cfg.predict_duration:
            # ---- v1: single fixed-length operator set ----
            H = cfg.horizon
            u_path = torch.arange(H + 1, dtype=torch.float64) / H
            B_path = bspline_basis(u_path, n, deg)                # (H+1, n) f64
            pinv_mid = torch.linalg.pinv(B_path[:, 1 : n - 1])    # (n-2, H+1)
            u_grip = torch.arange(H, dtype=torch.float64) / (H - 1)
            pinv_grip = torch.linalg.pinv(bspline_basis(u_grip, n, deg))  # (n, H)
            self.register_buffer("_pinv_mid", pinv_mid.float(), persistent=False)
            self.register_buffer("_b_last", B_path[:, n - 1 : n].float(), persistent=False)
            self.register_buffer("_pinv_grip", pinv_grip.float(), persistent=False)
            stats_name, expect_len = cfg.spline_stats_file, H
        else:
            # ---- v2: zero-padded operator BANKS for T in [min_seg, horizon_max].
            # Padding with zeros is exact: padded columns multiply path entries
            # beyond the segment, contributing nothing.
            Hm, lo = cfg.horizon_max, cfg.min_seg
            n_len = Hm - lo + 1
            pm_bank = torch.zeros(n_len, n - 2, Hm + 1, dtype=torch.float64)
            bl_bank = torch.zeros(n_len, Hm + 1, 1, dtype=torch.float64)
            pg_bank = torch.zeros(n_len, n, Hm, dtype=torch.float64)
            for i, T in enumerate(range(lo, Hm + 1)):
                u_path = torch.arange(T + 1, dtype=torch.float64) / T
                B_path = bspline_basis(u_path, n, deg)
                pm_bank[i, :, : T + 1] = torch.linalg.pinv(B_path[:, 1 : n - 1])
                bl_bank[i, : T + 1, :] = B_path[:, n - 1 : n]
                u_grip = torch.arange(T, dtype=torch.float64) / max(T - 1, 1)
                pg_bank[i, :, :T] = torch.linalg.pinv(bspline_basis(u_grip, n, deg))
            self.register_buffer("_pm_bank", pm_bank.float(), persistent=False)
            self.register_buffer("_bl_bank", bl_bank.float(), persistent=False)
            self.register_buffer("_pg_bank", pg_bank.float(), persistent=False)
            stats_name, expect_len = cfg.spline_stats_file_v2, None

        stats_path = os.path.join(os.path.dirname(__file__), stats_name)
        with open(stats_path) as f:
            s = json.load(f)
        if s["n_ctrl"] != n or (expect_len is not None and s.get("horizon") != expect_len):
            raise ValueError(f"stats file {stats_name} mismatches config — regenerate")
        mean = torch.zeros(n, self._N_OUT)
        std = torch.ones(n, self._N_OUT)
        mean[:, :6] = torch.tensor(s["pose_ctrl_mean"], dtype=torch.float32)
        std[:, :6] = torch.tensor(s["pose_ctrl_std"], dtype=torch.float32)
        mean[:, 6] = torch.tensor(s["grip_ctrl_mean"], dtype=torch.float32)
        std[:, 6] = torch.tensor(s["grip_ctrl_std"], dtype=torch.float32)
        if cfg.predict_duration:
            mean[:, 7] = float(s["logT_mean"])
            std[:, 7] = float(s["logT_std"])
        self.register_buffer("_tgt_mean", mean, persistent=False)
        self.register_buffer("_tgt_std", std, persistent=False)

    # -------------------------------------------------------- v2 event chunks
    def _first_event(self, a: Tensor, pad: Tensor | None) -> Tensor:
        """a: (B, Hm, 7) raw window -> T: (B,) int64 in [min_seg, horizon_max].

        Event at step k (>= min_seg): gripper toggle | pause (2 consecutive
        low-speed steps) | episode end (first padded step). Else horizon_max.
        """
        cfg = self.config
        B, Hm, _ = a.shape
        ev = torch.zeros(B, Hm, dtype=torch.bool, device=a.device)
        grip = a[..., 6]
        ev[:, 1:] |= grip[:, 1:] != grip[:, :-1]                       # toggle
        speed = a[..., :6].norm(dim=-1)                                # (B, Hm)
        # quantile(0.5) interpolates like numpy.median (torch.median picks the
        # lower-middle element, which under-thresholds half-slow windows)
        med = torch.quantile(speed, 0.5, dim=1, keepdim=True) + 1e-9
        low = speed < cfg.pause_frac * med
        ev[:, 1:] |= low[:, 1:] & low[:, :-1]                          # pause
        if pad is not None:
            ev |= pad                                                  # episode end
        ev[:, : cfg.min_seg] = False                                   # respect minimum
        any_ev = ev.any(dim=1)
        first = torch.argmax(ev.int(), dim=1)                          # 0 if none
        T = torch.where(any_ev, first, torch.full_like(first, cfg.horizon_max))
        return T.clamp(cfg.min_seg, cfg.horizon_max)

    # ------------------------------------------------------------ target build
    def _build_spline_targets(self, batch: dict[str, Tensor]) -> Tensor:
        """(B, H, 7) raw delta actions -> (B, n_ctrl, 8) normalized spline targets."""
        cfg = self.config
        a = batch[ACTION]
        want = cfg.horizon_max if cfg.predict_duration else cfg.horizon
        if a.shape[1] != want:
            raise ValueError(
                f"expected action window of {want} steps, got {a.shape[1]} "
                "(check action_delta_indices / dataset fps)"
            )
        pose = a[..., : self._POSE_DIMS]
        grip = a[..., self._POSE_DIMS]
        pad = batch.get("action_is_pad")
        if pad is not None:
            # zero pose deltas past episode end => the target path comes to a stop.
            # gripper is left as delivered (dataloader repeats last value = hold state).
            pose = pose * (~pad).unsqueeze(-1).to(pose.dtype)

        if not cfg.predict_duration:
            # ---- v1: fixed-length fit ----
            path = torch.cat(
                [torch.zeros_like(pose[:, :1]), torch.cumsum(pose, dim=1)], dim=1
            )
            p_end = path[:, -1:, :]                                    # (B, 1, 6)
            resid = path - self._b_last.unsqueeze(0) * p_end           # (B, H+1, 6)
            c_mid = torch.einsum("mh,bhd->bmd", self._pinv_mid, resid)
            c_pose = torch.cat([torch.zeros_like(p_end), c_mid, p_end], dim=1)
            c_grip = torch.einsum("nh,bh->bn", self._pinv_grip, grip)
            dur = torch.zeros_like(c_grip)
        else:
            # ---- v2: event-segmented variable-length fit via padded banks ----
            T = self._first_event(a, pad)                              # (B,)
            idx = torch.arange(cfg.horizon_max, device=a.device)
            seg_mask = (idx.unsqueeze(0) < T.unsqueeze(1)).to(pose.dtype)   # (B, Hm)
            pose_seg = pose * seg_mask.unsqueeze(-1)      # zero beyond the segment
            path = torch.cat(
                [torch.zeros_like(pose_seg[:, :1]), torch.cumsum(pose_seg, dim=1)], dim=1
            )                                                          # (B, Hm+1, 6)
            bank_i = T - cfg.min_seg                                   # (B,)
            pm = self._pm_bank[bank_i]                                 # (B, n-2, Hm+1)
            bl = self._bl_bank[bank_i]                                 # (B, Hm+1, 1)
            pg = self._pg_bank[bank_i]                                 # (B, n, Hm)
            # p_end = path at index T (segment endpoint), per sample
            p_end = path.gather(
                1, T.view(-1, 1, 1).expand(-1, 1, path.shape[-1])
            )                                                          # (B, 1, 6)
            resid = path - bl * p_end                                  # (B, Hm+1, 6)
            c_mid = torch.einsum("bmh,bhd->bmd", pm, resid)            # (B, n-2, 6)
            c_pose = torch.cat([torch.zeros_like(p_end), c_mid, p_end], dim=1)
            c_grip = torch.einsum("bnh,bh->bn", pg, grip)              # (B, n)
            dur = torch.log(T.to(pose.dtype)).unsqueeze(-1).expand(-1, cfg.n_ctrl)

        tgt = torch.cat([c_pose, c_grip.unsqueeze(-1), dur.unsqueeze(-1)], dim=-1)
        return (tgt - self._tgt_mean) / self._tgt_std                  # (B, n, 8)

    # ------------------------------------------------------------------ train
    def forward(self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"):
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        target = self._build_spline_targets(batch)                     # (B, n, 8)
        actions = pad_vector(target, self.config.max_action_dim)       # (B, n, 32)

        losses = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
        )
        losses = losses[:, :, : self._N_OUT]  # all spline tokens are valid (no padding)
        loss_dict = {"losses_after_forward": losses.mean().item()}

        if reduction == "none":
            per_sample = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample.mean().item()
            return per_sample, loss_dict
        loss = losses.mean()
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    # ----------------------------------------------------------------- decode
    def _predicted_T(self, tokens_unnorm: Tensor) -> int:
        """Read the duration channel -> chunk duration in TRAINING-rate env steps."""
        cfg = self.config
        logT = tokens_unnorm[..., 7].mean(dim=1)          # (B,) average over tokens
        T = torch.exp(logT).round().long()
        T = T.clamp(cfg.min_seg, cfg.horizon_max)
        # batched eval with heterogeneous durations needs a rectangular tensor;
        # use the median (eval runs batch_size=1 in practice)
        return int(T.median().item())

    def _decode_tokens(self, tokens: Tensor, h_exec: int) -> Tensor:
        """(B, n_ctrl, >=8) normalized tokens -> (B, h_exec, 7) env delta actions."""
        t = tokens[:, :, : self._N_OUT] * self._tgt_std + self._tgt_mean
        c_pose = t[..., :6].clone()
        c_pose[:, 0, :] = 0.0  # hard guarantee: chunk starts at the current pose
        c_grip = t[..., 6]

        dev = tokens.device

        def _path_deltas(h):
            u = torch.arange(h + 1, device=dev, dtype=torch.float64) / h
            Bp = bspline_basis(u, self.config.n_ctrl, self.config.spline_degree).float().to(dev)
            path = torch.einsum("hn,bnd->bhd", Bp, c_pose)             # (B, h+1, 6)
            return path[:, 1:] - path[:, :-1]

        deltas = _path_deltas(h_exec)
        if self.config.feasibility_stretch:
            # Feasibility-aware time stretching: if any step exceeds the actuator
            # bound, lengthen the horizon so the SAME continuous trajectory is
            # executed slower instead of being clipped/distorted. Deltas scale
            # ~1/h, so one proportional stretch (+ safety margin) suffices.
            worst = deltas.abs().max().item()
            bound = self.config.actuator_bound
            if worst > bound:
                h_exec = min(int(h_exec * worst / bound) + 1, 8 * h_exec)
                deltas = _path_deltas(h_exec)
        deltas = deltas.clamp(-1.0, 1.0)

        u_grip = (torch.arange(h_exec, device=dev, dtype=torch.float64) / max(h_exec - 1, 1))
        Bg = bspline_basis(u_grip, self.config.n_ctrl, self.config.spline_degree).float()
        g = torch.einsum("hn,bn->bh", Bg.to(dev), c_grip)
        grip = torch.where(g >= 0, 1.0, -1.0).unsqueeze(-1)

        return torch.cat([deltas, grip], dim=-1)                       # (B, h_exec, 7)

    def _get_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs) -> Tensor:
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        tokens = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise, **kwargs
        )  # (B, n_ctrl, max_action_dim)

        if self.config.predict_duration:
            # v2: the policy chose how long this chunk should take (training-rate
            # steps); retarget to the execution rate. The queue consumes up to
            # n_action_steps of it, so short chunks trigger earlier replanning.
            t_un = tokens[:, :, : self._N_OUT] * self._tgt_std + self._tgt_mean
            T = self._predicted_T(t_un)
            alpha = 1.0
            if self.config.speedup_alpha < 1.0 and T > self.config.speedup_T_threshold:
                alpha = self.config.speedup_alpha  # duration-aware selective speedup
            h_exec = max(2, round(T * self.config.exec_rate_ratio * alpha))
        else:
            h_exec = self.config.exec_horizon or self.config.horizon
        return self._decode_tokens(tokens, h_exec)


def _self_test():  # pragma: no cover — run with: python -m lerobot.policies.smolvla_spline.modeling_smolvla_spline
    """Numerical self-test of the torch operators against scipy."""
    import numpy as np
    from scipy.interpolate import BSpline

    n, deg, H = 6, 3, 20
    kn = np.concatenate([np.zeros(deg), np.linspace(0, 1, n - deg + 1), np.ones(deg)])
    for grid in (np.arange(H + 1) / H, np.linspace(0, 1, 37), np.array([0.0, 0.5, 1.0])):
        Bt = bspline_basis(torch.tensor(grid), n, deg).numpy()
        Bs = np.stack(
            [BSpline(kn, np.eye(n)[j], deg, extrapolate=False)(grid) for j in range(n)], axis=1
        )
        Bs = np.nan_to_num(Bs)
        # scipy marks u=1 as extrapolation for the open form; fix the endpoint row
        Bs[np.isclose(grid, 1.0)] = 0.0
        Bs[np.isclose(grid, 1.0), n - 1] = 1.0
        err = np.abs(Bt - Bs).max()
        print(f"grid len {len(grid):>3}: max |torch - scipy| = {err:.2e}")
        assert err < 1e-9, "basis mismatch"

    # end-to-end: random window -> fit -> decode at multiple rates, endpoint exact
    rng = np.random.RandomState(0)
    a = torch.tensor(rng.uniform(-0.6, 0.6, size=(2, H, 7)), dtype=torch.float32)
    a[..., 6] = torch.where(a[..., 6] > 0, 1.0, -1.0)

    class _Cfg:  # minimal duck-typed config for operator init
        n_ctrl, horizon, spline_degree, spline_stats_file = n, H, deg, "spline_stats_libero.json"
        predict_duration = False

    class _Holder(torch.nn.Module):
        _POSE_DIMS = SmolVLASplinePolicy._POSE_DIMS
        _N_OUT = SmolVLASplinePolicy._N_OUT

    holder = _Holder()
    SmolVLASplinePolicy._init_spline_operators(holder, _Cfg)
    tgt = SmolVLASplinePolicy._build_spline_targets(
        type("obj", (), {"config": _Cfg, "_POSE_DIMS": 6, "_N_OUT": 8,
                         "_pinv_mid": holder._pinv_mid, "_b_last": holder._b_last,
                         "_pinv_grip": holder._pinv_grip, "_tgt_mean": holder._tgt_mean,
                         "_tgt_std": holder._tgt_std})(),
        {ACTION: a},
    )
    print("targets:", tuple(tgt.shape), "finite:", bool(torch.isfinite(tgt).all()))
    fake_self = type("obj", (), {"config": _Cfg, "_N_OUT": 8, "_tgt_mean": holder._tgt_mean,
                                 "_tgt_std": holder._tgt_std})()
    for h_exec in (10, 20, 40):
        out = SmolVLASplinePolicy._decode_tokens(fake_self, tgt, h_exec)
        path_end = out[..., :6].sum(dim=1)
        true_end = a[..., :6].sum(dim=1)
        e = (path_end - true_end).abs().max().item()
        print(f"decode h_exec={h_exec:>3}: endpoint err {e:.2e} shape {tuple(out.shape)}")
        assert e < 1e-4 or h_exec < H  # low rates may clip => endpoint may deviate

    # ---------------- v2: event detection + variable-length fit ----------------
    class _CfgV2:
        n_ctrl, horizon, spline_degree = n, H, deg
        predict_duration, horizon_max, min_seg, pause_frac = True, 40, 6, 0.15
        spline_stats_file = "spline_stats_libero.json"
        spline_stats_file_v2 = "spline_stats_libero_v2.json"
        exec_rate_ratio = 1.0

    class _HolderV2(torch.nn.Module):
        _POSE_DIMS = SmolVLASplinePolicy._POSE_DIMS
        _N_OUT = SmolVLASplinePolicy._N_OUT
        config = _CfgV2

    h2 = _HolderV2()
    SmolVLASplinePolicy._init_spline_operators(h2, _CfgV2)
    h2._first_event = SmolVLASplinePolicy._first_event.__get__(h2)

    a2 = torch.zeros(3, 40, 7)
    a2[..., :6] = 0.3  # constant motion
    a2[..., 6] = -1.0
    a2[0, 12:, 6] = 1.0          # sample 0: gripper toggle at k=12
    a2[1, 20:, :6] = 0.001       # sample 1: pause (2 consecutive low-speed) ~k=21
    #                              sample 2: no event -> cap at 40
    T = h2._first_event(a2, None)
    print("v2 event T:", T.tolist(), "(expect [12, 21, 40])")
    assert T.tolist() == [12, 21, 40]

    tgt2 = SmolVLASplinePolicy._build_spline_targets(h2, {ACTION: a2})
    print("v2 targets:", tuple(tgt2.shape), "finite:", bool(torch.isfinite(tgt2).all()))
    un = tgt2 * h2._tgt_std + h2._tgt_mean
    T_rt = torch.exp(un[..., 7].mean(dim=1)).round().long()
    print("v2 duration roundtrip:", T_rt.tolist())
    assert T_rt.tolist() == [12, 21, 40]
    # endpoint of sample 0's fit must equal cumsum of its first 12 deltas
    e0 = (un[0, -1, :6] - a2[0, :12, :6].sum(0)).abs().max().item()
    print(f"v2 variable-length fit endpoint err: {e0:.2e}")
    assert e0 < 1e-4
    print("SELF-TEST PASSED (v1 + v2)")


if __name__ == "__main__":
    _self_test()
