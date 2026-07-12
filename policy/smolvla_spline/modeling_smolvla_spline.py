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

    def reset(self):
        super().reset()
        self._prev_step_vel = None  # start each episode at rest (chain_velocity)
        self.n_chunks_generated = 0  # policy invocations this episode (analysis hook)

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
            bp_bank = torch.zeros(n_len, Hm + 1, n, dtype=torch.float64)  # full basis (decode-consistency)
            for i, T in enumerate(range(lo, Hm + 1)):
                u_path = torch.arange(T + 1, dtype=torch.float64) / T
                B_path = bspline_basis(u_path, n, deg)
                if cfg.fit_end_weight is not None:
                    # contact-weighted LSQ: weight ramps 1 -> W over the last 25%
                    # of the (event-terminated) chunk; rank-safe via sqrt-W pinv.
                    W = float(cfg.fit_end_weight)
                    sw = torch.sqrt(1.0 + (W - 1.0) * ((u_path - 0.75) / 0.25).clamp(0, 1))
                    pm = torch.linalg.pinv(sw.unsqueeze(1) * B_path[:, 1 : n - 1]) * sw.unsqueeze(0)
                else:
                    pm = torch.linalg.pinv(B_path[:, 1 : n - 1])
                pm_bank[i, :, : T + 1] = pm
                bl_bank[i, : T + 1, :] = B_path[:, n - 1 : n]
                bp_bank[i, : T + 1, :] = B_path
                u_grip = torch.arange(T, dtype=torch.float64) / max(T - 1, 1)
                pg_bank[i, :, :T] = torch.linalg.pinv(bspline_basis(u_grip, n, deg))
            self.register_buffer("_pm_bank", pm_bank.float(), persistent=False)
            self.register_buffer("_bl_bank", bl_bank.float(), persistent=False)
            self.register_buffer("_pg_bank", pg_bank.float(), persistent=False)
            self.register_buffer("_bp_bank", bp_bank.float(), persistent=False)
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

    # ------------------------------------------- speed-heterogeneous demos
    def _speed_factors(self, batch: dict[str, Tensor], dtype, device) -> Tensor:
        """Per-sample synthetic slowdown s = speed_aug[episode_index % len]."""
        ep = batch.get("episode_index")
        if ep is None:
            raise ValueError("speed_aug requires 'episode_index' in the batch")
        s_vals = torch.tensor(self.config.speed_aug, dtype=dtype, device=device)
        return s_vals[ep.long().view(-1) % len(s_vals)]                # (B,)

    @staticmethod
    def _lerp_path(path: Tensor, x: Tensor) -> Tensor:
        """Linear interp of cumulative path (B, K, D) at fractional indices x (B, M)."""
        K = path.shape[1]
        x0 = x.floor().long().clamp(0, K - 1)
        x1 = (x0 + 1).clamp(max=K - 1)
        w = (x - x0.to(x.dtype)).unsqueeze(-1)
        D = path.shape[-1]
        g0 = path.gather(1, x0.unsqueeze(-1).expand(-1, -1, D))
        g1 = path.gather(1, x1.unsqueeze(-1).expand(-1, -1, D))
        return g0 * (1 - w) + g1 * w                                   # (B, M, D)

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
            H = cfg.horizon
            if cfg.speed_aug:
                # Synthetic slow demo: the chunk is the first `horizon` steps of
                # the path executed s x slower = path prefix up to raw index H/s,
                # sampled at H+1 synthetic ticks. Speed contaminates the shape
                # target here — that is the phenomenon under study.
                s = self._speed_factors(batch, path.dtype, path.device)  # (B,)
                j = torch.arange(H + 1, dtype=path.dtype, device=path.device)
                path = self._lerp_path(path, j.unsqueeze(0) / s.unsqueeze(1))
                gx = (
                    (torch.arange(H, dtype=path.dtype, device=path.device).unsqueeze(0)
                     / s.unsqueeze(1)).round().long().clamp(max=H - 1)
                )                                                       # (B, H)
                grip = grip.gather(1, gx)
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
            self._last_T_batch = T                                     # decode-consistency hook
            T_lab = T.to(pose.dtype)
            if cfg.speed_aug:
                # Time allocation factorizes shape from timing: the slowed demo
                # has the SAME event-segmented spatial chunk (fit above is
                # untouched); only the duration label scales. This one line is
                # the representation-level hypothesis of the speed-aug study.
                T_lab = T_lab * self._speed_factors(batch, pose.dtype, pose.device)
            dur = torch.log(T_lab).unsqueeze(-1).expand(-1, cfg.n_ctrl)

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

        dc_w = self.config.decode_consistency_weight
        if dc_w > 0:
            losses, flow_err, t_used = self._flow_forward_with_error(
                images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
            )
        else:
            losses = self.model.forward(
                images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
            )
        losses = losses[:, :, : self._N_OUT]  # all spline tokens are valid (no padding)
        loss_dict = {"losses_after_forward": losses.mean().item()}

        dc_aux = None
        if dc_w > 0:
            # x0-space PATH error: x0_hat - x0 = t*(u_t - v_t); decode is linear,
            # so project the signed flow error through the (std-scaled) basis.
            e_pose = flow_err[:, :, :6] * self._tgt_std[:, :6]          # (B, n, 6) raw scale
            bp = self._bp_bank[self._last_T_batch - self.config.min_seg]  # (B, Hm+1, n)
            path_err = torch.einsum("bhn,bnd->bhd", bp, e_pose)         # zero rows beyond T+1
            per = (t_used[:, None, None] ** 2 * path_err.pow(2)).sum(dim=(1, 2))
            dc_aux = (per / ((self._last_T_batch + 1).to(per.dtype) * 6)).mean()
            loss_dict["dc_aux"] = dc_aux.item()

        if reduction == "none":
            per_sample = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample.mean().item()
            return per_sample, loss_dict
        loss = losses.mean()
        if dc_aux is not None:
            loss = loss + dc_w * dc_aux
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    def _flow_forward_with_error(self, images, img_masks, lang_tokens, lang_masks,
                                 state, actions, noise=None, time=None):
        """Replicates VLAFlowMatching.forward but also returns the SIGNED flow
        error (v_t - u_t) and the sampled time — needed by the decode-consistency
        auxiliary loss (the stock forward only returns squared errors)."""
        import torch.nn.functional as F

        from ..smolvla.modeling_smolvla import make_att_2d_masks

        m = self.model
        if noise is None:
            noise = m.sample_noise(actions.shape, actions.device)
        if time is None:
            time = m.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_embs, prefix_pad_masks, prefix_att_masks = m.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = m.embed_suffix(x_t, time)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = m.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
        v_t = m.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses, v_t - u_t, time

    # ----------------------------------------------------------------- decode
    def _predicted_T_batch(self, tokens_unnorm: Tensor) -> Tensor:
        """Per-sample predicted chunk duration (TRAINING-rate env steps), (B,)."""
        cfg = self.config
        logT = tokens_unnorm[..., 7].mean(dim=1)          # (B,) average over tokens
        T = torch.exp(logT).round().long().clamp(cfg.min_seg, cfg.horizon_max)
        thr = getattr(cfg, "duration_snap_threshold", None)
        if thr is not None:
            # bimodal-target fix: cap-chunk predictions smear toward ~30 (measured
            # bias -9.5), making transports execute ~33% faster than their fitted
            # shape intended; snap them back to the cap.
            T = torch.where(T >= thr, torch.full_like(T, cfg.horizon_max), T)
        return T

    def _predicted_T(self, tokens_unnorm: Tensor) -> int:
        """Scalar duration for decode: batched eval with heterogeneous durations
        needs a rectangular tensor; use the median (eval runs batch_size=1)."""
        return int(self._predicted_T_batch(tokens_unnorm).median().item())

    def _decode_tokens(self, tokens: Tensor, h_exec: int) -> Tensor:
        """(B, n_ctrl, >=8) normalized tokens -> (B, h_exec, 7) env delta actions."""
        t = tokens[:, :, : self._N_OUT] * self._tgt_std + self._tgt_mean
        c_pose = t[..., :6].clone()
        c_pose[:, 0, :] = 0.0  # hard guarantee: chunk starts at the current pose
        if self.config.chain_velocity:
            # velocity continuity across replans: clamped cubic with interior
            # span 1/3 has s'(0) = 9*(c1 - c0) in u-units; per-step v0 = 9*c1/h.
            # Set c1 so v0 equals the previous chunk's replan-point velocity.
            v_prev = getattr(self, "_prev_step_vel", None)
            if v_prev is not None:
                c_pose[:, 1, :] = v_prev.to(c_pose.device) * h_exec / 9.0
        c_grip = t[..., 6]

        dev = tokens.device

        ease = self.config.ease_out if getattr(self, "_ease_this_chunk", False) else None

        def _u_grid(h):
            u = torch.arange(h + 1, device=dev, dtype=torch.float64) / h
            if ease is None or h < 4:
                return u
            # re-time the last 3 path steps over 3+ease steps, cosine-spaced:
            # same path, exact endpoint, velocity -> ~0 at the event boundary.
            m = 3
            k = torch.arange(m + ease + 1, device=dev, dtype=torch.float64) / (m + ease)
            tail = (h - m) / h + (m / h) * torch.sin(k * torch.pi / 2)
            return torch.cat([u[: h - m], tail])

        def _path_deltas(h):
            u = _u_grid(h)
            Bp = bspline_basis(u, self.config.n_ctrl, self.config.spline_degree).float().to(dev)
            path = torch.einsum("hn,bnd->bhd", Bp, c_pose)             # (B, len(u), 6)
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

        # Profile-selective retiming: compress time only where the chunk's own
        # predicted speed is high; slow (precision/contact) intervals keep dt=1.
        final_u = None
        prof_a = self.config.profile_alpha if self.config.predict_duration else None
        if prof_a is not None and ease is None and h_exec >= 4 and deltas.shape[0] == 1:
            u0 = torch.arange(h_exec + 1, device=dev, dtype=torch.float64) / h_exec
            sp = deltas[0, :, :6].norm(dim=-1).double()                # (h,) per-interval speed
            thr = self.config.profile_speed_threshold * sp.mean()
            dt = torch.where(sp > thr, torch.tensor(prof_a, dtype=torch.float64, device=dev),
                             torch.tensor(1.0, dtype=torch.float64, device=dev))
            cum = torch.cat([torch.zeros(1, device=dev, dtype=torch.float64), torch.cumsum(dt, 0)])
            hp = max(2, int(round(float(cum[-1]))))
            tq = torch.linspace(0, float(cum[-1]), hp + 1, device=dev, dtype=torch.float64)
            # invert the monotone piecewise-linear time map t(u): u_new = u(tq)
            idx = torch.searchsorted(cum, tq, right=True).clamp(1, h_exec)
            c0, c1 = cum[idx - 1], cum[idx]
            w = ((tq - c0) / (c1 - c0).clamp_min(1e-9)).clamp(0, 1)
            u_new = u0[idx - 1] + w * (u0[idx] - u0[idx - 1])
            u_new[0], u_new[-1] = 0.0, 1.0                             # exact endpoints
            Bp = bspline_basis(u_new, self.config.n_ctrl, self.config.spline_degree).float().to(dev)
            path = torch.einsum("hn,bnd->bhd", Bp, c_pose)
            deltas = (path[:, 1:] - path[:, :-1]).clamp(-1.0, 1.0)
            final_u = u_new
            h_exec = hp

        if final_u is not None:
            u_grip = (final_u[:-1] + final_u[1:]) / 2   # interval midpoints, len = n_deltas
        elif ease is not None and h_exec >= 4:
            ug_full = _u_grid(h_exec)
            u_grip = (ug_full[:-1] + ug_full[1:]) / 2   # interval midpoints, len = n_deltas
        else:
            u_grip = (torch.arange(h_exec, device=dev, dtype=torch.float64) / max(h_exec - 1, 1))
        Bg = bspline_basis(u_grip, self.config.n_ctrl, self.config.spline_degree).float()
        g = torch.einsum("hn,bn->bh", Bg.to(dev), c_grip)
        grip = torch.where(g >= 0, 1.0, -1.0).unsqueeze(-1)

        if self.config.chain_velocity:
            # per-step velocity at the upcoming replan point u_c = consumed/h,
            # via central finite difference of the basis (float64, exact enough)
            u_c = min(self.config.n_action_steps, h_exec) / h_exec
            du = 1e-5
            u_pair = torch.tensor([max(u_c - du, 0.0), min(u_c + du, 1.0)],
                                  dtype=torch.float64, device=dev)
            Bp2 = bspline_basis(u_pair, self.config.n_ctrl, self.config.spline_degree).float().to(dev)
            pts = torch.einsum("hn,bnd->bhd", Bp2, c_pose)             # (B, 2, 6)
            s_prime = (pts[:, 1] - pts[:, 0]) / float(u_pair[1] - u_pair[0])
            self._prev_step_vel = (s_prime / h_exec).detach()          # (B, 6) per-step

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
            self.last_predicted_T_batch = self._predicted_T_batch(t_un)  # (B,) analysis hook
            T = self._predicted_T(t_un)
            self.last_predicted_T = T  # scalar hook for recording tools
            alpha = 1.0
            if self.config.speedup_alpha < 1.0 and T > self.config.speedup_T_threshold:
                alpha = self.config.speedup_alpha  # duration-aware selective speedup
            h_exec = max(2, round(T * self.config.exec_rate_ratio * alpha))
            # soft landing only into predicted EVENTS (contact), not cap chunks
            self._ease_this_chunk = (
                self.config.ease_out is not None and T < self.config.horizon_max
            )
        else:
            h_exec = self.config.exec_horizon or self.config.horizon
            self._ease_this_chunk = False
        chunk = self._decode_tokens(tokens, h_exec)
        if self.config.predict_duration and self.config.replan_frac is not None:
            # Self-paced replanning: execute replan_frac of the chunk's own
            # predicted duration, then let the queue drain -> replan. The
            # duration head, not a fixed cadence, schedules the replans.
            n_exec = max(2, min(int(round(chunk.shape[1] * self.config.replan_frac)), chunk.shape[1]))
            chunk = chunk[:, :n_exec]
        elif self.config.predict_duration and self.config.replan_margin is not None:
            # Absolute-margin variant: replan a fixed few steps BEFORE the
            # predicted chunk end, so predicted events (gripper toggles) are
            # never placed at the chunk boundary where duration error clips them.
            n_exec = max(2, chunk.shape[1] - self.config.replan_margin)
            chunk = chunk[:, :n_exec]
        self.n_chunks_generated = getattr(self, "n_chunks_generated", 0) + 1
        return chunk


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

    class _StubBase:  # complete decode-flag defaults; extend for variants
        predict_duration = False
        chain_velocity = False
        feasibility_stretch = False
        actuator_bound = 1.0
        speedup_alpha = 1.0
        speedup_T_threshold = 0
        exec_rate_ratio = 1.0
        exec_horizon = None
        n_action_steps = 10
        replan_frac = None
        replan_margin = None
        speed_aug = None
        fit_end_weight = None
        ease_out = None
        decode_consistency_weight = 0.0
        profile_alpha = None
        profile_speed_threshold = 0.7

    class _Cfg(_StubBase):  # minimal duck-typed config for operator init
        n_ctrl, horizon, spline_degree, spline_stats_file = n, H, deg, "spline_stats_libero.json"

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
    class _CfgV2(_StubBase):
        n_ctrl, horizon, spline_degree = n, H, deg
        predict_duration, horizon_max, min_seg, pause_frac = True, 40, 6, 0.15
        spline_stats_file = "spline_stats_libero.json"
        spline_stats_file_v2 = "spline_stats_libero_v2.json"

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
    # ---------------- speed-aug: shape/timing factorization math ----------------
    # v1 (B arm): with constant deltas d, the s-slowed chunk covers d*H/s of path
    # -> pinned endpoint control point must be exactly p_end/s.
    class _CfgSaug(_Cfg):
        speed_aug = [1.0, 2.0]

    a3 = torch.zeros(2, H, 7)
    a3[..., :6] = 0.25
    a3[..., 6] = 1.0
    hs = _Holder()
    SmolVLASplinePolicy._init_spline_operators(hs, _CfgSaug)
    obj = type("obj", (), {"config": _CfgSaug, "_POSE_DIMS": 6, "_N_OUT": 8,
                           "_pinv_mid": hs._pinv_mid, "_b_last": hs._b_last,
                           "_pinv_grip": hs._pinv_grip, "_tgt_mean": hs._tgt_mean,
                           "_tgt_std": hs._tgt_std,
                           "_speed_factors": SmolVLASplinePolicy._speed_factors,
                           "_lerp_path": staticmethod(SmolVLASplinePolicy._lerp_path)})()
    obj._speed_factors = SmolVLASplinePolicy._speed_factors.__get__(obj)
    tgt3 = SmolVLASplinePolicy._build_spline_targets(
        obj, {ACTION: a3, "episode_index": torch.tensor([0, 1])})
    un3 = tgt3 * hs._tgt_std + hs._tgt_mean
    end_s1 = un3[0, -1, :6]                       # s=1: full-chunk displacement
    end_s2 = un3[1, -1, :6]                       # s=2: half of it
    e = (end_s2 - end_s1 / 2).abs().max().item()
    print(f"speed-aug v1: s=2 endpoint == s=1 endpoint / 2, err {e:.2e}")
    assert e < 1e-5

    # v2 (C arm): shape control points must be IDENTICAL across s (factorization);
    # only the duration channel scales: logT' = logT + log s.
    class _CfgV2S(_CfgV2):
        speed_aug = [1.0, 2.0]

    h3 = _HolderV2()
    h3.config = _CfgV2S
    SmolVLASplinePolicy._init_spline_operators(h3, _CfgV2S)
    h3._first_event = SmolVLASplinePolicy._first_event.__get__(h3)
    h3._speed_factors = SmolVLASplinePolicy._speed_factors.__get__(h3)
    a4 = torch.zeros(2, 40, 7)
    a4[..., :6] = 0.3
    a4[..., 6] = -1.0
    a4[:, 12:, 6] = 1.0                          # same toggle at k=12 in both samples
    tgt4 = SmolVLASplinePolicy._build_spline_targets(
        h3, {ACTION: a4, "episode_index": torch.tensor([0, 1])})
    un4 = tgt4 * h3._tgt_std + h3._tgt_mean
    shape_diff = (un4[0, :, :7] - un4[1, :, :7]).abs().max().item()
    T0 = float(torch.exp(un4[0, :, 7].mean()))
    T1 = float(torch.exp(un4[1, :, 7].mean()))
    print(f"speed-aug v2: shape ctrl-pt diff across s = {shape_diff:.2e} (expect 0); "
          f"durations {T0:.1f}/{T1:.1f} (expect 12/24)")
    assert shape_diff < 1e-6 and abs(T0 - 12) < 0.1 and abs(T1 - 24) < 0.1

    # ---------------- contact-weighted fit: WLS exactness vs numpy ----------------
    class _CfgV2W(_CfgV2):
        fit_end_weight = 9.0

    h4 = _HolderV2()
    h4.config = _CfgV2W
    SmolVLASplinePolicy._init_spline_operators(h4, _CfgV2W)
    h4._first_event = SmolVLASplinePolicy._first_event.__get__(h4)
    tgt5 = SmolVLASplinePolicy._build_spline_targets(h4, {ACTION: a2})
    un5 = tgt5 * h4._tgt_std + h4._tgt_mean
    # numpy WLS reference on sample 0 (T=12): pinned both ends, weight ramp
    T0w = 12
    seg = a2[0, :T0w, :6].numpy()
    pathw = np.concatenate([np.zeros((1, 6)), np.cumsum(seg, axis=0)], axis=0)
    uw = np.arange(T0w + 1) / T0w
    from scipy.interpolate import BSpline as _BS
    knw = np.concatenate([np.zeros(deg), np.linspace(0, 1, n - deg + 1), np.ones(deg)])
    Bw = np.stack([_BS(knw, np.eye(n)[j], deg, extrapolate=False)(uw) for j in range(n)], axis=1)
    Bw = np.nan_to_num(Bw)
    sww = np.sqrt(1 + 8.0 * np.clip((uw - 0.75) / 0.25, 0, 1))
    p_endw = pathw[-1:, :]
    residw = pathw - Bw[:, n - 1 : n] @ p_endw
    c_midw = (np.linalg.pinv(sww[:, None] * Bw[:, 1 : n - 1]) * sww[None, :]) @ residw
    err_w = np.abs(un5[0, 1 : n - 1, :6].numpy() - c_midw).max()
    print(f"contact-weighted fit: torch vs numpy WLS ctrl-pt err {err_w:.2e}; "
          f"endpoint still exact: {float((un5[0, -1, :6] - torch.tensor(p_endw[0], dtype=torch.float32)).abs().max()):.2e}")
    assert err_w < 1e-4

    # ------------- velocity-chaining math: s'(0) coefficient + continuity -------------
    u_pair = torch.tensor([0.0, 1e-6], dtype=torch.float64)
    Bp = bspline_basis(u_pair, n, deg)
    dB0 = (Bp[1] - Bp[0]) / 1e-6                     # basis derivative at u=0
    # s'(0) = sum_i c_i B_i'(0); with c = e1 (only c1=1): expect +9
    coeff = float(dB0[1])
    print(f"velocity coefficient s'(0) per c1: {coeff:.4f} (expect 9)")
    assert abs(coeff - 9.0) < 1e-3

    print("SELF-TEST PASSED (v1 + v2 + chaining math)")


if __name__ == "__main__":
    _self_test()
