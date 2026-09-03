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

import os
import warnings

import torch
from torch import Tensor

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from ..smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector
from .configuration_smolvla_spline import SmolVLASplineConfig
from .event_targets import (
    bspline_basis,
    build_event_spline_targets,
    first_event_indices,
    make_event_operator_banks,
)
from .stats_contract import consume_legacy_normalization_missing_keys, resolve_spline_stats


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

        # Action-vector layout: where pose/grip/passthrough live in the env's
        # action dim. Token layout stays [pose(6) | grip(1) | dur(1) | pass(k)]
        # so eef7 targets are bit-identical to before the extension.
        layout = getattr(cfg, "action_layout", "eef7")
        if layout == "robocasa12":
            self._pose_lo, self._grip_idx, self._pass_dims = 5, 11, (0, 1, 2, 3, 4)
        else:  # eef7
            self._pose_lo, self._grip_idx, self._pass_dims = 0, 6, ()
        self._N_OUT = 8 + len(self._pass_dims)  # instance attr shadows class default

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
        else:
            # ---- v2: zero-padded operator BANKS for T in [min_seg, horizon_max].
            # Padding with zeros is exact: padded columns multiply path entries
            # beyond the segment, contributing nothing.
            Hm, lo = cfg.horizon_max, cfg.min_seg
            pm_bank, bl_bank, pg_bank, bp_bank = make_event_operator_banks(
                n_ctrl=n,
                degree=deg,
                min_seg=lo,
                horizon_max=Hm,
                fit_end_weight=cfg.fit_end_weight,
            )
            self.register_buffer("_pm_bank", pm_bank, persistent=False)
            self.register_buffer("_bl_bank", bl_bank, persistent=False)
            self.register_buffer("_pg_bank", pg_bank, persistent=False)
            self.register_buffer("_bp_bank", bp_bank, persistent=False)

        # New checkpoints carry the exact validated payload in config.json and
        # in the persistent buffers below. Fresh runs and legacy checkpoints
        # retain a one-time source-JSON fallback.
        self._spline_stats_were_embedded = getattr(cfg, "embedded_spline_stats", None) is not None
        s, self._spline_stats_origin = resolve_spline_stats(cfg, os.path.dirname(__file__))
        mean = torch.zeros(n, self._N_OUT)
        std = torch.ones(n, self._N_OUT)
        mean[:, :6] = torch.tensor(s["pose_ctrl_mean"], dtype=torch.float32)
        std[:, :6] = torch.tensor(s["pose_ctrl_std"], dtype=torch.float32)
        mean[:, 6] = torch.tensor(s["grip_ctrl_mean"], dtype=torch.float32)
        std[:, 6] = torch.tensor(s["grip_ctrl_std"], dtype=torch.float32)
        if cfg.predict_duration:
            mean[:, 7] = float(s["logT_mean"])
            std[:, 7] = float(s["logT_std"])
        if self._pass_dims:
            # passthrough ctrl stats optional in older JSONs (default 0/1)
            if "pass_ctrl_mean" in s:
                mean[:, 8:] = torch.tensor(s["pass_ctrl_mean"], dtype=torch.float32)
                std[:, 8:] = torch.tensor(s["pass_ctrl_std"], dtype=torch.float32)
        self.register_buffer("_tgt_mean", mean, persistent=True)
        self.register_buffer("_tgt_std", std, persistent=True)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Load both checkpoint generations without weakening strict loading.

        Pre-fix checkpoints contain neither target-stat buffer. For that exact
        legacy case, keep the validated JSON fallback constructed above and
        remove only those two missing keys. A partially present pair remains a
        hard error, as do all other missing/unexpected parameters.
        """

        mean_key = prefix + "_tgt_mean"
        std_key = prefix + "_tgt_std"
        if (mean_key in state_dict) != (std_key in state_dict):
            error_msgs.append(
                "incomplete spline normalization in checkpoint: _tgt_mean and _tgt_std "
                "must either both be present (new checkpoint) or both be absent (legacy checkpoint)"
            )
        if self._spline_stats_were_embedded and mean_key in state_dict and std_key in state_dict:
            # config.json and the tensor state are deliberately redundant. If
            # both are present they must describe one exact normalization;
            # accepting disagreement would make load behaviour order-dependent.
            for key, expected in (
                (mean_key, self._tgt_mean),
                (std_key, self._tgt_std),
            ):
                incoming = state_dict[key]
                if incoming.shape == expected.shape and not torch.equal(
                    incoming.to(device=expected.device, dtype=expected.dtype), expected
                ):
                    error_msgs.append(
                        f"checkpoint normalization mismatch for {key}: persistent buffer "
                        "does not equal config-embedded spline stats"
                    )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if consume_legacy_normalization_missing_keys(state_dict, prefix, missing_keys):
            warnings.warn(
                "Loading a legacy spline checkpoint without embedded normalization buffers; "
                f"using validated stats from {self._spline_stats_origin}. Re-save the policy "
                "once to make the checkpoint self-contained.",
                UserWarning,
                stacklevel=2,
            )

    # -------------------------------------------------------- v2 event chunks
    def _first_event(self, a: Tensor, pad: Tensor | None) -> Tensor:
        """a: (B, Hm, 7) raw window -> T: (B,) int64 in [min_seg, horizon_max].

        Event at step k (>= min_seg): gripper toggle | pause (2 consecutive
        low-speed steps) | episode end (first padded step). Else horizon_max.
        """
        cfg = self.config
        return first_event_indices(
            a,
            pad,
            pose_lo=getattr(self, "_pose_lo", 0),
            grip_idx=getattr(self, "_grip_idx", 6),
            min_seg=cfg.min_seg,
            horizon_max=cfg.horizon_max,
            pause_frac=cfg.pause_frac,
        )

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
        p_lo = getattr(self, "_pose_lo", 0)
        p_dims = getattr(self, "_pass_dims", ())
        pose = a[..., p_lo : p_lo + self._POSE_DIMS]
        grip = a[..., getattr(self, "_grip_idx", 6)]
        pas = a[..., list(p_dims)] if p_dims else None
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
            raw = build_event_spline_targets(
                a,
                pad,
                pm_bank=self._pm_bank,
                bl_bank=self._bl_bank,
                pg_bank=self._pg_bank,
                pose_lo=p_lo,
                grip_idx=getattr(self, "_grip_idx", 6),
                pass_dims=p_dims,
                min_seg=cfg.min_seg,
                horizon_max=cfg.horizon_max,
                pause_frac=cfg.pause_frac,
            )
            c_pose, c_grip, c_pas, T = raw
            self._last_T_batch = T                                     # decode-consistency hook
            T_lab = T.to(pose.dtype)
            if cfg.speed_aug:
                # Time allocation factorizes shape from timing: the slowed demo
                # has the SAME event-segmented spatial chunk (fit above is
                # untouched); only the duration label scales. This one line is
                # the representation-level hypothesis of the speed-aug study.
                T_lab = T_lab * self._speed_factors(batch, pose.dtype, pose.device)
            dur = torch.log(T_lab).unsqueeze(-1).expand(-1, cfg.n_ctrl)

        parts = [c_pose, c_grip.unsqueeze(-1), dur.unsqueeze(-1)]
        if pas is not None:
            # passthrough channels (base/mode): same low-order curve fit as the
            # gripper channel, one per dim, no sign snap at decode
            if not cfg.predict_duration:
                c_pas = torch.einsum("nh,bhk->bnk", self._pinv_grip, pas)
            else:
                # v2 passthrough targets are produced by the shared raw-target
                # builder above, alongside pose, gripper, and duration.
                assert c_pas is not None
            parts.append(c_pas)
        tgt = torch.cat(parts, dim=-1)
        tgt = (tgt - self._tgt_mean) / self._tgt_std                   # (B, n, N_OUT)
        if pas is not None:
            # passthrough channels are near-constant on fixed-base episodes
            # (std -> eps) but active on mobile ones -> rare 100-sigma outliers
            # that blow up the flow loss (observed: rc loss 33). Winsorize.
            tgt = torch.cat([tgt[..., :8], tgt[..., 8:].clamp(-5.0, 5.0)], dim=-1)
        return tgt

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

        # Profile retiming: reallocate time using the chunk's own speed profile.
        # Fast intervals (transport) can be compressed (profile_alpha < 1); the
        # slowest intervals (precision/contact micro-motion) can be dilated
        # (profile_slow_alpha > 1). Either alone, or both = bidirectional.
        final_u = None
        prof_a = self.config.profile_alpha if self.config.predict_duration else None
        prof_s = getattr(self.config, "profile_slow_alpha", None) if self.config.predict_duration else None
        if (prof_a is not None or prof_s is not None) and ease is None and h_exec >= 4 \
                and deltas.shape[0] == 1:
            u0 = torch.arange(h_exec + 1, device=dev, dtype=torch.float64) / h_exec
            sp = deltas[0, :, :6].norm(dim=-1).double()                # (h,) per-interval speed
            dt = torch.ones(h_exec, device=dev, dtype=torch.float64)
            if prof_a is not None:
                thr = self.config.profile_speed_threshold * sp.mean()
                if getattr(self.config, "profile_soft", False):
                    # smooth gate: dt ramps 1 -> alpha around the threshold, avoiding
                    # bang-bang velocity steps between adjacent intervals
                    w = torch.sigmoid((sp - thr) / (0.15 * sp.mean() + 1e-9))
                    dt = 1.0 - (1.0 - prof_a) * w
                else:
                    dt = torch.where(sp > thr, torch.tensor(prof_a, dtype=torch.float64, device=dev), dt)
            if prof_s is not None:
                thr_s = self.config.profile_slow_threshold * sp.mean()
                dt = torch.where(sp < thr_s, torch.tensor(prof_s, dtype=torch.float64, device=dev), dt)
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
            deltas = path[:, 1:] - path[:, :-1]
            if self.config.feasibility_stretch:
                # feasibility must hold AFTER retiming: compressed intervals can
                # re-violate the actuator bound (composition ordering — measured:
                # stack at half rate 69 vs stretch-alone 75 before this fix).
                # Dilate only the violating intervals' dt and rebuild once.
                sp2 = deltas[0, :, :6].norm(dim=-1).double()
                over = deltas[0].abs().max(dim=-1).values.double() / self.config.actuator_bound
                if float(over.max()) > 1.0:
                    dt2 = torch.ones(hp, device=dev, dtype=torch.float64)
                    dt2 = torch.maximum(dt2, over)         # stretch violators
                    cum2 = torch.cat([torch.zeros(1, device=dev, dtype=torch.float64),
                                      torch.cumsum(dt2, 0)])
                    hp2 = max(2, int(round(float(cum2[-1]))))
                    tq2 = torch.linspace(0, float(cum2[-1]), hp2 + 1, device=dev, dtype=torch.float64)
                    idx2 = torch.searchsorted(cum2, tq2, right=True).clamp(1, hp)
                    d0, d1 = cum2[idx2 - 1], cum2[idx2]
                    w2 = ((tq2 - d0) / (d1 - d0).clamp_min(1e-9)).clamp(0, 1)
                    u_new = u_new[idx2 - 1] + w2 * (u_new[idx2] - u_new[idx2 - 1])
                    u_new[0], u_new[-1] = 0.0, 1.0
                    Bp = bspline_basis(u_new, self.config.n_ctrl, self.config.spline_degree).float().to(dev)
                    path = torch.einsum("hn,bnd->bhd", Bp, c_pose)
                    deltas = path[:, 1:] - path[:, :-1]
                    hp = hp2
            deltas = deltas.clamp(-1.0, 1.0)
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

        p_dims = getattr(self, "_pass_dims", ())
        if not p_dims:
            return torch.cat([deltas, grip], dim=-1)                   # (B, h_exec, 7)

        # layout with passthrough channels (robocasa12): sample them at the same
        # u grid (continuous, no snap) and place every block at its env position
        c_pas = t[..., 8: 8 + len(p_dims)]                             # (B, n, k)
        pas = torch.einsum("hn,bnk->bhk", Bg.to(dev), c_pas)           # (B, h, k)
        D = len(p_dims) + self._POSE_DIMS + 1
        out = torch.zeros(deltas.shape[0], deltas.shape[1], D, device=dev, dtype=deltas.dtype)
        out[..., list(p_dims)] = pas
        out[..., self._pose_lo : self._pose_lo + self._POSE_DIMS] = deltas
        out[..., self._grip_idx] = grip.squeeze(-1)
        return out                                                     # (B, h_exec, D)

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
            elif self.config.slowdown_alpha > 1.0 and T <= self.config.slowdown_T_threshold:
                # mirror knob: short T = precision/contact motion -> more time
                alpha = self.config.slowdown_alpha
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
        profile_soft = False
        profile_slow_alpha = None
        profile_slow_threshold = 0.4
        slowdown_alpha = 1.0
        slowdown_T_threshold = 0

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
        embedded_spline_stats = {**_Cfg.embedded_spline_stats, "speed_aug": speed_aug}

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
        embedded_spline_stats = {**_CfgV2.embedded_spline_stats, "speed_aug": speed_aug}

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
        embedded_spline_stats = {
            **_CfgV2.embedded_spline_stats,
            "fit_end_weight": fit_end_weight,
        }

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

    # ---------------- slow-down decode: interval dilation + bidirectional ----------------
    # Build a token whose decoded chunk has a fast first half and slow second half,
    # then check: (a) slow-only dilation lengthens execution and dilates ONLY the
    # slow intervals; (b) bidirectional (compress fast + dilate slow) preserves the
    # exact endpoint; (c) chunk-level slowdown_alpha stretches h_exec.
    class _CfgSlow(_CfgV2):
        profile_slow_alpha = 1.5
        profile_slow_threshold = 0.5

    a_sl = torch.zeros(1, 40, 7)
    a_sl[:, :20, :6] = 0.30      # fast transport half
    a_sl[:, 20:, :6] = 0.03      # slow precision half
    a_sl[..., 6] = -1.0          # no toggle -> cap chunk
    h5 = _HolderV2()
    h5.config = _CfgSlow
    SmolVLASplinePolicy._init_spline_operators(h5, _CfgSlow)
    h5._first_event = SmolVLASplinePolicy._first_event.__get__(h5)
    tgt_sl = SmolVLASplinePolicy._build_spline_targets(h5, {ACTION: a_sl})
    fake_sl = type("obj", (), {"config": _CfgSlow, "_N_OUT": 8, "_tgt_mean": h5._tgt_mean,
                               "_tgt_std": h5._tgt_std, "_ease_this_chunk": False,
                               "_chain_c1": None})()
    fake_sl.config.chain_velocity = False
    out_sl = SmolVLASplinePolicy._decode_tokens(fake_sl, tgt_sl, 20)
    e_end = (out_sl[..., :6].sum(1) - a_sl[..., :6].sum(1)).abs().max().item()
    assert out_sl.shape[1] > 20, f"slow dilation must lengthen execution, got {out_sl.shape[1]}"
    print(f"slow-down decode: 20 -> {out_sl.shape[1]} steps, endpoint err {e_end:.2e}")
    assert e_end < 5e-2

    class _CfgBidir(_CfgSlow):   # bidirectional: compress fast AND dilate slow
        profile_alpha = 0.6
        profile_speed_threshold = 0.7
        # compressing this hot chunk (deltas ~0.67/dim at h=20) violates the
        # actuator bound; post-warp feasibility dilation must repair it
        feasibility_stretch = True

    fake_bd = type("obj", (), {"config": _CfgBidir, "_N_OUT": 8, "_tgt_mean": h5._tgt_mean,
                               "_tgt_std": h5._tgt_std, "_ease_this_chunk": False,
                               "_chain_c1": None})()
    out_bd = SmolVLASplinePolicy._decode_tokens(fake_bd, tgt_sl, 20)
    e_bd = (out_bd[..., :6].sum(1) - a_sl[..., :6].sum(1)).abs().max().item()
    worst_bd = out_bd[..., :6].abs().max().item()
    print(f"bidirectional decode (+feasibility): 20 -> {out_bd.shape[1]} steps, "
          f"endpoint err {e_bd:.2e}, worst |delta| {worst_bd:.3f}")
    assert e_bd < 5e-2 and worst_bd <= 1.0 + 1e-6

    # ---------------- robocasa12 action layout: round-trip ----------------
    class _CfgRC(_CfgV2):
        action_layout = "robocasa12"
        embedded_spline_stats = {
            **_CfgV2.embedded_spline_stats,
            "action_layout": action_layout,
            "pose_dims": [5, 6, 7, 8, 9, 10],
            "grip_idx": 11,
            "pass_dims": [0, 1, 2, 3, 4],
            "pass_ctrl_mean": [[0.0] * 5 for _ in range(n)],
            "pass_ctrl_std": [[1.0] * 5 for _ in range(n)],
        }

    hrc = _HolderV2()
    hrc.config = _CfgRC
    SmolVLASplinePolicy._init_spline_operators(hrc, _CfgRC)
    hrc._first_event = SmolVLASplinePolicy._first_event.__get__(hrc)
    a_rc = torch.zeros(2, 40, 12)
    a_rc[..., 0:4] = 0.05          # base motion (passthrough)
    a_rc[..., 4] = 1.0             # control mode (passthrough, constant)
    a_rc[..., 5:11] = 0.2          # EE pose deltas
    a_rc[..., 11] = -1.0
    a_rc[0, 12:, 11] = 1.0         # toggle at 12 for sample 0
    tgt_rc = SmolVLASplinePolicy._build_spline_targets(hrc, {ACTION: a_rc})
    assert tuple(tgt_rc.shape) == (2, n, 13), f"robocasa targets shape {tuple(tgt_rc.shape)}"
    fake_rc = type("obj", (), {"config": _CfgRC, "_N_OUT": 13, "_tgt_mean": hrc._tgt_mean,
                               "_tgt_std": hrc._tgt_std, "_pose_lo": 5, "_grip_idx": 11,
                               "_pass_dims": (0, 1, 2, 3, 4), "_POSE_DIMS": 6,
                               "_ease_this_chunk": False, "_chain_c1": None})()
    out_rc = SmolVLASplinePolicy._decode_tokens(fake_rc, tgt_rc, 12)
    assert tuple(out_rc.shape) == (2, 12, 12), f"robocasa decode shape {tuple(out_rc.shape)}"
    e_pose = (out_rc[0, :, 5:11].sum(0) - a_rc[0, :12, 5:11].sum(0)).abs().max().item()
    e_pass = (out_rc[..., 4] - 1.0).abs().max().item()
    ok_grip = bool(((out_rc[..., 11] == 1.0) | (out_rc[..., 11] == -1.0)).all())
    print(f"robocasa12 layout: pose endpoint err {e_pose:.2e}, mode-channel err {e_pass:.2e}, "
          f"grip snapped: {ok_grip}")
    assert e_pose < 1e-3 and e_pass < 5e-2 and ok_grip

    print("SELF-TEST PASSED (v1 + v2 + chaining + slow-down + robocasa12)")


if __name__ == "__main__":
    _self_test()
