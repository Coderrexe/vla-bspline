# Copyright 2026 — VLA B-spline project.
"""SmolVLA-Interp: the STRONG waypoint baseline for rate-mismatch experiments.

Takes the ordinary waypoint SmolVLA (unchanged weights — evaluated straight from
a trained `smolvla` checkpoint via a config-type swap) and adds the decode-time
rate adaptation any practitioner would implement: linearly resample the chunk's
cumulative commanded path to the execution rate, then emit consecutive diffs.
Optionally with the same feasibility stretch we give the spline policy.

CORRECTNESS NOTE (the trap): SmolVLA normalizes actions per-step (MEAN_STD), and
the postprocessor adds the per-step mean back on every emitted action. Emitting
H' != chunk_size steps would bias the total path by (chunk_size - H')·mean.
Therefore we load the checkpoint's own unnormalizer stats, resample in RAW
action space, and re-normalize before returning — the postprocessor then
reconstructs exact raw deltas.
"""

import os
from dataclasses import dataclass

import torch
from torch import Tensor

from lerobot.configs import PreTrainedConfig

from ..smolvla.configuration_smolvla import SmolVLAConfig
from ..smolvla.modeling_smolvla import SmolVLAPolicy


@PreTrainedConfig.register_subclass("smolvla_interp")
@dataclass
class SmolVLAInterpConfig(SmolVLAConfig):
    exec_horizon: int | None = None      # resampled step count; None -> chunk_size
    feasibility_stretch: bool = False
    actuator_bound: float = 1.0


class SmolVLAInterpPolicy(SmolVLAPolicy):
    config_class = SmolVLAInterpConfig
    name = "smolvla_interp"

    def _action_stats(self, device):
        """Lazy-load per-step action mean/std from the checkpoint's unnormalizer."""
        if getattr(self, "_act_mu", None) is not None:
            return self._act_mu.to(device), self._act_sigma.to(device)
        from safetensors.torch import load_file

        path = getattr(self.config, "pretrained_path", None)
        assert path, "smolvla_interp needs config.pretrained_path to locate stats"
        cand = [f for f in os.listdir(path) if "unnormalizer" in f and f.endswith(".safetensors")]
        assert cand, f"no unnormalizer stats found in {path}"
        st = load_file(os.path.join(path, cand[0]))
        mu_k = [k for k in st if "action" in k and k.endswith("mean")]
        sd_k = [k for k in st if "action" in k and k.endswith("std")]
        assert mu_k and sd_k, f"action mean/std not found; keys: {list(st)[:8]}"
        self._act_mu = st[mu_k[0]].float()
        self._act_sigma = st[sd_k[0]].float().clamp_min(1e-6)
        return self._act_mu.to(device), self._act_sigma.to(device)

    def _get_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs) -> Tensor:
        chunk_n = super()._get_action_chunk(batch, noise, **kwargs)   # (B, chunk, 7) NORMALIZED
        h_exec = self.config.exec_horizon or chunk_n.shape[1]
        if h_exec == chunk_n.shape[1] and not self.config.feasibility_stretch:
            return chunk_n

        dev = chunk_n.device
        mu, sigma = self._action_stats(dev)
        raw = chunk_n * sigma + mu                                    # raw deltas
        pose, grip = raw[..., :6], raw[..., 6]

        B, H, _ = pose.shape
        path = torch.cat([torch.zeros_like(pose[:, :1]), torch.cumsum(pose, dim=1)], dim=1)  # (B,H+1,6)

        def resample(h):
            # linear interpolation of the cumulative path onto h+1 points
            u = torch.linspace(0, H, h + 1, device=dev)
            i0 = u.floor().long().clamp(max=H - 1)
            frac = (u - i0.float()).view(1, -1, 1)
            p = path[:, i0] * (1 - frac) + path[:, i0 + 1] * frac
            d = p[:, 1:] - p[:, :-1]
            # gripper: nearest original step, sign snap
            gi = torch.linspace(0, H - 1, h, device=dev).round().long()
            g = torch.where(grip[:, gi] >= 0, 1.0, -1.0)
            return d, g

        deltas, g = resample(h_exec)
        if self.config.feasibility_stretch:
            worst = deltas.abs().max().item()
            if worst > self.config.actuator_bound:
                h_exec = min(int(h_exec * worst / self.config.actuator_bound) + 1, 8 * h_exec)
                deltas, g = resample(h_exec)
        deltas = deltas.clamp(-1.0, 1.0)

        raw_out = torch.cat([deltas, g.unsqueeze(-1)], dim=-1)        # (B, h_exec, 7)
        return (raw_out - mu) / sigma                                 # back to normalized
