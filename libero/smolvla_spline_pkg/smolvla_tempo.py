# Copyright 2026 — VLA B-spline project.
"""SmolVLA-Tempo: the SPEED-AS-INPUT baseline (TempoVLA-style) for the
head-to-head against our speed-as-output duration head.

Mechanism (mirrors TempoVLA, arXiv 2606.06491, on our stack):
  * VSTA-style retiming augmentation: each training sample's action window is
    re-timed to a random speed v in `speeds` (v>1 = faster: same cumulative
    path resampled at v x coarser steps -> larger deltas; v<1 = slower).
  * Scalar speed conditioning: v is written into the last (padded) slot of the
    state vector — SmolVLA pads state to max_state_dim=32 and LIBERO uses 8,
    so the slot is otherwise always zero. (Equivalent in spirit to TempoVLA's
    scalar-embedding conditioning variant.)
  * Inference: config.exec_speed commands the speed; the policy was trained to
    modulate action magnitudes accordingly. Chunk semantics unchanged.

Shares the raw-space resampling + dataset-stats normalization discipline of
smolvla_speedaug (affine normalization does not commute through resampling).
NOTE: v>1 targets can exceed the [-1,1] actuator range exactly as in TempoVLA
(their "controller tracking mismatch" failure mode) — left unclipped on purpose.
"""

import json
import os
from dataclasses import dataclass, field

import torch
from torch import Tensor

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import ACTION

from ..smolvla.configuration_smolvla import SmolVLAConfig
from ..smolvla.modeling_smolvla import SmolVLAPolicy


@PreTrainedConfig.register_subclass("smolvla_tempo")
@dataclass
class SmolVLATempoConfig(SmolVLAConfig):
    speeds: list[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 2.0])
    exec_speed: float = 1.0                     # commanded speed at inference
    action_stats_file: str = "waypoint_action_stats_libero.json"

    def __post_init__(self):
        super().__post_init__()
        if min(self.speeds) <= 0:
            raise ValueError("speeds must be positive")

    @property
    def action_delta_indices(self) -> list:
        # fetch enough raw steps to synthesize the fastest retiming
        import math
        return list(range(int(math.ceil(self.chunk_size * max(self.speeds)))))


class SmolVLATempoPolicy(SmolVLAPolicy):
    config_class = SmolVLATempoConfig
    name = "smolvla_tempo"

    def _dataset_action_stats(self, device, dtype):
        if getattr(self, "_ta_mu", None) is None:
            path = os.path.join(os.path.dirname(__file__), self.config.action_stats_file)
            with open(path) as f:
                s = json.load(f)
            self._ta_mu = torch.tensor(s["mean"], dtype=torch.float32)
            self._ta_sigma = torch.tensor(s["std"], dtype=torch.float32).clamp_min(1e-6)
        return self._ta_mu.to(device=device, dtype=dtype), self._ta_sigma.to(device=device, dtype=dtype)

    def prepare_state(self, batch):
        state = super().prepare_state(batch)
        # speed conditioning in the last padded state slot (always zero otherwise)
        v = getattr(self, "_v_batch", None)
        if v is None:  # inference: commanded speed
            v = torch.full((state.shape[0],), float(self.config.exec_speed),
                           device=state.device, dtype=state.dtype)
        state = state.clone()
        state[:, -1] = v.to(state.device, state.dtype)
        return state

    def _retime_actions(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        a_n = batch[ACTION]                                     # (B, Hf, 7) normalized
        B, Hf, _ = a_n.shape
        Hc = self.config.chunk_size
        dev, dt = a_n.device, a_n.dtype
        mu, sigma = self._dataset_action_stats(dev, dt)
        raw = a_n * sigma + mu

        pad = batch.get("action_is_pad")
        pose, grip = raw[..., :6], raw[..., 6]
        if pad is not None:
            pose = pose * (~pad).unsqueeze(-1).to(dt)

        sp = torch.tensor(self.config.speeds, dtype=dt, device=dev)
        v = sp[torch.randint(len(sp), (B,), device=dev)]        # (B,) random per sample

        path = torch.cat([torch.zeros_like(pose[:, :1]), torch.cumsum(pose, dim=1)], dim=1)
        # synthetic step j covers raw indices [j*v, (j+1)*v] -> sample path at j*v
        x = (torch.arange(Hc + 1, dtype=dt, device=dev).unsqueeze(0) * v.unsqueeze(1)).clamp(max=Hf)
        x0 = x.floor().long().clamp(0, Hf)
        x1 = (x0 + 1).clamp(max=Hf)
        w = (x - x0.to(dt)).unsqueeze(-1)
        p = path.gather(1, x0.unsqueeze(-1).expand(-1, -1, 6)) * (1 - w) \
            + path.gather(1, x1.unsqueeze(-1).expand(-1, -1, 6)) * w
        syn_pose = p[:, 1:] - p[:, :-1]                         # (B, Hc, 6)

        gx = ((torch.arange(Hc, dtype=dt, device=dev).unsqueeze(0) + 0.5)
              * v.unsqueeze(1)).long().clamp(max=Hf - 1)        # (B, Hc)
        syn_grip = grip.gather(1, gx)

        out = dict(batch)
        out[ACTION] = (torch.cat([syn_pose, syn_grip.unsqueeze(-1)], dim=-1) - mu) / sigma
        if pad is not None:
            out["action_is_pad"] = pad.gather(1, gx)
        self._v_batch = v                                       # consumed by prepare_state
        return out

    def forward(self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"):
        try:
            return super().forward(self._retime_actions(batch), noise=noise, time=time, reduction=reduction)
        finally:
            self._v_batch = None
