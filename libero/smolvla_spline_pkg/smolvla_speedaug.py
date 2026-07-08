# Copyright 2026 — VLA B-spline project.
"""SmolVLA-SpeedAug: the WAYPOINT arm of the speed-heterogeneous-demos study.

Standard waypoint SmolVLA, but each training episode is presented as the same
spatial path executed s x slower (s = speed_aug[episode_index % len]). The
transformation happens in forward() on the fetched action window; observations
are untouched (SmolVLA conditions on the current frame only), and inference is
completely unchanged — deploy/eval exactly like a plain smolvla checkpoint.

NORMALIZATION NOTE: the preprocessor delivers batch[ACTION] normalized with the
DATASET's per-dim mean/std. We unnormalize with those same stats (shipped as
waypoint_action_stats_libero.json), build the synthetic slow chunk in RAW space
(cumulative path -> linear resample -> diffs; affine normalization does NOT
commute through resampling because of the mean term), and re-normalize with the
same stats. Synthetic slow deltas are smaller, so the model sees slightly
non-unit variance for s>1 episodes — accepted and documented in the design doc.
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


@PreTrainedConfig.register_subclass("smolvla_speedaug")
@dataclass
class SmolVLASpeedAugConfig(SmolVLAConfig):
    # per-episode slowdown factors; s >= 1 only (speeding up the commanded path
    # would clip deltas at the [-1,1] actuator bound -> corrupted demos)
    speed_aug: list[float] = field(default_factory=lambda: [1.0, 1.5, 2.0])
    action_stats_file: str = "waypoint_action_stats_libero.json"

    def __post_init__(self):
        super().__post_init__()
        if len(self.speed_aug) < 2:
            raise ValueError("speed_aug needs >= 2 factors to create heterogeneity")
        if min(self.speed_aug) < 1.0:
            raise ValueError("speed_aug factors must be >= 1")


class SmolVLASpeedAugPolicy(SmolVLAPolicy):
    config_class = SmolVLASpeedAugConfig
    name = "smolvla_speedaug"

    def _dataset_action_stats(self, device, dtype):
        if getattr(self, "_sa_mu", None) is None:
            path = os.path.join(os.path.dirname(__file__), self.config.action_stats_file)
            with open(path) as f:
                s = json.load(f)
            self._sa_mu = torch.tensor(s["mean"], dtype=torch.float32)
            self._sa_sigma = torch.tensor(s["std"], dtype=torch.float32).clamp_min(1e-6)
        return self._sa_mu.to(device=device, dtype=dtype), self._sa_sigma.to(device=device, dtype=dtype)

    def _slow_down_actions(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        a_n = batch[ACTION]                                            # (B, H, 7) normalized
        B, H, _ = a_n.shape
        dev, dt = a_n.device, a_n.dtype
        mu, sigma = self._dataset_action_stats(dev, dt)
        raw = a_n * sigma + mu

        pad = batch.get("action_is_pad")
        pose, grip = raw[..., :6], raw[..., 6]
        if pad is not None:
            pose = pose * (~pad).unsqueeze(-1).to(dt)                  # stop at episode end

        ep = batch.get("episode_index")
        if ep is None:
            raise ValueError("speed_aug requires 'episode_index' in the batch")
        s_vals = torch.tensor(self.config.speed_aug, dtype=dt, device=dev)
        s = s_vals[ep.long().view(-1) % len(s_vals)]                   # (B,)

        # cumulative path -> sample at raw index j/s (<= H since s >= 1) -> diffs
        path = torch.cat([torch.zeros_like(pose[:, :1]), torch.cumsum(pose, dim=1)], dim=1)
        x = torch.arange(H + 1, dtype=dt, device=dev).unsqueeze(0) / s.unsqueeze(1)  # (B, H+1)
        x0 = x.floor().long().clamp(0, H)
        x1 = (x0 + 1).clamp(max=H)
        w = (x - x0.to(dt)).unsqueeze(-1)
        p = path.gather(1, x0.unsqueeze(-1).expand(-1, -1, 6)) * (1 - w) \
            + path.gather(1, x1.unsqueeze(-1).expand(-1, -1, 6)) * w
        syn_pose = p[:, 1:] - p[:, :-1]                                # (B, H, 6)

        # gripper + pad mask: nearest raw sample for each synthetic step
        gx = ((torch.arange(H, dtype=dt, device=dev).unsqueeze(0) + 0.5)
              / s.unsqueeze(1)).long().clamp(max=H - 1)                # (B, H)
        syn_grip = grip.gather(1, gx)

        out = dict(batch)
        syn_raw = torch.cat([syn_pose, syn_grip.unsqueeze(-1)], dim=-1)
        out[ACTION] = (syn_raw - mu) / sigma
        if pad is not None:
            out["action_is_pad"] = pad.gather(1, gx)
        return out

    def forward(self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"):
        return super().forward(self._slow_down_actions(batch), noise=noise, time=time, reduction=reduction)
