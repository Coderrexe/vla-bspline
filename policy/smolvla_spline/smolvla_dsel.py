# Copyright 2026 — VLA B-spline project.
"""SmolVLA-DSel: the DATA-LEVEL selectivity baseline (DemoSpeedup/ESPADA
mechanism on our stack) — third arm of the speed-control triad:

  input-conditioned (smolvla_tempo)  |  data-level (THIS)  |  decode-level (ours)

Each training window is retimed by an event-aware rule: if the window's first
`horizon` steps contain NO gripper/pause event (pure transit), the chunk is
accelerated by `accel`; event-containing (contact) windows stay at 1x. The
policy learns fast transit + careful contact *baked into the weights* — fixed
at training time, unlike our decode-α. No conditioning input; inference is a
plain smolvla checkpoint.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from lerobot.configs import PreTrainedConfig

from .smolvla_tempo import SmolVLATempoConfig, SmolVLATempoPolicy

_PAUSE_FRAC = 0.15


@PreTrainedConfig.register_subclass("smolvla_dsel")
@dataclass
class SmolVLADSelConfig(SmolVLATempoConfig):
    accel: float = 1.5

    def __post_init__(self):
        self.speeds = [1.0, self.accel]  # drives the fetch-window property
        super().__post_init__()


class SmolVLADSelPolicy(SmolVLATempoPolicy):
    config_class = SmolVLADSelConfig
    name = "smolvla_dsel"

    def prepare_state(self, batch):
        # no speed conditioning — data-level selectivity is baked into weights;
        # skip the tempo class's state-slot injection entirely
        return super(SmolVLATempoPolicy, self).prepare_state(batch)

    def _retime_actions(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # choose v per sample BY EVENT CONTENT of the near window, then reuse
        # the tempo resampling machinery with that fixed v
        from lerobot.utils.constants import ACTION

        a_n = batch[ACTION]
        B, Hf, _ = a_n.shape
        dt, dev = a_n.dtype, a_n.device
        mu, sigma = self._dataset_action_stats(dev, dt)
        raw = a_n * sigma + mu
        look = min(24, Hf)
        grip = raw[:, :look, 6]
        toggles = (grip[:, 1:].sign() != grip[:, :-1].sign()).any(dim=1)
        speed = raw[:, :look, :6].norm(dim=-1)
        med = speed.median(dim=1, keepdim=True).values + 1e-9
        low = speed < _PAUSE_FRAC * med
        pauses = (low[:, 1:] & low[:, :-1]).any(dim=1)
        transit = ~(toggles | pauses)                     # no event ahead -> accelerate
        v = torch.where(transit, torch.full((B,), float(self.config.accel), device=dev, dtype=dt),
                        torch.ones(B, device=dev, dtype=dt))
        self._forced_v = v
        out = super()._retime_actions(batch)
        self._forced_v = None
        return out
