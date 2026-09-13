"""Learn the moving arm's six delta commands and analog gripper opening.

The 32-value state and 16-value action interface match the native Apollo schema.
The camera arm and both rails are fixed: the dataset exporter verifies this.
Shared spline geometry is unchanged. Event detection thresholds the gripper only
for boundary detection; fitting and decoding retain the analog opening values.
"""
import torch

from lerobot.utils.constants import ACTION
from ..smolvla_spline.event_targets import bspline_basis, first_event_indices
from ..smolvla_spline.modeling_smolvla_spline import SmolVLASplinePolicy
from .configuration_smolvla_apollo import SmolVLAApolloConfig


def apollo_raw_targets(actions, pad, cfg, pm_bank, bl_bank, pg_bank):
    if actions.ndim != 3 or actions.shape[-1] != 16:
        raise ValueError("Expected native Apollo actions (B,H,16)")
    segmentation = actions.clone()
    segmentation[..., 6] = (actions[..., 6] >= cfg.gripper_event_threshold).to(actions.dtype)
    duration = first_event_indices(segmentation, pad, pose_lo=0, grip_idx=6,
                                   min_seg=cfg.min_seg, horizon_max=cfg.horizon_max,
                                   pause_frac=cfg.pause_frac)
    mask = torch.arange(cfg.horizon_max, device=actions.device)[None, :] < duration[:, None]
    if pad is not None:
        mask = mask & ~pad.bool()
    pose = actions[..., :6] * mask[..., None]
    path = torch.cat([torch.zeros_like(pose[:, :1]), pose.cumsum(1)], 1)
    bank_index = duration - cfg.min_seg
    end = path.gather(1, duration[:, None, None].expand(-1, 1, 6))
    residual = path - bl_bank[bank_index] * end
    middle = torch.einsum("bmh,bhd->bmd", pm_bank[bank_index], residual)
    pose_ctrl = torch.cat([torch.zeros_like(end), middle, end], 1)
    grip_ctrl = torch.einsum("bnh,bh->bn", pg_bank[bank_index], actions[..., 6])
    log_duration = duration.to(actions.dtype).log()[:, None].expand(-1, cfg.n_ctrl)
    targets = torch.cat([pose_ctrl, grip_ctrl[..., None], log_duration[..., None]], -1)
    return targets, duration


class SmolVLAApolloPolicy(SmolVLASplinePolicy):
    config_class = SmolVLAApolloConfig
    name = "smolvla_apollo"

    def __init__(self, config, **kwargs):
        stats = config.embedded_spline_stats or {}
        if stats.get("apollo_gripper_decode") != "continuous_open_fraction_0_to_1":
            raise ValueError("Apollo requires its own embedded analog-gripper target statistics")
        if stats.get("apollo_gripper_event_threshold") != config.gripper_event_threshold:
            raise ValueError("Apollo statistics and gripper segmentation disagree")
        super().__init__(config, **kwargs)

    def _build_spline_targets(self, batch):
        raw, duration = apollo_raw_targets(batch[ACTION], batch.get("action_is_pad"),
                                          self.config, self._pm_bank, self._bl_bank, self._pg_bank)
        self._last_T_batch = duration
        return (raw - self._tgt_mean) / self._tgt_std

    def _decode_tokens(self, tokens, h_exec):
        # Decode the shared cumulative six-dimensional command spline. Apollo's
        # analog gripper replaces the simulator's signed gripper in the result.
        decoded = super()._decode_tokens(tokens, h_exec)
        raw = tokens[..., :8] * self._tgt_std + self._tgt_mean
        u = torch.arange(h_exec, device=tokens.device, dtype=torch.float64) / max(h_exec - 1, 1)
        basis = bspline_basis(u, self.config.n_ctrl, self.config.spline_degree).float()
        opening = torch.einsum("hn,bn->bh", basis, raw[..., 6]).clamp(0, 1)
        out = torch.zeros((*decoded.shape[:2], 16), device=tokens.device, dtype=decoded.dtype)
        out[..., :6] = decoded[..., :6]
        out[..., 6] = opening
        out[..., 14] = 1.0
        return out
