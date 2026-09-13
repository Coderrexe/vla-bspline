"""Prediction-only Apollo adapter. This module does not connect to a robot.

Inputs use the native demonstration interface: 32 state values and two RGB
uint8 HWC images. Outputs are raw 16-channel commands, not normalized actions.
The caller owns command timing, safety checks, and all physical execution.
"""
from pathlib import Path
import json

import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla_apollo.configuration_smolvla_apollo import SmolVLAApolloConfig
from lerobot.policies.smolvla_apollo.modeling_smolvla_apollo import SmolVLAApolloPolicy


class ApolloPredictor:
    """Load a checkpoint and return one chunk for a single observation.

    Do not feed these commands to a joint-position or velocity controller.
    They are native recorded Apollo delta-EE commands plus absolute gripper
    openings. Pose frames are labeled arm_base; this alone does not establish
    how angular increments are composed in Apollo's action-application code.
    """

    def __init__(self, checkpoint, device="cuda"):
        self.checkpoint = Path(checkpoint)
        provenance = json.loads((self.checkpoint / "dataset_provenance.json").read_text())
        self.task = provenance["task"]
        self.interface = json.loads((self.checkpoint / "apollo_interface.json").read_text())
        self.cfg = PreTrainedConfig.from_pretrained(self.checkpoint, local_files_only=True)
        self.cfg.device = device
        self.cfg.load_vlm_weights = False
        if isinstance(self.cfg, SmolVLAApolloConfig):
            policy_class = SmolVLAApolloPolicy
        elif self.cfg.type == "smolvla":
            policy_class = SmolVLAPolicy
        else:
            raise ValueError(f"Unsupported Apollo policy type: {self.cfg.type}")
        if tuple(self.cfg.action_feature.shape) != (16,) or tuple(self.cfg.input_features["observation.state"].shape) != (32,):
            raise ValueError("Checkpoint is not the native Apollo 32-state/16-action interface")
        self.policy = policy_class.from_pretrained(
            self.checkpoint, config=self.cfg, strict=True, local_files_only=True
        ).eval()
        self.pre, self.post = make_pre_post_processors(
            self.cfg, pretrained_path=str(self.checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.reset()

    def reset(self):
        """Call between episodes, after an operator intervention, or on restart."""
        self.policy.reset()
        for processor in (self.pre, self.post):
            if hasattr(processor, "reset"):
                processor.reset()

    def prepare(self, state, view_rgb, grip_rgb):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (32,) or not np.isfinite(state).all():
            raise ValueError("state must be a finite vector with the recorded 32-value ordering")
        batch = {"observation.state": torch.from_numpy(state.copy()), "task": self.task}
        for name, image in (("view_wrist", view_rgb), ("grip_wrist", grip_rgb)):
            image = np.asarray(image)
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise ValueError(f"{name} must be RGB uint8 HWC (480,640,3); convert BGR explicitly")
            batch[f"observation.images.{name}"] = (
                torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255)
            )
        return self.pre(batch)

    @torch.inference_mode()
    def predict_chunk(self, state, view_rgb, grip_rgb, *, noise=None):
        batch = self.prepare(state, view_rgb, grip_rgb)
        actions = self.post(self.policy.predict_action_chunk(batch, noise=noise))
        raw = actions.detach().cpu().float().numpy()[0]
        if raw.ndim != 2 or raw.shape[1] != 16 or not np.isfinite(raw).all():
            raise RuntimeError("Invalid predicted action chunk; do not execute")
        # Both rails and the camera arm were inactive in every demonstration.
        # The waypoint head has small numerical errors on constant channels;
        # enforce the same parked-arm interface for both heads.
        raw = raw.copy()
        raw[:, 7:14] = 0
        raw[:, 14] = 1
        raw[:, 15] = 0
        raw[:, 6] = np.clip(raw[:, 6], 0, 1)
        # A caller should consume at most this prefix before observing again.
        # No pose clipping or safety guarantee is implemented by this adapter.
        return raw[:self.cfg.n_action_steps]
