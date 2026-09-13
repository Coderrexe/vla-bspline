"""Apollo adapter for the project's event spline action representation."""
from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from ..smolvla_spline.configuration_smolvla_spline import SmolVLASplineConfig


@PreTrainedConfig.register_subclass("smolvla_apollo")
@dataclass
class SmolVLAApolloConfig(SmolVLASplineConfig):
    n_ctrl: int = 8
    min_seg: int = 8
    horizon_max: int = 24
    predict_duration: bool = True
    n_action_steps: int = 8
    gripper_event_threshold: float = 0.5
    apollo_schema_version: int = 1
    native_fps: int = 25

    def __post_init__(self):
        super().__post_init__()
        if self.action_layout != "eef7" or not self.predict_duration:
            raise ValueError("Apollo adapter supports the event head on leading EE7 channels")
        if not 0 < self.gripper_event_threshold < 1:
            raise ValueError("Gripper event threshold must lie in (0,1)")
        if self.speed_aug or self.ease_out or self.profile_alpha or self.profile_slow_alpha or self.chain_velocity or self.feasibility_stretch:
            raise ValueError("Apollo v1 uses a uniform decode grid; unsupported timing profile")
