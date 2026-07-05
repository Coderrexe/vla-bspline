# SmolVLA with a B-spline + time-allocation action head.
#
# The policy's flow-matching expert generates `n_ctrl` SPLINE TOKENS instead of
# `chunk_size` waypoints. Each token row is:
#     [ 6 pose-path control-point dims | 1 gripper control point | 1 duration channel ]
# The pose control points parametrize a clamped uniform cubic B-spline over the
# CUMULATIVE commanded delta-action path of the next `horizon` env steps, with
# control point 0 pinned to 0 (chunk starts exactly at the current pose) and the
# last control point pinned to the chunk displacement (exact endpoint).
# A downstream decode samples the spline at ANY execution step count and emits
# per-step deltas — decoupling execution rate from the training rate.

from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig

from ..smolvla.configuration_smolvla import SmolVLAConfig


@PreTrainedConfig.register_subclass("smolvla_spline")
@dataclass
class SmolVLASplineConfig(SmolVLAConfig):
    # --- spline head ---
    n_ctrl: int = 6           # spline control points per dim (== expert token count)
    horizon: int = 20         # v1: env steps per fixed training chunk
    spline_degree: int = 3    # cubic
    exec_horizon: int | None = None   # v1 decode step count; None -> horizon (native rate)
    # normalization stats file for control-point targets (relative to this module)
    spline_stats_file: str = "spline_stats_libero.json"

    # --- v2: time allocation (variable-duration, event-segmented chunks) ---
    predict_duration: bool = False    # False = v1 fixed-T (Baseline B); True = ours
    horizon_max: int = 40             # v2: fetched window / duration cap (env steps)
    min_seg: int = 6                  # v2: minimum chunk duration (>= n_ctrl fit limit)
    pause_frac: float = 0.15          # v2: pause event = speed < frac * window median
    exec_rate_ratio: float = 1.0      # v2 decode: exec control rate / training rate
    spline_stats_file_v2: str = "spline_stats_libero_v2.json"

    # Feasibility-aware time stretching (decode-time, works for v1 and v2):
    # if the decoded per-step deltas exceed the actuator bound, stretch the
    # execution horizon until every step is feasible — the robot slows down
    # instead of clipping. Unique to continuous-trajectory action heads.
    feasibility_stretch: bool = False
    actuator_bound: float = 1.0

    # Duration-aware selective speedup (v2 decode): execute chunks with
    # predicted duration ABOVE the threshold at alpha x fewer steps (faster),
    # while short (grasp-critical) chunks run at full care. Only expressible
    # with a duration head — the throughput/success Pareto experiment.
    speedup_alpha: float = 1.0        # <1.0 = faster execution of long chunks
    speedup_T_threshold: int = 0      # apply alpha only when predicted T > this

    # --- overridden SmolVLA defaults ---
    chunk_size: int = 6       # forced to n_ctrl in __post_init__
    n_action_steps: int = 10  # env steps consumed per model invocation (replan horizon)

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            # IDENTITY is load-bearing: control-point targets are normalized
            # internally with per-token stats. Per-step MEAN_STD on deltas would
            # break exact Hz-retargeting (the per-step mean does not telescope).
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self):
        # Deliberately skip SmolVLAConfig.__post_init__: its
        # `n_action_steps <= chunk_size` check compares against the TOKEN count,
        # but our tokens are control points — the decoded chunk has
        # `exec_horizon or horizon` env steps. Validate against that instead.
        PreTrainedConfig.__post_init__(self)

        self.chunk_size = self.n_ctrl  # expert sequence length == spline tokens
        decoded_len = self.exec_horizon if self.exec_horizon is not None else self.horizon
        if self.n_action_steps > decoded_len:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= decoded chunk "
                f"length ({decoded_len} env steps)."
            )
        if self.n_ctrl < self.spline_degree + 1:
            raise ValueError(f"cubic B-spline needs n_ctrl >= {self.spline_degree + 1}")
        if self.horizon + 1 < self.n_ctrl:
            raise ValueError("horizon too short for the number of control points")
        if self.use_delta_joint_actions_aloha or self.adapt_to_pi_aloha:
            raise NotImplementedError("aloha adaptations are not supported by the spline head")

    @property
    def action_delta_indices(self) -> list:
        # Fetch the RAW action window from the dataset; the policy converts it
        # to spline targets on the fly in forward(). v2 fetches the max window
        # and picks the event-defined prefix per sample.
        return list(range(self.horizon_max if self.predict_duration else self.horizon))
