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

    # Duration mode-snap (decode-only): the duration target is bimodal
    # (events ~22 vs cap 40); flow matching smears cap chunks down to ~30,
    # making transports execute ~33% faster than their fitted shape intended
    # (measured: cap bias -9.5, event MAE 3.7 / r=0.84). Snap T-hat >= threshold
    # back to horizon_max so transport chunks run at intended speed.
    duration_snap_threshold: int | None = None

    # Ease-out decode (v2, decode-only): demonstrators decelerate to ~0.55x
    # cruise speed around gripper toggles; the LSQ fit smooths that terminal
    # deceleration away (endpoint POSITION is pinned, terminal VELOCITY is not),
    # so the policy contacts at ~0.88x cruise (measured, libero_spatial). For
    # event-terminated chunks (T_hat < horizon_max = contact imminent), re-time
    # the final 3 path steps over (3 + ease_out) steps with cosine spacing:
    # same path, exact endpoint, velocity tapering to ~0 at the event.
    ease_out: int | None = None

    # Decode-consistency auxiliary loss (v2 only): the flow loss lives in
    # control-point space (6 tokens x 8 channels) where fine terminal detail is
    # a tiny loss fraction; waypoint policies supervise 50 per-step targets at
    # equalized scale. This auxiliary term projects the SIGNED flow error
    # through the decode basis — mathematically the x0-space PATH error
    # (x0_hat - x0 = t*(u_t - v_t), decode is linear) — unifying the two
    # supervision geometries while keeping every spline capability.
    # 0 = off. Suggested 5.0 (aux ~ comparable magnitude to flow loss early).
    decode_consistency_weight: float = 0.0

    # Contact-weighted fit (v2 only): event-terminated chunks place contact at
    # u=1, so end-weighting the LSQ fit re-allocates representation error away
    # from the precise terminal motions (offline mechanism study: tail rel-err
    # -23% at n6 / -31% at n8 for weight 9, body cost ~+4%; knot densification
    # BACKFIRES at this control-point budget). Weight ramps 1 -> fit_end_weight
    # over the last 25% of the chunk. Decode is completely unchanged.
    fit_end_weight: float | None = None

    # --- speed-heterogeneous demonstrations (training-time experiment) ---
    # Per-episode synthetic speed factor s = speed_aug[episode_index % len]:
    # the same spatial path executed s x slower (s >= 1 only: speeding up the
    # commanded path would clip deltas at the [-1,1] actuator bound).
    # v1 (fixed-T): the chunk becomes the first `horizon` SYNTHETIC steps of the
    #   slowed path -> shape supervision is speed-contaminated (the point).
    # v2 (time alloc): event segmentation + shape fit stay on the RAW segment;
    #   ONLY the duration label scales to s*T -> shape supervision stays exact.
    # Stats files must be regenerated for the augmented distributions.
    speed_aug: list[float] | None = None

    # Self-paced replanning (v2 decode-only): execute replan_frac of each
    # chunk's PREDICTED duration before replanning, instead of a fixed
    # n_action_steps cadence. The policy's own duration head schedules the
    # replans (chunks end at motion events, so replans align with events).
    # Requires n_action_steps >= horizon_max so the fixed cadence never cuts
    # a chunk short. None = fixed-cadence behavior (default).
    replan_frac: float | None = None
    # Variant: replan a FIXED number of steps before the predicted chunk end
    # (absolute margin, not fractional). Tests the boundary-clipping hypothesis:
    # executing exactly TO the predicted event places gripper toggles at chunk
    # boundaries where duration error clips them; a small margin should recover.
    replan_margin: int | None = None

    # Velocity-continuous chunk chaining (decode-only): pin the new chunk's
    # second control point so its initial velocity matches the previous chunk's
    # velocity at the replan point. Removes the replan-boundary velocity
    # discontinuity (measured boundary_ratio ~4) that c0-pinning alone leaves.
    chain_velocity: bool = False

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
        # v2 decodes a DYNAMIC chunk length (predicted T, up to horizon_max);
        # v1 decodes a fixed one. Validate the replan cadence against the max.
        decoded_len = (
            self.horizon_max
            if self.predict_duration
            else (self.exec_horizon if self.exec_horizon is not None else self.horizon)
        )
        if self.n_action_steps > decoded_len:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= decoded chunk "
                f"length ({decoded_len} env steps)."
            )
        if self.replan_frac is not None:
            if not self.predict_duration:
                raise ValueError("replan_frac (self-paced replanning) requires predict_duration=true")
            if not (0.0 < self.replan_frac <= 1.0):
                raise ValueError(f"replan_frac must be in (0, 1], got {self.replan_frac}")
        if self.ease_out is not None:
            if not self.predict_duration:
                raise ValueError("ease_out requires predict_duration=true (event-terminated chunks)")
            if self.ease_out < 1:
                raise ValueError(f"ease_out must be >= 1 extra step, got {self.ease_out}")
        if self.decode_consistency_weight > 0 and not self.predict_duration:
            raise NotImplementedError("decode_consistency_weight currently implemented for the v2 (event-segmented) head only")
        if self.fit_end_weight is not None:
            if not self.predict_duration:
                raise ValueError("fit_end_weight requires predict_duration=true (contact = chunk end only for event-terminated chunks)")
            if self.fit_end_weight <= 1.0:
                raise ValueError(f"fit_end_weight must be > 1, got {self.fit_end_weight}")
        if self.speed_aug is not None:
            if len(self.speed_aug) < 2:
                raise ValueError("speed_aug needs >= 2 factors to create heterogeneity")
            if min(self.speed_aug) < 1.0 and not self.predict_duration:
                # v1 resamples the PATH: s<1 doubles deltas -> actuator clipping.
                # v2 scales only the duration LABEL: any s>0 is safe (shape untouched).
                raise ValueError("speed_aug factors must be >= 1 for the v1 head (s<1 clips deltas)")
            if min(self.speed_aug) <= 0:
                raise ValueError("speed_aug factors must be positive")
        if self.replan_margin is not None:
            if not self.predict_duration:
                raise ValueError("replan_margin (self-paced replanning) requires predict_duration=true")
            if self.replan_frac is not None:
                raise ValueError("set replan_frac or replan_margin, not both")
            if self.replan_margin < 0:
                raise ValueError(f"replan_margin must be >= 0, got {self.replan_margin}")
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
