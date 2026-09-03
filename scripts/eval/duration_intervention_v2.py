"""Strict, single-process decode-time duration interventions.

The production spline policy computes ``last_predicted_T_batch`` immediately
before calling its scalar ``_predicted_T`` helper. This controller patches only
that scalar helper and consumes the already-computed batch. Predicted, fixed,
and shuffled episodes therefore share one exact model/prediction code path and
differ only in the scalar returned to the decoder.
"""

from __future__ import annotations

import hashlib
import json
import random
import types
from pathlib import Path
from typing import Any


VALID_MODES = ("predicted", "fixed", "shuffled")
PROTOCOL = "already_computed_batch1_duration_intervention_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def load_complete_fixed_manifest(
    path: Path, config_path: Path
) -> tuple[int, dict[str, Any]]:
    """Load a complete training-target prior bound to ``config_path``."""

    path = Path(path)
    payload = json.loads(path.read_text())
    required = {
        "complete_training_scan",
        "dataset_frames_expected",
        "dataset_frames_scanned",
        "action_schedule_sha256",
        "checkpoint_config_sha256",
        "event_target_source_sha256",
        "global_duration",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"fixed manifest is missing required fields: {missing}")
    if payload["complete_training_scan"] is not True:
        raise ValueError("fixed manifest must certify complete_training_scan=true")
    expected_frames = payload["dataset_frames_expected"]
    scanned_frames = payload["dataset_frames_scanned"]
    if (
        isinstance(expected_frames, bool)
        or isinstance(scanned_frames, bool)
        or not isinstance(expected_frames, int)
        or not isinstance(scanned_frames, int)
        or expected_frames <= 0
        or scanned_frames != expected_frames
    ):
        raise ValueError(
            "fixed manifest must scan every positive training frame: "
            f"expected={expected_frames!r}, scanned={scanned_frames!r}"
        )
    if payload["checkpoint_config_sha256"] != sha256(config_path):
        raise ValueError("fixed manifest is bound to a different checkpoint config")
    for field in ("action_schedule_sha256", "event_target_source_sha256"):
        value = payload[field]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"fixed manifest has invalid {field}")
    duration = payload["global_duration"]
    if isinstance(duration, bool) or not isinstance(duration, int):
        raise ValueError("fixed manifest global_duration must be an integer")
    return duration, payload


class DurationInterventionV2:
    """Patch one loaded policy and select an intervention per reset episode."""

    def __init__(self, *, fixed_duration: int, shuffle_seed: int = 73_921) -> None:
        if isinstance(fixed_duration, bool) or not isinstance(fixed_duration, int):
            raise TypeError("fixed_duration must be an integer")
        self.fixed_duration = fixed_duration
        self.shuffle_seed = int(shuffle_seed)
        self.policy: Any | None = None
        self.mode: str | None = None
        self.task_id: int | None = None
        self.state_id: int | None = None
        self._last_prediction_batch: Any | None = None
        self._counterfactual: list[int] = []
        self._predicted: list[int] = []
        self._executed: list[int] = []
        self._reused_values = 0

    def attach(self, policy: Any) -> None:
        if self.policy is not None:
            raise RuntimeError("duration intervention is already attached")
        if not bool(getattr(policy.config, "predict_duration", False)):
            raise ValueError("duration intervention requires predict_duration=true")
        if not callable(getattr(policy, "_predicted_T", None)):
            raise TypeError("policy does not expose callable _predicted_T")
        self.policy = policy
        controller = self

        def overridden(_policy: Any, _tokens_unnorm: Any) -> int:
            # Production has already computed last_predicted_T_batch from these
            # exact tokens immediately before entering this method.
            return controller._consume_and_choose()

        policy._predicted_T = types.MethodType(overridden, policy)

    def _bounded(self, duration: int) -> int:
        assert self.policy is not None
        low = int(self.policy.config.min_seg)
        high = int(self.policy.config.horizon_max)
        if not low <= duration <= high:
            raise ValueError(
                f"duration {duration} is outside checkpoint range [{low}, {high}]"
            )
        return int(duration)

    def begin_episode(
        self,
        mode: str,
        task_id: int,
        state_id: int,
        *,
        shuffle_source: list[int] | None = None,
    ) -> None:
        if self.policy is None:
            raise RuntimeError("attach the intervention before beginning an episode")
        if self.mode is not None:
            raise RuntimeError("previous episode has not ended")
        if mode not in VALID_MODES:
            raise ValueError(f"duration mode must be one of {VALID_MODES}, got {mode!r}")
        if mode == "shuffled" and not shuffle_source:
            raise ValueError("shuffled mode requires a nonempty in-process source trace")
        if mode != "shuffled" and shuffle_source is not None:
            raise ValueError(f"{mode} mode must not receive a shuffle source")

        self.mode = mode
        self.task_id = int(task_id)
        self.state_id = int(state_id)
        self._last_prediction_batch = None
        self._counterfactual = []
        self._predicted = []
        self._executed = []
        self._reused_values = 0

        if mode == "shuffled":
            assert shuffle_source is not None
            source = [self._bounded(int(value)) for value in shuffle_source]
            indices = list(range(len(source)))
            if len(indices) > 1:
                # Sattolo: one index cycle and therefore no fixed source index.
                seed = (
                    self.shuffle_seed
                    + self.task_id * 1_000_003
                    + self.state_id * 10_007
                )
                rng = random.Random(seed)
                for index in range(len(indices) - 1, 0, -1):
                    other = rng.randrange(index)
                    indices[index], indices[other] = indices[other], indices[index]
            self._counterfactual = [source[index] for index in indices]

    def _consume_model_prediction(self) -> int:
        assert self.policy is not None
        if not hasattr(self.policy, "last_predicted_T_batch"):
            raise RuntimeError(
                "policy did not populate last_predicted_T_batch before _predicted_T"
            )
        batch = self.policy.last_predicted_T_batch
        if batch is self._last_prediction_batch:
            raise RuntimeError(
                "the same last_predicted_T_batch was consumed twice; expected one "
                "fresh batch per generated chunk"
            )
        # Retain the object so Python cannot recycle its identity between calls.
        self._last_prediction_batch = batch
        numel_method = getattr(batch, "numel", None)
        if not callable(numel_method) or int(numel_method()) != 1:
            count = int(numel_method()) if callable(numel_method) else None
            raise RuntimeError(f"duration intervention requires batch size 1, got numel={count}")
        median_method = getattr(batch, "median", None)
        if not callable(median_method):
            raise TypeError("last_predicted_T_batch does not expose median()")
        scalar = median_method().item()
        predicted = int(scalar)
        if predicted != scalar:
            raise ValueError(f"nonintegral predicted duration {scalar!r}")
        return self._bounded(predicted)

    def _consume_and_choose(self) -> int:
        if self.mode is None or self.task_id is None or self.state_id is None:
            raise RuntimeError("begin_episode must be called before policy inference")

        # Unconditional and before mode branching: all arms consume the same
        # single production-computed prediction.
        predicted = self._consume_model_prediction()
        call_index = len(self._predicted)
        self._predicted.append(predicted)
        if self.mode == "predicted":
            executed = predicted
        elif self.mode == "fixed":
            executed = self._bounded(self.fixed_duration)
        else:
            if call_index >= len(self._counterfactual):
                self._reused_values += 1
            executed = self._bounded(
                self._counterfactual[call_index % len(self._counterfactual)]
            )
        self._executed.append(executed)
        return executed

    def end_episode(self) -> dict[str, Any]:
        if self.mode is None or self.task_id is None or self.state_id is None:
            raise RuntimeError("no active episode")
        if not self._predicted:
            raise RuntimeError("episode ended without any duration calls")
        changes = [
            abs(executed - predicted)
            for predicted, executed in zip(self._predicted, self._executed, strict=True)
        ]
        chunks = [
            {
                "chunk_index": index,
                "predicted_duration": predicted,
                "executed_duration": executed,
                "absolute_duration_intervention": change,
            }
            for index, (predicted, executed, change) in enumerate(
                zip(self._predicted, self._executed, changes, strict=True)
            )
        ]
        changed = sum(change > 0 for change in changes)
        record = {
            "duration_mode": self.mode,
            "duration_chunks": chunks,
            "predicted_durations": list(self._predicted),
            "executed_durations": list(self._executed),
            "n_duration_calls": len(chunks),
            "n_intervened_duration_calls": changed,
            "intervention_call_fraction": changed / len(chunks),
            "mean_absolute_duration_intervention": sum(changes) / len(chunks),
            "reused_counterfactual_values": self._reused_values,
            "counterfactual_source_length": (
                len(self._counterfactual) if self.mode == "shuffled" else None
            ),
            "counterfactual_unique_durations": (
                len(set(self._counterfactual)) if self.mode == "shuffled" else None
            ),
        }
        self.mode = None
        self.task_id = self.state_id = None
        self._last_prediction_batch = None
        return record
