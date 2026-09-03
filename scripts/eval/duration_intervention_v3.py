"""Production-path duration wrapper for stochastic paired-block evaluation.

Every arm first calls the policy's bound original ``_predicted_T(tokens)``.
This preserves the deployed recomputation exactly. The wrapper records that
canonical scalar and only then returns predicted, fixed, or shuffled duration.
"""

from __future__ import annotations

import random
import types
from typing import Any, Callable


VALID_MODES = ("predicted", "fixed", "shuffled")
PROTOCOL = "bound_original_predicted_T_then_scalar_intervention_v3"


class DurationInterventionV3:
    def __init__(self, *, fixed_duration: int, shuffle_seed: int = 73_921) -> None:
        if isinstance(fixed_duration, bool) or not isinstance(fixed_duration, int):
            raise TypeError("fixed_duration must be an integer")
        self.fixed_duration = fixed_duration
        self.shuffle_seed = int(shuffle_seed)
        self.policy: Any | None = None
        self._original: Callable[[Any], int] | None = None
        self.mode: str | None = None
        self.task_id: int | None = None
        self.state_id: int | None = None
        self.repeat_index: int | None = None
        self._counterfactual: list[int] = []
        self._predicted: list[int] = []
        self._executed: list[int] = []
        self._reused_values = 0

    def attach(self, policy: Any) -> None:
        if self.policy is not None:
            raise RuntimeError("duration intervention is already attached")
        original = getattr(policy, "_predicted_T", None)
        if not callable(original):
            raise TypeError("policy does not expose callable _predicted_T")
        if not bool(getattr(policy.config, "predict_duration", False)):
            raise ValueError("duration intervention requires predict_duration=true")
        self.policy = policy
        self._original = original
        controller = self

        def overridden(_policy: Any, tokens_unnorm: Any) -> int:
            return controller._consume_and_choose(tokens_unnorm)

        policy._predicted_T = types.MethodType(overridden, policy)

    def _bounded(self, duration: int) -> int:
        assert self.policy is not None
        low = int(self.policy.config.min_seg)
        high = int(self.policy.config.horizon_max)
        if not low <= duration <= high:
            raise ValueError(f"duration {duration} outside [{low}, {high}]")
        return int(duration)

    def begin_episode(
        self,
        mode: str,
        task_id: int,
        state_id: int,
        repeat_index: int,
        *,
        shuffle_source: list[int] | None = None,
    ) -> None:
        if self.policy is None:
            raise RuntimeError("attach before begin_episode")
        if self.mode is not None:
            raise RuntimeError("previous episode has not ended")
        if mode not in VALID_MODES:
            raise ValueError(f"unknown mode {mode!r}")
        if mode == "shuffled" and not shuffle_source:
            raise ValueError("shuffled mode requires a nonempty donor trace")
        if mode != "shuffled" and shuffle_source is not None:
            raise ValueError(f"{mode} mode must not receive shuffle_source")

        # Reset every per-rollout field, including the permutation and reuse
        # counters, before accepting the next fully reset environment episode.
        self.mode = mode
        self.task_id = int(task_id)
        self.state_id = int(state_id)
        self.repeat_index = int(repeat_index)
        self._counterfactual = []
        self._predicted = []
        self._executed = []
        self._reused_values = 0
        if mode == "shuffled":
            assert shuffle_source is not None
            source = [self._bounded(int(value)) for value in shuffle_source]
            indices = list(range(len(source)))
            if len(indices) > 1:
                seed = (
                    self.shuffle_seed
                    + self.task_id * 1_000_003
                    + self.state_id * 10_007
                    + self.repeat_index * 101
                )
                rng = random.Random(seed)
                for index in range(len(indices) - 1, 0, -1):
                    other = rng.randrange(index)
                    indices[index], indices[other] = indices[other], indices[index]
            self._counterfactual = [source[index] for index in indices]

    def _consume_and_choose(self, tokens_unnorm: Any) -> int:
        if self.mode is None:
            raise RuntimeError("begin_episode must precede policy inference")
        assert self._original is not None

        # Load-bearing invariant: every mode executes the exact deployed scalar
        # compute before any arm-specific selection.
        predicted = self._bounded(int(self._original(tokens_unnorm)))
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
        if self.mode is None:
            raise RuntimeError("no active episode")
        if not self._predicted:
            raise RuntimeError("episode ended without duration calls")
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
        self.task_id = self.state_id = self.repeat_index = None
        self._counterfactual = []
        self._predicted = []
        self._executed = []
        self._reused_values = 0
        return record
