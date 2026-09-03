"""Decode-time duration interventions for causal VLA policy evaluation.

This module deliberately does not modify the policy checkpoint or production
implementation.  It replaces only ``policy._predicted_T`` after loading a
duration-enabled spline policy, leaving the sampled shape/gripper tokens
untouched.  The caller must establish an episode context before inference and
persist :meth:`end_episode` in its evaluation record.

The strongest counterfactual control is ``shuffled``.  It takes the duration
sequence recorded by a prior ``predicted`` rollout on the *same task and
initial state* and applies a deterministic permutation to its call order.  It
therefore preserves that episode's predicted-duration multiset while breaking
the alignment between the current observation/action primitive and duration.
If an intervention causes more policy calls than the source rollout, the
permutation is cycled and the reuse count is reported.
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import types
from pathlib import Path
from typing import Any, Callable


VALID_MODES = ("predicted", "fixed", "fixed_task", "shuffled")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_task_duration_map(path: Path) -> dict[str, int]:
    """Load a task-duration map, accepting either a bare or wrapped mapping."""

    payload = json.loads(path.read_text())
    mapping = payload.get("durations_by_task_description", payload.get("durations_by_task", payload))
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"empty or invalid task-duration map: {path}")
    try:
        return {str(task): int(duration) for task, duration in mapping.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"task-duration map must contain integer duration values: {path}") from exc


def load_predicted_traces(path: Path) -> tuple[dict[tuple[int, int], list[int]], dict[str, Any]]:
    """Load same-state duration traces from this evaluator's predicted-mode JSON."""

    payload = json.loads(path.read_text())
    if payload.get("duration_mode") != "predicted":
        raise ValueError("shuffle source must be a predicted-mode duration-ablation result")
    traces: dict[tuple[int, int], list[int]] = {}
    for task in payload.get("tasks", []):
        task_id = int(task["task_id"])
        for episode in task.get("episodes", []):
            state_id = int(episode["state_id"])
            trace = [int(value) for value in episode.get("predicted_durations", [])]
            if not trace:
                raise ValueError(f"empty predicted trace for task={task_id}, state={state_id}")
            key = (task_id, state_id)
            if key in traces:
                raise ValueError(f"duplicate trace for task={task_id}, state={state_id}")
            traces[key] = trace
    if not traces:
        raise ValueError(f"no predicted duration traces in {path}")
    return traces, payload


class DurationIntervention:
    """Patch one loaded spline policy with a traceable duration intervention."""

    def __init__(
        self,
        mode: str,
        *,
        fixed_duration: int | None = None,
        task_durations: dict[str, int] | None = None,
        source_traces: dict[tuple[int, int], list[int]] | None = None,
        shuffle_seed: int = 73_921,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"duration mode must be one of {VALID_MODES}, got {mode!r}")
        if mode == "fixed" and fixed_duration is None:
            raise ValueError("fixed mode requires fixed_duration")
        if mode == "fixed_task" and not task_durations:
            raise ValueError("fixed_task mode requires a nonempty task-duration map")
        if mode == "shuffled" and not source_traces:
            raise ValueError("shuffled mode requires predicted source traces")
        self.mode = mode
        self.fixed_duration = fixed_duration
        self.task_durations = task_durations or {}
        self.source_traces = source_traces or {}
        self.shuffle_seed = int(shuffle_seed)
        self.policy: Any | None = None
        self._original: Callable[[Any], int] | None = None
        self.task_id: int | None = None
        self.state_id: int | None = None
        self.task_duration_key: str | None = None
        self.predicted: list[int] = []
        self.executed: list[int] = []
        self._counterfactual: list[int] = []
        self.reused_values = 0

    def attach(self, policy: Any) -> None:
        if self.policy is not None:
            raise RuntimeError("duration intervention is already attached")
        if not bool(getattr(policy.config, "predict_duration", False)):
            raise ValueError("duration intervention requires predict_duration=true")
        original = getattr(policy, "_predicted_T", None)
        if original is None or not callable(original):
            raise TypeError("policy does not expose callable _predicted_T")
        self.policy = policy
        self._original = original

        controller = self

        def overridden(_policy: Any, tokens_unnorm: Any) -> int:
            return controller._choose(tokens_unnorm)

        policy._predicted_T = types.MethodType(overridden, policy)

    def begin_episode(
        self, task_id: int, state_id: int, task_duration_key: str | None = None
    ) -> None:
        if self.policy is None:
            raise RuntimeError("attach the intervention before beginning an episode")
        self.task_id, self.state_id = int(task_id), int(state_id)
        self.task_duration_key = task_duration_key or str(self.task_id)
        self.predicted = []
        self.executed = []
        self.reused_values = 0
        self._counterfactual = []
        if self.mode == "fixed_task" and self.task_duration_key not in self.task_durations:
            raise KeyError(f"no fixed duration for task {self.task_duration_key!r}")
        if self.mode == "shuffled":
            key = (self.task_id, self.state_id)
            if key not in self.source_traces:
                raise KeyError(f"no shuffle source for task={key[0]}, state={key[1]}")
            source = list(self.source_traces[key])
            # A seeded Sattolo cycle preserves the exact multiset while ensuring
            # that no *index* remains aligned when there is more than one call.
            # Repeated duration values can still make some effective values
            # equal, which is measured explicitly in ``end_episode``.
            seed = self.shuffle_seed + self.task_id * 1_000_003 + self.state_id * 10_007
            rng = random.Random(seed)
            indices = list(range(len(source)))
            for index in range(len(indices) - 1, 0, -1):
                other = rng.randrange(index)
                indices[index], indices[other] = indices[other], indices[index]
            self._counterfactual = [source[index] for index in indices]

    def _bounded(self, duration: int) -> int:
        assert self.policy is not None
        low = int(self.policy.config.min_seg)
        high = int(self.policy.config.horizon_max)
        if not low <= duration <= high:
            raise ValueError(f"intervened duration {duration} outside checkpoint range [{low}, {high}]")
        return int(duration)

    def _choose(self, tokens_unnorm: Any) -> int:
        if self.task_id is None or self.state_id is None:
            raise RuntimeError("begin_episode must be called before policy inference")
        assert self._original is not None
        predicted = self._bounded(int(self._original(tokens_unnorm)))
        call_index = len(self.predicted)
        self.predicted.append(predicted)

        if self.mode == "predicted":
            executed = predicted
        elif self.mode == "fixed":
            assert self.fixed_duration is not None
            executed = self._bounded(int(self.fixed_duration))
        elif self.mode == "fixed_task":
            assert self.task_duration_key is not None
            executed = self._bounded(int(self.task_durations[self.task_duration_key]))
        else:
            if call_index >= len(self._counterfactual):
                self.reused_values += 1
            executed = self._bounded(self._counterfactual[call_index % len(self._counterfactual)])
        self.executed.append(executed)
        return executed

    def end_episode(self) -> dict[str, Any]:
        if self.task_id is None or self.state_id is None:
            raise RuntimeError("no active episode")
        absolute_changes = [
            abs(executed - predicted)
            for predicted, executed in zip(self.predicted, self.executed, strict=True)
        ]
        changed = sum(value > 0 for value in absolute_changes)
        record = {
            "predicted_durations": list(self.predicted),
            "executed_durations": list(self.executed),
            "n_duration_calls": len(self.predicted),
            "n_intervened_duration_calls": changed,
            "intervention_call_fraction": changed / len(self.predicted) if self.predicted else 0.0,
            "mean_absolute_duration_intervention": (
                sum(absolute_changes) / len(absolute_changes) if absolute_changes else 0.0
            ),
            "reused_counterfactual_values": self.reused_values,
            "counterfactual_source_length": (
                len(self._counterfactual) if self.mode == "shuffled" else None
            ),
            "counterfactual_unique_durations": (
                len(set(self._counterfactual)) if self.mode == "shuffled" else None
            ),
        }
        self.task_id = self.state_id = None
        self.task_duration_key = None
        return record


def task_medians_from_result(path: Path) -> dict[int, int]:
    """Diagnostic helper, not the preregistered training-target fixed control."""

    traces, _ = load_predicted_traces(path)
    by_task: dict[int, list[int]] = {}
    for (task_id, _), trace in traces.items():
        by_task.setdefault(task_id, []).extend(trace)
    return {task: int(round(statistics.median(values))) for task, values in by_task.items()}
