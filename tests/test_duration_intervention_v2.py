from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.eval.duration_intervention_v2 import (
    DurationInterventionV2,
    canonical_json_sha256,
    load_complete_fixed_manifest,
)


class FakeBatch:
    def __init__(self, values: list[int]):
        self.values = values

    def numel(self) -> int:
        return len(self.values)

    def median(self) -> "FakeBatch":
        ordered = sorted(self.values)
        return FakeBatch([ordered[(len(ordered) - 1) // 2]])

    def item(self) -> int:
        if len(self.values) != 1:
            raise ValueError("not scalar")
        return self.values[0]


class DummyPolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            predict_duration=True, min_seg=6, horizon_max=24
        )
        self.original_scalar_calls = 0
        self.batch_recompute_calls = 0

    def _predicted_T(self, _tokens: object) -> int:
        self.original_scalar_calls += 1
        raise AssertionError("patched scalar method must not call the original")

    def _predicted_T_batch(self, _tokens: object) -> FakeBatch:
        self.batch_recompute_calls += 1
        raise AssertionError("intervention must not recompute the duration batch")


def _call(policy: DummyPolicy, value: int) -> int:
    policy.last_predicted_T_batch = FakeBatch([value])
    return policy._predicted_T(None)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DurationInterventionV2Test(unittest.TestCase):
    def test_all_modes_consume_already_computed_batch_through_one_hook(self) -> None:
        policy = DummyPolicy()
        control = DurationInterventionV2(fixed_duration=24, shuffle_seed=5)
        control.attach(policy)

        control.begin_episode("predicted", 0, 0)
        self.assertEqual(
            [_call(policy, value) for value in (7, 11, 19)], [7, 11, 19]
        )
        predicted = control.end_episode()

        control.begin_episode("fixed", 0, 0)
        self.assertEqual(
            [_call(policy, value) for value in (7, 11, 19)], [24, 24, 24]
        )
        fixed = control.end_episode()

        control.begin_episode(
            "shuffled", 0, 0, shuffle_source=predicted["predicted_durations"]
        )
        shuffled_values = [_call(policy, value) for value in (7, 11, 19)]
        shuffled = control.end_episode()

        self.assertEqual(policy.original_scalar_calls, 0)
        self.assertEqual(policy.batch_recompute_calls, 0)
        self.assertEqual(fixed["predicted_durations"], [7, 11, 19])
        self.assertEqual(fixed["executed_durations"], [24, 24, 24])
        self.assertEqual(shuffled["predicted_durations"], [7, 11, 19])
        self.assertEqual(sorted(shuffled_values), [7, 11, 19])
        self.assertTrue(
            all(
                left != right
                for left, right in zip(
                    shuffled_values, [7, 11, 19], strict=True
                )
            )
        )

    def test_rejects_stale_or_non_batch1_prediction(self) -> None:
        policy = DummyPolicy()
        control = DurationInterventionV2(fixed_duration=24)
        control.attach(policy)
        control.begin_episode("predicted", 0, 0)
        batch = FakeBatch([8])
        policy.last_predicted_T_batch = batch
        self.assertEqual(policy._predicted_T(None), 8)
        with self.assertRaisesRegex(RuntimeError, "consumed twice"):
            policy._predicted_T(None)

        policy_two = DummyPolicy()
        control_two = DurationInterventionV2(fixed_duration=24)
        control_two.attach(policy_two)
        control_two.begin_episode("predicted", 0, 0)
        policy_two.last_predicted_T_batch = FakeBatch([8, 12])
        with self.assertRaisesRegex(RuntimeError, "batch size 1"):
            policy_two._predicted_T(None)

    def test_episode_mode_contract_and_shuffle_reuse(self) -> None:
        policy = DummyPolicy()
        control = DurationInterventionV2(fixed_duration=24)
        control.attach(policy)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            control.begin_episode("shuffled", 1, 2)
        with self.assertRaisesRegex(ValueError, "must not receive"):
            control.begin_episode("predicted", 1, 2, shuffle_source=[6])

        control.begin_episode("shuffled", 1, 2, shuffle_source=[6, 8])
        executed = [_call(policy, value) for value in (7, 9, 11)]
        record = control.end_episode()
        self.assertEqual(executed[2], executed[0])
        self.assertEqual(record["reused_counterfactual_values"], 1)
        self.assertEqual(record["counterfactual_source_length"], 2)

    def test_fixed_manifest_must_be_complete_and_config_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text('{"predict_duration":true}\n')
            manifest = root / "prior.json"
            payload = {
                "complete_training_scan": True,
                "dataset_frames_expected": 100,
                "dataset_frames_scanned": 100,
                "action_schedule_sha256": "a" * 64,
                "checkpoint_config_sha256": _digest(config),
                "event_target_source_sha256": "b" * 64,
                "global_duration": 24,
            }
            manifest.write_text(json.dumps(payload))
            duration, loaded = load_complete_fixed_manifest(manifest, config)
            self.assertEqual(duration, 24)
            self.assertEqual(loaded, payload)

            payload["dataset_frames_scanned"] = 99
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "every positive training frame"):
                load_complete_fixed_manifest(manifest, config)

            payload["dataset_frames_scanned"] = 100
            payload["checkpoint_config_sha256"] = "c" * 64
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "different checkpoint config"):
                load_complete_fixed_manifest(manifest, config)

    def test_canonical_manifest_digest_is_key_order_independent(self) -> None:
        self.assertEqual(
            canonical_json_sha256({"a": 1, "b": [2, 3]}),
            canonical_json_sha256({"b": [2, 3], "a": 1}),
        )


if __name__ == "__main__":
    unittest.main()
