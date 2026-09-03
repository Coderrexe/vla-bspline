from __future__ import annotations

import unittest

from scripts.eval.stochastic_duration_v3 import (
    EFFICACY_PERMUTATIONS,
    aggregate_predicted_pairs,
    block_seeds,
    component_first_differences,
    observation_component_hashes,
    predicted_pair_diagnostic,
    validate_embedded_sha256,
    materialize_task_order_manifest,
)


def _episode(success: bool, action: list[str], durations: list[int]) -> dict:
    steps = len(action)
    return {
        "success": success,
        "steps": steps,
        "initial_raw_observation_sha256": "r",
        "initial_processed_observation_sha256": "p",
        "initial_mujoco_integration_state_sha256": "m",
        "raw_observation_step_sha256": ["r"] * steps,
        "processed_observation_step_sha256": ["p"] * steps,
        "raw_observation_component_step_sha256": {
            "/pixels": ["rp"] * steps,
            "/robot_state": ["rs"] * steps,
        },
        "processed_observation_component_step_sha256": {
            "/observation.images.image": ["pi"] * steps,
            "/observation.state": ["ps"] * steps,
        },
        "mujoco_integration_state_step_sha256": ["m"] * steps,
        "policy_replan_inputs": [{"chunk": index} for index in range(len(durations))],
        "executed_action_trace_sha256": "".join(action),
        "executed_action_step_sha256": action,
        "executed_action_prefix_sha256": action,
        "predicted_durations": durations,
        "n_duration_calls": len(durations),
    }


class StochasticDurationV3Test(unittest.TestCase):
    @staticmethod
    def _digest(value: object) -> str:
        import hashlib

        return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()

    def _manifest(self, blocks: int, order_seed: int = 17) -> dict:
        return materialize_task_order_manifest(
            suite="libero_10",
            task_id=3,
            state_ids=list(range(blocks)),
            repeats=1,
            seed_base=100_000,
            order_seed=order_seed,
        )

    def test_task_local_exact_quotas(self) -> None:
        for blocks, expected in ((6, (1, 1)), (12, (2, 2)), (10, (1, 2)), (50, (8, 9))):
            counts = list(self._manifest(blocks)["permutation_counts"].values())
            self.assertEqual((min(counts), max(counts)), expected)
            self.assertEqual(sum(counts), blocks)

    def test_engineering_one_state_six_repeats_covers_every_order(self) -> None:
        manifest = materialize_task_order_manifest(
            suite="libero_10",
            task_id=2,
            state_ids=[0],
            repeats=6,
            seed_base=100_000,
            order_seed=20_260_821,
        )
        self.assertEqual(set(manifest["permutation_counts"].values()), {1})

    def test_manifest_is_seeded_randomized_and_checkpoint_independent(self) -> None:
        first = self._manifest(12, order_seed=17)
        repeat = self._manifest(12, order_seed=17)
        changed = self._manifest(12, order_seed=18)
        self.assertEqual(first, repeat)
        self.assertNotEqual(first["blocks"], changed["blocks"])
        self.assertEqual(
            {tuple(block["efficacy_permutation"]) for block in first["blocks"]},
            set(EFFICACY_PERMUTATIONS),
        )
        self.assertNotIn("checkpoint", first)
        self.assertEqual(len(first["manifest_sha256"]), 64)
        validate_embedded_sha256(first)
        corrupted = dict(first)
        corrupted["order_seed"] += 1
        with self.assertRaises(ValueError):
            validate_embedded_sha256(corrupted)

    def test_env_and_policy_seeds_have_declared_widths(self) -> None:
        env_seed, policy_seed = block_seeds(100_000, 3, 9, 2)
        self.assertGreaterEqual(env_seed, 0)
        self.assertLessEqual(env_seed, 0xFFFFFFFF)
        self.assertGreaterEqual(policy_seed, 0)
        self.assertLessEqual(policy_seed, 0xFFFFFFFFFFFFFFFF)
        self.assertEqual((env_seed, policy_seed), block_seeds(100_000, 3, 9, 2))

    def test_pairwise_repeatability_localizes_first_divergence(self) -> None:
        left = _episode(True, ["a", "b", "c"], [8, 12, 24])
        right = _episode(False, ["a", "b", "x", "y"], [8, 12, 20])
        right["raw_observation_component_step_sha256"]["/pixels"] = [
            "rp", "camera-drift", "camera-drift", "camera-drift"
        ]
        right["processed_observation_component_step_sha256"][
            "/observation.images.image"
        ] = ["pi", "pi", "processed-camera-drift", "processed-camera-drift"]
        diagnostic = predicted_pair_diagnostic(left, right)
        self.assertFalse(diagnostic["outcome_agreement"])
        self.assertEqual(diagnostic["outcome_pair"], "10")
        self.assertEqual(diagnostic["first_different_action_step"], 2)
        self.assertEqual(diagnostic["first_different_predicted_duration_chunk"], 2)
        self.assertEqual(
            diagnostic["first_different_raw_observation_step_by_component"][
                "/pixels"
            ],
            1,
        )
        self.assertEqual(
            diagnostic["first_different_raw_observation_step_by_component"][
                "/robot_state"
            ],
            3,
        )
        self.assertEqual(
            diagnostic[
                "first_different_processed_observation_step_by_component"
            ]["/observation.images.image"],
            2,
        )
        aggregate = aggregate_predicted_pairs([diagnostic])
        self.assertEqual(aggregate["outcome_pairs"]["10"], 1)
        self.assertEqual(aggregate["first_action_divergence_step"]["median"], 2.0)

    def test_component_hashes_include_nested_camera_and_state_groups(self) -> None:
        observation = {
            "pixels": {"agent": b"a", "wrist": b"w"},
            "robot_state": {"eef": {"pos": (1, 2, 3)}, "gripper": 0},
        }
        components = observation_component_hashes(observation, self._digest)
        self.assertEqual(
            set(components),
            {
                "/pixels",
                "/pixels/agent",
                "/pixels/wrist",
                "/robot_state",
                "/robot_state/eef",
                "/robot_state/eef/pos",
                "/robot_state/gripper",
            },
        )
        self.assertTrue(all(len(digest) == 64 for digest in components.values()))

    def test_component_difference_marks_schema_drift_at_initial_step(self) -> None:
        differences = component_first_differences(
            {"/pixels": ["a", "b"]},
            {"/pixels": ["a", "x"], "/state": ["s", "s"]},
        )
        self.assertEqual(differences, {"/pixels": 1, "/state": 0})


if __name__ == "__main__":
    unittest.main()
