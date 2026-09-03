"""Focused tests for checkpoint-contained spline normalization.

These contract tests intentionally use only the Python standard library, so
they can run on a login/development machine without importing LeRobot/PyTorch.
The production event-target tensor tests remain in ``test_event_targets.py``.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "policy" / "smolvla_spline" / "stats_contract.py"
SPEC = importlib.util.spec_from_file_location("spline_stats_contract", MODULE_PATH)
stats_contract = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(stats_contract)


def config(**overrides):
    values = {
        "n_ctrl": 6,
        "spline_degree": 3,
        "predict_duration": True,
        "horizon": 20,
        "horizon_max": 24,
        "min_seg": 6,
        "pause_frac": 0.15,
        "action_layout": "eef7",
        "fit_end_weight": None,
        "speed_aug": None,
        "spline_stats_file": "v1.json",
        "spline_stats_file_v2": "v2.json",
        "embedded_spline_stats": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def v2_stats(**overrides):
    values = {
        "n_ctrl": 6,
        "degree": 3,
        "horizon_max": 24,
        "min_seg": 6,
        "pause_frac": 0.15,
        "action_layout": "eef7",
        "pose_dims": list(range(6)),
        "grip_idx": 6,
        "pass_dims": [],
        "boundary_semantics": stats_contract.EVENT_BOUNDARY_SEMANTICS,
        "pose_ctrl_mean": [[float(row + col) for col in range(6)] for row in range(6)],
        "pose_ctrl_std": [[1.0 + 0.1 * col for col in range(6)] for _ in range(6)],
        "grip_ctrl_mean": [0.0] * 6,
        "grip_ctrl_std": [1.0] * 6,
        "logT_mean": 3.0,
        "logT_std": 0.5,
    }
    values.update(overrides)
    return values


class SplineStatsContractTest(unittest.TestCase):
    def test_repository_stats_are_explicit_and_match_their_configs(self):
        cases = (
            ("spline_stats_libero.json", config(predict_duration=False)),
            ("spline_stats_libero_v2.json", config(horizon_max=40)),
            ("spline_stats_libero_v2_h24.json", config(horizon_max=24)),
        )
        stats_dir = ROOT / "policy" / "smolvla_spline"
        for name, cfg in cases:
            with self.subTest(name=name), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                payload = json.loads((stats_dir / name).read_text(encoding="utf-8"))
                stats_contract.validate_spline_stats(payload, cfg, name)
                self.assertEqual(caught, [])

    def test_valid_v2_contract_is_canonicalized(self):
        result = stats_contract.validate_spline_stats(v2_stats(), config(), "fixture")
        self.assertEqual(result["horizon_max"], 24)
        self.assertEqual(result["boundary_semantics"], stats_contract.EVENT_BOUNDARY_SEMANTICS)
        self.assertEqual(result["stats_contract_version"], stats_contract.STATS_CONTRACT_VERSION)

    def test_rejects_head_metadata_mismatches(self):
        cases = {
            "horizon_max": {"horizon_max": 40},
            "min_seg": {"min_seg": 8},
            "action_layout": {"action_layout": "robocasa12"},
            "boundary_semantics": {"boundary_semantics": "event_at_k_inclusive"},
            "stats_contract_version": {"stats_contract_version": 999},
        }
        for field, override in cases.items():
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                stats_contract.validate_spline_stats(v2_stats(**override), config(), "fixture")

    def test_rejects_weighted_fit_stats_for_unweighted_head(self):
        with self.assertRaisesRegex(ValueError, "fit_end_weight"):
            stats_contract.validate_spline_stats(
                v2_stats(fit_end_weight=9.0), config(), "weighted.json"
            )

    def test_speed_aug_stats_must_match_training_config(self):
        with self.assertRaisesRegex(ValueError, "speed_aug"):
            stats_contract.validate_spline_stats(
                v2_stats(speed_aug=[1.0, 2.0]), config(), "augmented.json"
            )
        with self.assertRaisesRegex(ValueError, "speed_aug"):
            stats_contract.validate_spline_stats(
                v2_stats(), config(speed_aug=[1.0, 2.0]), "base.json"
            )
        result = stats_contract.validate_spline_stats(
            v2_stats(speed_aug=[1.0, 2.0]),
            config(speed_aug=[1.0, 2.0]),
            "augmented.json",
        )
        self.assertEqual(result["speed_aug"], [1.0, 2.0])

    def test_rejects_conflicting_horizon_aliases(self):
        payload = v2_stats(h_max=40)
        with self.assertRaisesRegex(ValueError, "conflicting horizon"):
            stats_contract.validate_spline_stats(payload, config(), "fixture")

    def test_legacy_metadata_is_inferred_with_warning(self):
        payload = v2_stats()
        for key in (
            "degree",
            "action_layout",
            "pose_dims",
            "grip_idx",
            "pass_dims",
            "boundary_semantics",
        ):
            payload.pop(key)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = stats_contract.validate_spline_stats(payload, config(), "legacy.json")
        self.assertGreaterEqual(len(caught), 2)
        self.assertEqual(result["degree"], 3)
        self.assertEqual(result["action_layout"], "eef7")
        self.assertEqual(result["boundary_semantics"], stats_contract.EVENT_BOUNDARY_SEMANTICS)

    def test_first_resolution_embeds_exact_payload_and_future_load_ignores_json(self):
        cfg = config()
        original = v2_stats()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v2.json"
            path.write_text(json.dumps(original), encoding="utf-8")
            resolved, origin = stats_contract.resolve_spline_stats(cfg, directory)
            self.assertEqual(origin, str(path))
            self.assertEqual(cfg.embedded_spline_stats, resolved)

            # Simulate mutation/deletion of the package JSON after saving. A
            # checkpoint config carrying the embedded payload remains exact.
            path.unlink()
            checkpoint_cfg = config(embedded_spline_stats=copy.deepcopy(cfg.embedded_spline_stats))
            reloaded, reloaded_origin = stats_contract.resolve_spline_stats(
                checkpoint_cfg, directory
            )
            self.assertEqual(reloaded_origin, "checkpoint-embedded config")
            self.assertEqual(reloaded, resolved)

    def test_embedded_payload_is_revalidated_against_modified_config(self):
        payload = v2_stats()
        cfg = config(horizon_max=40, embedded_spline_stats=payload)
        with self.assertRaisesRegex(ValueError, "horizon_max"):
            stats_contract.resolve_spline_stats(cfg, "/does/not/matter")

    def test_legacy_state_dict_exception_requires_both_buffers_absent(self):
        missing = ["policy._tgt_mean", "policy._tgt_std", "policy.other"]
        consumed = stats_contract.consume_legacy_normalization_missing_keys(
            {"policy.weight": object()}, "policy.", missing
        )
        self.assertTrue(consumed)
        self.assertEqual(missing, ["policy.other"])

        partial_missing = ["policy._tgt_std"]
        consumed = stats_contract.consume_legacy_normalization_missing_keys(
            {"policy._tgt_mean": object()}, "policy.", partial_missing
        )
        self.assertFalse(consumed)
        self.assertEqual(partial_missing, ["policy._tgt_std"])

    def test_model_registers_both_target_buffers_as_persistent(self):
        model_path = ROOT / "policy" / "smolvla_spline" / "modeling_smolvla_spline.py"
        tree = ast.parse(model_path.read_text(encoding="utf-8"))
        persistence = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "register_buffer" or not node.args:
                continue
            if isinstance(node.args[0], ast.Constant) and node.args[0].value in {
                "_tgt_mean",
                "_tgt_std",
            }:
                persistent = next(
                    (keyword.value for keyword in node.keywords if keyword.arg == "persistent"),
                    None,
                )
                persistence[node.args[0].value] = (
                    isinstance(persistent, ast.Constant) and persistent.value is True
                )
        self.assertEqual(persistence, {"_tgt_mean": True, "_tgt_std": True})


if __name__ == "__main__":
    unittest.main()
