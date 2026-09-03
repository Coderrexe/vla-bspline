from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from renderer_comparability import analyze, array_metrics  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RendererComparabilityTest(unittest.TestCase):
    def make_probe(
        self,
        root: Path,
        backend: str,
        replicate: int,
        *,
        raw_offset: int = 0,
        model_hash: str = "model",
    ) -> Path:
        arrays: dict[str, np.ndarray] = {}
        episodes = []
        for state_id in (0, 1):
            raw = np.full((1, 8, 8, 3), 80 + state_id, dtype=np.uint8)
            if raw_offset:
                raw = raw.copy()
                raw[:, 0, 0, 0] = np.clip(
                    raw[:, 0, 0, 0].astype(int) + raw_offset, 0, 255
                )
            processed = np.moveaxis(raw.astype(np.float32) / 255.0, -1, 1)
            raw_key = f"raw_{state_id}"
            processed_key = f"processed_{state_id}"
            arrays[raw_key] = raw
            arrays[processed_key] = processed
            episodes.append(
                {
                    "task_id": 0,
                    "task_description": "test",
                    "state_id": state_id,
                    "env_seed_u32": 100000 + state_id,
                    "policy_seed_u64": 100000 + state_id,
                    "initial_raw_observation_sha256": "raw",
                    "initial_processed_observation_sha256": "processed",
                    "initial_mujoco_integration_state_sha256": f"state-{state_id}",
                    "compiled_model_xml_sha256": "xml",
                    "raw_cameras": {
                        "/pixels/image": {
                            "array_key": raw_key,
                            "shape": list(raw.shape),
                            "dtype": str(raw.dtype),
                            "sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                        }
                    },
                    "processed_cameras": {
                        "/observation.images.image": {
                            "array_key": processed_key,
                            "shape": list(processed.shape),
                            "dtype": str(processed.dtype),
                            "sha256": hashlib.sha256(processed.tobytes()).hexdigest(),
                        }
                    },
                    "success": state_id == 0,
                    "steps": 10,
                    "first_action_sha256": f"first-{state_id}",
                    "executed_action_trace_sha256": f"trace-{state_id}",
                }
            )
        arrays_path = root / f"{backend}_{replicate}.npz"
        np.savez_compressed(arrays_path, **arrays)
        payload = {
            "schema_version": 1,
            "protocol": "libero_renderer_comparability_probe_v1",
            "renderer_backend": backend,
            "replicate_index": replicate,
            "checkpoint_config_sha256": "config",
            "model_sha256": model_hash,
            "suite": "libero_10",
            "task_ids": [0],
            "state_ids": [0, 1],
            "seed_base": 100000,
            "control_frequency_hz": 20,
            "episodes": episodes,
            "arrays_file": str(arrays_path),
            "arrays_sha256": sha256(arrays_path),
            "opengl": {"renderer": backend},
            "runtime": {
                "python_version": "3.12.0",
                "software_versions": {
                    "mujoco": "3.3",
                    "robosuite": "1.5",
                    "libero": "0.1",
                    "lerobot": "0.4",
                },
                "determinism": {
                    "numpy_version": "2.0",
                    "torch_version": "2.7",
                    "torch_cuda_version": "12.8",
                    "cudnn_version": 91000,
                    "cuda_devices": [
                        {
                            "name": "L40S",
                            "capability": [8, 9],
                            "total_memory_bytes": 1,
                        }
                    ],
                },
            },
            "source": {
                "collector_sha256": "collector",
                "locked_reset_helper_sha256": "reset",
                "structured_hash_helper_sha256": "structured",
                "hidden_state_helper_sha256": "hidden",
                "policy_source_sha256": "policy-source",
                "env_source_sha256": "env-source",
                "stats_sha256": "stats",
                "lerobot_commit": "commit",
            },
        }
        path = root / f"{backend}_{replicate}.json"
        path.write_text(json.dumps(payload))
        return path

    def test_metrics_exact(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        metrics = array_metrics(image, image.copy(), raw=True)
        self.assertEqual(metrics["mae"], 0)
        self.assertEqual(metrics["exact_fraction"], 1)
        self.assertEqual(metrics["psnr_db"], 999)

    def test_matching_probe_set_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                self.make_probe(root, "egl", 0),
                self.make_probe(root, "egl", 1, raw_offset=1),
                self.make_probe(root, "osmesa", 0),
                self.make_probe(root, "osmesa", 1),
            ]
            result = analyze(paths)
            self.assertEqual(result["decision"], "PASS_EQUIVALENCE_GATE")
            self.assertTrue(
                result["gates"]["osmesa_repeat_arrays_actions_outcomes_exact"]
            )
            self.assertIn(
                "/pixels/image",
                result["metric_summaries"]["cross_backend"]["raw_camera"][
                    "by_camera"
                ],
            )

    def test_large_renderer_shift_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                self.make_probe(root, "egl", 0),
                self.make_probe(root, "egl", 1),
                self.make_probe(root, "osmesa", 0, raw_offset=100),
                self.make_probe(root, "osmesa", 1, raw_offset=100),
            ]
            result = analyze(paths)
            self.assertEqual(
                result["decision"], "FAIL_EQUIVALENCE_GATE_DO_NOT_MIX_BACKENDS"
            )
            self.assertFalse(result["gates"]["raw_pixel_absolute"])

    def test_identity_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                self.make_probe(root, "egl", 0),
                self.make_probe(root, "egl", 1),
                self.make_probe(root, "osmesa", 0),
                self.make_probe(root, "osmesa", 1, model_hash="other"),
            ]
            with self.assertRaisesRegex(ValueError, "identities differ"):
                analyze(paths)


if __name__ == "__main__":
    unittest.main()
