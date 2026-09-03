from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "scripts" / "eval" / "libero_renderer_probe.py"


class LiberoRendererProbeContractTest(unittest.TestCase):
    def test_backend_is_explicit_and_replicate_does_not_change_seed(self) -> None:
        source = COLLECTOR.read_text()
        self.assertIn('backend not in {"egl", "osmesa"}', source)
        self.assertIn("_episode_seed(args.seed_base, task_id, state_id)", source)
        self.assertNotIn("args.replicate_index +", source)

    def test_records_distribution_and_closed_loop_endpoints(self) -> None:
        source = COLLECTOR.read_text()
        for field in (
            "raw_cameras",
            "processed_cameras",
            "initial_mujoco_integration_state_sha256",
            "compiled_model_xml_sha256",
            "first_action_sha256",
            "executed_action_trace_sha256",
            "success",
            "opengl",
            "arrays_sha256",
        ):
            self.assertIn(field, source)

    def test_one_npz_sidecar_and_no_overwrite_publication(self) -> None:
        source = COLLECTOR.read_text()
        self.assertIn("np.savez_compressed", source)
        self.assertIn("os.link(temporary, path)", source)
        self.assertIn("refusing to overwrite", source)


if __name__ == "__main__":
    unittest.main()
