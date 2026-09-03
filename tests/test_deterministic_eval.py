from __future__ import annotations

import json
import sys
import tempfile
import unittest
from importlib import import_module
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
_helpers = import_module("deterministic_eval")
ExecutedActionTrace = _helpers.ExecutedActionTrace
atomic_write_json_new = _helpers.atomic_write_json_new


def _trace(*actions: np.ndarray) -> tuple[str, int]:
    trace = ExecutedActionTrace()
    for action in actions:
        trace.update(action)
    return trace.hexdigest(), trace.count


class DeterministicEvalTest(unittest.TestCase):
    def test_action_trace_is_stable_and_order_sensitive(self) -> None:
        a = np.asarray([[1.0, -0.0]], dtype=np.float32)
        b = np.asarray([[2.0, 3.0]], dtype=np.float32)
        first, count = _trace(a, b)
        repeat, _ = _trace(a.copy(), b.copy())
        reversed_digest, _ = _trace(b, a)
        self.assertEqual(count, 2)
        self.assertEqual(first, repeat)
        self.assertNotEqual(first, reversed_digest)

    def test_action_trace_commits_to_shape_and_dtype(self) -> None:
        values = np.asarray([1.0, 2.0], dtype=np.float32)
        flat, _ = _trace(values)
        reshaped, _ = _trace(values.reshape(1, 2))
        widened, _ = _trace(values.astype(np.float64))
        self.assertEqual(len({flat, reshaped, widened}), 3)

    def test_action_trace_rejects_object_dtype(self) -> None:
        trace = ExecutedActionTrace()
        with self.assertRaisesRegex(TypeError, "object-dtype"):
            trace.update(np.asarray([object()], dtype=object))

    def test_atomic_json_publish_is_complete_and_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            payload = {"schema_version": 2, "values": [1, 2, 3]}
            atomic_write_json_new(path, payload)
            self.assertEqual(json.loads(path.read_text()), payload)
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                atomic_write_json_new(path, {"different": True})
            self.assertEqual(json.loads(path.read_text()), payload)
            self.assertFalse(list(path.parent.glob(".result.json.tmp.*")))

    def test_cuda_workspace_is_set_before_torch_import(self) -> None:
        source = (ROOT / "scripts" / "eval" / "libero_locked_eval_v2.py").read_text()
        workspace = source.index('os.environ["CUBLAS_WORKSPACE_CONFIG"]')
        torch_import = source.index("import torch")
        lerobot_import = source.index("from lerobot")
        configure_call = source.index("\n_configure_determinism()\n", torch_import)
        self.assertLess(workspace, torch_import)
        self.assertLess(torch_import, configure_call)
        self.assertLess(configure_call, lerobot_import)


if __name__ == "__main__":
    unittest.main()
