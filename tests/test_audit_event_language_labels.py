from __future__ import annotations

import unittest

from scripts.analysis.audit_event_language_labels import audit
from scripts.data.molmo_segment_labels import BOUNDARY_VERSION, PHASE_BANKS, PROMPT_VERSION


TASK = "put both the alphabet soup and the cream cheese box in the basket"


def row(episode: int, segment: int, phase: int) -> dict:
    return {
        "episode_index": episode,
        "seg_start": segment * 8,
        "seg_end": (segment + 1) * 8,
        "task": TASK,
        "label": PHASE_BANKS[TASK][phase],
        "phase_id": f"P{phase:02d}",
        "boundary_version": BOUNDARY_VERSION,
        "prompt_version": PROMPT_VERSION,
        "label_mode": "phase_bank",
    }


class EventLanguageAuditTest(unittest.TestCase):
    def test_counts_temporal_inversions(self):
        report = audit([row(1, index, phase) for index, phase in enumerate([0, 1, 3, 2, 8])])
        self.assertEqual(report["adjacent_pairs"], 4)
        self.assertEqual(report["adjacent_inversions"], 1)
        self.assertEqual(report["gross_adjacent_inversions"], 0)

    def test_rejects_phase_label_mismatch(self):
        bad = row(1, 0, 2)
        bad["label"] = "wrong"
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            audit([bad])


if __name__ == "__main__":
    unittest.main()
