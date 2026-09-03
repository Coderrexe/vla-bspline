from __future__ import annotations

import unittest

from scripts.analysis.audit_language_tokenizer import audit_pairs


class FakeTokenizer:
    def __call__(self, text, *, padding, add_special_tokens, truncation, max_length=None):
        ids = [1] + [ord(char) for char in text] + [2]
        if truncation:
            ids = ids[:max_length]
        return {"input_ids": ids}


class TokenizerAuditTest(unittest.TestCase):
    def test_detects_post_truncation_collision_and_distinction(self):
        pairs = [
            {"task_index": 40, "original_label": "abcX", "shuffled_label": "abcY"},
            {"task_index": 41, "original_label": "cat", "shuffled_label": "dog"},
        ]
        result = audit_pairs(pairs, FakeTokenizer(), max_length=4)
        self.assertEqual(result["pair_count"], 2)
        self.assertEqual(result["token_id_collision_count"], 1)
        self.assertEqual(result["token_id_collision_task_indices"], [40])
        self.assertEqual(result["pairs_with_any_truncation"], 2)


if __name__ == "__main__":
    unittest.main()
