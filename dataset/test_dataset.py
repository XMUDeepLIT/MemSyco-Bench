from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from dataset import iter_dataset, load_dataset, load_manifest, task_names


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REQUIRED_FIELDS = {"id", "task", "dialogue", "question", "memory", "evaluation", "metadata"}


class DatasetReleaseTest(unittest.TestCase):
    def test_manifest_counts_and_hashes(self) -> None:
        manifest = load_manifest()
        self.assertEqual(manifest["schema_version"], "1.0")
        self.assertEqual(manifest["total_samples"], 1550)
        self.assertEqual(tuple(manifest["tasks"]), task_names())

        total = 0
        for task, spec in manifest["tasks"].items():
            path = DATA_DIR / spec["file"]
            rows = load_dataset(task)
            self.assertEqual(len(rows), spec["samples"])
            self.assertTrue(all(row["task"] == task for row in rows))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, spec["sha256"])
            total += len(rows)
        self.assertEqual(total, manifest["total_samples"])

    def test_common_shape_and_unique_ids(self) -> None:
        ids = set()
        for row in iter_dataset():
            self.assertEqual(set(row), REQUIRED_FIELDS)
            self.assertNotIn(row["id"], ids)
            ids.add(row["id"])
            self.assertTrue(row["dialogue"])
            self.assertTrue(row["question"].strip())
            self.assertTrue(row["memory"]["items"])
            self.assertTrue(row["evaluation"]["reference_answer"].strip())
            for message in row["dialogue"]:
                self.assertIn(message["role"], {"user", "assistant"})
                self.assertFalse(message["content"].startswith(("User:", "Assistant:")))

    def test_schema_is_valid_json(self) -> None:
        schema = json.loads((DATA_DIR / "schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["type"], "object")
        self.assertEqual(set(schema["required"]), REQUIRED_FIELDS)


if __name__ == "__main__":
    unittest.main()
