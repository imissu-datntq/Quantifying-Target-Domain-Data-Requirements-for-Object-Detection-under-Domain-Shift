"""Checks for split semantics, strict parsing and leakage rejection."""
import csv
import json
import math
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_revised_bdd100k as protocol
from eda_revised_bdd100k import probability_absent


class ProtocolTests(unittest.TestCase):
    def test_stratified_split_is_order_independent_and_exhaustive(self):
        rows = [{"name": f"image_{i:06d}.jpg", "timeofday": ("night", "daytime")[i % 2], "weather": ("clear", "rainy", "undefined")[i % 3], "scene": f"scene_{i % 7}"} for i in range(69863)]
        first, strata = protocol.split_membership(rows, 20260916)
        reordered, _ = protocol.split_membership(list(reversed(rows)), 20260916)
        other_seed, _ = protocol.split_membership(rows, 20260917)
        self.assertEqual(first, reordered)
        self.assertNotEqual(first, other_seed)
        self.assertEqual(Counter(first.values()), {"train": 55000, "test": 14863})
        self.assertEqual(len(first), len(rows))
        self.assertLess(max(abs(row["rounding_error"]) for row in strata), 2)

    def test_apportion_rare_groups_and_fixed_total(self):
        quotas = protocol.apportion({"singleton": 1, "rare": 2, "large": 1000}, 789)
        self.assertEqual(sum(quotas.values()), 789)
        self.assertTrue(0 <= quotas["singleton"] <= 1)
        self.assertTrue(0 <= quotas["rare"] <= 2)

    def test_content_repair_keeps_stratum_counts_and_duplicate_groups(self):
        rows = [{"name": name, "timeofday": "night", "weather": "clear", "scene": "city street"} for name in ("a", "b", "c", "d")]
        membership = {"a": "train", "b": "test", "c": "train", "d": "test"}
        hashes = {"a": "duplicate", "b": "duplicate", "c": "unique_c", "d": "unique_d"}
        repairs = protocol.keep_content_groups_together(membership, rows, hashes)
        self.assertEqual(membership["a"], membership["b"])
        self.assertEqual(Counter(membership.values()), {"train": 2, "test": 2})
        self.assertEqual(len(repairs), 1)

    def test_json_reader_rejects_truncation_and_missing_separators(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            for text in ('[{"name":"x"}', '[{"name":"x"}{"name":"y"}]', '[{"name":"x"}]garbage'):
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    list(protocol.stream_records(path))
            path.write_text('[{"name":"x"}, {"name":"y"}]', encoding="utf-8")
            self.assertEqual([r["name"] for r in protocol.stream_records(path)], ["x", "y"])

    def test_hypergeometric_absence_and_edge_cases(self):
        expected = math.comb(17, 5) / math.comb(20, 5)
        self.assertAlmostEqual(probability_absent(20, 3, 5), expected)
        self.assertEqual(probability_absent(20, 0, 20), 1)
        self.assertEqual(probability_absent(20, 3, 20), 0)

    def test_validator_rejects_content_leakage_and_missing_images(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(protocol, "EXPECTED", {"train": 1, "test": 1, "val": 1}):
            root = Path(directory)
            (root / "manifests").mkdir()
            (root / "annotations").mkdir()
            for split in protocol.EXPECTED:
                images = root / "images" / split
                images.mkdir(parents=True)
                name = f"{split}.jpg"
                (images / name).write_bytes(b"fixture")
                (root / "annotations" / f"bdd100k_labels_images_{split}.json").write_text(json.dumps([{"name": name}]), encoding="utf-8")
                with (root / "manifests" / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["name", "sha256"])
                    writer.writeheader()
                    writer.writerow({"name": name, "sha256": split})
            self.assertTrue(protocol.validate(root)["passed"])
            manifest = root / "manifests" / "test.csv"
            original = manifest.read_text(encoding="utf-8")
            manifest.write_text(original.replace("test.jpg,test", "test.jpg,train"), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "Exact content leakage"):
                protocol.validate(root)
            manifest.write_text(original, encoding="utf-8")
            (root / "images" / "test" / "test.jpg").unlink()
            with self.assertRaises(AssertionError):
                protocol.validate(root)


if __name__ == "__main__":
    unittest.main()
