from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import bdd100k_research_split as split


class LargestRemainderTests(unittest.TestCase):
    def test_exact_total_and_lexicographic_tie_break(self) -> None:
        result = split.largest_remainder({("b",): 1, ("a",): 1, ("c",): 1}, 2)
        self.assertEqual(result, {("b",): 1, ("a",): 1, ("c",): 0})
        self.assertEqual(sum(result.values()), 2)

    def test_rejects_budget_overflow(self) -> None:
        with self.assertRaisesRegex(split.SplitError, "population"):
            split.largest_remainder({"only": 2}, 3)

    def test_published_cell_quota_rejects_infeasible_census(self) -> None:
        counts = {
            cell: train + test
            for cell, (train, test, _val) in split.PUBLISHED_CELL_QUOTAS.items()
        }
        counts[("night", "clear")] -= 1
        with self.assertRaisesRegex(split.SplitError, "Published cell"):
            split.build_cell_train_quotas(counts)


class StreamingJsonTests(unittest.TestCase):
    def test_streams_nested_records(self) -> None:
        records = [{"name": "a.jpg", "labels": [{"box2d": {"x1": 0}}]}, {"name": "b.jpg"}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.json"
            path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            self.assertEqual(list(split.json_array_items(path, chunk_size=7)), records)

    def test_duplicate_annotation_records_are_rejected(self) -> None:
        records = [
            {"name": "same.jpg", "attributes": {}, "labels": []},
            {"name": "same.jpg", "attributes": {}, "labels": []},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation_path = root / "records.json"
            annotation_path.write_text(json.dumps(records), encoding="utf-8")
            image_path = root / "same.jpg"
            # A minimal JPEG is unnecessary: duplicate detection occurs first
            # on the second record, after the first record has been audited.
            # Patch the dimension reader to keep this fixture focused.
            original = split.jpeg_dimensions
            split.jpeg_dimensions = lambda _path: (1280, 720)
            try:
                with self.assertRaisesRegex(split.SplitError, "Duplicate annotation"):
                    split.audit_annotations(
                        annotation_path,
                        {"same.jpg": image_path},
                        decode_images=False,
                        expected_count=2,
                    )
            finally:
                split.jpeg_dimensions = original


class ImageIndexTests(unittest.TestCase):
    def test_nested_index_rejects_duplicate_basenames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one").mkdir()
            (root / "two").mkdir()
            (root / "one" / "same.jpg").write_bytes(b"x")
            (root / "two" / "same.jpg").write_bytes(b"y")
            with self.assertRaisesRegex(split.SplitError, "Duplicate image basenames"):
                split.index_images(root)

    def test_missing_annotation_image_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.json"
            path.write_text(
                json.dumps([{"name": "missing.jpg", "attributes": {}, "labels": []}]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(split.SplitError, "no image"):
                split.audit_annotations(path, {}, decode_images=False, expected_count=1)

    def test_symlink_view_uses_a_relative_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw" / "image.jpg"
            destination = root / "view" / "image.jpg"
            source.parent.mkdir()
            source.write_bytes(b"jpeg-placeholder")
            self.assertEqual(split.create_image_link(source, destination, "symlink"), "symlink")
            self.assertTrue(destination.is_symlink())
            self.assertFalse(Path(destination.readlink()).is_absolute())
            self.assertEqual(destination.read_bytes(), b"jpeg-placeholder")


class DeterminismTests(unittest.TestCase):
    def test_stratum_seed_is_stable_and_distinct(self) -> None:
        first = split.stratum_seed(20260916, ("daytime", "clear", "city street"))
        second = split.stratum_seed(20260916, ("daytime", "clear", "city street"))
        other = split.stratum_seed(20260916, ("night", "clear", "city street"))
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)


if __name__ == "__main__":
    unittest.main()
