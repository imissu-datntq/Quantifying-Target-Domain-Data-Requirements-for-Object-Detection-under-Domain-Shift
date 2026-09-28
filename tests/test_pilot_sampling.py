import csv
import tempfile
import unittest
from pathlib import Path

from pilot_sampling import read_manifest_sample, read_preparation_counts


class PilotSamplingTests(unittest.TestCase):
    def write_manifest(self, root: Path, rows: list[dict]) -> Path:
        path = root / "manifest.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["name", "image", "coco_image_id"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_selects_exact_deterministic_prefix(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.write_manifest(root, [
                dict(name="c.jpg", image="images/train/c.jpg", coco_image_id=3),
                dict(name="a.jpg", image="images/train/a.jpg", coco_image_id=1),
                dict(name="b.jpg", image="images/train/b.jpg", coco_image_id=2),
            ])
            rows = read_manifest_sample(manifest, count=2, split="train")
            self.assertEqual([row["name"] for row in rows], ["a.jpg", "b.jpg"])

    def test_rejects_invalid_count_membership_and_split_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            valid = self.write_manifest(root, [
                dict(name="a.jpg", image="images/train/a.jpg", coco_image_id=1),
            ])
            with self.assertRaises(ValueError):
                read_manifest_sample(valid, count=0, split="train")
            with self.assertRaises(ValueError):
                read_manifest_sample(valid, count=2, split="train")
            duplicate = self.write_manifest(root, [
                dict(name="a.jpg", image="images/train/a.jpg", coco_image_id=1),
                dict(name="a.jpg", image="images/train/a.jpg", coco_image_id=1),
            ])
            with self.assertRaises(ValueError):
                read_manifest_sample(duplicate, count=2, split="train")
            wrong_split = self.write_manifest(root, [
                dict(name="a.jpg", image="images/val/a.jpg", coco_image_id=1),
            ])
            with self.assertRaises(ValueError):
                read_manifest_sample(wrong_split, count=1, split="train")

    def test_reads_exact_training_counts_from_preparation_sidecar(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data.yaml"
            data.write_text("{}")
            (root / "preparation.json").write_text(
                '{"source_train_images": 1000, "target_train_images": 1000}')
            self.assertEqual(read_preparation_counts(data), (1000, 1000))
            (root / "preparation.json").write_text(
                '{"source_train_images": 1000, "target_train_images": 0}')
            with self.assertRaises(ValueError):
                read_preparation_counts(data)


if __name__ == "__main__":
    unittest.main()
