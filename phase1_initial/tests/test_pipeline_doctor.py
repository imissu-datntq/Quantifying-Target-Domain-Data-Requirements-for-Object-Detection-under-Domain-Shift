from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pipeline_doctor as doctor


class RepositoryCheckTests(unittest.TestCase):
    def test_missing_repository_is_reported_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = doctor.check_repository(
                Path(directory) / "missing",
                expected_sha="abc123",
                required_paths=("train.py",),
            )
        self.assertEqual(result.status, "fail")
        self.assertIn("missing", result.detail.lower())

    def test_text_markers_require_every_expected_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.py"
            path.write_text("alpha\nbeta\n", encoding="utf-8")
            passing = doctor.check_text_markers(path, ("alpha", "beta"))
            failing = doctor.check_text_markers(path, ("alpha", "gamma"))
        self.assertEqual(passing.status, "pass")
        self.assertEqual(failing.status, "fail")
        self.assertIn("gamma", failing.detail)


class EnvironmentCheckTests(unittest.TestCase):
    def test_sf_adapter_markers_do_not_certify_runtime_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / 'sf-yolo'
            (repo / 'utils').mkdir(parents=True)
            (repo / 'train_sf-yolo.py').write_text(
                'labels = np.concatenate(dataset.labels, 0)\n'
                'labels_to_class_weights(dataset.labels, nc)\n'
                'for i, (imgs, targets, paths, _) in pbar\n'
                '--unlabeled-target create_unlabeled_dataloader concatenate_pseudo_labels\n'
            )
            (repo / 'utils/unlabeled_target.py').touch()
            result = next(check for check in doctor.compatibility_checks(root)
                          if check.name == 'compatibility:sf-unlabeled-loader')
            self.assertEqual(result.status, 'pending')

    def test_missing_required_asset_is_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = doctor.check_asset(Path(directory) / "weights.pt", required=True)
        self.assertEqual(result.status, "fail")

    def test_missing_optional_asset_is_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = doctor.check_asset(Path(directory) / "weights.pt", required=False)
        self.assertEqual(result.status, "pending")


class SummaryTests(unittest.TestCase):
    def test_summary_preserves_failure_and_pending_counts(self) -> None:
        results = [
            doctor.Check("a", "pass", "ok"),
            doctor.Check("b", "fail", "bad"),
            doctor.Check("c", "pending", "later"),
        ]
        self.assertEqual(doctor.summarize(results), {"pass": 1, "fail": 1, "pending": 1})


if __name__ == "__main__":
    unittest.main()
