"""Campaign grid, label-free prediction, schema-3 scoring and recovery uncertainty."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate_frcnn  # noqa: E402
import freeze_evaluation_plan  # noqa: E402
from bdd_pipeline.campaign import POLICY, grid, require_locked, train_arguments  # noqa: E402
from bdd_pipeline.data import BDDImages, CLASSES, EXPECTED_DOMAINS  # noqa: E402
from bdd_pipeline.evaluation import bootstrap_class_ap, coco_evaluator, fast_bootstrap_class_ap  # noqa: E402
from bdd_pipeline.model import MODEL_CONTRACT  # noqa: E402
from bdd_pipeline.recovery import (band, bootstrap_recovery, fit_power_curve, ladder_bracket, power_threshold,  # noqa: E402
                                   threshold_interval)

LOCKED = dict(POLICY, floor_steps=500, epochs=5, lr=0.001)


def synthetic_detection(seed: int, images: int = 12):
    rng = np.random.default_rng(seed)
    image_rows = [{"id": i, "file_name": f"{i}.jpg", "width": 1280, "height": 720} for i in range(1, images + 1)]
    annotations, predictions = [], []
    for image in image_rows:
        for _ in range(rng.integers(0, 5)):
            x, y, w, h = rng.uniform(0, 600), rng.uniform(0, 300), rng.uniform(8, 200), rng.uniform(8, 200)
            category = int(rng.choice([1, 3, 9, 10]))
            annotations.append({"id": len(annotations) + 1, "image_id": image["id"], "category_id": category,
                                "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            if rng.random() < 0.8:
                predictions.append({"image_id": image["id"], "category_id": category, "score": float(rng.random()),
                                    "bbox": [x + rng.normal(0, 4), y + rng.normal(0, 4), w, h]})
        for _ in range(2):
            predictions.append({"image_id": image["id"], "category_id": int(rng.choice([1, 3, 9, 10])), "score": float(rng.random()),
                                "bbox": [rng.uniform(0, 900), rng.uniform(0, 500), 30.0, 30.0]})
    gt = {"info": {}, "images": image_rows, "annotations": annotations,
          "categories": [{"id": i + 1, "name": name} for i, name in enumerate(CLASSES)]}
    return gt, predictions


class BootstrapTests(unittest.TestCase):
    def test_fast_bootstrap_matches_exact_resamples(self):
        for seed in (3, 11):
            gt, predictions = synthetic_detection(seed)
            evaluator = coco_evaluator(gt, predictions)
            exact = bootstrap_class_ap(evaluator, 25, seed=7)
            fast = fast_bootstrap_class_ap(evaluator, 25, seed=7)
            self.assertTrue(np.allclose(np.nan_to_num(exact, nan=-1), np.nan_to_num(fast, nan=-1)))


class RecoveryUncertaintyTests(unittest.TestCase):
    def test_bootstrap_recovery_fixes_ceiling_and_matches_noise_free_point(self):
        replicates = 50
        zero = np.full(replicates, 0.20)
        ceilings = [np.full(replicates, 0.30), np.full(replicates, 0.30)]
        runs = {10: [np.full(replicates, 0.22)] * 3, 100: [np.full(replicates, 0.28)] * 3}
        budgets, curves = bootstrap_recovery(zero, ceilings, runs, 1000, seed=1)
        self.assertEqual(budgets, [10, 100, 1000])
        self.assertTrue(np.allclose(curves, [[0.2, 0.8, 1.0]] * replicates))
        _, tiny = bootstrap_recovery(zero, [np.full(replicates, 0.205)], runs, 1000, min_denominator=0.01)
        self.assertTrue(np.isnan(tiny).all())

    def test_bootstrap_recovery_spreads_with_seed_noise(self):
        rng = np.random.default_rng(0)
        zero = 0.20 + rng.normal(0, 0.002, 200)
        ceilings = [0.30 + rng.normal(0, 0.002, 200) for _ in range(2)]
        runs = {10: [0.22 + s + rng.normal(0, 0.002, 200) for s in (-0.02, 0.0, 0.02)]}
        _, curves = bootstrap_recovery(zero, ceilings, runs, 1000, seed=2)
        lower, upper = band(curves)
        self.assertLess(lower[0], 0.2)
        self.assertGreater(upper[0], 0.2)
        self.assertEqual((lower[1], upper[1]), (1.0, 1.0))

    def test_threshold_interval_and_ladder_bracket(self):
        budgets = [10, 100, 1000, 2000]
        lower, upper = np.array([0.1, 0.6, 0.85, 1.0]), np.array([0.4, 0.92, 0.99, 1.0])
        self.assertEqual(threshold_interval(budgets, lower, upper, 0.90, strict=False), {"from": 100, "to": 2000})
        curve = {10: 0.2, 100: 0.8, 1000: 0.95, 2000: 1.0}
        self.assertEqual(ladder_bracket(curve, 0.90, strict=False), {"last_failing": 100, "first_passing": 1000})
        self.assertEqual(ladder_bracket(curve, 0.10, strict=False), {"last_failing": None, "first_passing": 10})

    def test_power_fit_recovers_known_curve_and_inverts(self):
        full, a, b = 2000, 3.0, 0.5
        budgets = [10, 25, 50, 100, 250, 500, 1000]
        values = [1 - a * (n ** -b - full ** -b) for n in budgets]
        fit = fit_power_curve(budgets, values, full)
        self.assertAlmostEqual(fit["a"], a, places=1)
        self.assertAlmostEqual(fit["b"], b, places=1)
        n90 = power_threshold(fit, 0.90, full)
        self.assertAlmostEqual(1 - fit["a"] * (n90 ** -fit["b"] - full ** -fit["b"]), 0.90, places=6)
        self.assertIsNone(fit_power_curve([10], [0.5], full))


class GridTests(unittest.TestCase):
    def test_grid_matches_proposal_counts_and_requires_lock(self):
        rows = grid(LOCKED)
        self.assertEqual(len(rows), 87)
        self.assertEqual(len({row["run_id"] for row in rows}), len(rows))
        by_stage = {}
        for row in rows:
            by_stage[row["stage"]] = by_stage.get(row["stage"], 0) + 1
            if row["budget"] not in ("full", "0"):
                self.assertLessEqual(int(row["budget"]), EXPECTED_DOMAINS[row["domain"]]["train"])
        self.assertEqual(by_stage, {"zero-shot": 1, "fine-tune": 80, "control": 6})
        self.assertEqual(rows[1]["domain"], "compound")  # cheapest complete curve first
        require_locked(POLICY)
        with self.assertRaisesRegex(ValueError, "not locked"):
            require_locked(dict(POLICY, epochs=None))

    def test_train_arguments_by_stage(self):
        rows = {row["stage"]: row for row in grid(LOCKED)}
        fine = train_arguments(rows["fine-tune"], LOCKED, Path("d"), Path("o"), Path("src.pt"))
        self.assertIn("--source-checkpoint", fine)
        self.assertIn("--amp", fine)
        control = train_arguments(rows["control"], LOCKED, Path("d"), Path("o"), None)
        self.assertEqual(control[control.index("--initialization") + 1], "coco")
        self.assertNotIn("--source-checkpoint", control)
        with self.assertRaises(ValueError):
            train_arguments(rows["fine-tune"], LOCKED, Path("d"), Path("o"), None)


class LabelFreePredictionTests(unittest.TestCase):
    def test_image_dataset_needs_no_annotation_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images" / "test").mkdir(parents=True)
            (root / "images" / "val").mkdir(parents=True)
            Image.new("RGB", (1280, 720)).save(root / "images" / "test" / "b.jpg")
            Image.new("RGB", (1280, 720)).save(root / "images" / "val" / "a.jpg")
            dataset = BDDImages(root, "test", [{"name": "b.jpg", "split": "test"}, {"name": "a.jpg", "split": "val"}])
            self.assertFalse((root / "annotations").exists())
            self.assertEqual(dataset.image_ids, {"a.jpg": 1, "b.jpg": 2})
            image, target = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 720, 1280))
            self.assertEqual(int(target["image_id"]), 1)

    def test_freeze_rejects_smoke_incomplete_and_modified_predictions(self):
        metadata = {"manifest_sha256": {"train": "a", "test": "b", "val": "c"}}
        # Compound's final test merges physical test and its held-out val images.
        manifests = {"train": [], "test": [{"name": "1.jpg", "split": "test", "timeofday": "night", "weather": "rainy"}],
                     "val": [{"name": "2.jpg", "split": "val", "timeofday": "night", "weather": "rainy"}]}
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(freeze_evaluation_plan.EXPECTED_DOMAINS, {"compound": {"final_test": 2}}):
            folder = Path(directory)
            (folder / "predictions_compound.json").write_text("[]", encoding="utf-8")
            entry = {"file": "predictions_compound.json", "sha256": freeze_evaluation_plan.sha256(folder / "predictions_compound.json"),
                     "images": 2, "image_names_sha256": freeze_evaluation_plan.image_names_sha256(manifests, "compound")}
            run = {"stage": "fine-tune", "domain": "compound", "budget": "10", "seed": 0, "smoke": False,
                   "completed": True, "manifest_sha256": metadata["manifest_sha256"]}
            manifest = {"schema": 1, "role": "final_test", "model_contract": MODEL_CONTRACT, "checkpoint_sha256": "x",
                        "run": run, "limited": False, "files": {"compound": entry}}
            path = folder / "prediction_manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(freeze_evaluation_plan.prediction_set(path, metadata, manifests)["files"]["compound"]["sha256"], entry["sha256"])
            for change, message in (({"run": dict(run, smoke=True)}, "smoke"), ({"run": dict(run, completed=False)}, "incomplete"),
                                    ({"files": {"compound": dict(entry, images=1)}}, "cover")):
                path.write_text(json.dumps(dict(manifest, **change)), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    freeze_evaluation_plan.prediction_set(path, metadata, manifests)
            path.write_text(json.dumps(manifest), encoding="utf-8")
            (folder / "predictions_compound.json").write_text("[{}]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                freeze_evaluation_plan.prediction_set(path, metadata, manifests)

    def test_scoring_maps_predictions_by_file_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "annotations").mkdir()
            records = [{"name": "b.jpg", "labels": [{"category": "car", "box2d": {"x1": 10, "y1": 10, "x2": 60, "y2": 60}}]},
                       {"name": "a.jpg", "labels": [{"category": "car", "box2d": {"x1": 100, "y1": 100, "x2": 200, "y2": 160}}]}]
            (root / "annotations" / "bdd100k_labels_images_test.json").write_text(json.dumps(records), encoding="utf-8")
            # Prediction image_ids deliberately disagree with GT ids; only file_name may be trusted.
            predictions = [{"image_id": 99, "file_name": "b.jpg", "category_id": 3, "bbox": [10, 10, 50, 50], "score": 0.9},
                           {"image_id": 98, "file_name": "a.jpg", "category_id": 3, "bbox": [100, 100, 100, 60], "score": 0.8}]
            (root / "p.json").write_text(json.dumps(predictions), encoding="utf-8")
            rows = [{"name": "a.jpg", "split": "test"}, {"name": "b.jpg", "split": "test"}]
            plan = {"domains": ["compound"], "bootstrap": {"resamples": 5, "seed": 1},
                    "prediction_sets": [{"label": "000_x", "run": {"stage": "fine-tune"}, "checkpoint_sha256": "c",
                                         "files": {"compound": {"path": str(root / "p.json")}}}]}
            args = Namespace(dataset_root=root, output_dir=root / "out")
            with patch.object(evaluate_frcnn, "final_test_rows", return_value=rows):
                results = evaluate_frcnn.score_prediction_plan(plan, "digest", {}, args)
            self.assertAlmostEqual(results[0]["AP50_95"], 1.0)
            self.assertEqual(np.load(root / "out" / "000_x" / "compound" / "bootstrap_class_ap.npy").shape, (5, len(CLASSES)))
            # An interrupted pass skips sets already scored instead of re-reading their predictions.
            (root / "p.json").unlink()
            with patch.object(evaluate_frcnn, "final_test_rows", return_value=rows):
                self.assertEqual(evaluate_frcnn.score_prediction_plan(plan, "digest", {}, args), results)

    def test_resume_interrupted_only_continues_the_same_unfinished_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, output = root / "plan.json", root / "scored"
            plan = {"schema": 3, "domains": [], "prediction_sets": []}
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            args = Namespace(final_plan=plan_path, output_dir=output, dataset_root=root, checkpoint=None, domain=None,
                             limit=0, resume_interrupted=False)
            with patch.object(evaluate_frcnn, "read_manifests", return_value=({}, {})), \
                    patch.object(evaluate_frcnn, "validate_prediction_plan"), \
                    patch.object(evaluate_frcnn, "score_prediction_plan", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    evaluate_frcnn.evaluate_final(args)
            self.assertTrue(plan_path.with_suffix(".json.opened").exists())
            with patch.object(evaluate_frcnn, "read_manifests", return_value=({}, {})), \
                    patch.object(evaluate_frcnn, "validate_prediction_plan"):
                with self.assertRaises(FileExistsError):  # a plain rerun is still refused
                    evaluate_frcnn.evaluate_final(args)
                args.resume_interrupted = True
                evaluate_frcnn.evaluate_final(args)
                self.assertTrue((output / "results.json").exists())
                with self.assertRaisesRegex(FileExistsError, "already completed"):
                    evaluate_frcnn.evaluate_final(args)
                plan_path.write_text(json.dumps(dict(plan, domains=["source"])), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "different plan"):
                    evaluate_frcnn.evaluate_final(args)


if __name__ == "__main__":
    unittest.main()
