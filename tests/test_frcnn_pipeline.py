"""Semantic checks for native geometry, COCO metrics and protocol boundaries."""
from __future__ import annotations

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np

from bdd_pipeline.data import (BDDDetection, CLASSES, RELIABLE_CLASSES, final_test_rows, holdout_names,
                               horizontal_flip, nested_rows, sanitize_labels, selection_rows)
from bdd_pipeline.evaluation import class_ap, class_precision, coco_evaluator, score_predictions
from bdd_pipeline.recovery import curve_summary, recovery, recovery_table, sustained_threshold
from bdd_pipeline.model import MODEL_CONTRACT, build_model
from bdd_pipeline.training import learning_rate_multiplier, schedule_size
from evaluate_frcnn import validate_plan


class PipelineTests(unittest.TestCase):
    def test_sanitize_clips_and_rejects_geometry_consistently(self):
        def label(box):
            return {"category": "car", "box2d": dict(zip(("x1", "y1", "x2", "y2"), box))}
        boxes, categories, audit = sanitize_labels([
            label([-2, 2, 12, 8]), label([10, 2, 4, 8]),
            label([float("nan"), 0, 1, 1]), label([20, 0, 30, 1])], 20, 10)
        self.assertEqual(boxes, [[0, 2, 12, 8]])
        self.assertEqual(categories, [3])
        self.assertEqual((audit["clipped"], audit["nonpositive"], audit["nonfinite"], audit["empty_after_clipping"]), (2, 1, 1, 1))
        with self.assertRaises(ValueError):
            sanitize_labels([{"category": "unexpected"}], 20, 10)

    def test_horizontal_flip_maps_edges_and_is_involutive(self):
        image = torch.arange(60).reshape(3, 4, 5)
        boxes = torch.tensor([[0., 1., 2., 3.], [3., 0., 5., 4.]])
        flipped_image, flipped_boxes = horizontal_flip(image, boxes)
        self.assertTrue(torch.equal(flipped_boxes, torch.tensor([[3., 1., 5., 3.], [0., 0., 2., 4.]])))
        recovered_image, recovered_boxes = horizontal_flip(flipped_image, flipped_boxes)
        self.assertTrue(torch.equal(image, recovered_image))
        self.assertTrue(torch.equal(boxes, recovered_boxes))

    def test_native_model_transform_normalizes_once_and_keeps_geometry(self):
        torch.set_num_threads(2)
        model = build_model("none").eval()
        image = torch.ones(3, 720, 1280)
        boxes = torch.tensor([[0., 0., 1280., 720.]])
        transformed, targets = model.transform([image], [{"boxes": boxes}])
        self.assertEqual(transformed.image_sizes, [(720, 1280)])
        self.assertEqual(tuple(transformed.tensors.shape), (1, 3, 736, 1280))
        self.assertTrue(torch.equal(targets[0]["boxes"], boxes))
        expected = (1 - MODEL_CONTRACT["image_mean"][0]) / MODEL_CONTRACT["image_std"][0]
        self.assertAlmostEqual(float(transformed.tensors[0, 0, 0, 0]), expected, places=5)
        self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, 11)
        self.assertFalse(model.backbone.body.conv1.weight.requires_grad)
        self.assertTrue(model.backbone.body.layer2[0].conv1.weight.requires_grad)

    def test_nested_budget_independent_of_input_order(self):
        rows = [{"name": str(i)} for i in range(100)]
        small = nested_rows(rows, "10", 7)
        large = nested_rows(list(reversed(rows)), "25", 7)
        self.assertEqual(small, large[:10])
        with self.assertRaises(ValueError):
            nested_rows(rows, "101", 7)

    def test_step_floor_epoch_term_and_short_accumulation_window(self):
        self.assertEqual(schedule_size(3, 1, 2, 1, 0), (2, 2))
        self.assertEqual(schedule_size(10, 1, 2, 30, 2000), (2000, 5))
        self.assertEqual(schedule_size(18015, 1, 2, 30, 2000), (270240, 9008))
        self.assertAlmostEqual(learning_rate_multiplier(9, 10, 0), 0.01)
        self.assertAlmostEqual(learning_rate_multiplier(0, 10, 5), 0.28)

    def test_dataset_ids_targets_and_gt_share_sanitation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "annotations").mkdir()
            (root / "images" / "val").mkdir(parents=True)
            Image.new("RGB", (1280, 720)).save(root / "images" / "val" / "one.jpg")
            record = {"name": "one.jpg", "labels": [{"category": "pedestrian", "box2d": {"x1": -1, "y1": 0, "x2": 20, "y2": 10}}]}
            (root / "annotations" / "bdd100k_labels_images_val.json").write_text(json.dumps([record]), encoding="utf-8")
            dataset = BDDDetection(root, "val", [{"name": "one.jpg"}])
            image, target = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 720, 1280))
            self.assertEqual(target["labels"].tolist(), [1])
            self.assertEqual(target["area"].tolist(), [200])
            gt = dataset.coco_ground_truth()["annotations"][0]
            self.assertEqual(gt["bbox"], [0, 0, 20, 10])
            self.assertEqual(gt["image_id"], int(target["image_id"]))

    def test_coco_perfect_empty_absent_classes_and_threshold(self):
        gt = {"info": {}, "images": [{"id": 1, "width": 1280, "height": 720}],
              "categories": [{"id": i + 1, "name": name} for i, name in enumerate(CLASSES)],
              "annotations": [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}]}
        predictions = [{"image_id": 1, "category_id": 3, "bbox": [10, 10, 20, 20], "score": .99}]
        perfect = score_predictions(gt, predictions, ("car",))
        empty = score_predictions(gt, [], ("car",))
        roster = score_predictions(gt, predictions)
        self.assertAlmostEqual(perfect["AP50_95"], 1)
        self.assertAlmostEqual(perfect["reliable_AP50_95"], 1)
        self.assertEqual(perfect["reliable_classes"], ["car"])
        self.assertIsNone(perfect["per_class"][0]["AP50_95"])
        self.assertEqual(empty["AP50_95"], 0)
        # The fixed four-class roster is undefined, not silently shrunk, when a class lacks ground truth.
        self.assertEqual(roster["reliable_classes"], list(RELIABLE_CLASSES))
        self.assertIsNone(roster["reliable_AP50_95"])
        self.assertEqual(roster["reliable_classes_without_ground_truth"], ["pedestrian", "traffic light", "traffic sign"])
        with self.assertRaises(ValueError):
            score_predictions(gt, predictions, ("unknown",))
        with self.assertRaises(ValueError):
            score_predictions(gt, [dict(predictions[0], image_id=99)])

    def test_final_plan_rejects_smoke_modified_and_wrong_dataset(self):
        metadata = {"manifest_sha256": {"train": "a", "val": "b", "test": "c"}}
        plan = {"schema": 2, "model_contract": MODEL_CONTRACT, **metadata,
                "domains": ["source"], "checkpoints": [{"path": "fixture.pt", "sha256": "original"}],
                "reliable_classes": list(RELIABLE_CLASSES), "holdout_val_domains": ["weather", "compound"]}
        checkpoint = {"training_completed": True, "config": {"smoke": False, **metadata}}
        with patch("evaluate_frcnn.sha256", return_value="original"), patch("evaluate_frcnn.load_checkpoint", return_value=checkpoint):
            validate_plan(plan, metadata)
            checkpoint["config"]["smoke"] = True
            with self.assertRaisesRegex(ValueError, "smoke/incomplete"):
                validate_plan(plan, metadata)
        with patch("evaluate_frcnn.sha256", return_value="modified"):
            with self.assertRaisesRegex(ValueError, "changed after freezing"):
                validate_plan(plan, metadata)
        with self.assertRaisesRegex(ValueError, "dataset differs"):
            validate_plan(plan, {"manifest_sha256": {"train": "wrong"}})
        with self.assertRaisesRegex(ValueError, "amended"):
            validate_plan(dict(plan, reliable_classes=["car"]), metadata)

    def test_bootstrap_precision_matches_cocoeval_accumulate(self):
        rng = np.random.default_rng(3)
        images = [{"id": i, "width": 1280, "height": 720} for i in range(1, 9)]
        annotations, predictions = [], []
        for image in images:
            for _ in range(rng.integers(0, 5)):
                x, y, w, h = rng.uniform(0, 600), rng.uniform(0, 300), rng.uniform(8, 200), rng.uniform(8, 200)
                category = int(rng.choice([1, 3, 9]))
                annotations.append({"id": len(annotations) + 1, "image_id": image["id"], "category_id": category,
                                    "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
                if rng.random() < 0.8:
                    predictions.append({"image_id": image["id"], "category_id": category, "score": float(rng.random()),
                                        "bbox": [x + rng.normal(0, 4), y + rng.normal(0, 4), w, h]})
            for _ in range(2):
                predictions.append({"image_id": image["id"], "category_id": int(rng.choice([1, 3, 9])), "score": float(rng.random()),
                                    "bbox": [rng.uniform(0, 900), rng.uniform(0, 500), 30.0, 30.0]})
        gt = {"info": {}, "images": images, "annotations": annotations,
              "categories": [{"id": i + 1, "name": name} for i, name in enumerate(CLASSES)]}
        evaluator = coco_evaluator(gt, predictions)
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator.accumulate()
        expected = evaluator.eval["precision"][:, :, :, 0, -1]
        self.assertTrue(np.allclose(class_precision(evaluator), expected))
        ap = class_ap(class_precision(evaluator))
        self.assertTrue(np.isnan(ap[1]))  # rider has no ground truth
        # Repeating every image twice keeps precision and recall, so AP is unchanged.
        doubled = class_ap(class_precision(evaluator, np.repeat(np.arange(len(images)), 2)))
        self.assertTrue(np.allclose(np.nan_to_num(doubled, nan=-1), np.nan_to_num(ap, nan=-1)))

    def test_amended_roles_merge_holdout_val_and_block_selection(self):
        def row(name, split, timeofday, weather):
            return {"name": name, "split": split, "timeofday": timeofday, "weather": weather}
        manifests = {"train": [row("a", "train", "daytime", "rainy")],
                     "test": [row("b", "test", "daytime", "rainy"), row("c", "test", "daytime", "clear")],
                     "val": [row("d", "val", "daytime", "rainy"), row("e", "val", "daytime", "clear"), row("f", "val", "night", "rainy")]}
        self.assertEqual([r["name"] for r in final_test_rows(manifests, "weather")], ["b", "d"])
        self.assertEqual([r["name"] for r in final_test_rows(manifests, "source")], ["c"])
        self.assertEqual([r["name"] for r in selection_rows(manifests, "source")], ["e"])
        for domain in ("weather", "compound"):
            with self.assertRaisesRegex(ValueError, "final test"):
                selection_rows(manifests, domain)
        self.assertEqual(holdout_names(manifests), {"b", "c", "d", "f"})

    def test_dataset_reads_each_physical_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "annotations").mkdir()
            for split, name in (("test", "t.jpg"), ("val", "v.jpg")):
                (root / "images" / split).mkdir(parents=True)
                Image.new("RGB", (1280, 720)).save(root / "images" / split / name)
                record = {"name": name, "labels": [{"category": "car", "box2d": {"x1": 1, "y1": 1, "x2": 9, "y2": 9}}]}
                (root / "annotations" / f"bdd100k_labels_images_{split}.json").write_text(json.dumps([record]), encoding="utf-8")
            dataset = BDDDetection(root, "test", [{"name": "t.jpg", "split": "test"}, {"name": "v.jpg", "split": "val"}])
            self.assertEqual(len(dataset.coco_ground_truth()["annotations"]), 2)
            self.assertEqual(tuple(dataset[1][0].shape), (3, 720, 1280))
            with self.assertRaises(ValueError):
                BDDDetection(root, "test", [{"name": "v.jpg", "split": "val"}], augment=True)

    def test_recovery_guard_thresholds_and_per_class_rows(self):
        self.assertAlmostEqual(recovery(0.35, 0.30, 0.40), 0.5)
        self.assertAlmostEqual(recovery(0.25, 0.30, 0.40), -0.5)
        self.assertIsNone(recovery(0.31, 0.30, 0.305, min_denominator=0.01))
        self.assertIsNone(recovery(0.31, 0.30, 0.29))
        curve = {10: -0.1, 25: 0.2, 50: 0.1, 100: 0.8, 250: 0.95, 1000: 1.0}
        self.assertEqual(sustained_threshold(curve, 0.0, strict=True), 25)
        self.assertEqual(sustained_threshold(curve, 0.75, strict=False), 100)
        self.assertEqual(sustained_threshold(curve, 0.90, strict=False), 250)
        self.assertIsNone(sustained_threshold({10: 0.5, 20: 0.6}, 0.9, strict=False))

        def metrics(reliable, car, light):
            return {"reliable_AP50_95": reliable, "AP50_95": reliable,
                    "per_class": [{"class": "car", "AP50_95": car}, {"class": "traffic light", "AP50_95": light}]}
        records = [{"domain": "weather", "budget": 0, "seed": "-", "metrics": metrics(0.30, 0.50, 0.20)},
                   {"domain": "weather", "budget": "full", "seed": 0, "metrics": metrics(0.40, 0.505, 0.40)},
                   {"domain": "weather", "budget": 10, "seed": 0, "metrics": metrics(0.35, 0.502, 0.36)}]
        rows = recovery_table(records, {"weather": 1985}, min_denominator=0.01)
        by_class = {(r["class"], r["budget"]): r["recovery"] for r in rows}
        self.assertAlmostEqual(by_class[("reliable", 10)], 0.5)
        self.assertAlmostEqual(by_class[("traffic light", 10)], 0.8)
        self.assertIsNone(by_class[("car", 10)])  # car ceiling gain is below the denominator guard
        summary = {entry["class"]: entry for entry in curve_summary(rows)}
        self.assertEqual(summary["traffic light"]["N75"], 10)


if __name__ == "__main__":
    unittest.main()
