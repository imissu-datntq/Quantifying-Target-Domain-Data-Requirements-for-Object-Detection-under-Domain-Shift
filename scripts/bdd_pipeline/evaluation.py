"""Shared COCO bbox AP/AR path; exact same ground-truth policy as training."""
from __future__ import annotations

import contextlib
import copy
import io
import time
from collections import Counter
from contextlib import nullcontext

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Subset

from .data import BDDDetection, CLASSES, RELIABLE_CLASSES, collate


def mean_defined(values: np.ndarray) -> float | None:
    valid = values[values >= 0]
    return float(valid.mean()) if valid.size else None


def coco_evaluator(ground_truth: dict, predictions: list[dict]) -> COCOeval:
    """Validated per-image COCO matching (evaluate only; no accumulation)."""
    image_ids = {image["id"] for image in ground_truth["images"]}
    category_ids = {category["id"] for category in ground_truth["categories"]}
    for prediction in predictions:
        if prediction["image_id"] not in image_ids or prediction["category_id"] not in category_ids:
            raise ValueError("Prediction image/category does not belong to this evaluation")
        box, score = prediction["bbox"], prediction["score"]
        if len(box) != 4 or not np.isfinite(box).all() or box[2] <= 0 or box[3] <= 0 or not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid prediction geometry or score")
    coco = COCO()
    coco.dataset = copy.deepcopy(ground_truth)
    coco.createIndex()
    if predictions:
        with contextlib.redirect_stdout(io.StringIO()):
            detected = coco.loadRes(copy.deepcopy(predictions))
    else:
        detected = COCO()
        detected.dataset = {"images": copy.deepcopy(ground_truth["images"]),
                            "categories": copy.deepcopy(ground_truth["categories"]), "annotations": []}
        detected.createIndex()
    evaluator = COCOeval(coco, detected, "bbox")
    evaluator.params.imgIds = sorted(image_ids)
    evaluator.params.catIds = sorted(category_ids)
    evaluator.params.maxDets = [1, 10, 100]
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
    return evaluator


def class_precision(evaluator: COCOeval, positions: np.ndarray | None = None) -> np.ndarray:
    """COCO precision [IoU, recall, class] for area=all and maxDets=100 over image positions.

    Mirrors COCOeval.accumulate but accepts repeated image positions, so test-image
    bootstrap resamples reuse one per-image matching pass.
    """
    params = evaluator._paramsEval
    images, areas = len(params.imgIds), len(params.areaRng)
    positions = np.arange(images) if positions is None else np.asarray(positions)
    thresholds = params.recThrs
    precision = -np.ones((len(params.iouThrs), len(thresholds), len(params.catIds)))
    max_detections = params.maxDets[-1]
    for k in range(len(params.catIds)):
        offset = k * areas * images  # area index 0 is "all"
        entries = [evaluator.evalImgs[offset + i] for i in positions]
        entries = [entry for entry in entries if entry is not None]
        if not entries:
            continue
        positives = np.count_nonzero(np.concatenate([entry["gtIgnore"] for entry in entries]) == 0)
        if positives == 0:
            continue
        scores = np.concatenate([entry["dtScores"][:max_detections] for entry in entries])
        order = np.argsort(-scores, kind="mergesort")
        matched = np.concatenate([entry["dtMatches"][:, :max_detections] for entry in entries], axis=1)[:, order].astype(bool)
        kept = ~np.concatenate([entry["dtIgnore"][:, :max_detections] for entry in entries], axis=1)[:, order].astype(bool)
        true_positive = np.cumsum(matched & kept, axis=1).astype(float)
        false_positive = np.cumsum(~matched & kept, axis=1).astype(float)
        for t in range(len(params.iouThrs)):
            tp, fp = true_positive[t], false_positive[t]
            recall = tp / positives
            values = tp / (fp + tp + np.spacing(1))
            if len(values):
                values = np.maximum.accumulate(values[::-1])[::-1]
            indices = np.searchsorted(recall, thresholds, side="left")
            valid = indices < len(values)
            row = np.zeros(len(thresholds))
            row[valid] = values[indices[valid]]
            precision[t, :, k] = row
    return precision


def class_ap(precision: np.ndarray) -> np.ndarray:
    """AP50:95 per class from class_precision; NaN where a class has no ground truth."""
    defined = (precision >= 0).all(axis=(0, 1))
    return np.where(defined, precision.clip(min=0).mean(axis=(0, 1)), np.nan)


def bootstrap_class_ap(evaluator: COCOeval, resamples: int, seed: int) -> np.ndarray:
    """[resample, class] AP50:95 from resampling evaluation images with replacement."""
    if resamples < 1:
        raise ValueError("Bootstrap needs at least one resample")
    rng = np.random.default_rng(seed)
    images = len(evaluator._paramsEval.imgIds)
    return np.stack([class_ap(class_precision(evaluator, rng.integers(0, images, images))) for _ in range(resamples)])


def fast_bootstrap_class_ap(evaluator: COCOeval, resamples: int, seed: int) -> np.ndarray:
    """bootstrap_class_ap with the same resamples, vectorised for full test sets.

    Detections are sorted once per class (stable, image order, as COCOeval.accumulate);
    a resample repeats each detection by its image's multiplicity, which places the copies
    adjacently. Results equal bootstrap_class_ap except for exact score ties across images.
    """
    if resamples < 1:
        raise ValueError("Bootstrap needs at least one resample")
    params = evaluator._paramsEval
    images, areas = len(params.imgIds), len(params.areaRng)
    thresholds, max_detections = params.recThrs, params.maxDets[-1]
    classes = []
    for k in range(len(params.catIds)):
        offset = k * areas * images
        image_index, scores, matched, kept, positives = [], [], [], [], np.zeros(images, dtype=np.int64)
        for i in range(images):
            entry = evaluator.evalImgs[offset + i]
            if entry is None:
                continue
            positives[i] = np.count_nonzero(entry["gtIgnore"] == 0)
            count = len(entry["dtScores"][:max_detections])
            image_index.append(np.full(count, i))
            scores.append(np.asarray(entry["dtScores"][:max_detections]))
            matched.append(entry["dtMatches"][:, :max_detections].astype(bool))
            kept.append(~entry["dtIgnore"][:, :max_detections].astype(bool))
        if image_index:
            image_index, scores = np.concatenate(image_index), np.concatenate(scores)
            order = np.argsort(-scores, kind="mergesort")
            matched = np.concatenate(matched, axis=1)[:, order]
            kept = np.concatenate(kept, axis=1)[:, order]
            image_index = image_index[order]
        classes.append((image_index, matched, kept, positives))
    rng = np.random.default_rng(seed)
    result = np.full((resamples, len(classes)), np.nan)
    for r in range(resamples):
        weights = np.bincount(rng.integers(0, images, images), minlength=images)
        for k, (image_index, matched, kept, positives) in enumerate(classes):
            total = int(weights @ positives)
            if total == 0:
                continue
            if len(image_index) == 0:
                result[r, k] = 0.0
                continue
            repeat = weights[image_index]
            m, keep = np.repeat(matched, repeat, axis=1), np.repeat(kept, repeat, axis=1)
            true_positive = np.cumsum(m & keep, axis=1).astype(float)
            false_positive = np.cumsum(~m & keep, axis=1).astype(float)
            curve = np.zeros((len(params.iouThrs), len(thresholds)))
            for t in range(len(params.iouThrs)):
                tp, fp = true_positive[t], false_positive[t]
                if not len(tp):
                    continue
                values = np.maximum.accumulate((tp / (fp + tp + np.spacing(1)))[::-1])[::-1]
                indices = np.searchsorted(tp / total, thresholds, side="left")
                valid = indices < len(values)
                curve[t, valid] = values[indices[valid]]
            result[r, k] = curve.mean()
    return result


def reliable_indices(category_ids: list[int], reliable_classes: tuple[str, ...]) -> list[int]:
    unknown = set(reliable_classes) - set(CLASSES)
    if unknown or not reliable_classes:
        raise ValueError(f"Invalid reliable class roster: {sorted(unknown)}")
    return [index for index, category in enumerate(category_ids) if CLASSES[category - 1] in reliable_classes]


def score_predictions(ground_truth: dict, predictions: list[dict],
                      reliable_classes: tuple[str, ...] = RELIABLE_CLASSES, evaluator: COCOeval | None = None) -> dict:
    """COCO metrics; pass the evaluator from coco_evaluator to reuse its per-image matching (e.g. for bootstrap)."""
    evaluator = evaluator or coco_evaluator(ground_truth, predictions)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.accumulate()
    precision, recall = evaluator.eval["precision"], evaluator.eval["recall"]
    category_ids = [int(category) for category in evaluator.params.catIds]
    support = Counter(annotation["category_id"] for annotation in ground_truth["annotations"])
    positive = {category: len({annotation["image_id"] for annotation in ground_truth["annotations"] if annotation["category_id"] == category}) for category in category_ids}
    per_class = []
    for index, category in enumerate(category_ids):
        per_class.append({"class": CLASSES[category - 1], "category_id": category,
                          "instances": support[category], "positive_images": positive[category],
                          "reliable": CLASSES[category - 1] in reliable_classes,
                          "AP50_95": mean_defined(precision[:, :, index, 0, -1]),
                          "AP50": mean_defined(precision[0, :, index, 0, -1]),
                          "AR100": mean_defined(recall[:, index, 0, -1])})
    reliable = reliable_indices(category_ids, reliable_classes)
    missing = [CLASSES[category_ids[i] - 1] for i in reliable if support[category_ids[i]] == 0]
    complete = len(reliable) == len(reliable_classes) and not missing
    return {"images": len(ground_truth["images"]), "instances": len(ground_truth["annotations"]), "predictions": len(predictions),
            "units": "fractions in [0,1]; unsupported metrics null",
            "AP50_95": mean_defined(precision[:, :, :, 0, -1]),
            "AP50": mean_defined(precision[0, :, :, 0, -1]),
            "AP75": mean_defined(precision[5, :, :, 0, -1]),
            "AR100": mean_defined(recall[:, :, 0, -1]),
            "AP_small": mean_defined(precision[:, :, :, 1, -1]),
            "AP_medium": mean_defined(precision[:, :, :, 2, -1]),
            "AP_large": mean_defined(precision[:, :, :, 3, -1]),
            # Primary metric: defined only when every fixed reliable class has ground truth.
            "reliable_AP50_95": mean_defined(precision[:, :, reliable, 0, -1]) if complete else None,
            "reliable_AP50": mean_defined(precision[0, :, reliable, 0, -1]) if complete else None,
            "reliable_classes": list(reliable_classes),
            "reliable_classes_without_ground_truth": missing,
            "reliable_policy": "Fixed roster shared by every domain (protocol amendment 2026-10-19); ten-class AP50_95 is a sensitivity analysis",
            "full_mean_policy": "COCO excludes categories with no ground-truth instances; ten-class roster reported explicitly",
            "per_class": per_class, "iou_thresholds": evaluator.params.iouThrs.tolist(), "max_detections": 100}


@torch.inference_mode()
def predict(model, dataset: BDDDetection, device: torch.device, *, batch_size: int = 1, workers: int = 0,
            amp: bool = False, label_map: dict[int, int] | None = None,
            indices: list[int] | None = None) -> tuple[list, float]:
    """COCO-format predictions; label_map translates model labels and drops unmapped ones."""
    if dataset.augment:
        raise ValueError("Evaluation dataset cannot augment images")
    selected = dataset if indices is None else Subset(dataset, indices)
    loader = DataLoader(selected, batch_size=batch_size, shuffle=False, num_workers=workers,
                        collate_fn=collate, pin_memory=device.type == "cuda")
    model.eval()
    predictions = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for images, targets in loader:
        context = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
        with context:
            outputs = model([image.to(device) for image in images])
        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"])
            for box, category, score in zip(output["boxes"].cpu().tolist(), output["labels"].cpu().tolist(), output["scores"].cpu().tolist()):
                if label_map is not None:
                    if category not in label_map:
                        continue
                    category = label_map[category]
                x1, y1, x2, y2 = box
                if x2 <= x1 or y2 <= y1:
                    continue
                predictions.append({"image_id": image_id, "category_id": category,
                                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
    if device.type == "cuda":
        torch.cuda.synchronize()
    return predictions, time.perf_counter() - start


def evaluate_model(model, dataset: BDDDetection, device: torch.device, *, batch_size: int = 1,
                   workers: int = 0, amp: bool = False,
                   reliable_classes: tuple[str, ...] = RELIABLE_CLASSES) -> tuple[dict, list]:
    predictions, inference_seconds = predict(model, dataset, device, batch_size=batch_size, workers=workers, amp=amp)
    result = score_predictions(dataset.coco_ground_truth(), predictions, reliable_classes)
    result.update(inference_seconds=inference_seconds, inference_images_per_second=len(dataset) / inference_seconds,
                  annotation_audit=dict(dataset.audit))
    return result, predictions
