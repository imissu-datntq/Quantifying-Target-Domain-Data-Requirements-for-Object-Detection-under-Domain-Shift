"""Common COCO bbox evaluation; this entry point never selects a test split."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_coverage(coverage: dict, image_ids: list[int], prediction_hash: str) -> None:
    """Require explicit inference coverage, including images with zero detections."""
    covered = coverage.get('evaluated_image_ids', [])
    if (coverage.get('status') != 'completed' or coverage.get('split') != 'val'
            or len(covered) != len(set(covered))
            or set(covered) != set(image_ids)):
        raise ValueError('Inference coverage must exactly match the selected validation sample')
    if coverage.get('prediction_sha256') != prediction_hash:
        raise ValueError('Coverage record belongs to a different prediction file')
    checkpoint_hash = coverage.get('checkpoint_sha256', '')
    if len(checkpoint_hash) != 64 or any(c not in '0123456789abcdef' for c in checkpoint_hash):
        raise ValueError('Coverage must record the evaluated checkpoint SHA-256')


def evaluate(ground_truth: dict, predictions: list, image_ids: list[int]) -> dict:
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images = {image['id']: image for image in ground_truth['images']}
    categories = {category['id'] for category in ground_truth['categories']}
    selected = set(image_ids)
    if not selected or len(selected) != len(image_ids) or not selected <= images.keys():
        raise ValueError('Evaluation IDs must be unique, nonempty and in validation')
    for item in predictions:
        if item['image_id'] not in selected or item['category_id'] not in categories:
            raise ValueError('Prediction is outside the selected validation IDs/classes')
        box = item['bbox']
        if len(box) != 4 or not all(math.isfinite(v) for v in box):
            raise ValueError('Expected finite original-image COCO xywh coordinates')
        if box[2] <= 0 or box[3] <= 0:
            raise ValueError('Prediction box must have positive area')
        if not math.isfinite(item['score']) or not 0 <= item['score'] <= 1:
            raise ValueError('Prediction score must be in [0, 1]')
    gt = COCO()
    gt.dataset = copy.deepcopy(ground_truth)
    gt.createIndex()
    if predictions:
        dt = gt.loadRes(copy.deepcopy(predictions))
    else:
        dt = COCO()
        dt.dataset = dict(images=gt.dataset['images'],
                          categories=gt.dataset['categories'], annotations=[])
        dt.createIndex()
    evaluator = COCOeval(gt, dt, 'bbox')
    evaluator.params.imgIds = sorted(selected)
    evaluator.params.catIds = sorted(categories)
    evaluator.params.iouThrs = np.linspace(.5, .95, 10)
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    precision = evaluator.eval['precision']
    per_class = {}
    for index, category in enumerate(evaluator.params.catIds):
        values = precision[:, :, index, 0, -1]
        present = values[values >= 0]
        per_class[str(category)] = float(present.mean() * 100) if present.size else None
    return dict(map_50_95=float(evaluator.stats[0] * 100),
                map_50=float(evaluator.stats[1] * 100), per_class_ap=per_class,
                images=len(selected), predictions=len(predictions),
                absent_class_policy='null; excluded from aggregate AP',
                iou_thresholds=evaluator.params.iouThrs.tolist(), max_detections=100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path('data/research_inputs_v1'))
    parser.add_argument('--domain', choices=['source', 'lighting', 'weather', 'compound'], required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--coverage', type=Path, required=True,
                        help='Inference manifest with every evaluated image ID and prediction/checkpoint SHA-256')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-images', type=int, default=64,
                        help='Must match the deterministic prefix used by the exporter')
    args = parser.parse_args()
    if args.max_images <= 0:
        parser.error('--max-images must be positive')
    annotations = args.inputs / 'coco/instances_val.json'
    manifest = args.inputs / 'manifests/domains' / args.domain / 'val.csv'
    with manifest.open(newline='') as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: Path(row['image']).name)
        rows = rows[:args.max_images]
    if any(not row['image'].startswith('images/val/') for row in rows):
        raise ValueError('Only official validation image paths are allowed')
    ids = [int(row['coco_image_id']) for row in rows]
    prediction_hash = digest(args.predictions)
    coverage = json.loads(args.coverage.read_text())
    verify_coverage(coverage, ids, prediction_hash)
    if args.output.exists():
        raise FileExistsError(args.output)
    metrics = evaluate(json.loads(annotations.read_text()),
                       json.loads(args.predictions.read_text()), ids)
    result = dict(split='val', domain=args.domain, metrics=metrics,
                  selection_evidence=coverage.get('selection_evidence', False),
                  candidate=coverage.get('candidate'),
                  annotation_sha256=digest(annotations), manifest_sha256=digest(manifest),
                  prediction_sha256=prediction_hash, coverage_sha256=digest(args.coverage),
                  checkpoint_sha256=coverage['checkpoint_sha256'],
                  evaluator='official pycocotools COCOeval bbox',
                  validation_subset='lexicographic-name-prefix', max_images=args.max_images,
                  coordinate_format='original-image xywh')
    serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
    # Exclusive creation also protects against collisions after evaluation.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        output.write(serialized)


if __name__ == '__main__':
    main()
