"""BDD100K registration for the pinned Adaptive Teacher implementation.

Unlabeled records are constructed exclusively from image manifests. This module
does not open canonical annotations when loading an unlabeled dataset.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


def load_records(root: Path, manifest: Path, split: str, labeled: bool,
                 split_root: Path = None) -> list[dict]:
    if split not in {"train", "val"} or (not labeled and split != "train"):
        raise ValueError("Only train/val are available; unlabeled data must be train")
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = [row["name"] for row in rows]
    if not rows or len(names) != len(set(names)):
        raise ValueError("Empty or duplicate manifest membership")
    records = {}
    for row in rows:
        relative = Path(row["image"])
        if relative.parts != ("images", split, row["name"]):
            raise ValueError("Image path does not belong to the declared split")
        image = (split_root or root) / relative
        if not image.is_file():
            raise ValueError(f"Missing image: {image}")
        image_id = int(row["coco_image_id"])
        if image_id in records:
            raise ValueError("Duplicate image ID")
        records[image_id] = dict(file_name=str(image.absolute()), image_id=image_id,
                                 height=720, width=1280, annotations=[])
    if labeled:
        with (root / "coco" / f"instances_{split}.json").open() as stream:
            coco = json.load(stream)
        categories = sorted(coco["categories"], key=lambda item: item["id"])
        if len(categories) != 10:
            raise ValueError("Expected ten classes")
        mapping = {category["id"]: index for index, category in enumerate(categories)}
        selected = {item["id"]: item for item in coco["images"] if item["id"] in records}
        if set(selected) != set(records):
            raise ValueError("Manifest IDs absent from canonical annotations")
        for row in rows:
            item = selected[int(row["coco_image_id"])]
            if Path(item["file_name"]).name != row["name"] or (item["width"], item["height"]) != (1280, 720):
                raise ValueError("Canonical image identity/geometry mismatch")
        for annotation in coco["annotations"]:
            record = records.get(annotation["image_id"])
            if record is not None:
                record["annotations"].append(dict(
                    bbox=annotation["bbox"], bbox_mode=1,
                    category_id=mapping[annotation["category_id"]],
                    iscrowd=annotation.get("iscrowd", 0)))
    return list(records.values())


def dataset_specifications(domain: str, budget: int, seed: int,
                           train_limit: int, validation_limit: int) -> dict:
    if domain not in {'lighting', 'weather', 'compound'} or budget not in {100, 1000} or seed not in {42, 43, 44}:
        raise ValueError('Unsupported pilot domain/budget/seed')
    if train_limit <= 0 or validation_limit <= 0:
        raise ValueError('Pilot limits must be positive')
    return {
        "bdd_source_train": ("domains/source/train.csv", "train", True, train_limit),
        "bdd_target_unlabeled": (f"unlabeled/{domain}/full_train_pool.csv", "train", False, train_limit),
        "bdd_source_val": ("domains/source/val.csv", "val", True, validation_limit),
        "bdd_target_val": (f"domains/{domain}/val.csv", "val", True, validation_limit),
        "bdd_finetune_train": (f"labeled/{domain}/seed_{seed}/n_{budget}.csv", "train", True, train_limit),
    }


def register(root: Path, smoke: bool = False, domain: str = 'lighting',
             budget: int = 100, seed: int = 42, train_limit: int = 128,
             validation_limit: int = 64) -> None:
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.structures import BoxMode
    specifications = dataset_specifications(
        domain, budget, seed, train_limit, validation_limit)
    classes = json.loads((root / 'metadata/classes.json').read_text())['classes']
    classes = sorted(classes, key=lambda category: category['coco_id'])
    for name, (relative, split, labeled, limit) in specifications.items():
        def loader(relative=relative, split=split, labeled=labeled, limit=limit):
            records = load_records(root, root / "manifests" / relative, split, labeled,
                                   root.parent / 'research_split_v1')
            for record in records:
                for annotation in record['annotations']:
                    annotation['bbox_mode'] = BoxMode.XYWH_ABS
            return sorted(records, key=lambda row: row["image_id"])[:limit] if smoke else records
        DatasetCatalog.register(name, loader)
        MetadataCatalog.get(name).set(
            evaluator_type="coco", thing_classes=[category['name'] for category in classes],
            thing_dataset_id_to_contiguous_id={category['coco_id']: i for i, category in enumerate(classes)})
