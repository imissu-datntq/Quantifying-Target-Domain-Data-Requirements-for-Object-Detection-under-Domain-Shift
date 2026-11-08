"""Checked split membership, nested sampling and native-resolution targets."""
from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from prepare_revised_bdd100k import (CLASSES, DOMAINS, EXPECTED, EXPECTED_DOMAINS, HOLDOUT_VAL_DOMAINS,
                                     RELIABLE_CLASSES, sha256, stream_records)

CLASS_TO_ID = {name: index + 1 for index, name in enumerate(CLASSES)}
WIDTH, HEIGHT = 1280, 720


def read_manifests(root: Path) -> tuple[dict[str, Any], dict[str, list[dict]]]:
    if not (root / "COMPLETE").is_file():
        raise ValueError("Use a completed revised dataset")
    metadata = json.loads((root / "dataset_metadata.json").read_text(encoding="utf-8"))
    if metadata["counts"] != EXPECTED or tuple(metadata["classes"]) != CLASSES:
        raise ValueError("Dataset count/class contract differs from the proposal")
    manifests, names, hashes = {}, {}, {}
    for split, expected in EXPECTED.items():
        path = root / "manifests" / f"{split}.csv"
        if sha256(path) != metadata["manifest_sha256"][split]:
            raise ValueError(f"Modified manifest: {split}")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        manifests[split] = sorted(rows, key=lambda row: row["name"])
        names[split] = {row["name"] for row in rows}
        hashes[split] = {row["sha256"] for row in rows}
        if len(rows) != len(names[split]) or len(rows) != expected:
            raise ValueError(f"Incorrect/duplicate {split} membership")
        if any(row["split"] != split or (int(row["width"]), int(row["height"])) != (WIDTH, HEIGHT) for row in rows):
            raise ValueError(f"Split or geometry mismatch: {split}")
    for a, b in (("train", "test"), ("train", "val"), ("test", "val")):
        if names[a] & names[b] or hashes[a] & hashes[b]:
            raise ValueError(f"Cross-split leakage: {a}/{b}")
    check_domain_roles(manifests)
    return metadata, manifests


def domain_rows(manifests: dict, split: str, domain: str) -> list[dict]:
    timeofday, weather = DOMAINS[domain]
    rows = [row for row in manifests[split] if (row["timeofday"], row["weather"]) == (timeofday, weather)]
    if not rows:
        raise ValueError(f"Empty domain: {split}/{domain}")
    return rows


def selection_rows(manifests: dict, domain: str) -> list[dict]:
    """Validation rows allowed for pilot/checkpoint decisions; none for holdout-val domains."""
    if domain in HOLDOUT_VAL_DOMAINS:
        raise ValueError(f"{domain} validation images belong to the final test set (protocol amendment 2026-10-19)")
    return domain_rows(manifests, "val", domain)


def final_test_rows(manifests: dict, domain: str) -> list[dict]:
    """Held-out rows; weather/compound also include their preserved-val images."""
    rows = domain_rows(manifests, "test", domain)
    if domain in HOLDOUT_VAL_DOMAINS:
        rows = rows + domain_rows(manifests, "val", domain)
    return sorted(rows, key=lambda row: row["name"])


def holdout_names(manifests: dict) -> set[str]:
    names = {row["name"] for row in manifests["test"]}
    for domain in HOLDOUT_VAL_DOMAINS:
        names |= {row["name"] for row in domain_rows(manifests, "val", domain)}
    return names


def check_domain_roles(manifests: dict) -> None:
    for domain, expected in EXPECTED_DOMAINS.items():
        actual = {"train": len(domain_rows(manifests, "train", domain)),
                  "final_test": len(final_test_rows(manifests, domain)),
                  "selection_val": 0 if domain in HOLDOUT_VAL_DOMAINS else len(selection_rows(manifests, domain))}
        if actual != expected:
            raise ValueError(f"Domain role counts differ for {domain}: {actual} != {expected}")


def nested_rows(rows: list[dict], budget: str, seed: int) -> list[dict]:
    pool = sorted(rows, key=lambda row: row["name"])
    amount = len(pool) if budget == "full" else int(budget)
    if not 1 <= amount <= len(pool):
        raise ValueError(f"Budget {budget} outside pool of {len(pool)}")
    random.Random(seed).shuffle(pool)
    return pool[:amount]


def sanitize_labels(labels: list[dict], width: int, height: int) -> tuple[list, list, Counter]:
    boxes, categories, audit = [], [], Counter()
    for label in labels:
        audit["input_boxes"] += 1
        category = label.get("category")
        if category not in CLASS_TO_ID:
            raise ValueError(f"Unexpected detection category: {category}")
        try:
            coordinates = [float(label["box2d"][key]) for key in ("x1", "y1", "x2", "y2")]
        except (KeyError, TypeError, ValueError):
            audit["malformed"] += 1
            continue
        if not all(math.isfinite(value) for value in coordinates):
            audit["nonfinite"] += 1
            continue
        x1, y1, x2, y2 = coordinates
        if x2 <= x1 or y2 <= y1:
            audit["nonpositive"] += 1
            continue
        clipped = [min(max(x1, 0), width), min(max(y1, 0), height), min(max(x2, 0), width), min(max(y2, 0), height)]
        audit["clipped"] += coordinates != clipped
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            audit["empty_after_clipping"] += 1
            continue
        boxes.append(clipped)
        categories.append(CLASS_TO_ID[category])
        audit["kept"] += 1
    return boxes, categories, audit


def horizontal_flip(image: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    width = image.shape[-1]
    flipped = boxes.clone()
    flipped[:, 0], flipped[:, 2] = width - boxes[:, 2], width - boxes[:, 0]
    return image.flip(-1), flipped


class BDDDetection(Dataset):
    """Only selected revised images; same sanitation for training and COCO GT.

    `split` is the evaluation role; each row's physical split (default: the role)
    selects its image directory and annotation file, so a final-test role may
    combine physical test and holdout-val rows.
    """

    def __init__(self, root: Path, split: str, rows: list[dict], *, augment: bool = False, seed: int = 0):
        if split == "test" and augment:
            raise ValueError("Test augmentation is forbidden")
        self.root, self.split, self.rows = root, split, rows
        self.augment, self.seed, self.epoch = augment, seed, 0
        self.image_ids = {row["name"]: i + 1 for i, row in enumerate(sorted(rows, key=lambda row: row["name"]))}
        self.sources = {row["name"]: row.get("split", split) for row in rows}
        selected = set(self.sources)
        if len(selected) != len(rows):
            raise ValueError("Duplicate selected image")
        self.annotations, self.audit = {}, Counter()
        for source in sorted(set(self.sources.values())):
            for record in stream_records(root / "annotations" / f"bdd100k_labels_images_{source}.json"):
                name = record["name"]
                if self.sources.get(name) == source:
                    if name in self.annotations:
                        raise ValueError(f"Duplicate annotation: {name}")
                    boxes, labels, audit = sanitize_labels(record.get("labels", []), WIDTH, HEIGHT)
                    self.annotations[name] = (boxes, labels)
                    self.audit.update(audit)
        if set(self.annotations) != selected:
            raise ValueError("Selected image/annotation mismatch")
        self.audit["empty_images"] = sum(not boxes for boxes, _ in self.annotations.values())

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        name = row["name"]
        with Image.open(self.root / "images" / self.sources[name] / name) as image:
            if image.size != (WIDTH, HEIGHT):
                raise ValueError(f"Non-native dimensions: {name}")
            array = np.array(image.convert("RGB"), copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255)
        raw_boxes, raw_labels = self.annotations[name]
        boxes = torch.tensor(raw_boxes, dtype=torch.float32).reshape(-1, 4)
        if self.augment and random.Random(f"{self.seed}/{self.epoch}/{name}").random() < 0.5:
            tensor, boxes = horizontal_flip(tensor, boxes)
        target = {"boxes": boxes, "labels": torch.tensor(raw_labels, dtype=torch.int64),
                  "image_id": torch.tensor(self.image_ids[name], dtype=torch.int64),
                  "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                  "iscrowd": torch.zeros(len(boxes), dtype=torch.int64)}
        return tensor, target

    def coco_ground_truth(self) -> dict:
        images, annotations = [], []
        for row in sorted(self.rows, key=lambda row: row["name"]):
            name, image_id = row["name"], self.image_ids[row["name"]]
            images.append({"id": image_id, "file_name": name, "width": WIDTH, "height": HEIGHT})
            boxes, labels = self.annotations[name]
            for box, category in zip(boxes, labels):
                x1, y1, x2, y2 = box
                annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": category,
                                    "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
        return {"info": {}, "images": images, "annotations": annotations,
                "categories": [{"id": CLASS_TO_ID[name], "name": name} for name in CLASSES]}


class BDDImages(Dataset):
    """Images only, for inference on held-out roles without opening any annotation file.

    image_ids follow BDDDetection's sorted-name order; scoring later remaps by file name.
    """

    augment = False

    def __init__(self, root: Path, split: str, rows: list[dict]):
        self.root, self.split, self.rows = root, split, sorted(rows, key=lambda row: row["name"])
        self.image_ids = {row["name"]: i + 1 for i, row in enumerate(self.rows)}
        self.sources = {row["name"]: row.get("split", split) for row in self.rows}
        if len(self.sources) != len(self.rows):
            raise ValueError("Duplicate selected image")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        name = self.rows[index]["name"]
        with Image.open(self.root / "images" / self.sources[name] / name) as image:
            if image.size != (WIDTH, HEIGHT):
                raise ValueError(f"Non-native dimensions: {name}")
            array = np.array(image.convert("RGB"), copy=True)
        tensor = torch.from_numpy(array).permute(2, 0, 1).float().div_(255)
        return tensor, {"image_id": torch.tensor(self.image_ids[name], dtype=torch.int64)}


def collate(batch: list) -> tuple[list, list]:
    images, targets = zip(*batch)
    return list(images), list(targets)
