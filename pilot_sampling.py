"""Deterministic manifest sampling shared by bounded model-selection pilots."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def read_manifest_sample(path: Path, count: int, split: str) -> list[dict[str, str]]:
    """Return an exact lexicographic prefix with canonical split membership."""
    if count <= 0:
        raise ValueError("Sample count must be positive")
    if split not in {"train", "val"}:
        raise ValueError("Pilot sampling supports train or val only")
    with path.open(newline="") as stream:
        all_rows = sorted(csv.DictReader(stream), key=lambda row: row["name"])
    names = [row["name"] for row in all_rows]
    image_ids = [row["coco_image_id"] for row in all_rows]
    if len(names) != len(set(names)) or len(image_ids) != len(set(image_ids)):
        raise ValueError("Duplicate manifest membership")
    if len(all_rows) < count:
        raise ValueError(f"Manifest has {len(all_rows)} rows; {count} required")
    rows = all_rows[:count]
    if any(row["image"] != f"images/{split}/" + row["name"] for row in rows):
        raise ValueError("Manifest row is outside the declared split")
    return rows


def read_preparation_counts(data_path: Path) -> tuple[int, int]:
    """Read positive source/target counts bound to a generated data config."""
    preparation = data_path.resolve().parent / "preparation.json"
    payload = json.loads(preparation.read_text())
    source = int(payload.get("source_train_images", 0))
    target = int(payload.get("target_train_images", 0))
    if source <= 0 or target <= 0:
        raise ValueError("Preparation must record positive source and target counts")
    return source, target
