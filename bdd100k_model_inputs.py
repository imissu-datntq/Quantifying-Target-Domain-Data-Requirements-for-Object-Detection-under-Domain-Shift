"""Export canonical model inputs from ``data/research_split_v1``.

The exporter creates one ten-class category mapping, COCO annotations, YOLO
labels, domain manifests, nested labeled target budgets, and separate unlabeled
target-pool manifests.  It never removes labels from the canonical split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from bdd100k_research_split import (
    DOMAINS,
    SplitError,
    jpeg_dimensions,
    json_array_items,
    membership_hash,
    metadata_from_record,
    write_json,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_SPLIT_ROOT = ROOT / "data" / "research_split_v1"
DEFAULT_OUTPUT = ROOT / "data" / "research_inputs_v1"
EXPORT_VERSION = "bdd100k-model-inputs-v1"
SEEDS = (42, 43, 44)
BASE_BUDGETS = (10, 25, 50, 100, 250, 500, 1000)
MATCHED_POOL_SIZE = 1738
CLASS_NAMES = (
    "car",
    "traffic sign",
    "traffic light",
    "person",
    "truck",
    "bus",
    "bike",
    "rider",
    "motor",
    "train",
)
CLASS_TO_YOLO = {name: index for index, name in enumerate(CLASS_NAMES)}
CLASS_TO_COCO = {name: index + 1 for index, name in enumerate(CLASS_NAMES)}


def image_id(name: str) -> int:
    """Return a deterministic positive 63-bit COCO image id."""

    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def normalized_box(
    box: Mapping[str, Any], width: int, height: int
) -> tuple[float, float, float, float, bool]:
    try:
        x1, y1, x2, y2 = (float(box[key]) for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError) as error:
        raise SplitError("Invalid box coordinates") from error
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise SplitError("Nonfinite box coordinates")
    clipped = x1 < 0 or y1 < 0 or x2 > width or y2 > height
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        raise SplitError("Degenerate box after clipping")
    box_width = x2 - x1
    box_height = y2 - y1
    return (
        (x1 + x2) / (2.0 * width),
        (y1 + y2) / (2.0 * height),
        box_width / width,
        box_height / height,
        clipped,
    )


def copy_json_lines_as_array(source: Path, destination_handle: Any) -> None:
    destination_handle.write("[")
    first = True
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not first:
                destination_handle.write(",")
            destination_handle.write(line.rstrip("\n"))
            first = False
    destination_handle.write("]")


def export_split(split_root: Path, output: Path, split: str) -> dict[str, int]:
    annotation_path = split_root / "annotations" / f"bdd100k_labels_images_{split}.json"
    image_root = split_root / "images" / split
    yolo_root = output / "yolo" / "labels" / split
    yolo_root.mkdir(parents=True, exist_ok=True)
    fragments = output / ".fragments"
    fragments.mkdir(parents=True, exist_ok=True)
    images_fragment = fragments / f"{split}-images.jsonl"
    annotations_fragment = fragments / f"{split}-annotations.jsonl"
    counters: Counter[str] = Counter()
    seen_image_ids: set[int] = set()
    annotation_id = 1

    with images_fragment.open("w", encoding="utf-8") as images_handle, annotations_fragment.open(
        "w", encoding="utf-8"
    ) as annotations_handle:
        for record in json_array_items(annotation_path):
            metadata = metadata_from_record(record, annotation_path)
            source_image = image_root / metadata.name
            if not source_image.is_file():
                raise SplitError(f"Missing canonical image view: {source_image}")
            width, height = jpeg_dimensions(source_image)
            current_image_id = image_id(metadata.name)
            if current_image_id in seen_image_ids:
                raise SplitError(f"Deterministic COCO image-id collision: {metadata.name}")
            seen_image_ids.add(current_image_id)
            json.dump(
                {
                    "id": current_image_id,
                    "file_name": f"images/{split}/{metadata.name}",
                    "width": width,
                    "height": height,
                },
                images_handle,
                separators=(",", ":"),
            )
            images_handle.write("\n")

            yolo_lines: list[str] = []
            for label in record.get("labels") or []:
                if not isinstance(label, Mapping):
                    counters["ignored_non_object_label"] += 1
                    continue
                category = label.get("category")
                box = label.get("box2d")
                if category not in CLASS_TO_YOLO or not isinstance(box, Mapping):
                    counters["ignored_non_detection_label"] += 1
                    continue
                x_center, y_center, box_width, box_height, clipped = normalized_box(
                    box, width, height
                )
                if clipped:
                    counters["clipped_boxes"] += 1
                yolo_lines.append(
                    f"{CLASS_TO_YOLO[str(category)]} {x_center:.10f} {y_center:.10f} "
                    f"{box_width:.10f} {box_height:.10f}"
                )
                x = x_center * width - box_width * width / 2.0
                y = y_center * height - box_height * height / 2.0
                coco_box_width = box_width * width
                coco_box_height = box_height * height
                json.dump(
                    {
                        "id": annotation_id,
                        "image_id": current_image_id,
                        "category_id": CLASS_TO_COCO[str(category)],
                        "bbox": [x, y, coco_box_width, coco_box_height],
                        "area": coco_box_width * coco_box_height,
                        "iscrowd": 0,
                    },
                    annotations_handle,
                    separators=(",", ":"),
                )
                annotations_handle.write("\n")
                annotation_id += 1
                counters["exported_boxes"] += 1
            label_path = yolo_root / f"{Path(metadata.name).stem}.txt"
            label_path.write_text(
                "\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8"
            )
            counters["images"] += 1

    coco_path = output / "coco" / f"instances_{split}.json"
    coco_path.parent.mkdir(parents=True, exist_ok=True)
    with coco_path.open("w", encoding="utf-8") as handle:
        handle.write('{"info":')
        json.dump({"description": EXPORT_VERSION, "version": "1"}, handle, separators=(",", ":"))
        handle.write(',"licenses":[],"categories":')
        json.dump(
            [
                {"id": index + 1, "name": name, "supercategory": "object"}
                for index, name in enumerate(CLASS_NAMES)
            ],
            handle,
            separators=(",", ":"),
        )
        handle.write(',"images":')
        copy_json_lines_as_array(images_fragment, handle)
        handle.write(',"annotations":')
        copy_json_lines_as_array(annotations_fragment, handle)
        handle.write("}\n")
    images_fragment.unlink()
    annotations_fragment.unlink()
    return dict(counters)


def read_split_manifest(split_root: Path, split: str) -> list[dict[str, str]]:
    path = split_root / "manifests" / f"{split}.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Iterable[Mapping[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "image", "yolo_label", "coco_image_id"])
        for row in rows:
            name = row["name"]
            split = row["split"]
            writer.writerow(
                [
                    name,
                    f"images/{split}/{name}",
                    f"yolo/labels/{split}/{Path(name).stem}.txt",
                    image_id(name),
                ]
            )
            count += 1
    return count


def write_unlabeled_rows(path: Path, rows: Iterable[Mapping[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "image", "coco_image_id"])
        for row in rows:
            name = row["name"]
            split = row["split"]
            writer.writerow([name, f"images/{split}/{name}", image_id(name)])
            count += 1
    return count


def nested_order(names: Iterable[str], seed: int) -> list[str]:
    result = sorted(names)
    random.Random(seed).shuffle(result)
    return result


def domain_rows(rows: Iterable[dict[str, str]], domain: str, split: str) -> list[dict[str, str]]:
    timeofday, weather = DOMAINS[domain]
    selected: list[dict[str, str]] = []
    for original in rows:
        if original["timeofday"] == timeofday and original["weather"] == weather:
            row = dict(original)
            row["split"] = split
            selected.append(row)
    return sorted(selected, key=lambda row: row["name"])


def budget_values(domain: str, available: int) -> list[tuple[str, int]]:
    values = [(str(value), value) for value in BASE_BUDGETS if value <= available]
    if domain in {"lighting", "weather"} and MATCHED_POOL_SIZE < available:
        values.append((str(MATCHED_POOL_SIZE), MATCHED_POOL_SIZE))
    if domain == "lighting":
        values.extend((str(value), value) for value in (2000, 5000, 10000) if value < available)
    values.append(("full", available))
    deduplicated = {amount: label for label, amount in values}
    return [(deduplicated[amount], amount) for amount in sorted(deduplicated)]


def export_manifests(split_root: Path, output: Path) -> dict[str, Any]:
    source_rows = {
        split: read_split_manifest(split_root, split) for split in ("train", "val", "test")
    }
    report: dict[str, Any] = {"domains": {}, "budgets": {}}
    for domain in DOMAINS:
        report["domains"][domain] = {}
        for split in ("train", "val", "test"):
            rows = domain_rows(source_rows[split], domain, split)
            count = write_rows(output / "manifests" / "domains" / domain / f"{split}.csv", rows)
            report["domains"][domain][split] = count

    for domain in ("lighting", "weather", "compound"):
        target_rows = domain_rows(source_rows["train"], domain, "train")
        row_by_name = {row["name"]: row for row in target_rows}
        report["budgets"][domain] = {}
        unlabeled_count = write_unlabeled_rows(
            output / "manifests" / "unlabeled" / domain / "full_train_pool.csv",
            target_rows,
        )
        report["budgets"][domain]["unlabeled_pool"] = unlabeled_count
        for seed in SEEDS:
            order = nested_order(row_by_name, seed)
            report["budgets"][domain][str(seed)] = {}
            previous: set[str] = set()
            for label, amount in budget_values(domain, len(order)):
                selected_names = order[:amount]
                selected = set(selected_names)
                if not previous.issubset(selected):
                    raise AssertionError("Internal error: labeled target budgets are not nested")
                count = write_rows(
                    output / "manifests" / "labeled" / domain / f"seed_{seed}" / f"n_{label}.csv",
                    (row_by_name[name] for name in selected_names),
                )
                report["budgets"][domain][str(seed)][label] = count
                previous = selected
    return report


def verify_source(split_root: Path) -> dict[str, str]:
    audit_path = split_root / "metadata" / "audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Canonical split audit not found: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for split in ("train", "val", "test"):
        rows = read_split_manifest(split_root, split)
        actual = membership_hash(row["name"] for row in rows)
        expected = audit["membership_sha256"][split]
        if actual != expected:
            raise SplitError(f"Canonical {split} membership hash mismatch")
    return dict(audit["membership_sha256"])


def prepare(args: argparse.Namespace) -> None:
    if args.output.exists() or args.output.is_symlink():
        raise SplitError(f"Refusing to overwrite existing destination: {args.output}")
    source_hashes = verify_source(args.split_root)
    staging = args.output.parent / f".{args.output.name}.staging-{os.getpid()}-{int(time.time())}"
    staging.mkdir(parents=True)
    try:
        export_report = {
            split: export_split(args.split_root, staging, split)
            for split in ("train", "val", "test")
        }
        manifest_report = export_manifests(args.split_root, staging)
        write_json(
            staging / "metadata" / "classes.json",
            {
                "classes": [
                    {"name": name, "yolo_id": index, "coco_id": index + 1}
                    for index, name in enumerate(CLASS_NAMES)
                ],
                "background": {"coco_id": 0, "note": "Framework-only; not an annotation category"},
            },
        )
        write_json(
            staging / "metadata" / "export.json",
            {
                "export_version": EXPORT_VERSION,
                "source_split": str(args.split_root),
                "source_membership_sha256": source_hashes,
                "splits": export_report,
                "manifests": manifest_report,
                "seeds": list(SEEDS),
                "canonical_labels_preserved": True,
                "unlabeled_manifests_contain_label_paths": False,
            },
        )
        fragments = staging / ".fragments"
        if fragments.exists():
            fragments.rmdir()
        verify_output(staging)
        staging.replace(args.output)
    except BaseException:
        print(f"Preparation failed; incomplete staging retained at {staging}", file=sys.stderr)
        raise
    print(f"Created verified model inputs: {args.output}")


def csv_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def csv_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [row["name"] for row in csv.DictReader(handle)]


def verify_output(output: Path) -> dict[str, Any]:
    metadata = json.loads((output / "metadata" / "export.json").read_text(encoding="utf-8"))
    expected_images = {"train": 55_000, "val": 10_000, "test": 14_863}
    for split, expected in expected_images.items():
        if metadata["splits"][split]["images"] != expected:
            raise SplitError(f"Export metadata count mismatch for {split}")
        label_count = sum(1 for _ in (output / "yolo" / "labels" / split).glob("*.txt"))
        if label_count != expected:
            raise SplitError(f"Expected {expected} YOLO label files for {split}, found {label_count}")
        if not (output / "coco" / f"instances_{split}.json").is_file():
            raise SplitError(f"Missing COCO export for {split}")

    for domain in ("lighting", "weather", "compound"):
        unlabeled = output / "manifests" / "unlabeled" / domain / "full_train_pool.csv"
        with unlabeled.open("r", encoding="utf-8", newline="") as handle:
            fields = csv.DictReader(handle).fieldnames or []
        if "yolo_label" in fields:
            raise SplitError(f"Unlabeled manifest exposes a label path: {unlabeled}")
        for seed in SEEDS:
            previous: set[str] = set()
            budget_dir = output / "manifests" / "labeled" / domain / f"seed_{seed}"
            paths = sorted(
                budget_dir.glob("n_*.csv"),
                key=lambda path: csv_count(path),
            )
            for path in paths:
                current = set(csv_names(path))
                if not previous.issubset(current):
                    raise SplitError(f"Budgets are not nested: {path}")
                previous = current
    return {
        "output": str(output),
        "counts": expected_images,
        "source_membership_sha256": metadata["source_membership_sha256"],
        "clipped_boxes": {
            split: metadata["splits"][split].get("clipped_boxes", 0)
            for split in expected_images
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    args.split_root = args.split_root.resolve()
    args.output = args.output.resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args)
        else:
            print(json.dumps(verify_output(args.output), indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, OSError, SplitError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
