"""Audit and materialize the versioned BDD100K research split.

The tool keeps the raw BDD100K layout unchanged.  It creates one canonical,
deterministic 55,000/14,863 split of the labeled original training records and
preserves the official 10,000-image validation split.  Image views contain
hardlinks when possible and relative symlinks otherwise; image bytes are never
copied.

Examples:

    python bdd100k_research_split.py audit --decode-images
    python bdd100k_research_split.py materialize --decode-images
    python bdd100k_research_split.py verify
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import struct
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGES_ROOT = ROOT / "data" / "bdd100k" / "bdd100k" / "images" / "100k"
DEFAULT_LABELS_ROOT = ROOT / "data" / "labels_raw"
DEFAULT_OUTPUT = ROOT / "data" / "research_split_v1"

ALGORITHM_VERSION = "bdd100k-research-split-v1"
DEFAULT_SEED = 20260916
TRAIN_TOTAL = 55_000
TEST_TOTAL = 14_863
VAL_TOTAL = 10_000
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg"})

Cell = tuple[str, str]
Stratum = tuple[str, str, str]

# Exact cells published in docs/proposal_dataset_aligned.docx.  The raw count
# for each cell must equal train + held-out test; validation is unchanged.
PUBLISHED_CELL_QUOTAS: dict[Cell, tuple[int, int, int]] = {
    ("night", "clear"): (18_015, 4_869, 3_274),
    ("daytime", "clear"): (9_804, 2_650, 1_764),
    ("daytime", "overcast"): (5_943, 1_608, 1_039),
    ("daytime", "partly cloudy"): (3_355, 907, 638),
    ("daytime", "snowy"): (2_252, 610, 422),
    ("daytime", "rainy"): (1_985, 537, 396),
    ("night", "snowy"): (1_771, 478, 273),
    ("night", "rainy"): (1_738, 470, 286),
    ("dawn/dusk", "clear"): (1_577, 427, 307),
    ("night", "overcast"): (56, 16, 18),
    ("night", "foggy"): (53, 14, 7),
    ("daytime", "foggy"): (38, 10, 5),
}

DOMAINS: dict[str, Cell] = {
    "source": ("daytime", "clear"),
    "lighting": ("night", "clear"),
    "weather": ("daytime", "rainy"),
    "compound": ("night", "rainy"),
}


class SplitError(ValueError):
    """Raised when source data or a generated split violates the protocol."""


@dataclass(frozen=True)
class RecordMetadata:
    name: str
    timeofday: str
    weather: str
    scene: str

    @property
    def cell(self) -> Cell:
        return (self.timeofday, self.weather)

    @property
    def stratum(self) -> Stratum:
        return (self.timeofday, self.weather, self.scene)


@dataclass(frozen=True)
class AuditResult:
    metadata: tuple[RecordMetadata, ...]
    image_index: Mapping[str, Path]
    unlabeled_images: tuple[str, ...]
    cell_counts: Mapping[Cell, int]
    scene_counts: Mapping[Stratum, int]
    box_counts: Mapping[str, int]


def json_array_items(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    """Stream objects from a top-level JSON array."""

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        started = False
        eof = False
        while True:
            if not eof and len(buffer) - position < chunk_size:
                buffer = buffer[position:]
                position = 0
                piece = handle.read(chunk_size)
                if piece:
                    buffer += piece
                else:
                    eof = True

            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    if eof:
                        raise SplitError(f"Empty JSON file: {path}")
                    continue
                if buffer[position] != "[":
                    raise SplitError(f"Expected a top-level JSON array: {path}")
                position += 1
                started = True
                continue

            while position < len(buffer) and (
                buffer[position].isspace() or buffer[position] == ","
            ):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            if position >= len(buffer):
                if eof:
                    raise SplitError(f"Unexpected end of JSON array: {path}")
                continue
            try:
                item, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                buffer = buffer[position:]
                position = 0
                piece = handle.read(chunk_size)
                if piece:
                    buffer += piece
                else:
                    eof = True
                continue
            if not isinstance(item, dict):
                raise SplitError(f"Expected object in {path}, got {type(item).__name__}")
            yield item
            position = end


def annotation_path(labels_root: Path, split: str) -> Path:
    return labels_root / f"bdd100k_labels_images_{split}.json"


def normalize_attribute(value: Any) -> str:
    if value is None or value == "":
        return "undefined"
    return str(value).strip().lower()


def metadata_from_record(record: Mapping[str, Any], path: Path) -> RecordMetadata:
    name = record.get("name")
    if not isinstance(name, str) or not name:
        raise SplitError(f"Annotation record without a valid name in {path}")
    if Path(name).name != name:
        raise SplitError(f"Annotation name must be a basename, got {name!r} in {path}")
    attributes = record.get("attributes")
    if not isinstance(attributes, Mapping):
        attributes = {}
    return RecordMetadata(
        name=name,
        timeofday=normalize_attribute(attributes.get("timeofday")),
        weather=normalize_attribute(attributes.get("weather")),
        scene=normalize_attribute(attributes.get("scene")),
    )


def index_images(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    result: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        previous = result.get(path.name)
        if previous is not None:
            duplicates[path.name].extend((previous, path))
        else:
            result[path.name] = path
    if duplicates:
        examples = "; ".join(
            f"{name}: {', '.join(str(item) for item in dict.fromkeys(paths))}"
            for name, paths in sorted(duplicates.items())[:5]
        )
        raise SplitError(f"Duplicate image basenames under {root}: {examples}")
    return result


def jpeg_dimensions(path: Path) -> tuple[int, int]:
    """Read JPEG dimensions from its frame header using the standard library."""

    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            raise SplitError(f"Not a JPEG image: {path}")
        while True:
            byte = handle.read(1)
            if not byte:
                break
            if byte != b"\xff":
                continue
            while byte == b"\xff":
                byte = handle.read(1)
            marker = byte[0]
            if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue
            length_bytes = handle.read(2)
            if len(length_bytes) != 2:
                break
            length = struct.unpack(">H", length_bytes)[0]
            if length < 2:
                raise SplitError(f"Invalid JPEG segment in {path}")
            if marker in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                payload = handle.read(5)
                if len(payload) != 5:
                    break
                height, width = struct.unpack(">HH", payload[1:])
                if width <= 0 or height <= 0:
                    raise SplitError(f"Invalid JPEG dimensions in {path}")
                return width, height
            handle.seek(length - 2, os.SEEK_CUR)
    raise SplitError(f"JPEG frame header not found: {path}")


def full_decode_image(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as error:
        raise SplitError(
            "--decode-images requires Pillow in the prepared environment"
        ) from error
    try:
        with Image.open(path) as image:
            image.load()
            return image.size
    except Exception as error:  # Pillow exposes several decoder-specific errors.
        raise SplitError(f"Image decode failed for {path}: {error}") from error


def audit_boxes(
    record: Mapping[str, Any], width: int, height: int, counters: Counter[str], path: Path
) -> None:
    labels = record.get("labels") or []
    if not isinstance(labels, list):
        raise SplitError(f"labels must be a list for {record.get('name')} in {path}")
    for label in labels:
        if not isinstance(label, Mapping):
            counters["non_object_labels"] += 1
            continue
        box = label.get("box2d")
        if box is None:
            counters["labels_without_box2d"] += 1
            continue
        if not isinstance(box, Mapping):
            raise SplitError(f"box2d must be an object for {record.get('name')} in {path}")
        try:
            x1, y1, x2, y2 = (float(box[key]) for key in ("x1", "y1", "x2", "y2"))
        except (KeyError, TypeError, ValueError) as error:
            raise SplitError(f"Invalid box coordinates for {record.get('name')} in {path}") from error
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise SplitError(f"Nonfinite box for {record.get('name')} in {path}")
        if x2 <= x1 or y2 <= y1:
            raise SplitError(f"Degenerate box for {record.get('name')} in {path}: {box}")
        counters["box2d"] += 1
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            counters["out_of_bounds_box2d"] += 1


def audit_annotations(
    annotations: Path,
    image_index: Mapping[str, Path],
    *,
    decode_images: bool,
    expected_count: int,
) -> AuditResult:
    if not annotations.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotations}")
    metadata: list[RecordMetadata] = []
    names: set[str] = set()
    cell_counts: Counter[Cell] = Counter()
    scene_counts: Counter[Stratum] = Counter()
    box_counts: Counter[str] = Counter()
    missing: list[str] = []

    for record in json_array_items(annotations):
        item = metadata_from_record(record, annotations)
        if item.name in names:
            raise SplitError(f"Duplicate annotation record: {item.name} in {annotations}")
        names.add(item.name)
        image_path = image_index.get(item.name)
        if image_path is None:
            missing.append(item.name)
            continue
        header_size = jpeg_dimensions(image_path)
        if decode_images:
            decoded_size = full_decode_image(image_path)
            if decoded_size != header_size:
                raise SplitError(
                    f"Header/decoder dimension mismatch for {image_path}: "
                    f"{header_size} != {decoded_size}"
                )
        audit_boxes(record, *header_size, box_counts, annotations)
        metadata.append(item)
        cell_counts[item.cell] += 1
        scene_counts[item.stratum] += 1

    if missing:
        raise SplitError(
            f"{len(missing)} annotation records have no image under the selected root; "
            f"examples: {', '.join(sorted(missing)[:10])}"
        )
    if len(metadata) != expected_count:
        raise SplitError(
            f"Expected {expected_count:,} records in {annotations}, found {len(metadata):,}"
        )
    extra = sorted(set(image_index) - names)
    return AuditResult(
        metadata=tuple(metadata),
        image_index=image_index,
        unlabeled_images=tuple(extra),
        cell_counts=dict(cell_counts),
        scene_counts=dict(scene_counts),
        box_counts=dict(box_counts),
    )


def largest_remainder(counts: Mapping[Any, int], total: int) -> dict[Any, int]:
    """Allocate an integer total proportionally with deterministic tie breaking."""

    if total < 0:
        raise SplitError(f"Allocation total must be nonnegative, got {total}")
    population = sum(counts.values())
    if total > population:
        raise SplitError(f"Cannot allocate {total} items from a population of {population}")
    if population == 0:
        if total:
            raise SplitError("Cannot allocate a nonzero total from an empty population")
        return {key: 0 for key in counts}
    allocation = {key: (value * total) // population for key, value in counts.items()}
    remaining = total - sum(allocation.values())
    order = sorted(
        counts,
        key=lambda key: (-(counts[key] * total % population), tuple_key(key)),
    )
    for key in order[:remaining]:
        allocation[key] += 1
    return allocation


def tuple_key(value: Any) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    return (str(value),)


def build_cell_train_quotas(cell_counts: Mapping[Cell, int]) -> dict[Cell, int]:
    quotas: dict[Cell, int] = {}
    for cell, (train_count, test_count, _) in PUBLISHED_CELL_QUOTAS.items():
        actual = cell_counts.get(cell, 0)
        expected = train_count + test_count
        if actual != expected:
            raise SplitError(
                f"Published cell {cell} requires {expected:,} raw records, found {actual:,}"
            )
        quotas[cell] = train_count
    remaining_counts = {
        cell: count for cell, count in cell_counts.items() if cell not in quotas
    }
    remaining_total = TRAIN_TOTAL - sum(quotas.values())
    quotas.update(largest_remainder(remaining_counts, remaining_total))
    if sum(quotas.values()) != TRAIN_TOTAL:
        raise AssertionError("Internal error: train cell quotas do not sum to protocol total")
    return quotas


def validate_val_counts(cell_counts: Mapping[Cell, int]) -> None:
    for cell, (_, _, expected_val) in PUBLISHED_CELL_QUOTAS.items():
        actual = cell_counts.get(cell, 0)
        if actual != expected_val:
            raise SplitError(
                f"Published validation cell {cell} requires {expected_val:,} records, "
                f"found {actual:,}"
            )


def stratum_seed(base_seed: int, stratum: Stratum) -> int:
    value = f"{ALGORITHM_VERSION}\0{base_seed}\0" + "\0".join(stratum)
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:16], "big")


def assign_train_test(
    metadata: Sequence[RecordMetadata], base_seed: int
) -> tuple[set[str], set[str], dict[Cell, int], dict[Stratum, int]]:
    cell_counts = Counter(item.cell for item in metadata)
    cell_quotas = build_cell_train_quotas(cell_counts)
    by_stratum: dict[Stratum, list[str]] = defaultdict(list)
    for item in metadata:
        by_stratum[item.stratum].append(item.name)

    stratum_quotas: dict[Stratum, int] = {}
    strata_by_cell: dict[Cell, dict[Stratum, int]] = defaultdict(dict)
    for stratum, names in by_stratum.items():
        strata_by_cell[stratum[:2]][stratum] = len(names)
    for cell in sorted(strata_by_cell):
        stratum_quotas.update(largest_remainder(strata_by_cell[cell], cell_quotas[cell]))

    train: set[str] = set()
    test: set[str] = set()
    for stratum in sorted(by_stratum):
        names = sorted(by_stratum[stratum])
        random.Random(stratum_seed(base_seed, stratum)).shuffle(names)
        cutoff = stratum_quotas[stratum]
        train.update(names[:cutoff])
        test.update(names[cutoff:])
    if len(train) != TRAIN_TOTAL or len(test) != TEST_TOTAL:
        raise SplitError(
            f"Generated counts are {len(train):,}/{len(test):,}, expected "
            f"{TRAIN_TOTAL:,}/{TEST_TOTAL:,}"
        )
    if train & test or train | test != {item.name for item in metadata}:
        raise AssertionError("Internal error: generated membership is not a partition")
    return train, test, cell_quotas, stratum_quotas


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def membership_hash(names: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def image_index_hash(index: Mapping[str, Path]) -> str:
    digest = hashlib.sha256()
    for name, path in sorted(index.items()):
        stat = path.stat()
        digest.update(f"{name}\t{stat.st_size}\n".encode("utf-8"))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_json_array_stream(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write("[\n")
        first = True
        for record in records:
            if not first:
                handle.write(",\n")
            json.dump(record, handle, ensure_ascii=False, separators=(",", ":"))
            first = False
            count += 1
        handle.write("\n]\n")
    temporary.replace(path)
    return count


def selected_records(path: Path, names: set[str]) -> Iterator[Mapping[str, Any]]:
    found: set[str] = set()
    for record in json_array_items(path):
        name = record.get("name")
        if name in names:
            found.add(str(name))
            yield record
    missing = names - found
    if missing:
        raise SplitError(
            f"Failed to export {len(missing)} selected annotation records from {path}"
        )


def create_image_link(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise SplitError(f"Refusing to overwrite image view entry: {destination}")
    if mode in {"auto", "hardlink"}:
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    relative_source = os.path.relpath(source, destination.parent)
    destination.symlink_to(relative_source)
    return "symlink"


def cell_label(cell: Cell) -> str:
    return f"{cell[0]}|{cell[1]}"


def stratum_label(stratum: Stratum) -> str:
    return "|".join(stratum)


def census_payload(
    train_metadata: Sequence[RecordMetadata],
    val_metadata: Sequence[RecordMetadata],
    train_names: set[str],
    test_names: set[str],
) -> dict[str, Any]:
    by_split_cell: dict[str, Counter[Cell]] = {
        "train": Counter(item.cell for item in train_metadata if item.name in train_names),
        "test": Counter(item.cell for item in train_metadata if item.name in test_names),
        "val": Counter(item.cell for item in val_metadata),
    }
    by_split_stratum: dict[str, Counter[Stratum]] = {
        "train": Counter(item.stratum for item in train_metadata if item.name in train_names),
        "test": Counter(item.stratum for item in train_metadata if item.name in test_names),
        "val": Counter(item.stratum for item in val_metadata),
    }
    return {
        "cells": {
            split: {cell_label(key): value for key, value in sorted(counts.items())}
            for split, counts in by_split_cell.items()
        },
        "strata": {
            split: {stratum_label(key): value for key, value in sorted(counts.items())}
            for split, counts in by_split_stratum.items()
        },
    }


def domain_names(item: RecordMetadata) -> str:
    return ";".join(name for name, cell in DOMAINS.items() if item.cell == cell)


def write_manifest(
    output: Path,
    split: str,
    metadata: Sequence[RecordMetadata],
    selected_names: set[str],
) -> None:
    path = output / "manifests" / f"{split}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["name", "image", "annotation_file", "timeofday", "weather", "scene", "domains"]
        )
        for item in sorted(metadata, key=lambda value: value.name):
            if item.name not in selected_names:
                continue
            writer.writerow(
                [
                    item.name,
                    f"images/{split}/{item.name}",
                    f"annotations/bdd100k_labels_images_{split}.json",
                    item.timeofday,
                    item.weather,
                    item.scene,
                    domain_names(item),
                ]
            )


def audit_sources(args: argparse.Namespace) -> tuple[AuditResult, AuditResult]:
    train_images = index_images(args.images_root / "train")
    val_images = index_images(args.images_root / "val")
    train_audit = audit_annotations(
        annotation_path(args.labels_root, "train"),
        train_images,
        decode_images=args.decode_images,
        expected_count=TRAIN_TOTAL + TEST_TOTAL,
    )
    val_audit = audit_annotations(
        annotation_path(args.labels_root, "val"),
        val_images,
        decode_images=args.decode_images,
        expected_count=VAL_TOTAL,
    )
    if len(train_images) != 70_000:
        raise SplitError(f"Expected 70,000 original train images, found {len(train_images):,}")
    if len(val_images) != VAL_TOTAL:
        raise SplitError(f"Expected {VAL_TOTAL:,} validation images, found {len(val_images):,}")
    if len(train_audit.unlabeled_images) != 137:
        raise SplitError(
            f"Expected 137 unlabeled original-train images, found "
            f"{len(train_audit.unlabeled_images):,}"
        )
    if val_audit.unlabeled_images:
        raise SplitError(
            f"Validation has {len(val_audit.unlabeled_images)} images without annotation records"
        )
    validate_val_counts(val_audit.cell_counts)
    return train_audit, val_audit


def audit_summary(
    train_audit: AuditResult,
    val_audit: AuditResult,
    train_names: set[str],
    test_names: set[str],
    seed: int,
    decode_images: bool,
) -> dict[str, Any]:
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "seed": seed,
        "full_image_decode": decode_images,
        "counts": {
            "raw_train_images": len(train_audit.image_index),
            "labeled_train_records": len(train_audit.metadata),
            "excluded_unlabeled_train_images": len(train_audit.unlabeled_images),
            "research_train": len(train_names),
            "research_test": len(test_names),
            "official_val": len(val_audit.metadata),
        },
        "box_audit": {
            "raw_train": train_audit.box_counts,
            "official_val": val_audit.box_counts,
        },
        "membership_sha256": {
            "train": membership_hash(train_names),
            "test": membership_hash(test_names),
            "val": membership_hash(item.name for item in val_audit.metadata),
        },
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    train_audit, val_audit = audit_sources(args)
    train_names, test_names, _, _ = assign_train_test(train_audit.metadata, args.seed)
    summary = audit_summary(
        train_audit, val_audit, train_names, test_names, args.seed, args.decode_images
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def materialize(args: argparse.Namespace) -> None:
    if args.output.exists() or args.output.is_symlink():
        raise SplitError(f"Refusing to overwrite existing destination: {args.output}")
    train_audit, val_audit = audit_sources(args)
    train_names, test_names, cell_quotas, stratum_quotas = assign_train_test(
        train_audit.metadata, args.seed
    )
    val_names = {item.name for item in val_audit.metadata}
    parent = args.output.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{args.output.name}.staging-{os.getpid()}-{int(time.time())}"
    if staging.exists():
        raise SplitError(f"Staging destination already exists: {staging}")
    staging.mkdir()

    try:
        annotation_train = annotation_path(args.labels_root, "train")
        annotation_val = annotation_path(args.labels_root, "val")
        exported = {
            "train": write_json_array_stream(
                staging / "annotations" / "bdd100k_labels_images_train.json",
                selected_records(annotation_train, train_names),
            ),
            "test": write_json_array_stream(
                staging / "annotations" / "bdd100k_labels_images_test.json",
                selected_records(annotation_train, test_names),
            ),
            "val": write_json_array_stream(
                staging / "annotations" / "bdd100k_labels_images_val.json",
                selected_records(annotation_val, val_names),
            ),
        }
        expected = {"train": TRAIN_TOTAL, "test": TEST_TOTAL, "val": VAL_TOTAL}
        if exported != expected:
            raise SplitError(f"Annotation export counts {exported} do not match {expected}")

        link_counts: Counter[str] = Counter()
        memberships = {"train": train_names, "test": test_names, "val": val_names}
        indexes = {
            "train": train_audit.image_index,
            "test": train_audit.image_index,
            "val": val_audit.image_index,
        }
        for split in ("train", "test", "val"):
            for name in sorted(memberships[split]):
                kind = create_image_link(
                    indexes[split][name], staging / "images" / split / name, args.link_mode
                )
                link_counts[kind] += 1

        write_manifest(staging, "train", train_audit.metadata, train_names)
        write_manifest(staging, "test", train_audit.metadata, test_names)
        write_manifest(staging, "val", val_audit.metadata, val_names)
        write_json(
            staging / "manifests" / "excluded_unlabeled_train_images.json",
            list(train_audit.unlabeled_images),
        )
        write_json(
            staging / "metadata" / "census.json",
            census_payload(train_audit.metadata, val_audit.metadata, train_names, test_names),
        )
        write_json(
            staging / "metadata" / "quotas.json",
            {
                "cell_train": {
                    cell_label(key): value for key, value in sorted(cell_quotas.items())
                },
                "stratum_train": {
                    stratum_label(key): value
                    for key, value in sorted(stratum_quotas.items())
                },
                "published_cells": {
                    cell_label(key): {"train": value[0], "test": value[1], "val": value[2]}
                    for key, value in sorted(PUBLISHED_CELL_QUOTAS.items())
                },
            },
        )
        summary = audit_summary(
            train_audit, val_audit, train_names, test_names, args.seed, args.decode_images
        )
        summary["link_counts"] = dict(link_counts)
        write_json(staging / "metadata" / "audit.json", summary)
        write_json(
            staging / "metadata" / "provenance.json",
            {
                "algorithm_version": ALGORITHM_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "seed": args.seed,
                "source": {
                    "images_root": str(args.images_root.absolute()),
                    "labels_root": str(args.labels_root.absolute()),
                    "annotation_sha256": {
                        "train": sha256_file(annotation_train),
                        "val": sha256_file(annotation_val),
                    },
                    "image_index_sha256": {
                        "train": image_index_hash(train_audit.image_index),
                        "val": image_index_hash(val_audit.image_index),
                    },
                },
            },
        )
        verify_output(staging)
        staging.replace(args.output)
    except BaseException:
        print(f"Materialization failed; incomplete staging retained at {staging}", file=sys.stderr)
        raise
    print(f"Created verified split: {args.output}")


def read_manifest_names(path: Path) -> tuple[list[str], list[str]]:
    names: list[str] = []
    images: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"name", "image", "annotation_file"}
        if not expected.issubset(reader.fieldnames or []):
            raise SplitError(f"Manifest missing required columns: {path}")
        for row in reader:
            names.append(row["name"])
            images.append(row["image"])
    return names, images


def verify_output(output: Path) -> dict[str, Any]:
    expected = {"train": TRAIN_TOTAL, "test": TEST_TOTAL, "val": VAL_TOTAL}
    memberships: dict[str, set[str]] = {}
    for split, count in expected.items():
        manifest_path = output / "manifests" / f"{split}.csv"
        names, image_paths = read_manifest_names(manifest_path)
        if len(names) != count or len(set(names)) != count:
            raise SplitError(
                f"Manifest {split} contains {len(names):,} rows and {len(set(names)):,} unique names; "
                f"expected {count:,}"
            )
        expected_prefix = f"images/{split}/"
        if any(not value.startswith(expected_prefix) for value in image_paths):
            raise SplitError(f"Manifest {split} contains a non-view image path")
        for relative in image_paths:
            path = output / relative
            if not path.is_file():
                raise SplitError(f"Missing or broken image view entry: {path}")
        annotation_names = {
            metadata_from_record(record, output / "annotations").name
            for record in json_array_items(
                output / "annotations" / f"bdd100k_labels_images_{split}.json"
            )
        }
        if annotation_names != set(names):
            raise SplitError(f"Manifest/annotation membership mismatch for {split}")
        memberships[split] = set(names)
    if memberships["train"] & memberships["test"]:
        raise SplitError("Research train and held-out test overlap")
    if memberships["train"] & memberships["val"]:
        raise SplitError("Research train and official validation overlap")
    if memberships["test"] & memberships["val"]:
        raise SplitError("Held-out test and official validation overlap")
    audit_path = output / "metadata" / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for split in expected:
        actual_hash = membership_hash(memberships[split])
        if actual_hash != audit["membership_sha256"][split]:
            raise SplitError(f"Membership hash mismatch for {split}")
    census = json.loads((output / "metadata" / "census.json").read_text(encoding="utf-8"))
    for cell, quotas in PUBLISHED_CELL_QUOTAS.items():
        label = cell_label(cell)
        for split, expected_count in zip(("train", "test", "val"), quotas):
            actual = census["cells"][split].get(label, 0)
            if actual != expected_count:
                raise SplitError(
                    f"Census mismatch for {split} {label}: {actual} != {expected_count}"
                )
    return {
        "output": str(output),
        "counts": expected,
        "membership_sha256": audit["membership_sha256"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("audit", "materialize", "verify"), help="Operation to perform"
    )
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--decode-images",
        action="store_true",
        help="Fully decode every labeled image with Pillow in addition to JPEG header checks",
    )
    parser.add_argument(
        "--link-mode",
        choices=("auto", "hardlink", "symlink"),
        default="auto",
        help="Image view link type; auto tries hardlinks and falls back to relative symlinks",
    )
    args = parser.parse_args(argv)
    args.images_root = args.images_root.resolve()
    args.labels_root = args.labels_root.resolve()
    args.output = args.output.resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "audit":
            run_audit(args)
        elif args.command == "materialize":
            materialize(args)
        else:
            print(json.dumps(verify_output(args.output), indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, OSError, SplitError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
