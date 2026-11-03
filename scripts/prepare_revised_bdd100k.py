"""Materialize a fresh, deterministic stratified BDD100K detection dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

from PIL import Image

SEED = 20260916
EXPECTED = {"train": 55000, "test": 14863, "val": 10000}
CLASSES = ("pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign")
ALIASES = {"person": "pedestrian", "motor": "motorcycle", "bike": "bicycle"}
DOMAINS = {"source": ("daytime", "clear"), "lighting": ("night", "clear"), "weather": ("daytime", "rainy"), "compound": ("night", "rainy")}
# Protocol amendment 2026-10-19 (evaluation roles only; physical split membership is unchanged).
# Preserved-val images of these domains join the final held-out test set and are never used for selection.
HOLDOUT_VAL_DOMAINS = ("weather", "compound")
EXPECTED_DOMAINS = {
    "source": {"train": 9804, "final_test": 2650, "selection_val": 1764},
    "lighting": {"train": 18015, "final_test": 4869, "selection_val": 3274},
    "weather": {"train": 1985, "final_test": 537 + 396, "selection_val": 0},
    "compound": {"train": 1738, "final_test": 470 + 286, "selection_val": 0},
}
# One fixed reliable-class roster for every domain (primary metric); the ten-class mean is a sensitivity analysis.
RELIABLE_CLASSES = ("pedestrian", "car", "traffic light", "traffic sign")


def stream_records(path: Path) -> Iterator[dict]:
    """Read a large JSON array with a bounded buffer and strict final delimiter."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = handle.read(8 << 20)
        position = 0
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if position == len(buffer) or buffer[position] != "[":
            raise ValueError(f"Expected JSON array: {path}")
        position += 1
        first = True
        while True:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                buffer = handle.read(8 << 20)
                position = 0
                if not buffer:
                    raise ValueError(f"Truncated array: {path}")
            if buffer[position] == "]":
                if buffer[position + 1:].strip() or handle.read().strip():
                    raise ValueError(f"Trailing JSON data: {path}")
                return
            if not first:
                if buffer[position] != ",":
                    raise ValueError(f"Missing JSON separator: {path}")
                position += 1
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                try:
                    record, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError:
                    extra = handle.read(8 << 20)
                    if not extra:
                        raise ValueError(f"Truncated record: {path}")
                    buffer = buffer[position:] + extra
                    position = 0
            if not isinstance(record, dict):
                raise ValueError(f"Non-object record: {path}")
            yield record
            first = False
            position = end
            if position > (16 << 20):
                buffer, position = buffer[position:], 0


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def apportion(counts: dict, total: int) -> dict:
    """Hamilton allocation: floor quotas, then largest fractional remainders."""
    population = sum(counts.values())
    if not 0 <= total <= population:
        raise ValueError("Requested count outside population")
    numerators = {key: value * total for key, value in counts.items()}
    result = {key: value // population for key, value in numerators.items()}
    ranked = sorted(counts, key=lambda key: (-(numerators[key] % population), key))
    for key in ranked[:total - sum(result.values())]:
        result[key] += 1
    assert sum(result.values()) == total
    assert all(0 <= result[key] <= counts[key] for key in counts)
    return result


def split_membership(rows: list[dict], seed: int) -> tuple[dict[str, str], list[dict]]:
    cells = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cells[(row["timeofday"], row["weather"])][row["scene"]].append(row["name"])
    quotas = apportion({key: sum(map(len, scenes.values())) for key, scenes in cells.items()}, EXPECTED["train"])
    rng = random.Random(seed)
    membership = {}
    allocation = []
    for cell in sorted(cells):
        scenes = cells[cell]
        scene_quotas = apportion({scene: len(names) for scene, names in scenes.items()}, quotas[cell])
        for scene in sorted(scenes):
            names = sorted(scenes[scene])
            rng.shuffle(names)
            train_count = scene_quotas[scene]
            membership.update({name: "train" if index < train_count else "test" for index, name in enumerate(names)})
            allocation.append({"timeofday": cell[0], "weather": cell[1], "scene": scene, "original_labeled": len(names), "train": train_count, "test": len(names) - train_count, "ideal_train": len(names) * 55000 / 69863, "rounding_error": train_count - len(names) * 55000 / 69863})
    assert Counter(membership.values()) == Counter({"train": 55000, "test": 14863})
    return membership, allocation


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"Empty output table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def keep_content_groups_together(membership: dict, rows: list[dict], image_hashes: dict) -> list[dict]:
    """Repair cross-split duplicate content using same-stratum singleton swaps."""
    by_name = {row["name"]: row for row in rows}
    groups = defaultdict(list)
    for name, digest in image_hashes.items():
        groups[digest].append(name)
    singletons = {names[0] for names in groups.values() if len(names) == 1}
    def stratum(name):
        row = by_name[name]
        return row["timeofday"], row["weather"], row["scene"]
    before = Counter((stratum(name), split) for name, split in membership.items())
    repairs = []
    for digest, names in sorted(groups.items()):
        if len({membership[name] for name in names}) < 2:
            continue
        preferred = sorted(("train", "test"), key=lambda split: (-sum(membership[name] == split for name in names), 0 if split == "train" else 1))
        for target in preferred:
            needed = Counter(stratum(name) for name in names if membership[name] != target)
            available = Counter(stratum(name) for name in singletons if membership[name] == target)
            if all(available[key] >= count for key, count in needed.items()):
                break
        else:
            raise ValueError("Duplicate content cannot be grouped while preserving all stratum quotas")
        for name in sorted(names):
            old = membership[name]
            if old == target:
                continue
            donor = next(candidate for candidate in sorted(singletons) if membership[candidate] == target and stratum(candidate) == stratum(name))
            membership[name], membership[donor] = target, old
            repairs.append({"sha256": digest, "moved_image": name, "swapped_image": donor, "kept_split": target, "stratum": stratum(name)})
    assert before == Counter((stratum(name), split) for name, split in membership.items())
    assert all(len({membership[name] for name in names}) == 1 for names in groups.values())
    return repairs


def prepare(verification: Path, destination: Path, seed: int):
    provenance = json.loads(verification.read_text(encoding="utf-8"))
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite dataset: {destination}")
    source_images = defaultdict(dict)
    source_labels = {}
    for row in provenance["files"]:
        name = Path(row["archive_name"]).name
        if name.endswith(".jpg"):
            source_images[row["original_split"]][name] = row
        else:
            source_labels[row["original_split"]] = row
    metadata = defaultdict(list)
    names = defaultdict(set)
    for original_split in ("train", "val"):
        label = source_labels[original_split]
        if sha256(Path(label["local_path"])) != label["sha256"]:
            raise ValueError("Source labels changed since verification")
        for record in stream_records(Path(label["local_path"])):
            name = record["name"]
            if name in names[original_split]:
                raise ValueError(f"Duplicate annotation: {name}")
            if name not in source_images[original_split]:
                raise FileNotFoundError(f"Missing source image: {name}")
            names[original_split].add(name)
            attrs = record.get("attributes") or {}
            metadata[original_split].append({"name": name, **{key: attrs.get(key, "undefined") for key in ("timeofday", "weather", "scene")}})
        print(f"Inventoried {len(names[original_split]):,} labeled {original_split} images", flush=True)
    assert len(names["train"]) == 69863
    assert len(names["val"]) == 10000
    assert not names["train"] & names["val"]
    excluded = sorted(set(source_images["train"]) - names["train"])
    assert len(excluded) == 137
    membership, allocation = split_membership(metadata["train"], seed)
    assert membership == split_membership(list(reversed(metadata["train"])), seed)[0]
    content_repairs = keep_content_groups_together(membership, metadata["train"], {name: source_images["train"][name]["sha256"] for name in names["train"]})
    print(f"Grouped duplicate content with {len(content_repairs)} same-stratum swap(s)", flush=True)
    membership.update({name: "val" for name in names["val"]})
    new_rows = {row["name"]: dict(row, split=membership[row["name"]]) for split in ("train", "val") for row in metadata[split]}
    hashes = defaultdict(set)
    for original_split in ("train", "val"):
        for name in names[original_split]:
            row = source_images[original_split][name]
            new_rows[name]["sha256"] = row["sha256"]
            hashes[membership[name]].add(row["sha256"])
    for a, b in (("train", "test"), ("train", "val"), ("test", "val")):
        overlap = hashes[a] & hashes[b]
        if overlap:
            raise ValueError(f"Exact content leakage {a}/{b}: {len(overlap)} image hashes")
    for split in EXPECTED:
        (destination / "images" / split).mkdir(parents=True, exist_ok=True)
    (destination / "annotations").mkdir()
    (destination / "manifests").mkdir()
    jobs = [(source_images[original_split][name], membership[name]) for original_split in ("train", "val") for name in sorted(names[original_split])]

    def copy_image(job):
        entry, split = job
        name = Path(entry["archive_name"]).name
        target = destination / "images" / split / name
        shutil.copyfile(entry["local_path"], target)
        if sha256(target) != entry["sha256"]:
            raise ValueError(f"Copied content differs: {name}")
        with Image.open(target) as image:
            width, height = image.size
            image.verify()
        return name, width, height

    with ThreadPoolExecutor(max_workers=8) as pool:
        for index, (name, width, height) in enumerate(pool.map(copy_image, jobs), 1):
            new_rows[name].update(width=width, height=height)
            if index % 10000 == 0:
                print(f"Copied and verified {index:,}/{len(jobs):,} images", flush=True)
    annotation_counts = Counter()
    dropped_nondetection = Counter()
    handles = {split: (destination / "annotations" / f"bdd100k_labels_images_{split}.json").open("w", encoding="utf-8") for split in EXPECTED}
    try:
        for handle in handles.values():
            handle.write("[\n")
        for original_split in ("train", "val"):
            for record in stream_records(Path(source_labels[original_split]["local_path"])):
                split = membership[record["name"]]
                detections = []
                for label in record.get("labels") or []:
                    category = ALIASES.get(label.get("category"), label.get("category"))
                    if category in CLASSES and isinstance(label.get("box2d"), dict):
                        detections.append(dict(label, category=category))
                    else:
                        dropped_nondetection[split] += 1
                item = {key: value for key, value in record.items() if key != "labels"}
                item["labels"] = detections
                if annotation_counts[split]:
                    handles[split].write(",\n")
                json.dump(item, handles[split], ensure_ascii=False, separators=(",", ":"))
                annotation_counts[split] += 1
        for handle in handles.values():
            handle.write("\n]\n")
    finally:
        for handle in handles.values():
            handle.close()
    assert dict(annotation_counts) == EXPECTED
    for split in EXPECTED:
        rows = [new_rows[name] for name in sorted(new_rows) if membership[name] == split]
        write_csv(destination / "manifests" / f"{split}.csv", rows)
    write_csv(destination / "manifests" / "stratification.csv", allocation)
    (destination / "manifests" / "duplicate_content_repairs.json").write_text(json.dumps(content_repairs, indent=2), encoding="utf-8")
    (destination / "manifests" / "excluded_unannotated.txt").write_text("\n".join(excluded) + "\n", encoding="utf-8")
    census = []
    for split in EXPECTED:
        counts = Counter((row["timeofday"], row["weather"]) for row in new_rows.values() if row["split"] == split)
        census.extend({"split": split, "timeofday": key[0], "weather": key[1], "images": count} for key, count in sorted(counts.items()))
    write_csv(destination / "manifests" / "domain_census.csv", census)
    integrity = validate(destination)
    result = {"seed": seed, "algorithm": "Hierarchical Hamilton allocation by (timeofday, weather), then scene; sorted cells, scenes and basenames; Python random.Random(seed) shuffle per stratum; exact-content groups repaired by deterministic same-stratum singleton swaps", "source_dataset": provenance["dataset_url"], "source_version": provenance["version"], "source_verification": provenance["verification"], "source_label_sha256": {key: row["sha256"] for key, row in source_labels.items()}, "counts": EXPECTED, "excluded_unannotated": len(excluded), "excluded_public_unlabeled_test": 20000, "classes": CLASSES, "aliases": ALIASES, "annotation_policy": "Detection only; original box coordinates/attributes preserved, no clipping or dropping of invalid geometry; lane/drivable-area polygons excluded", "images": "Independent byte-identical copies, not links", "duplicate_content_repairs": content_repairs, "within_split_repeated_content_images": {split: EXPECTED[split] - len(hashes[split]) for split in EXPECTED}, "nondetection_annotations_omitted": dict(dropped_nondetection), "dimensions": dict(Counter(f'{r["width"]}x{r["height"]}' for r in new_rows.values())), "max_abs_joint_stratum_rounding_error": max(abs(row["rounding_error"]) for row in allocation), "domain_census": census, "integrity": integrity, "manifest_sha256": {split: sha256(destination / "manifests" / f"{split}.csv") for split in EXPECTED}}
    (destination / "dataset_metadata.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "COMPLETE").write_text("All dataset invariants passed. Treat test membership as frozen.\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def validate(root: Path, rehash: bool = False) -> dict:
    names = {}
    hashes = {}
    metadata_path = root / "dataset_metadata.json"
    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
    for split, count in EXPECTED.items():
        if saved_metadata:
            assert sha256(root / "manifests" / f"{split}.csv") == saved_metadata["manifest_sha256"][split], f"Manifest modified: {split}"
        with (root / "manifests" / f"{split}.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        manifest = [row["name"] for row in rows]
        names[split] = set(manifest)
        hashes[split] = {row["sha256"] for row in rows}
        images = {path.name for path in (root / "images" / split).glob("*.jpg")}
        annotations = [record["name"] for record in stream_records(root / "annotations" / f"bdd100k_labels_images_{split}.json")]
        assert len(rows) == len(names[split]) == len(images) == len(annotations) == len(set(annotations)) == count
        assert images == names[split] == set(annotations)
        if rehash:
            with ThreadPoolExecutor(max_workers=12) as pool:
                actual = list(pool.map(sha256, (root / "images" / split / row["name"] for row in rows)))
            assert all(value == row["sha256"] for value, row in zip(actual, rows)), f"Image content modified: {split}"
            print(f"Rehashed all {count:,} {split} images", flush=True)
    pairs = (("train", "test"), ("train", "val"), ("test", "val"))
    for a, b in pairs:
        assert not names[a] & names[b], f"Basename leakage: {a}/{b}"
        assert not hashes[a] & hashes[b], f"Exact content leakage: {a}/{b}"
    return {"passed": True, "image_annotation_manifest_counts": EXPECTED, "basename_overlap": {f"{a}/{b}": 0 for a, b in pairs}, "sha256_content_overlap": {f"{a}/{b}": 0 for a, b in pairs}, "validation_membership": "Preserved official val exactly", "test_label_access": "Construction and structural validation only; no model evaluation or test-class EDA", "limitation": "Exact hashes do not detect near-duplicate images or geographic/video-level dependence"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified-source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate(args.output_dir, rehash=True), indent=2))
    else:
        if args.verified_source is None:
            parser.error("--verified-source is required for construction")
        prepare(args.verified_source, args.output_dir, args.seed)
