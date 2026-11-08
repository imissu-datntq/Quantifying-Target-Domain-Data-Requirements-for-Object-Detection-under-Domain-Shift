"""Record label source/version and cross-check a fixed label sample against an official release.

The Kaggle CRC32/SHA256 check only proves that local files equal the Kaggle mirror.
`fingerprint` records what the mirror's labels are (schema, vocabulary, counts,
hashes) and freezes a seeded sample of records. `compare` checks that sample against
an official BDD100K label download (zip or extracted JSON), after verifying the zip
MD5 published in the official BDD100K documentation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

from prepare_revised_bdd100k import ALIASES, sha256, stream_records

# MD5 values published in github.com/bdd100k/bdd100k doc/source/download.rst (master, fetched 2026-10-19).
OFFICIAL_MD5 = {"e21be3e7d6a07ee439faf61e769667e4": "BDD100K Labels (2018 release, bdd100k_labels_release; deprecated by det_20)",
                "b86a3e1b7edbcad421b7dad2b3987c94": "BDD100K Detection 2020 Labels (det_20)"}
SCHEMA_2018 = {"person", "bike", "motor"}
SCHEMA_2020 = {"pedestrian", "bicycle", "motorcycle", "other person", "other vehicle", "trailer"}


def canonical_digest(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def detection_boxes(record: dict) -> list[tuple]:
    boxes = []
    for label in record.get("labels") or []:
        if isinstance(label.get("box2d"), dict):
            box = tuple(round(float(label["box2d"][key]), 3) for key in ("x1", "y1", "x2", "y2"))
            boxes.append((ALIASES.get(label["category"], label["category"]), box))
    return sorted(boxes)


def fingerprint(paths: dict[str, Path], per_split: int, seed: int, output: Path) -> dict:
    result = {"files": {}, "sample_seed": seed, "sample": {}}
    for split, path in paths.items():
        categories, attributes, label_keys = Counter(), Counter(), Counter()
        records, names = 0, []
        for record in stream_records(path):
            records += 1
            names.append(record["name"])
            for key, value in (record.get("attributes") or {}).items():
                attributes[f"{key}={value}"] += 1
            for label in record.get("labels") or []:
                categories[label.get("category")] += 1
                label_keys.update(label.keys())
        vocabulary = set(categories)
        schema = "2018 release" if SCHEMA_2018 & vocabulary else "2020 det_20" if SCHEMA_2020 & vocabulary else "unknown"
        chosen = set(random.Random(f"{seed}/{split}").sample(sorted(names), per_split))
        sample = {record["name"]: {"canonical_sha256": canonical_digest(record),
                                   "attributes": record.get("attributes"), "boxes": detection_boxes(record)}
                  for record in stream_records(path) if record["name"] in chosen}
        result["files"][split] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path),
                                  "records": records, "inferred_schema": schema,
                                  "categories": dict(categories.most_common()), "label_keys": dict(label_keys),
                                  "image_attributes": dict(sorted(attributes.items()))}
        result["sample"][split] = sample
        print(f"{split}: {records:,} records, schema {schema}, sampled {len(sample)}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "label_fingerprint.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def official_records(path: Path):
    """Yield (member, record) from an official zip or JSON file; zip MD5 must be published."""
    if path.suffix == ".zip":
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest not in OFFICIAL_MD5:
            raise ValueError(f"Zip MD5 {digest} is not a published official BDD100K label MD5")
        print(f"Official archive verified: {OFFICIAL_MD5[digest]}", flush=True)
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if member.endswith(".json"):
                    temporary = Path(archive.extract(member, path.parent / "_extracted"))
                    yield from ((member, record) for record in stream_records(temporary))
    else:
        yield from ((path.name, record) for record in stream_records(path))


def compare(fingerprint_path: Path, official: list[Path], output: Path) -> dict:
    saved = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    wanted = {name: entry for split in saved["sample"].values() for name, entry in split.items()}
    found = {}
    for path in official:
        for member, record in official_records(path):
            if record.get("name") in wanted:
                found[record["name"]] = (member, record)
    rows = []
    for name, entry in sorted(wanted.items()):
        if name not in found:
            rows.append({"name": name, "present": False})
            continue
        member, record = found[name]
        official_boxes = [[category, list(box)] for category, box in detection_boxes(record)]
        local_boxes = [[category, list(box)] for category, box in entry["boxes"]]
        rows.append({"name": name, "present": True, "member": member,
                     "byte_identical_record": canonical_digest(record) == entry["canonical_sha256"],
                     "same_attributes": (record.get("attributes") or {}) == (entry["attributes"] or {}),
                     "same_boxes": official_boxes == local_boxes,
                     "local_boxes": len(local_boxes), "official_boxes": len(official_boxes)})
    present = [row for row in rows if row["present"]]
    summary = {"sampled": len(rows), "present_in_official": len(present),
               "byte_identical": sum(row["byte_identical_record"] for row in present),
               "same_attributes": sum(row["same_attributes"] for row in present),
               "same_boxes": sum(row["same_boxes"] for row in present), "rows": rows}
    output.mkdir(parents=True, exist_ok=True)
    (output / "official_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("fingerprint")
    first.add_argument("--source-verified", type=Path, default=Path("data/raw/kaggle_verified/source_verified.json"))
    first.add_argument("--per-split", type=int, default=200)
    first.add_argument("--seed", type=int, default=20261019)
    first.add_argument("--output-dir", type=Path, default=Path("runs/label_provenance"))
    second = commands.add_parser("compare")
    second.add_argument("--fingerprint", type=Path, default=Path("runs/label_provenance/label_fingerprint.json"))
    second.add_argument("--official", type=Path, nargs="+", required=True, help="Official label zip or JSON files")
    second.add_argument("--output-dir", type=Path, default=Path("runs/label_provenance"))
    args = parser.parse_args()
    if args.command == "fingerprint":
        provenance = json.loads(args.source_verified.read_text(encoding="utf-8"))
        paths = {row["original_split"]: Path(row["local_path"]) for row in provenance["files"] if row["archive_name"].endswith(".json")}
        for split, path in paths.items():
            expected = next(row["sha256"] for row in provenance["files"] if row["local_path"] == str(path))
            if sha256(path) != expected:
                raise ValueError(f"Label file changed since Kaggle verification: {split}")
        result = fingerprint(paths, args.per_split, args.seed, args.output_dir)
        result["kaggle"] = {key: provenance[key] for key in ("dataset_url", "version", "last_updated", "verification")}
        (args.output_dir / "label_fingerprint.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        compare(args.fingerprint, args.official, args.output_dir)


if __name__ == "__main__":
    main()
