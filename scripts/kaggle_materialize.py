"""Rebuild data/bdd100k_revised on Kaggle from the frozen protocol files and the Kaggle mirror.

Annotations, manifests and metadata come byte-for-byte from the private protocol
dataset (checked against expected SHA256). Images are symlinked from the public
solesensei_bdd100k mirror and every linked image is rehashed against its manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from prepare_revised_bdd100k import EXPECTED, sha256, validate


def materialize(protocol: Path, mirror_images: Path, output: Path, workers: int, rehash_sample: int = 0) -> dict:
    expected = json.loads((protocol / "expected_sha256.json").read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to reuse a non-empty dataset root: {output}")
    for directory in ("annotations", "manifests"):
        shutil.copytree(protocol / "data" / directory, output / directory)
    for name in ("dataset_metadata.json", "COMPLETE"):
        shutil.copyfile(protocol / "data" / name, output / name)
    for relative, digest in expected.items():
        if sha256(output / relative) != digest:
            raise ValueError(f"Protocol file differs from local frozen copy: {relative}")
    index = {}
    for original in ("train", "val"):
        found = {}
        for root, _, files in os.walk(mirror_images / original):
            for name in files:
                if name.endswith(".jpg"):
                    if name in found:
                        raise ValueError(f"Duplicate mirror basename: {original}/{name}")
                    found[name] = Path(root) / name
        index[original] = found
        print(f"Mirror {original}: {len(found):,} images", flush=True)
    jobs, missing = [], []
    for split in EXPECTED:
        (output / "images" / split).mkdir(parents=True)
        with (output / "manifests" / f"{split}.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                source = index["val" if split == "val" else "train"].get(row["name"])
                if source is None:
                    missing.append(f"{split}/{row['name']}")
                    continue
                target = output / "images" / split / row["name"]
                os.symlink(source, target)
                jobs.append((target, row["sha256"]))
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest images absent from the mirror, e.g. {missing[:3]}")

    def check(job):
        target, digest = job
        return sha256(target) == digest

    # Full rehash by default; a GPU session may rehash a seeded sample after a full CPU check passed.
    checked = jobs if not rehash_sample else random.Random(20261020).sample(jobs, min(rehash_sample, len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        mismatched = sum(not ok for ok in pool.map(check, checked))
    if mismatched:
        raise ValueError(f"{mismatched} mirror images differ from the frozen manifests")
    integrity = validate(output)
    return {"images_linked": len(jobs), "images_rehashed": len(checked), "protocol_files_checked": len(expected), "integrity": integrity}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True, help="Directory with data/ and expected_sha256.json")
    parser.add_argument("--mirror-images", type=Path, required=True, help="Mirror images/100k directory (train/, val/)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rehash-sample", type=int, default=0, help="0 rehashes every image")
    args = parser.parse_args()
    print(json.dumps(materialize(args.protocol, args.mirror_images, args.output, args.workers, args.rehash_sample), indent=2))


if __name__ == "__main__":
    main()
