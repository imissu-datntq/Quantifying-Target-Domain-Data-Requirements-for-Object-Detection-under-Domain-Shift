"""Verify local BDD100K source bytes against the public Kaggle ZIP directory.

ZIP sizes and CRC32 establish content consistency, not a cryptographic signature.
Only the ZIP central directory is downloaded; image data are read locally.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

URL = "https://www.kaggle.com/api/v1/datasets/download/solesensei/solesensei_bdd100k"
DATASET_URL = "https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k"


class RangeReader(io.RawIOBase):
    """Seekable HTTP range reader, refusing servers that ignore Range."""

    def __init__(self, url: str):
        self.url = url
        self.position = 0
        self.cache: list[tuple[int, bytes]] = []
        request = urllib.request.Request(url, headers={"Range": "bytes=-65557"})
        with urllib.request.urlopen(request, timeout=120) as response:
            if response.status != 206:
                raise RuntimeError("Server must support HTTP range requests")
            start_end, total = response.headers["Content-Range"].split("/")
            start = int(start_end.split()[1].split("-")[0])
            self.size = int(total)
            self.cache.append((start, response.read()))

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        start = self.position
        for cached_start, data in self.cache:
            if cached_start <= start and start + size <= cached_start + len(data):
                self.position += size
                return data[start - cached_start:start - cached_start + size]
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + size - 1}"})
        with urllib.request.urlopen(request, timeout=180) as response:
            if response.status != 206:
                raise RuntimeError("HTTP range ignored")
            data = response.read()
        if len(data) != size:
            raise IOError("Incomplete HTTP range")
        self.cache.append((start, data))
        self.position += size
        return data


def checksums(path: Path) -> tuple[int, int, str]:
    crc = 0
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            size += len(chunk)
            crc = zlib.crc32(chunk, crc)
            sha.update(chunk)
    return size, crc & 0xFFFFFFFF, sha.hexdigest()


def verify(images: Path, labels: Path, output: Path, workers: int):
    output.mkdir(parents=True, exist_ok=True)
    print("Reading public Kaggle ZIP central directory", flush=True)
    directory_path = output / "kaggle_zip_directory.json"
    if directory_path.exists():
        entries = {name: SimpleNamespace(file_size=row["size"], CRC=row["crc32"]) for name, row in json.loads(directory_path.read_text(encoding="utf-8")).items()}
    else:
        with zipfile.ZipFile(RangeReader(URL)) as archive:
            entries = {entry.filename: entry for entry in archive.infolist()}
    selected = {}
    for name, entry in entries.items():
        if any(name.startswith(f"bdd100k/bdd100k/images/100k/{split}/") for split in ("train", "val")) and name.endswith(".jpg"):
            selected[name] = entry
        elif name.endswith(("bdd100k_labels_images_train.json", "bdd100k_labels_images_val.json")):
            selected[name] = entry
    if len(selected) != 80002:
        raise ValueError(f"Expected 80,000 images and two label files; found {len(selected)}")
    (output / "kaggle_zip_directory.json").write_text(json.dumps({name: {"size": e.file_size, "crc32": e.CRC} for name, e in selected.items()}, indent=2), encoding="utf-8")
    indexes = {}
    for split in ("train", "val"):
        paths = sorted((images / split).rglob("*.jpg"))
        if len(paths) != len({path.name for path in paths}):
            raise ValueError(f"Duplicate basename in raw {split}")
        indexes[split] = {path.name: path for path in paths}
    rows = []
    def verify_one(item):
        name, entry = item
        if name.endswith(".jpg"):
            split = name.split("/100k/")[1].split("/")[0]
            path = indexes[split].get(Path(name).name)
        else:
            split = "train" if name.endswith("_train.json") else "val"
            path = labels / Path(name).name
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Missing Kaggle source file: {name}")
        size, crc, sha = checksums(path)
        if (size, crc) != (entry.file_size, entry.CRC):
            raise ValueError(f"Kaggle content mismatch: {name}")
        return {"archive_name": name, "local_path": str(path.resolve()), "size": size, "crc32": crc, "sha256": sha, "original_split": split}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, row in enumerate(pool.map(verify_one, sorted(selected.items())), 1):
            rows.append(row)
            if index % 5000 == 0:
                print(f"Verified {index:,}/{len(selected):,} source files", flush=True)
    metadata = json.load(urllib.request.urlopen("https://www.kaggle.com/api/v1/datasets/view/solesensei/solesensei_bdd100k", timeout=60))
    result = {"dataset_url": DATASET_URL, "version": metadata["currentVersionNumber"], "last_updated": metadata["lastUpdated"], "verification": "Every source image and label file matches Kaggle ZIP size and CRC32; SHA256 computed locally", "files": rows}
    (output / "source_verified.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"PASS: {len(rows):,} source files verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    verify(args.images_root, args.labels_root, args.output_dir, args.workers)
