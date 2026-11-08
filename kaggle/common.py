"""Shared Kaggle kernel bootstrap: locate inputs, stage code, rebuild the frozen dataset."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
CODE = Path("/tmp/code")
DATASET = Path("/tmp/bdd100k_revised")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def find_directory(marker: str, max_depth: int = 6) -> Path:
    """Find the directory containing `marker` without walking into image folders."""
    for root, directories, files in os.walk(INPUT):
        depth = Path(root).relative_to(INPUT).parts
        if marker in files or marker in directories:
            return Path(root)
        directories[:] = [d for d in directories if d not in ("train", "val", "test")] if len(depth) < max_depth else []
    raise FileNotFoundError(f"{marker} not found under {INPUT}")


def find_mirror_images() -> Path:
    """The mirror's images/100k directory (with train/ and val/), not a segmentation subset."""
    for root, directories, _ in os.walk(INPUT):
        path = Path(root)
        if path.name == "images" and (path / "100k" / "train").is_dir() and (path / "100k" / "val").is_dir():
            return path / "100k"
        directories[:] = [d for d in directories if d not in ("train", "val", "test")]
    raise FileNotFoundError("solesensei_bdd100k mirror images/100k not attached")


def run(arguments: list[str], log_name: str | None = None, check: bool = True) -> int:
    log("$ " + " ".join(arguments))
    with (open(WORK / "logs" / log_name, "w", encoding="utf-8") if log_name else open(os.devnull, "w")) as sink:
        process = subprocess.Popen(arguments, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            sink.write(line)
            if "index created" not in line and "creating index" not in line:
                print(line, end="", flush=True)
        code = process.wait()
    if check and code:
        raise RuntimeError(f"Command failed ({code}): {' '.join(arguments)}")
    return code


def bootstrap(rehash_sample: int = 0) -> dict:
    (WORK / "logs").mkdir(parents=True, exist_ok=True)
    protocol = find_directory("expected_sha256.json")
    mirror_images = find_mirror_images()
    log(f"protocol={protocol} mirror_images={mirror_images}")
    with open(WORK / "logs" / "mirror_tree.txt", "w", encoding="utf-8") as tree:
        for root, directories, files in os.walk(mirror_images.parents[2]):
            tree.write(f"{root}: {len(directories)} dirs, {len(files)} files\n")
    if CODE.exists():
        shutil.rmtree(CODE)
    shutil.copytree(protocol / "code", CODE)
    try:
        import pycocotools  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pycocotools"])
    if not (DATASET / "COMPLETE").exists():
        run([sys.executable, "scripts/kaggle_materialize.py", "--protocol", str(protocol),
             "--mirror-images", str(mirror_images), "--output", str(DATASET), "--workers", "8",
             "--rehash-sample", str(rehash_sample)], "materialize.log")
    info = {"protocol": str(protocol), "mirror_images": str(mirror_images), "dataset": str(DATASET)}
    (WORK / "bootstrap.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
