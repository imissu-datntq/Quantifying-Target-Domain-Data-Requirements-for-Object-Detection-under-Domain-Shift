"""BDD100K comparison driver for SSDA-YOLO, MS-DAYOLO, and SF-YOLO.

This file does not reimplement the papers under convenient look-alike names.
The three methods use incompatible official codebases, so this driver:

1. converts the already-split BDD100K JSON annotations to one shared YOLO set;
2. hides target-train labels from all three adaptation methods;
3. invokes each official repository with an adapter matching its real CLI;
4. evaluates on the same target test set and builds a common mAP50 report.

Official implementations expected by this driver:
  SSDA-YOLO: https://github.com/hnuzhy/SSDA-YOLO (YOLOv5 v5.0)
  MS-DAYOLO: https://github.com/Mazin-Hnewa/MS-DAYOLO (Darknet/YOLOv4)
  SF-YOLO:   https://github.com/vs-cv/sf-yolo (YOLOv5)

The methods have genuinely different prerequisites:
  * SSDA-YOLO needs source->target and target->source translated image sets.
  * MS-DAYOLO needs a compiled Darknet binary and yolov4.conv.137.
  * SF-YOLO needs a source checkpoint and trained TargetAugment assets.

Typical workflow (examples only; nothing runs when this file is created):

  python train_da_yolo_comparison.py bootstrap
  python train_da_yolo_comparison.py prepare --domain lighting

  python train_da_yolo_comparison.py train --model ssda-yolo --domain lighting \
      --initial-weights weights/yolov5l.pt \
      --source-fake-images /path/to/daytime_to_night/images \
      --target-fake-images /path/to/night_to_daytime/images

  python train_da_yolo_comparison.py train --model ms-dayolo --domain lighting \
      --initial-weights weights/yolov4.conv.137

  python train_da_yolo_comparison.py train-source --domain lighting \
      --initial-weights weights/yolov5l.pt
  python train_da_yolo_comparison.py train --model sf-yolo --domain lighting \
      --source-weights runs/lighting/sf-source/weights/best.pt \
      --decoder /path/to/decoder.pth --encoder /path/to/vgg16.pth \
      --fc1 /path/to/fc1.pth --fc2 /path/to/fc2.pth \
      --style-image /path/to/target_style.jpg

  python train_da_yolo_comparison.py evaluate --model ssda-yolo \
      --domain lighting --weights /path/to/best_student.pt
  python train_da_yolo_comparison.py evaluate --model ms-dayolo \
      --domain lighting --weights /path/to/ms-dayolo_best.weights
  python train_da_yolo_comparison.py evaluate --model sf-yolo \
      --domain lighting --weights /path/to/best_teacher.pt
  python train_da_yolo_comparison.py compare --domain lighting

Only Python's standard library is required by this driver. Each official model
repository must be installed in its own compatible environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGES_ROOT = ROOT / "data" / "bdd100k" / "bdd100k" / "images" / "100k"
DEFAULT_LABELS_ROOT = (
    ROOT / "data" / "bdd100k_labels_release" / "bdd100k" / "labels"
)
DEFAULT_PREPARED_ROOT = ROOT / "prepared_da_yolo"
DEFAULT_REPOS_ROOT = ROOT / "external"
DEFAULT_RUNS_ROOT = ROOT / "runs" / "da_yolo_comparison"

OFFICIAL_REPOSITORIES = {
    "ssda-yolo": "https://github.com/hnuzhy/SSDA-YOLO.git",
    "ms-dayolo": "https://github.com/Mazin-Hnewa/MS-DAYOLO.git",
    "sf-yolo": "https://github.com/vs-cv/sf-yolo.git",
}
REPOSITORY_DIRECTORIES = {
    "ssda-yolo": "SSDA-YOLO",
    "ms-dayolo": "MS-DAYOLO",
    "sf-yolo": "sf-yolo",
}

# One identical class order is used by every model.
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
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class Domain:
    timeofday: str
    weather: str


DOMAINS = {
    "source": Domain("daytime", "clear"),
    "lighting": Domain("night", "clear"),
    "weather": Domain("daytime", "rainy"),
    "compound": Domain("night", "rainy"),
}


@dataclass(frozen=True)
class Sample:
    name: str
    boxes: tuple[tuple[int, float, float, float, float], ...]


def json_array_items(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    """Read a large top-level JSON array without retaining it in memory."""

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
                        raise ValueError(f"Empty JSON file: {path}")
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"Expected a JSON array: {path}")
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
                    raise ValueError(f"Unexpected end of JSON array: {path}")
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
                raise ValueError(f"Expected an object in {path}")
            yield item
            position = end


def annotation_path(labels_root: Path, split: str) -> Path:
    return labels_root / f"bdd100k_labels_images_{split}.json"


def read_domain_samples(labels_root: Path, split: str, domain: Domain) -> list[Sample]:
    path = annotation_path(labels_root, split)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    samples: list[Sample] = []
    for record in json_array_items(path):
        attributes = record.get("attributes") or {}
        if attributes.get("timeofday") != domain.timeofday:
            continue
        if attributes.get("weather") != domain.weather:
            continue

        boxes: list[tuple[int, float, float, float, float]] = []
        for label in record.get("labels") or []:
            category = label.get("category")
            box = label.get("box2d")
            if category not in CLASS_TO_ID or not isinstance(box, dict):
                continue
            try:
                boxes.append(
                    (
                        CLASS_TO_ID[category],
                        float(box["x1"]),
                        float(box["y1"]),
                        float(box["x2"]),
                        float(box["y2"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        samples.append(Sample(name=str(record["name"]), boxes=tuple(boxes)))

    samples.sort(key=lambda item: item.name)
    if not samples:
        raise ValueError(f"No samples matched {domain} in split '{split}'")
    return samples


def nested_subset(samples: list[Sample], budget: str, seed: int) -> list[Sample]:
    if budget == "full":
        return samples
    try:
        amount = int(budget)
    except ValueError as error:
        raise ValueError("Target budget must be 'full' or a positive integer") from error
    if amount <= 0 or amount > len(samples):
        raise ValueError(f"Invalid target budget {amount}; available images: {len(samples)}")
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:amount]


def create_image_link(source: Path, destination: Path, mode: str) -> None:
    if destination.exists() or destination.is_symlink():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)

    if mode in {"hardlink", "auto"}:
        try:
            os.link(source, destination)
            return
        except OSError:
            if mode == "hardlink":
                raise
    if mode in {"symlink", "auto"}:
        try:
            destination.symlink_to(source)
            return
        except OSError:
            if mode == "symlink":
                raise
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    raise OSError(
        f"Could not link {source}. Use --link-mode copy explicitly if copying is acceptable."
    )


def yolo_label_text(sample: Sample, width: int, height: int) -> str:
    lines: list[str] = []
    for class_id, raw_x1, raw_y1, raw_x2, raw_y2 in sample.boxes:
        x1 = min(max(raw_x1, 0.0), float(width))
        y1 = min(max(raw_y1, 0.0), float(height))
        x2 = min(max(raw_x2, 0.0), float(width))
        y2 = min(max(raw_y2, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        center_x = ((x1 + x2) / 2.0) / width
        center_y = ((y1 + y2) / 2.0) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        lines.append(
            f"{class_id} {center_x:.8f} {center_y:.8f} "
            f"{box_width:.8f} {box_height:.8f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def stage_samples(
    samples: Sequence[Sample],
    *,
    source_image_dir: Path,
    prepared_domain_root: Path,
    group: str,
    split: str,
    label_policy: str,
    width: int,
    height: int,
    link_mode: str,
) -> tuple[Path, dict[str, Any]]:
    image_dir = prepared_domain_root / group / "images" / split
    label_dir = prepared_domain_root / group / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    class_counts: Counter[str] = Counter()
    metadata_targets = 0
    for sample_index, sample in enumerate(samples):
        original_image = source_image_dir / sample.name
        if not original_image.is_file():
            raise FileNotFoundError(f"Image referenced by annotation is missing: {original_image}")
        staged_image = image_dir / sample.name
        create_image_link(original_image.resolve(), staged_image, link_mode)
        image_paths.append(staged_image.resolve())

        label_path = label_dir / f"{Path(sample.name).stem}.txt"
        if label_policy == "ground_truth":
            label_path.write_text(yolo_label_text(sample, width, height), encoding="utf-8")
            for class_id, *_ in sample.boxes:
                class_counts[CLASS_NAMES[class_id]] += 1
        elif label_policy == "unlabeled":
            # MS-DAYOLO explicitly expects dummy target annotations. Empty files
            # also ensure SSDA/SF cannot accidentally consume target GT labels.
            label_path.write_text("", encoding="utf-8")
        elif label_policy == "sf_metadata_only":
            # The official SF-YOLO script computes label metadata before replacing
            # training targets with teacher pseudo-labels. One synthetic box per
            # class keeps that initialization functional without exposing target GT.
            if sample_index < len(CLASS_NAMES):
                label_path.write_text(
                    f"{sample_index} 0.50000000 0.50000000 0.05000000 0.05000000\n",
                    encoding="utf-8",
                )
                metadata_targets += 1
            else:
                label_path.write_text("", encoding="utf-8")
        else:
            raise ValueError(f"Unknown label policy: {label_policy}")

    manifest_dir = prepared_domain_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / f"{group}_{split}.txt"
    manifest.write_text(
        "\n".join(path.as_posix() for path in image_paths) + "\n", encoding="utf-8"
    )
    return manifest, {
        "images": len(samples),
        "label_policy": label_policy,
        "synthetic_metadata_targets": metadata_targets,
        "instances": sum(class_counts.values()),
        "instances_per_class": {name: class_counts[name] for name in CLASS_NAMES},
    }


def yaml_quote(value: str | Path) -> str:
    return json.dumps(Path(value).as_posix() if isinstance(value, Path) else value)


def write_standard_yaml(path: Path, train: Path, val: Path, test: Path) -> None:
    names = json.dumps(list(CLASS_NAMES), ensure_ascii=False)
    path.write_text(
        "\n".join(
            [
                f"train: {yaml_quote(train.resolve())}",
                f"val: {yaml_quote(val.resolve())}",
                f"test: {yaml_quote(test.resolve())}",
                f"nc: {len(CLASS_NAMES)}",
                f"names: {names}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_ms_data(
    path: Path,
    source_train: Path,
    target_train: Path,
    target_eval: Path,
    names_path: Path,
    backup_dir: Path,
) -> None:
    path.write_text(
        "\n".join(
            [
                f"classes = {len(CLASS_NAMES)}",
                f"train = {source_train.resolve().as_posix()}",
                f"train_target = {target_train.resolve().as_posix()}",
                f"valid = {target_eval.resolve().as_posix()}",
                f"names = {names_path.resolve().as_posix()}",
                f"backup = {backup_dir.resolve().as_posix()}/",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_ms_cfg(
    source_cfg: Path,
    destination_cfg: Path,
    *,
    classes: int,
    image_size: int,
    batch_size: int,
    subdivisions: int,
    max_batches: int,
) -> None:
    """Adapt the official YOLOv4 DA cfg without changing its DA architecture."""

    if image_size % 32 != 0:
        raise ValueError("MS-DAYOLO image size must be divisible by 32")
    if subdivisions <= 0 or batch_size % subdivisions != 0:
        raise ValueError("MS-DAYOLO batch size must be divisible by subdivisions")

    lines = source_cfg.read_text(encoding="utf-8").splitlines()
    section_starts = [
        index
        for index, line in enumerate(lines)
        if line.strip().startswith("[") and line.strip().endswith("]")
    ]

    def section_end(start: int) -> int:
        later = [index for index in section_starts if index > start]
        return min(later) if later else len(lines)

    def replace_key(start: int, end: int, key: str, value: int | str) -> None:
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
        for index in range(start + 1, end):
            if pattern.match(lines[index]) and not lines[index].lstrip().startswith("#"):
                lines[index] = f"{key}={value}"
                return
        raise ValueError(f"Key '{key}' not found in section starting at line {start + 1}")

    net_starts = [index for index in section_starts if lines[index].strip() == "[net]"]
    if len(net_starts) != 1:
        raise ValueError(f"Expected one [net] section in {source_cfg}")
    net_start = net_starts[0]
    net_end = section_end(net_start)
    replace_key(net_start, net_end, "batch", batch_size)
    replace_key(net_start, net_end, "subdivisions", subdivisions)
    replace_key(net_start, net_end, "width", image_size)
    replace_key(net_start, net_end, "height", image_size)
    replace_key(net_start, net_end, "max_batches", max_batches)
    replace_key(net_start, net_end, "steps", f"{int(max_batches * 0.8)},{int(max_batches * 0.9)}")

    yolo_starts = [index for index in section_starts if lines[index].strip() == "[yolo]"]
    if not yolo_starts:
        raise ValueError(f"No [yolo] heads found in {source_cfg}")
    for yolo_start in yolo_starts:
        yolo_end = section_end(yolo_start)
        mask_count = 3
        for index in range(yolo_start + 1, yolo_end):
            if re.match(r"^\s*mask\s*=", lines[index]):
                mask_count = len(lines[index].split("=", 1)[1].split(","))
                break
        replace_key(yolo_start, yolo_end, "classes", classes)

        previous_sections = [index for index in section_starts if index < yolo_start]
        convolution_start = max(
            index for index in previous_sections if lines[index].strip() == "[convolutional]"
        )
        replace_key(
            convolution_start,
            section_end(convolution_start),
            "filters",
            (classes + 5) * mask_count,
        )

    destination_cfg.parent.mkdir(parents=True, exist_ok=True)
    destination_cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepared_domain_root(args: argparse.Namespace) -> Path:
    return args.prepared_root / args.domain


def protocol_path(args: argparse.Namespace) -> Path:
    return prepared_domain_root(args) / "protocol.json"


def load_protocol(args: argparse.Namespace) -> dict[str, Any]:
    path = protocol_path(args)
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared protocol not found: {path}. Run the 'prepare' command first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(args: argparse.Namespace) -> None:
    destination = prepared_domain_root(args)
    destination.mkdir(parents=True, exist_ok=True)

    source_train = read_domain_samples(args.labels_root, "train", DOMAINS["source"])
    source_val = read_domain_samples(args.labels_root, "val", DOMAINS["source"])
    target_train_all = read_domain_samples(args.labels_root, "train", DOMAINS[args.domain])
    target_train = nested_subset(target_train_all, args.target_budget, args.sampling_seed)
    target_val = read_domain_samples(args.labels_root, "val", DOMAINS[args.domain])
    target_test = read_domain_samples(args.labels_root, "test", DOMAINS[args.domain])

    staged: dict[str, tuple[Path, dict[str, Any]]] = {}
    definitions = (
        ("source_train", source_train, "train", "source", "ground_truth"),
        ("source_val", source_val, "val", "source", "ground_truth"),
        ("target_train", target_train, "train", "target", "unlabeled"),
        ("target_sf_train", target_train, "train", "target_sf", "sf_metadata_only"),
        ("target_val", target_val, "val", "target", "ground_truth"),
        ("target_test", target_test, "test", "target", "ground_truth"),
    )
    for key, samples, original_split, group, policy in definitions:
        manifest, summary = stage_samples(
            samples,
            source_image_dir=args.images_root / original_split,
            prepared_domain_root=destination,
            group=group,
            split=key.split("_", 1)[1],
            label_policy=policy,
            width=args.image_width,
            height=args.image_height,
            link_mode=args.link_mode,
        )
        staged[key] = manifest, summary

    config_dir = destination / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    write_standard_yaml(
        config_dir / "source_yolo.yaml",
        staged["source_train"][0],
        staged["source_val"][0],
        staged["source_val"][0],
    )
    write_standard_yaml(
        config_dir / "target_yolo.yaml",
        staged["target_sf_train"][0],
        staged["target_val"][0],
        staged["target_test"][0],
    )

    names_path = config_dir / "bdd100k.names"
    names_path.write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    ms_backup = args.runs_root / args.domain / "ms-dayolo" / "weights"
    ms_backup.mkdir(parents=True, exist_ok=True)
    write_ms_data(
        config_dir / "ms_dayolo_train.data",
        staged["source_train"][0],
        staged["target_train"][0],
        staged["target_val"][0],
        names_path,
        ms_backup,
    )
    write_ms_data(
        config_dir / "ms_dayolo_test.data",
        staged["source_train"][0],
        staged["target_train"][0],
        staged["target_test"][0],
        names_path,
        ms_backup,
    )

    protocol = {
        "dataset": "BDD100K dataset-aligned split",
        "domain": args.domain,
        "source_filter": DOMAINS["source"].__dict__,
        "target_filter": DOMAINS[args.domain].__dict__,
        "target_budget": args.target_budget,
        "sampling_seed": args.sampling_seed,
        "class_names": list(CLASS_NAMES),
        "image_size": [args.image_width, args.image_height],
        "target_train_ground_truth_exposed": False,
        "sf_yolo_metadata_note": (
            "The official SF-YOLO loader requires non-empty label metadata. "
            "Its training manifest contains one synthetic box per class; real target-train "
            "boxes remain hidden and --noautoanchor is enforced."
        ),
        "selection_split": "target val",
        "final_split": "target test",
        "sets": {key: summary for key, (_, summary) in staged.items()},
        "manifests": {key: str(path.resolve()) for key, (path, _) in staged.items()},
    }
    protocol_path(args).write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


def bootstrap(args: argparse.Namespace) -> None:
    args.repos_root.mkdir(parents=True, exist_ok=True)
    for model, url in OFFICIAL_REPOSITORIES.items():
        destination = args.repos_root / REPOSITORY_DIRECTORIES[model]
        if destination.exists():
            print(f"exists: {destination}")
            continue
        run_command(["git", "clone", url, str(destination)], cwd=args.repos_root)


def repo_path(args: argparse.Namespace, model: str) -> Path:
    explicit = getattr(args, model.replace("-", "_") + "_repo", None)
    path = explicit or args.repos_root / REPOSITORY_DIRECTORIES[model]
    if not path.is_dir():
        raise FileNotFoundError(
            f"Official {model} repository not found: {path}. Run 'bootstrap' or pass its repo path."
        )
    return path.resolve()


def require_file(path: Path | None, description: str) -> Path:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path.resolve()


def require_directory(path: Path | None, description: str) -> Path:
    if path is None or not path.is_dir():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path.resolve()


def write_ssda_yaml(
    path: Path,
    protocol: dict[str, Any],
    source_fake_images: Path,
    target_fake_images: Path,
    evaluation_split: str,
) -> None:
    eval_key = "target_test" if evaluation_split == "test" else "target_val"
    names = json.dumps(list(CLASS_NAMES), ensure_ascii=False)
    path.write_text(
        "\n".join(
            [
                f"train_source_real: {yaml_quote(protocol['manifests']['source_train'])}",
                f"train_source_fake: {yaml_quote(source_fake_images)}",
                f"train_target_real: {yaml_quote(protocol['manifests']['target_train'])}",
                f"train_target_fake: {yaml_quote(target_fake_images)}",
                f"test_target_real: {yaml_quote(protocol['manifests'][eval_key])}",
                f"nc: {len(CLASS_NAMES)}",
                f"names: {names}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def command_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(item) for item in command])


def run_command(
    command: Sequence[str | Path],
    *,
    cwd: Path,
    log_path: Path | None = None,
    dry_run: bool = False,
) -> float:
    normalized = [str(item) for item in command]
    print(f"cwd: {cwd}")
    print(f"command: {command_text(normalized)}")
    if dry_run:
        return 0.0

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    process = subprocess.Popen(
        normalized,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    log_handle = log_path.open("w", encoding="utf-8") if log_path else None
    try:
        for line in process.stdout:
            print(line, end="")
            if log_handle:
                log_handle.write(line)
    finally:
        if log_handle:
            log_handle.close()
    return_code = process.wait()
    elapsed = time.perf_counter() - started
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, normalized)
    return elapsed


def model_run_dir(args: argparse.Namespace, model: str) -> Path:
    path = args.runs_root / args.domain / model
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def train_ssda(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    repo = repo_path(args, "ssda-yolo")
    script = require_file(repo / "ssda_yolov5_train.py", "SSDA-YOLO train script")
    initial_weights = require_file(args.initial_weights, "SSDA YOLOv5 v5.0 weights")
    source_fake = require_directory(args.source_fake_images, "source-to-target images")
    target_fake = require_directory(args.target_fake_images, "target-to-source images")
    output = model_run_dir(args, "ssda-yolo")
    data_yaml = prepared_domain_root(args) / "configs" / "ssda_yolo_train.yaml"
    write_ssda_yaml(data_yaml, protocol, source_fake, target_fake, "val")

    command = [
        args.ssda_python,
        script,
        "--weights",
        initial_weights,
        "--data",
        data_yaml,
        "--name",
        "adapt",
        "--project",
        output,
        "--exist-ok",
        "--img",
        str(args.img_size),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lambda_weight",
        str(args.ssda_lambda),
        "--consistency_loss",
        "--alpha_weight",
        str(args.ssda_alpha),
        "--noautoanchor",
    ]
    elapsed = run_command(
        command,
        cwd=repo,
        log_path=output / "train.log",
        dry_run=args.dry_run,
    )
    write_run_metadata(output, args, command, elapsed)


def train_ms(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    repo = repo_path(args, "ms-dayolo")
    darknet = args.ms_darknet or repo / ("darknet.exe" if os.name == "nt" else "darknet")
    darknet = require_file(darknet, "compiled MS-DAYOLO Darknet binary")
    base_cfg = require_file(args.ms_cfg or repo / "cfg" / "ms-dayolo.cfg", "MS-DAYOLO cfg")
    cfg = prepared_domain_root(args) / "configs" / "ms_dayolo_bdd100k.cfg"
    source_images = int(protocol["sets"]["source_train"]["images"])
    max_batches = max(1, args.epochs * math.ceil(source_images / args.batch_size))
    write_ms_cfg(
        base_cfg,
        cfg,
        classes=len(CLASS_NAMES),
        image_size=args.img_size,
        batch_size=args.batch_size,
        subdivisions=args.ms_subdivisions,
        max_batches=max_batches,
    )
    initial_weights = require_file(args.initial_weights, "yolov4.conv.137 weights")
    data_file = require_file(
        prepared_domain_root(args) / "configs" / "ms_dayolo_train.data",
        "MS-DAYOLO training data file",
    )
    output = model_run_dir(args, "ms-dayolo")
    command = [
        darknet,
        "detector",
        "train",
        data_file,
        cfg,
        initial_weights,
        "-dont_show",
        "-map",
        "-da",
    ]
    elapsed = run_command(
        command,
        cwd=repo,
        log_path=output / "train.log",
        dry_run=args.dry_run,
    )
    write_run_metadata(output, args, command, elapsed)


def train_sf(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    del protocol
    repo = repo_path(args, "sf-yolo")
    script = require_file(repo / "train_sf-yolo.py", "SF-YOLO train script")
    source_weights = require_file(args.source_weights, "SF-YOLO source checkpoint")
    decoder = require_file(args.decoder, "TargetAugment decoder")
    encoder = require_file(args.encoder, "TargetAugment VGG encoder")
    fc1 = require_file(args.fc1, "TargetAugment fc1")
    fc2 = require_file(args.fc2, "TargetAugment fc2")
    style_image = require_file(args.style_image, "target style image")
    data_yaml = require_file(
        prepared_domain_root(args) / "configs" / "target_yolo.yaml",
        "SF-YOLO target YAML",
    )
    output = model_run_dir(args, "sf-yolo")
    command = [
        args.sf_python,
        script,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--data",
        data_yaml,
        "--weights",
        source_weights,
        "--decoder_path",
        decoder,
        "--encoder_path",
        encoder,
        "--fc1",
        fc1,
        "--fc2",
        fc2,
        "--style_add_alpha",
        str(args.sf_style_alpha),
        "--style_path",
        style_image,
        "--SSM_alpha",
        str(args.sf_ssm_alpha),
        "--imgsz",
        str(args.img_size),
        "--device",
        args.device,
        "--project",
        output,
        "--name",
        "adapt",
        "--exist-ok",
        "--noautoanchor",
    ]
    elapsed = run_command(
        command,
        cwd=repo,
        log_path=output / "train.log",
        dry_run=args.dry_run,
    )
    write_run_metadata(output, args, command, elapsed)


def write_run_metadata(
    output: Path,
    args: argparse.Namespace,
    command: Sequence[str | Path],
    elapsed: float,
) -> None:
    serialized_arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "handler"
    }
    record = {
        "command": [str(item) for item in command],
        "elapsed_seconds": elapsed,
        "arguments": serialized_arguments,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def train(args: argparse.Namespace) -> None:
    protocol = load_protocol(args)
    if args.model == "ssda-yolo":
        train_ssda(args, protocol)
    elif args.model == "ms-dayolo":
        train_ms(args, protocol)
    elif args.model == "sf-yolo":
        train_sf(args, protocol)
    else:
        raise ValueError(args.model)


def train_source(args: argparse.Namespace) -> None:
    load_protocol(args)
    repo = repo_path(args, "sf-yolo")
    script = require_file(repo / "train_source.py", "SF-YOLO source training script")
    initial_weights = require_file(args.initial_weights, "YOLOv5 source initialization")
    data_yaml = require_file(
        prepared_domain_root(args) / "configs" / "source_yolo.yaml",
        "source-domain YAML",
    )
    output = model_run_dir(args, "sf-source")
    command = [
        args.sf_python,
        script,
        "--weights",
        initial_weights,
        "--data",
        data_yaml,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--imgsz",
        str(args.img_size),
        "--device",
        args.device,
        "--project",
        output,
        "--name",
        "source",
        "--exist-ok",
    ]
    elapsed = run_command(
        command,
        cwd=repo,
        log_path=output / "train.log",
        dry_run=args.dry_run,
    )
    write_run_metadata(output, args, command, elapsed)


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
MS_MAP_PATTERN = re.compile(
    r"mean average precision.*?=\s*([0-9]*\.?[0-9]+)(?:\s*,\s*or\s*([0-9]*\.?[0-9]+)\s*%)?",
    re.IGNORECASE,
)


def parse_metrics(log_text: str, model: str) -> dict[str, float | None]:
    clean = ANSI_ESCAPE.sub("", log_text)
    map50: float | None = None
    map50_95: float | None = None

    if model == "ms-dayolo":
        for match in MS_MAP_PATTERN.finditer(clean):
            raw = float(match.group(1))
            percent = float(match.group(2)) if match.group(2) else None
            map50 = raw if raw <= 1.0 else raw / 100.0
            if percent is not None:
                map50 = percent / 100.0
    else:
        for line in clean.splitlines():
            tokens = line.strip().split()
            if not tokens or tokens[0] != "all":
                continue
            numeric: list[float] = []
            for token in tokens[1:]:
                try:
                    numeric.append(float(token))
                except ValueError:
                    pass
            # SSDA's fork adds mAP75 between mAP50 and mAP50-95.
            if model == "ssda-yolo" and len(numeric) >= 7:
                map50 = numeric[-3]
                map50_95 = numeric[-1]
            # Standard YOLOv5 row: images, instances, P, R, mAP50, mAP50-95.
            elif len(numeric) >= 6:
                map50 = numeric[-2]
                map50_95 = numeric[-1]

    return {"map50": map50, "map50_95": map50_95}


def evaluate_model(args: argparse.Namespace) -> None:
    protocol = load_protocol(args)
    weights = require_file(args.weights, "evaluation checkpoint")
    output = model_run_dir(args, args.model)
    log_path = output / "evaluate.log"

    if args.model == "ssda-yolo":
        repo = repo_path(args, "ssda-yolo")
        source_fake = require_directory(args.source_fake_images, "source-to-target images")
        target_fake = require_directory(args.target_fake_images, "target-to-source images")
        data_yaml = prepared_domain_root(args) / "configs" / "ssda_yolo_test.yaml"
        write_ssda_yaml(data_yaml, protocol, source_fake, target_fake, "test")
        command: list[str | Path] = [
            args.ssda_python,
            require_file(repo / "ssda_yolov5_test.py", "SSDA-YOLO test script"),
            "--data",
            data_yaml,
            "--weights",
            weights,
            "--name",
            "final_test",
            "--img",
            str(args.img_size),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
        ]
    elif args.model == "ms-dayolo":
        repo = repo_path(args, "ms-dayolo")
        darknet = args.ms_darknet or repo / ("darknet.exe" if os.name == "nt" else "darknet")
        command = [
            require_file(darknet, "compiled MS-DAYOLO Darknet binary"),
            "detector",
            "map",
            require_file(
                prepared_domain_root(args) / "configs" / "ms_dayolo_test.data",
                "MS-DAYOLO test data file",
            ),
            require_file(
                args.ms_cfg
                or prepared_domain_root(args) / "configs" / "ms_dayolo_bdd100k.cfg",
                "BDD100K-adapted MS-DAYOLO cfg (run train or train --dry-run first)",
            ),
            weights,
        ]
    elif args.model == "sf-yolo":
        repo = repo_path(args, "sf-yolo")
        command = [
            args.sf_python,
            require_file(repo / "val.py", "SF-YOLO validation script"),
            "--data",
            require_file(
                prepared_domain_root(args) / "configs" / "target_yolo.yaml",
                "target YAML",
            ),
            "--weights",
            weights,
            "--task",
            "test",
            "--imgsz",
            str(args.img_size),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--project",
            output,
            "--name",
            "final_test",
            "--exist-ok",
        ]
    else:
        raise ValueError(args.model)

    elapsed = run_command(
        command,
        cwd=repo,
        log_path=log_path,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return
    metrics = parse_metrics(log_path.read_text(encoding="utf-8"), args.model)
    result = {
        "model": args.model,
        "domain": args.domain,
        "evaluation_split": "test",
        "test_images": protocol["sets"]["target_test"]["images"],
        **metrics,
        "evaluation_seconds": elapsed,
        "checkpoint": str(weights),
        "checkpoint_bytes": weights.stat().st_size,
        "metric_warning": (
            "mAP50 is the only metric emitted by the official MS-DAYOLO/Darknet evaluator"
            if args.model == "ms-dayolo"
            else None
        ),
    }
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def compare(args: argparse.Namespace) -> None:
    load_protocol(args)
    records: list[dict[str, Any]] = []
    for model in ("ssda-yolo", "ms-dayolo", "sf-yolo"):
        path = args.runs_root / args.domain / model / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation result for {model}: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("map50") is None:
            raise ValueError(
                f"Could not parse mAP50 for {model}; inspect its evaluate.log and record the result."
            )
        records.append(record)

    records.sort(key=lambda item: float(item["map50"]), reverse=True)
    for rank, record in enumerate(records, start=1):
        record["rank_by_map50"] = rank

    output_dir = args.runs_root / args.domain
    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "rank_by_map50",
                "model",
                "map50",
                "map50_95",
                "evaluation_seconds",
                "checkpoint_bytes",
                "checkpoint",
            ),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)

    report = {
        "domain": args.domain,
        "primary_common_metric": "mAP50 on the same held-out target test split",
        "recommended_by_accuracy": records[0]["model"],
        "ranking": records,
        "selection_note": (
            "Use this recommendation only when detection accuracy is the primary objective. "
            "The methods use different backbones and environments; deployment latency and "
            "memory should be benchmarked on the intended hardware before the final choice."
        ),
    }
    json_path = output_dir / "comparison.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--domain", choices=("lighting", "weather", "compound"), required=True)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--repos-root", type=Path, default=DEFAULT_REPOS_ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--ssda-yolo-repo", type=Path)
    parser.add_argument("--ms-dayolo-repo", type=Path)
    parser.add_argument("--sf-yolo-repo", type=Path)


def add_execution_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--img-size", type=int, default=960)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing it.")
    parser.add_argument("--ssda-python", default=sys.executable)
    parser.add_argument("--sf-python", default=sys.executable)
    parser.add_argument("--ms-darknet", type=Path)
    parser.add_argument("--ms-cfg", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare, train, evaluate, and compare three official DA-YOLO methods."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    bootstrap_parser = subparsers.add_parser("bootstrap", help="Clone the three official repositories.")
    bootstrap_parser.add_argument("--repos-root", type=Path, default=DEFAULT_REPOS_ROOT)
    bootstrap_parser.set_defaults(handler=bootstrap)

    prepare_parser = subparsers.add_parser("prepare", help="Create one leakage-safe YOLO dataset.")
    add_paths(prepare_parser)
    prepare_parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    prepare_parser.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS_ROOT)
    prepare_parser.add_argument("--target-budget", default="full")
    prepare_parser.add_argument("--sampling-seed", type=int, default=20260916)
    prepare_parser.add_argument("--image-width", type=int, default=1280)
    prepare_parser.add_argument("--image-height", type=int, default=720)
    prepare_parser.add_argument(
        "--link-mode", choices=("auto", "hardlink", "symlink", "copy"), default="auto"
    )
    prepare_parser.set_defaults(handler=prepare)

    train_parser = subparsers.add_parser("train", help="Train one official adaptation method.")
    add_paths(train_parser)
    add_execution_options(train_parser)
    train_parser.add_argument("--model", choices=tuple(OFFICIAL_REPOSITORIES), required=True)
    train_parser.add_argument("--epochs", type=int, default=60)
    train_parser.add_argument("--initial-weights", type=Path)
    train_parser.add_argument("--source-weights", type=Path)
    train_parser.add_argument("--source-fake-images", type=Path)
    train_parser.add_argument("--target-fake-images", type=Path)
    train_parser.add_argument("--ssda-lambda", type=float, default=0.005)
    train_parser.add_argument("--ssda-alpha", type=float, default=2.0)
    train_parser.add_argument("--ms-subdivisions", type=int, default=16)
    train_parser.add_argument("--decoder", type=Path)
    train_parser.add_argument("--encoder", type=Path)
    train_parser.add_argument("--fc1", type=Path)
    train_parser.add_argument("--fc2", type=Path)
    train_parser.add_argument("--style-image", type=Path)
    train_parser.add_argument("--sf-style-alpha", type=float, default=0.4)
    train_parser.add_argument("--sf-ssm-alpha", type=float, default=0.5)
    train_parser.set_defaults(handler=train)

    source_parser = subparsers.add_parser(
        "train-source", help="Train the YOLOv5 source checkpoint required by SF-YOLO."
    )
    add_paths(source_parser)
    add_execution_options(source_parser)
    source_parser.add_argument("--epochs", type=int, default=100)
    source_parser.add_argument("--initial-weights", type=Path, required=True)
    source_parser.set_defaults(handler=train_source)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate one checkpoint on the common target test split."
    )
    add_paths(evaluate_parser)
    add_execution_options(evaluate_parser)
    evaluate_parser.add_argument("--model", choices=tuple(OFFICIAL_REPOSITORIES), required=True)
    evaluate_parser.add_argument("--weights", type=Path, required=True)
    evaluate_parser.add_argument("--source-fake-images", type=Path)
    evaluate_parser.add_argument("--target-fake-images", type=Path)
    evaluate_parser.set_defaults(handler=evaluate_model)

    compare_parser = subparsers.add_parser(
        "compare", help="Rank completed runs by common held-out target mAP50."
    )
    add_paths(compare_parser)
    compare_parser.set_defaults(handler=compare)
    return parser


def validate_common_args(args: argparse.Namespace) -> None:
    if hasattr(args, "epochs") and args.epochs <= 0:
        raise ValueError("Epochs must be positive")
    if hasattr(args, "batch_size") and args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    if hasattr(args, "img_size") and args.img_size <= 0:
        raise ValueError("Image size must be positive")


def main() -> None:
    args = build_parser().parse_args()
    validate_common_args(args)
    args.handler(args)


if __name__ == "__main__":
    main()
