"""Bounded native-geometry MS-DAYOLO source/fine-tune smoke.

This adapter deliberately exercises Darknet's ordinary supervised path (no
``-da``), so the target fine-tune path cannot silently consume unlabeled target
data or target-domain detection losses.  It is a technical smoke only: the
source phase uses random initialization when no official MS source weight is
available, and neither phase is selection evidence.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path

from export_ms_validation import audit_weights


ROOT = Path(__file__).resolve().parent
UPSTREAM_SHA = "c655ceed3b99e6b499a9c907a3ae9c2deba98c1e"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def choose_rows(manifest: Path, count: int, seed: int) -> list[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: row["name"])
    if count > len(rows):
        raise ValueError(f"{manifest} contains {len(rows)} rows; need {count}")
    # Sampling is explicit and deterministic, while preserving the canonical
    # manifest membership and avoiding any access to validation or holdout rows.
    return random.Random(seed).sample(rows, count)


def write_view(output: Path, rows: list[dict[str, str]], role: str) -> Path:
    split_root = ROOT / "data/research_split_v1"
    inputs_root = ROOT / "data/research_inputs_v1"
    image_dir = output / role / "images"
    label_dir = output / role / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for row in rows:
        if row["image"] != "images/train/" + row["name"]:
            raise ValueError("supervised smoke accepts train images only")
        source_image = split_root / row["image"]
        source_label = inputs_root / row["yolo_label"]
        if not source_image.is_file() or not source_label.is_file():
            raise FileNotFoundError(row["name"])
        destination = image_dir / row["name"]
        label_destination = label_dir / (destination.stem + ".txt")
        destination.symlink_to(source_image)
        label_destination.symlink_to(source_label)
        paths.append(str(destination))
    path_file = output / f"{role}.txt"
    path_file.write_text("\n".join(paths) + "\n", encoding="utf-8")
    return path_file


def adapt_config(text: str, steps: int, pinned_memory_mb: int = 8192,
                 learning_rate: float = .0001) -> str:
    if text.count("filters=39") != 3 or text.count("classes=8") != 3:
        raise ValueError("unexpected upstream detector heads")
    text = text.replace("filters=39", "filters=45").replace("classes=8", "classes=10")
    # Ordinary supervised training consumes the full microbatch (unlike DA,
    # which splits it into source/target halves).  Batch=2 keeps the pinned
    # loader within the observed 4-GiB host allocation on this RTX 3060.
    values = dict(batch=2, subdivisions=2, width=1280, height=736,
                  learning_rate=learning_rate, burn_in=0, max_batches=steps,
                  policy="constant", mosaic=0, saturation=1, exposure=1, hue=0)
    for key, value in values.items():
        text, count = re.subn(r"^" + key + r"\s*=.*$", f"{key}={value}", text, flags=re.M)
        if count != 1:
            raise ValueError(f"expected one net setting: {key}")
    text = text.replace("[net]", f"[net]\nletter_box=1\nflip=0\noptimized_memory=2\nworkspace_size_limit_MB=256\npinned_memory_MB={pinned_memory_mb}", 1)
    text = text.replace("jitter=.3", "jitter=0\nresize=1\nrandom=0")
    text = text.replace("truth_thresh = 1\nrandom=0", "truth_thresh = 1")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["source", "finetune"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6, choices=range(6, 121))
    parser.add_argument("--domain", choices=["lighting", "weather", "compound"], default="lighting")
    parser.add_argument("--budget", type=int, choices=[100, 1000], default=100)
    parser.add_argument("--seed", type=int, choices=[42, 43], default=42)
    parser.add_argument("--initial-weights", type=Path,
                        help="Optional official convolutional initialization for source smoke")
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--pinned-memory-mb", type=int, choices=[4096, 8192], default=8192)
    parser.add_argument("--learning-rate", type=float, default=.0001)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    upstream = ROOT / "external/MS-DAYOLO"
    if not upstream.is_dir():
        raise FileNotFoundError(upstream)
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_SHA:
        raise RuntimeError(f"unexpected MS-DAYOLO revision: {revision}")
    if args.phase == "finetune" and args.source_checkpoint is None:
        parser.error("--source-checkpoint is required for fine-tuning")
    if args.phase == "source" and args.source_checkpoint is not None:
        parser.error("--source-checkpoint is only valid for fine-tuning; use --initial-weights")
    if args.initial_weights and not args.initial_weights.is_file():
        raise FileNotFoundError(args.initial_weights)
    if args.source_checkpoint and not args.source_checkpoint.is_file():
        raise FileNotFoundError(args.source_checkpoint)

    manifest = ROOT / "data/research_inputs_v1/manifests/domains"
    rows = choose_rows(manifest / ("source/train.csv" if args.phase == "source" else f"{args.domain}/train.csv"),
                       128 if args.phase == "source" else args.budget, args.seed)
    config = adapt_config((upstream / "cfg/ms-dayolo.cfg").read_text(encoding="utf-8"), args.steps,
                          args.pinned_memory_mb, args.learning_rate)
    command = ["stdbuf", "-oL", "-eL", str(upstream / "darknet"), "detector", "train",
               str(output / "bdd.data"), str(output / "smoke.cfg")]
    if args.initial_weights:
        command.append(str(args.initial_weights.resolve()))
    elif args.source_checkpoint:
        command.append(str(args.source_checkpoint.resolve()))
        command.append("-clear")
    command += ["-dont_show", "-gpus", "0"]
    print(json.dumps(dict(command=command, phase=args.phase,
                          domain="source" if args.phase == "source" else args.domain,
                          budget=len(rows), geometry=[1280, 736], adaptation=False,
                          selection_evidence=False), indent=2))
    if args.dry_run:
        return

    output.mkdir(parents=True)
    (output / "backup").mkdir()
    train_file = write_view(output, rows, "train")
    classes = json.loads((ROOT / "data/research_inputs_v1/metadata/classes.json").read_text())['classes']
    (output / "classes.names").write_text("\n".join(item["name"] for item in classes) + "\n", encoding="utf-8")
    (output / "smoke.cfg").write_text(config, encoding="utf-8")
    (output / "bdd.data").write_text(
        f"classes=10\ntrain={train_file}\nvalid={train_file}\nnames={output}/classes.names\nbackup={output}/backup\n",
        encoding="utf-8")
    state = dict(status="running", candidate="ms-dayolo", phase=args.phase,
                 domain="source" if args.phase == "source" else args.domain,
                 labeled_budget=0 if args.phase == "source" else len(rows), seed=args.seed,
                 steps=args.steps, geometry=[1280, 736], adaptation=False,
                 selection_evidence=False, holdout_access=False, pinned_memory_mb=args.pinned_memory_mb,
                 learning_rate=args.learning_rate,
                 upstream_sha=revision,
                 initial_weights_sha256=digest(args.initial_weights) if args.initial_weights else None,
                 source_checkpoint_sha256=digest(args.source_checkpoint) if args.source_checkpoint else None,
                 command=command)
    status = output / "status.json"
    status.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0",
               LD_LIBRARY_PATH=str(ROOT / ".venvs/ms-dayolo/lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))
    try:
        with (output / "train.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=output, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=3600)
        log_text = (output / "train.log").read_text(errors="replace")
        checkpoint = output / "backup/smoke_final.weights"
        checkpoint_audit = None
        checkpoint_seen = None
        if checkpoint.is_file():
            import struct
            with checkpoint.open("rb") as stream:
                header = stream.read(20)
                if len(header) == 20:
                    checkpoint_seen = struct.unpack("<iiiQ", header)[3]
            with checkpoint.open("rb") as stream:
                audit_weights(stream, 64745066, checkpoint_seen)
            checkpoint_audit = "passed"
        state.update(exit_code=result.returncode, checkpoint_exists=checkpoint.is_file(),
                     checkpoint_sha256=digest(checkpoint) if checkpoint.is_file() else None,
                     checkpoint_audit=checkpoint_audit, checkpoint_seen=checkpoint_seen,
                     nonfinite_log=bool(re.search(r"\b(?:nan|inf)\b", log_text, flags=re.I)),
                     iterations=len(re.findall(r"^[- ]+iteration \d+", log_text, flags=re.M)),
                     status="completed" if result.returncode == 0 and checkpoint.is_file() else "failed")
    except subprocess.TimeoutExpired:
        state.update(status="timeout", exit_code=None)
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        state["wall_seconds"] = time.monotonic() - started
        status.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, indent=2))
    if state["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
