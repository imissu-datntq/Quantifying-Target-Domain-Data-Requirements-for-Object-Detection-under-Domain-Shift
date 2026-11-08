"""Bounded native-resolution training/validation pilot; never final evidence."""
from __future__ import annotations

import argparse
from pathlib import Path

from bdd_pipeline.training import parser as training_parser, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/bdd100k_revised"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--images", type=int, default=16)
    parser.add_argument("--validation-images", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--initialization", choices=["coco", "imagenet", "none"], default="coco")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.steps, args.images, args.validation_images, args.batch_size, args.accumulation, args.threads) < 1:
        parser.error("Pilot sizes must be positive")
    # Use a one-epoch pool no larger than the requested update count.
    # The optimizer floor then gives exactly the requested number of updates.
    image_count = min(args.images, args.steps * args.batch_size * args.accumulation)
    command = ["--dataset-root", str(args.dataset_root), "--output-dir", str(args.output_dir),
               "--smoke", "--initialization", args.initialization, "--limit-train", str(image_count),
               "--limit-val", str(args.validation_images), "--epochs", "1", "--floor-steps", str(args.steps),
               "--batch-size", str(args.batch_size), "--accumulation", str(args.accumulation),
               "--eval-every-updates", str(args.steps), "--save-every-updates", str(args.steps),
               "--warmup-updates", "0", "--device", args.device,
               "--threads", str(args.threads)]
    if args.amp:
        command.append("--amp")
    train(training_parser().parse_args(command))


if __name__ == "__main__":
    main()
