"""CPU cost benchmark of torchvision detector candidates at native 1280x720.

Random weights (no download): timing and parameter counts only, never accuracy.
The architecture choice rests on public domain-adaptation implementations (proposal
6.1); this records what the two-stage choice costs relative to one-stage options.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fcos_resnet50_fpn, retinanet_resnet50_fpn

from bdd_pipeline.artifacts import environment
from bdd_pipeline.data import BDDDetection, CLASSES, domain_rows, read_manifests
from bdd_pipeline.model import MODEL_CONTRACT

CANDIDATES = {"fasterrcnn_resnet50_fpn": fasterrcnn_resnet50_fpn, "fcos_resnet50_fpn": fcos_resnet50_fpn,
              "retinanet_resnet50_fpn": retinanet_resnet50_fpn}


def timed(function, repeats: int) -> float:
    function()  # warm-up
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    return (time.perf_counter() - start) / repeats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/bdd100k_revised"))
    parser.add_argument("--images", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    _, manifests = read_manifests(args.dataset_root)
    dataset = BDDDetection(args.dataset_root, "train", domain_rows(manifests, "train", "source")[:args.images])
    samples = [dataset[i] for i in range(len(dataset))]
    results = {}
    for name, builder in CANDIDATES.items():
        model = builder(weights=None, weights_backbone=None, num_classes=len(CLASSES) + 1,
                        min_size=MODEL_CONTRACT["min_size"], max_size=MODEL_CONTRACT["max_size"])
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-4, momentum=0.9)
        state = {"index": 0}

        def infer():
            model.eval()
            with torch.inference_mode():
                model([samples[state["index"] % len(samples)][0]])
            state["index"] += 1

        def train_step():
            model.train()
            image, target = samples[state["index"] % len(samples)]
            losses = model([image], [{"boxes": target["boxes"], "labels": target["labels"]}])
            optimizer.zero_grad()
            sum(losses.values()).backward()
            optimizer.step()
            state["index"] += 1

        results[name] = {"parameters_millions": sum(p.numel() for p in model.parameters()) / 1e6,
                         "inference_seconds_per_image": timed(infer, args.images),
                         "train_seconds_per_image": timed(train_step, args.images)}
        print(name, json.dumps(results[name]), flush=True)
    baseline = results["fasterrcnn_resnet50_fpn"]
    for entry in results.values():
        entry["train_cost_vs_faster_rcnn"] = entry["train_seconds_per_image"] / baseline["train_seconds_per_image"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results, "images": args.images, "threads": args.threads,
                                       "note": "CPU, batch 1, random weights; relative cost only", "environment": environment()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
