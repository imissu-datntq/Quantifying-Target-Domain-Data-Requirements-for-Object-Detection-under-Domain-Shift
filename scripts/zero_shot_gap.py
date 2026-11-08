"""Pilot gate: zero-shot gap mAP_source - mAP(0) per target shift, without held-out labels.

The source score uses source *validation* images; each target uses its *training
pool*, which a source-only model has never seen and which fine-tuning may use
anyway. No final-test or holdout-val image is read. Image-level bootstrap gives a
confidence interval on each gap. A shift whose interval does not clear
--min-gap is flagged: its Recovery denominator mAP_c - mAP(0) is at risk of being ~0.

--coco-proxy evaluates the public COCO-pretrained Faster R-CNN on the three classes
COCO shares with the reliable roster (pedestrian/person, car, traffic light). It is a
pre-training difficulty probe, not the proposal's source-model gap.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

from bdd_pipeline.artifacts import atomic_json, environment, load_checkpoint
from bdd_pipeline.data import (BDDDetection, CLASSES, DOMAINS, RELIABLE_CLASSES, domain_rows, holdout_names,
                               read_manifests, selection_rows)
from bdd_pipeline.evaluation import bootstrap_class_ap, class_ap, class_precision, coco_evaluator, predict
from bdd_pipeline.model import MODEL_CONTRACT, build_model
from bdd_pipeline.training import choose_device
from prepare_revised_bdd100k import sha256

# torchvision COCO label ids -> revised BDD category ids (1-based CLASSES order)
COCO_TO_BDD = {1: CLASSES.index("pedestrian") + 1, 3: CLASSES.index("car") + 1, 10: CLASSES.index("traffic light") + 1}
PROXY_CLASSES = ("pedestrian", "car", "traffic light")
# Candidate replacement shifts probed on their train pool only (proposal 5: snow replicates the weather arm).
CANDIDATE_DOMAINS = {"snowy": ("daytime", "snowy")}


def probe_rows(manifests: dict, domain: str, count: int, seed: int) -> list[dict]:
    if domain == "source":
        rows = selection_rows(manifests, "source")
    elif domain in CANDIDATE_DOMAINS:
        rows = [row for row in manifests["train"] if (row["timeofday"], row["weather"]) == CANDIDATE_DOMAINS[domain]]
    else:
        rows = domain_rows(manifests, "train", domain)
    rows = sorted(rows, key=lambda row: row["name"])
    return rows if count == 0 or count >= len(rows) else sorted(random.Random(f"{seed}/{domain}").sample(rows, count), key=lambda row: row["name"])


def restrict(ground_truth: dict, classes: tuple[str, ...]) -> dict:
    keep = {CLASSES.index(name) + 1 for name in classes}
    return dict(ground_truth, categories=[c for c in ground_truth["categories"] if c["id"] in keep],
                annotations=[a for a in ground_truth["annotations"] if a["category_id"] in keep])


def coco_proxy_model() -> torch.nn.Module:
    return fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1, min_size=MODEL_CONTRACT["min_size"],
                                   max_size=MODEL_CONTRACT["max_size"], box_score_thresh=MODEL_CONTRACT["roi_score_threshold"],
                                   box_nms_thresh=MODEL_CONTRACT["roi_nms_threshold"],
                                   box_detections_per_img=MODEL_CONTRACT["detections_per_image"])


def interval(values: np.ndarray) -> list[float]:
    return [float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/bdd100k_revised"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Completed source-domain checkpoint")
    source.add_argument("--coco-proxy", action="store_true")
    parser.add_argument("--domains", nargs="+", choices=[d for d in DOMAINS if d != "source"] + list(CANDIDATE_DOMAINS), default=["weather", "lighting", "compound"])
    parser.add_argument("--images-per-domain", type=int, default=0, help="0 uses every eligible image")
    parser.add_argument("--sample-seed", type=int, default=20261019)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--min-gap", type=float, default=0.01, help="Smallest AP gap treated as a usable shift")
    parser.add_argument("--allow-smoke", action="store_true", help="Software check only; never a gate decision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-minutes", type=float, default=float("inf"), help="Stop after this wall time; cached predictions resume")
    parser.add_argument("--resume", action="store_true", help="Continue cached per-domain predictions in --output-dir")
    args = parser.parse_args()
    if args.images_per_domain < 0 or args.bootstrap < 1 or args.min_gap < 0:
        parser.error("Invalid sample/bootstrap/gap settings")
    if args.output_dir.exists() and not args.resume:
        parser.error("Choose a new output directory or pass --resume")
    invocation_start = time.perf_counter()
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    metadata, manifests = read_manifests(args.dataset_root)
    if args.coco_proxy:
        model, classes, label_map, model_id = coco_proxy_model(), PROXY_CLASSES, COCO_TO_BDD, "torchvision COCO_V1 fasterrcnn_resnet50_fpn"
    else:
        checkpoint = load_checkpoint(args.checkpoint)
        config = checkpoint["config"]
        if config["stage"] != "source" or config["manifest_sha256"] != metadata["manifest_sha256"]:
            raise ValueError("Gap pilot needs a source-stage checkpoint for this dataset")
        if (config["smoke"] or not checkpoint.get("training_completed")) and not args.allow_smoke:
            raise ValueError("Gap pilot needs a completed non-smoke source checkpoint")
        model = build_model("none")
        model.load_state_dict(checkpoint["model"])
        # Per-class COCO AP ignores other classes, so keep only reliable-class detections.
        label_map = {CLASSES.index(name) + 1: CLASSES.index(name) + 1 for name in RELIABLE_CLASSES}
        classes, model_id = RELIABLE_CLASSES, f"source checkpoint {sha256(args.checkpoint)}"
    model.to(device)
    protected = holdout_names(manifests)
    domains = ["source"] + list(args.domains)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_ids = [CLASSES.index(name) + 1 for name in classes]
    assert class_ids == sorted(class_ids), "class order must follow COCO category ids"
    results, boots = {}, {}
    for offset, domain in enumerate(domains):
        rows = probe_rows(manifests, domain, args.images_per_domain, args.sample_seed)
        if {row["name"] for row in rows} & protected:
            raise AssertionError("Gap probe must never read held-out labels")
        dataset = BDDDetection(args.dataset_root, "val" if domain == "source" else "train", rows)
        start = time.perf_counter()
        cache_path = args.output_dir / f"partial_{domain}.json"
        names = [row["name"] for row in rows]
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {"images": names, "done": [], "predictions": [], "seconds": 0.0}
        if cache["images"] != names:
            raise ValueError(f"Cached {domain} sample differs; use a new output directory")
        done = set(cache["done"])
        order = [i for i in range(len(rows)) if dataset.image_ids[rows[i]["name"]] not in done]
        for begin in range(0, len(order), 10):
            if time.perf_counter() - invocation_start > args.max_minutes * 60:
                atomic_json(cache_path, cache)
                print(f"Time budget reached during {domain} ({len(cache['done'])}/{len(rows)} images); rerun with --resume", flush=True)
                return
            chunk = order[begin:begin + 10]
            chunk_predictions, chunk_seconds = predict(model, dataset, device, label_map=label_map, indices=chunk)
            cache["predictions"] += chunk_predictions
            cache["done"] += [dataset.image_ids[rows[i]["name"]] for i in chunk]
            cache["seconds"] += chunk_seconds
            atomic_json(cache_path, cache)
        predictions, seconds = cache["predictions"], cache["seconds"]
        ground_truth = restrict(dataset.coco_ground_truth(), classes)
        evaluator = coco_evaluator(ground_truth, predictions)
        point = class_ap(class_precision(evaluator))
        boots[domain] = bootstrap_class_ap(evaluator, args.bootstrap, args.sample_seed + offset)
        support = {name: sum(a["category_id"] == cid for a in ground_truth["annotations"]) for name, cid in zip(classes, class_ids)}
        results[domain] = {"split_role": "source val" if domain == "source" else "target train pool (unseen by model)",
                           "images": len(rows), "inference_seconds": seconds, "seconds_per_image": seconds / len(rows),
                           "reliable_AP50_95": float(np.mean(point)), "reliable_AP50_95_ci95": interval(boots[domain].mean(axis=1)),
                           "per_class_AP50_95": dict(zip(classes, map(float, point))), "instances": support}
        atomic_json(args.output_dir / f"predictions_{domain}.json", predictions)
        atomic_json(args.output_dir / f"images_{domain}.json", [row["name"] for row in rows])
        print(f"{domain}: {len(rows)} images, reliable AP50:95 {np.mean(point):.4f} "
              f"({time.perf_counter() - start:.0f}s)", flush=True)
    gaps = {}
    for domain in args.domains:
        difference = boots["source"].mean(axis=1) - boots[domain].mean(axis=1)
        low, high = interval(difference)
        point = results["source"]["reliable_AP50_95"] - results[domain]["reliable_AP50_95"]
        gaps[domain] = {"gap": point, "ci95": [low, high], "relative_gap": point / results["source"]["reliable_AP50_95"] if results["source"]["reliable_AP50_95"] > 0 else None,
                        "per_class_gap": {name: results["source"]["per_class_AP50_95"][name] - results[domain]["per_class_AP50_95"][name] for name in classes},
                        "gate": "usable shift" if low > args.min_gap else "no clear gap: Recovery denominator at risk"}
    summary = {"model": model_id, "proxy": args.coco_proxy, "classes": list(classes), "bootstrap_resamples": args.bootstrap,
               "sample_seed": args.sample_seed, "min_gap": args.min_gap, "heldout_labels_accessed": False,
               "domains": results, "gaps": gaps, "environment": environment()}
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(gaps, indent=2))


if __name__ == "__main__":
    main()
