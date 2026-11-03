"""Predict on final-test images without opening any annotation file.

Each completed run writes hashed prediction files for the final-test images of its own
target domain and of the source domain (forgetting axis). Scoring happens once, later,
from a frozen plan of these files (freeze_evaluation_plan.py --prediction-manifests),
so held-out labels are opened in a single pass after the whole grid is complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from bdd_pipeline.artifacts import atomic_json, load_checkpoint
from bdd_pipeline.data import BDDImages, DOMAINS, final_test_rows, read_manifests
from bdd_pipeline.evaluation import predict
from bdd_pipeline.model import MODEL_CONTRACT, build_model
from bdd_pipeline.training import choose_device
from prepare_revised_bdd100k import sha256

PREDICTION_SCHEMA = 1
IMPLEMENTATION = ("predict_frcnn.py", "bdd_pipeline/data.py", "bdd_pipeline/evaluation.py", "bdd_pipeline/model.py")


def run_identity(checkpoint: dict) -> dict:
    config = checkpoint["config"]
    return {key: config.get(key) for key in ("stage", "domain", "budget", "seed", "initialization", "epochs",
                                             "floor_steps", "total_updates", "effective_batch_size", "lr",
                                             "warmup_updates", "amp", "smoke", "manifest_sha256",
                                             "source_checkpoint_sha256")} | {
        "global_step": checkpoint["global_step"], "completed": checkpoint.get("training_completed", False)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/bdd100k_revised"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--domains", choices=DOMAINS, nargs="+",
                        help="Default: the run's own domain plus source (all four for the source model)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Smoke checkpoints only")
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1 or min(args.limit, args.workers) < 0:
        parser.error("Invalid batch/thread/limit/worker settings")
    torch.set_num_threads(args.threads)
    metadata, manifests = read_manifests(args.dataset_root)
    checkpoint = load_checkpoint(args.checkpoint)
    identity = run_identity(checkpoint)
    if identity["manifest_sha256"] != metadata["manifest_sha256"]:
        raise ValueError("Checkpoint dataset differs")
    if not identity["smoke"] and (not identity["completed"] or args.limit):
        raise ValueError("Research predictions need a completed checkpoint and no image limit")
    domains = args.domains or (list(DOMAINS) if identity["domain"] == "source" else [identity["domain"], "source"])
    if (args.output_dir / "prediction_manifest.json").exists():
        raise FileExistsError("Predictions already written for this output directory")
    device = choose_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("AMP requires CUDA")
    model = build_model("none").to(device)
    model.load_state_dict(checkpoint["model"])
    files = {}
    for domain in domains:
        rows = final_test_rows(manifests, domain)
        if args.limit:
            rows = rows[:args.limit]
        dataset = BDDImages(args.dataset_root, "test", rows)
        names = {image_id: name for name, image_id in dataset.image_ids.items()}
        predictions, seconds = predict(model, dataset, device, batch_size=args.batch_size, workers=args.workers, amp=args.amp)
        for prediction in predictions:
            prediction["file_name"] = names[prediction["image_id"]]
        path = args.output_dir / f"predictions_{domain}.json"
        atomic_json(path, predictions)
        image_list = "\n".join(row["name"] for row in dataset.rows).encode("utf-8")
        files[domain] = {"file": path.name, "sha256": sha256(path), "images": len(dataset),
                         "image_names_sha256": hashlib.sha256(image_list).hexdigest(),
                         "predictions": len(predictions), "inference_seconds": seconds}
        print(f"{domain}: {len(predictions)} predictions on {len(dataset)} images in {seconds:.1f}s", flush=True)
    atomic_json(args.output_dir / "prediction_manifest.json", {
        "schema": PREDICTION_SCHEMA, "role": "final_test", "model_contract": MODEL_CONTRACT,
        "checkpoint_sha256": sha256(args.checkpoint), "run": identity, "limited": bool(args.limit),
        "amp": args.amp, "batch_size": args.batch_size, "files": files,
        "implementation_sha256": {name: sha256(Path(__file__).parent / name) for name in IMPLEMENTATION},
        "test_labels_accessed": False})
    print(json.dumps({domain: entry["predictions"] for domain, entry in files.items()}))


if __name__ == "__main__":
    main()
