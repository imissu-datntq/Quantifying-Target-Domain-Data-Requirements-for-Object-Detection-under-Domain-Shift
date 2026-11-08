"""Validation evaluation, or final held-out evaluation of a frozen campaign plan (checkpoints or prediction files)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from bdd_pipeline.artifacts import atomic_json, load_checkpoint
from bdd_pipeline.data import (BDDDetection, CLASSES, DOMAINS, HOLDOUT_VAL_DOMAINS, RELIABLE_CLASSES, final_test_rows,
                               read_manifests, selection_rows)
from bdd_pipeline.evaluation import coco_evaluator, evaluate_model, fast_bootstrap_class_ap, score_predictions
from bdd_pipeline.model import MODEL_CONTRACT, build_model
from bdd_pipeline.training import choose_device
from prepare_revised_bdd100k import sha256


def evaluate_validation(args) -> None:
    if args.checkpoint is None or args.domain is None:
        raise ValueError("Validation requires --checkpoint and --domain")
    if args.output_dir.exists():
        raise FileExistsError("Choose a new evaluation output directory")
    metadata, manifests = read_manifests(args.dataset_root)
    checkpoint = load_checkpoint(args.checkpoint)
    if checkpoint["config"]["manifest_sha256"] != metadata["manifest_sha256"]:
        raise ValueError("Checkpoint dataset differs")
    val_path = args.dataset_root / "annotations" / "bdd100k_labels_images_val.json"
    if sha256(val_path) != checkpoint["config"]["development_annotation_sha256"]["val"]:
        raise ValueError("Validation annotations changed since training")
    rows = selection_rows(manifests, args.domain)  # refuses weather/compound: their val is held out
    if args.limit:
        if not checkpoint["config"]["smoke"]:
            raise ValueError("Limited validation is permitted only for smoke checkpoints")
        rows = rows[:args.limit]
    device = choose_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("AMP requires CUDA")
    model = build_model("none").to(device)
    model.load_state_dict(checkpoint["model"])
    dataset = BDDDetection(args.dataset_root, "val", rows)
    metrics, predictions = evaluate_model(model, dataset, device, batch_size=args.batch_size,
                                           workers=args.workers, amp=args.amp)
    metrics.update(split="val", domain=args.domain, checkpoint_sha256=sha256(args.checkpoint),
                   smoke=checkpoint["config"]["smoke"], limited=bool(args.limit))
    atomic_json(args.output_dir / "metrics.json", metrics)
    atomic_json(args.output_dir / "predictions.json", predictions)
    atomic_json(args.output_dir / "image_ids.json", dataset.image_ids)
    print(json.dumps(metrics, indent=2))


def validate_plan(plan: dict, metadata: dict) -> None:
    if plan.get("schema") != 2 or plan.get("model_contract") != MODEL_CONTRACT:
        raise ValueError("Invalid final-plan schema/model contract")
    if plan.get("reliable_classes") != list(RELIABLE_CLASSES) or plan.get("holdout_val_domains") != list(HOLDOUT_VAL_DOMAINS):
        raise ValueError("Final plan does not use the amended reliable-class/holdout protocol")
    if plan["manifest_sha256"] != metadata["manifest_sha256"]:
        raise ValueError("Final-plan dataset differs")
    for name, digest in plan.get("evaluation_implementation_sha256", {}).items():
        if sha256(Path(__file__).parent / name) != digest:
            raise ValueError("Evaluation implementation changed after freezing")
    if not plan.get("checkpoints") or not plan.get("domains"):
        raise ValueError("Final plan must specify checkpoints and domains")
    if set(plan["domains"]) - set(DOMAINS) or len(set(plan["domains"])) != len(plan["domains"]):
        raise ValueError("Invalid/duplicate final domains")
    for entry in plan["checkpoints"]:
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError("Checkpoint changed after freezing")
        checkpoint = load_checkpoint(path)
        if checkpoint["config"]["smoke"] or not checkpoint.get("training_completed", False):
            raise ValueError("Final plan contains smoke/incomplete checkpoint")
        if checkpoint["config"]["manifest_sha256"] != metadata["manifest_sha256"]:
            raise ValueError("Final checkpoint belongs to different dataset")


def validate_prediction_plan(plan: dict, metadata: dict) -> None:
    """Schema 3: every frozen prediction file and the scoring code must be unchanged."""
    if plan.get("model_contract") != MODEL_CONTRACT or plan["manifest_sha256"] != metadata["manifest_sha256"]:
        raise ValueError("Final-plan dataset/model contract differs")
    if plan.get("reliable_classes") != list(RELIABLE_CLASSES) or plan.get("holdout_val_domains") != list(HOLDOUT_VAL_DOMAINS):
        raise ValueError("Final plan does not use the amended reliable-class/holdout protocol")
    for name, digest in plan["evaluation_implementation_sha256"].items():
        if sha256(Path(__file__).parent / name) != digest:
            raise ValueError("Evaluation implementation changed after freezing")
    if not plan.get("prediction_sets") or set(plan["domains"]) - set(DOMAINS):
        raise ValueError("Final plan must specify prediction sets and valid domains")
    for entry in plan["prediction_sets"]:
        if entry["run"]["smoke"] or not entry["run"]["completed"]:
            raise ValueError("Final plan contains smoke/incomplete predictions")
        if sha256(Path(entry["manifest"])) != entry["manifest_sha256"]:
            raise ValueError("Prediction manifest changed after freezing")
        for file in entry["files"].values():
            if sha256(Path(file["path"])) != file["sha256"]:
                raise ValueError("Prediction file changed after freezing")


def score_prediction_plan(plan: dict, plan_digest: str, manifests: dict, args) -> list[dict]:
    """Open each domain's held-out labels once and score every frozen prediction set on it."""
    results = []
    for domain in plan["domains"]:
        dataset = BDDDetection(args.dataset_root, "test", final_test_rows(manifests, domain))
        ground_truth = dataset.coco_ground_truth()
        for entry in plan["prediction_sets"]:
            if domain not in entry["files"]:
                continue
            target = args.output_dir / entry["label"] / domain
            if (target / "metrics.json").exists():  # scored before an interruption; written last, so complete
                metrics = json.loads((target / "metrics.json").read_text(encoding="utf-8"))
                results.append({key: metrics[key] for key in ("label", "domain", "run", "reliable_AP50_95", "AP50_95", "AP50")})
                continue
            predictions = json.loads(Path(entry["files"][domain]["path"]).read_text(encoding="utf-8"))
            for prediction in predictions:
                prediction["image_id"] = dataset.image_ids[prediction.pop("file_name")]  # image_id maps by file name
            evaluator = coco_evaluator(ground_truth, predictions)
            metrics = score_predictions(ground_truth, predictions, evaluator=evaluator)
            bootstrap = fast_bootstrap_class_ap(evaluator, plan["bootstrap"]["resamples"], plan["bootstrap"]["seed"])
            metrics.update(split="test", domain=domain, label=entry["label"], run=entry["run"],
                           checkpoint_sha256=entry["checkpoint_sha256"], frozen_plan_sha256=plan_digest,
                           bootstrap={**plan["bootstrap"], "file": "bootstrap_class_ap.npy", "class_order": list(CLASSES)})
            target.mkdir(parents=True, exist_ok=True)
            with open(target / "bootstrap_class_ap.npy.tmp", "wb") as handle:
                np.save(handle, bootstrap)
            os.replace(target / "bootstrap_class_ap.npy.tmp", target / "bootstrap_class_ap.npy")
            atomic_json(target / "metrics.json", metrics)
            results.append({key: metrics[key] for key in ("label", "domain", "run", "reliable_AP50_95", "AP50_95", "AP50")})
            print(f"{entry['label']} on {domain}: reliable {metrics['reliable_AP50_95']}", flush=True)
    return results


def evaluate_final(args) -> None:
    if args.checkpoint or args.domain or args.limit:
        raise ValueError("Final execution uses only the frozen plan; checkpoint/domain/limit overrides forbidden")
    plan = json.loads(args.final_plan.read_text(encoding="utf-8"))
    metadata, manifests = read_manifests(args.dataset_root)
    if plan.get("schema") == 3:
        # Validate every frozen prediction file before accessing any test annotations.
        validate_prediction_plan(plan, metadata)
        plan_digest = sha256(args.final_plan)
        lock = args.final_plan.with_suffix(args.final_plan.suffix + ".opened")
        if args.resume_interrupted:
            # Finish the same one-shot pass: same plan, same output, nothing published yet.
            opened = json.loads(lock.read_text(encoding="utf-8"))
            if opened["plan_sha256"] != plan_digest or opened["output_dir"] != str(args.output_dir.resolve()):
                raise ValueError("Lock belongs to a different plan or output directory")
            if (args.output_dir / "results.json").exists():
                raise FileExistsError("This plan's scoring already completed")
            if json.loads((args.output_dir / "frozen_plan.json").read_text(encoding="utf-8")) != plan:
                raise ValueError("Output directory holds a different frozen plan")
        else:
            if args.output_dir.exists():
                raise FileExistsError("Final output directory already exists")
            with lock.open("x", encoding="utf-8") as handle:
                json.dump({"plan_sha256": plan_digest, "output_dir": str(args.output_dir.resolve()),
                           "status": "Holdout access started; only --resume-interrupted of this pass may continue it"}, handle)
            args.output_dir.mkdir(parents=True)
            atomic_json(args.output_dir / "frozen_plan.json", plan)
        results = score_prediction_plan(plan, plan_digest, manifests, args)
        atomic_json(args.output_dir / "results.json", {"plan_sha256": plan_digest, "results": results, "completed": True})
        print(f"Completed locked final scoring: {args.output_dir}")
        return
    if args.resume_interrupted:
        raise ValueError("--resume-interrupted applies to prediction-file plans (schema 3) only")
    # Validate every selected checkpoint before accessing any test annotations.
    validate_plan(plan, metadata)
    device = choose_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("AMP requires CUDA")
    plan_digest = sha256(args.final_plan)
    if args.amp != plan["amp"] or args.batch_size != plan["eval_batch_size"]:
        raise ValueError("Evaluation precision/batch differs from frozen plan")
    # One-shot lock adjacent to plan survives changing the output directory.
    lock = args.final_plan.with_suffix(args.final_plan.suffix + ".opened")
    if args.output_dir.exists():
        raise FileExistsError("Final output directory already exists")
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"plan_sha256": plan_digest, "output_dir": str(args.output_dir.resolve()),
                   "status": "Holdout access started; this plan cannot be rerun"}, handle)
    args.output_dir.mkdir(parents=True)
    atomic_json(args.output_dir / "frozen_plan.json", plan)
    results = []
    datasets = {domain: BDDDetection(args.dataset_root, "test", final_test_rows(manifests, domain)) for domain in plan["domains"]}
    for entry in plan["checkpoints"]:
        checkpoint = load_checkpoint(Path(entry["path"]))
        model = build_model("none").to(device)
        model.load_state_dict(checkpoint["model"])
        for domain, dataset in datasets.items():
            metrics, predictions = evaluate_model(model, dataset, device, batch_size=args.batch_size,
                                                   workers=args.workers, amp=args.amp)
            metrics.update(split="test", domain=domain, checkpoint_sha256=entry["sha256"],
                           checkpoint_label=entry["label"], frozen_plan_sha256=plan_digest)
            target = args.output_dir / entry["label"] / domain
            atomic_json(target / "metrics.json", metrics)
            atomic_json(target / "predictions.json", predictions)
            atomic_json(target / "image_ids.json", dataset.image_ids)
            results.append(metrics)
        del model, checkpoint
    atomic_json(args.output_dir / "results.json", {"plan_sha256": plan_digest, "results": results, "completed": True})
    print(f"Completed locked final evaluation: {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/bdd100k_revised"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--domain", choices=DOMAINS)
    parser.add_argument("--final-plan", type=Path, help="Explicit held-out execution; must be frozen before reading test labels")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume-interrupted", action="store_true",
                        help="Finish an interrupted schema-3 pass of the same plan into the same output directory")
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1 or min(args.limit, args.workers) < 0:
        parser.error("Invalid batch/thread/limit/worker settings")
    torch.set_num_threads(args.threads)
    (evaluate_final if args.final_plan else evaluate_validation)(args)


if __name__ == "__main__":
    main()
