"""Common update schedule, weighted accumulation, resumable detection training."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from prepare_revised_bdd100k import DOMAINS, sha256
from .artifacts import (SCHEMA, atomic_checkpoint, atomic_json, environment,
                        load_checkpoint, random_state, restore_random_state, seed_everything)
from .data import (BDDDetection, RELIABLE_CLASSES, collate, domain_rows, nested_rows, read_manifests,
                   selection_rows)
from .evaluation import evaluate_model
from .model import MODEL_CONTRACT, build_model

CLASS_NAMES = MODEL_CONTRACT["class_names"]


def schedule_size(images: int, batch_size: int, accumulation: int, epochs: int, floor_steps: int) -> tuple[int, int]:
    if min(images, batch_size, accumulation, epochs) < 1 or floor_steps < 0:
        raise ValueError("Invalid schedule parameters")
    per_epoch = math.ceil(math.ceil(images / batch_size) / accumulation)
    return max(floor_steps, epochs * per_epoch), per_epoch


def learning_rate_multiplier(step: int, total: int, warmup: int) -> float:
    warmup = min(warmup, total)
    if warmup and step < warmup:
        return 0.1 + 0.9 * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return 0.01 + 0.99 * (1 + math.cos(math.pi * min(1, max(0, progress)))) / 2


def choose_device(value: str) -> torch.device:
    device = torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")
    return device


def train(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("AMP in this pipeline requires CUDA")
    if args.stage == "source" and (args.domain != "source" or args.budget != "full"):
        raise ValueError("Source training requires domain=source and budget=full")
    if args.stage != "source" and args.domain == "source":
        raise ValueError("Target fine-tuning/control must use a target domain")
    if (args.limit_train or args.limit_val or args.initialization == "none") and not args.smoke:
        raise ValueError("Limits/random initialization are only allowed with --smoke")
    if args.stage == "fine-tune" and args.source_checkpoint is None:
        raise ValueError("Fine-tuning requires --source-checkpoint")
    if args.stage != "fine-tune" and args.source_checkpoint is not None:
        raise ValueError("Source checkpoint is used only for fine-tune")
    if not args.smoke and args.floor_steps < 1:
        raise ValueError("Set a positive common step floor for research runs")
    root = args.dataset_root.resolve()
    metadata, manifests = read_manifests(root)
    pool = domain_rows(manifests, "train", args.domain)
    selected = nested_rows(pool, args.budget, args.seed)
    # Only source training selects a checkpoint on validation. Target runs keep the final update of the
    # common fixed schedule, so every shift uses one rule and weather/compound val stays held out.
    validation = selection_rows(manifests, args.domain) if args.stage == "source" else []
    if args.limit_train:
        selected = selected[:args.limit_train]
    if args.limit_val:
        validation = validation[:args.limit_val]
    total, per_epoch = schedule_size(len(selected), args.batch_size, args.accumulation, args.epochs, args.floor_steps)
    source = load_checkpoint(args.source_checkpoint) if args.source_checkpoint else None
    if source and (source["config"]["stage"] != "source" or source["config"]["manifest_sha256"] != metadata["manifest_sha256"]):
        raise ValueError("Source checkpoint stage/dataset mismatch")
    if source and not args.smoke and (source["config"]["smoke"] or not source.get("training_completed", False)):
        raise ValueError("Research fine-tuning requires a completed non-smoke source checkpoint")
    config = {"stage": args.stage, "domain": args.domain, "budget": args.budget,
              "implementation_sha256": {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
              "runtime_packages": environment()["packages"],
              "pool_images": len(pool), "train_images": len(selected), "val_images": len(validation),
              "seed": args.seed, "manifest_sha256": metadata["manifest_sha256"],
              "development_annotation_sha256": {split: sha256(root / "annotations" / f"bdd100k_labels_images_{split}.json") for split in ("train", "val")},
              "initialization": args.initialization if source is None else "source-checkpoint",
              "source_checkpoint_sha256": sha256(args.source_checkpoint) if source else None,
              "epochs": args.epochs, "floor_steps": args.floor_steps,
              "total_updates": total, "updates_per_epoch": per_epoch,
              "batch_size": args.batch_size, "accumulation": args.accumulation,
              "effective_batch_size": args.batch_size * args.accumulation,
              "lr": args.lr, "momentum": args.momentum, "weight_decay": args.weight_decay,
              "warmup_updates": args.warmup_updates, "lr_schedule": "linear warmup then cosine to 1% base LR, indexed by optimizer update",
              "clip_grad_norm": args.clip_grad_norm, "amp": args.amp, "device_type": device.type,
              "horizontal_flip": not args.no_flip,
              "sanitation": "reject nonfinite/nonpositive; clip bounds; drop empty clipped boxes",
              "reliable_classes": list(RELIABLE_CLASSES),
              "checkpoint_selection": "best source-val checkpoint_metric" if validation else "final update of fixed schedule",
              "checkpoint_metric": args.checkpoint_metric, "eval_every_updates": args.eval_every_updates,
              "eval_batch_size": args.eval_batch_size,
              "save_every_updates": args.save_every_updates,
              "smoke": args.smoke, "threads": args.threads, "workers": args.workers}
    if not args.smoke and args.checkpoint_metric != "reliable_AP50_95":
        raise ValueError("Proposal primary selection metric is reliable_AP50_95")
    if args.dry_run:
        print(json.dumps({"validated_configuration": config, "model_contract": MODEL_CONTRACT,
                          "test_annotations_accessed": False, "model_created": False}, indent=2))
        return config
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if checkpoint and checkpoint["config"] != config:
        differences = [key for key in config if checkpoint["config"].get(key) != config[key]]
        raise ValueError(f"Resume configuration differs: {differences}")
    run = args.output_dir.resolve()
    if args.resume and args.resume.resolve() != run / "last.pt":
        raise ValueError("Exact resume must use last.pt in the original run directory")
    if run.exists() and any(run.iterdir()) and checkpoint is None:
        raise FileExistsError(f"Use a new output directory or --resume: {run}")
    if checkpoint and (not (run / "config.json").exists() or json.loads((run / "config.json").read_text(encoding="utf-8")) != config):
        raise ValueError("Resume must use its original run directory and configuration")
    seed_everything(args.seed)
    model = build_model("none" if checkpoint or source else args.initialization).to(device)
    if checkpoint or source:
        model.load_state_dict((checkpoint or source)["model"])
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    train_data = BDDDetection(root, "train", selected, augment=not args.no_flip, seed=args.seed)
    val_data = BDDDetection(root, "val", validation) if validation else None
    if val_data is not None and not args.smoke:
        present = {CLASS_NAMES[annotation["category_id"] - 1] for annotation in val_data.coco_ground_truth()["annotations"]}
        if set(RELIABLE_CLASSES) - present:
            raise ValueError("Validation lacks a fixed reliable class; primary selection metric undefined")
    run.mkdir(parents=True, exist_ok=True)
    if checkpoint is None:
        atomic_json(run / "config.json", config)
        atomic_json(run / "environment.json", environment())
        atomic_json(run / "model_contract.json", MODEL_CONTRACT)
        atomic_json(run / "selected_images.json", {"train": [row["name"] for row in selected], "val": [row["name"] for row in validation]})
        atomic_json(run / "annotation_audit.json", {"train": dict(train_data.audit), "val": dict(val_data.audit) if val_data else None})
    step = checkpoint["global_step"] if checkpoint else 0
    best_score = checkpoint["best_score"] if checkpoint else -1.0
    processed = checkpoint["processed_images"] if checkpoint else 0
    training_seconds = checkpoint["training_seconds"] if checkpoint else 0.0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_random_state(checkpoint["random_state"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    invocation_start, completed_this_invocation = time.perf_counter(), 0
    limit = args.stop_after_updates or total
    log_path = run / "training.jsonl"
    # Trim log rows after the last durable checkpoint after a interrupted write.
    if checkpoint and log_path.exists():
        lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if json.loads(line)["step"] <= step]
        log_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    while step < total and completed_this_invocation < limit:
        epoch, offset = divmod(step, per_epoch)
        train_data.epoch = epoch
        order = list(range(len(train_data)))
        random.Random(f"{args.seed}/epoch/{epoch}").shuffle(order)
        start_image = offset * args.batch_size * args.accumulation
        batches = [order[i:i + args.batch_size] for i in range(start_image, len(order), args.batch_size)]
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = iter(DataLoader(train_data, batch_sampler=batches, num_workers=args.workers,
                                 collate_fn=collate, pin_memory=device.type == "cuda", generator=generator))
        remaining_microbatches = len(batches)
        while remaining_microbatches and step < total and completed_this_invocation < limit:
            if device.type == "cuda":
                torch.cuda.synchronize()
            tick = time.perf_counter()
            model.train()
            micro_count = min(args.accumulation, remaining_microbatches)
            images_this_step = sum(len(batch) for batch in batches[len(batches) - remaining_microbatches:len(batches) - remaining_microbatches + micro_count])
            optimizer.zero_grad(set_to_none=True)
            lr = args.lr * learning_rate_multiplier(step, total, args.warmup_updates)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss_values = {}
            for _ in range(micro_count):
                images, targets = next(loader)
                weight = len(images) / images_this_step
                moved_targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
                context = torch.autocast("cuda", dtype=torch.float16) if args.amp else nullcontext()
                with context:
                    losses = model([image.to(device) for image in images], moved_targets)
                    loss = sum(losses.values()) * weight
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite training loss at step {step}")
                scaler.scale(loss).backward()
                for key, value in losses.items():
                    loss_values[key] = loss_values.get(key, 0.0) + float(value.detach()) * weight
            scaler.unscale_(optimizer)
            # Under AMP an overflowing loss scale yields nonfinite gradients; GradScaler skips that update
            # and lowers the scale. Without AMP a nonfinite gradient is an error.
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm, error_if_nonfinite=not args.amp)
            skipped = not bool(torch.isfinite(norm))
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - tick
            training_seconds += elapsed
            processed += images_this_step
            step += 1
            completed_this_invocation += 1
            remaining_microbatches -= micro_count
            metrics = None
            if val_data is not None and (step % args.eval_every_updates == 0 or step == total):
                metrics, predictions = evaluate_model(model, val_data, device, batch_size=args.eval_batch_size,
                                                       workers=args.workers, amp=args.amp)
                metrics.update(split="val", domain=args.domain, step=step, smoke=args.smoke)
                score = metrics[args.checkpoint_metric]
                if score is None:
                    if not args.smoke:
                        raise ValueError("Checkpoint selection metric is undefined")
                    score = 0.0  # tiny smoke validation subsets may miss a reliable class
                improved = score > best_score
                if improved:
                    best_score = score
                atomic_json(run / f"validation_step_{step}.json", metrics)
                atomic_json(run / "latest_val_predictions.json", predictions)
            else:
                improved = False
            record = {"step": step, "epoch": epoch, "images": images_this_step, "loss": sum(loss_values.values()),
                      "losses": loss_values, "lr": lr, "grad_norm": None if skipped else float(norm), "amp_step_skipped": skipped, "seconds": elapsed,
                      "validation": metrics}
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            state = {"schema": SCHEMA, "model_contract": MODEL_CONTRACT, "config": config,
                     "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                     "global_step": step, "epoch": step // per_epoch, "update_in_epoch": step % per_epoch,
                     "best_score": best_score, "random_state": random_state(), "processed_images": processed,
                     "training_seconds": training_seconds, "validation": metrics,
                     "training_completed": step >= total, "trained_updates": step,
                     "lr_state": {"completed_updates": step, "total_updates": total}}
            if (step % args.save_every_updates == 0 or metrics is not None or remaining_microbatches == 0
                    or step == total or completed_this_invocation == limit):
                atomic_checkpoint(run / "last.pt", state)
            if improved:
                atomic_checkpoint(run / "best.pt", state)
            print(f"step={step}/{total} loss={record['loss']:.4f} train_seconds={elapsed:.2f}", flush=True)
    if step >= total and (run / "best.pt").exists():
        best = load_checkpoint(run / "best.pt")
        best.update(training_completed=True, trained_updates=total)
        atomic_checkpoint(run / "best.pt", best)
    result = {"completed": step >= total, "smoke": args.smoke, "global_step": step, "total_updates": total,
              "processed_images": processed, "training_seconds": training_seconds,
              "training_images_per_second": processed / training_seconds if training_seconds else None,
              "invocation_wall_seconds": time.perf_counter() - invocation_start,
              "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
              "best_validation_score": best_score if best_score >= 0 else None,
              "checkpoint_metric": args.checkpoint_metric, "last_checkpoint": str(run / "last.pt"),
              "best_checkpoint": str(run / "best.pt") if (run / "best.pt").exists() else None,
              "selected_checkpoint": str(run / ("best.pt" if val_data is not None else "last.pt")),
              "test_labels_accessed": False}
    atomic_json(run / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="Train fixed-protocol Faster R-CNN on revised BDD100K train/val")
    cli.add_argument("--dataset-root", type=Path, default=Path("data/bdd100k_revised"))
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--stage", choices=["source", "fine-tune", "control"], default="source")
    cli.add_argument("--domain", choices=DOMAINS, default="source")
    cli.add_argument("--budget", default="full")
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--initialization", choices=["coco", "imagenet", "none"], default="coco")
    cli.add_argument("--source-checkpoint", type=Path)
    cli.add_argument("--resume", type=Path)
    cli.add_argument("--epochs", type=int, default=30)
    cli.add_argument("--floor-steps", type=int, default=2000)
    cli.add_argument("--batch-size", type=int, default=1)
    cli.add_argument("--accumulation", type=int, default=2)
    cli.add_argument("--eval-batch-size", type=int, default=1)
    cli.add_argument("--eval-every-updates", type=int, default=1000)
    cli.add_argument("--save-every-updates", type=int, default=1000)
    cli.add_argument("--lr", type=float, default=0.005)
    cli.add_argument("--momentum", type=float, default=0.9)
    cli.add_argument("--weight-decay", type=float, default=0.0005)
    cli.add_argument("--warmup-updates", type=int, default=100)
    cli.add_argument("--clip-grad-norm", type=float, default=10.0)
    cli.add_argument("--amp", action="store_true")
    cli.add_argument("--no-flip", action="store_true")
    cli.add_argument("--checkpoint-metric", choices=["reliable_AP50_95", "AP50_95"], default="reliable_AP50_95")
    cli.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    cli.add_argument("--workers", type=int, default=0)
    cli.add_argument("--threads", type=int, default=4)
    cli.add_argument("--smoke", action="store_true", help="Mark bounded verification runs; never final research evidence")
    cli.add_argument("--dry-run", action="store_true", help="Validate data/budget/config without creating a model or training")
    cli.add_argument("--limit-train", type=int, default=0)
    cli.add_argument("--limit-val", type=int, default=0)
    cli.add_argument("--stop-after-updates", type=int, help="Stop after this invocation's updates; preserve full configured schedule for resume")
    return cli


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    if min(args.batch_size, args.accumulation, args.eval_batch_size, args.eval_every_updates, args.save_every_updates, args.threads) < 1:
        cli.error("Batch/accumulation/evaluation/threads/support settings must be positive")
    if min(args.seed, args.workers, args.limit_train, args.limit_val, args.warmup_updates) < 0 or args.lr <= 0 or args.clip_grad_norm <= 0 or not 0 <= args.momentum < 1 or args.weight_decay < 0:
        cli.error("Invalid worker/limit/learning rate/gradient settings")
    if args.stop_after_updates is not None and args.stop_after_updates < 1:
        cli.error("--stop-after-updates must be positive")
    train(args)
