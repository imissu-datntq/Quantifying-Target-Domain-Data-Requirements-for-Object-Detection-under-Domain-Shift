"""Resumable ALDI++ (`adapt`) and supervised source-replay (`replay`) runs from the source model.

Per update: a labeled stream of 2 images (2 source at N=0; 1 source + 1 labeled target at N>0)
and, for `adapt`, an unlabeled stream of 2 target-train images read without annotations. The EMA
teacher is the kept model: last.pt["model"] holds it, so predict_frcnn.py / the frozen-plan scorer
work unchanged. Data order, flips and augmentation are pure functions of (seed, stream, index),
so resume is exact. Constant LR after warmup (ALDI); updates = max(T, 2N).
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from prepare_revised_bdd100k import DOMAINS, sha256
from bdd_pipeline.artifacts import (SCHEMA, atomic_checkpoint, atomic_json, environment, load_checkpoint,
                                    random_state, restore_random_state, seed_everything)
from bdd_pipeline.data import (BDDDetection, BDDImages, collate, domain_rows, horizontal_flip, nested_rows,
                               read_manifests, selection_rows)
from bdd_pipeline.evaluation import evaluate_model
from bdd_pipeline.model import MODEL_CONTRACT, build_model
from bdd_pipeline.training import choose_device

from .augment import generator_for, strong
from .distill import THRESHOLD, distillation_losses, ema_update

LABELED_PER_STEP, UNLABELED_PER_STEP = 2, 2
PILOT_VAL_IMAGES = 1000


def run_updates(floor: int, budget: int) -> int:
    """max(T, 2N): every labeled target image is seen at least twice (1 per update), as FT's E=2."""
    return max(floor, 2 * budget)


class Stream(Dataset):
    """Item k of an endless stream: epoch k // n of a seeded permutation, seeded horizontal flip."""

    def __init__(self, dataset: Dataset, name: str, seed: int, labeled: bool):
        self.dataset, self.name, self.seed, self.labeled = dataset, name, seed, labeled
        self._orders: dict[int, list[int]] = {}

    def index(self, k: int) -> int:
        epoch, offset = divmod(k, len(self.dataset))
        if epoch not in self._orders:
            order = list(range(len(self.dataset)))
            random.Random(f"{self.seed}/{self.name}/epoch/{epoch}").shuffle(order)
            self._orders = {epoch: order}
        return self._orders[epoch][offset]

    def __len__(self) -> int:
        return 2 ** 62

    def __getitem__(self, k: int):
        image, target = self.dataset[self.index(k)]
        if self.labeled:
            target = {"boxes": target["boxes"], "labels": target["labels"]}
            if random.Random(f"{self.seed}/{self.name}/flip/{k}").random() < 0.5:
                image, target["boxes"] = horizontal_flip(image, target["boxes"])
            return image, target, k
        if random.Random(f"{self.seed}/{self.name}/flip/{k}").random() < 0.5:
            image = image.flip(-1)
        return image, None, k


def transpose(batch: list) -> tuple:
    return tuple(zip(*batch))


def loader(stream: Stream, per_step: int, start: int, stop: int, workers: int, pin: bool):
    batches = [list(range(step * per_step, (step + 1) * per_step)) for step in range(start, stop)]
    # An own generator: creating the iterator must not draw from the global RNG, or a resumed run's
    # (restored) RNG would be shifted relative to the uninterrupted run.
    return iter(DataLoader(stream, batch_sampler=batches, num_workers=workers, pin_memory=pin, collate_fn=transpose,
                           generator=torch.Generator().manual_seed(start)))


def labeled_plan(budget: int) -> list[tuple[str, int]]:
    return [("source", LABELED_PER_STEP)] if budget == 0 else [("source", 1), ("target", 1)]


def train(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("AMP in this pipeline requires CUDA")
    budget = int(args.budget)
    if budget < 0:
        raise ValueError("Budget must be >= 0")
    if args.stage == "adapt" and args.domain == "source":
        raise ValueError("Adaptation needs a target domain")
    if args.stage == "replay" and args.domain == "source" and budget:
        raise ValueError("Source-domain replay is the N=0 control only")
    if args.eval_every_updates and args.domain != "lighting":
        raise ValueError("Only lighting val may be used for decisions (pilot)")
    if (args.limit_pool or args.source_checkpoint is None) and not args.smoke:
        raise ValueError("Pool limits / random initialization only with --smoke")
    if not 0 < args.ema_alpha < 1 or args.floor_updates < 1:
        raise ValueError("Invalid EMA alpha / update floor")
    root = args.dataset_root.resolve()
    metadata, manifests = read_manifests(root)
    source_rows = domain_rows(manifests, "train", "source")
    pool = domain_rows(manifests, "train", args.domain) if args.domain != "source" else []
    labeled_target = nested_rows(pool, str(budget), args.seed) if budget else []
    unlabeled = pool if args.stage == "adapt" else []
    validation = []
    if args.eval_every_updates:
        validation = sorted(selection_rows(manifests, "lighting"), key=lambda row: row["name"])
        random.Random(0).shuffle(validation)
        validation = validation[:args.limit_val or PILOT_VAL_IMAGES]
    if args.limit_pool:
        source_rows, unlabeled = source_rows[:args.limit_pool], unlabeled[:args.limit_pool]
    total = run_updates(args.floor_updates, budget)
    source = load_checkpoint(args.source_checkpoint) if args.source_checkpoint else None
    if source and (source["config"]["stage"] != "source" or source["config"]["manifest_sha256"] != metadata["manifest_sha256"]):
        raise ValueError("Source checkpoint stage/dataset mismatch")
    if source and not args.smoke and (source["config"]["smoke"] or not source.get("training_completed", False)):
        raise ValueError("Research runs require the completed non-smoke source checkpoint")
    config = {"stage": args.stage, "method": "ALDI++ (soft distillation, MIC, strong aug, EMA teacher)" if args.stage == "adapt"
              else "supervised source replay, same augmentation and EMA teacher, no unlabeled stream",
              "domain": args.domain, "budget": str(budget), "seed": args.seed,
              "implementation_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
              "runtime_packages": environment()["packages"], "manifest_sha256": metadata["manifest_sha256"],
              "initialization": "source-checkpoint" if source else "none",
              "source_checkpoint_sha256": sha256(args.source_checkpoint) if source else None,
              "source_images": len(source_rows), "labeled_target_images": len(labeled_target),
              "unlabeled_target_images": len(unlabeled), "target_unlabeled_labels_accessed": False,
              "labeled_stream": labeled_plan(budget), "unlabeled_per_update": UNLABELED_PER_STEP if unlabeled else 0,
              "floor_updates": args.floor_updates, "total_updates": total, "epochs": None, "floor_steps": args.floor_updates,
              "effective_batch_size": LABELED_PER_STEP + (UNLABELED_PER_STEP if unlabeled else 0),
              "lr": args.lr, "lr_schedule": "linear warmup then constant", "warmup_updates": args.warmup_updates,
              "momentum": args.momentum, "weight_decay": args.weight_decay, "clip_grad_norm": args.clip_grad_norm,
              "ema_alpha": args.ema_alpha, "pseudo_label_threshold": THRESHOLD,
              "augmentation": {"labeled": "flip + strong photometric + random erasing",
                               "unlabeled_teacher": "flip", "unlabeled_student": "same flip + strong photometric + MIC 0.5/32px"},
              "kept_model": "final EMA teacher", "amp": args.amp, "device_type": device.type,
              "val_images": len(validation), "eval_every_updates": args.eval_every_updates,
              "save_every_updates": args.save_every_updates, "smoke": args.smoke, "limit_pool": args.limit_pool,
              "workers": args.workers, "threads": args.threads}
    if args.dry_run:
        print(json.dumps({"validated_configuration": config, "test_annotations_accessed": False}, indent=2))
        return config
    checkpoint = load_checkpoint(args.resume) if args.resume else None
    if checkpoint and checkpoint["config"] != config:
        raise ValueError(f"Resume configuration differs: {[k for k in config if checkpoint['config'].get(k) != config[k]]}")
    run = args.output_dir.resolve()
    if args.resume and args.resume.resolve() != run / "last.pt":
        raise ValueError("Exact resume must use last.pt in the original run directory")
    if run.exists() and any(run.iterdir()) and checkpoint is None:
        raise FileExistsError(f"Use a new output directory or --resume: {run}")
    seed_everything(args.seed)
    student = build_model("none").to(device)
    initial = checkpoint["student"] if checkpoint else source["model"] if source else None
    if initial is not None:
        student.load_state_dict(initial)
    teacher = copy.deepcopy(student)
    if checkpoint:
        teacher.load_state_dict(checkpoint["model"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.SGD([p for p in student.parameters() if p.requires_grad], lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    streams = {"source": Stream(BDDDetection(root, "train", source_rows), "source", args.seed, True)}
    if labeled_target:
        streams["target"] = Stream(BDDDetection(root, "train", labeled_target), "target", args.seed, True)
    if unlabeled:
        streams["unlabeled"] = Stream(BDDImages(root, "train", unlabeled), "unlabeled", args.seed, False)
    val_data = BDDDetection(root, "val", validation) if validation else None
    run.mkdir(parents=True, exist_ok=True)
    if checkpoint is None:
        atomic_json(run / "config.json", config)
        atomic_json(run / "environment.json", environment())
        atomic_json(run / "model_contract.json", MODEL_CONTRACT)
        atomic_json(run / "selected_images.json", {"labeled_target": [r["name"] for r in labeled_target],
                                                   "unlabeled_target": [r["name"] for r in unlabeled],
                                                   "val": [r["name"] for r in validation]})
    step = checkpoint["global_step"] if checkpoint else 0
    processed = checkpoint["processed_images"] if checkpoint else 0
    training_seconds = checkpoint["training_seconds"] if checkpoint else 0.0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_random_state(checkpoint["random_state"])
    log_path = run / "training.jsonl"
    if checkpoint and log_path.exists():
        lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if json.loads(line)["step"] <= step]
        log_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def validate(at: int) -> dict | None:
        if val_data is None or (at % args.eval_every_updates and at != total) or (run / f"validation_step_{at}.json").exists():
            return None
        metrics, _ = evaluate_model(teacher, val_data, device, batch_size=4, workers=args.workers, amp=args.amp)
        metrics.update(split="val", domain="lighting", step=at, model="ema_teacher", smoke=args.smoke)
        atomic_json(run / f"validation_step_{at}.json", metrics)
        print(f"val step={at} reliable_AP50_95={metrics['reliable_AP50_95']}", flush=True)
        return metrics

    validate(step)
    limit = min(total, step + (args.stop_after_updates or total))
    pin = device.type == "cuda"
    plan = labeled_plan(budget)
    loaders = {name: loader(streams[name], count, step, limit, args.workers, pin) for name, count in plan}
    if unlabeled:
        loaders["unlabeled"] = loader(streams["unlabeled"], UNLABELED_PER_STEP, step, limit, args.workers, pin)
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if args.amp else nullcontext
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    invocation_start = time.perf_counter()
    while step < limit:
        if device.type == "cuda":
            torch.cuda.synchronize()
        tick = time.perf_counter()
        student.train()
        lr = args.lr * (0.1 + 0.9 * (step + 1) / args.warmup_updates if step < args.warmup_updates else 1.0)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        images, targets = [], []
        for name, _ in plan:
            batch_images, batch_targets, indices = next(loaders[name])
            for image, target, k in zip(batch_images, batch_targets, indices):
                image = strong(image.to(device, non_blocking=True), generator_for(args.seed, name, k), erase=True, mic=False)
                images.append(image)
                targets.append({key: value.to(device) for key, value in target.items()})
        with autocast():
            supervised = student(images, targets)
            loss = sum(supervised.values())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite supervised loss at step {step}")
        scaler.scale(loss).backward()
        values = {key: float(value.detach()) for key, value in supervised.items()}
        diagnostics = {}
        if unlabeled:
            batch_images, _, indices = next(loaders["unlabeled"])
            weak = [image.to(device, non_blocking=True) for image in batch_images]
            strong_views = [strong(image, generator_for(args.seed, "unlabeled", k), erase=False, mic=True)
                            for image, k in zip(weak, indices)]
            with autocast():
                distilled, diagnostics = distillation_losses(student, teacher, weak, strong_views)
                distill_loss = sum(distilled.values())
            if not torch.isfinite(distill_loss):
                raise FloatingPointError(f"Nonfinite distillation loss at step {step}")
            scaler.scale(distill_loss).backward()
            values.update({key: float(value.detach()) for key, value in distilled.items()})
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad_norm, error_if_nonfinite=not args.amp)
        skipped = not bool(torch.isfinite(norm))
        scaler.step(optimizer)
        scaler.update()
        ema_update(teacher, student, args.ema_alpha)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - tick
        training_seconds += elapsed
        processed += len(images) + (UNLABELED_PER_STEP if unlabeled else 0)
        step += 1
        metrics = validate(step)
        record = {"step": step, "loss": sum(values.values()), "losses": values, "lr": lr,
                  "grad_norm": None if skipped else float(norm), "amp_step_skipped": skipped, "seconds": elapsed,
                  "diagnostics": diagnostics, "validation": None if metrics is None else metrics["reliable_AP50_95"]}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        if step % args.save_every_updates == 0 or step == limit:
            atomic_checkpoint(run / "last.pt", {
                "schema": SCHEMA, "model_contract": MODEL_CONTRACT, "config": config,
                "model": teacher.state_dict(), "student": student.state_dict(),
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "global_step": step, "random_state": random_state(), "processed_images": processed,
                "training_seconds": training_seconds, "training_completed": step >= total, "trained_updates": step})
        if step % 25 == 0 or step == limit:
            print(f"step={step}/{total} loss={record['loss']:.4f} seconds={elapsed:.2f} {diagnostics}", flush=True)
    result = {"completed": step >= total, "smoke": args.smoke, "global_step": step, "total_updates": total,
              "processed_images": processed, "training_seconds": training_seconds,
              "training_images_per_second": processed / training_seconds if training_seconds else None,
              "seconds_per_update": training_seconds / step if step else None,
              "invocation_wall_seconds": time.perf_counter() - invocation_start,
              "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
              "last_checkpoint": str(run / "last.pt"), "kept_model": "last.pt['model'] (EMA teacher)",
              "test_labels_accessed": False, "target_unlabeled_labels_accessed": False}
    atomic_json(run / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--dataset-root", type=Path, default=Path("data/bdd100k_revised"))
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--stage", choices=["adapt", "replay"], required=True)
    cli.add_argument("--domain", choices=DOMAINS, required=True)
    cli.add_argument("--budget", default="0", help="Labeled target images (0 = unsupervised)")
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--source-checkpoint", type=Path)
    cli.add_argument("--resume", type=Path)
    cli.add_argument("--floor-updates", type=int, required=True, help="T; the run lasts max(T, 2N) updates")
    cli.add_argument("--ema-alpha", type=float, default=0.9996)
    cli.add_argument("--lr", type=float, default=0.001)
    cli.add_argument("--warmup-updates", type=int, default=50)
    cli.add_argument("--momentum", type=float, default=0.9)
    cli.add_argument("--weight-decay", type=float, default=0.0005)
    cli.add_argument("--clip-grad-norm", type=float, default=10.0)
    cli.add_argument("--amp", action="store_true")
    cli.add_argument("--eval-every-updates", type=int, default=0, help="Pilot only: EMA teacher on the lighting-val subset")
    cli.add_argument("--limit-val", type=int, default=0, help="Smoke only")
    cli.add_argument("--save-every-updates", type=int, default=500)
    cli.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    cli.add_argument("--workers", type=int, default=0)
    cli.add_argument("--threads", type=int, default=4)
    cli.add_argument("--smoke", action="store_true")
    cli.add_argument("--limit-pool", type=int, default=0, help="Smoke only: cap source/unlabeled pools")
    cli.add_argument("--dry-run", action="store_true")
    cli.add_argument("--stop-after-updates", type=int)
    return cli


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    if min(args.save_every_updates, args.threads, args.warmup_updates) < 1 or min(args.workers, args.limit_pool, args.eval_every_updates) < 0:
        cli.error("Invalid save/thread/warmup/worker/limit settings")
    if args.limit_val and not args.smoke:
        cli.error("--limit-val is smoke only")
    train(args)
