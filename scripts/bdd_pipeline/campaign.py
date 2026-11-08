"""The supervised campaign grid (proposal 6.4/10) under one locked target fine-tuning policy."""
from __future__ import annotations

from pathlib import Path

from .data import EXPECTED_DOMAINS
from .training import schedule_size

# Locked 2026-11-05 by the 2026-10-30 rule applied to the local RTX 3060 schedule pilot
# (runs/local/schedule_pilot/, .agent/DECISIONS.md). The Kaggle lock of 2026-10-30 had epochs=2.
# None would mean not yet locked: grid generation for launching refuses.
POLICY = {
    "floor_steps": 500, "epochs": 3, "lr": 0.001,
    "batch_size": 2, "accumulation": 1, "warmup_updates": 50, "amp": True,
    "sampling_seeds": (0, 1, 2), "ceiling_seeds": (0, 1), "control_seed": 0,
}
TARGETS = ("compound", "weather", "lighting")  # launch order: cheapest complete curve first
LADDER = (10, 25, 50, 100, 250, 500, 1000)
EXTRA_RUNGS = {"lighting": (2000, 5000, 10000)}
MATCHED_POOL = EXPECTED_DOMAINS["compound"]["train"]  # 1,738: smallest target pool (RQ4)
MATCHED_DOMAINS = ("lighting", "weather")
CONTROL_BUDGETS = (100, 1000)


def require_locked(policy: dict) -> None:
    missing = [key for key in ("floor_steps", "epochs", "lr") if policy.get(key) is None]
    if missing:
        raise ValueError(f"Policy not locked yet: {missing}; lock it in DECISIONS.md and campaign.POLICY first")


def run_id(stage: str, domain: str, budget: str, seed: int) -> str:
    return f"{stage}_{domain}_N{budget}_s{seed}"


def grid(policy: dict) -> list[dict]:
    """Every run of the campaign in launch order; one row per (stage, domain, budget, seed)."""
    rows = [{"run_id": "zero-shot_all_N0_s0", "stage": "zero-shot", "domain": "all", "budget": "0", "seed": 0,
             "initialization": "source-checkpoint", "purpose": "mAP_source and mAP(0) for every shift"}]
    for domain in TARGETS:
        for seed in policy["ceiling_seeds"]:
            rows.append(dict(stage="fine-tune", domain=domain, budget="full", seed=seed, purpose="ceiling mAP_c"))
        for budget in LADDER + EXTRA_RUNGS.get(domain, ()):
            for seed in policy["sampling_seeds"]:
                rows.append(dict(stage="fine-tune", domain=domain, budget=str(budget), seed=seed, purpose="ladder"))
        if domain in MATCHED_DOMAINS:
            rows.append(dict(stage="fine-tune", domain=domain, budget=str(MATCHED_POOL), seed=0, purpose="matched-pool ceiling"))
        if domain == "weather":  # controls after the two small shifts, before the long lighting block
            for control in TARGETS:
                for budget in CONTROL_BUDGETS:
                    rows.append(dict(stage="control", domain=control, budget=str(budget), seed=policy["control_seed"],
                                     purpose="generic-pretrained control"))
    for row in rows[1:]:
        row.setdefault("initialization", "source-checkpoint" if row["stage"] == "fine-tune" else "coco")
        row["run_id"] = run_id(row["stage"], row["domain"], row["budget"], row["seed"])
    if len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate run ids")
    return rows


def estimate(row: dict, policy: dict) -> dict:
    """Optimizer updates and image-passes of one run under the policy's max(S, E epochs) rule."""
    if row["stage"] == "zero-shot":
        return {"images": 0, "updates": 0, "image_passes": 0}
    pool = EXPECTED_DOMAINS[row["domain"]]["train"]
    images = pool if row["budget"] == "full" else int(row["budget"])
    updates, _ = schedule_size(images, policy["batch_size"], policy["accumulation"], policy["epochs"], policy["floor_steps"])
    return {"images": images, "updates": updates, "image_passes": updates * policy["batch_size"] * policy["accumulation"]}


def train_arguments(row: dict, policy: dict, dataset: Path, output: Path, source_checkpoint: Path | None,
                    *, workers: int = 4, save_every_updates: int = 1000) -> list[str]:
    """train_frcnn.py arguments for one grid row; identical across sessions so resume is exact."""
    require_locked(policy)
    if row["stage"] == "zero-shot":
        raise ValueError("The zero-shot row only predicts with the source checkpoint")
    arguments = ["--dataset-root", str(dataset), "--output-dir", str(output), "--stage", row["stage"],
                 "--domain", row["domain"], "--budget", row["budget"], "--seed", str(row["seed"]),
                 "--floor-steps", str(policy["floor_steps"]), "--epochs", str(policy["epochs"]),
                 "--batch-size", str(policy["batch_size"]), "--accumulation", str(policy["accumulation"]),
                 "--lr", str(policy["lr"]), "--warmup-updates", str(policy["warmup_updates"]),
                 "--save-every-updates", str(save_every_updates), "--workers", str(workers), "--threads", "4"]
    if row["stage"] == "fine-tune":
        if source_checkpoint is None:
            raise ValueError("Fine-tuning needs the source checkpoint")
        arguments += ["--source-checkpoint", str(source_checkpoint)]
    else:
        arguments += ["--initialization", row["initialization"]]
    if policy["amp"]:
        arguments.append("--amp")
    return arguments
