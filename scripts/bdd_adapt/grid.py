"""The RQ5 grid (46 runs) under one adaptation policy locked by the lighting-val pilot."""
from __future__ import annotations

from pathlib import Path

from .train import run_updates

# T (floor_updates) and ema_alpha were set by the pre-registered pilot rule (.agent/DECISIONS.md,
# 2026-11-03; locked 2026-11-04 from runs/rq5/pilot_lock.json). None = not locked: launching refuses.
POLICY = {"floor_updates": 2000, "ema_alpha": 0.9996, "lr": 0.001, "warmup_updates": 50, "amp": True,
          "sampling_seeds": (0, 1, 2), "unsupervised_seeds": (0, 1), "replay_seed": 0}
TARGETS = ("compound", "weather", "lighting")
BUDGETS = {"compound": (10, 100, 1000), "weather": (10, 100, 1000), "lighting": (10, 100, 1000, 5000)}
REPLAY_BUDGETS = (10, 100, 1000)
# Pilot: adapt N=0 and N=100 on lighting (seed 0) for each alpha; teacher on lighting val every 500.
PILOT = {"floor_updates": 6000, "alphas": (0.999, 0.9996), "budgets": (0, 100), "eval_every_updates": 500}


def require_locked(policy: dict) -> None:
    missing = [key for key in ("floor_updates", "ema_alpha") if policy.get(key) is None]
    if missing:
        raise ValueError(f"RQ5 policy not locked: {missing}")


def run_id(stage: str, domain: str, budget: int, seed: int) -> str:
    return f"{stage}_{domain}_N{budget}_s{seed}"


def grid(policy: dict) -> list[dict]:
    """Launch order: compound, weather, replay controls, lighting <= 1000, lighting 5000."""
    rows = []
    for domain in TARGETS:
        block = [dict(stage="adapt", domain=domain, budget=0, seed=s, purpose="unsupervised line")
                 for s in policy["unsupervised_seeds"]]
        block += [dict(stage="adapt", domain=domain, budget=n, seed=s, purpose="semi-supervised curve")
                  for n in BUDGETS[domain] for s in policy["sampling_seeds"]]
        if domain == "lighting":
            rows += [r for r in block if r["budget"] <= 1000]
            late = [r for r in block if r["budget"] > 1000]
        else:
            rows += block
        if domain == "weather":
            rows.append(dict(stage="replay", domain="source", budget=0, seed=policy["replay_seed"], purpose="replay control"))
            rows += [dict(stage="replay", domain=d, budget=n, seed=policy["replay_seed"], purpose="replay control")
                     for d in TARGETS for n in REPLAY_BUDGETS]
    rows += late
    for row in rows:
        row["run_id"] = run_id(row["stage"], row["domain"], row["budget"], row["seed"])
    if len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate run ids")
    return rows


def estimate(row: dict, policy: dict, adapt_seconds: float = 0.75, replay_seconds: float = 0.35) -> dict:
    updates = run_updates(policy["floor_updates"], row["budget"])
    return {"updates": updates, "train_hours": updates * (adapt_seconds if row["stage"] == "adapt" else replay_seconds) / 3600}


def train_arguments(row: dict, policy: dict, dataset: Path, output: Path, source_checkpoint: Path,
                    *, workers: int = 4, save_every_updates: int = 500) -> list[str]:
    """train_adapt.py arguments for one row; identical across sessions so resume is exact."""
    require_locked(policy)
    arguments = ["--dataset-root", str(dataset), "--output-dir", str(output), "--stage", row["stage"],
                 "--domain", row["domain"], "--budget", str(row["budget"]), "--seed", str(row["seed"]),
                 "--source-checkpoint", str(source_checkpoint), "--floor-updates", str(policy["floor_updates"]),
                 "--ema-alpha", str(policy["ema_alpha"]), "--lr", str(policy["lr"]),
                 "--warmup-updates", str(policy["warmup_updates"]), "--save-every-updates", str(save_every_updates),
                 "--workers", str(workers), "--threads", "4"]
    return arguments + (["--amp"] if policy["amp"] else [])
