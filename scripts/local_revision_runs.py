"""Local GPU: the 42 revision runs asked for by the lecturer review of 2026-11-21.

Rules: .agent/DECISIONS.md 2026-11-21 and its addendum, written before any of these runs was
scored. Four blocks, in launch order (short blocks first, the long lighting ceilings near the end):

- replay_seeds (12 runs): the `replay` control at N = 10 and 100, sampling seeds 1 and 2, under
  the locked RQ5 policy. Seed 0 already exists in runs/local/rq5/grid/.
- matched_passes (18 runs): `adapt` and `replay` at N = 1,000, seeds 0/1/2, with 3,000 updates
  instead of 2,000, so that labeled target image-passes equal fine-tuning's 3,000 (E = 3).
- ceiling_e12 (10 runs): full-pool fine-tunes at E = 12 for every shift, seeds 0/1, plus
  matched-pool (1,738) fine-tunes at E = 12 for lighting and weather, seeds 0/1.
- matched_pool_e3 (2 runs): the matched-pool fine-tunes for lighting and weather at the locked
  E = 3 with seed 1, so that the E12 - E3 comparison at 1,738 is paired on two seeds. Seed 0
  already exists in runs/local/campaign/campaign/.

The frozen packages are used as they are: only the policy passed to their `train_arguments`
differs (epochs, floor_updates), so every run records the same implementation hashes as the
original grids. Per-run logic is that of local_campaign.py and local_rq5_grid.py: skip runs whose
final-test predictions exist, resume an interrupted run exactly from last.pt, predict on the
final-test images of the target and source domain (no labels opened), then drop the optimizer
checkpoint; the ceilings keep their model weights. Re-running is safe once no driver, trainer or
predictor process is alive (the driver refuses to start otherwise). --dry-run prints the plan.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bdd_adapt import grid as adapt_grid  # noqa: E402
from bdd_adapt.train import run_updates  # noqa: E402
from bdd_pipeline import campaign  # noqa: E402

TARGETS = ("compound", "weather", "lighting")  # cheapest first, as in the original grids
REPLAY_SEED_BUDGETS = (10, 100)
REPLAY_NEW_SEEDS = (1, 2)
MATCHED_BUDGET = 1000
MATCHED_SEEDS = (0, 1, 2)
CEILING_SEEDS = (0, 1)
MATCHED_POOL_E3_SEED = 1  # seed 0 is in the original campaign
# Fine-tuning at N = 1,000 runs E epochs at batch 2: E * N labeled target image-passes. The
# adaptation arms see one labeled target image per update, so they need E * N updates to match.
MATCHED_FLOOR_UPDATES = campaign.POLICY["epochs"] * MATCHED_BUDGET
CEILING_EPOCHS = 12  # the source model's epoch count
ADAPT_POLICY = dict(adapt_grid.POLICY)
MATCHED_POLICY = dict(adapt_grid.POLICY, floor_updates=MATCHED_FLOOR_UPDATES)
CEILING_POLICY = dict(campaign.POLICY, epochs=CEILING_EPOCHS)
# Measured on this machine (RTX 3060): FT 8.24 img/s at batch 2 on the lighting full pool, adapt
# ~0.6 s/update, replay ~0.25 s/update. Used only for the --dry-run time column; prediction adds
# about 3.5 min (compound, weather) to 7 min (lighting) per run.
SECONDS_PER_UPDATE = {"fine-tune": 2 / 8.24, "adapt": 0.6, "replay": 0.25}
PROCESS_PATTERN = r"scripts/(local_revision_runs|train_adapt|train_frcnn|predict_frcnn)\.py"


def rows() -> list[dict]:
    """Every revision run in launch order; `block` is the output subfolder, `policy` what is passed on."""
    out = []
    for domain in TARGETS:
        for budget in REPLAY_SEED_BUDGETS:
            for seed in REPLAY_NEW_SEEDS:
                out.append(dict(block="replay_seeds", kind="adapt", stage="replay", domain=domain, budget=budget, seed=seed,
                                policy=ADAPT_POLICY, purpose="replay control, extra sampling seed"))
    for domain in TARGETS:
        for stage in ("adapt", "replay"):
            for seed in MATCHED_SEEDS:
                out.append(dict(block="matched_passes", kind="adapt", stage=stage, domain=domain, budget=MATCHED_BUDGET, seed=seed,
                                policy=MATCHED_POLICY, purpose="labeled passes matched to fine-tuning"))
    ceilings = [(domain, "full") for domain in ("compound", "weather")]
    ceilings += [(domain, str(campaign.MATCHED_POOL)) for domain in campaign.MATCHED_DOMAINS]
    ceilings.append(("lighting", "full"))  # about 7 h per seed: last
    for domain, budget in ceilings:
        for seed in CEILING_SEEDS:
            out.append(dict(block="ceiling_e12", kind="fine-tune", stage="fine-tune", domain=domain, budget=budget, seed=seed,
                            initialization="source-checkpoint", policy=CEILING_POLICY,
                            purpose="ceiling mAP_c at E=12" if budget == "full" else "matched-pool ceiling at E=12"))
    for domain in campaign.MATCHED_DOMAINS:
        out.append(dict(block="matched_pool_e3", kind="fine-tune", stage="fine-tune", domain=domain, budget=str(campaign.MATCHED_POOL),
                        seed=MATCHED_POOL_E3_SEED, initialization="source-checkpoint", policy=dict(campaign.POLICY),
                        purpose="matched-pool ceiling at E=3, second seed"))
    for row in out:
        row["run_id"] = f"{row['stage']}_{row['domain']}_N{row['budget']}_s{row['seed']}"
    if len({(row["block"], row["run_id"]) for row in out}) != len(out):
        raise ValueError("Duplicate revision runs")
    return out


def updates_of(row: dict) -> int:
    if row["kind"] == "fine-tune":
        return campaign.estimate(row, row["policy"])["updates"]
    return run_updates(row["policy"]["floor_updates"], row["budget"])


def train_command(row: dict, python: str, dataset: Path, train_dir: Path, source: Path) -> list[str]:
    if row["kind"] == "fine-tune":
        return [python, "scripts/train_frcnn.py"] + campaign.train_arguments(row, row["policy"], dataset, train_dir, source) + ["--device", "cuda"]
    return [python, "scripts/train_adapt.py"] + adapt_grid.train_arguments(row, row["policy"], dataset, train_dir, source) + ["--device", "cuda"]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def run(arguments: list[str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as sink:
        return subprocess.call(arguments, cwd=ROOT, stdout=sink, stderr=subprocess.STDOUT)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def other_processes() -> list[str]:
    """Other live driver, trainer or predictor processes: a second writer would corrupt the run in progress."""
    listing = subprocess.run(["pgrep", "-af", PROCESS_PATTERN], capture_output=True, text=True).stdout.splitlines()
    # Only Python processes count: a launching shell also carries the script name on its command line.
    return [line for line in listing if line.split()[0] != str(os.getpid()) and Path(line.split()[1]).name.startswith("python")]


def prepare_train_dir(row: dict, train_dir: Path, name: str) -> None:
    """Start clean when a run stopped before its first checkpoint; never delete a finished ceiling's kept weights."""
    if not train_dir.exists() or (train_dir / "last.pt").exists():
        return
    if row["kind"] == "fine-tune" and (train_dir / "result.json").exists():
        raise RuntimeError(f"{name}: training finished and its checkpoint was dropped, but the predictions are missing. "
                           f"Remove {train_dir} by hand to retrain it.")
    shutil.rmtree(train_dir)  # adapt/replay: also when the checkpoint was dropped without predictions


def finalize(row: dict, train_dir: Path, name: str, summary: dict, summary_path: Path) -> None:
    """Idempotent tail of a run whose predictions exist: keep ceiling weights, drop checkpoints, record the run."""
    last = train_dir / "last.pt"
    if row["kind"] == "fine-tune" and last.exists():  # ceilings keep their weights, as in local_campaign.py
        import torch
        temporary = train_dir / "weights.pt.tmp"
        torch.save(torch.load(last, map_location="cpu", weights_only=False)["model"], temporary)
        os.replace(temporary, train_dir / "weights.pt")
    for checkpoint in train_dir.glob("*.pt"):
        if checkpoint.name != "weights.pt":
            checkpoint.unlink()
    if name not in summary["runs"] and (train_dir / "result.json").exists():
        result = json.loads((train_dir / "result.json").read_text())
        summary["runs"][name] = {"updates": result["global_step"],
                                 **{key: result[key] for key in ("training_seconds", "training_images_per_second", "seconds_per_update")
                                    if key in result}}
        write_json(summary_path, summary)
        log(f"{name}: {json.dumps(summary['runs'][name])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "bdd100k_revised")
    parser.add_argument("--source", type=Path, default=ROOT / "runs" / "local" / "source" / "source_seed0" / "best.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "local" / "revision")
    parser.add_argument("--blocks", nargs="*", choices=("replay_seeds", "matched_passes", "ceiling_e12", "matched_pool_e3"),
                        help="Blocks to run (default: all, in launch order)")
    parser.add_argument("--dry-run", action="store_true", help="Print every run, its update count and command; run nothing")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    campaign.require_locked(campaign.POLICY)
    adapt_grid.require_locked(adapt_grid.POLICY)
    plan = [row for row in rows() if not args.blocks or row["block"] in args.blocks]
    python = sys.executable
    grid_dir, logs = args.output_dir / "grid", args.output_dir / "logs"
    if args.dry_run:
        total = 0.0
        for index, row in enumerate(plan, 1):
            updates = updates_of(row)
            hours = updates * SECONDS_PER_UPDATE[row["stage"]] / 3600
            total += hours
            train_dir = grid_dir / row["block"] / row["run_id"] / "train"
            print(f"[{index}/{len(plan)}] {row['block']}/{row['run_id']}: {updates} updates, ~{hours:.2f} h training ({row['purpose']})")
            print("    " + " ".join(train_command(row, python, args.dataset_root, train_dir, args.source)))
        print(f"{len(plan)} runs, ~{total:.1f} h of training plus final-test prediction")
        return
    logs.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir / "driver.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"Another local_revision_runs.py holds {args.output_dir / 'driver.lock'}") from None
    alive = other_processes()
    if alive:
        raise SystemExit("Refusing to start while these processes are alive (stop the whole process group first):\n" + "\n".join(alive))
    summary_path = args.output_dir / "session_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"runs": {}}
    summary["policies"] = {"replay_seeds": ADAPT_POLICY, "matched_passes": MATCHED_POLICY, "ceiling_e12": CEILING_POLICY,
                           "matched_pool_e3": dict(campaign.POLICY)}
    log(f"revision: {len(plan)} runs")
    for index, row in enumerate(plan, 1):
        name = f"{row['block']}/{row['run_id']}"
        log_stem = f"{row['block']}_{row['run_id']}"
        folder = grid_dir / row["block"] / row["run_id"]
        predictions, train_dir = folder / "predictions", folder / "train"
        if (predictions / "prediction_manifest.json").exists():
            finalize(row, train_dir, name, summary, summary_path)  # repairs a tail cut short; a no-op for a finished run
            continue
        shutil.rmtree(predictions, ignore_errors=True)  # partial prediction folder: redo it
        log(f"[{index}/{len(plan)}] {name}")
        result_path = train_dir / "result.json"
        prepare_train_dir(row, train_dir, name)
        done = result_path.exists() and json.loads(result_path.read_text()).get("completed") and (train_dir / "last.pt").exists()
        if not done:
            folder.mkdir(parents=True, exist_ok=True)
            arguments = train_command(row, python, args.dataset_root, train_dir, args.source)
            if (train_dir / "last.pt").exists():
                arguments += ["--resume", str(train_dir / "last.pt")]
            code = run(arguments, logs / f"{log_stem}_train.log")
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            if code or not result.get("completed"):
                raise RuntimeError(f"{name} failed (exit {code}); see logs/{log_stem}_train.log")
        result = json.loads(result_path.read_text())
        expected = updates_of(row)
        if result["global_step"] != expected:
            raise RuntimeError(f"{name}: {result['global_step']} updates, the pre-registered count is {expected}")
        code = run([python, "scripts/predict_frcnn.py", "--dataset-root", str(args.dataset_root), "--checkpoint", str(train_dir / "last.pt"),
                    "--output-dir", str(predictions), "--device", "cuda", "--amp", "--batch-size", "4", "--workers", "4"],
                   logs / f"{log_stem}_predict.log")
        if code:
            raise RuntimeError(f"{name}: prediction failed (exit {code}); see logs/{log_stem}_predict.log")
        finalize(row, train_dir, name, summary, summary_path)
    log("REVISION RUNS COMPLETE")


if __name__ == "__main__":
    main()
