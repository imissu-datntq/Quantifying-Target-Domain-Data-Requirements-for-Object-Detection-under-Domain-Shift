"""Kaggle GPU: work through a queue of campaign runs, resuming across sessions.

Each session restores campaign/ from the previous version's output (attached as a kernel
source), skips runs whose final-test predictions already exist, resumes an interrupted run
exactly from last.pt, and stops training before the session limit. A completed run predicts on
the final-test images of its target and the source domain (no labels opened), then drops its
optimizer checkpoints; full-pool ceilings keep their model weights. QUEUE and SESSION_HOURS
are set in the pushed copy of this script.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

QUEUE: list[str] = []  # campaign run ids for this session, in order
SESSION_HOURS = 11.0  # Kaggle stops GPU sessions at 12 h
PREDICTION_RESERVE_HOURS = {"lighting": 0.6, "weather": 0.3, "compound": 0.3, "all": 1.0}
START = time.time()


def locate(marker: str, skip=("train", "val", "test", "images")) -> Path | None:
    for root, directories, files in os.walk("/kaggle/input"):
        if marker in files:
            return Path(root)
        directories[:] = [d for d in directories if d not in skip]
    return None


sys.path.insert(0, str(locate("expected_sha256.json") / "code" / "kaggle"))
from common import CODE, DATASET, WORK, bootstrap, log, run  # noqa: E402

bootstrap(rehash_sample=2000)  # full rehash passed in kizhowl/bdd100k-bootstrap-check
sys.path.insert(0, str(CODE / "scripts"))
from bdd_pipeline.campaign import POLICY, grid, train_arguments  # noqa: E402

CAMPAIGN = WORK / "campaign"
rows = {row["run_id"]: row for row in grid(POLICY)}
unknown = [run_id for run_id in QUEUE if run_id not in rows]
if unknown:
    raise ValueError(f"Unknown run ids: {unknown}")


def find_directory(name: str, contains: str | None = None, skip=("train", "val", "test", "images", "code")) -> Path | None:
    """First input directory called `name` (optionally holding `contains`), never walking image folders."""
    for root, directories, files in os.walk("/kaggle/input"):
        if Path(root).name == name and (contains is None or contains in files):
            return Path(root)
        directories[:] = [d for d in directories if d not in skip]
    return None


source_dir = find_directory("source_seed0", contains="best.pt")
if source_dir is None:
    raise FileNotFoundError("Attach the completed kizhowl/bdd100k-source-train output")
source = source_dir / "best.pt"
previous = find_directory("campaign")
if previous is not None and not CAMPAIGN.exists():
    log(f"Restoring previous sessions from {previous}")
    shutil.copytree(previous, CAMPAIGN)
CAMPAIGN.mkdir(exist_ok=True)


def hours_left() -> float:
    return SESSION_HOURS - (time.time() - START) / 3600


def timed(arguments: list[str], log_name: str, seconds: float) -> int | None:
    """Run with a wall-clock limit; None if stopped (last.pt is written atomically, so resume is exact)."""
    log(f"$ {' '.join(arguments)}  [limit {seconds / 3600:.2f} h]")
    with open(WORK / "logs" / log_name, "a", encoding="utf-8") as sink:
        process = subprocess.Popen(arguments, cwd=CODE, stdout=sink, stderr=subprocess.STDOUT)
        try:
            return process.wait(timeout=max(seconds, 1))
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=120)
            return None


def predict(checkpoint: Path, output: Path, log_name: str) -> None:
    run([sys.executable, "scripts/predict_frcnn.py", "--dataset-root", str(DATASET), "--checkpoint", str(checkpoint),
         "--output-dir", str(output), "--device", "cuda", "--amp", "--batch-size", "4", "--workers", "4"], log_name)


summary = {"queue": QUEUE, "policy": {k: v for k, v in POLICY.items()}, "runs": {}}
for run_id in QUEUE:
    row, folder = rows[run_id], CAMPAIGN / run_id
    predictions = folder / "predictions"
    if (predictions / "prediction_manifest.json").exists():
        summary["runs"][run_id] = "already complete"
        continue
    reserve = PREDICTION_RESERVE_HOURS[row["domain"]]
    if hours_left() < reserve + 0.25:
        summary["runs"][run_id] = "not started: session time"
        break
    if row["stage"] == "zero-shot":
        predict(source, predictions, f"{run_id}_predict.log")
        summary["runs"][run_id] = "predicted"
        continue
    train_dir = folder / "train"
    if train_dir.exists() and not (train_dir / "last.pt").exists():
        shutil.rmtree(train_dir)  # stopped before its first checkpoint: nothing to resume, start clean
    arguments = [sys.executable, "scripts/train_frcnn.py"] + train_arguments(row, POLICY, DATASET, train_dir, source) + ["--device", "cuda"]
    if (train_dir / "last.pt").exists():
        arguments += ["--resume", str(train_dir / "last.pt")]
    code = timed(arguments, f"{run_id}_train.log", (hours_left() - reserve) * 3600)
    if code is None:
        summary["runs"][run_id] = "interrupted by session time; resumes next session"
        log(f"{run_id}: {summary['runs'][run_id]}")
        break
    result = json.loads((train_dir / "result.json").read_text()) if (train_dir / "result.json").exists() else {}
    if code or not result.get("completed"):
        raise RuntimeError(f"{run_id} failed (exit {code}); see logs/{run_id}_train.log")
    predict(train_dir / "last.pt", predictions, f"{run_id}_predict.log")
    if row["budget"] == "full":
        import torch
        torch.save(torch.load(train_dir / "last.pt", map_location="cpu", weights_only=False)["model"], train_dir / "weights.pt")
    for checkpoint in train_dir.glob("*.pt"):
        if checkpoint.name != "weights.pt":
            checkpoint.unlink()
    summary["runs"][run_id] = {"updates": result["global_step"], "training_seconds": result["training_seconds"],
                               "images_per_second": result["training_images_per_second"]}
    log(f"{run_id}: {json.dumps(summary['runs'][run_id])}")
summary["hours_used"] = (time.time() - START) / 3600
(WORK / "session_summary.json").write_text(json.dumps(summary, indent=2))
log("session complete")
