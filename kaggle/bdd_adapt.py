"""Kaggle GPU: work through a queue of RQ5 runs (bdd_adapt.grid), resuming across sessions.

Same contract as bdd_campaign.py: restore rq5/ from the previous version's output, skip runs whose
final-test predictions exist, resume an interrupted run exactly from last.pt, stop before the
session limit. A completed run predicts with its EMA teacher (last.pt["model"]) on the final-test
images of its target and the source domain (all four for the source-domain replay), opening no
labels, then drops its checkpoint. QUEUE and SESSION_HOURS are set in the pushed copy.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

QUEUE: list[str] = []  # run ids for this session, in order
SESSION_HOURS = 11.0
PREDICTION_RESERVE_HOURS = {"lighting": 0.6, "weather": 0.3, "compound": 0.3, "source": 1.0}
START = time.time()


def locate(marker: str, skip=("train", "val", "test", "images")) -> Path | None:
    for root, directories, files in os.walk("/kaggle/input"):
        if marker in files:
            return Path(root)
        directories[:] = [d for d in directories if d not in skip]
    return None


def find_directory(name: str, contains: str | None = None, skip=("train", "val", "test", "images", "code")) -> Path | None:
    for root, directories, files in os.walk("/kaggle/input"):
        if Path(root).name == name and (contains is None or contains in files):
            return Path(root)
        directories[:] = [d for d in directories if d not in skip]
    return None


sys.path.insert(0, str(locate("expected_sha256.json") / "code" / "kaggle"))
from common import CODE, DATASET, WORK, bootstrap, log, run  # noqa: E402

bootstrap(rehash_sample=2000)
sys.path.insert(0, str(CODE / "scripts"))
from bdd_adapt.grid import POLICY, grid, train_arguments  # noqa: E402

rows = {row["run_id"]: row for row in grid(POLICY)}
unknown = [run_id for run_id in QUEUE if run_id not in rows]
if unknown:
    raise ValueError(f"Unknown run ids: {unknown}")
SOURCE_SHA256 = "2c62adf9f8fcc185c93e66c711614b65866cf875735a8f9d59f334cd9070e349"  # source_seed0/best.pt
# kizhowl attaches the bdd100k-source-train output (source_seed0/best.pt); buihuynhgiahuy attaches its
# private copy buihuynhgiahuy/bdd100k-source-model. Either way the same file, checked by hash.
source_dir = find_directory("source_seed0", contains="best.pt") or locate("best.pt")
if source_dir is None:
    raise FileNotFoundError("Attach the source model (kizhowl/bdd100k-source-train or buihuynhgiahuy/bdd100k-source-model)")
source = source_dir / "best.pt"
from prepare_revised_bdd100k import sha256  # noqa: E402
if sha256(source) != SOURCE_SHA256:
    raise ValueError(f"Unexpected source checkpoint: {source}")
RQ5 = WORK / "rq5"
previous = find_directory("rq5")
if previous is not None and not RQ5.exists():
    log(f"Restoring previous sessions from {previous}")
    shutil.copytree(previous, RQ5)
RQ5.mkdir(exist_ok=True)


def hours_left() -> float:
    return SESSION_HOURS - (time.time() - START) / 3600


summary = {"queue": QUEUE, "policy": POLICY, "runs": {}}
for run_id in QUEUE:
    row, folder = rows[run_id], RQ5 / run_id
    predictions, train_dir = folder / "predictions", folder / "train"
    if (predictions / "prediction_manifest.json").exists():
        summary["runs"][run_id] = "already complete"
        continue
    reserve = PREDICTION_RESERVE_HOURS[row["domain"]]
    if hours_left() < reserve + 0.25:
        summary["runs"][run_id] = "not started: session time"
        break
    if train_dir.exists() and not (train_dir / "last.pt").exists():
        shutil.rmtree(train_dir)
    arguments = [sys.executable, "scripts/train_adapt.py"] + train_arguments(row, POLICY, DATASET, train_dir, source) + ["--device", "cuda"]
    if (train_dir / "last.pt").exists():
        arguments += ["--resume", str(train_dir / "last.pt")]
    log(f"$ {' '.join(arguments)}")
    with open(WORK / "logs" / f"{run_id}_train.log", "a", encoding="utf-8") as sink:
        process = subprocess.Popen(arguments, cwd=CODE, stdout=sink, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=max(1, (hours_left() - reserve) * 3600))
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=120)
            summary["runs"][run_id] = "interrupted by session time; resumes next session"
            log(f"{run_id}: {summary['runs'][run_id]}")
            break
    result = json.loads((train_dir / "result.json").read_text()) if (train_dir / "result.json").exists() else {}
    if code or not result.get("completed"):
        raise RuntimeError(f"{run_id} failed (exit {code}); see logs/{run_id}_train.log")
    run([sys.executable, "scripts/predict_frcnn.py", "--dataset-root", str(DATASET), "--checkpoint", str(train_dir / "last.pt"),
         "--output-dir", str(predictions), "--device", "cuda", "--amp", "--batch-size", "4", "--workers", "4"],
        f"{run_id}_predict.log")
    (train_dir / "last.pt").unlink()
    summary["runs"][run_id] = {"updates": result["global_step"], "seconds_per_update": result["seconds_per_update"]}
    log(f"{run_id}: {json.dumps(summary['runs'][run_id])}")
summary["hours_used"] = (time.time() - START) / 3600
(WORK / "session_summary.json").write_text(json.dumps(summary, indent=2))
log("session complete")
