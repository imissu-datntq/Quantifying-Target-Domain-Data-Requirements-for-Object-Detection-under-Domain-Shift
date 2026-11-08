"""Kaggle GPU: train the daytime-clear source detector, resuming across sessions.

Each session restores runs/source_seed0 from the previous version's output (attached as a
kernel source), resumes exactly from last.pt and stops after SESSION_UPDATES updates so the
session ends inside Kaggle's limit. Settings below must stay identical across sessions.
"""
import json
import os
import shutil
import sys
from pathlib import Path

SETTINGS = {  # fixed after the GPU pilot; resume rejects any change
    "batch_size": 4, "accumulation": 1, "epochs": 12, "floor_steps": 1, "lr": 0.005,
    "warmup_updates": 500, "eval_every_updates": 2451, "save_every_updates": 500, "seed": 0,
}
SESSION_UPDATES = int(os.environ.get("SESSION_UPDATES", "0")) or None  # set in the pushed script copy


def locate(marker: str, skip=("train", "val", "test", "images")) -> Path | None:
    for root, directories, files in os.walk("/kaggle/input"):
        if marker in files:
            return Path(root)
        directories[:] = [d for d in directories if d not in skip]
    return None


sys.path.insert(0, str(locate("expected_sha256.json") / "code" / "kaggle"))
from common import DATASET, WORK, bootstrap, log, run  # noqa: E402

bootstrap(rehash_sample=2000)  # full rehash passed in kizhowl/bdd100k-bootstrap-check
run_dir = WORK / "runs" / "source_seed0"
previous = None
for root, _, files in os.walk("/kaggle/input"):
    if "last.pt" in files and Path(root).name == "source_seed0":
        previous = Path(root)
        break
if previous is not None and not run_dir.exists():
    log(f"Restoring previous session from {previous}")
    shutil.copytree(previous, run_dir)
arguments = [sys.executable, "scripts/train_frcnn.py", "--dataset-root", str(DATASET), "--output-dir", str(run_dir),
             "--stage", "source", "--domain", "source", "--budget", "full", "--initialization", "coco",
             "--device", "cuda", "--amp", "--workers", "4", "--threads", "4",
             "--batch-size", str(SETTINGS["batch_size"]), "--accumulation", str(SETTINGS["accumulation"]),
             "--epochs", str(SETTINGS["epochs"]), "--floor-steps", str(SETTINGS["floor_steps"]),
             "--lr", str(SETTINGS["lr"]), "--warmup-updates", str(SETTINGS["warmup_updates"]),
             "--eval-every-updates", str(SETTINGS["eval_every_updates"]),
             "--save-every-updates", str(SETTINGS["save_every_updates"]), "--seed", str(SETTINGS["seed"])]
if (run_dir / "last.pt").exists():
    arguments += ["--resume", str(run_dir / "last.pt")]
if SESSION_UPDATES:
    arguments += ["--stop-after-updates", str(SESSION_UPDATES)]
run(arguments, "train_source.log")
result = json.loads((run_dir / "result.json").read_text())
(WORK / "session_result.json").write_text(json.dumps({"settings": SETTINGS, "session_updates": SESSION_UPDATES, "result": result}, indent=2))
log(json.dumps(result))
