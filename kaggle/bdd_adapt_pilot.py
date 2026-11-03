"""Kaggle GPU: RQ5 adaptation pilot on lighting val only (pre-registered rule, DECISIONS 2026-11-03).

adapt N=0 and N=100 (seed 0) for EMA alpha 0.9996 and 0.999, 6,000 updates each; the EMA teacher
is evaluated on a fixed 1,000-image lighting-val subset at step 0 and every 500 updates. No test
image or weather/compound val image is touched. Resumes across versions: attach this kernel's
previous output and finished runs are skipped, an interrupted one continues from last.pt.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SESSION_HOURS = 11.0
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
from common import CODE, DATASET, WORK, bootstrap, log  # noqa: E402

bootstrap(rehash_sample=2000)
sys.path.insert(0, str(CODE / "scripts"))
from bdd_adapt.grid import PILOT  # noqa: E402

source_dir = find_directory("source_seed0", contains="best.pt")
if source_dir is None:
    raise FileNotFoundError("Attach the completed kizhowl/bdd100k-source-train output")
PILOT_DIR = WORK / "rq5_pilot"
previous = find_directory("rq5_pilot")
if previous is not None and not PILOT_DIR.exists():
    log(f"Restoring previous session from {previous}")
    shutil.copytree(previous, PILOT_DIR)
PILOT_DIR.mkdir(exist_ok=True)
RUNS = [(budget, alpha) for alpha in reversed(PILOT["alphas"]) for budget in PILOT["budgets"]]  # 0.9996 first


def hours_left() -> float:
    return SESSION_HOURS - (time.time() - START) / 3600


summary = {"pilot": PILOT, "runs": {}}
for budget, alpha in RUNS:
    name = f"adapt_lighting_N{budget}_a{alpha}"
    folder = PILOT_DIR / name
    if (folder / "result.json").exists() and json.loads((folder / "result.json").read_text())["completed"]:
        summary["runs"][name] = "already complete"
    else:
        if hours_left() < 0.3:
            summary["runs"][name] = "not started: session time"
            break
        if folder.exists() and not (folder / "last.pt").exists():
            shutil.rmtree(folder)
        arguments = [sys.executable, "scripts/train_adapt.py", "--dataset-root", str(DATASET), "--output-dir", str(folder),
                     "--stage", "adapt", "--domain", "lighting", "--budget", str(budget), "--seed", "0",
                     "--source-checkpoint", str(source_dir / "best.pt"), "--floor-updates", str(PILOT["floor_updates"]),
                     "--ema-alpha", str(alpha), "--eval-every-updates", str(PILOT["eval_every_updates"]),
                     "--save-every-updates", "500", "--device", "cuda", "--amp", "--workers", "4", "--threads", "4"]
        if (folder / "last.pt").exists():
            arguments += ["--resume", str(folder / "last.pt")]
        log("$ " + " ".join(arguments))
        with open(WORK / "logs" / f"{name}.log", "a", encoding="utf-8") as sink:
            process = subprocess.Popen(arguments, cwd=CODE, stdout=sink, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=max(1, (hours_left() - 0.2) * 3600))
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=120)
                summary["runs"][name] = "interrupted by session time; resumes next version"
                break
        if code:
            raise RuntimeError(f"{name} failed ({code}); see logs/{name}.log")
        (folder / "last.pt").unlink()  # pilot keeps curves only
    curve = {}
    for path in sorted(folder.glob("validation_step_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
        metrics = json.loads(path.read_text())
        curve[metrics["step"]] = metrics["reliable_AP50_95"]
    result = json.loads((folder / "result.json").read_text())
    summary["runs"][name] = {"curve": curve, "seconds_per_update": result["seconds_per_update"],
                             "peak_cuda_memory_gb": result["peak_cuda_memory_allocated_bytes"] / 2 ** 30}
    log(f"{name}: {json.dumps(summary['runs'][name])}")
summary["hours_used"] = (time.time() - START) / 3600
(WORK / "rq5_pilot_summary.json").write_text(json.dumps(summary, indent=2))
log("session complete")
