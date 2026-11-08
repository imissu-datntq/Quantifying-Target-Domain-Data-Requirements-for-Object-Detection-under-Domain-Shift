"""Local GPU: RQ5 adaptation pilot on lighting val only (2026-11-05 fresh rerun, no Kaggle).

Same runs and arguments as kaggle/bdd_adapt_pilot.py, without the session limit: adapt N=0 and
N=100 (seed 0) for each EMA alpha in bdd_adapt.grid.PILOT, teacher evaluated on the fixed
1,000-image lighting-val subset every 500 updates. No test or weather/compound val image is read.
Re-running is safe: finished runs are skipped, an interrupted one resumes from last.pt. The lock
rule is applied afterwards by scripts/rq5_pilot_lock.py.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bdd_adapt.grid import PILOT  # noqa: E402

DATASET = ROOT / "data" / "bdd100k_revised"
SOURCE = ROOT / "runs" / "local" / "source" / "source_seed0" / "best.pt"
PILOT_DIR = ROOT / "runs" / "local" / "rq5" / "pilot"
LOGS = ROOT / "runs" / "local" / "rq5" / "logs"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def main() -> None:
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    runs = [(budget, alpha) for alpha in reversed(PILOT["alphas"]) for budget in PILOT["budgets"]]  # 0.9996 first
    summary = {"pilot": PILOT, "source": str(SOURCE), "runs": {}}
    for budget, alpha in runs:
        name = f"adapt_lighting_N{budget}_a{alpha}"
        folder = PILOT_DIR / name
        result_path = folder / "result.json"
        if not (result_path.exists() and json.loads(result_path.read_text())["completed"]):
            if folder.exists() and not (folder / "last.pt").exists():
                shutil.rmtree(folder)
            arguments = [sys.executable, "scripts/train_adapt.py", "--dataset-root", str(DATASET), "--output-dir", str(folder),
                         "--stage", "adapt", "--domain", "lighting", "--budget", str(budget), "--seed", "0",
                         "--source-checkpoint", str(SOURCE), "--floor-updates", str(PILOT["floor_updates"]),
                         "--ema-alpha", str(alpha), "--eval-every-updates", str(PILOT["eval_every_updates"]),
                         "--save-every-updates", "500", "--device", "cuda", "--amp", "--workers", "4", "--threads", "4"]
            if (folder / "last.pt").exists():
                arguments += ["--resume", str(folder / "last.pt")]
            log(f"train {name}")
            with (LOGS / f"{name}.log").open("a", encoding="utf-8") as sink:
                code = subprocess.call(arguments, cwd=ROOT, stdout=sink, stderr=subprocess.STDOUT)
            if code:
                raise RuntimeError(f"{name} failed ({code}); see {LOGS / (name + '.log')}")
            (folder / "last.pt").unlink(missing_ok=True)  # pilot keeps curves only
        curve = {}
        for path in sorted(folder.glob("validation_step_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
            metrics = json.loads(path.read_text())
            curve[metrics["step"]] = metrics["reliable_AP50_95"]
        result = json.loads(result_path.read_text())
        summary["runs"][name] = {"curve": curve, "seconds_per_update": result["seconds_per_update"],
                                 "peak_cuda_memory_gb": result["peak_cuda_memory_allocated_bytes"] / 2 ** 30}
        log(f"{name}: {json.dumps(summary['runs'][name])}")
        (PILOT_DIR.parent / "pilot_summary.json").write_text(json.dumps(summary, indent=2))
    log("RQ5 PILOT COMPLETE: run scripts/rq5_pilot_lock.py runs/local/rq5/pilot --output runs/local/rq5/pilot_lock.json")


if __name__ == "__main__":
    main()
