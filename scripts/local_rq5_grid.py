"""Local GPU: run the 46-run RQ5 grid (bdd_adapt.grid) in order (2026-11-05 fresh rerun, no Kaggle).

Same per-run logic as kaggle/bdd_adapt.py without the session limit: skip runs whose final-test
predictions exist, resume an interrupted run exactly from last.pt, predict with the EMA teacher
(last.pt["model"]) on the final-test images (no labels opened), then drop the checkpoint.
Re-running is safe.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bdd_adapt.grid import POLICY, grid, require_locked, train_arguments  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def run(arguments: list[str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as sink:
        return subprocess.call(arguments, cwd=ROOT, stdout=sink, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "bdd100k_revised")
    parser.add_argument("--source", type=Path, default=ROOT / "runs" / "local" / "source" / "source_seed0" / "best.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "local" / "rq5")
    parser.add_argument("--only", nargs="*", help="Run ids to run (default: the whole grid in order)")
    args = parser.parse_args()
    require_locked(POLICY)
    rq5, logs = args.output_dir / "grid", args.output_dir / "logs"
    rq5.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    rows = grid(POLICY)
    if args.only:
        unknown = set(args.only) - {row["run_id"] for row in rows}
        if unknown:
            raise ValueError(f"Unknown run ids: {sorted(unknown)}")
        rows = [row for row in rows if row["run_id"] in set(args.only)]
    python = sys.executable
    summary_path = args.output_dir / "grid_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"runs": {}}
    summary["policy"] = dict(POLICY)
    log(f"RQ5 grid: {len(rows)} runs, policy {POLICY}")
    for index, row in enumerate(rows, 1):
        run_id, folder = row["run_id"], rq5 / row["run_id"]
        predictions, train_dir = folder / "predictions", folder / "train"
        if (predictions / "prediction_manifest.json").exists():
            continue
        shutil.rmtree(predictions, ignore_errors=True)
        log(f"[{index}/{len(rows)}] {run_id}")
        if train_dir.exists() and not (train_dir / "last.pt").exists():
            shutil.rmtree(train_dir)  # stopped before its first checkpoint, or checkpoint already dropped without predictions
        result_path = train_dir / "result.json"
        done = result_path.exists() and json.loads(result_path.read_text()).get("completed") and (train_dir / "last.pt").exists()
        if not done:
            arguments = [python, "scripts/train_adapt.py"] + train_arguments(row, POLICY, args.dataset_root, train_dir, args.source) + ["--device", "cuda"]
            if (train_dir / "last.pt").exists():
                arguments += ["--resume", str(train_dir / "last.pt")]
            code = run(arguments, logs / f"{run_id}_train.log")
            result = json.loads(result_path.read_text()) if result_path.exists() else {}
            if code or not result.get("completed"):
                raise RuntimeError(f"{run_id} failed (exit {code}); see logs/{run_id}_train.log")
        result = json.loads(result_path.read_text())
        code = run([python, "scripts/predict_frcnn.py", "--dataset-root", str(args.dataset_root), "--checkpoint", str(train_dir / "last.pt"),
                    "--output-dir", str(predictions), "--device", "cuda", "--amp", "--batch-size", "4", "--workers", "4"],
                   logs / f"{run_id}_predict.log")
        if code:
            raise RuntimeError(f"{run_id}: prediction failed (exit {code}); see logs/{run_id}_predict.log")
        (train_dir / "last.pt").unlink()
        summary["runs"][run_id] = {"updates": result["global_step"], "seconds_per_update": result["seconds_per_update"]}
        log(f"{run_id}: {json.dumps(summary['runs'][run_id])}")
        summary_path.write_text(json.dumps(summary, indent=2))
    log("RQ5 GRID COMPLETE")


if __name__ == "__main__":
    main()
