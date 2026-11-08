"""Runtime-test the exact optimizer-update stop used by both YOLO DA arms."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import run_sf_adaptation_benchmark
import run_ssda_adaptation_benchmark


ROOT = Path(__file__).resolve().parent
STEPS = 6
REPORT = ROOT / "runs/yolo-update-limit-smoke-v1.json"


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_command(candidate: str, output: Path, root: Path = ROOT) -> list[str]:
    if candidate == "ssda-yolo":
        command = run_ssda_adaptation_benchmark.build_command(
            python=root / ".venvs/ssda-yolo/bin/python",
            upstream=root / "external/SSDA-YOLO",
            output=output,
            source_checkpoint=root / "weights/yolov5l-v5.0.pt",
            data=root / "runs/ssda-adapt-inputs-001/data.yaml",
            hyp=root / "runs/ssda-adapt-inputs-001/hyp.yaml",
            epochs=36,
            max_optimizer_updates=STEPS,
        )
    elif candidate == "sf-yolo":
        assets = root / "runs/targetaugment-lighting-smoke-002/models"
        command = run_sf_adaptation_benchmark.build_command(
            python=root / ".venvs/sf-yolo/bin/python",
            upstream=root / "external/sf-yolo",
            output=output,
            source_checkpoint=root / "runs/sf-pretrained-source-smoke-001/weights/last.pt",
            data=root / "runs/sf-adapt-smoke-001/data.yaml",
            hyp=root / "external/sf-yolo/data/hyps/hyp.scratch-low.yaml",
            decoder=assets / "decoder_iter_120.pth",
            encoder=root / "weights/vgg16_ori.pth",
            fc1=assets / "fc1_iter_120.pth",
            fc2=assets / "fc2_iter_120.pth",
            epochs=60,
            max_optimizer_updates=STEPS,
        )
    else:
        raise ValueError(candidate)
    return command


def completed_evidence(output: Path) -> dict[str, Any] | None:
    path = output / "update_limit_status.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return value if (value.get("status") == "completed"
                     and value.get("optimizer_updates") == STEPS
                     and value.get("selection_evidence") is False
                     and value.get("holdout_access") is False) else None


def run_candidate(candidate: str, output: Path, root: Path = ROOT) -> dict[str, Any]:
    existing = completed_evidence(output)
    if existing:
        return existing
    log = output.with_name(output.name + ".log")
    if output.exists() or log.exists():
        raise FileExistsError(f"Incomplete immutable smoke identity: {output}")
    command = build_command(candidate, output, root)
    for item in (Path(command[0]), Path(command[1])):
        if not item.is_file():
            raise FileNotFoundError(item)
    started = time.monotonic()
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=Path(command[1]).parent,
                                env=environment, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    trace = output / "technical_batches.jsonl"
    records = [json.loads(line) for line in trace.read_text().splitlines()
               if line.strip()] if trace.is_file() else []
    updates = max((int(item.get("optimizer_updates", 0)) for item in records),
                  default=0)
    checkpoints = [output / "weights/last_student.pt",
                   output / "weights/last_teacher.pt"]
    nonfinite_log = bool(re.search(
        r"\b(?:nan|inf)\b", log.read_text(errors="replace"), re.I
    ))
    state = {
        "schema": "yolo-update-limit-smoke-v1",
        "status": "completed" if (
            result.returncode == 0 and updates == STEPS and records
            and all(math.isfinite(float(item["loss"])) for item in records)
            and all(path.is_file() for path in checkpoints)
            and not nonfinite_log
        ) else "failed",
        "candidate": candidate,
        "optimizer_updates": updates,
        "expected_optimizer_updates": STEPS,
        "trace_records": len(records),
        "checkpoint_paths": [str(path) for path in checkpoints if path.is_file()],
        "nonfinite_log": nonfinite_log,
        "exit_code": result.returncode,
        "wall_seconds": time.monotonic() - started,
        "command": command,
        "selection_evidence": False,
        "holdout_access": False,
        "production_asset": False,
    }
    if output.is_dir():
        atomic_json(output / "update_limit_status.json", state)
    if state["status"] != "completed":
        raise RuntimeError(f"{candidate} update-limit smoke failed: {state}")
    return state


def main() -> int:
    rows = []
    for candidate, name in (("ssda-yolo", "ssda-max-update-smoke-001"),
                            ("sf-yolo", "sf-max-update-smoke-001")):
        rows.append(run_candidate(candidate, ROOT / "runs" / name))
    report = {
        "schema": "yolo-update-limit-runtime-audit-v1",
        "status": "completed",
        "expected_optimizer_updates": STEPS,
        "candidates": rows,
        "selection_evidence": False,
        "holdout_access": False,
    }
    atomic_json(REPORT, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
