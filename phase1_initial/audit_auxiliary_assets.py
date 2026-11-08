"""Audit bounded auxiliary assets without promoting them to production inputs.

The audit is deliberately limited to integrity properties that can be checked
without judging translation quality: train-only provenance, pair counts,
native image geometry, output hashes, finite TargetAugment tensors, and the
instrumented loss summary.  A passing report is still diagnostic evidence;
``production_asset`` remains false until the reviewed auxiliary campaign is
run and its quality/selection gates are approved.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


PAIR_COUNT = 128
NATIVE_SIZE = (1280, 720)
PAIR_RUNS = {
    "source_a_to_b": Path("runs/cut-pairs-source-smoke-001"),
    "target_b_to_a": Path("runs/cut-pairs-target-smoke-001"),
}
TARGETAUGMENT_RUN = Path("runs/targetaugment-lighting-smoke-002")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_pairs(run: Path) -> dict:
    from PIL import Image

    provenance_path = run / "provenance.json"
    report = json.loads(provenance_path.read_text(encoding="utf-8"))
    errors = []
    records = report.get("images", [])
    if report.get("status") != "completed":
        errors.append("pair run is not completed")
    if report.get("split") != "train":
        errors.append("pair run is not train-only")
    if report.get("selection_evidence") is not False:
        errors.append("pair run incorrectly claims selection evidence")
    if report.get("production_asset") is not False:
        errors.append("pair run incorrectly claims production asset")
    if len(records) != PAIR_COUNT or len({row.get("name") for row in records}) != PAIR_COUNT:
        errors.append(f"expected {PAIR_COUNT} unique pair records")
    dimensions = {}
    verified = 0
    for row in records:
        image = run / row["name"]
        if not image.is_file():
            errors.append(f"missing output {row['name']}")
            continue
        if digest(image) != row.get("output_sha256"):
            errors.append(f"output hash mismatch {row['name']}")
        with Image.open(image) as handle:
            dimensions[str(handle.size)] = dimensions.get(str(handle.size), 0) + 1
            if handle.size != NATIVE_SIZE or handle.mode != "RGB":
                errors.append(f"unexpected geometry/mode {row['name']}")
        verified += 1
    return {
        "path": str(run),
        "records": len(records),
        "verified_outputs": verified,
        "dimensions": dimensions,
        "split": report.get("split"),
        "selection_evidence": report.get("selection_evidence"),
        "production_asset": report.get("production_asset"),
        "errors": errors,
    }


def _audit_targetaugment(run: Path) -> dict:
    import torch

    errors = []
    summary_path = run / "metrics.jsonl.summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("steps") != 120:
        errors.append("expected 120 instrumented TargetAugment steps")
    if summary.get("selection_evidence") is not False:
        errors.append("TargetAugment summary incorrectly claims selection evidence")
    if summary.get("optimizer_resume_verified") is not False:
        errors.append("TargetAugment summary incorrectly claims optimizer resume")
    for key in ("peak_allocated_bytes", "peak_reserved_bytes", "mean_update_seconds"):
        value = summary.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            errors.append(f"non-finite summary field {key}")
    weights = {}
    for path in sorted((run / "models").glob("*_iter_120.pth")):
        state = torch.load(str(path), map_location="cpu")
        tensors = [value for value in state.values() if torch.is_tensor(value)]
        finite = bool(tensors) and all(bool(torch.isfinite(value).all()) for value in tensors)
        weights[path.name] = {"tensor_count": len(tensors), "finite": finite}
        if not finite:
            errors.append(f"non-finite or empty TargetAugment state {path.name}")
    if set(weights) != {"decoder_iter_120.pth", "fc1_iter_120.pth", "fc2_iter_120.pth"}:
        errors.append("missing one or more TargetAugment components")
    return {
        "path": str(run),
        "steps": summary.get("steps"),
        "selection_evidence": summary.get("selection_evidence"),
        "optimizer_resume_verified": summary.get("optimizer_resume_verified"),
        "weights": weights,
        "errors": errors,
    }


def audit() -> dict:
    pairs = {name: _audit_pairs(path) for name, path in PAIR_RUNS.items()}
    targetaugment = _audit_targetaugment(TARGETAUGMENT_RUN)
    errors = [f"{name}: {error}" for name, item in pairs.items() for error in item["errors"]]
    errors.extend(f"targetaugment: {error}" for error in targetaugment["errors"])
    return {
        "status": "passed" if not errors else "failed",
        "selection_evidence": False,
        "production_asset": False,
        "train_only": True,
        "pairs": pairs,
        "targetaugment": targetaugment,
        "errors": errors,
        "note": "Integrity-only diagnostic audit; translation quality, semantic box alignment, optimizer resume and production approval remain open.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    report = audit()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
