"""Lock selected checkpoints, or prediction files, and evaluation settings before test access.

--checkpoints freezes model files (schema 2). --prediction-manifests freezes the hashed
final-test prediction files written by predict_frcnn.py (schema 3), so the campaign's
held-out labels are opened once, after every run is complete, without reloading models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bdd_pipeline.artifacts import checkpoint_fingerprint
from bdd_pipeline.data import (DOMAINS, EXPECTED_DOMAINS, HOLDOUT_VAL_DOMAINS, RELIABLE_CLASSES, final_test_rows,
                               read_manifests)
from bdd_pipeline.model import MODEL_CONTRACT
from prepare_revised_bdd100k import sha256

BOOTSTRAP = {"resamples": 1000, "seed": 20261030}
EVALUATION_FILES = ("evaluate_frcnn.py", "bdd_pipeline/evaluation.py", "bdd_pipeline/data.py", "bdd_pipeline/model.py")


def image_names_sha256(manifests: dict, domain: str) -> str:
    """Same digest predict_frcnn.py records for the images it predicted on."""
    return hashlib.sha256("\n".join(row["name"] for row in final_test_rows(manifests, domain)).encode("utf-8")).hexdigest()


def prediction_set(path: Path, metadata: dict, manifests: dict) -> dict:
    """Validate one run's prediction manifest without opening any annotation file."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    run = manifest["run"]
    if manifest.get("schema") != 1 or manifest.get("role") != "final_test" or manifest.get("model_contract") != MODEL_CONTRACT:
        raise ValueError(f"Invalid prediction manifest: {path}")
    if run["smoke"] or not run["completed"] or manifest["limited"]:
        raise ValueError(f"Cannot freeze smoke/incomplete/limited predictions: {path}")
    if run["manifest_sha256"] != metadata["manifest_sha256"]:
        raise ValueError(f"Prediction dataset mismatch: {path}")
    files = {}
    for domain, entry in manifest["files"].items():
        file = path.parent / entry["file"]
        if sha256(file) != entry["sha256"]:
            raise ValueError(f"Prediction file changed: {file}")
        if entry["images"] != EXPECTED_DOMAINS[domain]["final_test"] or entry["image_names_sha256"] != image_names_sha256(manifests, domain):
            raise ValueError(f"Predictions do not cover the {domain} final test set: {file}")
        files[domain] = {"path": str(file.resolve()), "sha256": entry["sha256"]}
    return {"manifest": str(path.resolve()), "manifest_sha256": sha256(path), "checkpoint_sha256": manifest["checkpoint_sha256"],
            "run": run, "files": files}


def freeze_predictions(args, metadata: dict, manifests: dict) -> dict:
    sets = [prediction_set(path, metadata, manifests) for path in args.prediction_manifests]
    if len({entry["checkpoint_sha256"] for entry in sets}) != len(sets):
        raise ValueError("Duplicate checkpoints in final plan")
    for index, entry in enumerate(sets):
        run = entry["run"]
        entry["label"] = f"{index:03d}_{run['stage']}_{run['domain']}_N{run['budget']}_seed{run['seed']}"
    domains = sorted({domain for entry in sets for domain in entry["files"]}, key=list(DOMAINS).index)
    return {"schema": 3, "model_contract": MODEL_CONTRACT, "manifest_sha256": metadata["manifest_sha256"],
            "evaluation_implementation_sha256": {name: sha256(Path(__file__).parent / name) for name in EVALUATION_FILES},
            "prediction_sets": sets, "domains": domains, "bootstrap": BOOTSTRAP,
            "reliable_classes": list(RELIABLE_CLASSES), "holdout_val_domains": list(HOLDOUT_VAL_DOMAINS),
            "final_test_images": {domain: EXPECTED_DOMAINS[domain]["final_test"] for domain in domains},
            "selection": "Fixed before accessing held-out labels; no backup selection",
            "primary_metric": "reliable_AP50_95", "test_labels_accessed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/bdd100k_revised"))
    frozen = parser.add_mutually_exclusive_group(required=True)
    frozen.add_argument("--checkpoints", type=Path, nargs="+")
    frozen.add_argument("--prediction-manifests", type=Path, nargs="+", help="prediction_manifest.json files (schema 3)")
    parser.add_argument("--domains", choices=DOMAINS, nargs="+", default=list(DOMAINS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if args.eval_batch_size < 1 or len(set(args.domains)) != len(args.domains):
        parser.error("Invalid threshold/batch or duplicate domains")
    metadata, manifests = read_manifests(args.dataset_root)
    if args.prediction_manifests:
        plan = freeze_predictions(args, metadata, manifests)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(plan, handle, ensure_ascii=False, indent=2)
        print(f"Frozen {len(plan['prediction_sets'])} prediction sets without test annotation access: {args.output}")
        return
    entries = []
    for index, path in enumerate(args.checkpoints):
        entry = checkpoint_fingerprint(path)
        if entry["smoke"] or not entry["completed"]:
            raise ValueError("Cannot freeze smoke/incomplete checkpoint for final research evaluation")
        if entry["manifest_sha256"] != metadata["manifest_sha256"]:
            raise ValueError("Checkpoint dataset mismatch")
        entry["label"] = f"checkpoint_{index:03d}_{entry['domain']}"
        entries.append(entry)
    if len({entry["sha256"] for entry in entries}) != len(entries):
        raise ValueError("Duplicate checkpoints in final plan")
    plan = {"schema": 2, "model_contract": MODEL_CONTRACT, "manifest_sha256": metadata["manifest_sha256"],
            "evaluation_implementation_sha256": {name: sha256(Path(__file__).parent / name) for name in ("evaluate_frcnn.py", "bdd_pipeline/evaluation.py", "bdd_pipeline/data.py", "bdd_pipeline/model.py")},
            "checkpoints": entries, "domains": args.domains,
            "reliable_classes": list(RELIABLE_CLASSES), "holdout_val_domains": list(HOLDOUT_VAL_DOMAINS),
            "final_test_images": {domain: EXPECTED_DOMAINS[domain]["final_test"] for domain in args.domains},
            "eval_batch_size": args.eval_batch_size,
            "amp": args.amp, "selection": "Fixed before accessing held-out labels; no backup selection",
            "primary_metric": "reliable_AP50_95", "test_labels_accessed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
    print(f"Frozen without test annotation access: {args.output}")


if __name__ == "__main__":
    main()
