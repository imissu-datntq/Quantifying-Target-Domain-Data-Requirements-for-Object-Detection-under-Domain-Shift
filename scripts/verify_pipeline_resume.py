"""Compare interrupted/resumed and uninterrupted smoke checkpoints exactly."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bdd_pipeline.artifacts import atomic_json, load_checkpoint


def same(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(same(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--uninterrupted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resumed = load_checkpoint(args.resumed)
    uninterrupted = load_checkpoint(args.uninterrupted)
    if not (resumed["config"]["smoke"] and uninterrupted["config"]["smoke"]):
        raise ValueError("This verification is intended for paired smoke runs")
    checks = {key: same(resumed[key], uninterrupted[key]) for key in
              ("config", "model", "optimizer", "scaler", "global_step", "epoch", "update_in_epoch",
               "random_state", "processed_images", "best_score", "lr_state", "training_completed")}
    max_error = max(float((tensor - uninterrupted["model"][key]).abs().max()) for key, tensor in resumed["model"].items() if tensor.numel())
    report = {"passed": all(checks.values()), "checks": checks,
              "model_tensor_count": len(resumed["model"]), "max_absolute_model_tensor_difference": max_error,
              "global_step": resumed["global_step"], "processed_images": resumed["processed_images"],
              "resumed_checkpoint": str(args.resumed), "uninterrupted_checkpoint": str(args.uninterrupted),
              "scope": "Exact CPU smoke resume; does not establish CUDA/multiworker reproducibility"}
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise AssertionError("Interrupted/resumed run differs from uninterrupted run")


if __name__ == "__main__":
    main()
