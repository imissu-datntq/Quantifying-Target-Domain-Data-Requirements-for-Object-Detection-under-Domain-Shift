"""Compare R101-C4 restart checkpoints without claiming exactness.

The report separates a same-segment control (two fresh deterministic runs) from
the uninterrupted-versus-resumed comparison.  This avoids attributing every
bitwise difference to checkpoint restore when the pinned CUDA/ROI stack is
already nondeterministic before a restart.
"""

import argparse
import json
import math
from pathlib import Path


def _compare(left, right, path=""):
    import torch

    result = {"tensor_count": 0, "tensor_mismatches": 0, "max_abs": 0.0,
              "mean_abs_sum": 0.0, "allclose_count": 0, "other_mismatches": 0}
    if torch.is_tensor(left):
        result["tensor_count"] = 1
        if left.shape != right.shape or left.dtype != right.dtype:
            result["tensor_mismatches"] = 1
            result["other_mismatches"] = 1
            return result
        delta = (left - right).abs()
        max_abs = float(delta.max()) if delta.numel() else 0.0
        mean_abs = float(delta.mean()) if delta.numel() else 0.0
        result["max_abs"] = max_abs
        result["mean_abs_sum"] = mean_abs
        if not torch.equal(left, right):
            result["tensor_mismatches"] = 1
        if torch.allclose(left, right, rtol=1e-5, atol=1e-6):
            result["allclose_count"] = 1
        return result
    if isinstance(left, dict):
        for key, value in left.items():
            if key not in right:
                result["other_mismatches"] += 1
                continue
            child = _compare(value, right[key], f"{path}/{key}")
            for name in ("tensor_count", "tensor_mismatches", "mean_abs_sum",
                         "allclose_count", "other_mismatches"):
                result[name] += child[name]
            result["max_abs"] = max(result["max_abs"], child["max_abs"])
        for key in right:
            if key not in left:
                result["other_mismatches"] += 1
        return result
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            result["other_mismatches"] += 1
            return result
        for index, (value, other) in enumerate(zip(left, right)):
            child = _compare(value, other, f"{path}/{index}")
            for name in ("tensor_count", "tensor_mismatches", "mean_abs_sum",
                         "allclose_count", "other_mismatches"):
                result[name] += child[name]
            result["max_abs"] = max(result["max_abs"], child["max_abs"])
        return result
    if left != right:
        result["other_mismatches"] += 1
    return result


def compare_checkpoint(left_path: Path, right_path: Path) -> dict:
    import torch

    left = torch.load(str(left_path), map_location="cpu")
    right = torch.load(str(right_path), map_location="cpu")
    sections = {}
    for key in ("model", "optimizer", "scheduler"):
        sections[key] = _compare(left[key], right[key], key)
    return {
        "left": str(left_path),
        "right": str(right_path),
        "iterations": [left.get("iteration"), right.get("iteration")],
        "sections": sections,
        "exact": all(item["tensor_mismatches"] == 0 and item["other_mismatches"] == 0
                      for item in sections.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninterrupted", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True,
                        help="Independent fresh run covering the same segment")
    parser.add_argument("--split", type=Path, required=True,
                        help="First segment checkpoint")
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema": "r101-restart-audit-v1",
        "selection_evidence": False,
        "deterministic_requested": True,
        "same_segment_control": compare_checkpoint(args.split, args.control),
        "uninterrupted_vs_resumed": compare_checkpoint(args.uninterrupted, args.resumed),
    }
    report["exact_restart"] = (report["same_segment_control"]["exact"]
                                and report["uninterrupted_vs_resumed"]["exact"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
