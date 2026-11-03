"""Apply the pre-registered RQ5 pilot rule (DECISIONS.md, 2026-11-03) to the pilot's val curves.

alpha: higher mean peak over its two runs (N=0, N=100); within delta of each other -> 0.9996.
T:     smallest evaluated step where both runs under alpha are within delta of their own peak and
       stay within delta at every later step. Gate: N=0 teacher at T no more than delta below step 0.
Input: a directory with adapt_lighting_N{0,100}_a{alpha}/validation_step_*.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DELTA, DEFAULT_ALPHA, ALPHAS, BUDGETS = 0.005, 0.9996, (0.999, 0.9996), (0, 100)


def curve(folder: Path) -> dict[int, float]:
    points = {}
    for path in folder.glob("validation_step_*.json"):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        points[int(metrics["step"])] = metrics["reliable_AP50_95"]
    return dict(sorted(points.items()))


def sustained_step(points: dict[int, float]) -> int | None:
    """None if the curve never stays within delta of its peak (it ends > delta below it)."""
    peak = max(points.values())
    steps = list(points)
    return next((s for i, s in enumerate(steps) if all(points[t] >= peak - DELTA for t in steps[i:])), None)


def lock(curves: dict[tuple[int, float], dict[int, float]]) -> dict:
    mean_peak = {alpha: sum(max(curves[(n, alpha)].values()) for n in BUDGETS) / len(BUDGETS) for alpha in ALPHAS}
    best = max(mean_peak, key=mean_peak.get)
    alpha = DEFAULT_ALPHA if abs(mean_peak[0.999] - mean_peak[0.9996]) <= DELTA else best
    per_run = {n: sustained_step(curves[(n, alpha)]) for n in BUDGETS}
    steps = sorted(set(curves[(0, alpha)]) & set(curves[(100, alpha)]))
    zero = curves[(0, alpha)]
    if None in per_run.values():  # a run collapses after its peak: no T exists; treat as a suspect port
        return {"mean_peak": mean_peak, "alpha": alpha, "sustained_step_per_run": per_run, "T": None,
                "T_is_pilot_maximum": None, "gate": {"N0_step0": zero[0], "N0_at_T": None, "passed": False},
                "curves": {f"N{n}_a{a}": c for (n, a), c in curves.items()}, "delta": DELTA}
    # both within delta from T on: the later of the two sustained steps, restricted to common steps
    t = next(s for s in steps if s >= max(per_run.values()))
    gate_ok = zero[t] >= zero[0] - DELTA
    return {"mean_peak": mean_peak, "alpha": alpha, "sustained_step_per_run": per_run, "T": t,
            "T_is_pilot_maximum": t == max(steps), "gate": {"N0_step0": zero[0], "N0_at_T": zero[t], "passed": gate_ok},
            "curves": {f"N{n}_a{a}": c for (n, a), c in curves.items()}, "delta": DELTA}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pilot_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    curves = {(n, a): curve(args.pilot_dir / f"adapt_lighting_N{n}_a{a}") for a in ALPHAS for n in BUDGETS}
    if any(len(c) < 2 for c in curves.values()):
        raise ValueError("Every pilot run needs its validation curve")
    result = lock(curves)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
