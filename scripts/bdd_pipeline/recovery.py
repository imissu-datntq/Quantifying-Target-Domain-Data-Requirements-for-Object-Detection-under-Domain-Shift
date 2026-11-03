"""Recovery(N), sustained thresholds and per-class recovery from evaluation metrics."""
from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean, stdev

import numpy as np

LEVELS = {"N_BE": 0.0, "N75": 0.75, "N90": 0.90, "N95": 0.95}


def recovery(score: float, zero_shot: float, ceiling: float, min_denominator: float = 0.0) -> float | None:
    """[mAP(N) - mAP(0)] / [mAP_c - mAP(0)]; undefined (None) unless the denominator exceeds min_denominator."""
    if min_denominator < 0:
        raise ValueError("Minimum denominator must be non-negative")
    denominator = ceiling - zero_shot
    if not denominator > max(min_denominator, 0.0):
        return None
    return (score - zero_shot) / denominator


def sustained_threshold(curve: dict[int, float], level: float, strict: bool) -> int | None:
    """Smallest positive budget from which every larger budget's mean recovery passes the level."""
    passing = None
    for budget in sorted(curve, reverse=True):
        value = curve[budget]
        if budget <= 0 or not (value > level if strict else value >= level):
            break
        passing = budget
    return passing


def budget_value(budget: str | int, pools: dict[str, int], domain: str) -> int:
    return pools[domain] if str(budget) == "full" else int(budget)


def recovery_table(records: list[dict], pools: dict[str, int], min_denominator: float = 0.0) -> list[dict]:
    """Rows of recovery per domain/budget/seed/class.

    Each record holds domain, budget (0 = source-only, "full" = ceiling, else N), seed and
    metrics (an evaluate_frcnn metrics dict). The ceiling and zero-shot references are the
    seed means. Class "reliable" uses reliable_AP50_95 and "all" the ten-class AP50_95.
    """
    scores: dict[tuple, dict[str, float]] = {}
    for record in records:
        metrics = record["metrics"]
        values = {"reliable": metrics["reliable_AP50_95"], "all": metrics["AP50_95"]}
        values.update({row["class"]: row["AP50_95"] for row in metrics["per_class"]})
        key = (record["domain"], budget_value(record["budget"], pools, record["domain"]), record["seed"])
        if key in scores:
            raise ValueError(f"Duplicate result: {key}")
        scores[key] = values
    references = defaultdict(lambda: defaultdict(list))
    for (domain, budget, _), values in scores.items():
        if budget in (0, pools[domain]):
            for name, value in values.items():
                if value is not None:
                    references[(domain, budget)][name].append(value)
    rows = []
    for (domain, budget, seed), values in sorted(scores.items(), key=lambda item: (item[0][0], item[0][1], str(item[0][2]))):
        zero, ceiling = references.get((domain, 0)), references.get((domain, pools[domain]))
        if zero is None or ceiling is None:
            raise ValueError(f"{domain} needs both source-only (budget 0) and full-pool results")
        for name, value in values.items():
            if value is None or not zero[name] or not ceiling[name]:
                continue
            zero_mean, ceiling_mean = mean(zero[name]), mean(ceiling[name])
            rows.append({"domain": domain, "budget": budget, "seed": seed, "class": name, "AP50_95": value,
                         "zero_shot": zero_mean, "ceiling": ceiling_mean, "denominator": ceiling_mean - zero_mean,
                         "raw_change": value - zero_mean,
                         "recovery": recovery(value, zero_mean, ceiling_mean, min_denominator)})
    return rows


def curve_summary(rows: list[dict]) -> list[dict]:
    """Seed mean/SD of recovery and sustained thresholds per domain and class."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["recovery"] is not None and row["budget"] > 0:
            grouped[(row["domain"], row["class"])][row["budget"]].append(row["recovery"])
    summary = []
    for (domain, name), by_budget in sorted(grouped.items()):
        curve = {budget: mean(values) for budget, values in by_budget.items()}
        entry = {"domain": domain, "class": name,
                 "curve": [{"budget": budget, "mean": curve[budget], "sd": stdev(values) if len(values) > 1 else math.nan,
                            "seeds": len(values)} for budget, values in sorted(by_budget.items())]}
        entry.update({label: sustained_threshold(curve, level, strict=label == "N_BE") for label, level in LEVELS.items()})
        summary.append(entry)
    return summary


# ---- Uncertainty and threshold estimation (proposal 7.3) --------------------------------------
# Bootstrap inputs are per-run vectors over the same test-image resamples: every run of a domain
# is scored with one seed and one image count, so replicate t means the same resampled test set.


def bootstrap_recovery(zero: np.ndarray, ceilings: list[np.ndarray], runs: dict[int, list[np.ndarray]],
                       ceiling_budget: int, *, seed: int = 0, min_denominator: float = 0.0) -> tuple[list[int], np.ndarray]:
    """Replicate curves [replicate, budget] of seed-mean Recovery with three sources of uncertainty.

    Replicate t uses test resample t for every run (test-set precision), redraws the sampling
    seeds at each budget with replacement (which images a budget drew) and redraws the ceiling
    seeds (noise in the denominator). The ceiling rung reuses the redrawn ceiling, so it is 1.
    Replicates whose denominator does not exceed min_denominator are NaN.
    """
    if not ceilings or any(not values for values in runs.values()):
        raise ValueError("Need at least one ceiling run and one run per budget")
    rng = np.random.default_rng(seed)
    budgets = sorted(set(runs) | {ceiling_budget})
    replicates = len(zero)
    curves = np.full((replicates, len(budgets)), np.nan)
    stacked = {budget: np.stack(values) for budget, values in runs.items()}
    ceiling_stack = np.stack(ceilings)
    for t in range(replicates):
        ceiling = ceiling_stack[rng.integers(0, len(ceilings), len(ceilings)), t].mean()
        denominator = ceiling - zero[t]
        if not denominator > max(min_denominator, 0.0):
            continue
        for j, budget in enumerate(budgets):
            if budget == ceiling_budget:
                curves[t, j] = 1.0
                continue
            values = stacked[budget]
            score = values[rng.integers(0, len(values), len(values)), t].mean()
            curves[t, j] = (score - zero[t]) / denominator
    return budgets, curves


def band(curves: np.ndarray, confidence: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Percentile band per budget over replicates, ignoring undefined (NaN) replicates."""
    tail = 100 * (1 - confidence) / 2
    return np.nanpercentile(curves, tail, axis=0), np.nanpercentile(curves, 100 - tail, axis=0)


def threshold_interval(budgets: list[int], lower: np.ndarray, upper: np.ndarray, level: float, strict: bool) -> dict:
    """Bounds on a sustained threshold: the optimistic band crosses first, the pessimistic band last."""
    return {"from": sustained_threshold(dict(zip(budgets, upper)), level, strict),
            "to": sustained_threshold(dict(zip(budgets, lower)), level, strict)}


def ladder_bracket(curve: dict[int, float], level: float, strict: bool) -> dict:
    """Last failing rung below the sustained threshold, and the threshold rung itself."""
    passing = sustained_threshold(curve, level, strict)
    below = [budget for budget in sorted(curve) if 0 < budget and (passing is None or budget < passing)]
    return {"last_failing": below[-1] if below else None, "first_passing": passing}


def fit_power_curve(budgets: list[int], values: list[float], full_budget: int) -> dict | None:
    """Least-squares Recovery(N) = 1 - a[N^-b - Nc^-b] through Recovery(Nc) = 1, on rungs 0 < N < Nc.

    a is closed-form for each b on a grid; returns a, b, RMSE and R^2, or None when the fit is
    undefined (fewer than two rungs, or no positive a).
    """
    points = [(n, v) for n, v in zip(budgets, values) if 0 < n < full_budget and np.isfinite(v)]
    if len(points) < 2:
        return None
    n = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    best = None
    for b in np.linspace(0.01, 3.0, 600):
        g = n ** -b - full_budget ** -b
        a = float(g @ (1 - y) / (g @ g))
        if a <= 0:
            continue
        sse = float(((1 - a * g) - y) @ ((1 - a * g) - y))
        if best is None or sse < best[0]:
            best = (sse, a, float(b))
    if best is None:
        return None
    sse, a, b = best
    total = float(((y - y.mean()) ** 2).sum())
    return {"a": a, "b": b, "rmse": (sse / len(y)) ** 0.5, "r2": 1 - sse / total if total > 0 else None,
            "rungs": len(y), "form": "1 - a[N^-b - Nc^-b]"}


def power_threshold(fit: dict | None, level: float, full_budget: int) -> float | None:
    """Budget at which the fitted curve reaches the level; None outside (0, Nc]."""
    if fit is None or level >= 1:
        return None
    base = full_budget ** -fit["b"] + (1 - level) / fit["a"]
    budget = base ** (-1 / fit["b"])
    return float(budget) if 0 < budget <= full_budget else None


def paired_difference(point_a: float, boot_a: np.ndarray, point_b: float, boot_b: np.ndarray,
                      confidence: float = 0.95) -> dict:
    """a - b with a percentile interval from replicate-wise differences."""
    tail = 100 * (1 - confidence) / 2
    difference = boot_a - boot_b
    return {"estimate": point_a - point_b, "ci_low": float(np.nanpercentile(difference, tail)),
            "ci_high": float(np.nanpercentile(difference, 100 - tail))}
