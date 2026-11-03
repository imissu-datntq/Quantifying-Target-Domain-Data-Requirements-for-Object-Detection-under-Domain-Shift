"""RQ1-RQ4 tables and figures from the one-shot scoring of a frozen prediction plan.

Input is the output directory of `evaluate_frcnn.py --final-plan <schema-3 plan>`: per prediction
set and domain, metrics.json and bootstrap_class_ap.npy (shared test resamples per domain).

Outputs: domain drop with CI and the compound-vs-sum interaction (RQ1); seed-mean recovery
curves with a hierarchical bootstrap band (RQ2); N_BE/N75/N90/N95 as point, band interval, ladder
bracket and power-fit estimate (RQ3); the same against matched-pool ceilings (RQ4); forgetting on
the source test set; generic-pretrained controls. Primary metric: reliable mAP50:95.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from bdd_pipeline.campaign import MATCHED_POOL, TARGETS
from bdd_pipeline.data import CLASSES, EXPECTED_DOMAINS, RELIABLE_CLASSES
from bdd_pipeline.recovery import (LEVELS, band, bootstrap_recovery, fit_power_curve, ladder_bracket, paired_difference,
                                   power_threshold, recovery, sustained_threshold, threshold_interval)

KEYS = ("reliable", "all") + RELIABLE_CLASSES  # "all" is the ten-class sensitivity analysis
PALETTE = {"reliable": "#1f2933", "pedestrian": "#2f6b9a", "car": "#2d997c", "traffic light": "#d48041", "traffic sign": "#8163a8"}


def load(directory: Path) -> list[dict]:
    entries = []
    for metrics_path in sorted(directory.glob("*/*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        classes = np.load(metrics_path.parent / "bootstrap_class_ap.npy")
        if metrics["bootstrap"]["class_order"] != list(CLASSES):
            raise ValueError(f"Unexpected bootstrap class order: {metrics_path}")
        reliable = [CLASSES.index(name) for name in RELIABLE_CLASSES]
        points = {"reliable": metrics["reliable_AP50_95"], "all": metrics["AP50_95"]}
        points.update({row["class"]: row["AP50_95"] for row in metrics["per_class"]})
        boots = {"reliable": classes[:, reliable].mean(axis=1), "all": np.nanmean(classes, axis=1)}
        boots.update({name: classes[:, CLASSES.index(name)] for name in RELIABLE_CLASSES})
        entries.append({"run": metrics["run"], "domain": metrics["domain"], "points": points, "boots": boots})
    if not entries:
        raise FileNotFoundError(f"No scored prediction sets under {directory}")
    return entries


def budget_of(run: dict) -> int:
    return EXPECTED_DOMAINS[run["domain"]]["train"] if run["budget"] == "full" else int(run["budget"])


def threshold_rows(domain: str, key: str, analysis: str, budgets: list[int], point: dict[int, float],
                   curves: np.ndarray, ceiling_budget: int) -> list[dict]:
    lower, upper = band(curves)
    fit = fit_power_curve(budgets, [point[b] for b in budgets], ceiling_budget)
    rows = []
    for label, level in LEVELS.items():
        strict = label == "N_BE"
        replicate_fits = [power_threshold(fit_power_curve(budgets, list(curve), ceiling_budget), level, ceiling_budget)
                          for curve in curves if np.isfinite(curve).all()] if label != "N_BE" else []
        replicate_fits = [value for value in replicate_fits if value is not None]
        rows.append({"domain": domain, "class": key, "analysis": analysis, "threshold": label, "level": level,
                     "ceiling_budget": ceiling_budget, "point": sustained_threshold(point, level, strict),
                     **{f"band_{k}": v for k, v in threshold_interval(budgets, lower, upper, level, strict).items()},
                     **{f"ladder_{k}": v for k, v in ladder_bracket(point, level, strict).items()},
                     "fit": power_threshold(fit, level, ceiling_budget) if label != "N_BE" else None,
                     "fit_ci_low": float(np.percentile(replicate_fits, 2.5)) if replicate_fits else None,
                     "fit_ci_high": float(np.percentile(replicate_fits, 97.5)) if replicate_fits else None,
                     "fit_replicates_defined": len(replicate_fits), "fit_b": fit and fit["b"], "fit_r2": fit and fit["r2"]})
    return rows


def curve_analysis(domain: str, key: str, analysis: str, zero: dict, ladder: dict[int, list[dict]],
                   ceilings: list[dict], ceiling_budget: int, min_denominator: float, seed: int):
    zero_point, ceiling_point = zero["points"][key], mean(c["points"][key] for c in ceilings)
    if zero_point is None or ceiling_point is None:
        return [], []
    runs = {b: [r["boots"][key] for r in rs] for b, rs in ladder.items() if b < ceiling_budget}
    point = {b: recovery(mean(r["points"][key] for r in rs), zero_point, ceiling_point, min_denominator)
             for b, rs in ladder.items() if b < ceiling_budget}
    point[ceiling_budget] = 1.0
    if any(value is None for value in point.values()):
        return [{"domain": domain, "class": key, "analysis": analysis, "budget": None, "recovery": None,
                 "note": f"undefined: ceiling gain {ceiling_point - zero_point:.4f} <= {min_denominator}"}], []
    budgets, curves = bootstrap_recovery(zero["boots"][key], [c["boots"][key] for c in ceilings], runs, ceiling_budget,
                                         seed=seed, min_denominator=min_denominator)
    lower, upper = band(curves)
    points = [{"domain": domain, "class": key, "analysis": analysis, "budget": b, "recovery": point[b],
               "ci_low": float(lo), "ci_high": float(hi), "seeds": len(ladder.get(b, ceilings)),
               "mAP_mean": mean(r["points"][key] for r in ladder.get(b, ceilings)), "zero_shot": zero_point,
               "ceiling": ceiling_point, "undefined_replicates": int(np.isnan(curves[:, j]).sum())}
              for j, (b, lo, hi) in enumerate(zip(budgets, lower, upper))]
    return points, threshold_rows(domain, key, analysis, budgets, point, curves, ceiling_budget)


def plot(points: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for domain in sorted({p["domain"] for p in points}):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for key in ("reliable",) + RELIABLE_CLASSES:
            rows = sorted((p for p in points if p["domain"] == domain and p["class"] == key
                           and p["analysis"] == "full-pool" and p["budget"]), key=lambda p: p["budget"])
            if not rows:
                continue
            x = [p["budget"] for p in rows]
            ax.plot(x, [p["recovery"] for p in rows], marker="o", lw=2.6 if key == "reliable" else 1.3,
                    color=PALETTE[key], label=key)
            if key == "reliable":
                ax.fill_between(x, [p["ci_low"] for p in rows], [p["ci_high"] for p in rows], color=PALETTE[key], alpha=0.15, lw=0)
        for level in (0.75, 0.90):
            ax.axhline(level, color="gray", ls=":", lw=1)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Labeled target images N")
        ax.set_ylabel("Recovery(N); band: 95% CI (reliable)")
        ax.set_title(f"{domain}: recovery against the full-pool ceiling")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(output / f"recovery_{domain}.png", dpi=160)
        plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scored", type=Path, required=True, help="evaluate_frcnn.py --final-plan output directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-denominator", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0, help="Seed for redrawing sampling/ceiling seeds")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    entries = load(args.scored)
    source_model = {e["domain"]: e for e in entries if e["run"]["stage"] == "source"}
    if set(source_model) != {"source", *TARGETS}:
        raise ValueError("The source model must be scored on the source and every target test set")
    reference = source_model["source"]
    drops, points, thresholds, forgetting, controls = [], [], [], [], []
    for domain in TARGETS:
        own = [e for e in entries if e["run"]["domain"] == domain and e["domain"] == domain]
        ladder = defaultdict(list)
        for entry in own:
            if entry["run"]["stage"] == "fine-tune":
                ladder[budget_of(entry["run"])].append(entry)
        pool = EXPECTED_DOMAINS[domain]["train"]
        ceilings = ladder.pop(pool, [])
        zero = source_model[domain]
        if not ceilings:
            raise ValueError(f"{domain}: no ceiling runs scored")
        for key in KEYS:
            if reference["points"][key] is not None and zero["points"][key] is not None:
                drop = paired_difference(reference["points"][key], reference["boots"][key], zero["points"][key], zero["boots"][key])
                ceiling = mean(c["points"][key] for c in ceilings)
                drops.append({"domain": domain, "class": key, "mAP_source": reference["points"][key], "mAP_0": zero["points"][key],
                              "mAP_c": ceiling, "drop": drop["estimate"], "drop_ci_low": drop["ci_low"], "drop_ci_high": drop["ci_high"],
                              "relative_drop": drop["estimate"] / reference["points"][key], "residual_gap": reference["points"][key] - ceiling,
                              "ceiling_seeds": len(ceilings), "test_images": EXPECTED_DOMAINS[domain]["final_test"]})
            # The single matched-pool run is a reference ceiling, not an extra rung of the full-pool ladder.
            rungs = {b: rs for b, rs in ladder.items() if b != MATCHED_POOL}
            for analysis, ceiling_runs, ceiling_budget in (("full-pool", ceilings, pool),
                                                            ("matched-pool", ladder.get(MATCHED_POOL, ceilings), MATCHED_POOL)):
                p, t = curve_analysis(domain, key, analysis, zero, rungs, ceiling_runs, ceiling_budget,
                                      args.min_denominator, args.seed)
                points += p
                thresholds += t
        for entry in entries:
            run = entry["run"]
            if run["domain"] == domain and entry["domain"] == "source" and run["stage"] == "fine-tune":
                forgetting.append({"domain": domain, "budget": budget_of(run), "seed": run["seed"],
                                   "source_reliable": entry["points"]["reliable"],
                                   "change_vs_source_model": entry["points"]["reliable"] - reference["points"]["reliable"]})
            if run["domain"] == domain and entry["domain"] == domain and run["stage"] == "control":
                matched = ladder.get(budget_of(run), [])
                controls.append({"domain": domain, "budget": budget_of(run), "control_reliable": entry["points"]["reliable"],
                                 "fine_tune_reliable_mean": mean(r["points"]["reliable"] for r in matched) if matched else None,
                                 "zero_shot": zero["points"]["reliable"]})
    interaction = {}
    by_domain = {d["domain"]: d for d in drops if d["class"] == "reliable"}
    if len(by_domain) == 3:
        boots = {d: reference["boots"]["reliable"] - source_model[d]["boots"]["reliable"] for d in TARGETS}
        estimate = by_domain["compound"]["drop"] - by_domain["lighting"]["drop"] - by_domain["weather"]["drop"]
        replicate = boots["compound"] - boots["lighting"] - boots["weather"]
        interaction = {"estimate": estimate, "ci_low": float(np.nanpercentile(replicate, 2.5)),
                       "ci_high": float(np.nanpercentile(replicate, 97.5)),
                       "meaning": "drop(compound) - drop(lighting) - drop(weather); > 0 means super-additive"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("rq1_drop", drops), ("recovery_points", points), ("thresholds", thresholds),
                       ("forgetting", forgetting), ("controls", controls)):
        write_csv(args.output_dir / f"{name}.csv", rows)
    clean = lambda value: None if isinstance(value, float) and math.isnan(value) else value  # noqa: E731
    summary = {"min_denominator": args.min_denominator, "interaction": interaction,
               "thresholds": [{k: clean(v) for k, v in row.items()} for row in thresholds
                              if row["class"] == "reliable" and row["analysis"] == "full-pool"]}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.no_plots:
        plot(points, args.output_dir)
    print(f"{len(entries)} scored sets -> {len(points)} curve points, {len(thresholds)} threshold rows: {args.output_dir}")


if __name__ == "__main__":
    main()
