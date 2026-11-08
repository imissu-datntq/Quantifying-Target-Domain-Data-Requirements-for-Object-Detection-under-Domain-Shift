"""RQ5 tables and figures: ALDI++ adaptation against plain fine-tuning (FT) on the final test set.

Inputs are two one-shot scoring outputs of `evaluate_frcnn.py --final-plan` (schema 3) that share
the test resamples (bootstrap seed 20261030, same final-test image lists): the FT campaign and the
RQ5 grid. Rules: .agent/DECISIONS.md 2026-11-03 (written before any RQ5 result).

- Delta(N) = mean over sampling seeds of [mAP_DA(N, s) - mAP_FT(N, s)], paired by seed (N = 0: the
  UDA seeds against the source model). CI: hierarchical bootstrap, test resample t shared by every
  run, seeds redrawn with replacement as pairs. Also in recovery units against FT's mAP(0) and
  ceiling (only where that denominator is at least --min-denominator).
- Meaningful advantage: Delta >= delta (0.005) and the lower 95% bound > 0.
- Crossover N_x: smallest tested N from which no larger tested N has a meaningful advantage.
- N_eq: FT budget whose seed-mean mAP equals mAP_DA(N), linear in log N between FT rungs. The FT
  curve is not monotone at small N (it dips below mAP(0)), so the sustained crossing is used: the
  last upward crossing of the FT curve, consistent with the sustained recovery thresholds.
  "<= N_min" when FT is at or above the target from its smallest rung on, "> full pool" above the
  ceiling. Bootstrap replicates redraw seeds per rung and the ceiling seeds.
- Attribution at N in {0, 10, 100, 1000}, seed 0: adapt - replay and replay - FT (test-resample CI).
- Secondary: per-class Delta (four reliable classes), forgetting on the source test set.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from analyze_campaign import load, write_csv
from bdd_pipeline.campaign import MATCHED_POOL
from bdd_pipeline.data import EXPECTED_DOMAINS, RELIABLE_CLASSES
from bdd_pipeline.recovery import paired_difference

TARGETS = ("compound", "weather", "lighting")
DELTA = 0.005
ATTRIBUTION_BUDGETS = (0, 10, 100, 1000)
KEYS = ("reliable",) + RELIABLE_CLASSES
# Reference categorical slots 1-3 (dataviz palette, validated light: PASS; aqua contrast WARN,
# relieved by distinct markers, the legend and the CSV tables).
COLORS = {"FT": "#2a78d6", "adapt": "#eb6834", "replay": "#1baf7a"}
MARKERS = {"FT": "o", "adapt": "s", "replay": "^"}


def tail(confidence: float = 0.95) -> float:
    return 100 * (1 - confidence) / 2


def seeds_of(entries: list[dict]) -> dict[int, dict]:
    out = {}
    for entry in entries:
        seed = int(entry["run"]["seed"])
        if seed in out:
            raise ValueError(f"Duplicate seed {seed}: {entry['run']}")
        out[seed] = entry
    return out


def paired_delta(da: dict[int, dict], ft: dict[int, dict] | dict, key: str, rng: np.random.Generator,
                 zero: dict | None = None) -> dict | None:
    """Seed-paired DA - FT; with zero given, every DA seed is paired with the source model."""
    seeds = sorted(da) if zero is not None else sorted(set(da) & set(ft))
    if not seeds:
        return None
    reference = (lambda s: zero) if zero is not None else (lambda s: ft[s])
    points = [da[s]["points"][key] - reference(s)["points"][key] for s in seeds
              if da[s]["points"][key] is not None and reference(s)["points"][key] is not None]
    if len(points) != len(seeds):
        return None
    differences = np.stack([da[s]["boots"][key] - reference(s)["boots"][key] for s in seeds])  # [seed, replicate]
    replicates = differences.shape[1]
    draws = rng.integers(0, len(seeds), (replicates, len(seeds)))
    boot = differences[draws, np.arange(replicates)[:, None]].mean(axis=1)
    return {"estimate": mean(points), "ci_low": float(np.nanpercentile(boot, tail())),
            "ci_high": float(np.nanpercentile(boot, 100 - tail())), "seeds": len(seeds), "boot": boot}


def crossover(rows: list[dict]) -> dict:
    """Smallest tested N from which no tested N has a meaningful advantage (rows sorted by N)."""
    budgets = [r["budget"] for r in rows]
    meaningful = [r["meaningful"] for r in rows]
    if meaningful and meaningful[-1]:
        return {"N_x": None, "bracket": f"> {budgets[-1]} (beyond the largest tested N)"}
    index = len(rows)
    while index > 0 and not meaningful[index - 1]:
        index -= 1
    if index == 0:
        return {"N_x": budgets[0], "bracket": f"<= {budgets[0]} (no meaningful advantage at any tested N)"}
    return {"N_x": budgets[index], "bracket": f"({budgets[index - 1]}, {budgets[index]}]"}


def equivalent_budget(target: float, budgets: list[int], values: list[float]) -> tuple[float | None, str]:
    """Sustained log-linear inverse of the FT curve (budgets ascending, last one = full pool)."""
    if not np.isfinite(target):
        return None, "undefined"
    if target > values[-1]:
        return None, f"> full pool ({budgets[-1]})"
    below = [i for i, v in enumerate(values) if v < target]
    if not below:
        return float(budgets[0]), f"<= {budgets[0]}"
    i = below[-1]  # last rung below the target; the next rung is at or above it and so are all later ones
    lo, hi = budgets[i], budgets[i + 1]
    fraction = (target - values[i]) / (values[i + 1] - values[i])
    value = math.exp(math.log(lo) + fraction * (math.log(hi) - math.log(lo)))
    return value, f"{value:.0f}"


def ft_curve(zero: dict, ladder: dict[int, dict[int, dict]], ceilings: dict[int, dict], pool: int):
    budgets = sorted(b for b in ladder if b < pool) + [pool]
    point = [mean(e["points"]["reliable"] for e in ladder[b].values()) for b in budgets[:-1]]
    point.append(mean(e["points"]["reliable"] for e in ceilings.values()))
    return budgets, point


def ft_curve_replicates(ladder, ceilings, budgets, rng) -> np.ndarray:
    """[replicate, budget] seed-mean FT reliable mAP with seeds redrawn per rung."""
    stacks = [np.stack([e["boots"]["reliable"] for e in ladder[b].values()]) for b in budgets[:-1]]
    stacks.append(np.stack([e["boots"]["reliable"] for e in ceilings.values()]))
    replicates = stacks[0].shape[1]
    out = np.empty((replicates, len(budgets)))
    for j, stack in enumerate(stacks):
        draws = rng.integers(0, len(stack), (replicates, len(stack)))
        out[:, j] = stack[draws, np.arange(replicates)[:, None]].mean(axis=1)
    return out


def plot(curves: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for domain, data in curves.items():
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        position = lambda n: 3 if n == 0 else n  # noqa: E731  N=0 drawn at x=3, labelled "0"
        for arm, label in (("FT", "fine-tune (FT)"), ("adapt", "ALDI++ adapt"), ("replay", "replay control (seed 0)")):
            rows = sorted(data[arm], key=lambda r: r[0])
            if not rows:
                continue
            x = [position(n) for n, _, _, _ in rows]
            y = [m for _, m, _, _ in rows]
            ax.plot(x, y, marker=MARKERS[arm], ms=8, lw=2, color=COLORS[arm], label=label,
                    mec="white", mew=1.5, zorder=3)
            lows = [lo for _, _, lo, _ in rows]
            if arm != "replay" and all(v is not None for v in lows):
                ax.fill_between(x, lows, [hi for _, _, _, hi in rows], color=COLORS[arm], alpha=0.12, lw=0)
        ax.axhline(data["ceiling"], color="#6b6b66", ls="--", lw=1)
        ax.text(position(10), data["ceiling"], f" FT full-pool ceiling {data['ceiling']:.3f}", va="bottom",
                fontsize=8.5, color="#4a4a46")
        ax.set_xscale("log")
        ticks = [3] + [b for b in (10, 100, 1000, 10000) if b <= max(r[0] for r in data["FT"])]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["0"] + [f"{t:,}" for t in ticks[1:]])
        ax.minorticks_off()
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", color="#e6e6e3", lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Labeled target images N (log scale)")
        ax.set_ylabel("Reliable mAP50:95, final test")
        ax.set_title(f"{domain}: ALDI++ vs fine-tuning (seed means; band: 95% CI)")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
        fig.tight_layout()
        fig.savefig(output / f"rq5_{domain}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ft-scored", type=Path, required=True, help="Scored FT campaign (evaluate_frcnn.py output)")
    parser.add_argument("--da-scored", type=Path, required=True, help="Scored RQ5 grid")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-denominator", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0, help="Seed for redrawing sampling seeds")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    ft_entries, da_entries = load(args.ft_scored), load(args.da_scored)
    source_model = {e["domain"]: e for e in ft_entries if e["run"]["stage"] == "source"}

    def select(entries, stage, run_domain, eval_domain):
        groups = defaultdict(list)
        for e in entries:
            run = e["run"]
            if run["stage"] == stage and run["domain"] == run_domain and e["domain"] == eval_domain:
                groups[run["budget"]].append(e)
        return {b: seeds_of(es) for b, es in groups.items()}

    deltas, equivalents, attribution, per_class, forgetting, summary, curves = [], [], [], [], [], {}, {}
    for domain in TARGETS:
        pool = EXPECTED_DOMAINS[domain]["train"]
        zero = source_model[domain]
        ft = select(ft_entries, "fine-tune", domain, domain)
        ceilings = ft.pop("full")
        ft_ladder = {int(b): runs for b, runs in ft.items() if int(b) != MATCHED_POOL}  # matched pool is a ceiling, not a rung
        adapt = {int(b): runs for b, runs in select(da_entries, "adapt", domain, domain).items()}
        replay = {int(b): runs for b, runs in select(da_entries, "replay", domain, domain).items()}
        replay_zero = select(da_entries, "replay", "source", domain).get("0", {})
        if replay_zero:
            replay[0] = replay_zero
        ceiling = mean(e["points"]["reliable"] for e in ceilings.values())
        denominator = ceiling - zero["points"]["reliable"]
        budgets, ft_point = ft_curve(zero, ft_ladder, ceilings, pool)
        ft_reps = ft_curve_replicates(ft_ladder, ceilings, budgets, np.random.default_rng(args.seed + 1))
        domain_rows = []
        for n in sorted(adapt):
            for key in KEYS:
                result = paired_delta(adapt[n], {} if n == 0 else ft_ladder.get(n, {}), key, rng,
                                      zero=zero if n == 0 else None)
                if result is None:
                    continue
                ft_mean = zero["points"][key] if n == 0 else mean(e["points"][key] for e in ft_ladder[n].values())
                row = {"domain": domain, "budget": n, "class": key, "seeds": result["seeds"],
                       "mAP_DA": mean(e["points"][key] for e in adapt[n].values()), "mAP_FT": ft_mean,
                       "delta": result["estimate"], "ci_low": result["ci_low"], "ci_high": result["ci_high"],
                       "meaningful": result["estimate"] >= DELTA and result["ci_low"] > 0}
                if key == "reliable":
                    row["delta_recovery_units"] = result["estimate"] / denominator if denominator >= args.min_denominator else None
                    row["recovery_denominator"] = denominator
                    deltas.append(row)
                    domain_rows.append(row)
                else:
                    per_class.append(row)
            # N_eq of the DA seed mean, with a bootstrap over test resamples and seeds.
            da_stack = np.stack([e["boots"]["reliable"] for e in adapt[n].values()])
            draws = rng.integers(0, len(da_stack), (da_stack.shape[1], len(da_stack)))
            da_reps = da_stack[draws, np.arange(da_stack.shape[1])[:, None]].mean(axis=1)
            target = mean(e["points"]["reliable"] for e in adapt[n].values())
            value, text = equivalent_budget(target, budgets, ft_point)
            replicate_values, above, at_min = [], 0, 0
            for t in range(len(da_reps)):
                v, label = equivalent_budget(float(da_reps[t]), budgets, list(ft_reps[t]))
                if label.startswith(">"):
                    above += 1
                elif label.startswith("<="):
                    at_min += 1
                if v is not None:
                    replicate_values.append(v)
            equivalents.append({"domain": domain, "budget": n, "mAP_DA": target, "N_eq": value, "N_eq_label": text,
                                "ci_low": float(np.percentile(replicate_values, 2.5)) if replicate_values else None,
                                "ci_high": float(np.percentile(replicate_values, 97.5)) if replicate_values else None,
                                "replicates_above_pool": above, "replicates_at_min_rung": at_min,
                                "replicates": len(da_reps), "ft_budgets": budgets, "ft_curve": ft_point})
        cross = crossover(sorted(domain_rows, key=lambda r: r["budget"]))
        for n in ATTRIBUTION_BUDGETS:
            a, r = adapt.get(n, {}).get(0), replay.get(n, {}).get(0)
            f = zero if n == 0 else ft_ladder.get(n, {}).get(0)
            if a is None or r is None or f is None:
                continue
            for name, left, right in (("adapt - replay", a, r), ("replay - FT", r, f), ("adapt - FT", a, f)):
                d = paired_difference(left["points"]["reliable"], left["boots"]["reliable"],
                                      right["points"]["reliable"], right["boots"]["reliable"])
                attribution.append({"domain": domain, "budget": n, "seed": 0, "comparison": name,
                                    "left": left["points"]["reliable"], "right": right["points"]["reliable"],
                                    "difference": d["estimate"], "ci_low": d["ci_low"], "ci_high": d["ci_high"]})
        reference = source_model["source"]["points"]["reliable"]
        for stage, groups in (("adapt", select(da_entries, "adapt", domain, "source")),
                              ("replay", select(da_entries, "replay", domain, "source")),
                              ("fine-tune", select(ft_entries, "fine-tune", domain, "source"))):
            for budget, runs in groups.items():
                if stage == "fine-tune" and budget not in {str(n) for n in adapt}:
                    continue
                forgetting.append({"domain": domain, "stage": stage, "budget": int(budget), "seeds": len(runs),
                                   "source_reliable_mean": mean(e["points"]["reliable"] for e in runs.values()),
                                   "change_vs_source_model": mean(e["points"]["reliable"] for e in runs.values()) - reference})
        summary[domain] = {"mAP_0": zero["points"]["reliable"], "ceiling": ceiling, "recovery_denominator": denominator,
                           "crossover": cross,
                           "delta": {r["budget"]: {k: r[k] for k in ("delta", "ci_low", "ci_high", "meaningful")}
                                     for r in domain_rows},
                           "N_eq": {e["budget"]: e["N_eq_label"] for e in equivalents if e["domain"] == domain}}

        def band(groups, n):
            boots = np.stack([e["boots"]["reliable"] for e in groups.values()])
            draws = rng.integers(0, len(boots), (boots.shape[1], len(boots)))
            reps = boots[draws, np.arange(boots.shape[1])[:, None]].mean(axis=1)
            return (n, mean(e["points"]["reliable"] for e in groups.values()),
                    float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5)))

        ft_rows = [(0, zero["points"]["reliable"], float(np.percentile(zero["boots"]["reliable"], 2.5)),
                    float(np.percentile(zero["boots"]["reliable"], 97.5)))]
        ft_rows += [band(ft_ladder[b], b) for b in sorted(ft_ladder) if b < pool]
        curves[domain] = {"FT": ft_rows, "adapt": [band(adapt[n], n) for n in sorted(adapt)],
                          "replay": [(n, r[0]["points"]["reliable"], None, None) for n, r in sorted(replay.items()) if 0 in r],
                          "ceiling": ceiling}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("rq5_delta", deltas), ("rq5_delta_per_class", per_class), ("rq5_equivalent_budget", equivalents),
                       ("rq5_attribution", attribution), ("rq5_forgetting", forgetting)):
        write_csv(args.output_dir / f"{name}.csv", rows)
    (args.output_dir / "summary.json").write_text(json.dumps(
        {"rules": "DECISIONS 2026-11-03", "delta_threshold": DELTA, "domains": summary,
         "attribution": attribution}, indent=2, default=str), encoding="utf-8")
    if not args.no_plots:
        plot(curves, args.output_dir)
    for domain, s in summary.items():
        print(f"{domain}: crossover {s['crossover']['bracket']}; N_eq {s['N_eq']}")
        for n, d in s["delta"].items():
            print(f"  N={n}: delta {d['delta']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]{' *' if d['meaningful'] else ''}")


if __name__ == "__main__":
    main()
