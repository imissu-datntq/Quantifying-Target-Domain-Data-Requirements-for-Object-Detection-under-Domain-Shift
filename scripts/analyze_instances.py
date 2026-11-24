"""Instances or images: does a class's AP gain along the fine-tune ladder follow its labeled instances or the images?

Direction 1.1 of the lecturer's follow-up (2026-11-24). Rules: .agent/plans/active/2026-11-24-direction-1-analyses.md,
section "1. Instances or images". They were written before this analysis was run but after the curves, thresholds and
RQ5 tables had been seen, so they are descriptive, not confirmatory.

Inputs: the campaign's scored directory (`evaluate_frcnn.py --final-plan` output) and the revised dataset root, of
which only manifests/ and annotations/bdd100k_labels_images_train.json are read; no test or val annotation file is
opened. Runs are told apart by their run fields, never by folder names.

- Counts: the train annotation file is streamed once, every image's boxes pass through sanitize_labels (the boxes
  training saw) and are counted per reliable class. Only the train pools of the source and the three targets are kept.
  A run's images are rebuilt with the training code's own call, nested_rows(domain_rows(manifests, "train", domain),
  budget, seed), with the budget and seed of its run fields.
- Rungs: per shift, the campaign fine-tune runs trained on that shift and scored on it, at the locked epoch count
  (POLICY["epochs"]), budgets in LADDER + EXTRA_RUNGS plus "full". The matched-pool runs at 1,738 of lighting and
  weather are not rungs. N = 0 is the source model scored on the shift (0 images, 0 instances, 0 updates).
- Observations: per shift and sampling seed, the rungs that seed has form a chain; each pair of consecutive rungs is a
  transition. One observation = (shift, seed, transition, class): dAP = AP_c(to) - AP_c(from), dN images added, dI
  instances of class c added, dln N = ln(N_to / N_from), dln I = ln(I_to / I_from). The 0 -> first-rung transitions
  are tabulated (from_zero) and left out of every regression. Observations with I_from = 0 are left out of the log form
  and counted (dropped_zero_instances). The update counts of both rungs are columns.
- Specifications, each by OLS in the log form (x_I = dln I, x_N = dln N) and the raw form (x_I = dI, x_N = dN):
  images (1 + x_N), instances (1 + x_I), both (1 + x_I + x_N), both+class (class intercepts + x_I + x_N),
  within (x_I after demeaning dAP and x_I within cells shift x class x transition; cells with at least two
  observations; within-R^2), and both per class. A rank-deficient design or a non-finite outcome leaves a fit
  undefined. Standardized coefficient = coef * sd(x) / sd(dAP) on the estimation sample (demeaned for within).
  corr_IN is the correlation of x_I and x_N on the estimation sample after the specification's intercepts are
  partialled out (the Pearson correlation for both, within class for both+class), so it is the collinearity of the
  coefficients that row reports, and vif = 1 / (1 - corr_IN^2), for the specifications with both. loso_rmse: fit on
  two shifts, predict the third, RMSE pooled over every held-out observation (not for within).
- Intervals: replicate t takes test resample t of every run, and per shift redraws the sampling seeds with
  replacement from one draw array shared by every row (default_rng(--seed), one (replicates, seeds) block per shift in
  TARGETS order). The chains of the drawn seeds are taken whole, with their instance counts. Every row is refitted on
  the same replicate; undefined replicates are counted and left out of the 95% percentile interval, which is given
  only next to a point estimate.
- Reading (instance_summary.json), from the within/log coefficient of x_I: "at a fixed image step, more instances of a
  class give more AP for that class" when its 95% interval is above 0; "no detectable instance effect at a fixed image
  step" when it covers 0; "at a fixed image step, more instances of a class go with less AP gain for that class" when
  it is below 0 (no support for the density explanation). The pooled specifications, their VIF and the out-of-shift
  RMSE of instances against images are reported next to it without a verdict.
- Figure instance_recovery.png (+ instance_recovery.csv): per-class recovery (seed means; reference the E = 3 full-pool
  runs; undefined unless the denominator exceeds --min-denominator) against images (top row) and against instances of
  the class (bottom row), one line per shift. A curve with an undefined denominator is left out and named in its panel.

Outputs: instance_pool_density.csv, instance_counts.csv, instance_transitions.csv, instance_regression.csv,
instance_recovery.csv, instance_summary.json, instance_recovery.png. A table without rows writes no file.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from analyze_campaign import load, require_one_schedule, write_tables
from analyze_rq5 import select, shared_resamples
from bdd_pipeline.campaign import EXTRA_RUNGS, LADDER, POLICY, TARGETS
from bdd_pipeline.data import CLASS_TO_ID, HEIGHT, WIDTH, domain_rows, nested_rows, read_manifests, sanitize_labels
from bdd_pipeline.recovery import recovery
from prepare_revised_bdd100k import RELIABLE_CLASSES, stream_records

TRAIN_ANNOTATION = Path("annotations") / "bdd100k_labels_images_train.json"
DOMAINS = ("source", *TARGETS)
ID_TO_CLASS = {index: name for name, index in CLASS_TO_ID.items()}
LECTURER_PEDESTRIANS = {"source": (13170, 9804), "lighting": (12265, 18015)}  # (instances, images), 2026-11-24
FORMS = {"log": ("dln_instances", "dln_images"), "raw": ("d_instances", "d_images")}  # form -> (x_I, x_N)
SPECS = ("images", "instances", "both", "both+class", "within")
POOLED = "pooled"
ROWS = ([(spec, form, POOLED) for form in FORMS for spec in SPECS]
        + [("both", form, name) for form in FORMS for name in RELIABLE_CLASSES])
READINGS = {"above": "at a fixed image step, more instances of a class give more AP for that class",
            "covers": "no detectable instance effect at a fixed image step",
            "below": ("at a fixed image step, more instances of a class go with less AP gain for that class "
                      "(no support for the density explanation)"),
            None: "not determined: the within/log coefficient of x_I or its interval is undefined"}
# Reference categorical slots 1-3 (dataviz palette, as in analyze_rq5.py), with distinct markers as secondary encoding.
COLORS = {"compound": "#2a78d6", "weather": "#1baf7a", "lighting": "#eb6834"}
MARKERS = {"compound": "o", "weather": "^", "lighting": "s"}


def count_instances(manifests: dict, annotation: Path) -> dict[str, dict[str, int]]:
    """Image name -> kept boxes per reliable class, for every train-pool image of the source and the three targets."""
    if annotation.name != TRAIN_ANNOTATION.name:
        raise ValueError(f"Only the train annotation file may be read, not {annotation}")
    wanted = {row["name"] for domain in DOMAINS for row in domain_rows(manifests, "train", domain)}
    counts = {}
    for record in stream_records(annotation):
        name = record["name"]
        if name in wanted:
            if name in counts:
                raise ValueError(f"Duplicate annotation: {name}")
            _, categories, _ = sanitize_labels(record.get("labels", []), WIDTH, HEIGHT)
            kept = Counter(ID_TO_CLASS[category] for category in categories)
            counts[name] = {key: kept[key] for key in RELIABLE_CLASSES}
    missing = sorted(wanted - set(counts))
    if missing:
        raise ValueError(f"{len(missing)} train-pool images have no annotation record, e.g. {missing[0]}")
    return counts


def tally(rows: list[dict], counts: dict[str, dict[str, int]]) -> dict:
    """Images, and instances and positive images per reliable class, of a set of manifest rows."""
    return {"images": len(rows),
            "instances": {name: sum(counts[row["name"]][name] for row in rows) for name in RELIABLE_CLASSES},
            "positive": {name: sum(counts[row["name"]][name] > 0 for row in rows) for name in RELIABLE_CLASSES}}


def run_subset(manifests: dict, run: dict) -> list[dict]:
    """The images a fine-tune run trained on, rebuilt with the training code's call from its run fields."""
    return nested_rows(domain_rows(manifests, "train", run["domain"]), str(run["budget"]), int(run["seed"]))


def pool_density(manifests: dict, counts: dict[str, dict[str, int]]) -> list[dict]:
    rows = []
    for domain in DOMAINS:
        pool = tally(domain_rows(manifests, "train", domain), counts)
        rows += [{"domain": domain, "class": name, "images": pool["images"], "instances": pool["instances"][name],
                  "instances_per_image": pool["instances"][name] / pool["images"],
                  "positive_images": pool["positive"][name]} for name in RELIABLE_CLASSES]
    return rows


def shift_chains(entries: list[dict], manifests: dict, counts: dict[str, dict[str, int]], shift: str) -> dict[int, list]:
    """Sampling seed -> its chain of rungs, N = 0 (the source model) first, then by image count.

    A rung holds budget (as recorded), images, instances and positive images per class, the scored entry and the
    update count. Only LADDER + EXTRA_RUNGS budgets and "full" are rungs, so the matched-pool runs at 1,738 are not.
    """
    zero = [entry for entry in entries if entry["run"]["stage"] == "source" and entry["domain"] == shift]
    if len(zero) != 1:
        raise ValueError(f"{shift}: expected one source model scored on the shift, found {len(zero)}")
    allowed = {str(budget) for budget in LADDER + EXTRA_RUNGS.get(shift, ())} | {"full"}
    ladder = {str(budget): runs for budget, runs in select(entries, "fine-tune", shift, shift).items()
              if str(budget) in allowed}
    require_one_schedule(shift, {budget: list(runs.values()) for budget, runs in ladder.items()})
    epochs = {entry["run"].get("epochs") for runs in ladder.values() for entry in runs.values()}
    if epochs != {POLICY["epochs"]} or "full" not in ladder:
        raise ValueError(f"{shift}: the ladder needs fine-tune runs at E = {POLICY['epochs']} only, the full pool "
                         f"included; found epochs {sorted(epochs, key=str)} at budgets {sorted(ladder)}")
    empty = dict.fromkeys(RELIABLE_CLASSES, 0)
    start = {"budget": "0", "images": 0, "instances": empty, "positive": empty, "entry": zero[0], "updates": 0}
    chains = {}
    for seed in sorted({seed for runs in ladder.values() for seed in runs}):
        rungs = [{"budget": budget, **tally(run_subset(manifests, runs[seed]["run"]), counts), "entry": runs[seed],
                  "updates": runs[seed]["run"].get("total_updates")} for budget, runs in ladder.items() if seed in runs]
        chains[seed] = [start] + sorted(rungs, key=lambda rung: rung["images"])
    return chains


def ap(rung: dict, name: str, replicate: int | None) -> float | None:
    point = rung["entry"]["points"][name]
    return point if point is None or replicate is None else float(rung["entry"]["boots"][name][replicate])


def transitions(shift: str, seed: int, chain: list[dict], replicate: int | None = None) -> list[dict]:
    """One observation per consecutive pair of rungs and reliable class; AP from test resample `replicate` if given."""
    rows = []
    for low, high in zip(chain, chain[1:]):
        for name in RELIABLE_CLASSES:
            ap_from, ap_to = ap(low, name, replicate), ap(high, name, replicate)
            i_from, i_to = low["instances"][name], high["instances"][name]
            rows.append({"shift": shift, "seed": seed, "class": name, "budget_from": low["budget"],
                         "budget_to": high["budget"], "images_from": low["images"], "images_to": high["images"],
                         "instances_from": i_from, "instances_to": i_to, "updates_from": low["updates"],
                         "updates_to": high["updates"], "ap_from": ap_from, "ap_to": ap_to,
                         "d_ap": None if ap_from is None or ap_to is None else ap_to - ap_from,
                         "d_images": high["images"] - low["images"], "d_instances": i_to - i_from,
                         "dln_images": math.log(high["images"] / low["images"]) if low["images"] else None,
                         "dln_instances": math.log(i_to / i_from) if i_from else None, "from_zero": low["images"] == 0})
    return rows


def replicate_observations(chains: dict[str, dict[int, list]], draws: dict[str, np.ndarray], t: int) -> list[dict]:
    """Observations of replicate t: per shift the chains of the seeds drawn in row t (duplicates kept), test resample t."""
    sample = []
    for shift, by_seed in chains.items():
        seeds = sorted(by_seed)
        for index in draws[shift][t]:
            sample += transitions(shift, seeds[index], by_seed[seeds[index]], t)
    return sample


def cell(observation: dict) -> tuple:
    return observation["shift"], observation["class"], observation["budget_from"], observation["budget_to"]


def estimation_sample(sample: list[dict], spec: str, form: str, scope: str) -> tuple[list[dict], int]:
    """The observations a row is fitted on, and how many I_from = 0 observations the log form left out."""
    usable = [o for o in sample if not o["from_zero"] and o["d_ap"] is not None and scope in (POOLED, o["class"])]
    kept = usable if form == "raw" else [o for o in usable if o["instances_from"] > 0]
    dropped = len(usable) - len(kept)
    if spec == "within":
        sizes = Counter(cell(o) for o in kept)
        kept = [o for o in kept if sizes[cell(o)] >= 2]
    return kept, dropped


def demean(values: np.ndarray, codes: np.ndarray) -> np.ndarray:
    means = np.bincount(codes, weights=values) / np.bincount(codes)
    return values - means[codes]


def design(sample: list[dict], spec: str, form: str, classes: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Outcome, design matrix and the column of x_I / x_N in it (within: both sides demeaned within cells)."""
    key_i, key_n = FORMS[form]
    y = np.array([o["d_ap"] for o in sample], dtype=float)
    x_i = np.array([o[key_i] for o in sample], dtype=float)
    x_n = np.array([o[key_n] for o in sample], dtype=float)
    if spec == "within":
        index = {}
        codes = np.array([index.setdefault(cell(o), len(index)) for o in sample], dtype=int)
        return demean(y, codes), demean(x_i, codes)[:, None], {"x_I": 0}
    if spec == "both+class":
        base = np.array([[o["class"] == name for name in classes] for o in sample], dtype=float).reshape(-1, len(classes))
    else:
        base = np.ones((len(sample), 1))
    regressors = {"images": {"x_N": x_n}, "instances": {"x_I": x_i}}.get(spec, {"x_I": x_i, "x_N": x_n})
    columns = {name: base.shape[1] + j for j, name in enumerate(regressors)}
    return y, np.column_stack([base, *regressors.values()]), columns


def ols(y: np.ndarray, x: np.ndarray) -> np.ndarray | None:
    """Least-squares coefficients; None without observations, with a non-finite value or a rank-deficient design."""
    if not len(y) or not np.isfinite(y).all() or np.linalg.matrix_rank(x) < x.shape[1]:
        return None
    return np.linalg.lstsq(x, y, rcond=None)[0]


def fit_row(sample: list[dict], spec: str, form: str, scope: str) -> dict | None:
    """Coefficients of x_I and x_N, standardized coefficients, R^2 (within-R^2), corr_IN and VIF; None if undefined."""
    kept, _ = estimation_sample(sample, spec, form, scope)
    y, x, columns = design(kept, spec, form, sorted({o["class"] for o in kept}))
    coef = ols(y, x)
    if coef is None:
        return None
    residual = y - x @ coef
    total, sd_y = float(((y - y.mean()) ** 2).sum()), float(y.std())
    out = {"r2": 1 - float(residual @ residual) / total if total > 0 else None, "corr_IN": None, "vif": None}
    for name, suffix in (("x_I", "I"), ("x_N", "N")):
        j = columns.get(name)
        out[f"coef_{suffix}"] = None if j is None else float(coef[j])
        out[f"std_coef_{suffix}"] = None if j is None or sd_y == 0 else float(coef[j] * x[:, j].std() / sd_y)
    if len(columns) == 2:  # x_I and x_N after the intercept columns in front of them are partialled out
        base, pair = x[:, :columns["x_I"]], x[:, [columns["x_I"], columns["x_N"]]]
        rest = pair - base @ np.linalg.lstsq(base, pair, rcond=None)[0]
        r = float(np.corrcoef(rest[:, 0], rest[:, 1])[0, 1])
        out.update({"corr_IN": r, "vif": 1 / (1 - r * r) if r * r < 1 else None})
    return out


def out_of_shift_rmse(sample: list[dict], spec: str, form: str, scope: str) -> float | None:
    """Fit on two shifts, predict the third; RMSE over every held-out observation. None for within or any failed fit."""
    if spec == "within":
        return None
    kept, _ = estimation_sample(sample, spec, form, scope)
    classes, errors = sorted({o["class"] for o in kept}), []
    for shift in sorted({o["shift"] for o in kept}):
        y, x, _ = design([o for o in kept if o["shift"] != shift], spec, form, classes)
        coef = ols(y, x)
        if coef is None:
            return None
        y, x, _ = design([o for o in kept if o["shift"] == shift], spec, form, classes)
        errors.append(y - x @ coef)
    return float(np.sqrt(np.mean(np.concatenate(errors) ** 2))) if errors else None


def seed_draws(chains: dict[str, dict[int, list]], replicates: int, seed: int) -> dict[str, np.ndarray]:
    """One (replicates, seeds) array of drawn seed indices per shift, in TARGETS order from default_rng(seed)."""
    rng = np.random.default_rng(seed)
    return {shift: rng.integers(0, len(chains[shift]), (replicates, len(chains[shift]))) for shift in TARGETS}


def regression_rows(observations: list[dict], chains: dict[str, dict[int, list]], draws: dict[str, np.ndarray]) -> list[dict]:
    """instance_regression.csv: every row of ROWS, point fit plus intervals refitted on the same replicates."""
    replicates = len(draws[TARGETS[0]])
    fits = {row: [] for row in ROWS}
    for t in range(replicates):
        sample = replicate_observations(chains, draws, t)  # one draw per replicate, shared by every row
        for row in ROWS:
            fits[row].append(fit_row(sample, *row))
    rows = []
    for spec, form, scope in ROWS:
        kept, dropped = estimation_sample(observations, spec, form, scope)
        point = fit_row(observations, spec, form, scope) or {}
        defined = [fit for fit in fits[(spec, form, scope)] if fit is not None]
        row = {"spec": spec, "form": form, "class": scope, "n_obs": len(kept), "dropped_zero_instances": dropped}
        for suffix in ("I", "N"):
            estimate = point.get(f"coef_{suffix}")
            values = [fit[f"coef_{suffix}"] for fit in defined] if estimate is not None else []
            row.update({f"coef_{suffix}": estimate,
                        f"coef_{suffix}_ci_low": float(np.percentile(values, 2.5)) if values else None,
                        f"coef_{suffix}_ci_high": float(np.percentile(values, 97.5)) if values else None})
        row.update({name: point.get(name) for name in ("std_coef_I", "std_coef_N", "r2", "corr_IN", "vif")})
        row.update({"r2_kind": "within" if spec == "within" else "ols",
                    "loso_rmse": out_of_shift_rmse(observations, spec, form, scope), "replicates": replicates,
                    "undefined_replicates": replicates - len(defined)})
        rows.append(row)
    return rows


def instance_reading(row: dict) -> str:
    low, high = row["coef_I_ci_low"], row["coef_I_ci_high"]
    if low is None or high is None:
        return READINGS[None]
    return READINGS["above" if low > 0 else "below" if high < 0 else "covers"]


def recovery_points(chains: dict[str, dict[int, list]], min_denominator: float) -> list[dict]:
    """Seed-mean recovery per shift, class and rung against the E = 3 full-pool runs, with images and mean instances."""
    rows = []
    for shift, by_seed in chains.items():
        rungs = defaultdict(list)
        for chain in by_seed.values():
            for rung in chain[1:]:
                rungs[rung["budget"]].append(rung)
        zero = next(iter(by_seed.values()))[0]["entry"]
        for name in RELIABLE_CLASSES:
            values = {budget: [rung["entry"]["points"][name] for rung in group] for budget, group in rungs.items()}
            zero_point = zero["points"][name]
            ceiling = None if None in values["full"] else mean(values["full"])
            if zero_point is None or ceiling is None or recovery(ceiling, zero_point, ceiling, min_denominator) is None:
                note = ("undefined: no AP for the class" if zero_point is None or ceiling is None
                        else f"undefined: ceiling gain {ceiling - zero_point:.4f} <= {min_denominator}")
                rows.append({"shift": shift, "class": name, "budget": None, "recovery": None, "note": note})
                continue
            for budget, group in sorted(rungs.items(), key=lambda item: item[1][0]["images"]):
                score = None if None in values[budget] else mean(values[budget])
                rows.append({"shift": shift, "class": name, "budget": budget, "images": group[0]["images"],
                             "instances": mean(rung["instances"][name] for rung in group), "seeds": len(group),
                             "mAP_mean": score, "zero_shot": zero_point, "ceiling": ceiling,
                             "recovery": None if score is None else recovery(score, zero_point, ceiling, min_denominator)})
    return rows


def plot(points: list[dict], output: Path) -> Path:
    """instance_recovery.png: recovery against images (top) and instances of the class (bottom), one line per shift."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    fig, axes = plt.subplots(2, len(RELIABLE_CLASSES), figsize=(16, 7.5), sharey=True)
    for column, name in enumerate(RELIABLE_CLASSES):
        undefined = [p["shift"] for p in points if p["class"] == name and p["budget"] is None]
        for row, key, label in ((0, "images", "Labeled target images N"), (1, "instances", f"Labeled {name} instances")):
            ax = axes[row, column]
            for shift in TARGETS:
                rungs = [p for p in points if p["shift"] == shift and p["class"] == name and p["budget"] is not None
                         and p["recovery"] is not None and p[key] > 0]
                if rungs:
                    ax.plot([p[key] for p in rungs], [p["recovery"] for p in rungs], color=COLORS[shift], lw=2,
                            marker=MARKERS[shift], ms=8, mec="white", mew=1.2, label=shift)
            ax.axhline(1.0, color="#6b6b66", ls="--", lw=1)
            ax.axhline(0.0, color="#6b6b66", lw=0.8)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:,.0f}"))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            ax.set_xlabel(label + ("\nundefined denominator: " + ", ".join(undefined) if undefined else ""))
            ax.grid(axis="y", color="#e6e6e3", lw=0.8)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        axes[0, column].set_title(name)
    for ax in axes[:, 0]:
        ax.set_ylabel("Recovery (E = 3 full pool = 1)")
    legend = {}
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            legend.setdefault(label, handle)
    fig.legend(legend.values(), legend.keys(), loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Per-class recovery against images and against instances of the class (seed means)")
    fig.tight_layout(rect=(0, 0, 0.94, 0.96))
    path = output / "instance_recovery.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scored", type=Path, required=True, help="evaluate_frcnn.py --final-plan output of the campaign")
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Revised dataset root (manifests/ and the train annotation file are read)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-denominator", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0, help="Seed for redrawing sampling seeds")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    entries = load(args.scored)
    shared_resamples([json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.scored.glob("*/*/metrics.json"))])
    _, manifests = read_manifests(args.dataset_root)
    counts = count_instances(manifests, args.dataset_root / TRAIN_ANNOTATION)
    density = pool_density(manifests, counts)
    chains = {shift: shift_chains(entries, manifests, counts, shift) for shift in TARGETS}
    lengths = {len(rung["entry"]["boots"][name]) for by_seed in chains.values() for chain in by_seed.values()
               for rung in chain for name in RELIABLE_CLASSES}
    if len(lengths) != 1:
        raise ValueError(f"Bootstrap vectors of different lengths {sorted(lengths)}: replicate t would not be shared")
    observations = [o for shift, by_seed in chains.items() for seed, chain in by_seed.items()
                    for o in transitions(shift, seed, chain)]
    regressions = regression_rows(observations, chains, seed_draws(chains, lengths.pop(), args.seed))
    table = []
    for o in observations:
        usable = not o["from_zero"] and o["d_ap"] is not None
        table.append({**o, "in_regressions": usable, "in_log_form": usable and o["instances_from"] > 0})
    counted = [{"shift": shift, "seed": seed, "budget": rung["budget"], "images": rung["images"], "class": name,
                "instances": rung["instances"][name], "positive_images": rung["positive"][name]}
               for shift, by_seed in chains.items() for seed, chain in by_seed.items() for rung in chain[1:]
               for name in RELIABLE_CLASSES]
    points = recovery_points(chains, args.min_denominator)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tables(args.output_dir, {"instance_pool_density": density, "instance_counts": counted,
                                   "instance_transitions": table, "instance_regression": regressions,
                                   "instance_recovery": points})
    within = next(row for row in regressions if (row["spec"], row["form"], row["class"]) == ("within", "log", POOLED))
    raw_both = next(row for row in regressions if (row["spec"], row["form"], row["class"]) == ("both", "raw", POOLED))
    log_both = next(row for row in regressions if (row["spec"], row["form"], row["class"]) == ("both", "log", POOLED))
    summary = {"rule": f"within/log coefficient of x_I: '{READINGS['above']}' when its 95% interval is above 0, "
                       f"'{READINGS['covers']}' when it covers 0, '{READINGS['below']}' when it is below 0 "
                       "(plan 2026-11-24, section 1; descriptive)",
               "reading": instance_reading(within), "within_log": within,
               "pooled": [row for row in regressions if row["class"] == POOLED and row["spec"] != "within"],
               "out_of_shift_rmse": {form: {spec: next(row["loso_rmse"] for row in regressions
                                                       if (row["spec"], row["form"], row["class"]) == (spec, form, POOLED))
                                            for spec in ("instances", "images")} for form in FORMS},
               "observations": {"transitions": sum(not o["from_zero"] for o in observations) // len(RELIABLE_CLASSES),
                                "from_zero_transitions": sum(o["from_zero"] for o in observations) // len(RELIABLE_CLASSES),
                                "raw_form": raw_both["n_obs"], "log_form": log_both["n_obs"],
                                "dropped_zero_instances_log": log_both["dropped_zero_instances"]},
               "pedestrian_density": {row["domain"]: {key: row[key] for key in ("instances", "images", "instances_per_image")}
                                      for row in density if row["class"] == "pedestrian"},
               "lecturer_pedestrian_density": {domain: {"instances": i, "images": n} for domain, (i, n)
                                               in LECTURER_PEDESTRIANS.items()},
               "seed": args.seed, "min_denominator": args.min_denominator}
    (args.output_dir / "instance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.no_plots:
        plot(points, args.output_dir)
    for row in density:
        if row["class"] == "pedestrian" and row["domain"] in LECTURER_PEDESTRIANS:
            instances, images = LECTURER_PEDESTRIANS[row["domain"]]
            print(f"pedestrian, {row['domain']} train pool: {row['instances']:,} / {row['images']:,} = "
                  f"{row['instances_per_image']:.2f} per image (lecturer: {instances:,} / {images:,} = "
                  f"{instances / images:.2f})")
    print(f"{summary['observations']['transitions']} transitions, {raw_both['n_obs']} observations in the raw form, "
          f"{log_both['n_obs']} in the log form -> {instance_reading(within)}: {args.output_dir}")


if __name__ == "__main__":
    main()
