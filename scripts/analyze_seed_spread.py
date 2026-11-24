"""Seed spread against N: how far one random draw of N labeled target images strays from another (direction 1.2).

Input is the scored FT campaign, the output directory of `evaluate_frcnn.py --final-plan`. Rules: section "2. Seed
spread against N" of .agent/plans/active/2026-11-24-direction-1-analyses.md, written before this analysis was run but
after the curves had been seen, so descriptive, not confirmatory. Runs are told apart by their run fields only, never
by folder names. The SD, floor and slope intervals are analytic; the intervals of the variance model come from a
seeded parametric bootstrap (--seed).

- Ladder: the fine-tune runs scored on their own shift with the locked epoch count (POLICY["epochs"], 3) at a budget
  of LADDER, plus EXTRA_RUNGS for lighting. The matched-pool runs at 1,738 of lighting and weather are not rungs, and
  the full pool is the floor below, not a rung. Keys: reliable mAP50:95 and the AP50:95 of the four reliable classes
  (not the ten-class mean). A rung enters a key with the sampling seeds that have a value of it, at least two.
- seed_spread.csv, per shift, key and rung: the sample SD over the sampling seeds (ddof 1) with its 95% interval from
  the chi-square distribution with nu = seeds - 1, [sqrt(nu s^2 / chi2_0.975(nu)), sqrt(nu s^2 / chi2_0.025(nu))];
  the SD in recovery units, SD / (seed-mean E = 3 full-pool AP - source model AP on the shift), only where that
  denominator exceeds --min-denominator (as bdd_pipeline.recovery); the finite-population factor (M - N) / (M - 1), M = pool size.
- seed_spread_floor.csv, the training-noise floor: the E = 3 full-pool runs (seeds 0 and 1) train on the same images,
  so their SD (nu = 1) is training noise only (order, flips, GPU nondeterminism). A shift without exactly these two
  runs stops the script, since they are also the recovery denominator. Pooled over the shifts per key:
  s^2 = sum(nu s^2) / sum(nu) with nu = sum(nu), for two seeds per shift the mean of the variances with nu = 3.
- seed_spread_fit.csv, on the rungs (never the full pool).
  Models "power_fpc" (primary) and "power", per shift and pooled over the shifts (shift intercepts, one slope):
  ln s = a + b ln N by weighted least squares. The response is ln s minus its bias (psi(nu/2) - ln(nu/2)) / 2; its
  known variance psi'(nu/2) / 4 sets the weights. "power_fpc" also subtracts ln(fpc) / 2: the subsets are drawn
  without replacement from a finite pool, so pure random sampling gives s ~ sqrt(fpc / N), and only after this
  correction is its slope -1/2 (without it, -0.54 to -0.58 on these ladders). "power" is the uncorrected 1/sqrt(N)
  reading of the review, for reference. The slope has a normal 95% interval from (X'WX)^-1, and chi2_df =
  sum(w r^2) / (rungs - parameters) flags misfit (about 1 when the model holds). The interval is approximate: it
  treats the rungs as independent, but the rungs of one seed are nested subsets (on synthetic nested data its
  coverage is about 89%, not 95%). Flags: slope_covers_minus_half, slope_below_zero (upper bound below 0). A rung with
  SD 0 has no logarithm and is left out (rungs_zero_sd). A per-shift fit needs 3 rungs, the pooled fit more rungs than
  parameters + 1; otherwise there is no row.
  Model "variance", per shift only: s^2 = sigma2 + c (1/N - 1/M), "sampling variance proportional to 1/N plus a floor"
  in its finite-population form. Ordinary least squares, then one weighted refinement with weights nu / (2 f^2), f the
  fitted s^2 clipped below at CLIP_SHARE of the mean observed s^2; 3 rungs needed. Reported: sigma2 as fitted,
  sigma_floor = sqrt(max(sigma2, 0)), c, N* = 1 / (sigma2 / c + 1/M), the budget where both parts are equal (only
  when sigma2 > 0 and c > 0), and sigma_floor over the training-noise SD of the shift and over the pooled one.
  Intervals by a parametric bootstrap (SIMULATIONS draws, generator row_rng(--seed + VARIANCE_SEED_OFFSET, shift
  index, key index)): s^2 of each rung redrawn as f chi2_nu / nu around the fitted f, the model refitted; percentile
  intervals for sigma_floor and c, and for N* over the draws where it exists (n_star_undefined counts the others).
  Like the slope interval it treats the rungs as independent. floor_above_training_noise: the lower bound of
  sigma_floor is above the upper bound of the pooled training-noise SD ("clearly above" in the reading).
- seed_spread_summary.json: the reading, the assumption and the limit below, and the reliable floor and fit rows.
- seed_spread_reliable.png: one panel per shift, SD against N on log axes with intervals, the fitted power law drawn
  at the scale of the observed SDs (the mean bias of ln s over the rungs added back to the intercept), a -1/2
  reference line through the centre of the points, the pooled training-noise band and the full-pool point.
  seed_spread_classes.png: the same per shift and reliable class.
- Quantiles without scipy: chi-square in closed form for nu = 1 (z_{(1+q)/2}^2) and nu = 2 (-2 ln(1 - q)), and for
  nu >= 3 by bisection on the exact CDF (regularized incomplete gamma by recurrence); digamma and trigamma at
  multiples of 1/2 by recurrence from 1/2 and 1.
- A table without rows writes no file and removes one of an earlier run.

Assumption: at a rung the seed changes the subset AND the training randomness, so SD^2 ~ sampling + training noise;
that the training noise does not depend on N is assumed. What the power-law fit takes for granted (independent rungs,
an unlimited pool behind the -1/2 reference): ASSUMPTION below. Reading and limit: READING and LIMIT below.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist, mean, stdev

import numpy as np

from analyze_campaign import LADDER_EPOCHS, PALETTE, budget_of, load, require_ceiling_seeds, write_tables
from analyze_rq5 import row_rng, seeds_of
from bdd_pipeline.campaign import EXTRA_RUNGS, LADDER, TARGETS
from bdd_pipeline.data import EXPECTED_DOMAINS, RELIABLE_CLASSES

KEYS = ("reliable",) + RELIABLE_CLASSES
EULER = 0.5772156649015329  # Euler-Mascheroni constant: psi(1) = -EULER
Z95 = NormalDist().inv_cdf(0.975)
MIN_RUNGS = 3  # per-shift fits; the pooled power law needs more rungs than parameters + 1
CLIP_SHARE = 0.01  # fitted s^2 of the weighted refinement is clipped below at this share of the mean observed s^2
SIMULATIONS = 1000  # parametric bootstrap draws of the variance model
VARIANCE_SEED_OFFSET = 40  # its generators: row_rng(--seed + VARIANCE_SEED_OFFSET, shift index, key index)
RULES = ".agent/plans/active/2026-11-24-direction-1-analyses.md, section 2 (descriptive, not confirmatory)"
ASSUMPTION = ("At a ladder rung the seed changes the subset and the training randomness, so SD^2 ~ sampling + training "
              "noise. That the training noise does not depend on N is an assumption. The power-law fit treats the rungs "
              "as independent, but the rungs of one seed are nested subsets, so the SDs at neighbouring rungs are "
              "correlated and the slope interval is approximate, likely too narrow (synthetic nested data: about 89% "
              "coverage). The same holds for the parametric intervals of the variance model. The -1/2 reference of "
              "'power' is that of an unlimited pool; 'power_fpc' corrects for drawing from the finite pool (fpc column), "
              "under which pure random sampling has slope -1/2 again.")
READING = ("A power_fpc slope interval that covers -1/2 and lies below 0 means the spread shrinks as averaging over randomly "
           "drawn images predicts. A fitted floor at the level of the training noise means the spread left at large N is "
           "training noise, which no choice of images removes; a floor clearly above it (floor_above_training_noise: the "
           "lower bound of sigma_floor above the upper bound of the pooled training-noise SD) means some "
           "subset-dependent spread remains: room for selection.")
LIMIT = ("The SD over random subsets says how far one random draw strays from another. It does not bound what an informed "
         "selector can reach, because active selection can beat every random subset: the result answers 'does the "
         "choice of random subset matter', not 'is active learning useless'.")


def digamma_trigamma(x: float) -> tuple[float, float]:
    """psi(x) and psi'(x) for x a positive multiple of 1/2, by recurrence up from x = 1/2 or x = 1.

    psi(1/2) = -gamma - 2 ln 2, psi(1) = -gamma, psi'(1/2) = pi^2 / 2, psi'(1) = pi^2 / 6;
    psi(x + 1) = psi(x) + 1/x and psi'(x + 1) = psi'(x) - 1/x^2.
    """
    if x <= 0 or (2 * x) % 1:
        raise ValueError(f"x must be a positive multiple of 1/2, got {x}")
    start = x % 1 or 1.0
    psi, trigamma = (-EULER - 2 * math.log(2), math.pi ** 2 / 2) if start == 0.5 else (-EULER, math.pi ** 2 / 6)
    while start < x:
        psi, trigamma, start = psi + 1 / start, trigamma - 1 / start ** 2, start + 1
    return psi, trigamma


def chi2_cdf(x: float, nu: int) -> float:
    """P(chi2_nu <= x) for integer nu: the regularized lower incomplete gamma P(nu/2, x/2), by recurrence up from
    P(1/2, y) = erf(sqrt(y)) or P(1, y) = 1 - exp(-y), with P(a + 1, y) = P(a, y) - y^a exp(-y) / Gamma(a + 1)."""
    y = x / 2
    a, p = (0.5, math.erf(math.sqrt(y))) if nu % 2 else (1.0, -math.expm1(-y))
    while a < nu / 2:
        p -= y ** a * math.exp(-y) / math.gamma(a + 1)
        a += 1
    return p


def chi2_quantile(q: float, nu: int) -> float:
    """Quantile q of the chi-square distribution: closed form for nu = 1 and 2, bisection on chi2_cdf above."""
    if nu < 1:
        raise ValueError(f"nu must be at least 1, got {nu}")
    if nu == 1:
        return NormalDist().inv_cdf((1 + q) / 2) ** 2
    if nu == 2:
        return -2 * math.log(1 - q)
    low, high = 0.0, float(nu)
    while chi2_cdf(high, nu) < q:
        high *= 2
    for _ in range(100):
        middle = (low + high) / 2
        low, high = (middle, high) if chi2_cdf(middle, nu) < q else (low, middle)
    return (low + high) / 2


def sd_row(sd: float, nu: int) -> dict:
    """An SD with nu degrees of freedom and its 95% chi-square interval."""
    return {"sd": sd, "nu": nu, "sd_ci_low": math.sqrt(nu * sd ** 2 / chi2_quantile(0.975, nu)),
            "sd_ci_high": math.sqrt(nu * sd ** 2 / chi2_quantile(0.025, nu))}


def seed_spread(runs: dict[int, dict], key: str) -> dict | None:
    """Sample SD (ddof 1) of one key over the seeds of `runs` (seed -> entry) that have a value; None below two."""
    values = {seed: entry["points"][key] for seed, entry in sorted(runs.items()) if entry["points"][key] is not None}
    if len(values) < 2:
        return None
    return {"seeds": len(values), "seed_list": " ".join(str(seed) for seed in values), "mean_AP": mean(values.values()),
            **sd_row(stdev(values.values()), len(values) - 1)}


def pooled_floor(rows: list[dict]) -> dict:
    """Training-noise SD pooled over floor rows: s^2 = sum(nu s^2) / sum(nu), nu = sum(nu)."""
    nu = sum(row["nu"] for row in rows)
    return {"shifts": " ".join(row["domain"] for row in rows),
            **sd_row(math.sqrt(sum(row["nu"] * row["sd"] ** 2 for row in rows) / nu), nu)}


def fine_tune_runs(entries: list[dict], domain: str) -> tuple[dict[int, dict[int, dict]], dict[int, dict]]:
    """Ladder rungs (budget -> seed -> entry) and full-pool runs (seed -> entry) of one shift.

    Fine-tune runs of the shift scored on its own test set with the locked epoch count; rungs are the budgets of LADDER
    and EXTRA_RUNGS, so the matched-pool runs at 1,738 of lighting and weather are none. A seed twice is an error.
    """
    groups = defaultdict(list)
    for entry in entries:
        run = entry["run"]
        if (run["stage"] == "fine-tune" and run["domain"] == domain and entry["domain"] == domain
                and run.get("epochs") == LADDER_EPOCHS):
            groups[budget_of(run)].append(entry)
    rungs = set(LADDER) | set(EXTRA_RUNGS.get(domain, ()))
    ladder = {budget: seeds_of(groups[budget]) for budget in sorted(groups) if budget in rungs}
    return ladder, seeds_of(groups.get(EXPECTED_DOMAINS[domain]["train"], []))


def spread_rows(domain: str, key: str, ladder: dict[int, dict[int, dict]], denominator: float | None,
                min_denominator: float) -> list[dict]:
    """seed_spread.csv rows of one shift and key; a rung with fewer than two seeds with a value has none."""
    pool = EXPECTED_DOMAINS[domain]["train"]
    usable = denominator is not None and denominator > min_denominator
    rows = []
    for budget, runs in sorted(ladder.items()):
        spread = seed_spread(runs, key)
        if spread:
            rows.append({"domain": domain, "class": key, "budget": budget, **spread, "recovery_denominator": denominator,
                         "sd_recovery": spread["sd"] / denominator if usable else None,
                         "fpc": (pool - budget) / (pool - 1)})
    return rows


def power_law_fit(rungs: list[dict], finite_pool: bool = False) -> dict | None:
    """ln s = a_shift + b ln N by weighted least squares over spread rows (domain, budget, sd, nu; fpc if finite_pool).

    One intercept per shift in `rungs`; response ln s - (psi(nu/2) - ln(nu/2)) / 2, minus ln(fpc) / 2 with
    finite_pool, weights 4 / psi'(nu/2). Rows with SD 0 are left out. None below MIN_RUNGS rungs for one shift, or at
    most parameters + 1 rungs pooled.
    """
    used = [row for row in rungs if row["sd"] > 0]
    shifts = list(dict.fromkeys(row["domain"] for row in used))
    parameters = len(shifts) + 1
    if len(used) < (MIN_RUNGS if len(shifts) == 1 else parameters + 2):
        return None
    psi, trigamma = np.array([digamma_trigamma(row["nu"] / 2) for row in used]).T
    nu = np.array([row["nu"] for row in used], dtype=float)
    y = np.log([row["sd"] for row in used]) - (psi - np.log(nu / 2)) / 2
    if finite_pool:  # pure random sampling from the finite pool: s ~ sqrt(fpc / N), so the slope is -1/2 again
        y -= np.log([row["fpc"] for row in used]) / 2
    w = 4 / trigamma
    x = np.column_stack([[float(row["domain"] == shift) for row in used] for shift in shifts]
                        + [np.log([row["budget"] for row in used])])
    covariance = np.linalg.inv(x.T @ (w[:, None] * x))
    beta = covariance @ (x.T @ (w * y))
    residual = y - x @ beta
    slope, se = float(beta[-1]), math.sqrt(covariance[-1, -1])
    low, high = slope - Z95 * se, slope + Z95 * se
    return {"rungs": len(used), "rungs_zero_sd": len(rungs) - len(used), "parameters": parameters,
            "intercept": float(beta[0]) if len(shifts) == 1 else None, "slope": slope, "slope_se": se,
            "slope_ci_low": low, "slope_ci_high": high, "chi2_df": float(w @ residual ** 2) / (len(used) - parameters),
            "slope_covers_minus_half": low <= -0.5 <= high, "slope_below_zero": high < 0}


def variance_model(rungs: list[dict], pool: int) -> dict | None:
    """s^2 = sigma2 + c (1/N - 1/M) over spread rows of one shift: OLS, then one WLS step with weights nu / (2 f^2)."""
    if len(rungs) < MIN_RUNGS:
        return None
    x = np.column_stack([np.ones(len(rungs)), [1 / row["budget"] - 1 / pool for row in rungs]])
    y = np.array([row["sd"] ** 2 for row in rungs])
    nu = np.array([row["nu"] for row in rungs], dtype=float)
    fitted = np.maximum(x @ np.linalg.lstsq(x, y, rcond=None)[0], CLIP_SHARE * y.mean())
    w = nu / (2 * fitted ** 2)
    sigma2, c = (float(value) for value in np.linalg.solve(x.T @ (w[:, None] * x), x.T @ (w * y)))
    return {"rungs": len(rungs), "sigma2": sigma2, "sigma_floor": math.sqrt(max(sigma2, 0.0)), "c": c,
            "n_star": 1 / (sigma2 / c + 1 / pool) if sigma2 > 0 and c > 0 else None}


def variance_interval(rungs: list[dict], pool: int, model: dict, rng: np.random.Generator) -> dict:
    """Parametric bootstrap of variance_model: each rung's s^2 redrawn SIMULATIONS times as f chi2_nu / nu around the
    fitted f (clipped as in the fit), the model refitted; 95% percentile intervals of sigma_floor, c and N*."""
    x = np.array([1 / row["budget"] - 1 / pool for row in rungs])
    y = np.array([row["sd"] ** 2 for row in rungs])
    nu = np.array([row["nu"] for row in rungs], dtype=float)
    fitted = np.maximum(model["sigma2"] + model["c"] * x, CLIP_SHARE * y.mean())
    floors, slopes, stars = [], [], []
    for _ in range(SIMULATIONS):
        draw = fitted * rng.chisquare(nu) / nu
        refit = variance_model([{**row, "sd": math.sqrt(value)} for row, value in zip(rungs, draw)], pool)
        floors.append(refit["sigma_floor"])
        slopes.append(refit["c"])
        if refit["n_star"] is not None:
            stars.append(refit["n_star"])
    columns = {"simulations": SIMULATIONS, "n_star_undefined": SIMULATIONS - len(stars)}
    for name, values in (("sigma_floor", floors), ("c", slopes), ("n_star", stars)):
        columns[f"{name}_ci_low"] = float(np.percentile(values, 2.5)) if values else None
        columns[f"{name}_ci_high"] = float(np.percentile(values, 97.5)) if values else None
    return columns


def floor_ratio(model: dict, floor: dict | None) -> float | None:
    """sigma_floor of a variance model over a measured training-noise SD; None without a positive one."""
    return model["sigma_floor"] / floor["sd"] if floor and floor["sd"] > 0 else None


def find(rows: list[dict], domain: str, key: str, model: str | None = None) -> dict | None:
    return next((row for row in rows if row["domain"] == domain and row["class"] == key
                 and (model is None or row["model"] == model)), None)


def draw_panel(ax, domain: str, key: str, spread: list[dict], floors: list[dict], fits: list[dict]) -> None:
    """SD against N with intervals, the fitted power law, the -1/2 reference, the training-noise band and the full pool."""
    color = PALETTE[key]
    rows = [row for row in spread if row["domain"] == domain and row["class"] == key and row["sd"] > 0]
    pooled, own, fit = find(floors, "pooled", key), find(floors, domain, key), find(fits, domain, key, "power")
    corrected = find(fits, domain, key, "power_fpc")
    if pooled:
        ax.axhspan(pooled["sd_ci_low"], pooled["sd_ci_high"], color="gray", alpha=0.12, lw=0,
                   label=f"training noise, pooled (nu = {pooled['nu']})")
        ax.axhline(pooled["sd"], color="gray", ls=":", lw=1)
    if rows:
        x, y = np.array([row["budget"] for row in rows]), np.array([row["sd"] for row in rows])
        errors = [y - [row["sd_ci_low"] for row in rows], [row["sd_ci_high"] for row in rows] - y]
        ax.errorbar(x, y, yerr=errors, fmt="o", ms=5, color=color, capsize=2, label="SD over sampling seeds")
        grid = np.geomspace(x.min(), x.max(), 50)
        centre = np.exp(np.log(x).mean()), np.exp(np.log(y).mean())
        ax.plot(grid, centre[1] * (grid / centre[0]) ** -0.5, color="gray", ls="--", lw=1, label="slope -1/2")
        if fit:  # at the scale of the observed SDs: the intercept estimates ln sigma, so the bias of ln s is added back
            bias = mean((digamma_trigamma(row["nu"] / 2)[0] - math.log(row["nu"] / 2)) / 2 for row in rows)
            label = f"fit: slope {fit['slope']:.2f} [{fit['slope_ci_low']:.2f}, {fit['slope_ci_high']:.2f}]"
            if corrected:
                label += (f"\nfinite-pool corrected: {corrected['slope']:.2f} [{corrected['slope_ci_low']:.2f}, "
                          f"{corrected['slope_ci_high']:.2f}]")
            ax.plot(grid, np.exp(fit["intercept"] + bias + fit["slope"] * np.log(grid)), color=color, lw=1.8, label=label)
    if own and own["sd"] > 0:
        ax.errorbar([own["budget"]], [own["sd"]], yerr=[[own["sd"] - own["sd_ci_low"]], [own["sd_ci_high"] - own["sd"]]],
                    fmt="D", ms=6, mfc="white", mec=color, color=color, capsize=2, label="full pool: training noise")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{domain}: {key}", fontsize=10)
    ax.set_xlabel("Labeled target images N")
    ax.set_ylabel("SD of AP50:95 over seeds")
    ax.legend(frameon=False, fontsize=7)


def plot(spread: list[dict], floors: list[dict], fits: list[dict], output: Path) -> None:
    """seed_spread_reliable.png (one panel per shift) and seed_spread_classes.png (shift x reliable class)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, keys, shape, size in (("reliable", ("reliable",), (1, len(TARGETS)), (14, 4.4)),
                                    ("classes", RELIABLE_CLASSES, (len(TARGETS), len(RELIABLE_CLASSES)), (18, 12))):
        fig, axes = plt.subplots(*shape, figsize=size, squeeze=False, layout="constrained")
        for ax, (domain, key) in zip(axes.flat, itertools.product(TARGETS, keys)):
            draw_panel(ax, domain, key, spread, floors, fits)
        fig.suptitle("Seed spread against N (95% chi-square intervals; band: pooled training noise)")
        fig.savefig(output / f"seed_spread_{name}.png", dpi=160)
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scored", type=Path, required=True,
                        help="evaluate_frcnn.py --final-plan output directory of the FT campaign")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-denominator", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0, help="Seed of the parametric bootstrap of the variance model")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    entries = load(args.scored)
    source_model = {entry["domain"]: entry for entry in entries if entry["run"]["stage"] == "source"}
    missing = [domain for domain in TARGETS if domain not in source_model]
    if missing:
        raise ValueError(f"The source model is not scored on {', '.join(missing)}")
    spread, floors, fits = [], [], []
    for domain in TARGETS:
        ladder, full = fine_tune_runs(entries, domain)
        require_ceiling_seeds(domain, f"{LADDER_EPOCHS}-epoch full pool (training-noise floor, recovery denominator)",
                              list(full.values()))
        for key in KEYS:
            values, zero = [entry["points"][key] for entry in full.values()], source_model[domain]["points"][key]
            denominator = mean(values) - zero if values and None not in values and zero is not None else None
            spread += spread_rows(domain, key, ladder, denominator, args.min_denominator)
            floor = seed_spread(full, key)
            if floor:
                floors.append({"domain": domain, "class": key, "budget": EXPECTED_DOMAINS[domain]["train"], **floor})
    for key in KEYS:
        own = [row for row in floors if row["class"] == key]
        if own:
            floors.append({"domain": "pooled", "class": key, "budget": None, **pooled_floor(own)})
        pooled = find(floors, "pooled", key)
        for domain in TARGETS:
            rungs = [row for row in spread if row["domain"] == domain and row["class"] == key]
            for name, finite_pool in (("power_fpc", True), ("power", False)):
                fit = power_law_fit(rungs, finite_pool)
                if fit:
                    fits.append({"domain": domain, "class": key, "model": name, **fit})
            pool = EXPECTED_DOMAINS[domain]["train"]
            model = variance_model(rungs, pool)
            if model:
                bounds = variance_interval(rungs, pool, model, row_rng(args.seed + VARIANCE_SEED_OFFSET,
                                                                       TARGETS.index(domain), KEYS.index(key)))
                fits.append({"domain": domain, "class": key, "model": "variance", **model, **bounds,
                             "floor_ratio_shift": floor_ratio(model, find(floors, domain, key)),
                             "floor_ratio_pooled": floor_ratio(model, pooled),
                             "floor_above_training_noise": (bounds["sigma_floor_ci_low"] > pooled["sd_ci_high"]
                                                            if pooled else None)})
        for name, finite_pool in (("power_fpc", True), ("power", False)):
            fit = power_law_fit([row for row in spread if row["class"] == key], finite_pool)
            if fit:
                fits.append({"domain": "pooled", "class": key, "model": name, **fit})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tables(args.output_dir, {"seed_spread": spread, "seed_spread_floor": floors, "seed_spread_fit": fits})
    summary = {"rules": RULES, "min_denominator": args.min_denominator, "seed": args.seed, "reading": READING,
               "assumption": ASSUMPTION,
               "limit": LIMIT, "floor": [row for row in floors if row["class"] == "reliable"],
               "fits": [row for row in fits if row["class"] == "reliable"]}
    (args.output_dir / "seed_spread_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.no_plots:
        plot(spread, floors, fits, args.output_dir)
    print(f"{len(entries)} scored sets -> {len(spread)} spread rows, {len(floors)} floor rows, {len(fits)} fit rows: "
          f"{args.output_dir}")


if __name__ == "__main__":
    main()
