"""RQ1-RQ4 tables and figures from the one-shot scoring of a frozen prediction plan.

Input is the output directory of `evaluate_frcnn.py --final-plan <schema-3 plan>`: per prediction
set and domain, metrics.json and bootstrap_class_ap.npy (shared test resamples per domain).

Outputs: domain drop with CI and the compound-vs-sum interaction with its detection limit (RQ1);
seed-mean recovery curves with a hierarchical bootstrap band (RQ2); N_BE/N75/N90 as point with a
status, band interval, ladder bracket and log-linear fit estimate (RQ3); the same against
matched-pool ceilings (RQ4); the recovery denominator per shift and class; forgetting on the
source test set; generic-pretrained controls. Primary metric: reliable mAP50:95.

Analysis rules of the lecturer-review revision (.agent/DECISIONS.md 2026-11-21). They were written
after the curves had been seen, so they are descriptive, not confirmatory.

- thresholds.csv: N95 is retired. `status` describes the point threshold: "determined" (a rung
  below the ceiling), "not determined within pool" (only the ceiling rung passes, and that rung is
  1 by construction, so the statement is "> ladder_last_failing") or "not reached". The fit columns
  come from the primary ("ci") region of the log-linear fit. `fit_inside_ladder_bracket` is False
  where the fitted line crosses the level outside (ladder_last_failing, ladder_first_passing]: a
  measured rung then contradicts the fit, and the ladder bracket is the measurement.
- fits.csv: Recovery = alpha + beta ln N by ordinary least squares; it replaces the power fit.
  The region runs from the first rung whose 95% band lower bound is above 0 at that and every
  larger rung ("ci"; sensitivity "point": the point-estimate N_BE) to the largest rung below the
  ceiling, which is excluded because it is 1 by construction. Fewer than four rungs: no fit, the
  row keeps the rung count and the ladder bracket is the result. R^2 and RMSE are on the region
  only. N75/N90 from the fit are interpolations inside the region. Their CI refits every bootstrap
  replicate on the same rungs and covers only the replicates where the threshold is defined (the
  count is reported next to the count of finite replicates); it is given only next to a point
  estimate. Such an interval cannot leave the region. `*_ci_censored` names the bound ("low",
  "high" or "both") that the same percentile over ALL finite replicates puts outside the region,
  with a replicate that crosses before the first region rung ranked below every defined one and one
  that crosses after the last rung, or never, ranked above: that printed number is not a bound and
  reads "< region_start" or "> region_end".
- fit_holdout.csv: the three lowest region rungs predict the remaining rungs and the ceiling rung.
- recovery_denominator.csv: ceiling - zero-shot with a bootstrap CI and its ratio to the floor.
  `replicates_at_or_below_floor` counts the same event as `undefined_replicates` in
  recovery_points.csv, but with its own redraw of the ceiling seeds: with more than one ceiling
  seed the two counts differ by Monte Carlo noise, and the recovery band uses `undefined_replicates`.
- summary.json: the interaction gains its bootstrap SE, the CI half-width, the 80%-power minimum
  detectable effect and their shares of the weather drop; plus the reliable full-pool thresholds
  and fits and the reliable denominators.

In the analyses above the rung at N = pool is read as the ceiling reference (recovery 1 by
construction), so runs pooled into one rung or one ceiling must share one epoch count: E = 12 runs
found under --scored next to the E = 3 full-pool runs still stop the script. They are read through
--revision-scored instead, which names the schedules. A table without rows is removed from the
output directory, so a rerun leaves no table of an earlier run behind; where figures are drawn, the
same holds for a recovery_<shift>_e12.png that the run does not draw.

Ceilings of the revision: --revision-scored DIR (.agent/DECISIONS.md 2026-11-21, "Ceiling",
"Materially higher" and "Bracket reading", and the addendum on the second E = 3 matched-pool seed).
DIR is the scored output of the revision plan, with the same test resamples. Only its fine-tune runs
are read, and each is told apart by its run fields alone, never by a folder name: E = 12 on the full
pool (seeds 0 and 1), E = 12 on the matched pool of 1,738 (lighting, weather; seeds 0 and 1), E = 3
on the matched pool (seed 1, the second seed of the short-schedule matched-pool ceiling). A fine-tune
run of any other kind or seed, and a run (domain, budget, epochs, seed) scored in both directories,
stops the script. So does a ceiling without both of its seeds: each E = 12 ceiling, and for lighting
and weather the E = 3 matched-pool ceiling (seed 0 from the campaign, seed 1 from DIR), for which
the full-pool runs never stand in. So does a test set whose bootstrap seed, resample count, image
count or vector length differs between the two directories: replicate t would then pair two
different resampled test sets. Without the argument every output is what it was before; with it,
every "full-pool" row and the compound "matched-pool" rows keep their values, and rq1_drop.csv,
forgetting.csv and controls.csv do not change.

- Analyses, in the `analysis` column: "full-pool" and "matched-pool" keep the E = 3 ceiling, the
  second now on two ceiling seeds for lighting and weather. "full-pool-e12" and "matched-pool-e12"
  take the seed mean of the E = 12 runs as mAP_c ("Ceiling"), whatever it turned out to be. Their
  ladder is the same E = 3 rungs plus the E = 3 runs at N = pool as a MEASURED top rung, with
  recovery (mAP - mAP(0)) / (mAP_c - mAP(0)): not 1 by construction, never clipped, above 1 where the
  short schedule scored higher than the long one. As before, the matched-pool runs at 1,738 are a
  ceiling and a top rung of the matched-pool analyses only, not a rung of a full-pool ladder.
- Every statistic of an "-e12" analysis follows the definitions above with the top rung measured.
  The band redraws the sampling seeds of the top rung like those of any rung, independently of the
  ceiling seeds. `status` is "determined" when the point threshold is a measured rung, the top rung
  included, and "not reached" when no rung passes; "not determined within pool" does not occur. The
  fit region may extend to the top rung, the held-out check predicts the remaining region rungs
  only (no rung is 1 by construction, so `is_ceiling_rung` is False), and `predicted_at_ceiling` is
  the fitted value at N = pool, to be held against the measured top rung. Recovery stays undefined
  where the E = 12 gain does not exceed the floor: one row with the "undefined" note, as before.
- A replicate whose redrawn E = 12 gain does not exceed the floor is undefined as well, as before, so
  a band, band_from / band_to and a fit interval are conditional on the replicates that are left;
  close to the floor the ones that drop out have the smallest denominators, so the recovery furthest
  from 0. The "-e12" rows of thresholds.csv therefore end with band_replicates and
  band_undefined_replicates (the count that recovery_points.csv gives as undefined_replicates). The
  E = 3 rows keep the columns they had.
- "Bracket reading": the rungs stay at E = 3. A threshold against the E = 3 ceiling is a lower bound;
  against the E = 12 ceiling it is an upper bound wherever large rungs are themselves undertrained.
  Both tables are published either way.
- ceiling_e12_vs_e3.csv: per shift, class and pool ("full", "matched"; the same runs for compound)
  the two seed-mean ceilings and E12 - E3 paired by seed over the seeds present in both. Replicate t
  uses test resample t for both runs of a pair and redraws the pairs together. `materially_higher`
  is the pre-registered rule: difference >= 0.005 AP with the lower 95% bound above 0.
  `materially_lower` (difference <= -0.005 with the upper bound below 0) is its mirror image and
  descriptive only. `residual_gap_*` is the source model on its own test set minus the ceiling.
- forgetting_e12.csv: forgetting.csv for the fine-tune runs of the revision, with their epoch count.
- summary.json gains thresholds_e12, fits_e12, recovery_denominator_e12 (reliable rows of both
  "-e12" analyses) and ceiling_e12_vs_e3 (reliable rows); recovery_denominator keeps the E = 3 rows.
- recovery_<shift>_e12.png: the "full-pool-e12" curves, the top rung drawn as a measured point.
- New draws come from generators of their own, so no earlier number moves: --seed + E12_SEED_OFFSET
  for the "-e12" bands, that plus DENOMINATOR_SEED_OFFSET for their denominators, and
  --seed + PAIRED_SEED_OFFSET for the paired differences.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist, mean

import numpy as np

from bdd_pipeline.campaign import MATCHED_DOMAINS, MATCHED_POOL, POLICY, TARGETS
from bdd_pipeline.data import CLASSES, EXPECTED_DOMAINS, RELIABLE_CLASSES
from bdd_pipeline.recovery import (band, bootstrap_recovery, ladder_bracket, paired_difference, recovery,
                                   sustained_threshold, threshold_interval)

KEYS = ("reliable", "all") + RELIABLE_CLASSES  # "all" is the ten-class sensitivity analysis
PALETTE = {"reliable": "#1f2933", "pedestrian": "#2f6b9a", "car": "#2d997c", "traffic light": "#d48041", "traffic sign": "#8163a8"}
# recovery.LEVELS still lists N95 because bdd_pipeline/ is frozen for provenance; N95 is retired from every reported table.
REPORTED_LEVELS = {"N_BE": 0.0, "N75": 0.75, "N90": 0.90}
FIT_LEVELS = {label: level for label, level in REPORTED_LEVELS.items() if label != "N_BE"}
REGION_RULES = ("ci", "point")  # primary: 95% band lower bound > 0; sensitivity: point recovery > 0
MIN_FIT_RUNGS = 4
HOLDOUT_FIT_RUNGS = 3
FIT_FORM = "alpha + beta * ln N"
DENOMINATOR_SEED_OFFSET = 20261121  # the denominator table draws from a generator of its own
E3_ANALYSES = ("full-pool", "matched-pool")  # against the E = 3 ceiling; the rung at N = pool is 1 by construction
E12_SUFFIX = "-e12"  # the same two ladders against the E = 12 ceiling; the rung at N = pool is measured
LADDER_EPOCHS = POLICY["epochs"]  # 3: the locked schedule of every rung and of the short-schedule ceilings
CEILING_EPOCHS = 12  # the source model's epoch count (DECISIONS 2026-11-21, "Ceiling")
CEILING_SEEDS = POLICY["ceiling_seeds"]  # 0 and 1: the seeds of every ceiling that --revision-scored reads or completes
# What a fine-tune run of the revision may be, by the (epochs, budget) of its run fields, and the seeds of each kind
# (seed 0 of "matched-e3" is the campaign's run).
REVISION_KINDS = {(CEILING_EPOCHS, "full"): "full-e12", (CEILING_EPOCHS, str(MATCHED_POOL)): "matched-e12",
                  (LADDER_EPOCHS, str(MATCHED_POOL)): "matched-e3"}
REVISION_SEEDS = {"full-e12": CEILING_SEEDS, "matched-e12": CEILING_SEEDS, "matched-e3": (1,)}
MATERIAL_DELTA = 0.005  # AP (DECISIONS 2026-11-21, "Materially higher")
# Generators of the revision outputs: --seed + E12_SEED_OFFSET for the "-e12" recovery bootstrap, that plus
# DENOMINATOR_SEED_OFFSET for the "-e12" denominators, --seed + PAIRED_SEED_OFFSET for the paired E12 - E3 differences.
E12_SEED_OFFSET = 20261123
PAIRED_SEED_OFFSET = 20261124
CEILING_SENTINEL = 0  # budget of the ceiling in bootstrap_recovery when the rung at N = pool is measured; N = 0 is never a rung


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


def require_one_schedule(domain: str, ladder: dict[int, list[dict]]) -> None:
    """Runs pooled into one seed mean must share one epoch count, or the mean is neither schedule's value.

    Under --scored the rung at N = pool is taken whole as the ceiling, so E = 12 ceilings scored next to the E = 3
    full-pool runs (DECISIONS 2026-11-21) stop the script there; --revision-scored reads them as a schedule of their own.
    Every group that is pooled after that split passes through here as well.
    """
    for budget, entries in sorted(ladder.items()):
        epochs = sorted({str(entry["run"].get("epochs")) for entry in entries})
        if len(epochs) > 1:
            raise ValueError(f"{domain}: the fine-tune runs at N = {budget} mix epoch counts {', '.join(epochs)}; "
                             "they would be pooled into one seed mean (at N = pool: into one ceiling). "
                             "Separate the schedules first (DECISIONS 2026-11-21, ceilings at E = 12)")


def revision_kind(run: dict) -> str:
    """Kind of a fine-tune run of the revision, from its run fields only; folder names and labels say nothing here.

    "full-e12": E = 12 on the full pool. "matched-e12": E = 12 on the matched pool of 1,738. "matched-e3": the locked
    E = 3 on the matched pool, a further seed of the short-schedule matched-pool ceiling. The two matched kinds exist
    for lighting and weather only: the compound pool is the matched pool, and its runs carry budget "full". The seed
    belongs to the kind (REVISION_SEEDS): 0 and 1 for the E = 12 ceilings, 1 for "matched-e3", whose seed 0 the campaign
    scored. A run with any other seed would enter a ceiling as a further seed, so it is of no known kind either.
    """
    domain, budget, epochs = run.get("domain"), str(run.get("budget")), run.get("epochs")
    kind = REVISION_KINDS.get((epochs, budget)) if domain in TARGETS else None
    if kind is None or (kind != "full-e12" and domain not in MATCHED_DOMAINS) or run.get("seed") not in REVISION_SEEDS[kind]:
        seeds = {name: " and ".join(str(seed) for seed in values) for name, values in REVISION_SEEDS.items()}
        raise ValueError(f"Revision fine-tune run of no known kind: domain {domain}, budget {budget}, epochs {epochs}, "
                         f"seed {run.get('seed')}. Known (DECISIONS 2026-11-21): E = {CEILING_EPOCHS} at budget full, "
                         f"seeds {seeds['full-e12']}; at budget {MATCHED_POOL} for {' and '.join(MATCHED_DOMAINS)}, "
                         f"E = {CEILING_EPOCHS} with seeds {seeds['matched-e12']} or E = {LADDER_EPOCHS} with seed "
                         f"{seeds['matched-e3']}")
    return kind


def run_identity(run: dict) -> tuple:
    """(domain, budget in images, epochs, seed): what makes a fine-tune run one run. Without the epoch count an E = 12
    ceiling and the E = 3 run of the same stage, domain, budget and seed would be merged."""
    return run["domain"], budget_of(run), run.get("epochs"), run["seed"]


def duplicate_runs(campaign: list[dict], revision: list[dict]) -> list[tuple]:
    """Identities of the fine-tune runs scored in both directories, or twice on one test set within either."""
    groups = [[entry for entry in entries if entry["run"]["stage"] == "fine-tune"] for entries in (campaign, revision)]
    across = {run_identity(entry["run"]) for entry in groups[0]} & {run_identity(entry["run"]) for entry in groups[1]}
    counts = Counter((run_identity(entry["run"]), entry["domain"]) for group in groups for entry in group)
    return sorted(across | {identity for (identity, _), count in counts.items() if count > 1}, key=str)


def resample_signatures(directory: Path, stage: str | None = None) -> dict[str, set[tuple]]:
    """Per test domain, every (bootstrap seed, resamples, test images, rows of bootstrap_class_ap.npy) under `directory`.

    load() keeps none of these. Replicate t of two scored sets is the same resampled test set only where all four
    agree: the resamples are drawn from the seed and the image count. `stage` keeps the runs of that stage only.
    """
    found = defaultdict(set)
    for metrics_path in sorted(directory.glob("*/*/metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if stage is None or metrics["run"]["stage"] == stage:
            rows = np.load(metrics_path.parent / "bootstrap_class_ap.npy", mmap_mode="r").shape[0]
            bootstrap = metrics["bootstrap"]
            found[metrics["domain"]].add((bootstrap.get("seed"), bootstrap.get("resamples"), metrics.get("images"), rows))
    return dict(found)


def require_shared_resamples(campaign: Path, revision: Path) -> None:
    """Stops unless the fine-tune sets of the revision were scored on the test resamples of the campaign.

    Every "-e12" band, "-e12" denominator and paired E12 - E3 interval takes replicate t of a campaign run and
    replicate t of a revision run as one resampled test set. Nothing in the vectors says whether that holds, and a
    directory scored with another bootstrap seed would be paired without a message.
    """
    ours = resample_signatures(campaign)
    for domain, theirs in sorted(resample_signatures(revision, "fine-tune").items()):
        found = sorted(ours.get(domain, set()) | theirs, key=str)
        if len(found) > 1:
            raise ValueError(f"Test set {domain}: {campaign} and the fine-tune sets of {revision} do not share one "
                             "(bootstrap seed, resamples, test images, rows of bootstrap_class_ap.npy): "
                             f"{', '.join(str(signature) for signature in found)}. Replicate t would not be the same "
                             "resampled test set in both")


def read_revision(directory: Path, campaign_directory: Path, campaign: list[dict]) -> list[dict]:
    """Fine-tune entries of the revision's scored directory, each with its `kind`; adapt and replay entries are not used.

    Stops on a fine-tune run of no known kind (revision_kind), on a run that would be counted twice (duplicate_runs)
    and on test resamples that are not those of the campaign directory (require_shared_resamples).
    """
    revision = [{**entry, "kind": revision_kind(entry["run"])} for entry in load(directory)
                if entry["run"]["stage"] == "fine-tune"]
    duplicates = duplicate_runs(campaign, revision)
    if duplicates:
        listed = "; ".join(f"{domain} N = {budget} E = {epochs} seed {seed}" for domain, budget, epochs, seed in duplicates)
        raise ValueError(f"Fine-tune runs scored more than once, in {directory} and the campaign directory or twice in "
                         f"one of them (domain, budget, epochs, seed): {listed}")
    require_shared_resamples(campaign_directory, directory)
    return revision


def revision_groups(domain: str, revision: list[dict]) -> dict[str, list[dict]]:
    """The revision runs of one shift scored on its own test set, per kind and in seed order."""
    own = sorted((entry for entry in revision if entry["run"]["domain"] == domain and entry["domain"] == domain),
                 key=lambda entry: entry["run"]["seed"])
    return {kind: [entry for entry in own if entry["kind"] == kind] for kind in REVISION_KINDS.values()}


def require_ceiling_seeds(domain: str, ceiling: str, runs: list[dict]) -> None:
    """A ceiling that the revision brings or completes is the mean of seeds 0 and 1, no fewer and no more.

    That is the registered ceiling (DECISIONS 2026-11-21, "Ceiling", and the addendum on the E = 3 matched pool). A
    missing seed would leave a one-seed ceiling and a paired E12 - E3 difference on one pair; a further one would move
    the mean. Both stop the script.
    """
    found = sorted(entry["run"]["seed"] for entry in runs)
    if found != sorted(CEILING_SEEDS):
        raise ValueError(f"{domain}: the {ceiling} needs seeds {' and '.join(str(seed) for seed in CEILING_SEEDS)}, found "
                         f"{' '.join(str(seed) for seed in found) or 'none'}")


def e12_ceilings(domain: str, pool: int, groups: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """The E = 12 ceiling runs of one shift at its full pool and at the matched pool, from revision_groups.

    The shift whose pool is the matched pool has one set of runs for both. Each must exist, with both of its seeds
    (require_ceiling_seeds): the "-e12" analyses and the E12 - E3 comparison are not written for some shifts only.
    """
    e12 = {"full": groups["full-e12"], "matched": groups["full-e12"] if pool == MATCHED_POOL else groups["matched-e12"]}
    for name, budget in (("full", pool), ("matched", MATCHED_POOL)):
        if not e12[name]:
            raise ValueError(f"{domain}: no {CEILING_EPOCHS}-epoch {name}-pool ceiling run (N = {budget}) in the revision "
                             "directory")
        require_one_schedule(domain, {budget: e12[name]})
        require_ceiling_seeds(domain, f"{CEILING_EPOCHS}-epoch {name}-pool ceiling (N = {budget})", e12[name])
    return e12


def matched_e3_ceiling(domain: str, pool: int, ceilings: list[dict], campaign: list[dict],
                       revision: list[dict] | None = None) -> list[dict]:
    """The E = 3 ceiling runs at the matched pool of 1,738.

    `campaign`: the campaign's runs at 1,738, none for the shift whose pool is the matched pool. Without the revision
    (`revision` None) these are the ceiling, or the full-pool `ceilings` where there are none, as before. With it, the
    full-pool runs stand in only where pool is the matched pool, where they are those runs. Any other shift needs the
    campaign's seed 0 and the revision's seed 1 at 1,738 (require_ceiling_seeds): runs trained on a larger pool would
    otherwise be written as the measured top rung at N = 1,738 and as the E = 3 side of the matched comparison.
    """
    if revision is None or pool == MATCHED_POOL:
        return campaign or ceilings
    require_ceiling_seeds(domain, f"{LADDER_EPOCHS}-epoch matched-pool ceiling (N = {MATCHED_POOL}; seed 0 from the "
                                  "campaign, seed 1 from the revision)", campaign + revision)
    return campaign + revision


def threshold_status(point: int | None, ceiling_budget: int, top_measured: bool = False) -> str:
    """Whether the sustained point threshold was measured; the ceiling rung is 1 by construction, so it measures nothing.

    By default the rung at ceiling_budget is the ceiling reference itself. top_measured says that it is a measured rung
    (the E = 3 run at N = pool against an E = 12 ceiling): it then counts like any other, and a threshold that no rung
    passes is "not reached".
    """
    if point is None:
        return "not reached"
    return "determined" if top_measured or point < ceiling_budget else "not determined within pool"


def inside_ladder_bracket(estimate: float | None, point: dict[int, float], level: float) -> bool | None:
    """Whether a fit threshold agrees with the measured rungs: above the last failing rung, at most the first passing one."""
    if estimate is None:
        return None
    bracket = ladder_bracket(point, level, strict=False)
    return ((bracket["last_failing"] is None or estimate > bracket["last_failing"])
            and (bracket["first_passing"] is None or estimate <= bracket["first_passing"]))


def fit_region(start_curve: dict[int, float], ceiling_budget: int, top_measured: bool = False) -> list[int]:
    """Rungs from the first one where start_curve is above 0 at it and at every larger rung, up to the last below the ceiling.

    The rung at ceiling_budget is left out as 1 by construction, unless top_measured says that it is a measured rung
    (see threshold_status): the region then extends to it.
    """
    start = sustained_threshold(start_curve, 0.0, strict=True)
    if start is None:
        return []
    return [budget for budget in sorted(start_curve)
            if start <= budget and (budget < ceiling_budget or (top_measured and budget == ceiling_budget))]


def log_linear_ols(budgets: list[int], values: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares (alpha, beta) of value = alpha + beta ln N; `values` may hold one curve per leading row."""
    x = np.log(np.asarray(budgets, dtype=float))
    y = np.asarray(values, dtype=float)
    centered = x - x.mean()
    beta = y @ centered / (centered @ centered)
    return y.mean(axis=-1) - beta * x.mean(), beta


def fit_log_linear(region: list[int], values: list[float]) -> dict | None:
    """Recovery = alpha + beta ln N on the region rungs, R^2 and RMSE on those rungs only; None below MIN_FIT_RUNGS."""
    if len(region) < MIN_FIT_RUNGS:
        return None
    y = np.asarray(values, dtype=float)
    alpha, beta = (float(value) for value in log_linear_ols(region, y))
    residual = y - (alpha + beta * np.log(np.asarray(region, dtype=float)))
    sse, total = float(residual @ residual), float(((y - y.mean()) ** 2).sum())
    return {"form": FIT_FORM, "alpha": alpha, "beta": beta, "r2": 1 - sse / total if total > 0 else None,
            "rmse": (sse / len(y)) ** 0.5}


def fit_threshold(alpha: float, beta: float, level: float, low: int, high: int) -> float | None:
    """Budget where the fitted line reaches the level: only for a rising line, and only inside [low, high] (interpolation)."""
    if not beta > 0:
        return None
    try:
        budget = math.exp((level - alpha) / beta)
    except OverflowError:
        return None
    return budget if low <= budget <= high else None


def censored_bounds(below: int, defined: int, above: int) -> str | None:
    """Which bounds of the 95% percentile interval over ALL replicates fall outside the fitted range.

    The replicates are ranked `below` (the line crosses before the first region rung), then `defined`, then `above`
    (it crosses after the last region rung, or never). "low" / "high" / "both": the 2.5th / 97.5th percentile of
    that ranking is not among the defined replicates, so the bound printed from the defined ones alone is not a
    bound. "no": both are. None without a defined replicate.
    """
    if not defined:
        return None
    outside = []
    for name, per_mille in (("low", 25), ("high", 975)):
        # np.percentile interpolates between the order statistics at the floor and the ceiling of (count - 1) * share.
        lower, remainder = divmod((below + defined + above - 1) * per_mille, 1000)
        if lower < below or lower + (remainder > 0) >= below + defined:
            outside.append(name)
    return "both" if len(outside) == 2 else outside[0] if outside else "no"


def bootstrap_fit_thresholds(budgets: list[int], curves: np.ndarray, region: list[int]) -> dict[str, dict]:
    """Percentile CI of each fit threshold, every replicate curve refitted on the SAME region rungs as the point curve.

    Replicates with a non-finite value in the region are skipped. A replicate threshold is defined under the
    interpolation rule of fit_threshold, so each interval is conditional on the count reported with it and cannot
    leave the region; ci_censored (see censored_bounds) says where that hides the bound.
    """
    sub = curves[:, [budgets.index(budget) for budget in region]]
    sub = sub[np.isfinite(sub).all(axis=1)]
    alpha, beta = (values.tolist() for values in log_linear_ols(region, sub))
    out = {}
    for label, level in FIT_LEVELS.items():
        defined, below = [], 0
        for a, b in zip(alpha, beta):
            value = fit_threshold(a, b, level, region[0], region[-1])
            if value is not None:
                defined.append(value)
            elif fit_threshold(a, b, level, 0, region[0]) is not None:
                below += 1
        out[label] = {"ci_low": float(np.percentile(defined, 2.5)) if defined else None,
                      "ci_high": float(np.percentile(defined, 97.5)) if defined else None,
                      "replicates_defined": len(defined), "replicates_finite": len(sub),
                      "ci_censored": censored_bounds(below, len(defined), len(sub) - len(defined) - below)}
    return out


def holdout_check(region: list[int], point: dict[int, float], ceiling_budget: int, top_measured: bool = False) -> list[dict]:
    """Fit the three lowest region rungs, then predict every remaining region rung and the ceiling rung (observed 1).

    With top_measured no rung is 1 by construction: the remaining region rungs, which may end at the measured rung at
    ceiling_budget, are the whole check.
    """
    if len(region) < MIN_FIT_RUNGS:
        return []
    fitted = region[:HOLDOUT_FIT_RUNGS]
    alpha, beta = (float(value) for value in log_linear_ols(fitted, [point[budget] for budget in fitted]))
    rows = []
    for budget in region[HOLDOUT_FIT_RUNGS:] + ([] if top_measured else [ceiling_budget]):
        predicted = alpha + beta * math.log(budget)
        rows.append({"fit_rungs": " ".join(str(rung) for rung in fitted), "budget": budget, "observed": point[budget],
                     "predicted": predicted, "error": predicted - point[budget],
                     "is_ceiling_rung": not top_measured and budget == ceiling_budget})
    return rows


def region_fits(budgets: list[int], point: dict[int, float], curves: np.ndarray, ceiling_budget: int,
                top_measured: bool = False) -> dict[str, dict]:
    """Per region rule: the region rungs, the log-linear fit (None below MIN_FIT_RUNGS), its thresholds and the held-out check.

    top_measured: the rung at ceiling_budget is measured (see threshold_status), so the region may extend to it and
    predicted_at_ceiling is the fitted value at that measured rung.
    """
    lower, _ = band(curves)
    start_curves = {"ci": {budget: float(value) for budget, value in zip(budgets, lower)}, "point": point}
    out = {}
    for rule in REGION_RULES:
        region = fit_region(start_curves[rule], ceiling_budget, top_measured)
        values = [point[budget] for budget in region]
        fit = fit_log_linear(region, values)
        entry = {"region": region, "fit": fit, "monotone": bool(np.all(np.diff(values) >= 0)) if region else None,
                 "thresholds": {}, "predicted_at_ceiling": None,
                 "holdout": holdout_check(region, point, ceiling_budget, top_measured)}
        if fit:
            intervals = bootstrap_fit_thresholds(budgets, curves, region)
            for label, level in FIT_LEVELS.items():
                estimate = fit_threshold(fit["alpha"], fit["beta"], level, region[0], region[-1])
                # Without a point estimate the replicate percentiles are not its interval; the count of defined replicates stays.
                interval = intervals[label] if estimate is not None else {**intervals[label], "ci_low": None, "ci_high": None,
                                                                          "ci_censored": None}
                entry["thresholds"][label] = {"fit": estimate, **interval,
                                              "inside_ladder_bracket": inside_ladder_bracket(estimate, point, level)}
            entry["predicted_at_ceiling"] = fit["alpha"] + fit["beta"] * math.log(ceiling_budget)
        out[rule] = entry
    return out


def fit_rows(domain: str, key: str, analysis: str, fits: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """fits.csv rows, one per region rule and written even when no fit exists, and the fit_holdout.csv rows."""
    rows, holdout = [], []
    for rule, entry in fits.items():
        fit, region, identity = entry["fit"] or {}, entry["region"], {"domain": domain, "class": key, "analysis": analysis}
        row = {**identity, "region_rule": rule, **{name: fit.get(name) for name in ("form", "alpha", "beta", "r2", "rmse")},
               "rungs": len(region), "region_start": region[0] if region else None,
               "region_end": region[-1] if region else None, "monotone": entry["monotone"]}
        estimates = {label: entry["thresholds"].get(label, {}) for label in FIT_LEVELS}
        for label, estimate in estimates.items():
            row.update({label: estimate.get("fit"), f"{label}_ci_low": estimate.get("ci_low"),
                        f"{label}_ci_high": estimate.get("ci_high")})
        row["predicted_at_ceiling"] = entry["predicted_at_ceiling"]
        row.update({f"{label}_replicates_defined": estimate.get("replicates_defined") for label, estimate in estimates.items()})
        row["replicates_finite"] = next((e["replicates_finite"] for e in estimates.values() if e), None)
        for name in ("ci_censored", "inside_ladder_bracket"):
            row.update({f"{label}_{name}": estimate.get(name) for label, estimate in estimates.items()})
        rows.append(row)
        holdout += [{**identity, "region_rule": rule, **check} for check in entry["holdout"]]
    return rows, holdout


def denominator_row(zero_point: float | None, ceiling_points: list[float | None], zero_boot: np.ndarray,
                    ceiling_boots: list[np.ndarray], min_denominator: float, seed: int) -> dict | None:
    """The recovery denominator, seed-mean ceiling - zero-shot, with a bootstrap CI and its ratio to the floor.

    Mirrors bootstrap_recovery: replicate t uses test resample t for both terms and redraws the ceiling seeds with
    replacement, from a generator of its own so that no other number moves. Non-finite replicates are left out of
    the CI and both counts. None when the class has no value.
    """
    if len(ceiling_points) != len(ceiling_boots):
        raise ValueError(f"{len(ceiling_points)} ceiling values but {len(ceiling_boots)} ceiling bootstrap vectors")
    if zero_point is None or any(value is None for value in ceiling_points):
        return None
    floor = max(min_denominator, 0.0)
    stack = np.stack(ceiling_boots)
    seeds, replicates = stack.shape
    drawn = np.random.default_rng(seed).integers(0, seeds, (replicates, seeds))
    replicate = stack[drawn, np.arange(replicates)[:, None]].mean(axis=1) - zero_boot
    finite = replicate[np.isfinite(replicate)]
    ceiling = mean(ceiling_points)
    denominator = ceiling - zero_point
    return {"zero_shot": zero_point, "ceiling": ceiling, "denominator": denominator,
            "ci_low": float(np.percentile(finite, 2.5)) if len(finite) else None,
            "ci_high": float(np.percentile(finite, 97.5)) if len(finite) else None,
            "ratio_to_floor": denominator / min_denominator if min_denominator > 0 else None, "defined": denominator > floor,
            "replicates_at_or_below_floor": int((finite <= floor).sum()), "replicates": len(finite), "ceiling_seeds": seeds}


def paired_seed_difference(points_a: dict[int, float], boots_a: dict[int, np.ndarray], points_b: dict[int, float],
                           boots_b: dict[int, np.ndarray], seed: int) -> dict:
    """a - b paired by sampling seed, over the seeds present in both, with a 95% percentile interval.

    The point value is the mean of the per-seed differences. Replicate t uses test resample t for both runs of every
    pair and redraws the pairs together with replacement (a redrawn seed brings its a run and its b run), from a
    generator of its own. Non-finite replicates are left out of the interval and the count. Without a common seed
    there is no difference.
    """
    common = sorted(set(points_a) & set(points_b))
    out = {"paired_seeds": " ".join(str(value) for value in common), "difference": None, "ci_low": None, "ci_high": None,
           "replicates": 0}
    if not common:
        return out
    stack = np.stack([np.asarray(boots_a[value], dtype=float) - boots_b[value] for value in common])
    pairs, replicates = stack.shape
    drawn = np.random.default_rng(seed).integers(0, pairs, (replicates, pairs))
    replicate = stack[drawn, np.arange(replicates)[:, None]].mean(axis=1)
    finite = replicate[np.isfinite(replicate)]
    out.update({"difference": mean(points_a[value] - points_b[value] for value in common), "replicates": len(finite)})
    if len(finite):
        out.update({"ci_low": float(np.percentile(finite, 2.5)), "ci_high": float(np.percentile(finite, 97.5))})
    return out


def material_flags(difference: float | None, ci_low: float | None, ci_high: float | None,
                   delta: float = MATERIAL_DELTA) -> dict:
    """How an E12 - E3 difference reads (DECISIONS 2026-11-21, "Materially higher").

    materially_higher is the pre-registered rule: difference >= delta with the lower 95% bound above 0.
    materially_lower is its mirror image, difference <= -delta with the upper bound below 0; it was not pre-registered
    and is descriptive only. Both None without a difference or an interval.
    """
    if difference is None or ci_low is None or ci_high is None:
        return {"materially_higher": None, "materially_lower": None}
    return {"materially_higher": bool(difference >= delta and ci_low > 0),
            "materially_lower": bool(difference <= -delta and ci_high < 0)}


def ceiling_comparison(e3: list[dict], e12: list[dict], key: str, source_point: float | None, seed: int) -> dict | None:
    """One ceiling_e12_vs_e3.csv row: both seed-mean ceilings, the paired E12 - E3 difference and how it reads.

    The seed means are over all seeds of each schedule, the difference over the seeds present in both. residual_gap_*
    is the source model on its own test set minus the ceiling, as residual_gap in rq1_drop.csv. None when a run has
    no value for the class.
    """
    by_seed = [{entry["run"]["seed"]: entry for entry in runs} for runs in (e3, e12)]
    if any(len(group) != len(runs) for group, runs in zip(by_seed, (e3, e12))):
        raise ValueError("Two ceiling runs of one schedule share a seed")
    points = [{value: entry["points"][key] for value, entry in group.items()} for group in by_seed]
    if not all(points) or any(value is None for group in points for value in group.values()):
        return None
    boots = [{value: entry["boots"][key] for value, entry in group.items()} for group in by_seed]
    e3_mean, e12_mean = (mean(group.values()) for group in points)
    paired = paired_seed_difference(points[1], boots[1], points[0], boots[0], seed)
    gap = lambda ceiling: None if source_point is None else source_point - ceiling  # noqa: E731
    return {"mAP_c_e3": e3_mean, "mAP_c_e12": e12_mean, "seeds_e3": " ".join(str(value) for value in sorted(points[0])),
            "seeds_e12": " ".join(str(value) for value in sorted(points[1])), **paired, "delta": MATERIAL_DELTA,
            **material_flags(paired["difference"], paired["ci_low"], paired["ci_high"]), "mAP_source": source_point,
            "residual_gap_e3": gap(e3_mean), "residual_gap_e12": gap(e12_mean)}


def detection_limit(ci_low: float, ci_high: float, replicate: np.ndarray, drops: dict[str, float]) -> dict:
    """How small an interaction the test could have seen, in AP and as a share of the weather drop.

    se is the bootstrap SD of the replicate vector (ddof=1, NaN ignored). mde_80 = (z_0.975 + z_0.80) * se is the
    |interaction| that a two-sided 5% test detects with 80% power. largest_compatible_abs is the far end of the CI.
    The shares are None unless the weather drop is positive: a limit has no share of a drop that is not there.
    """
    normal = NormalDist()
    se = float(np.nanstd(replicate, ddof=1))
    limits = {"ci_half_width": (ci_high - ci_low) / 2, "mde_80": (normal.inv_cdf(0.975) + normal.inv_cdf(0.80)) * se,
              "largest_compatible_abs": max(abs(ci_low), abs(ci_high))}
    weather = drops["weather"]
    return {"se": se, **limits, **{f"drop_{domain}": drops[domain] for domain in ("compound", "lighting", "weather")},
            **{f"{name}_over_weather_drop": value / weather if weather > 0 else None for name, value in limits.items()}}


def threshold_rows(domain: str, key: str, analysis: str, budgets: list[int], point: dict[int, float],
                   curves: np.ndarray, ceiling_budget: int, fits: dict[str, dict] | None = None,
                   top_measured: bool = False) -> list[dict]:
    """thresholds.csv rows of one curve; top_measured: the rung at ceiling_budget is measured (see threshold_status).

    Such a row also ends with band_replicates and band_undefined_replicates: the band, band_from / band_to and the fit
    interval cover only the replicates whose redrawn denominator exceeds the floor, and the count has to travel with
    them. It is the largest count over the rungs, the same at every rung unless a run has a non-finite bootstrap value
    (undefined_replicates in recovery_points.csv). The rows of the E = 3 analyses keep the columns they had.
    """
    lower, upper = band(curves)
    primary = (fits or region_fits(budgets, point, curves, ceiling_budget, top_measured))["ci"]
    fit, region = primary["fit"], primary["region"]
    conditional = {"band_replicates": len(curves),
                   "band_undefined_replicates": int(np.isnan(curves).sum(axis=0).max())} if top_measured else {}
    rows = []
    for label, level in REPORTED_LEVELS.items():
        strict = label == "N_BE"
        threshold = sustained_threshold(point, level, strict)
        estimate = primary["thresholds"].get(label, {})
        rows.append({"domain": domain, "class": key, "analysis": analysis, "threshold": label, "level": level,
                     "ceiling_budget": ceiling_budget, "point": threshold,
                     "status": threshold_status(threshold, ceiling_budget, top_measured),
                     **{f"band_{k}": v for k, v in threshold_interval(budgets, lower, upper, level, strict).items()},
                     **{f"ladder_{k}": v for k, v in ladder_bracket(point, level, strict).items()},
                     "fit": estimate.get("fit"), "fit_ci_low": estimate.get("ci_low"), "fit_ci_high": estimate.get("ci_high"),
                     "fit_replicates_defined": estimate.get("replicates_defined", 0), "fit_r2": fit and fit["r2"],
                     "fit_region_start": region[0] if region else None,
                     "fit_replicates_finite": estimate.get("replicates_finite"), "fit_ci_censored": estimate.get("ci_censored"),
                     "fit_inside_ladder_bracket": estimate.get("inside_ladder_bracket"), **conditional})
    return rows


def bootstrap_recovery_measured_top(zero: np.ndarray, ceilings: list[np.ndarray], runs: dict[int, list[np.ndarray]], *,
                                    seed: int = 0, min_denominator: float = 0.0) -> tuple[list[int], np.ndarray]:
    """bootstrap_recovery for a ladder whose rungs are all measured, the rung at N = pool included.

    The frozen function writes 1.0 at budget == ceiling_budget, which is right only where that rung IS the ceiling.
    Here the ceiling runs trained on another schedule and are no rung of `runs`, so the ceiling is given the sentinel
    budget 0 (N = 0 is the source model, never a rung) and that column is dropped. A replicate then uses test resample
    t for every run, redraws the ceiling seeds once and the sampling seeds of each rung, the top rung like any other,
    all independently; where the redrawn denominator does not exceed min_denominator the whole replicate is NaN.
    """
    if any(budget <= CEILING_SENTINEL for budget in runs):
        raise ValueError(f"Every rung needs a budget above {CEILING_SENTINEL}, the sentinel budget of the ceiling")
    budgets, curves = bootstrap_recovery(zero, ceilings, runs, CEILING_SENTINEL, seed=seed, min_denominator=min_denominator)
    return budgets[1:], curves[:, 1:]  # the sentinel sorts first


def curve_analysis(domain: str, key: str, analysis: str, zero: dict, ladder: dict[int, list[dict]],
                   ceilings: list[dict], ceiling_budget: int, min_denominator: float, seed: int, *,
                   top_measured: bool = False):
    """Curve points, threshold rows, fit rows and held-out rows of one curve.

    By default the rung at ceiling_budget is the ceiling itself: recovery 1 in the point curve and in every replicate,
    and `ladder` holds the rungs below it. top_measured: `ceilings` trained on another schedule and `ladder` also
    holds the runs at ceiling_budget, a measured top rung whose recovery is computed like that of any rung.
    """
    zero_point, ceiling_point = zero["points"][key], mean(c["points"][key] for c in ceilings)
    if zero_point is None or ceiling_point is None:
        return [], [], [], []
    if top_measured and not ladder.get(ceiling_budget):
        raise ValueError(f"{domain} {analysis}: no measured runs at the top rung N = {ceiling_budget}")
    measured = {b: rs for b, rs in ladder.items() if b < ceiling_budget or (top_measured and b == ceiling_budget)}
    runs = {b: [r["boots"][key] for r in rs] for b, rs in measured.items()}
    point = {b: recovery(mean(r["points"][key] for r in rs), zero_point, ceiling_point, min_denominator)
             for b, rs in measured.items()}
    if not top_measured:
        point[ceiling_budget] = 1.0
    if any(value is None for value in point.values()):
        return [{"domain": domain, "class": key, "analysis": analysis, "budget": None, "recovery": None,
                 "note": f"undefined: ceiling gain {ceiling_point - zero_point:.4f} <= {min_denominator}"}], [], [], []
    ceiling_boots = [c["boots"][key] for c in ceilings]
    if top_measured:
        budgets, curves = bootstrap_recovery_measured_top(zero["boots"][key], ceiling_boots, runs, seed=seed,
                                                          min_denominator=min_denominator)
    else:
        budgets, curves = bootstrap_recovery(zero["boots"][key], ceiling_boots, runs, ceiling_budget,
                                             seed=seed, min_denominator=min_denominator)
    lower, upper = band(curves)
    rung_runs = measured if top_measured else {ceiling_budget: ceilings, **ladder}
    points = [{"domain": domain, "class": key, "analysis": analysis, "budget": b, "recovery": point[b],
               "ci_low": float(lo), "ci_high": float(hi), "seeds": len(rung_runs[b]),
               "mAP_mean": mean(r["points"][key] for r in rung_runs[b]), "zero_shot": zero_point,
               "ceiling": ceiling_point, "undefined_replicates": int(np.isnan(curves[:, j]).sum())}
              for j, (b, lo, hi) in enumerate(zip(budgets, lower, upper))]
    fits = region_fits(budgets, point, curves, ceiling_budget, top_measured)
    return (points, threshold_rows(domain, key, analysis, budgets, point, curves, ceiling_budget, fits, top_measured),
            *fit_rows(domain, key, analysis, fits))


def plot(points: list[dict], output: Path, analysis: str = "full-pool", suffix: str = "",
         ceiling: str = "full-pool ceiling", top_measured: bool = False) -> list[Path]:
    """One recovery_<shift><suffix>.png per shift for one analysis; returns the figures written.

    top_measured (the "-e12" analyses): the largest rung of every curve is the E = 3 run at N = pool, a measurement; it
    is drawn as an open diamond, and the ceiling, which is no rung, as a dashed line at recovery 1. A shift without a
    defined curve gets no figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    written = []
    for domain in sorted({p["domain"] for p in points}):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for key in ("reliable",) + RELIABLE_CLASSES:
            rows = sorted((p for p in points if p["domain"] == domain and p["class"] == key
                           and p["analysis"] == analysis and p["budget"]), key=lambda p: p["budget"])
            if not rows:
                continue
            x = [p["budget"] for p in rows]
            ax.plot(x, [p["recovery"] for p in rows], marker="o", lw=2.6 if key == "reliable" else 1.3,
                    color=PALETTE[key], label=key)
            if key == "reliable":
                ax.fill_between(x, [p["ci_low"] for p in rows], [p["ci_high"] for p in rows], color=PALETTE[key], alpha=0.15, lw=0)
            if top_measured:
                ax.plot(x[-1:], [rows[-1]["recovery"]], ls="none", marker="D", ms=9 if key == "reliable" else 7,
                        mfc="white", mec=PALETTE[key], mew=1.8, zorder=3)
        if top_measured:
            if not ax.lines:
                plt.close(fig)
                continue
            ax.plot([], [], ls="none", marker="D", ms=7, mfc="white", mec="gray", mew=1.8,
                    label=f"N = pool: {LADDER_EPOCHS}-epoch runs, measured")
            ax.axhline(1.0, color="gray", ls="--", lw=1, label=f"{ceiling} (recovery 1)")
        for level in (0.75, 0.90):
            ax.axhline(level, color="gray", ls=":", lw=1)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Labeled target images N")
        ax.set_ylabel("Recovery(N); band: 95% CI (reliable)")
        ax.set_title(f"{domain}: recovery against the {ceiling}")
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        written.append(output / f"recovery_{domain}{suffix}.png")
        fig.savefig(written[-1], dpi=160)
        plt.close(fig)
    return written


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def write_tables(directory: Path, tables: dict[str, list[dict]]) -> None:
    """One CSV per table. A table without rows removes its file, so no table of an earlier run into the directory stays."""
    for name, rows in tables.items():
        path = directory / f"{name}.csv"
        if rows:
            write_csv(path, rows)
        else:
            path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scored", type=Path, required=True, help="evaluate_frcnn.py --final-plan output directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-denominator", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0, help="Seed for redrawing sampling/ceiling seeds")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--revision-scored", type=Path,
                        help="evaluate_frcnn.py --final-plan output directory of the revision plan; adds the analyses "
                             "against the 12-epoch ceilings (only its fine-tune runs are read)")
    args = parser.parse_args(argv)
    entries = load(args.scored)
    with_e12 = args.revision_scored is not None
    revision = read_revision(args.revision_scored, args.scored, entries) if with_e12 else []
    source_model = {e["domain"]: e for e in entries if e["run"]["stage"] == "source"}
    if set(source_model) != {"source", *TARGETS}:
        raise ValueError("The source model must be scored on the source and every target test set")
    reference = source_model["source"]
    drops, points, thresholds, forgetting, controls = [], [], [], [], []
    fits, holdout, denominators = [], [], []
    comparisons, forgetting_e12 = [], []
    for domain in TARGETS:
        own = [e for e in entries if e["run"]["domain"] == domain and e["domain"] == domain]
        ladder = defaultdict(list)
        for entry in own:
            if entry["run"]["stage"] == "fine-tune":
                ladder[budget_of(entry["run"])].append(entry)
        require_one_schedule(domain, ladder)
        pool = EXPECTED_DOMAINS[domain]["train"]
        ceilings = ladder.pop(pool, [])
        zero = source_model[domain]
        if not ceilings:
            raise ValueError(f"{domain}: no ceiling runs scored")
        # E = 3 ceiling at the matched pool: the campaign's run(s) there, then the revision's further seed; for the shift
        # whose pool is the matched pool, its full-pool ceilings.
        groups = revision_groups(domain, revision)
        matched_e3 = matched_e3_ceiling(domain, pool, ceilings, ladder.get(MATCHED_POOL, []),
                                        groups["matched-e3"] if with_e12 else None)
        require_one_schedule(domain, {MATCHED_POOL: matched_e3})
        # (analysis, ceiling runs, ceiling budget, measured top rung or None where the rung at N = pool is the ceiling)
        analyses = [("full-pool", ceilings, pool, None), ("matched-pool", matched_e3, MATCHED_POOL, None)]
        e12 = e12_ceilings(domain, pool, groups) if with_e12 else {}
        if with_e12:
            analyses += [(f"full-pool{E12_SUFFIX}", e12["full"], pool, ceilings),
                         (f"matched-pool{E12_SUFFIX}", e12["matched"], MATCHED_POOL, matched_e3)]
        for key in KEYS:
            if reference["points"][key] is not None and zero["points"][key] is not None:
                drop = paired_difference(reference["points"][key], reference["boots"][key], zero["points"][key], zero["boots"][key])
                ceiling = mean(c["points"][key] for c in ceilings)
                drops.append({"domain": domain, "class": key, "mAP_source": reference["points"][key], "mAP_0": zero["points"][key],
                              "mAP_c": ceiling, "drop": drop["estimate"], "drop_ci_low": drop["ci_low"], "drop_ci_high": drop["ci_high"],
                              "relative_drop": drop["estimate"] / reference["points"][key], "residual_gap": reference["points"][key] - ceiling,
                              "ceiling_seeds": len(ceilings), "test_images": EXPECTED_DOMAINS[domain]["final_test"]})
            # The matched-pool runs are a reference ceiling, not an extra rung of the full-pool ladder.
            rungs = {b: rs for b, rs in ladder.items() if b != MATCHED_POOL}
            for analysis, ceiling_runs, ceiling_budget, top_rung in analyses:
                top_measured = top_rung is not None
                offset = E12_SEED_OFFSET if top_measured else 0  # the "-e12" analyses draw from generators of their own
                measured = {**rungs, ceiling_budget: top_rung} if top_measured else rungs
                p, t, f, h = curve_analysis(domain, key, analysis, zero, measured, ceiling_runs, ceiling_budget,
                                            args.min_denominator, args.seed + offset, top_measured=top_measured)
                points += p
                thresholds += t
                fits += f
                holdout += h
                denominator = denominator_row(zero["points"][key], [c["points"][key] for c in ceiling_runs], zero["boots"][key],
                                              [c["boots"][key] for c in ceiling_runs], args.min_denominator,
                                              args.seed + DENOMINATOR_SEED_OFFSET + offset)
                if denominator:
                    denominators.append({"domain": domain, "class": key, "analysis": analysis, **denominator})
            if with_e12:
                for name, images, e3_runs in (("full", pool, ceilings), ("matched", MATCHED_POOL, matched_e3)):
                    comparison = ceiling_comparison(e3_runs, e12[name], key, reference["points"][key],
                                                    args.seed + PAIRED_SEED_OFFSET)
                    if comparison:
                        comparisons.append({"domain": domain, "class": key, "pool": name, "pool_images": images, **comparison})
        for entry in revision:
            run = entry["run"]
            if run["domain"] == domain and entry["domain"] == "source":
                forgetting_e12.append({"domain": domain, "budget": budget_of(run), "seed": run["seed"], "epochs": run["epochs"],
                                       "source_reliable": entry["points"]["reliable"],
                                       "change_vs_source_model": entry["points"]["reliable"] - reference["points"]["reliable"]})
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
        interaction.update(detection_limit(interaction["ci_low"], interaction["ci_high"], replicate,
                                           {d: by_domain[d]["drop"] for d in TARGETS}))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tables(args.output_dir, {"rq1_drop": drops, "recovery_points": points, "thresholds": thresholds,
                                   "forgetting": forgetting, "controls": controls, "fits": fits, "fit_holdout": holdout,
                                   "recovery_denominator": denominators, "ceiling_e12_vs_e3": comparisons,
                                   "forgetting_e12": forgetting_e12})
    clean = lambda value: None if isinstance(value, float) and math.isnan(value) else value  # noqa: E731
    primary = lambda row: row["class"] == "reliable" and row["analysis"] == "full-pool"  # noqa: E731
    reliable = lambda rows, keep: [{k: clean(v) for k, v in row.items()} for row in rows  # noqa: E731
                                   if row["class"] == "reliable" and keep(row)]
    summary = {"min_denominator": args.min_denominator, "interaction": {k: clean(v) for k, v in interaction.items()},
               "thresholds": [{k: clean(v) for k, v in row.items()} for row in thresholds if primary(row)],
               "fits": [{k: clean(v) for k, v in row.items()} for row in fits if primary(row)],
               "recovery_denominator": reliable(denominators, lambda row: row["analysis"] in E3_ANALYSES)}
    if with_e12:
        against_e12 = lambda row: row["analysis"].endswith(E12_SUFFIX)  # noqa: E731
        summary.update({"thresholds_e12": reliable(thresholds, against_e12), "fits_e12": reliable(fits, against_e12),
                        "recovery_denominator_e12": reliable(denominators, against_e12),
                        "ceiling_e12_vs_e3": reliable(comparisons, lambda row: True)})
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.no_plots:
        plot(points, args.output_dir)
        drawn = plot(points, args.output_dir, analysis=f"full-pool{E12_SUFFIX}", suffix="_e12",
                     ceiling=f"{CEILING_EPOCHS}-epoch full-pool ceiling", top_measured=True) if with_e12 else []
        # As with the tables: an "-e12" figure that this run does not draw is removed, so none of an earlier run stays.
        for domain in TARGETS:
            figure = args.output_dir / f"recovery_{domain}_e12.png"
            if figure not in drawn:
                figure.unlink(missing_ok=True)
    counted = f"{len(entries)} scored sets" + (f" and {len(revision)} fine-tune sets of the revision" if with_e12 else "")
    print(f"{counted} -> {len(points)} curve points, {len(thresholds)} threshold rows, "
          f"{len(fits)} fit rows: {args.output_dir}")


if __name__ == "__main__":
    main()
