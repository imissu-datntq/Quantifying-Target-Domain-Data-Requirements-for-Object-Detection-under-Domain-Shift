"""Source retention against target gain: the trade-off of every (method, N) between the target and the source test set.

Rules: .agent/plans/active/2026-11-24-direction-1-analyses.md, section 3 "Source retention" (direction 1 of the
lecturer's follow-up of 2026-11-24). They were written before this analysis was run but after the curves, thresholds
and RQ5 tables had been seen, so they are descriptive, not confirmatory.

Inputs are one-shot scoring outputs of `evaluate_frcnn.py --final-plan` (schema 3): the fine-tune campaign
(--ft-scored), the RQ5 grid (--da-scored) and, optionally, the revision runs (--revision-scored). main() stops unless
every scored set of these directories has one bootstrap seed and resample count and one image count per test domain
(analyze_rq5.shared_resamples): every interval takes test resample t of every run as one resampled test set. Runs are
told apart by their run fields alone, never by folder names. The source model must be scored exactly once on the source
and on every target test set. The revision runs are classified by the existing functions
(analyze_campaign.read_revision for the fine-tune runs, analyze_rq5.split_revision and merge_replay_seeds for adapt and
replay), which stop on runs of no known kind, on duplicates and on test resamples that are not shared.

Points (shift, method, N), each the seed mean of runs scored on the target test set AND on the source test set; a seed
without both entries, or with no reliable mAP, stops the script:
- "source model": N = 0, the source checkpoint (gain 0, retention 1).
- "FT": the campaign's fine-tune ladder rungs (3 epochs, seeds 0/1/2) and the full pool (N = pool, 3 epochs, seeds 0
  and 1, both required: it is the recovery reference). A campaign fine-tune run with another epoch count than the
  locked one (POLICY["epochs"]) stops the script. The matched-pool runs at 1,738 of lighting and weather are not rungs
  (as in analyze_campaign.py).
- "FT 12 epochs" (revision only): the 12-epoch full-pool runs, seeds 0 and 1, both required.
- "adapt": ALDI++ in the RQ5 grid at N = 0 (seeds 0, 1), 10, 100, 1,000 (+ 5,000 for lighting; seeds 0/1/2).
- "replay": the replay control at N = 0 (the source-only replay run, seed 0), 10, 100 and 1,000 (seed 0). With the
  revision its further seeds 1 and 2 at N = 10 and 100 (2,000 updates) join the arm.
- "adapt_matched", "replay_matched" (revision only): N = 1,000 with 3,000 updates, seeds 0/1/2, arms of their own.
Without --revision-scored the revision points are absent and every other row is unchanged.

Axes, on reliable mAP50:95. Target: gain = mAP_target - mAP(0) and recovery_e3 = gain / (FT full pool mean - mAP(0)),
the reference of the RQ5 tables ("FT full pool, 3 epochs"); None where that denominator does not exceed
--min-denominator (as bdd_pipeline.recovery).
With the revision also recovery_e12 against the 12-epoch full-pool mean, with the count and fraction of undefined
replicates; recovery_e12_determined is False where more than 5% of them are undefined (or the point is). Source:
retention = mAP_source(model) / mAP_source(source model) and
source_change = mAP_source(model) - mAP_source(source model).

Intervals: 95% percentile intervals over the defined (finite) replicates, the count of undefined replicates next to each
(<column>_undefined); a non-finite test resample of a run leaves the replicates that use it undefined. Replicate t takes
target resample t and source resample t of every run. The point's sampling seeds are redrawn with replacement ONCE per
replicate, and that draw serves both axes. The seeds of each recovery reference are redrawn ONCE per replicate and
shift, and every row of the shift divides by those same denominators, as the rungs of one curve share the ceiling redraw
in analyze_campaign.py; the point that is the reference itself (FT at N = pool for recovery_e3, FT 12 epochs for
recovery_e12) takes that draw as its own, so its recovery is 1 in every defined replicate. A replicate whose redrawn
denominator does not exceed --min-denominator is undefined; a recovery without a point value has no interval. The
replicates that a reference leaves undefined are therefore a property of the shift: retention_summary.json gives their
count per shift and reference (denominator_<reference>_undefined) and, for the 12-epoch reference,
recovery_e12_undefined_fraction and recovery_e12_determined; a row's count adds only the replicates that its own gain
leaves undefined. recovery_e12_determined is decided on the drawn replicates, as in analyze_rq5.py; next to it,
denominator_<reference>_expected_undefined_fraction is the same fraction over every equally likely redraw of the
reference seeds and every test resample, with no random draw. Where the two sit on either side of 5%, the verdict
depends on --seed: quote it only together with the expected fraction.

pareto (retention_points.csv): per shift, True when no other point has a gain and a retention at least as high, one of
them strictly. Point estimates only, descriptive; the source model and the FT full pool are included. Gain and recovery
order the points of a shift the same way, so the front does not depend on the denominator.

retention_paired.csv, per shift and N: adapt - FT and replay - FT at every N >= 10 both arms have, adapt_matched - FT
and replay_matched - FT at N = 1,000, and at N = 0 adapt - source model and replay - source model. Paired by sampling seed
over the seeds both arms have (at N = 0 every seed of the arm is paired with the one source model); the seed pairs are
redrawn together, once per replicate for both axes, and replicate t is shared by both arms. delta_target in AP and in
recovery units (divided by the point E = 3 denominator, None below the floor, no interval), delta_retention =
delta(mAP_source) / mAP_source(source model), their intervals and undefined counts, and flags (False where an interval
does not exist): dominates (both lower bounds above 0), dominated (both upper bounds below 0), trade_off (both intervals
exclude 0, with opposite signs).

Caveat, stated with the result: adapt and replay keep one labeled source image in every update (1 source + 1 target
labeled image per update); FT trains on target images only. High retention for those arms is source rehearsal by
design: the tables measure the trade-off of each recipe as run, not of adaptation alone. The replay arm, which has the
same labeled stream without adaptation, separates the two, as in RQ5.

Outputs (a table without rows writes no file and removes an earlier one): retention_points.csv, retention_paired.csv,
retention_summary.json and, unless --no-plots, retention_<shift>.png: x = recovery against "FT full pool, 3 epochs",
y = retention, one line per method through its budgets with N labels, the source model at (0, 1), Pareto points
circled, a dashed line at retention 1. A shift whose E = 3 denominator is below the floor gets no figure.

Random draws: every point row draws its seeds from a generator of its own, a child of SeedSequence(seed + 30) with
spawn key (shift index, method index, N); each recovery reference draws once per shift from a child of
SeedSequence(seed + 32) with spawn key (shift index, reference index: 0 for E = 3, 1 for E = 12); the paired rows draw
from children of SeedSequence(seed + 31) with spawn key (shift index, comparison index, N). So no row moves when another
one gains seeds, and the revision leaves the other columns of every row whose runs it does not change as they were.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import mean

import numpy as np

from analyze_campaign import LADDER_EPOCHS, budget_of, load, read_revision, require_ceiling_seeds, write_tables
from analyze_rq5 import (COLORS, MARKERS, MATCHED_ARMS, MAX_UNDEFINED_FRACTION, arm_runs, merge_replay_seeds, row_rng,
                         seeds_of, select, shared_resamples, split_revision)
from bdd_pipeline.campaign import MATCHED_DOMAINS, MATCHED_POOL, TARGETS
from bdd_pipeline.data import EXPECTED_DOMAINS

METHODS = ("source model", "FT", "FT 12 epochs", "adapt", "replay", "adapt_matched", "replay_matched")
COMPARISONS = (("adapt", "FT"), ("replay", "FT"), ("adapt_matched", "FT"), ("replay_matched", "FT"),
               ("adapt", "source model"), ("replay", "source model"))
CEILINGS = {"e3": "FT", "e12": "FT 12 epochs"}  # recovery reference -> the method whose point at N = pool it is
POINT_SEED_OFFSET = 30
PAIRED_SEED_OFFSET = 31
REFERENCE_SEED_OFFSET = 32
CAVEAT = ("adapt and replay keep one labeled source image in every update (1 source + 1 target labeled image per "
          "update); FT trains on target images only. High retention for those arms is source rehearsal by design: the "
          "tables measure the trade-off of each recipe as run, not of adaptation alone. The replay arm separates the "
          "two, as in RQ5.")
RECOVERY_REFERENCE = ("recovery_e3 and the x axis of the figures: gain / (FT full pool, 3 epochs, seed mean - mAP(0)), "
                      "the reference of the RQ5 tables; recovery_e12 (revision only): against the 12-epoch full-pool mean")
LABELS = {"source model": "source model (N = 0)", "FT": "fine-tune (FT), 3 epochs", "FT 12 epochs": "FT full pool, 12 epochs",
          "adapt": "ALDI++ adapt", "replay": "replay control",
          "adapt_matched": "ALDI++ adapt, matched passes (3,000 updates)",
          "replay_matched": "replay control, matched passes (3,000 updates)"}
HOLLOW = {"FT 12 epochs": "FT", **MATCHED_ARMS}  # drawn as hollow markers in the colour of this arm
INK, MUTED = "#1f2933", "#6b6b66"


def both_tests(domain: str, method: str, budget: int, target: dict[int, dict], source: dict[int, dict]) -> dict:
    """The runs of one point: every seed needs a reliable mAP on the shift's test set and on the source test set."""
    for test, runs, other in ((domain, target, source), ("source", source, target)):
        missing = sorted(set(other) - set(runs))
        if missing:
            raise ValueError(f"{domain} {method} N = {budget}: seed(s) {missing} have no entry on the {test} test set")
    for entry in (*target.values(), *source.values()):
        if entry["points"]["reliable"] is None:
            raise ValueError(f"{domain} {method} N = {budget}: run {entry['run']} has no reliable mAP on the "
                             f"{entry['domain']} test set")
    return {"target": target, "source": source}


def shift_points(domain: str, ft: list[dict], da: list[dict], source_model: dict[str, dict],
                 revision_ft: list[dict] | None = None, revision_da: dict[str, list[dict]] | None = None
                 ) -> dict[tuple[str, int], dict]:
    """(method, N) -> {"target": seed -> entry on the shift's test set, "source": seed -> entry on the source test set}.

    In METHODS order, N ascending. revision_ft (read_revision) adds "FT 12 epochs"; revision_da (split_revision) adds
    the further replay seeds and the matched-pass arms. Every FT run must have the locked epoch count, and both full-pool
    references need seeds 0 and 1.
    """
    tests, pool = (domain, "source"), EXPECTED_DOMAINS[domain]["train"]

    def fine_tune(entries: list[dict], test: str) -> dict[int, dict[int, dict]]:
        groups = select(entries, "fine-tune", domain, test)
        if domain in MATCHED_DOMAINS:
            groups.pop(str(MATCHED_POOL), None)  # the matched-pool runs are a ceiling of RQ4, not a rung
        return {budget_of(next(iter(runs.values()))["run"]): runs for runs in groups.values()}

    arms = {"source model": tuple({0: seeds_of([source_model[test]])} for test in tests),
            "FT": tuple(fine_tune(ft, test) for test in tests),
            "adapt": tuple(arm_runs(da, "adapt", domain, test) for test in tests)}
    for groups in arms["FT"]:
        other = {entry["run"].get("epochs") for runs in groups.values() for entry in runs.values()} - {LADDER_EPOCHS}
        if other:
            raise ValueError(f"{domain}: campaign fine-tune runs with epochs {sorted(other, key=str)}; the FT arm is the "
                             f"locked {LADDER_EPOCHS}-epoch schedule (12-epoch runs are read through --revision-scored)")
    require_ceiling_seeds(domain, "3-epoch full-pool reference", list(arms["FT"][0].get(pool, {}).values()))
    replay = []
    for test in tests:
        groups = arm_runs(da, "replay", domain, test)
        source_only = arm_runs(da, "replay", "source", test).get(0)  # the source-only replay run is N = 0 of every shift
        if source_only:
            groups[0] = source_only
        replay.append(groups)
    if revision_da is not None:
        replay = [merge_replay_seeds(groups, arm_runs(revision_da["replay_seeds"], "replay", domain, test))
                  for groups, test in zip(replay, tests)]
        for arm, stage in MATCHED_ARMS.items():
            arms[arm] = tuple(arm_runs(revision_da["matched"], stage, domain, test) for test in tests)
    arms["replay"] = tuple(replay)
    if revision_ft is not None:
        e12 = [entry for entry in revision_ft if entry["kind"] == "full-e12"]
        arms["FT 12 epochs"] = tuple(fine_tune(e12, test) for test in tests)
        require_ceiling_seeds(domain, "12-epoch full-pool ceiling", list(arms["FT 12 epochs"][0].get(pool, {}).values()))
    points = {}
    for method in (name for name in METHODS if name in arms):
        target, source = arms[method]
        for budget in sorted(set(target) | set(source)):
            points[(method, budget)] = both_tests(domain, method, budget, target.get(budget, {}), source.get(budget, {}))
    return points


def redrawn(runs: dict[int, dict], draw: np.ndarray) -> np.ndarray:
    """Seed-mean reliable mAP per replicate: replicate t averages test resample t of the seeds (sorted) drawn in draw[t]."""
    stack = np.stack([runs[seed]["boots"]["reliable"] for seed in sorted(runs)])
    return stack[draw, np.arange(stack.shape[1])[:, None]].mean(axis=1)


def reference_denominator(runs: dict[int, dict], zero: dict, draw: np.ndarray) -> tuple[float, np.ndarray]:
    """A recovery reference's denominator, seed mean - mAP(0), and per replicate with the reference seeds of draw[t]."""
    return (mean(entry["points"]["reliable"] for entry in runs.values()) - zero["points"]["reliable"],
            redrawn(runs, draw) - zero["boots"]["reliable"])


def expected_undefined_fraction(runs: dict[int, dict], zero: dict, min_denominator: float) -> float:
    """Share of the replicates whose denominator does not exceed the floor, over every ordered redraw of the reference
    seeds (each equally likely) and every test resample: what the drawn count of one --seed scatters around."""
    stack = np.stack([runs[seed]["boots"]["reliable"] for seed in sorted(runs)])
    redraws = itertools.product(range(len(stack)), repeat=len(stack))
    return float(np.mean([~(stack[list(draw)].mean(axis=0) - zero["boots"]["reliable"] > min_denominator)
                          for draw in redraws]))


def interval(name: str, replicates: np.ndarray) -> dict:
    """<name>_ci_low and <name>_ci_high, the 95% percentile interval over the finite replicates (None without one), and
    <name>_undefined, the count of the others."""
    defined = replicates[np.isfinite(replicates)]
    low, high = (float(np.percentile(defined, 2.5)), float(np.percentile(defined, 97.5))) if len(defined) else (None, None)
    return {f"{name}_ci_low": low, f"{name}_ci_high": high, f"{name}_undefined": len(replicates) - len(defined)}


def determined(defined: bool, undefined: int, replicates: int) -> dict:
    """recovery_e12_undefined_fraction, and recovery_e12_determined: defined, with at most 5% undefined replicates."""
    return {"recovery_e12_undefined_fraction": undefined / replicates,
            "recovery_e12_determined": defined and undefined / replicates <= MAX_UNDEFINED_FRACTION}


def point_row(point: dict, zero: dict, reference: dict, denominators: dict[str, tuple[float, np.ndarray]],
              min_denominator: float, rng: np.random.Generator, draw: np.ndarray | None = None) -> dict:
    """One retention_points.csv row, without its identity and Pareto flag.

    zero / reference: the source model on the shift's / the source test set. denominators: reference name ("e3", then
    "e12" with the revision) -> reference_denominator(), the same for every row of the shift. draw: the seed draw of the
    reference that this point is; without it the point draws its seeds from rng, once for both axes.
    """
    seeds = sorted(point["target"])
    zero_boot, reference_boot = zero["boots"]["reliable"], reference["boots"]["reliable"]
    replicates = len(zero_boot)
    if draw is None:
        draw = rng.integers(0, len(seeds), (replicates, len(seeds)))
    target, source = (mean(point[side][seed]["points"]["reliable"] for seed in seeds) for side in ("target", "source"))
    zero_point, reference_point = zero["points"]["reliable"], reference["points"]["reliable"]
    gain = redrawn(point["target"], draw) - zero_boot
    source_boot = redrawn(point["source"], draw)  # the same draw: one redraw of the seeds serves both axes
    row = {"seeds": len(seeds), "seed_list": " ".join(str(seed) for seed in seeds), "target_mAP": target,
           "source_mAP": source, "gain": target - zero_point, **interval("gain", gain)}
    for name, (denominator, denominator_boot) in denominators.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            recovery = np.where(denominator_boot > min_denominator, gain / denominator_boot, np.nan)
        value = row["gain"] / denominator if denominator > min_denominator else None
        columns = interval(f"recovery_{name}", recovery)
        if value is None:
            columns.update({f"recovery_{name}_ci_low": None, f"recovery_{name}_ci_high": None})
        row.update({f"recovery_{name}": value, **columns})
        if name == "e12":
            row.update(determined(value is not None, columns["recovery_e12_undefined"], replicates))
    row.update({"retention": source / reference_point, **interval("retention", source_boot / reference_boot),
                "source_change": source - reference_point, **interval("source_change", source_boot - reference_boot),
                "replicates": replicates})
    return row


def pareto(values: list[tuple[float, float]]) -> list[bool]:
    """Per (gain, retention): True when no other point is at least as high on both and strictly higher on one."""
    return [not any(g >= gain and r >= retention and (g > gain or r > retention) for g, r in values)
            for gain, retention in values]


def paired_row(left: dict, right: dict, reference: dict, denominator: float, min_denominator: float,
               rng: np.random.Generator) -> dict | None:
    """left - right paired by sampling seed over the seeds both have; None without a common seed.

    left / right: {"target", "source"} as in shift_points (for the source model: its runs under every seed of left).
    Replicate t takes test resample t of every run on both test sets and one redraw of the seed pairs, shared by the
    target and the retention difference. denominator: the point E = 3 denominator of the shift (recovery units).
    """
    seeds = sorted(set(left["target"]) & set(right["target"]))
    if not seeds:
        return None
    reference_boot = reference["boots"]["reliable"]
    draw = rng.integers(0, len(seeds), (len(reference_boot), len(seeds)))
    common = lambda runs: {seed: runs[seed] for seed in seeds}  # noqa: E731
    target, source = ((redrawn(common(left[side]), draw) - redrawn(common(right[side]), draw)) for side in ("target", "source"))
    differences = {side: mean(left[side][seed]["points"]["reliable"] - right[side][seed]["points"]["reliable"]
                              for seed in seeds) for side in ("target", "source")}
    target_columns = interval("delta_target", target)
    retention_columns = interval("delta_retention", source / reference_boot)
    target_low, target_high = target_columns["delta_target_ci_low"], target_columns["delta_target_ci_high"]
    retention_low, retention_high = retention_columns["delta_retention_ci_low"], retention_columns["delta_retention_ci_high"]
    both = target_low is not None and retention_low is not None  # the flags need an interval on both axes
    return {"seeds": len(seeds), "seed_list": " ".join(str(seed) for seed in seeds),
            "delta_target": differences["target"], **target_columns,
            "delta_target_recovery": differences["target"] / denominator if denominator > min_denominator else None,
            "delta_retention": differences["source"] / reference["points"]["reliable"], **retention_columns,
            "dominates": both and target_low > 0 and retention_low > 0,
            "dominated": both and target_high < 0 and retention_high < 0,
            "trade_off": both and ((target_low > 0 and retention_high < 0) or (target_high < 0 and retention_low > 0)),
            "replicates": len(reference_boot)}


def plot(rows: list[dict], output: Path) -> list[Path]:
    """One retention_<shift>.png per shift with a defined E = 3 denominator; returns the figures written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    written = []
    for domain in TARGETS:
        shift = [row for row in rows if row["domain"] == domain and row["recovery_e3"] is not None]
        path = output / f"retention_{domain}.png"
        if not shift:
            path.unlink(missing_ok=True)
            continue
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for method in METHODS:
            line = sorted((row for row in shift if row["method"] == method), key=lambda row: row["budget"])
            if not line:
                continue
            x, y = [row["recovery_e3"] for row in line], [row["retention"] for row in line]
            if method == "source model":
                ax.plot(x, y, ls="none", marker="*", ms=14, color=INK, label=LABELS[method], zorder=4)
            elif method in HOLLOW:
                arm = HOLLOW[method]
                ax.plot(x, y, ls="none", marker=MARKERS[arm], ms=9, mfc="white", mec=COLORS[arm], mew=1.8,
                        label=LABELS[method], zorder=4)
            else:
                ax.plot(x, y, marker=MARKERS[method], ms=8, lw=2, color=COLORS[method], mec="white", mew=1.5,
                        label=LABELS[method], zorder=3)
            for row in line:
                ax.annotate(f"{row['budget']:,}", (row["recovery_e3"], row["retention"]), textcoords="offset points",
                            xytext=(5, 5), fontsize=7.5, color=MUTED)
        front = [row for row in shift if row["pareto"]]
        ax.scatter([row["recovery_e3"] for row in front], [row["retention"] for row in front], s=260, facecolors="none",
                   edgecolors=INK, linewidths=1.2, zorder=5, label="Pareto front (gain, retention)")
        ax.axhline(1.0, color=MUTED, ls="--", lw=1)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(color="#e6e6e3", lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Recovery on the target test set (reference: FT full pool, 3 epochs)")
        ax.set_ylabel("Retention on the source test set (source model = 1)")
        ax.set_title(f"{domain}: target recovery vs source retention (seed means; labels: N)")
        ax.legend(frameon=False, fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ft-scored", type=Path, required=True, help="Scored FT campaign (evaluate_frcnn.py output)")
    parser.add_argument("--da-scored", type=Path, required=True, help="Scored RQ5 grid")
    parser.add_argument("--revision-scored", type=Path, default=None,
                        help="Scored revision runs: 12-epoch full-pool fine-tunes, further replay seeds at N = 10 and "
                             "100, matched-pass arms at N = 1,000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-denominator", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0, help="Seed for redrawing sampling seeds")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    ft, da = load(args.ft_scored), load(args.da_scored)
    scored = [args.ft_scored, args.da_scored] + ([args.revision_scored] if args.revision_scored else [])
    resamples = shared_resamples([json.loads(path.read_text(encoding="utf-8")) for directory in scored
                                  for path in sorted(directory.glob("*/*/metrics.json"))])
    with_revision = args.revision_scored is not None
    revision_ft = read_revision(args.revision_scored, args.ft_scored, ft) if with_revision else None
    revision_da = split_revision(load(args.revision_scored)) if with_revision else None
    if revision_da is not None and not any(revision_da.values()):
        raise ValueError(f"No adapt or replay run among the scored sets under {args.revision_scored}")
    sources = [entry for entry in ft if entry["run"]["stage"] == "source"]
    source_model = {entry["domain"]: entry for entry in sources}
    if len(source_model) != len(sources) or set(source_model) != {"source", *TARGETS}:
        raise ValueError("The source model must be scored exactly once on the source and on every target test set; "
                         f"found it on {', '.join(sorted(entry['domain'] for entry in sources)) or 'none'}")
    reference = source_model["source"]
    rows, paired, references = [], [], {}
    for shift, domain in enumerate(TARGETS):
        pool, zero = EXPECTED_DOMAINS[domain]["train"], source_model[domain]
        replicates = len(zero["boots"]["reliable"])
        points = shift_points(domain, ft, da, source_model, revision_ft, revision_da)
        ceilings = {name: points[(method, pool)]["target"] for name, method in CEILINGS.items() if (method, pool) in points}
        draws = {name: row_rng(args.seed + REFERENCE_SEED_OFFSET, shift, list(CEILINGS).index(name)).integers(
            0, len(runs), (replicates, len(runs))) for name, runs in ceilings.items()}
        denominators = {name: reference_denominator(runs, zero, draws[name]) for name, runs in ceilings.items()}
        shift_rows = []
        for (method, budget), point in points.items():
            own = next((draws[name] for name, owner in CEILINGS.items() if (owner, pool) == (method, budget)), None)
            rng = row_rng(args.seed + POINT_SEED_OFFSET, shift, METHODS.index(method), budget)
            shift_rows.append({"domain": domain, "method": method, "budget": budget,
                               **point_row(point, zero, reference, denominators, args.min_denominator, rng, own)})
        for row, flag in zip(shift_rows, pareto([(row["gain"], row["retention"]) for row in shift_rows])):
            row["pareto"] = flag
        rows += shift_rows
        references[domain] = {"mAP_0": zero["points"]["reliable"], "source_model_on_source": reference["points"]["reliable"]}
        for name, runs in ceilings.items():
            denominator, denominator_boot = denominators[name]
            undefined = int(np.count_nonzero(~(denominator_boot > args.min_denominator)))
            references[domain].update({f"ceiling_{name}": mean(entry["points"]["reliable"] for entry in runs.values()),
                                       f"denominator_{name}": denominator, f"seeds_{name}": sorted(runs),
                                       f"denominator_{name}_undefined": undefined,
                                       f"denominator_{name}_expected_undefined_fraction":
                                           expected_undefined_fraction(runs, zero, args.min_denominator)})
            if name == "e12":
                references[domain].update(determined(denominator > args.min_denominator, undefined, replicates))
        denominator = references[domain]["denominator_e3"]
        for index, (left, right) in enumerate(COMPARISONS):
            for budget in sorted(budget for method, budget in points if method == left):
                if right == "source model" and budget == 0:
                    seeds = points[(left, 0)]["target"]
                    other = {"target": dict.fromkeys(seeds, zero), "source": dict.fromkeys(seeds, reference)}
                elif right != "source model" and budget > 0 and (right, budget) in points:
                    other = points[(right, budget)]
                else:
                    continue
                result = paired_row(points[(left, budget)], other, reference, denominator, args.min_denominator,
                                    row_rng(args.seed + PAIRED_SEED_OFFSET, shift, index, budget))
                if result is not None:
                    paired.append({"domain": domain, "budget": budget, "comparison": f"{left} - {right}", **result})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_tables(args.output_dir, {"retention_points": rows, "retention_paired": paired})
    front = {domain: [{key: row[key] for key in ("method", "budget", "gain", "retention")}
                      for row in rows if row["domain"] == domain and row["pareto"]] for domain in TARGETS}
    summary = {"rules": ".agent/plans/active/2026-11-24-direction-1-analyses.md, section 3 (descriptive)",
               "caveat": CAVEAT, "recovery_reference": RECOVERY_REFERENCE, "min_denominator": args.min_denominator,
               "max_undefined_fraction": MAX_UNDEFINED_FRACTION, "revision": with_revision, "test_resamples": resamples,
               "references": references, "pareto_front": front, "paired": paired}
    (args.output_dir / "retention_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.no_plots:
        plot(rows, args.output_dir)
    print(f"{len(rows)} points, {len(paired)} paired comparisons: {args.output_dir}")
    signed = lambda value: "none" if value is None else f"{value:+.4f}"  # noqa: E731
    for row in paired:
        reading = ("dominates" if row["dominates"] else "dominated" if row["dominated"]
                   else "trade-off" if row["trade_off"] else "no clear order")
        print(f"  {row['domain']} N={row['budget']} {row['comparison']} (seeds {row['seed_list']}): target "
              f"{row['delta_target']:+.4f} [{signed(row['delta_target_ci_low'])}, {signed(row['delta_target_ci_high'])}], "
              f"retention {row['delta_retention']:+.4f} [{signed(row['delta_retention_ci_low'])}, "
              f"{signed(row['delta_retention_ci_high'])}]: {reading}")
    print(f"Caveat: {CAVEAT}")


if __name__ == "__main__":
    main()
