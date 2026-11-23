"""RQ5 tables and figures: ALDI++ adaptation against plain fine-tuning (FT) on the final test set.

Inputs are two one-shot scoring outputs of `evaluate_frcnn.py --final-plan` (schema 3) that share
the test resamples (bootstrap seed 20261030, same final-test image lists): the FT campaign and the
RQ5 grid. Rules: .agent/DECISIONS.md 2026-11-03 (written before any RQ5 result) and, for the replay
share and the comparison against no fine-tuning, .agent/DECISIONS.md 2026-11-21 (lecturer review;
written after the existing curves had been seen, so descriptive, not confirmatory). With
--revision-scored a third scoring output is read, the revision runs (see "Revision runs" below).

Arms: "adapt" is ALDI++ (labeled stream plus distillation on unlabeled target images), "replay" is
the control (same labeled stream, strong augmentation and EMA, no unlabeled stream, no
distillation), "FT" is plain fine-tuning.

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
- Replay share (2026-11-21, the control-arm decomposition): share(N) = (replay - FT) / (adapt - FT)
  on reliable mAP, over the sampling seeds common to all three arms, paired by seed; at N = 0 the
  reference is the source model and the row is tabulated separately. Undefined where
  |adapt - FT| < delta, for the point value and for each replicate. CI: test resample t shared by
  all arms, seeds redrawn with replacement as triples; undefined replicates are counted and left
  out of the percentiles, so the interval is conditional on the replicates where the share exists.
  The comparison with delta is made on floats: a gain equal to delta on paper (0.205 - 0.200) can
  fall on either side.
- Claim rule (2026-11-21, addendum 2): "most of the gain over fine-tuning is not adaptation" is
  supported at N = 10 or N = 100 when, in all three shifts and with three replay seeds,
  adapt - FT > 0, share(N) > 0.5 with the lower 95% bound above 0.5, and at most 5% of the
  replicates leave the share undefined. It is not stated for N = 0 or N = 1000: a share of a
  deficit says where the deficit sits and is not evidence for the headline. `quotable` marks the
  shares that may be cited with their interval (defined, at most 5% undefined replicates).
- Share against the source model (addendum 2), rq5_replay_share_vs_source.csv: the same ratio with
  the source model in place of FT, at N = 10, 100, 1000. FT at N <= 50 is below the source model,
  so at N = 10 the two references read differently: against FT the control reproduces avoided
  damage, against the source model it reproduces a gain only if replay itself beats zero-shot.
- Against no fine-tuning (2026-11-21): seed-mean mAP(N) - mAP(0) for FT, adapt and replay with CI,
  because FT at N <= 50 is below zero-shot. The adapt row at N = 0 is the estimand of the N = 0 row
  of rq5_delta.csv with the seeds redrawn by another generator, so its bounds differ in the fourth
  decimal: cite that interval from rq5_delta.csv.
- Secondary: per-class Delta (four reliable classes), forgetting on the source test set.
- Shared test resamples: every interval pairs replicate t across runs, so main() stops unless all
  scored sets have one bootstrap seed and resample count and one image count per domain.
- Random draws: the 2026-11-03 tables draw from default_rng(seed) and default_rng(seed + 1) in a
  fixed order. The 2026-11-21 tables draw only from children of SeedSequence(seed + 2) (replay
  share: one per shift and N) and SeedSequence(seed + 3) (against no fine-tuning: one per shift,
  arm and N). The earlier numbers stay bit-identical, and no row of the new tables changes when
  another row gains seeds. The share against the source model draws from children of
  SeedSequence(seed + 4).

Revision runs (--revision-scored; .agent/DECISIONS.md 2026-11-21 "RQ5 matched passes", "Replay
seeds" and "Claim rules", and addendum 2; all written before these runs were scored). Without the
option every output is what it was before the revision runs existed. With it, only the adapt and
replay entries of that directory are read (its fine-tune ceilings belong to analyze_campaign.py),
and they are told apart by fields of metrics["run"] alone (stage, domain, budget, seed,
total_updates), never by a folder name or label: many of them share stage, domain, budget and seed
with a run of the RQ5 grid. An adapt or replay entry of neither kind below stops the script.

- Replay seeds: replay at N = 10 or 100 with 2,000 updates is a further sampling seed of the
  original replay arm (seed 0 is in the RQ5 grid). The merged arm feeds rq5_replay_share.csv,
  rq5_replay_share_vs_source.csv, the headline check, the replay rows of rq5_vs_zero_shot.csv and
  the replay line of the plots, which therefore use three seeds at N = 10 and 100. The original
  replay at N = 0 and N = 1,000 keeps seed 0 only, and those rows do not change.
- Matched passes: adapt and replay at N = 1,000 with 3,000 updates are arms of their own,
  "adapt_matched" and "replay_matched", never pooled with the 2,000-update runs. One labeled target
  image enters each update, so 3,000 updates give the labeled target image-passes of FT (3 epochs
  of 1,000 images). The original runs stay the pre-registered result; the matched runs are the
  control of that confound.
  - rq5_matched_passes.csv (reliable mAP) and rq5_matched_passes_per_class.csv, per shift:
    adapt_matched - FT, replay_matched - FT, adapt_matched - replay_matched, each matched arm minus
    its original and, for reference, adapt_original - FT. Each difference is paired by sampling
    seed over the seeds present in both arms; CI as for Delta (replicate t takes test resample t
    for both arms, and the paired seeds are redrawn together). left and right are seed means over
    those paired seeds. The original replay at N = 1,000 has seed 0 only, so replay_matched -
    replay_original rests on one seed. The adapt_original - FT row is the estimand of the
    N = 1,000 row of rq5_delta.csv with the seeds redrawn by another generator, so its bounds
    differ in the fourth decimal: cite that interval from rq5_delta.csv, the pre-registered table.
  - Matched-pass rule ("Claim rules", "Matched passes"), applied per shift to adapt_matched - FT on
    reliable mAP: "FT is ahead from N = 1,000" stands (ft_ahead_stands) when the difference is
    <= -delta and the upper 95% bound is below 0. Otherwise the crossover claim is withdrawn and
    the reading is that there is no meaningful difference at N = 1,000. The rule names no seed
    count, so the reading is given on the seeds paired between the two arms, and seeds and
    seed_list are reported with it. A shift without such a pair is listed in summary.json and in
    the printed summary as not evaluated; it never passes silently. A non-finite test resample in
    the two arms of that row stops the script: the reading needs every test resample.
  - rq5_replay_share_matched.csv: the replay share at N = 1,000 from adapt_matched and
    replay_matched, against FT and against the source model, three seeds. A three-seed share at
    N = 1,000 exists only under matched passes. Against FT it is a share of a deficit wherever FT
    is ahead: it says where the deficit sits and is not evidence for the headline (addendum 2).
    Its gain against FT is the estimand of the adapt_matched - FT row with the seeds redrawn by
    another generator, so the bounds differ in the fourth decimal: the rule reads, and the report
    cites, the interval of rq5_matched_passes.csv.
    undefined_replicates counts one random redraw of the seeds per test resample, so the quotable
    flag of a share whose undefined fraction is near the 5% limit depends on --seed.
    expected_undefined_fraction is that fraction over every redraw of the paired seeds, with no
    random draw: quote the flag of such a share only together with it. quotable itself stays the
    flag of addendum 2 on the drawn replicates.
  - rq5_vs_zero_shot.csv gains the rows of the two matched arms at N = 1,000.
  - rq5_forgetting_revision.csv: source-test reliable mAP of the merged replay arm at N = 10 and
    100 and of the two matched arms, in the format of rq5_forgetting.csv.
- EMA lag: the scored model of the adapt and replay arms is the EMA teacher (alpha = 0.9996), not
  the student. After 3,000 updates the teacher still carries 0.9996^3000 = 0.30 of the source
  initialization (0.45 after 2,000). Matching the labeled passes therefore equalizes the student's
  exposure, not the teacher's, and the matched-pass result is to be read with that.
- The five 2026-11-03 tables (rq5_delta, rq5_delta_per_class, rq5_equivalent_budget,
  rq5_attribution, rq5_forgetting) describe the original, pre-registered runs only and are the same
  with and without the option; the attribution stays seed 0 by definition. summary.json keeps
  every key; its replay shares and headline entries then rest on the merged replay arm, and it
  gains the matched-pass rows, their reading per shift and the matched-pass shares.
- Ceiling of these outputs: "ceiling" and "recovery_denominator" of summary.json,
  delta_recovery_units of rq5_delta.csv and the dashed line of the figures are the seed mean of the
  3-epoch full-pool fine-tune, the reference the 2026-11-03 rules were written against. They are
  not mAP_c of DECISIONS 2026-11-21 "Ceiling" (the 12-epoch full-pool mean, read by
  analyze_campaign.py), under which the 3-epoch full pool is the top rung of the ladder. This
  script does not read the 12-epoch runs; with the option, summary.json says so under
  "ceiling_reference".
- Output directory: a table without rows writes no file. The script stops before writing anything
  when such a table of an earlier run is still in --output-dir (for example the revision tables
  after a run without the option), so one directory never mixes two runs.
- Random draws of the revision tables: children of SeedSequence(seed + 5) (matched passes: one per
  shift and comparison), SeedSequence(seed + 6) (the same per class: one per shift, class and
  comparison) and SeedSequence(seed + 7) (matched-pass share: one per shift and reference). The
  matched rows of rq5_vs_zero_shot.csv take the children of SeedSequence(seed + 3) of two arm
  indices that did not exist before. expected_undefined_fraction enumerates the redraws and takes
  no draw. No generator of an earlier table gains or loses a draw, so the rows whose inputs did
  not change are identical with and without the option: the changed rows are the replay shares at
  N = 10 and 100 (both references) and the replay rows of rq5_vs_zero_shot.csv at N = 10 and 100,
  which now redraw three seeds.
"""
from __future__ import annotations

import argparse
import itertools
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
ARMS = ("FT", "adapt", "replay")
DELTA = 0.005
ATTRIBUTION_BUDGETS = (0, 10, 100, 1000)
HEADLINE_BUDGETS = (10, 100)  # small labeled budgets only (DECISIONS 2026-11-21, addendum 2)
HEADLINE_REPLAY_SEEDS = 3
MAX_UNDEFINED_FRACTION = 0.05  # a share is quoted only when at most this fraction of its replicates is undefined
SOURCE_REFERENCE_BUDGETS = tuple(n for n in ATTRIBUTION_BUDGETS if n > 0)
CLAIM_RULE = ("DECISIONS 2026-11-21 with addendum 2: 'most of the gain over fine-tuning is not adaptation' is supported "
              "at budget N in {10, 100} when, in all three shifts and with three replay seeds, adapt - FT > 0, "
              "share(N) > 0.5 with the lower 95% bound above 0.5, and at most 5% of the replicates leave the share "
              "undefined. share(N) = (replay - FT) / (adapt - FT) on reliable mAP, undefined where "
              f"|adapt - FT| < {DELTA}. The same share against the source model is tabulated next to it: FT at "
              "N <= 50 is below the source model, so the two references read differently at N = 10.")
KEYS = ("reliable",) + RELIABLE_CLASSES
# Reference categorical slots 1-3 (dataviz palette, validated light: PASS; aqua contrast WARN,
# relieved by distinct markers, the legend and the CSV tables).
COLORS = {"FT": "#2a78d6", "adapt": "#eb6834", "replay": "#1baf7a"}
MARKERS = {"FT": "o", "adapt": "s", "replay": "^"}
# Revision runs (DECISIONS 2026-11-21, "Replay seeds" and "RQ5 matched passes"), told apart by run fields only.
REPLAY_SEED_BUDGETS = (10, 100)  # the original replay arm gains sampling seeds at these budgets
ORIGINAL_UPDATES = 2000  # run length of the original grid at N <= 1,000: max(2000, 2N)
MATCHED_BUDGET = 1000
MATCHED_UPDATES = 3000  # one labeled target image per update: the labeled image-passes of FT's 3 epochs at N = 1,000
MATCHED_ARMS = {"adapt_matched": "adapt", "replay_matched": "replay"}  # matched-pass arm -> its stage
# Generator index of an arm in rq5_vs_zero_shot.csv. The first three positions are those of ARMS and must not move.
STREAM_ARMS = ARMS + tuple(MATCHED_ARMS)
MATCHED_COMPARISONS = (("adapt_matched", "FT"), ("replay_matched", "FT"), ("adapt_matched", "replay_matched"),
                       ("adapt_matched", "adapt_original"), ("replay_matched", "replay_original"), ("adapt_original", "FT"))
PRE_REGISTERED_COMPARISON = "adapt_matched - FT"
READINGS = {True: "FT ahead from N = 1,000 stands", False: "crossover withdrawn: no meaningful difference at N = 1,000"}
NOT_EVALUATED = "not evaluated: no adapt_matched and FT run share a sampling seed at N = 1,000"
CEILING_REFERENCE = ("'ceiling' and 'recovery_denominator' under 'domains', delta_recovery_units of rq5_delta.csv and the dashed "
                     "line of the figures are the seed mean of the 3-epoch full-pool fine-tune, the reference of the 2026-11-03 "
                     "rules. They are not mAP_c of DECISIONS 2026-11-21, 'Ceiling' (the 12-epoch full-pool mean), which "
                     "analyze_campaign.py reports and under which the 3-epoch full pool is the top rung of the ladder. This "
                     "script does not read the 12-epoch runs.")
MATCHED_PASS_RULE = ("DECISIONS 2026-11-21, 'Claim rules', 'Matched passes', applied per shift to adapt_matched - FT at "
                     f"N = 1,000 on reliable mAP (adapt and replay run for {MATCHED_UPDATES} updates, so that labeled target "
                     "image-passes equal fine-tuning's): 'FT is ahead from N = 1,000' stands when the difference is "
                     f"<= -{DELTA} and the upper 95% bound is below 0; otherwise the crossover claim is withdrawn and there "
                     "is no meaningful difference at N = 1,000. The scored model of the adapt and replay arms is the EMA "
                     "teacher, which after 3,000 updates still carries 0.9996^3000 = 0.30 of the source initialization "
                     "(0.45 after 2,000): matching labeled passes equalizes the student's exposure, not the teacher's.")
MATCHED_LABELS = {"adapt_matched": "ALDI++ adapt, matched passes (3,000 updates)",
                  "replay_matched": "replay control, matched passes (3,000 updates)"}


def tail(confidence: float = 0.95) -> float:
    return 100 * (1 - confidence) / 2


def row_rng(seed: int, *row: int) -> np.random.Generator:
    """Generator of one table row: child `row` of SeedSequence(seed), so its draws do not depend on any other row."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=row))


def shared_resamples(metrics: list[dict]) -> dict:
    """Bootstrap seed, resample count and image count per domain shared by every scored set; ValueError if they differ.

    Every interval pairs test resample t across runs. That holds only when each set was resampled with the same seed and
    count over the same image list of its domain, which the loader does not check. `metrics` holds the metrics.json
    contents of every scored set. Equal image counts are necessary for a shared image list, not sufficient.
    """
    plans = {(m["bootstrap"]["seed"], m["bootstrap"]["resamples"]) for m in metrics}
    if len(plans) != 1:
        raise ValueError(f"Scored sets do not share one bootstrap (seed, resamples): {sorted(plans, key=str)}")
    images = defaultdict(set)
    for m in metrics:
        images[m["domain"]].add(m["images"])
    mixed = {domain: sorted(counts) for domain, counts in images.items() if len(counts) != 1}
    if mixed:
        raise ValueError(f"Scored sets of one domain differ in image count, so their test resamples are not shared: {mixed}")
    seed, resamples = plans.pop()
    return {"seed": seed, "resamples": resamples, "scored_sets": len(metrics),
            "images": {domain: counts.pop() for domain, counts in sorted(images.items())}}


def seeds_of(entries: list[dict]) -> dict[int, dict]:
    out = {}
    for entry in entries:
        seed = int(entry["run"]["seed"])
        if seed in out:
            raise ValueError(f"Duplicate seed {seed}: {entry['run']}")
        out[seed] = entry
    return out


def select(entries: list[dict], stage: str, run_domain: str, eval_domain: str) -> dict:
    """Scored entries of one stage and training domain on one test domain: budget as recorded -> seed -> entry."""
    groups = defaultdict(list)
    for e in entries:
        run = e["run"]
        if run["stage"] == stage and run["domain"] == run_domain and e["domain"] == eval_domain:
            groups[run["budget"]].append(e)
    return {b: seeds_of(es) for b, es in groups.items()}


def arm_runs(entries: list[dict], stage: str, run_domain: str, eval_domain: str) -> dict[int, dict[int, dict]]:
    """select() for an arm whose budgets are all numbers: budget -> seed -> entry."""
    return {int(b): runs for b, runs in select(entries, stage, run_domain, eval_domain).items()}


def revision_kind(run: dict) -> str | None:
    """Kind of one revision run, read from its run fields only (DECISIONS 2026-11-21, "Replay seeds", "RQ5 matched passes").

    "replay_seeds": replay on a target shift at N = 10 or 100 with 2,000 updates, a further sampling seed of the original
    replay arm. "matched": adapt or replay on a target shift at N = 1,000 with 3,000 updates, a matched-pass arm.
    None: neither adapt nor replay (the fine-tune runs of the revision belong to analyze_campaign.py). Any other adapt or
    replay run is an error: read as either kind it would enter an arm it does not belong to.
    """
    stage = run["stage"]
    if stage not in MATCHED_ARMS.values():
        return None
    budget = int(run["budget"]) if str(run["budget"]).isdigit() else None
    updates = run.get("total_updates")
    if run["domain"] in TARGETS:
        if stage == "replay" and budget in REPLAY_SEED_BUDGETS and updates == ORIGINAL_UPDATES:
            return "replay_seeds"
        if budget == MATCHED_BUDGET and updates == MATCHED_UPDATES:
            return "matched"
    raise ValueError(f"Revision {stage} run is neither a replay seed (replay, N in {REPLAY_SEED_BUDGETS}, {ORIGINAL_UPDATES} "
                     f"updates) nor a matched-pass run (N = {MATCHED_BUDGET}, {MATCHED_UPDATES} updates) on a target shift: "
                     f"domain {run['domain']!r}, budget {run['budget']!r}, seed {run.get('seed')!r}, total_updates {updates!r}")


def split_revision(entries: list[dict]) -> dict[str, list[dict]]:
    """Adapt and replay entries of the scored revision runs by kind; every other stage is left out.

    A run has one entry per test domain it was scored on. Two entries with the same (stage, domain, budget, total_updates,
    seed) on the same test domain are an error: the folder name is then the only thing that tells them apart. So is an
    entry without a reliable mAP (a reliable class has no value): every revision table is a seed mean of that number.
    """
    kinds, seen = {"replay_seeds": [], "matched": []}, set()
    for entry in entries:
        run = entry["run"]
        kind = revision_kind(run)
        if kind is None:
            continue
        identity = (run["stage"], run["domain"], int(run["budget"]), run["total_updates"], int(run["seed"]))
        if (identity, entry["domain"]) in seen:
            raise ValueError("Duplicate revision run (stage, domain, budget, total_updates, seed) scored on "
                             f"{entry['domain']!r}: {identity}")
        if entry["points"]["reliable"] is None:
            raise ValueError(f"Revision run without a reliable mAP on {entry['domain']!r}: {identity}")
        seen.add((identity, entry["domain"]))
        kinds[kind].append(entry)
    return kinds


def merge_replay_seeds(original: dict[int, dict[int, dict]], extra: dict[int, dict[int, dict]]) -> dict[int, dict[int, dict]]:
    """Original replay arm with its further sampling seeds: budget -> seed -> entry, seeds ascending.

    Budgets without further seeds keep their original runs. A seed present on both sides is an error, and so is a budget
    whose merged runs do not share one run length: their seed mean would belong to neither length.
    """
    merged = {budget: dict(runs) for budget, runs in original.items()}
    for budget, runs in extra.items():
        group = merged.setdefault(budget, {})
        repeated = sorted(set(group) & set(runs))
        if repeated:
            raise ValueError(f"Replay seeds {repeated} at N = {budget} are in the original arm and in the revision runs")
        group.update(runs)
        lengths = {entry["run"].get("total_updates") for entry in group.values()}
        if len(lengths) != 1:
            raise ValueError(f"Replay runs at N = {budget} differ in total_updates and cannot form one arm: "
                             f"{sorted(lengths, key=str)}")
        merged[budget] = dict(sorted(group.items()))
    return merged


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


def replay_share(adapt: dict[int, dict], replay: dict[int, dict], ft: dict[int, dict], rng: np.random.Generator,
                 zero: dict | None = None, key: str = "reliable", delta: float = DELTA) -> dict | None:
    """Share of the adapt - reference gain that the replay control reproduces (DECISIONS 2026-11-21).

    The reference is FT at the same sampling seed, or the source model for every seed when zero is
    given (N = 0). Only seeds present in every arm are used. gain = adapt - reference,
    replay_part = replay - reference, distill_part = adapt - replay, each a seed mean.
    share = replay_part / gain where |gain| >= delta, else undefined (None; NaN per replicate).
    Replicate t takes test resample t for all arms and one redraw of the seeds, applied to the
    whole (adapt, replay, reference) triple. A non-finite bootstrap replicate is an error, so
    undefined_replicates counts only the replicates with |gain| < delta.
    """
    if not delta > 0:
        raise ValueError(f"delta must be positive, got {delta}")
    seeds = sorted(set(adapt) & set(replay) if zero is not None else set(adapt) & set(replay) & set(ft))
    if not seeds:
        return None
    arms = {"adapt": adapt, "replay": replay, "reference": {s: zero if zero is not None else ft[s] for s in seeds}}
    if any(group[s]["points"][key] is None for group in arms.values() for s in seeds):
        return None
    points = {name: [group[s]["points"][key] for s in seeds] for name, group in arms.items()}
    boots = {name: np.stack([group[s]["boots"][key] for s in seeds]) for name, group in arms.items()}  # [seed, replicate]
    broken = [name for name, stack in boots.items() if not np.isfinite(stack).all()]
    if broken:
        raise ValueError(f"Non-finite bootstrap replicates of {key!r} in {broken}: the share needs every test resample")
    replicates = boots["adapt"].shape[1]
    draws = rng.integers(0, len(seeds), (replicates, len(seeds)))  # the same drawn seeds for the three arms
    out, reps = {"seeds": len(seeds), "seed_list": seeds}, {}
    for name, left, right in (("gain", "adapt", "reference"), ("replay_part", "replay", "reference"),
                              ("distill_part", "adapt", "replay")):
        reps[name] = (boots[left] - boots[right])[draws, np.arange(replicates)[:, None]].mean(axis=1)
        out.update({name: mean(a - b for a, b in zip(points[left], points[right])),
                    f"{name}_ci_low": float(np.nanpercentile(reps[name], tail())),
                    f"{name}_ci_high": float(np.nanpercentile(reps[name], 100 - tail()))})
    share = out["replay_part"] / out["gain"] if abs(out["gain"]) >= delta else None
    with np.errstate(divide="ignore", invalid="ignore"):
        share_reps = np.where(np.abs(reps["gain"]) >= delta, reps["replay_part"] / reps["gain"], np.nan)
    undefined = int(np.isnan(share_reps).sum())
    interval = share is not None and undefined < replicates
    ci_low = float(np.nanpercentile(share_reps, tail())) if interval else None
    ci_high = float(np.nanpercentile(share_reps, 100 - tail())) if interval else None
    out.update({"share": share, "share_ci_low": ci_low, "share_ci_high": ci_high, "undefined_replicates": undefined,
                "replicates": replicates, "direction": "advantage" if out["gain"] > 0 else "deficit",
                "majority_not_adaptation": share is not None and ci_low is not None and share > 0.5 and ci_low > 0.5})
    return out


def quotable(row: dict) -> bool:
    """A share may be quoted with its interval: it is defined and at most 5% of its replicates are undefined."""
    return row["share"] is not None and row["undefined_replicates"] <= MAX_UNDEFINED_FRACTION * row["replicates"]


def headline_check(rows: list[dict], budget: int, seeds_required: int = HEADLINE_REPLAY_SEEDS) -> dict:
    """Claim rule of DECISIONS 2026-11-21 (addendum 2) at one labeled budget, over the replay-share rows of every shift.

    `supported` needs, in every shift: share and its lower bound above 0.5, the required replay seeds, a gain to share
    out (adapt - FT > 0) and at most 5% undefined replicates. direction is "advantage" or "deficit" when the three shifts
    agree, "mixed" when they do not, None when a shift is missing; a share of a deficit says where the deficit sits and
    never supports the headline. largest_undefined_fraction is, over the shifts, the largest fraction of replicates left
    out of a share interval. share_majority_all_shifts is the share condition alone.
    """
    found = {r["domain"]: r for r in rows if r["budget"] == budget}
    seeds = min(found[d]["seeds"] if d in found else 0 for d in TARGETS)  # paired seeds in the weakest shift
    majority = all(d in found and found[d]["majority_not_adaptation"] for d in TARGETS)
    complete = all(d in found for d in TARGETS)
    directions = {found[d]["direction"] for d in TARGETS} if complete else set()
    direction = None if not complete else directions.pop() if len(directions) == 1 else "mixed"
    undefined = max(found[d]["undefined_replicates"] / found[d]["replicates"] for d in TARGETS) if complete else None
    supported = (majority and seeds >= seeds_required and direction == "advantage"
                 and undefined is not None and undefined <= MAX_UNDEFINED_FRACTION)
    return {"share_majority_all_shifts": majority, "replay_seeds": seeds, "supported": supported,
            "direction": direction, "largest_undefined_fraction": undefined}


def versus_zero_shot(zero: dict, arms: dict[str, dict[int, dict[int, dict]]], budgets: list[int],
                     stream: tuple[int, ...], key: str = "reliable") -> list[dict]:
    """Seed-mean mAP(N) - mAP(0) per arm and budget: what each arm buys over not fine-tuning at all.

    stream is (seed, shift index). Each row redraws its seeds from its own child (arm index, N) of that stream, so an
    interval does not move when another arm, budget or shift gains seeds. The arm index is the position in STREAM_ARMS:
    FT, adapt and replay keep 0, 1 and 2, and the matched-pass arms of the revision runs follow.
    """
    rows = []
    for arm, groups in arms.items():
        for n in budgets:
            result = (paired_delta(groups[n], {}, key, row_rng(*stream, STREAM_ARMS.index(arm), n), zero=zero)
                      if groups.get(n) else None)
            if result is None:
                continue
            rows.append({"arm": arm, "budget": n, "seeds": result["seeds"],
                         "mAP": mean(e["points"][key] for e in groups[n].values()), "mAP_0": zero["points"][key],
                         "gain_vs_zero_shot": result["estimate"], "ci_low": result["ci_low"], "ci_high": result["ci_high"]})
    return rows


def matched_pass_reading(difference: float, ci_high: float, delta: float = DELTA) -> dict:
    """Pre-registered reading of adapt_matched - FT at N = 1,000 in one shift (DECISIONS 2026-11-21, "Matched passes").

    "FT is ahead from N = 1,000" stands when the estimate is at most -delta and the upper 95% bound is below 0. In every
    other case, a positive difference included, the crossover claim is withdrawn.
    """
    stands = bool(difference <= -delta and ci_high < 0)
    return {"ft_ahead_stands": stands, "reading": READINGS[stands]}


def matched_pass_rows(arms: dict[str, dict[int, dict]], stream: tuple[int, ...], key: str = "reliable",
                      reading: bool = False, delta: float = DELTA) -> list[dict]:
    """The matched-pass comparisons of one shift at N = 1,000 (DECISIONS 2026-11-21, "RQ5 matched passes").

    arms maps adapt_matched, replay_matched, adapt_original, replay_original and FT to seed -> entry. Every comparison of
    MATCHED_COMPARISONS is left - right, paired by sampling seed over the seeds present in both arms, with the interval
    of paired_delta: replicate t takes test resample t for both arms and the paired seeds are redrawn together. left and
    right are seed means over the paired seeds. A comparison without a common seed, or with a missing value, has no row.
    Each comparison redraws its seeds from its own child (comparison index) of stream, so no row moves when an arm of
    another comparison gains seeds. reading adds the pre-registered reading to the adapt_matched - FT row; a non-finite
    bootstrap replicate in either arm of that row is then an error, since its bound would rest on part of the resamples.
    """
    rows = []
    for index, (left, right) in enumerate(MATCHED_COMPARISONS):
        result = paired_delta(arms[left], arms[right], key, row_rng(*stream, index))
        if result is None:
            continue
        seeds = sorted(set(arms[left]) & set(arms[right]))
        row = {"comparison": f"{left} - {right}", "seeds": result["seeds"], "seed_list": seeds,
               "left": mean(arms[left][s]["points"][key] for s in seeds),
               "right": mean(arms[right][s]["points"][key] for s in seeds),
               "difference": result["estimate"], "ci_low": result["ci_low"], "ci_high": result["ci_high"]}
        if reading and row["comparison"] == PRE_REGISTERED_COMPARISON:
            broken = [arm for arm in (left, right) if not all(np.isfinite(arms[arm][s]["boots"][key]).all() for s in seeds)]
            if broken:
                raise ValueError(f"Non-finite bootstrap replicates of {key!r} in {broken}: the matched-pass reading needs "
                                 "every test resample")
            row.update(matched_pass_reading(result["estimate"], result["ci_high"], delta))
        rows.append(row)
    return rows


def readings_by_shift(rows: list[dict]) -> dict[str, dict]:
    """Pre-registered reading of every target shift, from the rows of rq5_matched_passes.csv (each with its domain).

    seeds and seed_list say how many sampling seeds the reading rests on. A shift without an adapt_matched - FT row keeps
    its place with ft_ahead_stands None and the reading NOT_EVALUATED, so a missing shift cannot pass for a result.
    """
    found = {row["domain"]: row for row in rows if row["comparison"] == PRE_REGISTERED_COMPARISON}
    keys = ("difference", "ci_low", "ci_high", "seeds", "seed_list", "ft_ahead_stands", "reading")
    return {domain: {key: found[domain][key] for key in keys} if domain in found
            else {**dict.fromkeys(keys), "seeds": 0, "seed_list": [], "reading": NOT_EVALUATED} for domain in TARGETS}


def expected_undefined_fraction(differences: np.ndarray, delta: float = DELTA) -> float:
    """Expected fraction of undefined share replicates over every redraw of the seeds, with no random draw.

    differences is [seed, replicate]: adapt - reference at each paired seed and test resample. A replicate of
    replay_share redraws as many seeds as there are, with replacement, and is undefined where the mean difference of the
    drawn seeds is within delta of 0; its undefined_replicates is one draw per test resample. This is the mean over the
    test resamples and over all seeds^seeds ordered redraws, the value that count scatters around.
    """
    seeds = len(differences)
    redraws = itertools.product(range(seeds), repeat=seeds)
    return sum(float((np.abs(differences[list(draw)].mean(axis=0)) < delta).mean()) for draw in redraws) / seeds ** seeds


def matched_share_rows(adapt: dict[int, dict], replay: dict[int, dict], ft: dict[int, dict], zero: dict,
                       stream: tuple[int, ...]) -> list[dict]:
    """Replay share of one shift at N = 1,000 from the matched-pass arms, against FT and against the source model.

    The share, its interval and its flags are those of replay_share. Each reference redraws its seeds from its own child
    (0: FT, 1: source model) of stream. quotable is decided on the drawn replicates (addendum 2), so it can change with
    the stream when the undefined fraction is near the limit; expected_undefined_fraction is the same fraction over
    every redraw of the paired seeds and does not depend on the stream.
    """
    rows = []
    for index, (reference, group, source) in enumerate((("FT", ft, None), ("source model", {}, zero))):
        share = replay_share(adapt, replay, group, row_rng(*stream, index), zero=source)
        if share is not None:
            gains = np.stack([adapt[s]["boots"]["reliable"] - (group[s] if source is None else source)["boots"]["reliable"]
                              for s in share["seed_list"]])
            rows.append({"budget": MATCHED_BUDGET, "labeled_budget": True, "reference": reference, **share,
                         "quotable": quotable(share), "expected_undefined_fraction": expected_undefined_fraction(gains)})
    return rows


def source_test_rows(groups: list[tuple[str, int, dict[int, dict]]], reference: float) -> list[dict]:
    """Seed-mean reliable mAP on the source test set and its change against the source model, as rq5_forgetting.csv.

    groups holds (stage, budget, seed -> entry scored on the source test set); a group without runs has no row.
    """
    rows = []
    for stage, budget, runs in groups:
        if runs:
            value = mean(e["points"]["reliable"] for e in runs.values())
            rows.append({"stage": stage, "budget": budget, "seeds": len(runs), "source_reliable_mean": value,
                         "change_vs_source_model": value - reference})
    return rows


def stale_tables(directory: Path, tables: dict[str, list[dict]]) -> list[str]:
    """Files of tables that have no rows in this run and are still in the directory from an earlier one.

    write_csv writes nothing for a table without rows, so such a file would stay next to the outputs of this run.
    """
    return [f"{name}.csv" for name, rows in tables.items() if not rows and (directory / f"{name}.csv").exists()]


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
    """One figure per shift. The keys "replay_label", "matched" and "ceiling_label" exist only with the revision runs:
    the label of the merged replay line, matched-pass arm -> seed-mean mAP at N = 1,000 (a hollow marker of the arm's
    colour), and the name of the dashed line, which is then the 3-epoch full-pool mean and no longer the ceiling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for domain, data in curves.items():
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        position = lambda n: 3 if n == 0 else n  # noqa: E731  N=0 drawn at x=3, labelled "0"
        for arm, label in (("FT", "fine-tune (FT)"), ("adapt", "ALDI++ adapt"),
                           ("replay", data.get("replay_label", "replay control (seed 0)"))):
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
        for arm, value in data.get("matched", {}).items():
            ax.plot([position(MATCHED_BUDGET)], [value], ls="none", marker=MARKERS[MATCHED_ARMS[arm]], ms=8, mfc="white",
                    mec=COLORS[MATCHED_ARMS[arm]], mew=1.8, label=MATCHED_LABELS[arm], zorder=4)
        ax.axhline(data["ceiling"], color="#6b6b66", ls="--", lw=1)
        ax.text(position(10), data["ceiling"], f" {data.get('ceiling_label', 'FT full-pool ceiling')} {data['ceiling']:.3f}", va="bottom",
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ft-scored", type=Path, required=True, help="Scored FT campaign (evaluate_frcnn.py output)")
    parser.add_argument("--da-scored", type=Path, required=True, help="Scored RQ5 grid")
    parser.add_argument("--revision-scored", type=Path, default=None,
                        help="Scored revision runs: further replay seeds at N = 10 and 100, matched-pass arms at N = 1,000")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-denominator", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0, help="Seed for redrawing sampling seeds")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    # The 2026-11-21 tables and the revision tables take one generator per row (row_rng with seed + 2 to seed + 7):
    # nothing below may add a draw to rng or to default_rng(seed + 1).
    ft_entries, da_entries = load(args.ft_scored), load(args.da_scored)
    scored = [args.ft_scored, args.da_scored] + ([args.revision_scored] if args.revision_scored else [])
    resamples = shared_resamples([json.loads(path.read_text(encoding="utf-8")) for directory in scored
                                  for path in sorted(directory.glob("*/*/metrics.json"))])
    source_model = {e["domain"]: e for e in ft_entries if e["run"]["stage"] == "source"}
    # The revision entries never join da_entries: the 2026-11-03 tables read the original runs only.
    revision = split_revision(load(args.revision_scored)) if args.revision_scored else None
    if revision is not None and not any(revision.values()):
        raise ValueError(f"No adapt or replay run among the scored sets under {args.revision_scored}")

    deltas, equivalents, attribution, per_class, forgetting, summary, curves = [], [], [], [], [], {}, {}
    shares, shares_vs_source, vs_zero_shot = [], [], []
    matched_passes, matched_per_class, shares_matched, forgetting_revision = [], [], [], []
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
        # replay stays the original arm. replay_seeds adds the further sampling seeds of the revision runs and feeds the
        # 2026-11-21 tables; matched holds the matched-pass arms at N = 1,000, which are never pooled with adapt or replay.
        replay_seeds, matched = replay, {}
        if revision is not None:
            replay_seeds = merge_replay_seeds(replay, arm_runs(revision["replay_seeds"], "replay", domain, domain))
            matched = {arm: arm_runs(revision["matched"], stage, domain, domain).get(MATCHED_BUDGET, {})
                       for arm, stage in MATCHED_ARMS.items()}
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
        # Every seed common to the three arms: replay has seed 0 only, plus seeds 1 and 2 at N = 10 and 100 with the
        # revision runs.
        for n in ATTRIBUTION_BUDGETS:
            share = replay_share(adapt.get(n, {}), replay_seeds.get(n, {}), {} if n == 0 else ft_ladder.get(n, {}),
                                 row_rng(args.seed + 2, TARGETS.index(domain), n), zero=zero if n == 0 else None)
            if share is not None:
                shares.append({"domain": domain, "budget": n, "labeled_budget": n > 0, **share, "quotable": quotable(share)})
        for n in SOURCE_REFERENCE_BUDGETS:  # the same share with the source model as the reference, in its own stream
            share = replay_share(adapt.get(n, {}), replay_seeds.get(n, {}), {},
                                 row_rng(args.seed + 4, TARGETS.index(domain), n), zero=zero)
            if share is not None:
                shares_vs_source.append({"domain": domain, "budget": n, "reference": "source model", **share,
                                         "quotable": quotable(share)})
        vs_zero_shot += [{"domain": domain, **row} for row in versus_zero_shot(
            zero, {"FT": ft_ladder, "adapt": adapt, "replay": replay_seeds}, sorted(adapt),
            (args.seed + 3, TARGETS.index(domain)))]
        vs_zero_shot += [{"domain": domain, **row} for row in versus_zero_shot(
            zero, {arm: {MATCHED_BUDGET: runs} for arm, runs in matched.items()}, [MATCHED_BUDGET],
            (args.seed + 3, TARGETS.index(domain)))]
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
        if revision is not None:  # row_rng only: no draw from rng, so the band draws below keep their place
            shift = TARGETS.index(domain)
            arms = {**matched, "adapt_original": adapt.get(MATCHED_BUDGET, {}), "replay_original": replay.get(MATCHED_BUDGET, {}),
                    "FT": ft_ladder.get(MATCHED_BUDGET, {})}
            matched_passes += [{"domain": domain, **row} for row in matched_pass_rows(arms, (args.seed + 5, shift), reading=True)]
            for index, name in enumerate(RELIABLE_CLASSES):
                matched_per_class += [{"domain": domain, "class": name, **row}
                                      for row in matched_pass_rows(arms, (args.seed + 6, shift, index), key=name)]
            shares_matched += [{"domain": domain, **row} for row in matched_share_rows(
                matched["adapt_matched"], matched["replay_matched"], arms["FT"], zero, (args.seed + 7, shift))]
            source_replay = merge_replay_seeds(arm_runs(da_entries, "replay", domain, "source"),
                                               arm_runs(revision["replay_seeds"], "replay", domain, "source"))
            groups = [("replay", n, source_replay.get(n, {})) for n in REPLAY_SEED_BUDGETS]
            groups += [(arm, MATCHED_BUDGET, arm_runs(revision["matched"], stage, domain, "source").get(MATCHED_BUDGET, {}))
                       for arm, stage in MATCHED_ARMS.items()]
            forgetting_revision += [{"domain": domain, **row} for row in source_test_rows(groups, reference)]
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
        if revision is not None:  # the replay line becomes the seed mean of the merged arm; the matched arms are points
            several = ", ".join(str(n) for n, r in sorted(replay_seeds.items()) if len(r) > 1)
            curves[domain]["replay"] = [(n, mean(e["points"]["reliable"] for e in r.values()), None, None)
                                        for n, r in sorted(replay_seeds.items())]
            curves[domain]["replay_label"] = f"replay control (seed 0{'; seed mean at N = ' + several if several else ''})"
            curves[domain]["ceiling_label"] = "FT full pool, 3 epochs"  # mAP_c is the 12-epoch ceiling since 2026-11-21
            curves[domain]["matched"] = {arm: mean(e["points"]["reliable"] for e in runs.values())
                                         for arm, runs in matched.items() if runs}
    headline = {n: headline_check(shares, n) for n in HEADLINE_BUDGETS}
    tables = {"rq5_delta": deltas, "rq5_delta_per_class": per_class, "rq5_equivalent_budget": equivalents,
              "rq5_attribution": attribution, "rq5_forgetting": forgetting, "rq5_replay_share": shares,
              "rq5_replay_share_vs_source": shares_vs_source, "rq5_vs_zero_shot": vs_zero_shot,
              "rq5_matched_passes": matched_passes, "rq5_matched_passes_per_class": matched_per_class,
              "rq5_replay_share_matched": shares_matched, "rq5_forgetting_revision": forgetting_revision}
    stale = stale_tables(args.output_dir, tables)
    if stale:  # before anything is written: the directory keeps the earlier run whole
        raise ValueError(f"{args.output_dir} holds tables of an earlier run that this run does not write: {stale}. Use an "
                         "empty --output-dir, or remove them.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(args.output_dir / f"{name}.csv", rows)
    readings = readings_by_shift(matched_passes)
    revision_summary = {} if revision is None else {
        "matched_pass_rule": MATCHED_PASS_RULE, "matched_passes": matched_passes, "matched_pass_reading": readings,
        "replay_share_matched": shares_matched,
        "revision_scored_entries": {kind: len(entries) for kind, entries in revision.items()},
        "ceiling_reference": CEILING_REFERENCE}
    (args.output_dir / "summary.json").write_text(json.dumps(
        {"rules": "DECISIONS 2026-11-03", "delta_threshold": DELTA, "domains": summary,
         "attribution": attribution, "replay_share": shares, "replay_share_vs_source_model": shares_vs_source,
         "claim_rule": CLAIM_RULE,
         "headline_supported_all_shifts": {n: h["supported"] for n, h in headline.items()},
         "share_majority_all_shifts": {n: h["share_majority_all_shifts"] for n, h in headline.items()},
         "replay_seeds_available": {n: h["replay_seeds"] for n, h in headline.items()},
         "replay_seeds_required": HEADLINE_REPLAY_SEEDS,
         "direction_all_shifts": {n: h["direction"] for n, h in headline.items()},
         "largest_undefined_fraction": {n: h["largest_undefined_fraction"] for n, h in headline.items()},
         "test_resamples": resamples, **revision_summary}, indent=2, default=str), encoding="utf-8")
    if not args.no_plots:
        plot(curves, args.output_dir)
    for domain, s in summary.items():
        print(f"{domain}: crossover {s['crossover']['bracket']}; N_eq {s['N_eq']}")
        for n, d in s["delta"].items():
            print(f"  N={n}: delta {d['delta']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]{' *' if d['meaningful'] else ''}")
    print(f"Replay share = (replay - FT) / (adapt - FT), reliable mAP; undefined where |adapt - FT| < {DELTA}; N=0: FT is the "
          "source model; * = share and its lower 95% bound above 0.5")
    for row in shares:
        text = ("undefined" if row["share"] is None else f"{row['share']:.3f}" if row["share_ci_low"] is None
                else f"{row['share']:.3f} [{row['share_ci_low']:.3f}, {row['share_ci_high']:.3f}]")
        if row["undefined_replicates"]:
            text += f" ({row['undefined_replicates']}/{row['replicates']} replicates undefined)"
        print(f"  {row['domain']} N={row['budget']}: share {text}; {row['direction']} {row['gain']:+.4f}; "
              f"seeds {row['seed_list']}{' *' if row['majority_not_adaptation'] else ''}")
    print("Replay share against the source model = (replay - mAP(0)) / (adapt - mAP(0)); ~ = not quotable")
    for row in shares_vs_source:
        text = ("undefined" if row["share"] is None else f"{row['share']:.3f}" if row["share_ci_low"] is None
                else f"{row['share']:.3f} [{row['share_ci_low']:.3f}, {row['share_ci_high']:.3f}]")
        print(f"  {row['domain']} N={row['budget']}: share {text}; adapt gain {row['gain']:+.4f}, replay gain "
              f"{row['replay_part']:+.4f}; {row['undefined_replicates']}/{row['replicates']} undefined"
              f"{'' if row['quotable'] else ' ~'}")
    notes = {"advantage": "", "deficit": " (no gain at this budget: the rule then describes a deficit)",
             "mixed": " (a gain in some shifts only)", None: " (a shift is missing)"}
    for n, h in headline.items():
        undefined = h["largest_undefined_fraction"]
        print(f"  headline at N={n}: share > 0.5 with lower bound > 0.5 in all shifts: {h['share_majority_all_shifts']}; "
              f"replay seeds {h['replay_seeds']} of {HEADLINE_REPLAY_SEEDS} required; supported: {h['supported']}; "
              f"direction: {h['direction']}{notes[h['direction']]}; "
              f"largest undefined fraction: {'n/a' if undefined is None else format(undefined, '.3f')}")
    if revision is None:
        return
    print(f"Matched passes at N={MATCHED_BUDGET} ({MATCHED_UPDATES} updates for adapt and replay), reliable mAP, paired by "
          f"sampling seed. Rule on {PRE_REGISTERED_COMPARISON}: FT ahead stands when the difference <= -{DELTA} and the "
          "upper 95% bound < 0")
    for row in matched_passes:
        print(f"  {row['domain']} {row['comparison']}: {row['difference']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]; "
              f"seeds {row['seed_list']}{'; ' + row['reading'] if 'reading' in row else ''}")
    for domain, entry in readings.items():
        if entry["ft_ahead_stands"] is None:
            print(f"  {domain} {PRE_REGISTERED_COMPARISON}: {NOT_EVALUATED}")
    print("Matched-pass replay share at N=1000 (adapt_matched, replay_matched); ~ = not quotable (more than "
          f"{MAX_UNDEFINED_FRACTION:.0%} of the drawn replicates undefined); expected = the undefined fraction over every "
          "redraw of the seeds, which the drawn count scatters around with --seed")
    for row in shares_matched:
        text = ("undefined" if row["share"] is None else f"{row['share']:.3f}" if row["share_ci_low"] is None
                else f"{row['share']:.3f} [{row['share_ci_low']:.3f}, {row['share_ci_high']:.3f}]")
        print(f"  {row['domain']} against {row['reference']}: share {text}; {row['direction']} {row['gain']:+.4f}; "
              f"{row['undefined_replicates']}/{row['replicates']} undefined (expected "
              f"{row['expected_undefined_fraction']:.3f}); seeds {row['seed_list']}{'' if row['quotable'] else ' ~'}")
    print("Headline ('most of the gain over fine-tuning is not adaptation') with the merged replay arm: "
          + "; ".join(f"N={n} supported: {h['supported']} (replay seeds {h['replay_seeds']})" for n, h in headline.items()))


if __name__ == "__main__":
    main()
