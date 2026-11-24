"""Source retention analysis (direction 1.3): the Pareto flag, the point rows (one seed draw for both axes, test resample
t of the source test set paired with resample t of the source model, the recovery floor, the reference that reuses its
own draw, the undefined fraction of recovery_e12, non-finite replicates counted as undefined), the paired rows (common
seeds only, one draw of the seed pairs for both axes, resample t paired across arms, the flags), the check that every
seed of a point is scored on both test sets, and main() with and without --revision-scored (one redraw of each recovery
reference per shift, the locked FT schedule and its two full-pool seeds, one source model per test set).

Synthetic arrays and synthetic scored trees in a temporary directory only; nothing here reads runs/ or data/.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_retention import (both_tests, expected_undefined_fraction, main, paired_row, pareto,  # noqa: E402
                               point_row, reference_denominator)
from bdd_pipeline.data import CLASSES, EXPECTED_DOMAINS  # noqa: E402

TARGETS = ("compound", "weather", "lighting")


def entry(point: float, boot: np.ndarray | None = None, replicates: int = 100, test: str = "lighting") -> dict:
    return {"run": {"seed": 0}, "domain": test, "points": {"reliable": point},
            "boots": {"reliable": np.full(replicates, point) if boot is None else boot}}


def runs(values: dict[int, float], replicates: int = 100) -> dict[int, dict]:
    """Seed -> entry with a flat bootstrap vector at the point value."""
    return {seed: entry(value, replicates=replicates) for seed, value in values.items()}


def denominators(ceilings: dict[str, dict[int, dict]], zero: dict) -> dict:
    """reference_denominator of each reference, its seeds redrawn by a generator other than the point's."""
    replicates = len(zero["boots"]["reliable"])
    return {name: reference_denominator(ceiling, zero, np.random.default_rng(1).integers(
        0, len(ceiling), (replicates, len(ceiling)))) for name, ceiling in ceilings.items()}


class FirstDrawZero:
    """Stub generator: the first draw takes seed index 0 for every replicate, every later draw the last seed index."""

    def __init__(self) -> None:
        self.calls = 0

    def integers(self, low: int, high: int, size: tuple[int, int]) -> np.ndarray:
        self.calls += 1
        return np.full(size, 0 if self.calls == 1 else high - 1)


class ParetoTest(unittest.TestCase):
    def test_hand_built_front(self):
        points = [(0.0, 1.0),     # source model: highest retention
                  (0.05, 0.90),   # dominated by the next-but-two (same retention, more gain)
                  (0.03, 0.95),   # nothing has both more gain and at least its retention
                  (0.02, 0.85),   # dominated on both axes
                  (0.06, 0.90),   # front
                  (0.06, 0.90)]   # an exact tie dominates nothing: both stay on the front
        self.assertEqual(pareto(points), [True, False, True, False, True, True])
        self.assertEqual(pareto([(0.0, 1.0)]), [True])
        self.assertEqual(pareto([(0.01, 1.0), (0.0, 1.0)]), [True, False])  # equal retention, strictly more gain


class PointRowTest(unittest.TestCase):
    zero, reference = entry(0.25, replicates=400), entry(0.35, replicates=400, test="source")

    def test_values_and_hierarchical_intervals(self):
        point = {"target": runs({0: 0.30, 1: 0.32}, 400), "source": runs({0: 0.33, 1: 0.34}, 400)}
        ceilings = {"e3": runs({0: 0.35, 1: 0.37}, 400)}  # denominator 0.11
        row = point_row(point, self.zero, self.reference, denominators(ceilings, self.zero), 0.01, np.random.default_rng(0))
        self.assertEqual((row["seeds"], row["seed_list"], row["replicates"]), (2, "0 1", 400))
        self.assertEqual((row["gain_undefined"], row["retention_undefined"], row["source_change_undefined"]), (0, 0, 0))
        self.assertAlmostEqual(row["target_mAP"], 0.31)
        self.assertAlmostEqual(row["source_mAP"], 0.335)
        self.assertAlmostEqual(row["gain"], 0.06)
        self.assertAlmostEqual(row["recovery_e3"], 0.06 / 0.11)
        self.assertAlmostEqual(row["retention"], 0.335 / 0.35)
        self.assertAlmostEqual(row["source_change"], -0.015)
        # Each extreme (both draws on one seed) has probability 1/4, so the 2.5% and 97.5% points are the seed values.
        self.assertAlmostEqual(row["gain_ci_low"], 0.05)
        self.assertAlmostEqual(row["gain_ci_high"], 0.07)
        self.assertAlmostEqual(row["retention_ci_low"], 0.33 / 0.35)
        self.assertAlmostEqual(row["retention_ci_high"], 0.34 / 0.35)
        self.assertAlmostEqual(row["source_change_ci_low"], -0.02)
        # The denominator redraws the reference seeds on its own: lowest gain over highest denominator and vice versa.
        self.assertAlmostEqual(row["recovery_e3_ci_low"], 0.05 / 0.12)
        self.assertAlmostEqual(row["recovery_e3_ci_high"], 0.07 / 0.10)
        self.assertEqual(row["recovery_e3_undefined"], 0)
        self.assertNotIn("recovery_e12", row)

    def test_one_seed_draw_serves_both_axes(self):
        # Seed 0 is low on the target and high on the source, seed 1 the opposite. The stub puts the first draw on seed 0
        # and every later draw on the last seed: a second draw for the source axis would read 0.28 instead of 0.34.
        point = {"target": runs({0: 0.30, 1: 0.40}), "source": runs({0: 0.34, 1: 0.28})}
        zero = entry(0.25)
        last = {"e3": reference_denominator(runs({0: 0.35, 1: 0.45}), zero, np.ones((100, 2), dtype=int))}  # 0.20
        row = point_row(point, zero, entry(0.35, test="source"), last, 0.01, FirstDrawZero())
        self.assertAlmostEqual(row["gain_ci_low"], 0.05)
        self.assertAlmostEqual(row["gain_ci_high"], 0.05)
        self.assertAlmostEqual(row["retention_ci_low"], 0.34 / 0.35)
        self.assertAlmostEqual(row["retention_ci_high"], 0.34 / 0.35)
        self.assertAlmostEqual(row["source_change_ci_high"], -0.01)
        # The denominator comes from the reference's own draw (its last seed, 0.45): recovery 0.05 / 0.20.
        self.assertAlmostEqual(row["recovery_e3_ci_low"], 0.25)
        self.assertAlmostEqual(row["recovery_e3_ci_high"], 0.25)

    def test_source_resamples_pair_with_the_source_model(self):
        # Test-resample noise of the shift's (f) and of the source test set (e), shared by every run on that test set.
        f, e = np.random.default_rng(1).normal(0, 0.01, (2, 400))
        zero, reference = entry(0.25, 0.25 + f), entry(0.35, 0.35 + e, test="source")
        target = {0: entry(0.30, 0.30 + f), 1: entry(0.32, 0.32 + f)}
        ceilings = denominators({"e3": {0: entry(0.35, 0.35 + f), 1: entry(0.37, 0.37 + f)}}, zero)
        # Each source run is 98% / 96% of the source model in every resample: retention is exactly that, per drawn seed,
        # only when resample t of the run meets resample t of the source model.
        source = {0: entry(0.98 * 0.35, 0.98 * (0.35 + e)), 1: entry(0.96 * 0.35, 0.96 * (0.35 + e))}
        row = point_row({"target": target, "source": source}, zero, reference, ceilings, 0.01, np.random.default_rng(0))
        self.assertAlmostEqual(row["gain_ci_low"], 0.05)
        self.assertAlmostEqual(row["gain_ci_high"], 0.07)
        self.assertAlmostEqual(row["recovery_e3_ci_low"], 0.05 / 0.12)
        self.assertAlmostEqual(row["recovery_e3_ci_high"], 0.07 / 0.10)
        self.assertAlmostEqual(row["retention"], 0.97)
        self.assertAlmostEqual(row["retention_ci_low"], 0.96)
        self.assertAlmostEqual(row["retention_ci_high"], 0.98)
        # Each source run a constant below the source model: the change is exactly that constant, per drawn seed.
        source = {0: entry(0.33, 0.33 + e), 1: entry(0.34, 0.34 + e)}
        row = point_row({"target": target, "source": source}, zero, reference, ceilings, 0.01, np.random.default_rng(0))
        self.assertAlmostEqual(row["source_change_ci_low"], -0.02)
        self.assertAlmostEqual(row["source_change_ci_high"], -0.01)

    def test_the_reference_point_reuses_its_own_draw(self):
        full = runs({0: 0.30, 1: 0.36}, 400)
        point = {"target": full, "source": runs({0: 0.31, 1: 0.32}, 400)}
        draw = np.random.default_rng(1).integers(0, 2, (400, 2))
        shared = {"e3": reference_denominator(full, self.zero, draw)}
        own = point_row(point, self.zero, self.reference, shared, 0.01, FirstDrawZero(), draw)
        self.assertEqual((own["recovery_e3"], own["recovery_e3_ci_low"], own["recovery_e3_ci_high"]), (1.0, 1.0, 1.0))
        # The reference's draw serves both axes of its own point.
        self.assertEqual([own["gain_ci_low"], own["gain_ci_high"]], list(np.percentile(shared["e3"][1], [2.5, 97.5])))
        retention = np.array([0.31, 0.32])[draw].mean(axis=1) / 0.35
        self.assertEqual([own["retention_ci_low"], own["retention_ci_high"]], list(np.percentile(retention, [2.5, 97.5])))
        apart = point_row(point, self.zero, self.reference, shared, 0.01, np.random.default_rng(0))
        self.assertLess(apart["recovery_e3_ci_low"], 1.0)  # two independent draws of the same runs would scatter
        self.assertGreater(apart["recovery_e3_ci_high"], 1.0)

    def test_recovery_is_none_below_the_floor(self):
        point = {"target": runs({0: 0.30}), "source": runs({0: 0.33})}
        zero = entry(0.25)
        row = point_row(point, zero, entry(0.35, test="source"), denominators({"e3": runs({0: 0.255})}, zero), 0.01,
                        np.random.default_rng(0))
        self.assertIsNone(row["recovery_e3"])
        self.assertEqual((row["recovery_e3_ci_low"], row["recovery_e3_ci_high"], row["recovery_e3_undefined"]),
                         (None, None, 100))
        self.assertAlmostEqual(row["gain"], 0.05)  # the other axes do not need the denominator
        self.assertAlmostEqual(row["retention"], 0.33 / 0.35)
        # A denominator equal to the floor is undefined, as in bdd_pipeline.recovery (exactly representable values);
        # just above it, it is defined.
        at = point_row({"target": runs({0: 0.5}), "source": runs({0: 0.5})}, zero, entry(0.5, test="source"),
                       denominators({"e3": runs({0: 0.75})}, zero), 0.5, np.random.default_rng(0))
        self.assertEqual((at["recovery_e3"], at["recovery_e3_undefined"]), (None, 100))
        above = point_row({"target": runs({0: 0.5}), "source": runs({0: 0.5})}, zero, entry(0.5, test="source"),
                          denominators({"e3": runs({0: 0.75})}, zero), 0.25, np.random.default_rng(0))
        self.assertEqual((above["recovery_e3"], above["recovery_e3_undefined"]), (0.5, 0))

    def test_non_finite_replicates_are_undefined_and_counted(self):
        target, source_model = np.full(100, 0.30), np.full(100, 0.35)
        target[3] = np.nan
        source_model[[5, 6]] = np.nan
        zero = entry(0.25)
        point = {"target": {0: entry(0.30, target)}, "source": runs({0: 0.33})}
        row = point_row(point, zero, entry(0.35, source_model, test="source"), denominators({"e3": runs({0: 0.35})}, zero),
                        0.01, np.random.default_rng(0))
        self.assertEqual((row["gain_undefined"], row["recovery_e3_undefined"]), (1, 1))
        self.assertEqual((row["retention_undefined"], row["source_change_undefined"]), (2, 2))
        self.assertAlmostEqual(row["gain_ci_low"], 0.05)  # over the defined replicates
        self.assertAlmostEqual(row["recovery_e3_ci_high"], 0.5)
        self.assertAlmostEqual(row["retention_ci_high"], 0.33 / 0.35)

    def test_undefined_fraction_of_the_12_epoch_recovery(self):
        point = {"target": runs({0: 0.30}), "source": runs({0: 0.33})}
        zero = entry(0.25)
        for below, determined in ((5, True), (6, False)):
            boot = np.full(100, 0.30)
            boot[:below] = 0.255  # these replicates put the 12-epoch denominator under the floor
            ceilings = denominators({"e3": runs({0: 0.35}), "e12": {0: entry(0.30, boot)}}, zero)
            row = point_row(point, zero, entry(0.35, test="source"), ceilings, 0.01, np.random.default_rng(0))
            self.assertAlmostEqual(row["recovery_e12"], 1.0)
            self.assertEqual(row["recovery_e12_undefined"], below)
            self.assertAlmostEqual(row["recovery_e12_undefined_fraction"], below / 100)
            self.assertIs(row["recovery_e12_determined"], determined)
            self.assertAlmostEqual(row["recovery_e12_ci_low"], 1.0)  # the interval covers the defined replicates only
            self.assertEqual(row["recovery_e3_undefined"], 0)


class ExpectedUndefinedFractionTest(unittest.TestCase):
    def test_every_ordered_redraw_of_the_reference_seeds_counts_once(self):
        # Gains over mAP(0) = 0.25: seed 0 alone 0.005 (undefined), seed 1 alone 0.05, the mixed redraws 0.0275. Of the
        # four ordered redraws of two seeds, (0, 0) is undefined in every test resample: 1/4.
        zero = entry(0.25)
        self.assertEqual(expected_undefined_fraction(runs({0: 0.255, 1: 0.30}), zero, 0.01), 0.25)
        self.assertEqual(expected_undefined_fraction(runs({0: 0.30, 1: 0.30}), zero, 0.01), 0.0)
        # A denominator equal to the floor is undefined (exactly representable values).
        self.assertEqual(expected_undefined_fraction(runs({0: 0.75}), entry(0.25), 0.5), 1.0)

    def test_fraction_is_over_test_resamples_too(self):
        boot = np.array([0.255, 0.30, 0.30, 0.30])  # seed 0 is close to mAP(0) in the first resample only
        zero = entry(0.25, replicates=4)
        reference = {0: entry(0.28, boot), 1: entry(0.30, replicates=4)}
        # Resample 0: redraws (0, 0) -> 0.005 undefined, (0, 1) and (1, 0) -> 0.0275, (1, 1) -> 0.05. Others defined.
        self.assertEqual(expected_undefined_fraction(reference, zero, 0.01), 1 / 16)


class PairedRowTest(unittest.TestCase):
    reference = entry(0.35, test="source")

    def side(self, target: dict[int, float], source: dict[int, float]) -> dict:
        return {"target": runs(target), "source": runs(source)}

    def test_only_common_seeds_are_paired(self):
        left = self.side({0: 0.30, 1: 0.50, 2: 0.32}, {0: 0.34, 1: 0.20, 2: 0.33})
        right = self.side({0: 0.28, 2: 0.30, 3: 0.90}, {0: 0.33, 2: 0.32, 3: 0.10})
        row = paired_row(left, right, self.reference, 0.04, 0.01, np.random.default_rng(0))
        self.assertEqual((row["seeds"], row["seed_list"]), (2, "0 2"))
        self.assertAlmostEqual(row["delta_target"], 0.02)
        self.assertAlmostEqual(row["delta_target_ci_low"], 0.02)  # seeds 1 and 3 never enter a replicate
        self.assertAlmostEqual(row["delta_target_ci_high"], 0.02)
        self.assertAlmostEqual(row["delta_target_recovery"], 0.5)
        self.assertAlmostEqual(row["delta_retention"], 0.01 / 0.35)
        self.assertAlmostEqual(row["delta_retention_ci_high"], 0.01 / 0.35)
        self.assertEqual((row["dominates"], row["dominated"], row["trade_off"]), (True, False, False))
        self.assertIsNone(paired_row(self.side({0: 0.3}, {0: 0.3}), self.side({1: 0.3}, {1: 0.3}), self.reference, 0.04,
                                     0.01, np.random.default_rng(0)))

    def test_one_draw_of_the_seed_pairs_serves_both_axes(self):
        # Pair 0: +0.02 on the target, +0.01 on the source; pair 1: +0.12 and -0.05. The stub's first draw is pair 0.
        left = self.side({0: 0.30, 1: 0.40}, {0: 0.36, 1: 0.30})
        right = self.side({0: 0.28, 1: 0.28}, {0: 0.35, 1: 0.35})
        row = paired_row(left, right, self.reference, 0.04, 0.01, FirstDrawZero())
        self.assertAlmostEqual(row["delta_target_ci_low"], 0.02)
        self.assertAlmostEqual(row["delta_target_ci_high"], 0.02)
        self.assertAlmostEqual(row["delta_retention_ci_low"], 0.01 / 0.35)
        self.assertAlmostEqual(row["delta_retention_ci_high"], 0.01 / 0.35)
        self.assertTrue(row["dominates"])

    def test_flags(self):
        cases = [("dominated", {0: 0.28}, {0: 0.34}), ("trade_off", {0: 0.32}, {0: 0.30}), ("trade_off", {0: 0.28}, {0: 0.36})]
        right = self.side({0: 0.30}, {0: 0.35})
        for name, target, source in cases:
            row = paired_row(self.side(target, source), right, self.reference, 0.04, 0.01, np.random.default_rng(0))
            flags = {key for key in ("dominates", "dominated", "trade_off") if row[key]}
            self.assertEqual(flags, {name}, (target, source))
        # Pair differences +0.01 and -0.01 on the target: the interval covers 0, so no flag.
        mixed = paired_row(self.side({0: 0.31, 1: 0.29}, {0: 0.36, 1: 0.36}), self.side({0: 0.30, 1: 0.30}, {0: 0.35, 1: 0.35}),
                           self.reference, 0.04, 0.01, np.random.default_rng(0))
        self.assertLess(mixed["delta_target_ci_low"], 0)
        self.assertGreater(mixed["delta_target_ci_high"], 0)
        self.assertEqual((mixed["dominates"], mixed["dominated"], mixed["trade_off"]), (False, False, False))

    def test_recovery_units_are_none_below_the_floor(self):
        row = paired_row(self.side({0: 0.32}, {0: 0.35}), self.side({0: 0.30}, {0: 0.35}), self.reference, 0.005, 0.01,
                         np.random.default_rng(0))
        self.assertAlmostEqual(row["delta_target"], 0.02)
        self.assertIsNone(row["delta_target_recovery"])

    def test_resample_t_pairs_across_arms_and_with_the_source_model(self):
        f, e = np.random.default_rng(1).normal(0, 0.01, (2, 400))  # test-resample noise of the shift's / source test set
        reference = entry(0.35, 0.35 + e, test="source")
        # Pair 0: +0.02 on the target and 3% of the source model on the source test set in every resample; pair 1: +0.04
        # and 1%. Each replicate difference is exactly one of those per drawn pair only when resample t meets resample t.
        left = {"target": {0: entry(0.30, 0.30 + f), 1: entry(0.32, 0.32 + f)},
                "source": {0: entry(1.03 * 0.35, 1.03 * (0.35 + e)), 1: entry(1.01 * 0.35, 1.01 * (0.35 + e))}}
        right = {"target": {0: entry(0.28, 0.28 + f), 1: entry(0.28, 0.28 + f)},
                 "source": {0: entry(0.35, 0.35 + e), 1: entry(0.35, 0.35 + e)}}
        row = paired_row(left, right, reference, 0.04, 0.01, np.random.default_rng(0))
        self.assertAlmostEqual(row["delta_target"], 0.03)
        self.assertAlmostEqual(row["delta_target_ci_low"], 0.02)
        self.assertAlmostEqual(row["delta_target_ci_high"], 0.04)
        self.assertAlmostEqual(row["delta_retention"], 0.02)
        self.assertAlmostEqual(row["delta_retention_ci_low"], 0.01)
        self.assertAlmostEqual(row["delta_retention_ci_high"], 0.03)

    def test_undefined_replicates_are_counted_and_a_flag_needs_both_intervals(self):
        right = self.side({0: 0.30}, {0: 0.35})
        boot = np.full(100, 0.32)
        boot[3] = np.nan
        row = paired_row({"target": {0: entry(0.32, boot)}, "source": runs({0: 0.36})}, right, self.reference, 0.04, 0.01,
                         np.random.default_rng(0))
        self.assertEqual((row["delta_target_undefined"], row["delta_retention_undefined"]), (1, 0))
        self.assertAlmostEqual(row["delta_target_ci_low"], 0.02)  # over the defined replicates
        self.assertTrue(row["dominates"])
        for side in ("target", "source"):
            left = self.side({0: 0.32}, {0: 0.36})
            left[side] = {0: entry(0.36, np.full(100, np.nan))}
            row = paired_row(left, right, self.reference, 0.04, 0.01, np.random.default_rng(0))
            name = "delta_target" if side == "target" else "delta_retention"
            self.assertEqual((row[f"{name}_ci_low"], row[f"{name}_ci_high"], row[f"{name}_undefined"]), (None, None, 100))
            self.assertEqual((row["dominates"], row["dominated"], row["trade_off"]), (False, False, False), side)


class BothTestsTest(unittest.TestCase):
    def test_every_seed_needs_both_test_sets_and_a_reliable_mAP(self):
        with self.assertRaisesRegex(ValueError, r"lighting adapt N = 100: seed\(s\) \[1\] have no entry on the source test"):
            both_tests("lighting", "adapt", 100, runs({0: 0.3, 1: 0.3}), runs({0: 0.3}))
        with self.assertRaisesRegex(ValueError, r"weather FT N = 10: seed\(s\) \[2\] have no entry on the weather test"):
            both_tests("weather", "FT", 10, runs({0: 0.3}), runs({0: 0.3, 2: 0.3}))
        with self.assertRaisesRegex(ValueError, "has no reliable mAP"):
            both_tests("lighting", "replay", 10, runs({0: 0.3}), {0: entry(None, np.full(100, 0.3))})
        points = both_tests("lighting", "FT", 10, runs({0: 0.3}), runs({0: 0.33}))
        self.assertEqual(sorted(points), ["source", "target"])
        # A non-finite test resample does not stop the script: the replicates that use it are undefined and counted.
        boot = np.full(100, 0.3)
        boot[3] = np.nan
        self.assertEqual(sorted(both_tests("lighting", "replay", 10, {0: entry(0.3, boot)}, runs({0: 0.3}))),
                         ["source", "target"])


class MainTest(unittest.TestCase):
    """main() on three synthetic scored trees (campaign, RQ5 grid, revision).

    Every class of a run scores its level, and every bootstrap vector is that level plus test-resample noise shared by
    all runs on one test domain in all three trees. Set folders are numbered and carry no run name; the revision tree
    repeats (stage, domain, budget, seed) of campaign and RQ5 runs, so only the run fields tell them apart.
    """
    zero = {"compound": 0.22, "weather": 0.32, "lighting": 0.24}
    source_level = 0.35
    ladder = {"compound": (10, 100, 1000), "weather": (10, 100, 1000), "lighting": (10, 100, 1000, 5000)}
    ft_gain = {10: -0.05, 100: 0.0, 1000: 0.03, 5000: 0.035}
    ft_source = {10: 0.345, 100: 0.34, 1000: 0.33, 5000: 0.32, "full": 0.31}
    adapt_delta = {10: 0.06, 100: 0.016, 1000: -0.012, 5000: -0.02}
    replay_part = {10: (0.048, 0.051, 0.054), 100: (0.012, 0.014, 0.016), 1000: (-0.009,)}
    matched_delta = {"adapt": -0.008, "replay": -0.006}
    matched_source = {"adapt": 0.346, "replay": 0.344}

    def ft(self, domain: str, budget: int, seed: int) -> float:
        return self.zero[domain] + self.ft_gain[budget] + 0.002 * seed

    def write(self, tree: Path, noise: dict, stage: str, run_domain: str, budget, seed: int, test: str, level: float,
              epochs: int | None = None, updates: int | None = 2000) -> None:
        folder = tree / f"{len(list(tree.glob('*'))):03d}" / test
        folder.mkdir(parents=True)
        metrics = {"run": {"stage": stage, "domain": run_domain, "budget": str(budget), "seed": seed, "epochs": epochs,
                           "total_updates": updates},
                   "domain": test, "images": EXPECTED_DOMAINS[test]["final_test"], "AP50_95": level,
                   "reliable_AP50_95": level, "per_class": [{"class": name, "AP50_95": level} for name in CLASSES],
                   "bootstrap": {"class_order": list(CLASSES), "seed": 20261030, "resamples": len(noise[test])}}
        (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        np.save(folder / "bootstrap_class_ap.npy", level + noise[test])

    def both(self, tree: Path, noise: dict, stage: str, run_domain: str, budget, seed: int, test: str, level: float,
             source: float, **fields) -> None:
        """One run scored on the shift's test set and on the source test set."""
        self.write(tree, noise, stage, run_domain, budget, seed, test, level, **fields)
        self.write(tree, noise, stage, run_domain, budget, seed, "source", source, **fields)

    def build(self, root: Path, replicates: int = 80) -> dict[str, Path]:
        rng = np.random.default_rng(7)
        noise = {domain: rng.normal(0, 0.004, (replicates, len(CLASSES))) for domain in ("source", *TARGETS)}
        trees = {name: root / name for name in ("campaign", "rq5", "revision")}
        for tree in trees.values():
            tree.mkdir(parents=True)
        campaign, rq5, revision = trees["campaign"], trees["rq5"], trees["revision"]
        for test in ("source", *TARGETS):
            self.write(campaign, noise, "source", "source", "full", 0, test, self.zero.get(test, self.source_level),
                       epochs=12, updates=29412)
        self.write(rq5, noise, "replay", "source", 0, 0, "source", 0.3495)
        for domain in TARGETS:
            zero = self.zero[domain]
            for seed, shift in enumerate((-0.005, 0.005)):
                self.both(campaign, noise, "fine-tune", domain, "full", seed, domain, zero + 0.04 + shift,
                          self.ft_source["full"], epochs=3)
                self.both(revision, noise, "fine-tune", domain, "full", seed, domain, zero + 0.06, 0.30, epochs=12)
                self.both(rq5, noise, "adapt", domain, 0, seed, domain, zero + 0.01, 0.349)
            if domain != "compound":  # matched-pool runs at 1,738: a ceiling of RQ4, never a point here
                self.both(campaign, noise, "fine-tune", domain, 1738, 0, domain, zero + 0.038, 0.32, epochs=3)
                self.both(revision, noise, "fine-tune", domain, 1738, 1, domain, zero + 0.038, 0.32, epochs=3)
                for seed in (0, 1):
                    self.both(revision, noise, "fine-tune", domain, 1738, seed, domain, zero + 0.05, 0.30, epochs=12)
            self.write(campaign, noise, "control", domain, 100, 0, domain, zero - 0.05, epochs=3)
            self.write(rq5, noise, "replay", "source", 0, 0, domain, zero + 0.004)
            for budget in self.ladder[domain]:
                for seed in (0, 1, 2):
                    level = self.ft(domain, budget, seed)
                    self.both(campaign, noise, "fine-tune", domain, budget, seed, domain, level, self.ft_source[budget],
                              epochs=3)
                    self.both(rq5, noise, "adapt", domain, budget, seed, domain, level + self.adapt_delta[budget], 0.348)
                for seed, part in enumerate(self.replay_part.get(budget, ())):  # seed 0: RQ5 grid; 1 and 2: revision
                    self.both(rq5 if seed == 0 else revision, noise, "replay", domain, budget, seed, domain,
                              self.ft(domain, budget, seed) + part, 0.347 - 0.001 * seed)
            for stage in ("adapt", "replay"):
                for seed in (0, 1, 2):
                    self.both(revision, noise, stage, domain, 1000, seed, domain,
                              self.ft(domain, 1000, seed) + self.matched_delta[stage], self.matched_source[stage],
                              updates=3000)
        return trees

    def run_main(self, trees: dict[str, Path], output: Path, *options: str, revision: bool = False,
                 plots: bool = False) -> str:
        arguments = ["--ft-scored", str(trees["campaign"]), "--da-scored", str(trees["rq5"]), "--output-dir", str(output),
                     *options, *([] if plots else ["--no-plots"])]
        if revision:
            arguments += ["--revision-scored", str(trees["revision"])]
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            main(arguments)
        return printed.getvalue()

    def table(self, path: Path) -> list[dict]:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def outputs(self, directory: Path) -> dict:
        return {"files": sorted(path.name for path in directory.iterdir()),
                "points": {(row["domain"], row["method"], int(row["budget"])): row
                           for row in self.table(directory / "retention_points.csv")},
                "paired": {(row["domain"], row["comparison"], int(row["budget"])): row
                           for row in self.table(directory / "retention_paired.csv")},
                "summary": json.loads((directory / "retention_summary.json").read_text(encoding="utf-8"))}

    def test_points_pairs_and_the_revision_kinds(self):
        with tempfile.TemporaryDirectory() as folder:
            trees = self.build(Path(folder))
            self.run_main(trees, Path(folder) / "before")
            printed = self.run_main(trees, Path(folder) / "after", revision=True)
            before, after = self.outputs(Path(folder) / "before"), self.outputs(Path(folder) / "after")
        files = ["retention_paired.csv", "retention_points.csv", "retention_summary.json"]
        self.assertEqual((before["files"], after["files"]), (files, files))
        pools = {domain: EXPECTED_DOMAINS[domain]["train"] for domain in TARGETS}
        for domain in TARGETS:
            ladder, pool = list(self.ladder[domain]), pools[domain]
            expected = {"source model": [0], "FT": ladder + [pool], "adapt": [0] + ladder, "replay": [0, 10, 100, 1000]}
            methods = {method: sorted(n for d, m, n in before["points"] if d == domain and m == method)
                       for method in {m for d, m, _ in before["points"] if d == domain}}
            self.assertEqual(methods, expected, domain)  # no 12-epoch, matched-pass or matched-pool point without the option
            revised = {method: sorted(n for d, m, n in after["points"] if d == domain and m == method)
                       for method in {m for d, m, _ in after["points"] if d == domain}}
            self.assertEqual(revised, {**expected, "FT 12 epochs": [pool], "adapt_matched": [1000], "replay_matched": [1000]})
        lighting = before["points"]
        source = self.source_level
        ft100 = lighting[("lighting", "FT", 100)]
        self.assertEqual(ft100["seed_list"], "0 1 2")
        self.assertAlmostEqual(float(ft100["gain"]), 0.002, places=9)
        self.assertAlmostEqual(float(ft100["recovery_e3"]), 0.05, places=9)
        self.assertAlmostEqual(float(ft100["retention"]), 0.34 / source, places=9)
        self.assertAlmostEqual(float(ft100["source_change"]), -0.01, places=9)
        full = lighting[("lighting", "FT", pools["lighting"])]
        self.assertEqual(full["seed_list"], "0 1")
        self.assertEqual((full["recovery_e3"], full["recovery_e3_ci_low"], full["recovery_e3_ci_high"]), ("1.0",) * 3)
        model = lighting[("lighting", "source model", 0)]
        self.assertEqual((float(model["gain"]), float(model["retention"]), model["pareto"]), (0.0, 1.0, "True"))
        self.assertEqual(lighting[("lighting", "FT", 10)]["pareto"], "False")  # below the source model on both axes
        self.assertAlmostEqual(float(lighting[("lighting", "adapt", 10)]["gain"]), 0.012, places=9)
        self.assertAlmostEqual(float(lighting[("lighting", "replay", 0)]["retention"]), 0.3495 / source, places=9)
        self.assertNotIn("recovery_e12", ft100)
        # The revision adds three replay seeds at N = 10 and 100, not at 1,000; every other number of the rows it does not
        # change stays, interval included (each row has its own generator and draws the E = 12 reference last).
        for budget, seeds, gain in ((10, "0 1 2", -0.05 + 0.002 + 0.051), (100, "0 1 2", 0.002 + 0.014), (1000, "0", 0.021)):
            row = after["points"][("lighting", "replay", budget)]
            self.assertEqual(row["seed_list"], seeds)
            self.assertAlmostEqual(float(row["gain"]), gain, places=9)
        self.assertEqual(before["points"][("lighting", "replay", 10)]["seed_list"], "0")
        for key, row in before["points"].items():
            if key[1] != "replay" or key[2] not in (10, 100):
                changed = {name for name, value in row.items() if after["points"][key][name] != value}
                self.assertEqual(changed, {"pareto"} & changed, key)
        e12 = after["points"][("lighting", "FT 12 epochs", pools["lighting"])]
        self.assertAlmostEqual(float(e12["recovery_e3"]), 1.5, places=9)
        self.assertEqual((e12["recovery_e12"], e12["recovery_e12_ci_low"], e12["recovery_e12_ci_high"]), ("1.0",) * 3)
        self.assertEqual(e12["recovery_e12_determined"], "True")
        self.assertAlmostEqual(float(after["points"][("lighting", "FT", pools["lighting"])]["recovery_e12"]), 0.04 / 0.06,
                               places=9)
        matched = after["points"][("lighting", "adapt_matched", 1000)]
        self.assertEqual(matched["seed_list"], "0 1 2")
        self.assertAlmostEqual(float(matched["retention"]), 0.346 / source, places=9)
        # Paired rows: adapt - FT and replay - FT at every N >= 10 both arms have, the N = 0 rows against the source model.
        expected_pairs = {("lighting", "adapt - FT", n) for n in (10, 100, 1000, 5000)}
        expected_pairs |= {("lighting", "replay - FT", n) for n in (10, 100, 1000)}
        expected_pairs |= {("lighting", "adapt - source model", 0), ("lighting", "replay - source model", 0)}
        self.assertEqual({key for key in before["paired"] if key[0] == "lighting"}, expected_pairs)
        self.assertEqual({key for key in after["paired"] if key[0] == "lighting"},
                         expected_pairs | {("lighting", "adapt_matched - FT", 1000), ("lighting", "replay_matched - FT", 1000)})
        small = before["paired"][("lighting", "adapt - FT", 10)]
        self.assertAlmostEqual(float(small["delta_target"]), 0.06, places=9)
        self.assertAlmostEqual(float(small["delta_target_recovery"]), 1.5, places=9)
        self.assertAlmostEqual(float(small["delta_retention"]), 0.003 / source, places=9)
        self.assertEqual((small["seeds"], small["dominates"], small["trade_off"]), ("3", "True", "False"))
        large = before["paired"][("lighting", "adapt - FT", 1000)]
        self.assertAlmostEqual(float(large["delta_target"]), -0.012, places=9)
        self.assertEqual((large["dominates"], large["trade_off"]), ("False", "True"))
        zero_shot = before["paired"][("lighting", "adapt - source model", 0)]
        self.assertEqual((zero_shot["seed_list"], zero_shot["trade_off"]), ("0 1", "True"))
        self.assertAlmostEqual(float(zero_shot["delta_retention"]), -0.001 / source, places=9)
        self.assertEqual(before["paired"][("lighting", "replay - FT", 10)]["seed_list"], "0")
        self.assertEqual(after["paired"][("lighting", "replay - FT", 10)]["seed_list"], "0 1 2")
        self.assertEqual(after["paired"][("lighting", "adapt_matched - FT", 1000)]["seed_list"], "0 1 2")
        self.assertAlmostEqual(float(after["paired"][("lighting", "adapt_matched - FT", 1000)]["delta_target"]), -0.008,
                               places=9)
        self.assertEqual(after["paired"][("lighting", "adapt - FT", 10)], small)
        # Summary: references, the front and the caveat.
        references = after["summary"]["references"]["lighting"]
        self.assertAlmostEqual(references["denominator_e3"], 0.04, places=9)
        self.assertAlmostEqual(references["denominator_e12"], 0.06, places=9)
        self.assertEqual((references["seeds_e3"], references["seeds_e12"]), ([0, 1], [0, 1]))
        self.assertEqual((references["denominator_e3_undefined"], references["denominator_e12_undefined"]), (0, 0))
        self.assertEqual((references["denominator_e3_expected_undefined_fraction"],
                          references["denominator_e12_expected_undefined_fraction"]), (0.0, 0.0))
        self.assertIs(references["recovery_e12_determined"], True)
        self.assertNotIn("denominator_e12", before["summary"]["references"]["lighting"])
        self.assertEqual((before["summary"]["revision"], after["summary"]["revision"]), (False, True))
        self.assertIn({"method": "source model", "budget": 0, "gain": 0.0, "retention": 1.0},
                      after["summary"]["pareto_front"]["lighting"])
        self.assertIn("1 source + 1 target labeled image per update", after["summary"]["caveat"])
        self.assertTrue(printed.rstrip().splitlines()[-1].startswith("Caveat: adapt and replay keep one labeled source"))

    def test_figures_and_the_recovery_floor(self):
        with tempfile.TemporaryDirectory() as folder:
            trees = self.build(Path(folder), replicates=40)
            self.run_main(trees, Path(folder) / "plots", revision=True, plots=True)
            drawn = sorted(path.name for path in (Path(folder) / "plots").glob("*.png"))
            # A floor above every E = 3 denominator (0.04) leaves recovery_e3 undefined and draws no figure; the 12-epoch
            # denominator (0.06) is still above it.
            self.run_main(trees, Path(folder) / "floor", "--min-denominator", "0.05", revision=True, plots=True)
            floor = self.outputs(Path(folder) / "floor")
        self.assertEqual(drawn, [f"retention_{domain}.png" for domain in sorted(TARGETS)])
        self.assertFalse([name for name in floor["files"] if name.endswith(".png")])
        self.assertTrue(all(row["recovery_e3"] == "" for row in floor["points"].values()))
        self.assertTrue(all(row["recovery_e12"] != "" for row in floor["points"].values()))
        self.assertTrue(all(row["delta_target_recovery"] == "" for row in floor["paired"].values()))

    def matching(self, tree: Path, test: str, fields: tuple[str, ...], values: tuple) -> list[Path]:
        """metrics.json of the sets of `tree` scored on `test` whose run fields `fields` have `values`."""
        return [path for path in sorted(tree.glob(f"*/{test}/metrics.json"))
                if tuple(json.loads(path.read_text(encoding="utf-8"))["run"][name] for name in fields) == values]

    def drop(self, tree: Path, test: str, fields: tuple[str, ...], values: tuple) -> None:
        """Remove the sets of `tree` scored on `test` whose run fields `fields` have `values`."""
        for path in self.matching(tree, test, fields, values):
            shutil.rmtree(path.parent.parent)

    def test_rows_of_a_shift_share_the_redraw_of_each_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            trees = self.build(Path(folder))
            # The weather 12-epoch seed 0 is 0.002 above mAP(0) in the first 16 of 80 test resamples (seed 1: 0.06 in
            # all), so a redraw of both seeds on seed 0 leaves the 12-epoch denominator under the floor there.
            [path] = self.matching(trees["revision"], "weather", ("domain", "budget", "epochs", "seed"), ("weather", "full", 12, 0))
            boot = np.load(path.parent / "bootstrap_class_ap.npy")
            boot[:16] -= 0.058
            np.save(path.parent / "bootstrap_class_ap.npy", boot)
            self.run_main(trees, Path(folder) / "out", revision=True)
            outputs = self.outputs(Path(folder) / "out")
        weather = [row for key, row in outputs["points"].items() if key[0] == "weather"]
        shift = outputs["summary"]["references"]["weather"]
        self.assertGreater(shift["denominator_e12_undefined"], 0)
        self.assertEqual(shift["recovery_e12_undefined_fraction"], shift["denominator_e12_undefined"] / 80)
        # Every row of the shift divides by the same redrawn denominators: one count and one verdict for the shift.
        self.assertEqual({row["recovery_e12_undefined"] for row in weather}, {str(shift["denominator_e12_undefined"])})
        self.assertEqual({row["recovery_e12_determined"] for row in weather}, {str(shift["recovery_e12_determined"])})
        self.assertEqual({row["recovery_e3_undefined"] for row in weather}, {"0"})
        self.assertEqual(shift["denominator_e3_undefined"], 0)

    def test_a_non_finite_resample_is_counted_not_fatal(self):
        with tempfile.TemporaryDirectory() as folder:
            trees = self.build(Path(folder), replicates=40)
            [path] = self.matching(trees["campaign"], "lighting", ("stage",), ("source",))
            boot = np.load(path.parent / "bootstrap_class_ap.npy")
            boot[3, CLASSES.index("pedestrian")] = np.nan  # resample 3 of mAP(0) on lighting
            np.save(path.parent / "bootstrap_class_ap.npy", boot)
            self.run_main(trees, Path(folder) / "out")
            outputs = self.outputs(Path(folder) / "out")
        for (domain, method, budget), row in outputs["points"].items():
            undefined = "1" if domain == "lighting" else "0"
            self.assertEqual((row["gain_undefined"], row["recovery_e3_undefined"]), (undefined, undefined), (method, budget))
            self.assertEqual((row["retention_undefined"], row["source_change_undefined"]), ("0", "0"))
            self.assertNotEqual(row["gain_ci_low"], "")
        self.assertEqual(outputs["summary"]["references"]["lighting"]["denominator_e3_undefined"], 1)
        self.assertEqual(outputs["paired"][("lighting", "adapt - source model", 0)]["delta_target_undefined"], "1")
        self.assertEqual(outputs["paired"][("lighting", "adapt - FT", 10)]["delta_target_undefined"], "0")

    def test_the_ft_arm_is_the_locked_schedule_with_both_reference_seeds(self):
        with tempfile.TemporaryDirectory() as folder:
            trees = self.build(Path(folder) / "a", replicates=20)
            for test in ("lighting", "source"):
                self.drop(trees["campaign"], test, ("stage", "domain", "budget", "seed"), ("fine-tune", "lighting", "full", 1))
            with self.assertRaisesRegex(ValueError, "lighting: the 3-epoch full-pool reference needs seeds 0 and 1, found 0"):
                self.run_main(trees, Path(folder) / "out")
            trees = self.build(Path(folder) / "b", replicates=20)
            for test in ("lighting", "source"):
                for path in self.matching(trees["campaign"], test, ("stage", "domain", "budget"), ("fine-tune", "lighting", "full")):
                    metrics = json.loads(path.read_text(encoding="utf-8"))
                    metrics["run"]["epochs"] = 12
                    path.write_text(json.dumps(metrics), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"lighting: campaign fine-tune runs with epochs \[12\]"):
                self.run_main(trees, Path(folder) / "out")
            self.assertFalse((Path(folder) / "out").exists())

    def test_the_source_model_is_scored_once_per_test_set(self):
        with tempfile.TemporaryDirectory() as folder:
            trees = self.build(Path(folder), replicates=20)
            [path] = self.matching(trees["campaign"], "lighting", ("stage",), ("source",))
            shutil.copytree(path.parent.parent, trees["campaign"] / "999")
            with self.assertRaisesRegex(ValueError, "scored exactly once .* found it on compound, lighting, lighting, source"):
                self.run_main(trees, Path(folder) / "out")

    def test_missing_entries_stop_the_script(self):
        with tempfile.TemporaryDirectory() as folder:
            trees = self.build(Path(folder) / "a", replicates=20)
            self.drop(trees["rq5"], "source", ("stage", "domain", "budget", "seed"), ("adapt", "lighting", "100", 1))
            with self.assertRaisesRegex(ValueError, r"lighting adapt N = 100: seed\(s\) \[1\] have no entry on the source"):
                self.run_main(trees, Path(folder) / "out")
            self.assertFalse((Path(folder) / "out").exists())
            # The 12-epoch full-pool runs are a recovery reference: both seeds are required.
            trees = self.build(Path(folder) / "b", replicates=20)
            self.drop(trees["revision"], "weather", ("domain", "budget", "epochs", "seed"), ("weather", "full", 12, 1))
            with self.assertRaisesRegex(ValueError, "weather: the 12-epoch full-pool ceiling needs seeds 0 and 1, found 0"):
                self.run_main(trees, Path(folder) / "out", revision=True)
            self.run_main(trees, Path(folder) / "out")  # without the option the revision tree is not read


if __name__ == "__main__":
    unittest.main()
