"""Campaign analysis rules of the 2026-11-21 revision: threshold status, log-linear fit, denominator, detection limit,
and the analyses against the 12-epoch ceilings (run kinds and their seeds, shared test resamples, measured top rung,
paired E12 - E3 difference).

Synthetic arrays and synthetic scored trees in a temporary directory only; nothing here reads runs/.
"""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import inspect
import io
import json
import math
import sys
import tempfile
import unittest
import warnings
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_campaign import (CEILING_SEEDS, DENOMINATOR_SEED_OFFSET, E12_SEED_OFFSET, MATERIAL_DELTA,  # noqa: E402
                              PAIRED_SEED_OFFSET, REVISION_SEEDS, bootstrap_fit_thresholds,
                              bootstrap_recovery_measured_top, ceiling_comparison, censored_bounds, curve_analysis,
                              denominator_row, detection_limit, duplicate_runs, e12_ceilings, fit_log_linear, fit_region,
                              fit_rows, fit_threshold, holdout_check, inside_ladder_bracket, load, log_linear_ols, main,
                              matched_e3_ceiling, material_flags, paired_seed_difference, plot, region_fits,
                              require_ceiling_seeds, require_one_schedule, require_shared_resamples, resample_signatures,
                              revision_groups, revision_kind, threshold_rows, threshold_status, write_tables)
from bdd_pipeline.campaign import MATCHED_POOL, TARGETS  # noqa: E402
from bdd_pipeline.data import CLASSES, EXPECTED_DOMAINS, RELIABLE_CLASSES  # noqa: E402
from bdd_pipeline.recovery import bootstrap_recovery  # noqa: E402

BUDGETS = [10, 100, 250, 500, 1000, 2000, 5000]  # the last rung is the ceiling
CEILING = 5000
N75 = math.exp(7.25)  # where line() reaches 0.75: about 1408


def line(budget: float, alpha: float = -0.7, beta: float = 0.2) -> float:
    return alpha + beta * math.log(budget)


def ladder_point() -> dict[int, float]:
    """Negative at N = 10, on the line from N = 100, 1 at the ceiling: N75 is 2000 and N90 only the ceiling rung."""
    return {10: -1.0, **{n: line(n) for n in BUDGETS[1:-1]}, CEILING: 1.0}


def spread_curves(budgets: list[int], point: dict[int, float], width: dict[int, float], replicates: int = 201) -> np.ndarray:
    """Replicate curves [replicate, budget]: the point curve plus width[N] * o, o evenly spaced in [-1, 1].

    The 95% band at a rung is then point -+ 0.95 * width, and a width shared by all region rungs shifts alpha only.
    """
    offsets = np.linspace(-1, 1, replicates)
    return np.stack([point[n] + width.get(n, 0.0) * offsets for n in budgets], axis=1)


def ladder_curves() -> np.ndarray:
    """Band lower bound below 0 at N = 100 (0.22 - 0.95 * 0.3) and above 0 from N = 250 on."""
    return spread_curves(BUDGETS, ladder_point(), {10: 0.05, 100: 0.3, 250: 0.05, 500: 0.05, 1000: 0.05, 2000: 0.05})


class LogLinearFitTest(unittest.TestCase):
    def test_recovers_alpha_and_beta_exactly(self):
        region = [250, 500, 1000, 2000, 5000]
        fit = fit_log_linear(region, [line(n, -0.5, 0.15) for n in region])
        self.assertAlmostEqual(fit["alpha"], -0.5, places=12)
        self.assertAlmostEqual(fit["beta"], 0.15, places=12)
        self.assertAlmostEqual(fit["r2"], 1.0, places=12)
        self.assertAlmostEqual(fit["rmse"], 0.0, places=12)
        self.assertEqual(fit["form"], "alpha + beta * ln N")

    def test_fewer_than_four_rungs_gives_no_fit(self):
        self.assertIsNone(fit_log_linear([250, 500, 1000], [line(n) for n in (250, 500, 1000)]))
        self.assertIsNone(fit_log_linear([], []))
        self.assertAlmostEqual(fit_log_linear([250, 500, 1000, 2000], [line(n) for n in (250, 500, 1000, 2000)])["beta"], 0.2)

    def test_r2_and_rmse_on_scattered_points(self):
        # In log10 units the rungs are x = 0, 1, 2, 3: slope 3/5, intercept 0.1, residuals -0.1, 0.3, -0.3, 0.1.
        fit = fit_log_linear([1, 10, 100, 1000], [0.0, 1.0, 1.0, 2.0])
        self.assertAlmostEqual(fit["alpha"], 0.1)
        self.assertAlmostEqual(fit["beta"], 0.6 / math.log(10))
        self.assertAlmostEqual(fit["r2"], 0.9)
        self.assertAlmostEqual(fit["rmse"], math.sqrt(0.05))

    def test_r2_is_none_without_variance(self):
        fit = fit_log_linear([250, 500, 1000, 2000], [0.5] * 4)
        self.assertIsNone(fit["r2"])
        self.assertAlmostEqual(fit["alpha"], 0.5)
        self.assertAlmostEqual(fit["beta"], 0.0)

    def test_ols_fits_one_curve_per_row(self):
        region = [100, 200, 400]
        curves = np.array([[line(n, 0.1, 0.3) for n in region], [line(n, -2.0, 0.5) for n in region]])
        alpha, beta = log_linear_ols(region, curves)
        np.testing.assert_allclose(alpha, [0.1, -2.0], atol=1e-12)
        np.testing.assert_allclose(beta, [0.3, 0.5], atol=1e-12)


class FitRegionTest(unittest.TestCase):
    def test_starts_at_the_first_sustained_positive_rung_and_excludes_the_ceiling(self):
        curve = {10: -1.2, 100: -0.02, 250: 0.25, 500: 0.35, 1000: 0.45, 2000: 0.55, 5000: 1.0}
        self.assertEqual(fit_region(curve, 5000), [250, 500, 1000, 2000])

    def test_positive_rung_before_a_dip_does_not_start_the_region(self):
        self.assertEqual(fit_region({100: 0.05, 250: -0.01, 500: 0.3, 1000: 0.5, 2000: 1.0}, 2000), [500, 1000])

    def test_zero_is_not_above_zero(self):
        self.assertEqual(fit_region({250: 0.0, 500: 0.3, 1000: 1.0}, 1000), [500])

    def test_empty_when_only_the_ceiling_rung_is_positive_or_nothing_is(self):
        self.assertEqual(fit_region({500: -0.1, 1000: 1.0}, 1000), [])
        self.assertEqual(fit_region({500: 0.4, 1000: math.nan}, 1000), [])

    def test_ci_rule_uses_the_band_and_point_rule_the_point_estimate(self):
        fits = region_fits(BUDGETS, ladder_point(), ladder_curves(), CEILING)
        self.assertEqual(fits["ci"]["region"], [250, 500, 1000, 2000])
        self.assertEqual(fits["point"]["region"], [100, 250, 500, 1000, 2000])
        for rule in ("ci", "point"):
            self.assertAlmostEqual(fits[rule]["fit"]["alpha"], -0.7, places=12)
            self.assertAlmostEqual(fits[rule]["fit"]["beta"], 0.2, places=12)
            self.assertAlmostEqual(fits[rule]["predicted_at_ceiling"], line(CEILING), places=12)
            self.assertIs(fits[rule]["monotone"], True)

    def test_monotone_reports_a_dip_inside_the_region(self):
        point = {**ladder_point(), 1000: line(500) - 0.01}
        fits = region_fits(BUDGETS, point, spread_curves(BUDGETS, point, {}), CEILING)
        self.assertEqual(fits["ci"]["region"], [100, 250, 500, 1000, 2000])  # still above 0, so the dip stays in the region
        self.assertIs(fits["ci"]["monotone"], False)
        self.assertLess(fits["ci"]["fit"]["r2"], 1.0)

    def test_monotone_is_judged_inside_the_region_and_a_plateau_passes(self):
        # The fall from N = 100 to N = 250 is before the region; inside it the curve never falls, and 0.4 twice is no fall.
        point = {100: -0.2, 250: -0.5, 500: 0.3, 1000: 0.4, 2000: 0.4, 4000: 0.6, 8000: 0.7, 10000: 1.0}
        budgets = sorted(point)
        fits = region_fits(budgets, point, spread_curves(budgets, point, {}), 10000)
        self.assertEqual(fits["ci"]["region"], [500, 1000, 2000, 4000, 8000])
        self.assertIs(fits["ci"]["monotone"], True)

    def test_ci_rule_reads_the_95_percent_band(self):
        # Lower bounds at a point recovery of 0.1: width 0.108 gives -0.0026 at 95% (+0.0028 at 90%), so that rung is out;
        # width 0.103 gives +0.0022 at 95% (-0.0020 at 99%), so that rung is in.
        point = {100: 0.1, 250: 0.1, 500: 0.3, 1000: 0.5, 2000: 0.7, 5000: 1.0}
        budgets = sorted(point)
        fits = region_fits(budgets, point, spread_curves(budgets, point, {100: 0.108, 250: 0.103}), 5000)
        self.assertEqual(fits["ci"]["region"], [250, 500, 1000, 2000])
        self.assertEqual(fits["point"]["region"], [100, 250, 500, 1000, 2000])

    def test_a_measured_top_rung_extends_the_region(self):
        curve = {10: -1.2, 100: -0.02, 250: 0.25, 500: 0.35, 1000: 0.45, 2000: 0.55, 5000: 0.93}
        self.assertEqual(fit_region(curve, 5000, top_measured=True), [250, 500, 1000, 2000, 5000])
        self.assertEqual(fit_region(curve, 5000), [250, 500, 1000, 2000])  # read as the ceiling, the same rung is left out
        self.assertEqual(fit_region({500: -0.1, 1000: 0.3}, 1000, top_measured=True), [1000])
        # A measured top rung that is not above 0 leaves no rung from which the curve stays above 0.
        self.assertEqual(fit_region({250: 0.2, 500: 0.4, 1000: 0.0}, 1000, top_measured=True), [])

    def test_fit_reaches_the_measured_top_rung(self):
        # The rung at N = 5000 is measured on the line (1.0034), not 1 by construction: five region rungs instead of four,
        # and the line reaches 0.90 at exp(8) = 2981, which is now inside the region.
        point = {**ladder_point(), CEILING: line(CEILING)}
        curves = spread_curves(BUDGETS, point, {10: 0.05, 100: 0.3})
        fits = region_fits(BUDGETS, point, curves, CEILING, top_measured=True)
        self.assertEqual(fits["ci"]["region"], [250, 500, 1000, 2000, 5000])
        self.assertEqual(fits["point"]["region"], [100, 250, 500, 1000, 2000, 5000])
        for rule in ("ci", "point"):
            self.assertAlmostEqual(fits[rule]["fit"]["alpha"], -0.7, places=12)
            self.assertAlmostEqual(fits[rule]["fit"]["beta"], 0.2, places=12)
            self.assertAlmostEqual(fits[rule]["thresholds"]["N90"]["fit"], math.exp(8), places=6)
            self.assertAlmostEqual(fits[rule]["predicted_at_ceiling"], 1.0034386383, places=9)  # the fitted value at the top rung
            self.assertEqual([row["is_ceiling_rung"] for row in fits[rule]["holdout"]], [False] * len(fits[rule]["holdout"]))
        self.assertEqual([row["budget"] for row in fits["ci"]["holdout"]], [2000, 5000])
        self.assertAlmostEqual(fits["ci"]["holdout"][-1]["error"], 0.0, places=12)
        as_ceiling = region_fits(BUDGETS, point, curves, CEILING)
        self.assertEqual(as_ceiling["ci"]["region"], [250, 500, 1000, 2000])
        self.assertIsNone(as_ceiling["ci"]["thresholds"]["N90"]["fit"])
        self.assertEqual([(row["budget"], row["is_ceiling_rung"]) for row in as_ceiling["ci"]["holdout"]],
                         [(2000, False), (5000, True)])


class FitThresholdTest(unittest.TestCase):
    def test_interpolates_inside_the_region_only(self):
        value = fit_threshold(-0.5, 0.15, 0.75, 250, 5000)
        self.assertAlmostEqual(value, math.exp(1.25 / 0.15))
        self.assertTrue(4160 < value < 4161)
        self.assertIsNone(fit_threshold(-0.5, 0.15, 0.75, 250, 2000))  # above the largest region rung
        self.assertIsNone(fit_threshold(-0.5, 0.15, 0.75, 5000, 10000))  # below the smallest region rung

    def test_both_ends_of_the_region_are_inside(self):
        self.assertEqual(fit_threshold(0.75, 0.5, 0.75, 1, 10), 1.0)  # on the smallest rung; exp(0) is exact
        self.assertEqual(fit_threshold(0.75, 0.5, 0.75, 0.5, 1), 1.0)  # on the largest rung

    def test_needs_a_rising_line(self):
        self.assertIsNone(fit_threshold(0.75, 0.0, 0.75, 250, 5000))
        # A falling line crosses 0.75 at the same 4160 images; that is still not a threshold.
        self.assertIsNone(fit_threshold(2.0, -0.15, 0.75, 250, 5000))
        self.assertIsNone(fit_threshold(0.0, 1e-300, 0.75, 250, 5000))  # exp overflows
        self.assertIsNone(fit_threshold(math.nan, 0.15, 0.75, 250, 5000))

    def test_region_fits_reports_n75_inside_and_drops_n90_outside(self):
        fits = region_fits(BUDGETS, ladder_point(), ladder_curves(), CEILING)
        self.assertAlmostEqual(fits["ci"]["thresholds"]["N75"]["fit"], N75, places=6)
        self.assertIsNone(fits["ci"]["thresholds"]["N90"]["fit"])  # exp(8) = 2981 is beyond the last region rung, 2000
        self.assertNotIn("N_BE", fits["ci"]["thresholds"])


class HoldoutCheckTest(unittest.TestCase):
    def test_three_lowest_rungs_predict_the_rest_and_the_ceiling(self):
        region = [250, 500, 1000, 2000, 5000]
        point = {n: line(n, -0.5, 0.15) for n in region[:3]}
        point.update({2000: line(2000, -0.5, 0.15) + 0.02, 5000: line(5000, -0.5, 0.15) - 0.03, 10000: 1.0})
        rows = holdout_check(region, point, 10000)
        self.assertEqual([row["budget"] for row in rows], [2000, 5000, 10000])
        self.assertEqual([row["is_ceiling_rung"] for row in rows], [False, False, True])
        self.assertEqual({row["fit_rungs"] for row in rows}, {"250 500 1000"})
        for row, error in zip(rows, (-0.02, 0.03, -0.5 + 0.15 * math.log(10000) - 1.0)):
            self.assertAlmostEqual(row["predicted"], line(row["budget"], -0.5, 0.15), places=12)
            self.assertAlmostEqual(row["observed"], point[row["budget"]], places=12)
            self.assertAlmostEqual(row["error"], error, places=12)
        self.assertAlmostEqual(rows[-1]["error"], -0.118448947, places=8)

    def test_needs_four_region_rungs(self):
        self.assertEqual(holdout_check([250, 500, 1000], {250: 0.3, 500: 0.4, 1000: 0.5, 1738: 1.0}, 1738), [])

    def test_a_measured_top_rung_is_the_last_region_rung_and_no_rung_is_added(self):
        region = [250, 500, 1000, 2000, 5000]
        point = {n: line(n, -0.5, 0.15) for n in region[:3]}
        point.update({2000: line(2000, -0.5, 0.15) + 0.02, 5000: 0.9})
        rows = holdout_check(region, point, 5000, top_measured=True)
        self.assertEqual([row["budget"] for row in rows], [2000, 5000])
        self.assertEqual([row["is_ceiling_rung"] for row in rows], [False, False])
        self.assertAlmostEqual(rows[0]["error"], -0.02, places=12)
        self.assertAlmostEqual(rows[1]["observed"], 0.9, places=12)
        self.assertAlmostEqual(rows[1]["error"], -0.122421021, places=8)  # -0.5 + 0.15 ln 5000 - 0.9
        # A region that stops below the pool gets no row at the pool either: nothing there is 1 by construction.
        self.assertEqual([row["budget"] for row in holdout_check(region[:4], point, 5000, top_measured=True)], [2000])
        self.assertEqual([row["budget"] for row in holdout_check(region[:4], {**point, 5000: 1.0}, 5000)], [2000, 5000])


class BootstrapFitThresholdTest(unittest.TestCase):
    budgets = [50, 100, 200, 400, 800, 1600]
    region = [100, 200, 400, 800]

    def curve(self, threshold: float, beta: float) -> list[float]:
        """A replicate that is exactly log-linear and reaches 0.75 at `threshold`; 1 at the ceiling rung."""
        return [0.75 + beta * math.log(n / threshold) for n in self.budgets[:-1]] + [1.0]

    def test_refits_every_replicate_on_the_same_rungs_and_skips_nan(self):
        curves = np.array([self.curve(150, 0.2), self.curve(300, 0.2),
                           self.curve(600, 0.5),  # below 0 at N = 100: the region is not re-derived, so it still counts
                           self.curve(1000, 0.2),  # crosses 0.75 beyond the region: not an interpolation
                           self.curve(300, 0.2),  # NaN inside the region (set below): skipped
                           self.curve(300, 0.2),  # NaN outside the region (set below): kept
                           self.curve(300, -0.2)])  # falling line: no threshold
        curves[4, self.budgets.index(400)] = np.nan
        curves[5, self.budgets.index(50)] = np.nan
        self.assertLess(curves[2, self.budgets.index(100)], 0)
        out = bootstrap_fit_thresholds(self.budgets, curves, self.region)
        self.assertEqual(out["N75"]["replicates_finite"], 6)
        self.assertEqual(out["N75"]["replicates_defined"], 4)  # 150, 300, 600, 300
        self.assertAlmostEqual(out["N75"]["ci_low"], 161.25, places=6)  # 2.5th percentile of 150, 300, 300, 600
        self.assertAlmostEqual(out["N75"]["ci_high"], 577.5, places=6)
        # 0.90 is reached exp(0.15 / beta) times later: 317.6, 635.1 and 635.1 inside; 809.9 and 2117 outside the region.
        later = math.exp(0.75)
        self.assertEqual(out["N90"]["replicates_defined"], 3)
        self.assertAlmostEqual(out["N90"]["ci_low"], 150 * later + 0.05 * 150 * later, places=6)
        self.assertAlmostEqual(out["N90"]["ci_high"], 300 * later, places=6)

    def test_no_defined_replicate_gives_no_interval(self):
        curves = np.array([self.curve(5000, 0.2), [np.nan] * 6])
        out = bootstrap_fit_thresholds(self.budgets, curves, self.region)
        self.assertEqual(out["N75"], {"ci_low": None, "ci_high": None, "replicates_defined": 0, "replicates_finite": 1,
                                      "ci_censored": None})

    def test_censored_bounds_follow_the_percentile_over_all_replicates(self):
        # 1000 replicates: the 2.5th percentile lies between ranks 24 and 25, the 97.5th between ranks 974 and 975.
        self.assertEqual(censored_bounds(0, 1000, 0), "no")
        self.assertEqual(censored_bounds(24, 952, 24), "no")  # ranks 24..975 are defined
        self.assertEqual(censored_bounds(25, 951, 24), "low")  # rank 24 crosses before the region
        self.assertEqual(censored_bounds(24, 951, 25), "high")  # rank 975 crosses after the region
        self.assertEqual(censored_bounds(25, 950, 25), "both")
        self.assertEqual(censored_bounds(0, 891, 109), "high")
        self.assertEqual(censored_bounds(0, 10, 990), "both")  # even the 2.5th percentile is beyond the region
        self.assertEqual(censored_bounds(990, 10, 0), "both")
        # 201 replicates: the percentiles sit exactly on ranks 5 and 195.
        self.assertEqual(censored_bounds(5, 191, 5), "no")
        self.assertEqual(censored_bounds(6, 190, 5), "low")
        self.assertEqual(censored_bounds(5, 190, 6), "high")
        self.assertIsNone(censored_bounds(3, 0, 4))

    def test_interval_is_marked_where_the_region_cuts_the_replicates(self):
        # Replicate N75 every 5 images from 502 to 1502 and a region that ends at 800: the 60 replicates up to 797 cross
        # inside it, and their percentiles (509.4, 789.6) say nothing about the 141 that cross later.
        curves = np.array([self.curve(value, 0.2) for value in np.linspace(502, 1502, 201)])
        out = bootstrap_fit_thresholds(self.budgets, curves, self.region)["N75"]
        self.assertEqual((out["replicates_defined"], out["replicates_finite"]), (60, 201))
        self.assertAlmostEqual(out["ci_low"], 509.375, places=6)
        self.assertAlmostEqual(out["ci_high"], 789.625, places=6)
        self.assertEqual(out["ci_censored"], "high")
        # Replicate N75 every 0.5 image from 60.25 to 160.25 and a region that starts at 100: the 80 earlier ones are cut off.
        curves = np.array([self.curve(value, 0.2) for value in np.linspace(60.25, 160.25, 201)])
        early = bootstrap_fit_thresholds(self.budgets, curves, self.region)["N75"]
        self.assertEqual((early["replicates_defined"], early["replicates_finite"]), (121, 201))
        self.assertAlmostEqual(early["ci_low"], 101.75, places=6)
        self.assertAlmostEqual(early["ci_high"], 158.75, places=6)
        self.assertEqual(early["ci_censored"], "low")
        # A falling line never reaches the level, so it ranks above. The 97.5th percentile of 201 is rank 195 of 0..200.
        for falling, marker in ((5, "no"), (6, "high")):
            curves = np.array([self.curve(300, 0.2)] * (201 - falling) + [self.curve(300, -0.2)] * falling)
            out = bootstrap_fit_thresholds(self.budgets, curves, self.region)["N75"]
            self.assertEqual((out["replicates_defined"], out["ci_censored"]), (201 - falling, marker))
        # A replicate without a value in the region is skipped, not ranked: 10 of them do not move the 97.5th percentile.
        curves = np.array([self.curve(300, 0.2)] * 201 + [[np.nan] * 6] * 10)
        out = bootstrap_fit_thresholds(self.budgets, curves, self.region)["N75"]
        self.assertEqual((out["replicates_defined"], out["replicates_finite"], out["ci_censored"]), (201, 201, "no"))

    def test_region_fits_refits_the_replicates_on_exactly_its_region(self):
        # Widths that are not linear in ln N: a replicate is then no straight line, and a fit on other rungs gives another
        # threshold (on the three largest region rungs the interval would be 1082 to 1772).
        width = {10: 0.05, 100: 0.3, 250: 0.02, 500: 0.08, 1000: 0.03, 2000: 0.06}
        curves = spread_curves(BUDGETS, ladder_point(), width)
        estimate = region_fits(BUDGETS, ladder_point(), curves, CEILING)["ci"]["thresholds"]["N75"]
        region = [250, 500, 1000, 2000]
        expected = []
        for replicate in curves[:, [BUDGETS.index(n) for n in region]]:
            beta, alpha = np.polyfit(np.log(region), replicate, 1)
            expected.append(math.exp((0.75 - alpha) / beta))
        self.assertTrue(all(250 <= value <= 2000 for value in expected))
        self.assertEqual((estimate["replicates_defined"], estimate["replicates_finite"], estimate["ci_censored"]),
                         (201, 201, "no"))
        self.assertAlmostEqual(estimate["ci_low"], float(np.percentile(expected, 2.5)), places=6)
        self.assertAlmostEqual(estimate["ci_high"], float(np.percentile(expected, 97.5)), places=6)
        self.assertAlmostEqual(estimate["ci_low"], 1100.118, places=2)
        self.assertAlmostEqual(estimate["ci_high"], 1847.707, places=2)

    def test_interval_through_region_fits(self):
        # Every replicate is the line shifted by 0.05 * o, so its N75 is exp(-0.25 * o) times the point N75; all inside.
        fits = region_fits(BUDGETS, ladder_point(), ladder_curves(), CEILING)
        expected = N75 * np.exp(-0.25 * np.linspace(-1, 1, 201))
        estimate = fits["ci"]["thresholds"]["N75"]
        self.assertEqual(estimate["replicates_defined"], 201)
        self.assertAlmostEqual(estimate["ci_low"], float(np.percentile(expected, 2.5)), places=6)
        self.assertAlmostEqual(estimate["ci_high"], float(np.percentile(expected, 97.5)), places=6)
        self.assertTrue(1100 < estimate["ci_low"] < N75 < estimate["ci_high"] < 1800)
        self.assertEqual(fits["ci"]["thresholds"]["N90"]["replicates_defined"], 0)
        self.assertIsNone(fits["ci"]["thresholds"]["N90"]["ci_low"])

    def test_no_interval_without_a_point_estimate(self):
        # The point line reaches 0.90 at exp(8) = 2981, beyond the region; replicates shifted up by 0.1 or more reach it
        # inside. Their percentiles are not an interval for a threshold that has no point estimate; the count is kept.
        curves = spread_curves(BUDGETS, ladder_point(), {n: 0.2 for n in BUDGETS[1:-1]})
        estimate = region_fits(BUDGETS, ladder_point(), curves, CEILING)["point"]["thresholds"]["N90"]
        self.assertIsNone(estimate["fit"])
        self.assertEqual(estimate["replicates_defined"], 61)  # offsets o with 2981 * exp(-o) <= 2000: o >= 0.4
        self.assertEqual(estimate["replicates_finite"], 201)
        self.assertIsNone(estimate["ci_low"])
        self.assertIsNone(estimate["ci_high"])
        self.assertIsNone(estimate["ci_censored"])
        self.assertIsNone(estimate["inside_ladder_bracket"])


class ThresholdRowsTest(unittest.TestCase):
    def test_status_values(self):
        self.assertEqual(threshold_status(1000, 1738), "determined")
        self.assertEqual(threshold_status(1738, 1738), "not determined within pool")
        self.assertEqual(threshold_status(None, 1738), "not reached")

    def test_status_with_a_measured_top_rung(self):
        self.assertEqual(threshold_status(1000, 1738, top_measured=True), "determined")
        self.assertEqual(threshold_status(1738, 1738, top_measured=True), "determined")  # the top rung itself passes
        self.assertEqual(threshold_status(None, 1738, top_measured=True), "not reached")

    def test_rows_with_a_measured_top_rung(self):
        # The rung at N = 5000 is measured at 0.92 with a band of 0.8725 to 0.9675.
        point = {**ladder_point(), CEILING: 0.92}
        width = {10: 0.05, 100: 0.3, 250: 0.05, 500: 0.05, 1000: 0.05, 2000: 0.05, CEILING: 0.05}
        curves = spread_curves(BUDGETS, point, width)
        rows = threshold_rows("lighting", "reliable", "full-pool-e12", BUDGETS, point, curves, CEILING, top_measured=True)
        self.assertEqual([row["point"] for row in rows], [100, 2000, 5000])
        self.assertEqual([row["status"] for row in rows], ["determined"] * 3)
        # N90: the optimistic band passes at the top rung only, the pessimistic band (0.8725) nowhere.
        self.assertEqual([(row["band_from"], row["band_to"]) for row in rows], [(100, 250), (2000, 2000), (5000, None)])
        self.assertEqual([(row["ladder_last_failing"], row["ladder_first_passing"]) for row in rows],
                         [(10, 100), (1000, 2000), (2000, 5000)])
        self.assertEqual([row["fit_region_start"] for row in rows], [250] * 3)
        self.assertLess(rows[1]["fit_r2"], 1.0)  # the measured top rung is in the fit and lies below the line (1.0034)
        # The same numbers read with the rung at N = 5000 as the ceiling: N90 would not count as measured.
        as_ceiling = threshold_rows("lighting", "reliable", "full-pool", BUDGETS, point, curves, CEILING)
        self.assertEqual([row["status"] for row in as_ceiling], ["determined", "determined", "not determined within pool"])
        self.assertAlmostEqual(as_ceiling[1]["fit_r2"], 1.0, places=12)

    def test_a_top_rung_that_fails_leaves_the_threshold_not_reached(self):
        # 0.82 at N = 2000 passes 0.75, but the measured top rung (0.70) does not: no rung from which every larger one passes.
        point = {**ladder_point(), CEILING: 0.70}
        rows = threshold_rows("weather", "reliable", "full-pool-e12", BUDGETS, point, spread_curves(BUDGETS, point, {}),
                              CEILING, top_measured=True)
        self.assertEqual([row["point"] for row in rows], [100, None, None])
        self.assertEqual([row["status"] for row in rows], ["determined", "not reached", "not reached"])
        self.assertEqual([(row["ladder_last_failing"], row["ladder_first_passing"]) for row in rows],
                         [(10, 100), (5000, None), (5000, None)])
        # A top rung above 1 passes like any rung: recovery is not clipped.
        above = {**ladder_point(), CEILING: 1.4}
        rows = threshold_rows("weather", "reliable", "full-pool-e12", BUDGETS, above, spread_curves(BUDGETS, above, {}),
                              CEILING, top_measured=True)
        self.assertEqual([(row["point"], row["status"]) for row in rows],
                         [(100, "determined"), (2000, "determined"), (5000, "determined")])

    def test_rows_with_a_measured_top_rung_end_with_the_replicate_counts_of_the_band(self):
        # The top rung is measured at 0.935. Over all 201 replicates its band starts at 0.935 - 0.95 * 0.05 = 0.8875, below
        # 0.90, so the pessimistic band never reaches N90. Without the 30 lowest replicates (o from -1 to -0.71; the
        # 2.5th percentile of the other 171 is o = -0.6575) it starts at 0.9021 and N90 reads "at most 5000".
        point = {**ladder_point(), CEILING: 0.935}
        curves = spread_curves(BUDGETS, point, {n: 0.05 for n in BUDGETS})
        rows = threshold_rows("weather", "reliable", "full-pool-e12", BUDGETS, point, curves, CEILING, top_measured=True)
        self.assertEqual([list(row)[-2:] for row in rows], [["band_replicates", "band_undefined_replicates"]] * 3)
        self.assertEqual([(row["band_replicates"], row["band_undefined_replicates"]) for row in rows], [(201, 0)] * 3)
        self.assertEqual((rows[2]["band_from"], rows[2]["band_to"]), (5000, None))
        curves[:30] = np.nan  # as where the redrawn denominator does not exceed the floor: undefined at every rung
        rows = threshold_rows("weather", "reliable", "full-pool-e12", BUDGETS, point, curves, CEILING, top_measured=True)
        self.assertEqual([(row["band_replicates"], row["band_undefined_replicates"]) for row in rows], [(201, 30)] * 3)
        self.assertEqual((rows[2]["band_from"], rows[2]["band_to"]), (5000, 5000))  # a band of the 171 defined replicates
        self.assertEqual([row["fit_replicates_finite"] for row in rows], [None, 171, 171])
        # A replicate without a value at one rung only is counted too: the largest count over the rungs.
        curves[30, BUDGETS.index(500)] = np.nan
        rows = threshold_rows("weather", "reliable", "full-pool-e12", BUDGETS, point, curves, CEILING, top_measured=True)
        self.assertEqual({row["band_undefined_replicates"] for row in rows}, {31})
        # The rows of a 3-epoch analysis keep the columns they had.
        as_ceiling = threshold_rows("weather", "reliable", "full-pool", BUDGETS, point, curves, CEILING)
        self.assertEqual(list(as_ceiling[0])[-1], "fit_inside_ladder_bracket")
        self.assertEqual(list(rows[0])[:-2], list(as_ceiling[0]))

    def test_rows_have_no_n95_and_carry_status_and_the_ci_fit(self):
        rows = threshold_rows("lighting", "reliable", "full-pool", BUDGETS, ladder_point(), ladder_curves(), CEILING)
        self.assertEqual([row["threshold"] for row in rows], ["N_BE", "N75", "N90"])
        self.assertEqual([row["level"] for row in rows], [0.0, 0.75, 0.90])
        self.assertEqual([row["point"] for row in rows], [100, 2000, 5000])
        self.assertEqual([row["status"] for row in rows], ["determined", "determined", "not determined within pool"])
        self.assertEqual([row["ladder_last_failing"] for row in rows], [10, 1000, 2000])
        self.assertEqual([row["ladder_first_passing"] for row in rows], [100, 2000, 5000])
        self.assertEqual([(row["band_from"], row["band_to"]) for row in rows], [(100, 250), (2000, 2000), (5000, 5000)])
        self.assertEqual([row["fit_region_start"] for row in rows], [250, 250, 250])
        self.assertEqual([row["fit_replicates_defined"] for row in rows], [0, 201, 0])
        self.assertIsNone(rows[0]["fit"])  # no fit threshold for N_BE
        self.assertAlmostEqual(rows[1]["fit"], N75, places=6)
        self.assertIsNone(rows[2]["fit"])
        for row in rows:
            self.assertAlmostEqual(row["fit_r2"], 1.0, places=12)
            self.assertNotIn("fit_b", row)
        self.assertEqual([row["fit_replicates_finite"] for row in rows], [None, 201, 201])
        self.assertEqual([row["fit_ci_censored"] for row in rows], [None, "no", None])
        self.assertEqual([row["fit_inside_ladder_bracket"] for row in rows], [None, True, None])

    def test_interval_columns_keep_their_values_and_order_in_both_tables(self):
        # Every replicate is the line shifted by 0.05 * o, so its N75 is exp(-0.25 * o) times the point N75: 1121 to 1787.
        low, high = (float(value) for value in np.percentile(N75 * np.exp(-0.25 * np.linspace(-1, 1, 201)), [2.5, 97.5]))
        self.assertTrue(1100 < low < N75 < high < 1800)
        rows = threshold_rows("lighting", "reliable", "full-pool", BUDGETS, ladder_point(), ladder_curves(), CEILING)
        self.assertAlmostEqual(rows[1]["fit_ci_low"], low, places=6)
        self.assertAlmostEqual(rows[1]["fit_ci_high"], high, places=6)
        self.assertEqual([(row["fit_ci_low"], row["fit_ci_high"]) for row in (rows[0], rows[2])], [(None, None)] * 2)
        fits = region_fits(BUDGETS, ladder_point(), ladder_curves(), CEILING)
        table, _ = fit_rows("lighting", "reliable", "full-pool", fits)
        self.assertAlmostEqual(table[0]["N75_ci_low"], low, places=6)
        self.assertAlmostEqual(table[0]["N75_ci_high"], high, places=6)
        self.assertEqual((table[0]["N90_ci_low"], table[0]["N90_ci_high"]), (None, None))
        self.assertEqual((table[0]["N75_replicates_defined"], table[0]["N90_replicates_defined"], table[0]["replicates_finite"]),
                         (201, 0, 201))
        self.assertEqual((table[0]["N75_ci_censored"], table[0]["N90_ci_censored"]), ("no", None))
        self.assertEqual((table[0]["N75_inside_ladder_bracket"], table[0]["N90_inside_ladder_bracket"]), (True, None))

    def test_status_follows_the_point_threshold_not_the_band(self):
        # At N = 2000 the point recovery is 0.88 and the band 0.69 to 1.07.
        point = {100: 0.2, 250: 0.4, 500: 0.6, 1000: 0.8, 2000: 0.88, 5000: 1.0}
        budgets = sorted(point)
        curves = spread_curves(budgets, point, {2000: 0.2})
        rows = threshold_rows("lighting", "reliable", "full-pool", budgets, point, curves, 5000)
        self.assertEqual([row["point"] for row in rows], [100, 1000, 5000])
        # N75: the pessimistic band passes only at the ceiling rung, yet the point threshold is the measured rung 1000.
        # N90: the optimistic band passes at 2000, yet the point threshold is the ceiling rung.
        self.assertEqual([(row["band_from"], row["band_to"]) for row in rows], [(100, 100), (1000, 5000), (2000, 5000)])
        self.assertEqual([row["status"] for row in rows], ["determined", "determined", "not determined within pool"])
        # The fit (N75 = 991, N90 = 1859) is held against the point rungs as well: 0.88 at N = 2000 contradicts its N90.
        self.assertAlmostEqual(rows[1]["fit"], 991.316, places=2)
        self.assertAlmostEqual(rows[2]["fit"], 1858.506, places=2)
        self.assertEqual([row["fit_inside_ladder_bracket"] for row in rows], [None, True, False])

    def test_n_be_is_strict_and_the_other_levels_are_not(self):
        point = {100: 0.0, 250: 0.75, 500: 0.75, 1000: 0.9, 2000: 1.0}
        budgets = sorted(point)
        rows = threshold_rows("weather", "car", "full-pool", budgets, point, spread_curves(budgets, point, {}), 2000)
        self.assertEqual([row["point"] for row in rows], [250, 250, 1000])  # 0 is not above 0; 0.75 and 0.90 are reached

    def test_fit_is_checked_against_the_measured_rungs(self):
        point = {250: 0.5, 500: 0.6, 1000: 0.7, 2000: 0.8, 5000: 1.0}  # 0.75 lies in (1000, 2000], 0.90 in (2000, 5000]
        self.assertIs(inside_ladder_bracket(1500.0, point, 0.75), True)
        self.assertIs(inside_ladder_bracket(2000.0, point, 0.75), True)  # the first passing rung itself
        self.assertIs(inside_ladder_bracket(1000.0, point, 0.75), False)  # the rung measured below the level
        self.assertIs(inside_ladder_bracket(900.0, point, 0.75), False)
        self.assertIs(inside_ladder_bracket(2500.0, point, 0.75), False)  # past a rung that was measured above the level
        self.assertIs(inside_ladder_bracket(1900.0, point, 0.90), False)
        self.assertIs(inside_ladder_bracket(200.0, point, 0.50), True)  # every rung passes: no failing rung to contradict
        self.assertIs(inside_ladder_bracket(300.0, point, 0.50), False)
        unreached = {250: 0.5, 500: 0.6, 1000: 0.7}  # the ladder says "> 1000"
        self.assertIs(inside_ladder_bracket(1500.0, unreached, 0.75), True)
        self.assertIs(inside_ladder_bracket(900.0, unreached, 0.75), False)
        self.assertIsNone(inside_ladder_bracket(None, point, 0.75))

    def test_a_fit_below_a_failing_rung_is_flagged_in_both_tables(self):
        # The rung at 2000 was measured at 0.89, so N90 > 2000; the line through the four rungs is at 0.908 there and
        # crosses 0.90 at N = 1894. Its N75 = 683 agrees with the rungs: 0.71 at 500 and 0.83 at 1000.
        budgets = [250, 500, 1000, 2000, 5000]
        point = {250: 0.59, 500: 0.71, 1000: 0.83, 2000: 0.89, 5000: 1.0}
        curves = spread_curves(budgets, point, {})
        rows = threshold_rows("compound", "car", "full-pool", budgets, point, curves, 5000)
        self.assertEqual([(row["ladder_last_failing"], row["ladder_first_passing"]) for row in rows[1:]],
                         [(500, 1000), (2000, 5000)])
        self.assertAlmostEqual(rows[1]["fit"], 683.4844, places=3)
        self.assertAlmostEqual(rows[2]["fit"], 1894.1737, places=3)
        self.assertEqual(rows[2]["status"], "not determined within pool")
        self.assertEqual([row["fit_inside_ladder_bracket"] for row in rows], [None, True, False])
        table, _ = fit_rows("compound", "car", "full-pool", region_fits(budgets, point, curves, 5000))
        self.assertEqual([(row["N75_inside_ladder_bracket"], row["N90_inside_ladder_bracket"]) for row in table],
                         [(True, False)] * 2)

    def test_not_reached_and_no_fit_below_four_rungs(self):
        point = {250: 0.2, 500: 0.5, 1000: 0.7}  # a top rung that is not a ceiling, so neither level is reached
        budgets = [250, 500, 1000]
        rows = threshold_rows("weather", "car", "full-pool", budgets, point, spread_curves(budgets, point, {}), 1000)
        self.assertEqual([row["status"] for row in rows], ["determined", "not reached", "not reached"])
        self.assertEqual([row["point"] for row in rows], [250, None, None])
        for row in rows:
            self.assertIsNone(row["fit"])
            self.assertIsNone(row["fit_r2"])
            self.assertEqual(row["fit_replicates_defined"], 0)
            self.assertEqual(row["fit_region_start"], 250)

    def test_fit_rows_keep_the_rung_count_when_there_is_no_fit(self):
        budgets = [100, 250, 500, 1000, 1738]
        point = {**{n: line(n) for n in budgets[:-1]}, 1738: 1.0}
        fits = region_fits(budgets, point, spread_curves(budgets, point, {100: 0.3}), 1738)
        rows, holdout = fit_rows("compound", "reliable", "full-pool", fits)
        self.assertEqual([row["region_rule"] for row in rows], ["ci", "point"])
        self.assertEqual([row["rungs"] for row in rows], [3, 4])
        self.assertEqual([(row["region_start"], row["region_end"]) for row in rows], [(250, 1000), (100, 1000)])
        for name in ("form", "alpha", "beta", "r2", "rmse", "N75", "N75_ci_low", "N75_ci_high", "N90", "predicted_at_ceiling"):
            self.assertIsNone(rows[0][name], name)
        self.assertAlmostEqual(rows[1]["beta"], 0.2, places=12)
        self.assertAlmostEqual(rows[1]["predicted_at_ceiling"], line(1738), places=12)
        self.assertEqual({row["region_rule"] for row in holdout}, {"point"})  # the three-rung region has no held-out check
        self.assertEqual([(row["fit_rungs"], row["budget"], row["is_ceiling_rung"]) for row in holdout],
                         [("100 250 500", 1000, False), ("100 250 500", 1738, True)])


class DetectionLimitTest(unittest.TestCase):
    def test_arithmetic(self):
        drops = {"compound": 0.13, "lighting": 0.10, "weather": 0.02}
        out = detection_limit(-0.012, 0.016, np.array([-0.01, 0.0, 0.01, np.nan]), drops)
        self.assertAlmostEqual(out["se"], 0.01)  # SD with ddof=1 of the three finite replicates
        self.assertAlmostEqual(out["ci_half_width"], 0.014)
        self.assertAlmostEqual(out["mde_80"], 0.0280158522, places=9)  # (1.959964 + 0.841621) * 0.01
        self.assertAlmostEqual(out["largest_compatible_abs"], 0.016)
        self.assertEqual((out["drop_compound"], out["drop_lighting"], out["drop_weather"]), (0.13, 0.10, 0.02))
        self.assertAlmostEqual(out["ci_half_width_over_weather_drop"], 0.7)
        self.assertAlmostEqual(out["mde_80_over_weather_drop"], 1.40079261, places=7)
        self.assertAlmostEqual(out["largest_compatible_abs_over_weather_drop"], 0.8)

    def test_largest_compatible_is_the_far_end_of_the_interval(self):
        out = detection_limit(-0.03, 0.01, np.array([0.0, 0.02]), {"compound": 0.1, "lighting": 0.1, "weather": 0.0})
        self.assertAlmostEqual(out["largest_compatible_abs"], 0.03)
        self.assertAlmostEqual(out["ci_half_width"], 0.02)
        self.assertIsNone(out["mde_80_over_weather_drop"])  # no weather drop to compare against
        gain = detection_limit(-0.03, 0.01, np.array([0.0, 0.02]), {"compound": 0.1, "lighting": 0.1, "weather": -0.04})
        self.assertEqual(gain["drop_weather"], -0.04)
        for name in ("ci_half_width", "mde_80", "largest_compatible_abs"):
            self.assertIsNone(gain[f"{name}_over_weather_drop"], name)  # not -0.5: the model did not drop on weather


class DenominatorRowTest(unittest.TestCase):
    def test_point_values_interval_and_floor_count(self):
        ceiling_boot = np.array([0.625, 0.75, 1.0, 1.5])  # replicate denominators 0.125, 0.25, 0.5, 1.0 for both seeds
        row = denominator_row(0.5, [0.875, 1.125], np.full(4, 0.5), [ceiling_boot, ceiling_boot], 0.25, seed=0)
        self.assertEqual((row["zero_shot"], row["ceiling"], row["denominator"]), (0.5, 1.0, 0.5))
        self.assertEqual(row["ratio_to_floor"], 2.0)
        self.assertIs(row["defined"], True)
        self.assertEqual(row["replicates_at_or_below_floor"], 2)  # 0.125, and 0.25: the floor itself does not pass
        self.assertEqual((row["replicates"], row["ceiling_seeds"]), (4, 2))
        self.assertAlmostEqual(row["ci_low"], 0.134375)  # 2.5th and 97.5th percentiles of 0.125, 0.25, 0.5, 1.0
        self.assertAlmostEqual(row["ci_high"], 0.9625)

    def test_denominator_at_or_below_the_floor_is_not_defined(self):
        row = denominator_row(0.3200, [0.3289], np.full(3, 0.3200), [np.full(3, 0.3289)], 0.01, seed=0)
        self.assertIs(row["defined"], False)
        self.assertAlmostEqual(row["ratio_to_floor"], 0.89)
        self.assertEqual((row["replicates_at_or_below_floor"], row["replicates"], row["ceiling_seeds"]), (3, 3, 1))
        at_floor = denominator_row(0.5, [0.75], np.full(3, 0.5), [np.full(3, 0.75)], 0.25, seed=0)
        self.assertEqual((at_floor["ratio_to_floor"], at_floor["defined"]), (1.0, False))  # recovery() needs it above the floor

    def test_same_test_resample_for_both_terms(self):
        zero_boot = np.array([0.1, 0.2, 0.3, 0.4])  # test-set noise shared by the ceiling cancels in the difference
        row = denominator_row(0.25, [0.30], zero_boot, [zero_boot + 0.05], 0.01, seed=0)
        self.assertAlmostEqual(row["ci_low"], 0.05)
        self.assertAlmostEqual(row["ci_high"], 0.05)
        self.assertEqual(row["replicates_at_or_below_floor"], 0)

    def test_ceiling_seeds_are_redrawn_with_replacement(self):
        # Ceiling seeds at 0.2 and 0.6 around a zero-shot of 0.3: a replicate is -0.1 (seed 0 twice), 0.3 or 0.1.
        boots = [np.full(2000, 0.2), np.full(2000, 0.6)]
        row = denominator_row(0.3, [0.2, 0.6], np.full(2000, 0.3), boots, 0.01, seed=3)
        self.assertAlmostEqual(row["denominator"], 0.1)
        self.assertAlmostEqual(row["ci_low"], -0.1)
        self.assertAlmostEqual(row["ci_high"], 0.3)
        self.assertTrue(400 < row["replicates_at_or_below_floor"] < 600)  # a quarter of 2000 replicates draw seed 0 twice
        self.assertEqual(row, denominator_row(0.3, [0.2, 0.6], np.full(2000, 0.3), boots, 0.01, seed=3))
        other = denominator_row(0.3, [0.2, 0.6], np.full(2000, 0.3), boots, 0.01, seed=4)
        self.assertNotEqual(row["replicates_at_or_below_floor"], other["replicates_at_or_below_floor"])  # the seed is used

    def test_replicate_ceiling_is_the_mean_of_the_redrawn_seeds(self):
        # With a floor of 0.2 the seed mean is at or below it for the draws (0, 0), (0, 1) and (1, 0): three quarters.
        boots = [np.full(2000, 0.2), np.full(2000, 0.6)]
        row = denominator_row(0.3, [0.2, 0.6], np.full(2000, 0.3), boots, 0.2, seed=3)
        self.assertTrue(1400 < row["replicates_at_or_below_floor"] < 1600)
        self.assertAlmostEqual(row["ratio_to_floor"], 0.5)
        self.assertIs(row["defined"], False)

    def test_non_finite_replicates_and_missing_values(self):
        row = denominator_row(0.2, [0.3], np.array([0.2, np.nan, 0.2]), [np.full(3, 0.3)], 0.01, seed=0)
        self.assertEqual((row["replicates"], row["replicates_at_or_below_floor"]), (2, 0))
        self.assertAlmostEqual(row["ci_low"], 0.1)  # the NaN replicate is left out of the interval too
        self.assertAlmostEqual(row["ci_high"], 0.1)
        self.assertIsNone(denominator_row(None, [0.3], np.zeros(3), [np.zeros(3)], 0.01, seed=0))
        self.assertIsNone(denominator_row(0.2, [0.3, None], np.zeros(3), [np.zeros(3), np.zeros(3)], 0.01, seed=0))

    def test_without_a_floor_there_is_no_ratio(self):
        row = denominator_row(0.2, [0.3], np.full(3, 0.2), [np.full(3, 0.3)], 0.0, seed=0)
        self.assertIsNone(row["ratio_to_floor"])
        self.assertIs(row["defined"], True)

    def test_one_bootstrap_vector_per_ceiling_seed(self):
        with self.assertRaisesRegex(ValueError, "2 ceiling values but 1 ceiling bootstrap"):
            denominator_row(0.2, [0.3, 0.31], np.full(3, 0.2), [np.full(3, 0.3)], 0.01, seed=0)


class CurveAnalysisTest(unittest.TestCase):
    noise = np.random.default_rng(0).normal(0, 0.01, 400)  # test-resample noise shared by every run: cancels in recovery

    def entry(self, score: float) -> dict:
        return {"points": {"reliable": score}, "boots": {"reliable": score + self.noise}}

    def scores(self, budget: int) -> list[float]:
        """Three sampling seeds; mAP(0) = 0.2 and the ceiling 0.3, so recovery is (score - 0.2) / 0.1."""
        if budget == 10:
            return [0.1] * 3
        if budget == 100:  # mean recovery +0.033, but one draw in 27 picks 0.19 three times: band lower bound -0.1
            return [0.19, 0.20, 0.22]
        return [0.2 + 0.1 * line(budget)] * 3

    def analyse(self, zero_shot: float = 0.2):
        ladder = {budget: [self.entry(score) for score in self.scores(budget)] for budget in BUDGETS[:-1]}
        return curve_analysis("lighting", "reliable", "full-pool", self.entry(zero_shot), ladder,
                              [self.entry(0.3), self.entry(0.3)], CEILING, 0.01, seed=0)

    def test_tables_from_one_curve(self):
        points, thresholds, fits, holdout = self.analyse()
        self.assertEqual([row["budget"] for row in points], BUDGETS)
        self.assertEqual([row["threshold"] for row in thresholds], ["N_BE", "N75", "N90"])
        self.assertEqual([row["point"] for row in thresholds], [100, 2000, 5000])
        self.assertEqual([row["status"] for row in thresholds], ["determined", "determined", "not determined within pool"])
        self.assertEqual([row["band_to"] for row in thresholds], [250, 2000, 5000])
        ci, point = fits
        self.assertEqual((ci["region_rule"], ci["rungs"], ci["region_start"], ci["region_end"]), ("ci", 4, 250, 2000))
        self.assertEqual((point["region_rule"], point["rungs"], point["region_start"]), ("point", 5, 100))
        self.assertAlmostEqual(ci["alpha"], -0.7, places=9)
        self.assertAlmostEqual(ci["beta"], 0.2, places=9)
        self.assertAlmostEqual(ci["r2"], 1.0, places=9)
        self.assertAlmostEqual(ci["N75"], N75, places=4)
        self.assertAlmostEqual(ci["N75_ci_low"], N75, places=4)  # no seed spread on the region rungs
        self.assertEqual(ci["N75_replicates_defined"], 400)
        self.assertIsNone(ci["N90"])
        self.assertAlmostEqual(ci["predicted_at_ceiling"], line(CEILING), places=9)
        self.assertIs(ci["monotone"], True)
        self.assertLess(point["r2"], 0.999)  # the N = 100 rung is off the line
        self.assertEqual(thresholds[1]["fit"], ci["N75"])
        self.assertEqual([(row["region_rule"], row["budget"]) for row in holdout],
                         [("ci", 2000), ("ci", 5000), ("point", 1000), ("point", 2000), ("point", 5000)])
        self.assertAlmostEqual(holdout[1]["error"], line(CEILING) - 1.0, places=9)

    def test_undefined_recovery_gives_a_note_and_no_other_rows(self):
        points, thresholds, fits, holdout = self.analyse(zero_shot=0.295)
        self.assertEqual(len(points), 1)
        self.assertIn("undefined", points[0]["note"])
        self.assertEqual((thresholds, fits, holdout), ([], [], []))


class MeasuredTopBootstrapTest(unittest.TestCase):
    def test_top_rung_keeps_its_measured_value(self):
        zero, ceilings = np.full(50, 0.2), [np.full(50, 0.3)]  # denominator 0.1 in every replicate
        runs = {100: [np.full(50, 0.22)], 1000: [np.full(50, 0.28)], 5000: [np.full(50, 0.33)]}
        budgets, curves = bootstrap_recovery_measured_top(zero, ceilings, runs, seed=0, min_denominator=0.01)
        self.assertEqual(budgets, [100, 1000, 5000])  # no column for the ceiling: it is no rung
        self.assertEqual(curves.shape, (50, 3))
        np.testing.assert_allclose(curves, np.tile([0.2, 0.8, 1.3], (50, 1)), atol=1e-12)  # 1.3 at the top: not 1, not clipped
        # The frozen function, given the same runs with the top rung as its ceiling budget, writes 1 there.
        _, forced = bootstrap_recovery(zero, ceilings, runs, 5000, seed=0, min_denominator=0.01)
        np.testing.assert_array_equal(forced[:, -1], np.ones(50))

    def test_top_rung_seeds_and_ceiling_seeds_are_redrawn_independently(self):
        # Redrawn ceiling means 0.3, 0.35, 0.4 and redrawn top-rung means 0.26, 0.29, 0.32 around a zero-shot of 0.2: nine
        # (numerator, denominator) pairs. Tied draws would give 0.6 every time, a forced top rung 1 every time.
        zero = np.full(4000, 0.2)
        ceilings = [np.full(4000, 0.3), np.full(4000, 0.4)]
        runs = {1000: [np.full(4000, 0.25)], 5000: [np.full(4000, 0.26), np.full(4000, 0.32)]}
        budgets, curves = bootstrap_recovery_measured_top(zero, ceilings, runs, seed=3, min_denominator=0.01)
        self.assertEqual(budgets, [1000, 5000])
        expected = sorted({round(numerator / denominator, 9)
                           for numerator in (0.06, 0.09, 0.12) for denominator in (0.1, 0.15, 0.2)})
        self.assertEqual(expected, [0.3, 0.4, 0.45, 0.6, 0.8, 0.9, 1.2])
        self.assertEqual(sorted(set(np.round(curves[:, 1], 9).tolist())), expected)
        self.assertTrue(180 < int(np.isclose(curves[:, 1], 1.2).sum()) < 320)  # one draw in 16: both low ceilings, both high tops
        self.assertTrue(180 < int(np.isclose(curves[:, 1], 0.3).sum()) < 320)
        self.assertEqual(sorted(set(np.round(curves[:, 0], 9).tolist())), [0.25, round(0.05 / 0.15, 9), 0.5])
        again = bootstrap_recovery_measured_top(zero, ceilings, runs, seed=3, min_denominator=0.01)[1]
        np.testing.assert_array_equal(curves, again)
        other = bootstrap_recovery_measured_top(zero, ceilings, runs, seed=4, min_denominator=0.01)[1]
        self.assertFalse(np.array_equal(curves, other))  # the seed is used

    def test_same_test_resample_for_every_run_and_small_denominators_are_undefined(self):
        noise = np.random.default_rng(7).normal(0, 0.01, 200)  # shared by all runs of a test set: cancels in recovery
        budgets, curves = bootstrap_recovery_measured_top(0.2 + noise, [0.3 + noise], {500: [0.25 + noise], 2000: [0.28 + noise]},
                                                          seed=0, min_denominator=0.01)
        np.testing.assert_allclose(curves, np.tile([0.5, 0.8], (200, 1)), atol=1e-9)
        # Third replicate: denominator 0.3 - 0.295 = 0.005, at or below the floor; the top rung is undefined there too.
        zero = np.array([0.2, 0.2, 0.295])
        _, curves = bootstrap_recovery_measured_top(zero, [np.full(3, 0.3)], {10: [np.full(3, 0.25)], 100: [np.full(3, 0.31)]},
                                                    seed=0, min_denominator=0.01)
        np.testing.assert_allclose(curves[:2], [[0.5, 1.1], [0.5, 1.1]], atol=1e-12)
        self.assertTrue(np.isnan(curves[2]).all())

    def test_every_rung_needs_a_budget_above_the_sentinel(self):
        with self.assertRaisesRegex(ValueError, "above 0, the sentinel budget of the ceiling"):
            bootstrap_recovery_measured_top(np.full(3, 0.2), [np.full(3, 0.3)], {0: [np.full(3, 0.25)]})


class MeasuredTopCurveTest(unittest.TestCase):
    """curve_analysis against a ceiling of another schedule: mAP(0) = 0.2, a ceiling of 0.3, three sampling seeds per
    rung on line() and the runs at N = 5000 as a measured top rung."""
    noise = np.random.default_rng(1).normal(0, 0.01, 400)  # test-resample noise shared by every run: cancels in recovery

    def entry(self, score: float) -> dict:
        return {"points": {"reliable": score}, "boots": {"reliable": score + self.noise}}

    def analyse(self, top: tuple[float, ...], ceilings: tuple[float, ...] = (0.3, 0.3), top_measured: bool = True):
        ladder = {budget: [self.entry(0.1 if budget == 10 else 0.2 + 0.1 * line(budget))] * 3 for budget in BUDGETS[:-1]}
        if top:
            ladder[CEILING] = [self.entry(score) for score in top]
        return curve_analysis("lighting", "reliable", "full-pool-e12", self.entry(0.2), ladder,
                              [self.entry(score) for score in ceilings], CEILING, 0.01, seed=0, top_measured=top_measured)

    def test_top_rung_is_a_measurement_in_the_point_curve_and_in_the_band(self):
        points, thresholds, fits, holdout = self.analyse(top=(0.295, 0.295))
        self.assertEqual([row["budget"] for row in points], BUDGETS)
        top = points[-1]
        self.assertAlmostEqual(top["recovery"], 0.95, places=9)  # (0.295 - 0.2) / (0.3 - 0.2), not 1
        self.assertAlmostEqual(top["ci_low"], 0.95, places=9)
        self.assertAlmostEqual(top["ci_high"], 0.95, places=9)
        self.assertEqual((top["seeds"], top["undefined_replicates"]), (2, 0))
        self.assertAlmostEqual(top["mAP_mean"], 0.295, places=12)  # the short-schedule runs, not the ceiling
        self.assertAlmostEqual(top["ceiling"], 0.3, places=12)
        self.assertEqual([row["point"] for row in thresholds], [100, 2000, 5000])
        self.assertEqual([row["status"] for row in thresholds], ["determined"] * 3)  # N90 is the measured top rung
        ci, point = fits
        self.assertEqual((ci["rungs"], ci["region_start"], ci["region_end"]), (6, 100, 5000))
        self.assertEqual((point["rungs"], point["region_start"], point["region_end"]), (6, 100, 5000))
        self.assertLess(ci["r2"], 1.0)  # the top rung is in the fit, 0.053 below the line through the other rungs
        self.assertEqual([(row["region_rule"], row["budget"], row["is_ceiling_rung"]) for row in holdout],
                         [(rule, budget, False) for rule in ("ci", "point") for budget in (1000, 2000, 5000)])
        self.assertAlmostEqual(holdout[2]["observed"], 0.95, places=9)
        self.assertAlmostEqual(holdout[2]["error"], line(CEILING) - 0.95, places=9)

    def test_the_same_runs_read_as_the_ceiling_would_be_one(self):
        # Without the marker the rung at N = 5000 is taken from the ceiling runs: recovery 1 and a band of 1 to 1.
        points, thresholds, _, _ = self.analyse(top=(), top_measured=False)
        self.assertEqual((points[-1]["budget"], points[-1]["recovery"], points[-1]["ci_low"], points[-1]["ci_high"]),
                         (5000, 1.0, 1.0, 1.0))
        self.assertEqual([row["status"] for row in thresholds], ["determined", "determined", "not determined within pool"])

    def test_band_follows_the_sampling_seeds_of_the_top_rung_and_recovery_is_not_clipped(self):
        # Top-rung seeds at recovery 1.1 and 1.2: a replicate redraws them to 1.1, 1.15 or 1.2.
        points, thresholds, _, _ = self.analyse(top=(0.31, 0.32))
        self.assertAlmostEqual(points[-1]["recovery"], 1.15, places=9)
        self.assertAlmostEqual(points[-1]["ci_low"], 1.1, places=9)
        self.assertAlmostEqual(points[-1]["ci_high"], 1.2, places=9)
        self.assertEqual([(row["point"], row["status"]) for row in thresholds],
                         [(100, "determined"), (2000, "determined"), (5000, "determined")])

    def test_band_redraws_the_ceiling_seeds_independently_of_the_top_rung(self):
        # Denominators 0.08, 0.10, 0.12 and top-rung numerators 0.06, 0.08, 0.10: the band reaches 0.06 / 0.12 and
        # 0.10 / 0.08 (one replicate in 16 each). Tied draws would stay between 0.75 and 0.8333.
        points, _, _, _ = self.analyse(top=(0.26, 0.30), ceilings=(0.28, 0.32))
        self.assertAlmostEqual(points[-1]["recovery"], 0.8, places=9)
        self.assertAlmostEqual(points[-1]["ci_low"], 0.5, places=9)
        self.assertAlmostEqual(points[-1]["ci_high"], 1.25, places=9)

    def test_a_failing_top_rung_gives_not_reached(self):
        _, thresholds, fits, _ = self.analyse(top=(0.27, 0.27))  # recovery 0.70 after 0.82 at N = 2000
        self.assertEqual([row["point"] for row in thresholds], [100, None, None])
        self.assertEqual([row["status"] for row in thresholds], ["determined", "not reached", "not reached"])
        self.assertEqual([row["ladder_last_failing"] for row in thresholds], [10, 5000, 5000])
        self.assertIs(fits[0]["monotone"], False)
        self.assertEqual(fits[0]["region_end"], 5000)

    def test_undefined_recovery_and_a_missing_top_rung(self):
        points, thresholds, fits, holdout = self.analyse(top=(0.295, 0.295), ceilings=(0.205, 0.205))
        self.assertEqual([row["note"] for row in points], ["undefined: ceiling gain 0.0050 <= 0.01"])
        self.assertEqual((thresholds, fits, holdout), ([], [], []))
        with self.assertRaisesRegex(ValueError, "lighting full-pool-e12: no measured runs at the top rung N = 5000"):
            self.analyse(top=())


class RevisionRunsTest(unittest.TestCase):
    def run_fields(self, domain: str, budget, epochs, seed: int = 0, stage: str = "fine-tune") -> dict:
        return {"stage": stage, "domain": domain, "budget": str(budget), "seed": seed, "epochs": epochs}

    def entry(self, domain: str, budget, epochs, seed: int = 0, test_domain: str | None = None, stage: str = "fine-tune",
              kind: str | None = None) -> dict:
        return {"run": self.run_fields(domain, budget, epochs, seed, stage), "domain": test_domain or domain, "kind": kind}

    def test_kind_is_read_from_the_run_fields(self):
        self.assertEqual(revision_kind(self.run_fields("lighting", "full", 12)), "full-e12")
        self.assertEqual(revision_kind(self.run_fields("compound", "full", 12, seed=1)), "full-e12")
        self.assertEqual(revision_kind(self.run_fields("weather", 1738, 12)), "matched-e12")
        self.assertEqual(revision_kind(self.run_fields("lighting", 1738, 3, seed=1)), "matched-e3")
        self.assertEqual(revision_kind({**self.run_fields("lighting", 1738, 12), "budget": 1738}), "matched-e12")
        self.assertEqual(MATCHED_POOL, 1738)
        # Every registered (kind, seed) is known: both seeds of an E = 12 ceiling, and seed 1 of the 3-epoch matched pool.
        self.assertEqual(CEILING_SEEDS, (0, 1))
        self.assertEqual(REVISION_SEEDS, {"full-e12": (0, 1), "matched-e12": (0, 1), "matched-e3": (1,)})
        self.assertEqual([revision_kind(self.run_fields(domain, "full", 12, seed)) for domain in TARGETS for seed in (0, 1)],
                         ["full-e12"] * 6)
        self.assertEqual([revision_kind(self.run_fields("weather", 1738, 12, seed)) for seed in (0, 1)], ["matched-e12"] * 2)
        self.assertEqual(revision_kind(self.run_fields("weather", 1738, 3, seed=1)), "matched-e3")

    def test_a_run_of_no_known_kind_stops_the_analysis(self):
        unknown = (("lighting", "full", 3),  # a further short-schedule full-pool ceiling: not part of the revision
                   ("lighting", 1000, 12),  # a ladder rung at E = 12
                   ("weather", 1738, 5),
                   ("lighting", "full", None),
                   ("compound", 1738, 12),  # the compound pool is the matched pool: its runs carry budget "full"
                   ("compound", 1738, 3),
                   ("source", "full", 12))
        for domain, budget, epochs in unknown:
            message = f"no known kind: domain {domain}, budget {budget}, epochs {epochs}, seed 0"
            with self.subTest(domain=domain, budget=budget, epochs=epochs), self.assertRaisesRegex(ValueError, message):
                revision_kind(self.run_fields(domain, budget, epochs))

    def test_a_run_with_a_seed_that_was_not_registered_is_of_no_known_kind(self):
        unknown = (("lighting", "full", 12, 7),  # would enter the 12-epoch ceiling as a third seed
                   ("compound", "full", 12, 2),
                   ("weather", 1738, 12, 2),
                   ("weather", 1738, 3, 2),  # would enter the 3-epoch matched-pool ceiling as a third seed
                   ("lighting", 1738, 3, 0),  # seed 0 of that ceiling is the campaign's run, not a run of the revision
                   ("lighting", "full", 12, None),
                   ("lighting", "full", 12, "1"))
        for domain, budget, epochs, seed in unknown:
            message = (f"no known kind: domain {domain}, budget {budget}, epochs {epochs}, seed {seed}. Known .*"
                       "E = 12 at budget full, seeds 0 and 1; at budget 1738 for lighting and weather, "
                       "E = 12 with seeds 0 and 1 or E = 3 with seed 1$")
            with self.subTest(domain=domain, budget=budget, epochs=epochs, seed=seed):
                with self.assertRaisesRegex(ValueError, message):
                    revision_kind(self.run_fields(domain, budget, epochs, seed))

    def test_runs_that_differ_in_the_epoch_count_are_two_runs(self):
        campaign = [self.entry("lighting", "full", 3), self.entry("lighting", "full", 3, test_domain="source"),
                    self.entry("lighting", 1738, 3), self.entry("lighting", 1000, 3, seed=1)]
        revision = [self.entry("lighting", "full", 12), self.entry("lighting", "full", 12, test_domain="source"),
                    self.entry("lighting", 1738, 12), self.entry("lighting", 1738, 3, seed=1),
                    # adaptation runs share stage-free fields with fine-tune runs and are no fine-tune runs
                    self.entry("lighting", 1000, None, seed=1, stage="replay"),
                    self.entry("lighting", 1000, 3, seed=1, stage="adapt")]
        self.assertEqual(duplicate_runs(campaign, revision), [])

    def test_a_run_scored_twice_is_reported_by_domain_budget_epochs_and_seed(self):
        campaign = [self.entry("lighting", 1738, 3), self.entry("compound", "full", 3), self.entry("weather", 1000, 3, seed=2)]
        self.assertEqual(duplicate_runs(campaign, [self.entry("lighting", 1738, 3)]), [("lighting", 1738, 3, 0)])
        # The same run scored on another test set in the other directory is still the same run.
        self.assertEqual(duplicate_runs(campaign, [self.entry("lighting", 1738, 3, test_domain="source")]),
                         [("lighting", 1738, 3, 0)])
        # Budget "1738" and budget "full" name the same compound images.
        self.assertEqual(duplicate_runs(campaign, [self.entry("compound", 1738, 3)]), [("compound", 1738, 3, 0)])
        # Twice on one test set inside the revision directory; its two test sets are one run scored once each.
        twice = [self.entry("weather", "full", 12, seed=1), self.entry("weather", "full", 12, seed=1)]
        self.assertEqual(duplicate_runs(campaign, twice), [("weather", 1985, 12, 1)])
        once = [self.entry("weather", "full", 12, seed=1), self.entry("weather", "full", 12, seed=1, test_domain="source")]
        self.assertEqual(duplicate_runs(campaign, once), [])
        self.assertEqual(duplicate_runs(campaign, [self.entry("lighting", 1738, 3), self.entry("weather", 1000, 3, seed=2)]),
                         [("lighting", 1738, 3, 0), ("weather", 1000, 3, 2)])

    def test_groups_hold_the_own_test_set_per_kind_in_seed_order(self):
        revision = [self.entry("lighting", "full", 12, seed=1, kind="full-e12"),
                    self.entry("lighting", "full", 12, seed=0, kind="full-e12"),
                    self.entry("lighting", "full", 12, seed=0, test_domain="source", kind="full-e12"),
                    self.entry("lighting", 1738, 3, seed=1, kind="matched-e3"),
                    self.entry("weather", 1738, 12, seed=0, kind="matched-e12")]
        groups = revision_groups("lighting", revision)
        self.assertEqual(sorted(groups), ["full-e12", "matched-e12", "matched-e3"])
        self.assertEqual([(entry["run"]["seed"], entry["domain"]) for entry in groups["full-e12"]],
                         [(0, "lighting"), (1, "lighting")])
        self.assertEqual(groups["matched-e12"], [])
        self.assertEqual([entry["run"]["seed"] for entry in groups["matched-e3"]], [1])
        self.assertEqual(revision_groups("compound", revision), {"full-e12": [], "matched-e12": [], "matched-e3": []})

    def test_e12_ceilings_exist_for_both_pools_and_compound_has_one_set(self):
        full = [self.entry("compound", "full", 12, seed=seed) for seed in (0, 1)]
        self.assertEqual(e12_ceilings("compound", 1738, {"full-e12": full, "matched-e12": [], "matched-e3": []}),
                         {"full": full, "matched": full})
        matched = [self.entry("lighting", 1738, 12, seed=seed) for seed in (0, 1)]
        self.assertEqual(e12_ceilings("lighting", 18015, {"full-e12": full, "matched-e12": matched, "matched-e3": []}),
                         {"full": full, "matched": matched})
        with self.assertRaisesRegex(ValueError, r"lighting: no 12-epoch matched-pool ceiling run \(N = 1738\)"):
            e12_ceilings("lighting", 18015, {"full-e12": full, "matched-e12": [], "matched-e3": []})
        with self.assertRaisesRegex(ValueError, r"weather: no 12-epoch full-pool ceiling run \(N = 1985\)"):
            e12_ceilings("weather", 1985, {"full-e12": [], "matched-e12": matched, "matched-e3": []})
        mixed = [self.entry("weather", "full", 12), self.entry("weather", "full", 3, seed=1)]
        with self.assertRaisesRegex(ValueError, "weather: the fine-tune runs at N = 1985 mix epoch counts 12, 3"):
            e12_ceilings("weather", 1985, {"full-e12": mixed, "matched-e12": matched, "matched-e3": []})

    def test_an_e12_ceiling_needs_both_of_its_seeds(self):
        full = [self.entry("lighting", "full", 12, seed=seed) for seed in (0, 1)]
        matched = [self.entry("lighting", 1738, 12, seed=seed) for seed in (0, 1)]
        # One seed would be a one-seed ceiling and one pair for "materially higher"; a third would move the seed mean.
        with self.assertRaisesRegex(ValueError, r"lighting: the 12-epoch full-pool ceiling \(N = 18015\) needs seeds 0 and 1, "
                                                "found 0$"):
            e12_ceilings("lighting", 18015, {"full-e12": full[:1], "matched-e12": matched, "matched-e3": []})
        with self.assertRaisesRegex(ValueError, r"lighting: the 12-epoch matched-pool ceiling \(N = 1738\) needs seeds 0 and 1, "
                                                "found 1$"):
            e12_ceilings("lighting", 18015, {"full-e12": full, "matched-e12": matched[1:], "matched-e3": []})
        third = self.entry("compound", "full", 12, seed=2)
        with self.assertRaisesRegex(ValueError, r"compound: the 12-epoch full-pool ceiling \(N = 1738\) needs seeds 0 and 1, "
                                                "found 0 1 2$"):
            e12_ceilings("compound", 1738, {"full-e12": [*full, third], "matched-e12": [], "matched-e3": []})
        require_ceiling_seeds("weather", "ceiling", [self.entry("weather", "full", 12, seed=seed) for seed in (1, 0)])
        with self.assertRaisesRegex(ValueError, "weather: the ceiling needs seeds 0 and 1, found 0 0$"):
            require_ceiling_seeds("weather", "ceiling", [self.entry("weather", "full", 12)] * 2)
        with self.assertRaisesRegex(ValueError, "weather: the ceiling needs seeds 0 and 1, found none$"):
            require_ceiling_seeds("weather", "ceiling", [])

    def test_the_e3_matched_pool_ceiling_is_never_the_full_pool_runs_of_a_larger_pool(self):
        ceilings = [self.entry("lighting", "full", 3, seed=seed) for seed in (0, 1)]
        ours, theirs = [self.entry("lighting", 1738, 3, seed=0)], [self.entry("lighting", 1738, 3, seed=1)]
        # Without the revision, as before: the campaign's runs at 1,738, or its full-pool ceilings where it has none.
        self.assertEqual(matched_e3_ceiling("lighting", 18015, ceilings, ours), ours)
        self.assertEqual(matched_e3_ceiling("lighting", 18015, ceilings, []), ceilings)
        self.assertEqual(matched_e3_ceiling("compound", 1738, ceilings, []), ceilings)
        # With it: seed 0 from the campaign and seed 1 from the revision, in that order.
        self.assertEqual(matched_e3_ceiling("lighting", 18015, ceilings, ours, theirs), ours + theirs)
        # The compound pool is the matched pool, so there its full-pool runs are the runs at 1,738.
        self.assertEqual(matched_e3_ceiling("compound", 1738, ceilings, [], []), ceilings)
        message = (r"{}: the 3-epoch matched-pool ceiling \(N = 1738; seed 0 from the campaign, seed 1 from the revision\) "
                   "needs seeds 0 and 1, found {}$")
        for domain, pool, campaign, revision, found in (("lighting", 18015, [], [], "none"),  # 18,015-image runs as N = 1738
                                                        ("lighting", 18015, [], theirs, "1"),
                                                        ("lighting", 18015, ours, [], "0"),
                                                        ("weather", 1985, ours + theirs, [self.entry("weather", 1738, 3, seed=2)],
                                                         "0 1 2")):
            with self.subTest(found=found), self.assertRaisesRegex(ValueError, message.format(domain, found)):
                matched_e3_ceiling(domain, pool, ceilings, campaign, revision)


class SharedResamplesTest(unittest.TestCase):
    """Two scored directories pair replicate t only when their bootstrap seed, resample count and test images agree."""

    def scored_set(self, directory: Path, name: str, test_domain: str, stage: str = "fine-tune", seed: int | None = 20261030,
                   resamples: int = 50, images: int = 756, rows: int | None = None) -> None:
        folder = directory / name / test_domain
        folder.mkdir(parents=True)
        metrics = {"run": {"stage": stage}, "domain": test_domain, "images": images,
                   "bootstrap": {"seed": seed, "resamples": resamples}}
        (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        np.save(folder / "bootstrap_class_ap.npy", np.zeros((resamples if rows is None else rows, len(CLASSES))))

    def campaign(self, directory: Path) -> Path:
        self.scored_set(directory, "000_source", "compound", stage="source")
        self.scored_set(directory, "000_source", "source", stage="source", images=2650)
        self.scored_set(directory, "001_fine-tune", "compound")
        return directory

    def test_signatures_per_test_domain_and_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign = self.campaign(Path(folder) / "campaign")
            self.scored_set(campaign, "002_replay", "compound", stage="replay", seed=7, resamples=40)
            self.assertEqual(resample_signatures(campaign),
                             {"compound": {(20261030, 50, 756, 50), (7, 40, 756, 40)}, "source": {(20261030, 50, 2650, 50)}})
            self.assertEqual(resample_signatures(campaign, "fine-tune"), {"compound": {(20261030, 50, 756, 50)}})
            self.assertEqual(resample_signatures(campaign, "adapt"), {})

    def test_the_same_resamples_pass_and_runs_that_are_not_read_do_not_count(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign, revision = self.campaign(Path(folder) / "campaign"), Path(folder) / "revision"
            self.scored_set(revision, "000_fine-tune", "compound")
            self.scored_set(revision, "000_fine-tune", "source", images=2650)
            self.scored_set(revision, "001_fine-tune", "lighting", seed=5, images=4869)  # a test set the campaign did not score
            self.scored_set(revision, "002_adapt", "compound", stage="adapt", seed=1)  # adapt and replay sets are not read
            require_shared_resamples(campaign, revision)

    def test_another_seed_count_image_count_or_vector_length_stops_the_analysis(self):
        cases = {"bootstrap seed": ({"seed": 1}, r"\(1, 50, 756, 50\), \(20261030, 50, 756, 50\)"),
                 "no bootstrap seed": ({"seed": None}, r"\(20261030, 50, 756, 50\), \(None, 50, 756, 50\)"),
                 "resample count": ({"resamples": 40}, r"\(20261030, 40, 756, 40\), \(20261030, 50, 756, 50\)"),
                 "test images": ({"images": 933}, r"\(20261030, 50, 756, 50\), \(20261030, 50, 933, 50\)"),
                 "rows of the vector file": ({"rows": 40}, r"\(20261030, 50, 756, 40\), \(20261030, 50, 756, 50\)")}
        for name, (fields, listed) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as folder:
                campaign, revision = self.campaign(Path(folder) / "campaign"), Path(folder) / "revision"
                self.scored_set(revision, "000_fine-tune", "source", images=2650)
                self.scored_set(revision, "000_fine-tune", "compound", **fields)
                message = (r"Test set compound: .*campaign and the fine-tune sets of .*revision do not share one "
                           r"\(bootstrap seed, resamples, test images, rows of bootstrap_class_ap.npy\): " + listed
                           + r"\. Replicate t would not be the same resampled test set in both$")
                with self.assertRaisesRegex(ValueError, message):
                    require_shared_resamples(campaign, revision)

    def test_a_campaign_directory_with_two_bootstraps_cannot_be_paired_either(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign, revision = self.campaign(Path(folder) / "campaign"), Path(folder) / "revision"
            self.scored_set(campaign, "002_control", "compound", stage="control", seed=3)
            self.scored_set(revision, "000_fine-tune", "compound")
            with self.assertRaisesRegex(ValueError, r"Test set compound: .*: \(20261030, 50, 756, 50\), \(3, 50, 756, 50\)\. "):
                require_shared_resamples(campaign, revision)


class CeilingComparisonTest(unittest.TestCase):
    noise = np.random.default_rng(2).normal(0, 0.01, 2000)  # test-resample noise shared by both schedules: cancels

    def boots(self, points: dict[int, float]) -> dict[int, np.ndarray]:
        return {seed: value + self.noise for seed, value in points.items()}

    def entry(self, seed: int, score: float | None) -> dict:
        return {"run": {"seed": seed}, "points": {"reliable": score},
                "boots": {"reliable": (0.0 if score is None else score) + self.noise}}

    def test_flags_and_the_boundary_at_exactly_delta(self):
        self.assertEqual(MATERIAL_DELTA, 0.005)
        both = lambda *values, **options: tuple(material_flags(*values, **options).values())  # noqa: E731
        self.assertEqual(list(material_flags(0.006, 0.001, 0.011)), ["materially_higher", "materially_lower"])
        self.assertEqual(both(0.006, 0.001, 0.011), (True, False))
        self.assertEqual(both(0.005, 0.001, 0.009), (True, False))  # exactly delta counts
        self.assertEqual(both(math.nextafter(0.005, 0.0), 0.001, 0.009), (False, False))
        self.assertEqual(both(0.008, 0.0, 0.016), (False, False))  # the lower bound has to be above 0
        self.assertEqual(both(0.008, -0.001, 0.017), (False, False))
        self.assertEqual(both(-0.006, -0.011, -0.001), (False, True))
        self.assertEqual(both(-0.005, -0.009, -0.001), (False, True))
        self.assertEqual(both(math.nextafter(-0.005, 0.0), -0.009, -0.001), (False, False))
        self.assertEqual(both(-0.008, -0.016, 0.0), (False, False))  # the upper bound has to be below 0
        self.assertEqual(both(0.0029, 0.0014, 0.0041), (False, False))  # significant and small: neither
        self.assertEqual(both(0.02, 0.01, 0.03, delta=0.03), (False, False))
        self.assertEqual(both(None, None, None), (None, None))
        self.assertEqual(both(0.006, None, None), (None, None))

    def test_difference_is_paired_by_seed_over_the_common_seeds(self):
        # Per-seed differences 0.01 (seed 1) and 0.03 (seed 2); seeds 0 and 5 have no partner. A replicate redraws the
        # two pairs: 0.01, 0.02 or 0.03. Redrawn apart, a would reach 0.31 against b at 0.40: -0.09.
        points_a, points_b = {0: 0.9, 1: 0.31, 2: 0.43}, {1: 0.30, 2: 0.40, 5: 0.1}
        out = paired_seed_difference(points_a, self.boots(points_a), points_b, self.boots(points_b), seed=0)
        self.assertEqual(list(out), ["paired_seeds", "difference", "ci_low", "ci_high", "replicates"])
        self.assertEqual((out["paired_seeds"], out["replicates"]), ("1 2", 2000))
        self.assertAlmostEqual(out["difference"], 0.02, places=12)
        self.assertAlmostEqual(out["ci_low"], 0.01, places=9)  # exact: the shared test resample cancels inside a pair
        self.assertAlmostEqual(out["ci_high"], 0.03, places=9)

    def test_generator_is_seeded_and_non_finite_replicates_are_left_out(self):
        points_a, points_b = {0: 0.31, 1: 0.43}, {0: 0.30, 1: 0.40}
        boots_a, boots_b = self.boots(points_a), self.boots(points_b)
        with mock.patch("analyze_campaign.np.random.default_rng", wraps=np.random.default_rng) as generator:
            paired_seed_difference(points_a, boots_a, points_b, boots_b, seed=11)
        generator.assert_called_once_with(11)
        boots_a[0] = boots_a[0].copy()
        boots_a[0][:100] = np.nan  # a replicate is lost when it draws seed 0 at least once: three in four of the first 100
        out = paired_seed_difference(points_a, boots_a, points_b, boots_b, seed=11)
        self.assertTrue(1900 < out["replicates"] < 1950)
        self.assertAlmostEqual(out["ci_low"], 0.01, places=9)

    def test_one_pair_and_no_pair(self):
        out = paired_seed_difference({0: 0.005}, {0: np.full(10, 0.005)}, {0: 0.0, 1: 0.2},
                                     {0: np.zeros(10), 1: np.full(10, 0.2)}, seed=0)
        self.assertEqual((out["paired_seeds"], out["difference"], out["ci_low"], out["ci_high"]), ("0", 0.005, 0.005, 0.005))
        self.assertEqual(material_flags(out["difference"], out["ci_low"], out["ci_high"])["materially_higher"], True)
        none = paired_seed_difference({0: 0.3}, {0: np.zeros(10)}, {1: 0.2}, {1: np.zeros(10)}, 0)
        self.assertEqual(none, {"paired_seeds": "", "difference": None, "ci_low": None, "ci_high": None, "replicates": 0})

    def test_row_materially_higher(self):
        e3 = [self.entry(0, 0.295), self.entry(1, 0.305)]
        e12 = [self.entry(0, 0.310), self.entry(1, 0.312)]  # per-seed differences 0.015 and 0.007
        row = ceiling_comparison(e3, e12, "reliable", 0.35, seed=0)
        self.assertEqual(list(row), ["mAP_c_e3", "mAP_c_e12", "seeds_e3", "seeds_e12", "paired_seeds", "difference", "ci_low",
                                     "ci_high", "replicates", "delta", "materially_higher", "materially_lower", "mAP_source",
                                     "residual_gap_e3", "residual_gap_e12"])
        self.assertAlmostEqual(row["mAP_c_e3"], 0.300, places=12)
        self.assertAlmostEqual(row["mAP_c_e12"], 0.311, places=12)
        self.assertEqual((row["seeds_e3"], row["seeds_e12"], row["paired_seeds"], row["delta"]), ("0 1", "0 1", "0 1", 0.005))
        self.assertAlmostEqual(row["difference"], 0.011, places=12)
        self.assertAlmostEqual(row["ci_low"], 0.007, places=9)
        self.assertAlmostEqual(row["ci_high"], 0.015, places=9)
        self.assertEqual((row["materially_higher"], row["materially_lower"]), (True, False))
        self.assertAlmostEqual(row["residual_gap_e3"], 0.05, places=12)
        self.assertAlmostEqual(row["residual_gap_e12"], 0.039, places=12)

    def test_row_materially_lower_and_neither(self):
        e3 = [self.entry(0, 0.330), self.entry(1, 0.340)]
        lower = ceiling_comparison(e3, [self.entry(0, 0.327), self.entry(1, 0.331)], "reliable", 0.35, seed=0)
        self.assertAlmostEqual(lower["difference"], -0.006, places=12)  # -0.003 and -0.009
        self.assertAlmostEqual(lower["ci_high"], -0.003, places=9)
        self.assertEqual((lower["materially_higher"], lower["materially_lower"]), (False, True))
        # Above delta on average, but one seed is lower: the interval reaches -0.002 and the rule does not fire.
        mixed = ceiling_comparison(e3, [self.entry(0, 0.328), self.entry(1, 0.356)], "reliable", 0.35, seed=0)
        self.assertAlmostEqual(mixed["difference"], 0.007, places=12)
        self.assertAlmostEqual(mixed["ci_low"], -0.002, places=9)
        self.assertEqual((mixed["materially_higher"], mixed["materially_lower"]), (False, False))
        # Clearly above 0 and below delta: neither.
        small = ceiling_comparison(e3, [self.entry(0, 0.333), self.entry(1, 0.343)], "reliable", 0.35, seed=0)
        self.assertAlmostEqual(small["ci_low"], 0.003, places=9)
        self.assertEqual((small["materially_higher"], small["materially_lower"]), (False, False))

    def test_seed_means_use_all_seeds_and_the_difference_the_common_ones(self):
        e3 = [self.entry(0, 0.295), self.entry(1, 0.305)]
        row = ceiling_comparison(e3, [self.entry(0, 0.310)], "reliable", None, seed=0)
        self.assertAlmostEqual(row["mAP_c_e3"], 0.300, places=12)
        self.assertAlmostEqual(row["mAP_c_e12"], 0.310, places=12)
        self.assertEqual((row["seeds_e3"], row["seeds_e12"], row["paired_seeds"]), ("0 1", "0", "0"))
        self.assertAlmostEqual(row["difference"], 0.015, places=12)  # seed 0 against seed 0, not 0.310 - 0.300
        self.assertEqual((row["mAP_source"], row["residual_gap_e3"], row["residual_gap_e12"]), (None, None, None))

    def test_runs_are_paired_by_their_seed_not_by_their_place_in_the_list(self):
        # E3 seeds 0 and 1 at 0.30 and 0.32, E12 seed 1 only at 0.33: the one pair is seed 1, +0.01. Paired by position
        # the first run of each list would meet: 0.33 - 0.30 = +0.03.
        row = ceiling_comparison([self.entry(0, 0.30), self.entry(1, 0.32)], [self.entry(1, 0.33)], "reliable", 0.35, seed=0)
        self.assertEqual((row["seeds_e3"], row["seeds_e12"], row["paired_seeds"]), ("0 1", "1", "1"))
        for name in ("difference", "ci_low", "ci_high"):
            self.assertAlmostEqual(row[name], 0.01, places=9, msg=name)
        self.assertEqual((row["materially_higher"], row["materially_lower"]), (True, False))
        # The runs of test_row_materially_higher with the 3-epoch list in the other order: still +0.015 (seed 0) and
        # +0.007 (seed 1). By position it would be 0.310 - 0.305 and 0.312 - 0.295: the same mean, from 0.005 to 0.017.
        row = ceiling_comparison([self.entry(1, 0.305), self.entry(0, 0.295)], [self.entry(0, 0.310), self.entry(1, 0.312)],
                                 "reliable", 0.35, seed=0)
        self.assertEqual((row["seeds_e3"], row["seeds_e12"], row["paired_seeds"]), ("0 1", "0 1", "0 1"))
        self.assertAlmostEqual(row["difference"], 0.011, places=12)
        self.assertAlmostEqual(row["ci_low"], 0.007, places=9)
        self.assertAlmostEqual(row["ci_high"], 0.015, places=9)

    def test_interval_is_the_95_percent_interval_of_the_replicate_differences(self):
        # One pair and test-resample noise that does not cancel: replicate t is that pair on resample t, and the 201
        # differences run from -0.0015 to 0.0385 in steps of 0.0002. The 2.5th percentile is the 6th of them, -0.0005,
        # and the 97.5th the 196th, 0.0375; a 90% interval would start at the 11th, +0.0005, and end at 0.0365.
        base = np.random.default_rng(3).normal(0.30, 0.01, 201)
        steps = np.linspace(-0.0015, 0.0385, 201)

        def pair(shift: float) -> tuple[list[dict], list[dict]]:
            return ([{"run": {"seed": 0}, "points": {"reliable": 0.30}, "boots": {"reliable": base}}],
                    [{"run": {"seed": 0}, "points": {"reliable": 0.3185 + shift}, "boots": {"reliable": base + steps + shift}}])

        out = paired_seed_difference({0: 0.3185}, {0: base + steps}, {0: 0.30}, {0: base}, seed=0)
        self.assertEqual((out["paired_seeds"], out["replicates"]), ("0", 201))
        self.assertAlmostEqual(out["difference"], 0.0185, places=12)
        self.assertAlmostEqual(out["ci_low"], -0.0005, places=9)
        self.assertAlmostEqual(out["ci_high"], 0.0375, places=9)
        # The 95% bound decides the registered flag: +0.0185 with a lower bound of -0.0005 is not materially higher,
        # and the same interval moved up by 0.001 (lower bound +0.0005) is.
        row = ceiling_comparison(*pair(0.0), "reliable", 0.35, seed=0)
        self.assertAlmostEqual(row["ci_low"], -0.0005, places=9)
        self.assertAlmostEqual(row["ci_high"], 0.0375, places=9)
        self.assertEqual((row["materially_higher"], row["materially_lower"]), (False, False))
        row = ceiling_comparison(*pair(0.001), "reliable", 0.35, seed=0)
        self.assertAlmostEqual(row["difference"], 0.0195, places=12)
        self.assertAlmostEqual(row["ci_low"], 0.0005, places=9)
        self.assertEqual((row["materially_higher"], row["materially_lower"]), (True, False))

    def test_no_row_without_a_value_and_no_two_runs_with_one_seed(self):
        e3 = [self.entry(0, 0.295), self.entry(1, 0.305)]
        self.assertIsNone(ceiling_comparison(e3, [self.entry(0, None)], "reliable", 0.35, seed=0))
        self.assertIsNone(ceiling_comparison(e3, [], "reliable", 0.35, seed=0))
        with self.assertRaisesRegex(ValueError, "share a seed"):
            ceiling_comparison(e3, [self.entry(0, 0.31), self.entry(0, 0.32)], "reliable", 0.35, seed=0)


class ScheduleGuardTest(unittest.TestCase):
    def runs(self, *epochs) -> list[dict]:
        return [{"run": {"epochs": value} if value else {}} for value in epochs]

    def test_one_epoch_count_per_rung_passes(self):
        require_one_schedule("weather", {1000: self.runs(3, 3, 3), 1985: self.runs(3, 3)})
        require_one_schedule("weather", {1000: self.runs(3, 3), 1985: self.runs(12, 12)})  # schedules differ between rungs only
        require_one_schedule("weather", {1000: self.runs(None, None)})  # scored outputs without the field

    def test_mixed_epoch_counts_in_one_rung_stop_the_analysis(self):
        with self.assertRaisesRegex(ValueError, "weather: the fine-tune runs at N = 1985 mix epoch counts 12, 3"):
            require_one_schedule("weather", {1000: self.runs(3, 3, 3), 1985: self.runs(3, 3, 12, 12)})
        with self.assertRaisesRegex(ValueError, "N = 1738 mix epoch counts 3, None"):
            require_one_schedule("lighting", {1738: self.runs(3, None)})


class WriteTablesTest(unittest.TestCase):
    def test_an_empty_table_removes_the_file_of_an_earlier_run(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_tables(directory, {"fits": [{"domain": "weather", "rungs": 3}], "fit_holdout": [{"budget": 2000}]})
            self.assertEqual(sorted(path.name for path in directory.iterdir()), ["fit_holdout.csv", "fits.csv"])
            write_tables(directory, {"fits": [{"domain": "compound", "rungs": 2}], "fit_holdout": [], "controls": []})
            self.assertEqual([path.name for path in directory.iterdir()], ["fits.csv"])
            self.assertEqual((directory / "fits.csv").read_text(encoding="utf-8").splitlines(), ["domain,rungs", "compound,2"])


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
class PlotTest(unittest.TestCase):
    def test_only_a_measured_top_rung_is_marked(self):
        import matplotlib.pyplot as plt
        points = [{"domain": "lighting", "class": key, "analysis": analysis, "budget": budget, "recovery": value,
                   "ci_low": value - 0.05, "ci_high": value + 0.05}
                  for analysis, top in (("full-pool", 1.0), ("full-pool-e12", 0.9)) for key in ("reliable", "car")
                  for budget, value in ((100, 0.2), (1000, 0.6), (5000, top))]
        points.append({"domain": "lighting", "class": "pedestrian", "analysis": "full-pool-e12", "budget": None, "recovery": None,
                       "note": "undefined: ceiling gain 0.0050 <= 0.01"})
        axes, subplots = [], plt.subplots

        def recording(*args, **options):
            figure, axis = subplots(*args, **options)
            axes.append(axis)
            return figure, axis

        with tempfile.TemporaryDirectory() as folder, mock.patch("matplotlib.pyplot.subplots", side_effect=recording):
            written = plot(points, Path(folder))
            written += plot(points, Path(folder), analysis="full-pool-e12", suffix="_e12", ceiling="12-epoch full-pool ceiling",
                            top_measured=True)
            self.assertEqual(written, [Path(folder) / "recovery_lighting.png", Path(folder) / "recovery_lighting_e12.png"])
            self.assertEqual(sorted(path.name for path in Path(folder).iterdir()),
                             ["recovery_lighting.png", "recovery_lighting_e12.png"])
        e3, e12 = axes
        markers = lambda axis: [line.get_marker() for line in axis.lines]  # noqa: E731
        labels = lambda axis: [text.get_text() for text in axis.get_legend().get_texts()]  # noqa: E731
        dashed_at_one = lambda axis: [line for line in axis.lines  # noqa: E731
                                      if line.get_linestyle() == "--" and list(line.get_ydata()) == [1.0, 1.0]]
        self.assertEqual((markers(e3).count("o"), markers(e3).count("D")), (2, 0))  # the ceiling rung is a point like any other
        self.assertEqual((markers(e12).count("o"), markers(e12).count("D")), (2, 3))  # one per curve and the legend entry
        self.assertEqual((len(e3.lines), len(e12.lines)), (2 + 3, 2 + 3 + 4))  # curves, diamonds, level lines
        diamonds = [line for line in e12.lines if line.get_marker() == "D"]
        self.assertEqual([(list(line.get_xdata()), list(line.get_ydata())) for line in diamonds],
                         [([5000], [0.9]), ([5000], [0.9]), ([], [])])  # at the measured recovery, not at 1
        self.assertEqual((len(dashed_at_one(e3)), len(dashed_at_one(e12))), (0, 1))
        self.assertEqual(e3.get_title(), "lighting: recovery against the full-pool ceiling")
        self.assertEqual(e12.get_title(), "lighting: recovery against the 12-epoch full-pool ceiling")
        self.assertEqual(labels(e3), ["reliable", "car"])
        self.assertEqual(labels(e12), ["reliable", "car", "N = pool: 3-epoch runs, measured",
                                       "12-epoch full-pool ceiling (recovery 1)"])

    def test_no_e12_figure_without_a_defined_curve(self):
        undefined = [{"domain": "weather", "class": "reliable", "analysis": "full-pool-e12", "budget": None, "recovery": None,
                      "note": "undefined: ceiling gain 0.0080 <= 0.01"}]
        with tempfile.TemporaryDirectory() as folder:
            written = plot(undefined, Path(folder), analysis="full-pool-e12", suffix="_e12",
                           ceiling="12-epoch full-pool ceiling", top_measured=True)
            self.assertEqual(written, [])  # main() removes the figure of an earlier run by this
            self.assertEqual(list(Path(folder).iterdir()), [])


class MainTest(unittest.TestCase):
    """main() on a synthetic scored tree: the wiring between the tables that the pure functions cannot see.

    Every class of a run scores a level plus the offset of the class, and every bootstrap vector is that score plus
    test-resample noise shared by all runs scored on the same test domain, so the noise cancels inside a domain.
    Per shift, mAP(0) is `zero`, the two full-pool ceilings are zero + gain -+ 0.005, the matched-pool run is
    zero + 0.95 gain, and a ladder rung has recovery line(N) at both sampling seeds.
    """
    offset = [0.001 * index for index in range(len(CLASSES))]
    source_level = 0.35
    zero = {"compound": 0.22, "lighting": 0.24, "weather": 0.32}
    gain = {"compound": 0.04, "lighting": 0.05, "weather": 0.014}
    ladder = {"compound": (10, 100, 250, 500, 1000), "lighting": (10, 100, 250, 500, 1000, 2000),
              "weather": (10, 100, 250, 500, 1000)}
    # The revision tree (build_revision): gain of the 12-epoch runs over mAP(0) at the full pool and at the matched pool.
    # Its second 3-epoch matched-pool seed is at 1.05 of the gain, so that the two seeds average the gain itself.
    e12_full = {"compound": 0.042, "lighting": 0.0625, "weather": 0.008}
    e12_matched = {"lighting": 0.04, "weather": 0.02}
    e3_files = ["controls.csv", "fit_holdout.csv", "fits.csv", "forgetting.csv", "recovery_denominator.csv",
                "recovery_points.csv", "rq1_drop.csv", "summary.json", "thresholds.csv"]
    curve_tables = ("recovery_points", "thresholds", "fits", "fit_holdout", "recovery_denominator")
    matched_seeds = {"lighting": (0,), "weather": (0,)}  # the campaign's runs at 1,738; the compound pool is that pool
    bootstrap_seed = 20261030

    def write(self, scored: Path, noise: dict, stage: str, run_domain: str, budget, seed: int, test_domain: str, level: float,
              epochs: int | None = 3, updates: int | None = None, named_by_epochs: bool = True,
              bootstrap_seed: int | None = None) -> None:
        name = f"{len(list(scored.glob('*'))):03d}_{stage}_{run_domain}_N{budget}_seed{seed}"
        folder = scored / (f"{name}_e{epochs}" if named_by_epochs else name) / test_domain
        folder.mkdir(parents=True)
        per_class = [level + value for value in self.offset]
        run = {"stage": stage, "domain": run_domain, "budget": str(budget), "seed": seed, "epochs": epochs}
        metrics = {"run": run if updates is None else {**run, "total_updates": updates},
                   "domain": test_domain, "images": EXPECTED_DOMAINS[test_domain]["final_test"],
                   "AP50_95": float(np.mean(per_class)),
                   "reliable_AP50_95": float(np.mean([per_class[CLASSES.index(name)] for name in RELIABLE_CLASSES])),
                   "per_class": [{"class": name, "AP50_95": value} for name, value in zip(CLASSES, per_class)],
                   "bootstrap": {"resamples": len(noise[test_domain]), "class_order": list(CLASSES),
                                 "seed": self.bootstrap_seed if bootstrap_seed is None else bootstrap_seed}}
        (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        np.save(folder / "bootstrap_class_ap.npy", np.asarray(per_class) + noise[test_domain])

    def resample_noise(self, replicates: int) -> dict[str, np.ndarray]:
        """Test-resample noise per test domain; the same for every scored tree, as two plans with one bootstrap seed."""
        rng = np.random.default_rng(42)
        return {domain: rng.normal(0, 0.004, (replicates, len(CLASSES))) for domain in ("source", *TARGETS)}

    def build(self, scored: Path, ladder: dict[str, tuple[int, ...]] | None = None, extras: bool = True,
              replicates: int = 80, epochs: int = 3, matched: dict[str, tuple[int, ...]] | None = None) -> None:
        """A campaign tree. `epochs`: the schedule of its fine-tune runs; `matched`: the seeds of its runs at 1,738 per shift."""
        noise = self.resample_noise(replicates)
        scored.mkdir(parents=True)
        for domain in ("source", *TARGETS):
            self.write(scored, noise, "source", "source", "full", 0, domain, self.zero.get(domain, self.source_level), epochs=12)
        for domain in TARGETS:
            zero, gain = self.zero[domain], self.gain[domain]
            for seed, shift in enumerate((-0.005, 0.005)):
                self.write(scored, noise, "fine-tune", domain, "full", seed, domain, zero + gain + shift, epochs=epochs)
            for budget in (ladder or self.ladder)[domain]:
                for seed in (0, 1):
                    self.write(scored, noise, "fine-tune", domain, budget, seed, domain, zero + gain * line(budget),
                               epochs=epochs)
            for seed in (self.matched_seeds if matched is None else matched).get(domain, ()):
                self.write(scored, noise, "fine-tune", domain, MATCHED_POOL, seed, domain, zero + 0.95 * gain, epochs=epochs)
        if extras:
            self.write(scored, noise, "control", "lighting", 100, 0, "lighting", self.zero["lighting"] - 0.05, epochs=epochs)
            self.write(scored, noise, "fine-tune", "lighting", 100, 0, "source", 0.33, epochs=epochs)

    def revision_runs(self) -> list[tuple]:
        """(run domain, budget, seed, epochs, level on the own test set) of the fine-tune runs of the revision.

        Lighting: E12 above E3 at the full pool, below at the matched pool. Weather: the E12 full-pool gain (0.008) is under
        the 0.01 floor. Compound: E12 is E3 + 0.002 at either seed. The matched runs exist for lighting and weather only.
        """
        runs = []
        for domain in TARGETS:
            zero = self.zero[domain]
            shifts = (-0.005, 0.005) if domain == "compound" else (0.0, 0.0)
            runs += [(domain, "full", seed, 12, zero + self.e12_full[domain] + shift) for seed, shift in enumerate(shifts)]
            if domain in self.e12_matched:
                runs += [(domain, MATCHED_POOL, seed, 12, zero + self.e12_matched[domain]) for seed in (0, 1)]
                runs.append((domain, MATCHED_POOL, 1, 3, zero + 1.05 * self.gain[domain]))
        return runs

    def build_revision(self, scored: Path, runs: list[tuple] | None = None, replicates: int = 80,
                       bootstrap_seed: int | None = None) -> None:
        """A revision tree: every fine-tune run scored on its own and on the source test set, in folders that do not name
        the epoch count, next to adaptation runs with the stage-free fields of campaign runs and a score of 0.9."""
        noise = self.resample_noise(replicates)
        scored.mkdir(parents=True)
        for domain, budget, seed, epochs, level in self.revision_runs() if runs is None else runs:
            for test_domain, value in ((domain, level), ("source", 0.31 if epochs == 12 else 0.33)):
                self.write(scored, noise, "fine-tune", domain, budget, seed, test_domain, value, epochs=epochs,
                           named_by_epochs=False, bootstrap_seed=bootstrap_seed)
        for stage, budget, seed, updates in (("adapt", 1000, 0, 3000), ("replay", 1000, 0, 3000), ("replay", 100, 1, 2000)):
            for test_domain in ("lighting", "source"):
                self.write(scored, noise, stage, "lighting", budget, seed, test_domain, 0.9, epochs=None, updates=updates,
                           named_by_epochs=False, bootstrap_seed=bootstrap_seed)

    def run_main(self, scored: Path, output: Path, *options: str, plots: bool = False) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--scored", str(scored), "--output-dir", str(output), *([] if plots else ["--no-plots"]), *options])

    def table(self, path: Path) -> list[dict]:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_tables_and_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            scored, output = Path(folder) / "scored", Path(folder) / "analysis"
            self.build(scored)
            with (mock.patch("analyze_campaign.denominator_row", wraps=denominator_row) as denominator_spy,
                  mock.patch("analyze_campaign.curve_analysis", wraps=curve_analysis) as curve_spy):
                self.run_main(scored, output, "--seed", "5")
            self.assertEqual(sorted(path.name for path in output.iterdir()),
                             ["controls.csv", "fit_holdout.csv", "fits.csv", "forgetting.csv", "recovery_denominator.csv",
                              "recovery_points.csv", "rq1_drop.csv", "summary.json", "thresholds.csv"])
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            entries = load(scored)
            denominators = {(row["domain"], row["class"], row["analysis"]): row
                            for row in self.table(output / "recovery_denominator.csv")}
            thresholds, fits = self.table(output / "thresholds.csv"), self.table(output / "fits.csv")
            controls, forgetting = self.table(output / "controls.csv"), self.table(output / "forgetting.csv")
        self.assertEqual(len(denominators), 3 * 6 * 2)  # shift x class key x analysis
        # The recovery bootstrap keeps --seed itself; the denominator table draws from a generator of its own, seeded
        # with --seed plus the fixed offset.
        self.assertEqual(DENOMINATOR_SEED_OFFSET, 20261121)
        for spy, function, seed in ((curve_spy, curve_analysis, 5), (denominator_spy, denominator_row, 5 + 20261121)):
            signature = inspect.signature(function)
            self.assertEqual(len(spy.call_args_list), 3 * 6 * 2)
            self.assertEqual({signature.bind(*call.args, **call.kwargs).arguments["seed"] for call in spy.call_args_list}, {seed})

        # The matched-pool row takes its own ceiling run: one seed at 0.95 of the gain, and noise that cancels exactly.
        matched = denominators[("lighting", "reliable", "matched-pool")]
        self.assertEqual(matched["ceiling_seeds"], "1")
        for name in ("denominator", "ci_low", "ci_high"):
            self.assertAlmostEqual(float(matched[name]), 0.0475, places=9, msg=name)
        full = denominators[("lighting", "reliable", "full-pool")]
        self.assertEqual(full["ceiling_seeds"], "2")
        self.assertAlmostEqual(float(full["denominator"]), 0.05, places=9)
        self.assertAlmostEqual(float(full["ci_low"]), 0.045, places=9)  # both redrawn seeds are the low ceiling
        self.assertAlmostEqual(float(full["ci_high"]), 0.055, places=9)
        # The compound pool is the matched pool, so its matched-pool row keeps both full-pool ceilings.
        self.assertEqual(denominators[("compound", "reliable", "matched-pool")]["ceiling_seeds"], "2")

        # Weather: 0.014 is 1.4 floors, and a replicate that draws the low ceiling twice (0.009) is at or below the floor.
        weather = denominators[("weather", "reliable", "full-pool")]
        self.assertAlmostEqual(float(weather["ratio_to_floor"]), 1.4, places=6)
        self.assertEqual(weather["defined"], "True")
        zero = next(e for e in entries if e["run"]["stage"] == "source" and e["domain"] == "weather")
        ceilings = [e for e in entries if e["run"]["stage"] == "fine-tune" and e["run"]["domain"] == "weather"
                    and e["run"]["budget"] == "full"]
        expected = denominator_row(zero["points"]["reliable"], [c["points"]["reliable"] for c in ceilings],
                                   zero["boots"]["reliable"], [c["boots"]["reliable"] for c in ceilings], 0.01,
                                   5 + DENOMINATOR_SEED_OFFSET)
        self.assertEqual(int(weather["replicates_at_or_below_floor"]), expected["replicates_at_or_below_floor"])
        self.assertTrue(8 < expected["replicates_at_or_below_floor"] < 32)  # about a quarter of 80
        self.assertEqual((weather["replicates"], weather["ceiling_seeds"]), ("80", "2"))

        interaction = summary["interaction"]
        boots = {e["domain"]: e["boots"]["reliable"] for e in entries if e["run"]["stage"] == "source"}
        replicate = boots["lighting"] + boots["weather"] - boots["compound"] - boots["source"]
        se = float(np.std(replicate, ddof=1))
        self.assertGreater(se, 0.001)
        self.assertAlmostEqual(interaction["estimate"], -0.01, places=9)  # 0.13 - 0.11 - 0.03
        self.assertAlmostEqual(interaction["se"], se, places=12)
        self.assertAlmostEqual(interaction["mde_80"], 2.8015852 * se, places=8)
        self.assertAlmostEqual(interaction["ci_half_width"], (interaction["ci_high"] - interaction["ci_low"]) / 2, places=12)
        for domain, drop in (("compound", 0.13), ("lighting", 0.11), ("weather", 0.03)):
            self.assertAlmostEqual(interaction[f"drop_{domain}"], drop, places=9)
        self.assertAlmostEqual(interaction["ci_half_width_over_weather_drop"], interaction["ci_half_width"] / 0.03, places=6)
        self.assertAlmostEqual(interaction["mde_80_over_weather_drop"], 2.8015852 * se / 0.03, places=6)

        self.assertEqual(len(fits), 3 * 6 * 2 * 2)  # ... x region rule
        self.assertEqual([(row["domain"], row["class"], row["analysis"], row["region_rule"]) for row in summary["fits"]],
                         [(domain, "reliable", "full-pool", rule) for domain in TARGETS for rule in ("ci", "point")])
        lighting = next(row for row in summary["fits"] if row["domain"] == "lighting" and row["region_rule"] == "ci")
        self.assertEqual((lighting["rungs"], lighting["region_start"], lighting["region_end"]), (5, 100, 2000))
        self.assertAlmostEqual(lighting["alpha"], -0.7, places=6)
        self.assertAlmostEqual(lighting["beta"], 0.2, places=6)
        self.assertAlmostEqual(lighting["N75"], N75, places=2)
        self.assertEqual(len(thresholds), 3 * 6 * 2 * 3)
        self.assertEqual({row["threshold"] for row in thresholds}, {"N_BE", "N75", "N90"})
        self.assertEqual([(row["domain"], row["threshold"]) for row in summary["thresholds"]],
                         [(domain, label) for domain in TARGETS for label in ("N_BE", "N75", "N90")])
        self.assertEqual([(row["domain"], row["class"], row["analysis"]) for row in summary["recovery_denominator"]],
                         [(domain, "reliable", analysis) for domain in TARGETS for analysis in ("full-pool", "matched-pool")])
        self.assertEqual(len(controls), 1)
        self.assertAlmostEqual(float(controls[0]["control_reliable"]) - float(controls[0]["zero_shot"]), -0.05, places=9)
        self.assertEqual(len(forgetting), 1)
        self.assertAlmostEqual(float(forgetting[0]["change_vs_source_model"]), -0.02, places=9)

    def test_a_rerun_leaves_no_table_of_the_earlier_run(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "analysis"
            self.build(Path(folder) / "scored")
            self.run_main(Path(folder) / "scored", output)
            self.assertTrue((output / "fit_holdout.csv").exists() and (output / "controls.csv").exists())
            # A ladder with two rungs above 0 has no region of four rungs: no fit, no held-out check, and here no control.
            self.build(Path(folder) / "short", ladder={domain: (10, 100, 1000) for domain in TARGETS}, extras=False)
            self.run_main(Path(folder) / "short", output)
            self.assertEqual(sorted(path.name for path in output.iterdir()),
                             ["fits.csv", "recovery_denominator.csv", "recovery_points.csv", "rq1_drop.csv", "summary.json",
                              "thresholds.csv"])
            fits = self.table(output / "fits.csv")
            self.assertEqual({(row["rungs"], row["alpha"]) for row in fits}, {("2", "")})
            self.assertEqual(json.loads((output / "summary.json").read_text(encoding="utf-8"))["fits"][0]["rungs"], 2)

    def test_summary_is_valid_json_when_the_interaction_has_no_bootstrap_spread(self):
        with tempfile.TemporaryDirectory() as folder, warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # one replicate: the SD with ddof=1 is undefined
            self.build(Path(folder) / "scored", replicates=1)
            self.run_main(Path(folder) / "scored", Path(folder) / "analysis")
            text = (Path(folder) / "analysis" / "summary.json").read_text(encoding="utf-8")
        self.assertNotIn("NaN", text)
        interaction = json.loads(text)["interaction"]
        self.assertEqual((interaction["se"], interaction["mde_80"], interaction["mde_80_over_weather_drop"]), (None, None, None))
        self.assertAlmostEqual(interaction["estimate"], -0.01, places=9)
        self.assertAlmostEqual(interaction["drop_weather"], 0.03, places=9)

    def test_ceilings_of_two_schedules_are_not_pooled(self):
        with tempfile.TemporaryDirectory() as folder:
            scored, output = Path(folder) / "scored", Path(folder) / "analysis"
            self.build(scored)
            noise = {"weather": np.zeros((80, len(CLASSES)))}
            for seed in (0, 1):
                self.write(scored, noise, "fine-tune", "weather", "full", seed, "weather", 0.354, epochs=12)
            with self.assertRaisesRegex(ValueError, "weather: the fine-tune runs at N = 1985 mix epoch counts 12, 3"):
                self.run_main(scored, output)
            self.assertFalse(output.exists())

    def test_tables_against_the_e12_ceilings(self):
        with tempfile.TemporaryDirectory() as folder:
            scored, revision, output = (Path(folder) / name for name in ("scored", "revision", "analysis"))
            self.build(scored)
            self.build_revision(revision)
            with (mock.patch("analyze_campaign.denominator_row", wraps=denominator_row) as denominator_spy,
                  mock.patch("analyze_campaign.curve_analysis", wraps=curve_analysis) as curve_spy,
                  mock.patch("analyze_campaign.ceiling_comparison", wraps=ceiling_comparison) as comparison_spy):
                self.run_main(scored, output, "--seed", "5", "--revision-scored", str(revision))
            files = sorted(path.name for path in output.iterdir())
            tables = {name: self.table(output / f"{name}.csv")
                      for name in (*self.curve_tables, "ceiling_e12_vs_e3", "forgetting_e12", "forgetting")}
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            campaign_entries, revision_entries = load(scored), load(revision)
        self.assertEqual(files, sorted([*self.e3_files, "ceiling_e12_vs_e3.csv", "forgetting_e12.csv"]))
        self.assertEqual(len(revision_entries), 12 * 2 + 3 * 2)  # fine-tune and adaptation runs, each on two test sets
        reliable = lambda name: [row for row in tables[name] if row["class"] == "reliable"]  # noqa: E731

        # Every draw of the revision outputs comes from a generator of its own; the 3-epoch analyses keep --seed.
        self.assertEqual((E12_SEED_OFFSET, PAIRED_SEED_OFFSET), (20261123, 20261124))
        bound = lambda spy, function: [inspect.signature(function).bind(*call.args, **call.kwargs).arguments  # noqa: E731
                                       for call in spy.call_args_list]
        curve_calls = bound(curve_spy, curve_analysis)
        self.assertEqual(len(curve_calls), 3 * 6 * 4)  # shift x class key x analysis
        self.assertEqual({(call["analysis"], call["top_measured"], call["seed"]) for call in curve_calls},
                         {("full-pool", False, 5), ("matched-pool", False, 5),
                          ("full-pool-e12", True, 5 + 20261123), ("matched-pool-e12", True, 5 + 20261123)})
        self.assertEqual(Counter(call["seed"] for call in bound(denominator_spy, denominator_row)),
                         {5 + 20261121: 36, 5 + 20261121 + 20261123: 36})
        self.assertEqual([call["seed"] for call in bound(comparison_spy, ceiling_comparison)], [5 + 20261124] * 36)

        # Lighting, full pool: the E12 ceiling is mAP(0) + 0.0625, so the 3-epoch full-pool runs (+0.05 -+ 0.005) are a
        # measured top rung at 0.8, and every lower rung is 0.8 of its 3-epoch recovery.
        curve = {(row["domain"], row["analysis"], row["budget"]): row for row in reliable("recovery_points")}
        top = curve[("lighting", "full-pool-e12", "18015")]
        self.assertAlmostEqual(float(top["recovery"]), 0.8, places=9)
        self.assertAlmostEqual(float(top["ci_low"]), 0.72, places=9)  # both redrawn top-rung seeds are the low run
        self.assertAlmostEqual(float(top["ci_high"]), 0.88, places=9)
        self.assertEqual((top["seeds"], top["undefined_replicates"]), ("2", "0"))
        self.assertAlmostEqual(float(top["mAP_mean"]) - float(top["zero_shot"]), 0.05, places=9)
        self.assertAlmostEqual(float(top["ceiling"]) - float(top["zero_shot"]), 0.0625, places=9)
        self.assertAlmostEqual(float(curve[("lighting", "full-pool-e12", "1000")]["recovery"]), 0.8 * line(1000), places=9)
        self.assertEqual(curve[("lighting", "full-pool", "18015")]["recovery"], "1.0")  # the 3-epoch analysis: by construction
        # Lighting, matched pool: 3-epoch runs at +0.0475 (campaign, seed 0) and +0.0525 (revision, seed 1) against an
        # E12 ceiling of +0.04: a top rung at 1.25, above 1 and not clipped.
        matched = curve[("lighting", "matched-pool-e12", "1738")]
        self.assertAlmostEqual(float(matched["recovery"]), 1.25, places=9)
        self.assertAlmostEqual(float(matched["ci_low"]), 1.1875, places=9)
        self.assertAlmostEqual(float(matched["ci_high"]), 1.3125, places=9)
        self.assertEqual(matched["seeds"], "2")
        e3_matched = curve[("lighting", "matched-pool", "1738")]
        self.assertEqual((e3_matched["recovery"], e3_matched["seeds"]), ("1.0", "2"))  # the 3-epoch ceiling has two seeds now
        self.assertAlmostEqual(float(e3_matched["ceiling"]) - float(e3_matched["zero_shot"]), 0.05, places=9)
        # The ladders: a matched-pool ladder ends at its measured top rung, so the lighting rung at N = 2000 stays out
        # of it (against the 12-epoch ceiling at 1,738 it would read 1.025), and the runs at 1,738 are no rung of a
        # full-pool ladder.
        budgets = lambda *key: [int(budget) for domain, analysis, budget in curve if (domain, analysis) == key]  # noqa: E731
        self.assertEqual(budgets("lighting", "matched-pool-e12"), [10, 100, 250, 500, 1000, 1738])
        self.assertEqual(budgets("lighting", "matched-pool"), [10, 100, 250, 500, 1000, 1738])
        self.assertEqual(budgets("lighting", "full-pool-e12"), [10, 100, 250, 500, 1000, 2000, 18015])
        self.assertEqual(budgets("weather", "matched-pool-e12"), [10, 100, 250, 500, 1000, 1738])
        # Per class key: compound 4 curves of 6 rungs; weather 3 of 6 and the undefined row; lighting 2 of 7 and 2 of 6.
        self.assertEqual(len(tables["recovery_points"]), 6 * (4 * 6 + (3 * 6 + 1) + (2 * 7 + 2 * 6)))
        # Compound: both the top-rung seeds (+0.035, +0.045) and the ceiling seeds (+0.037, +0.047) are redrawn.
        compound = curve[("compound", "full-pool-e12", "1738")]
        self.assertAlmostEqual(float(compound["recovery"]), 0.04 / 0.042, places=9)
        own = lambda entries, **fields: [entry["boots"]["reliable"] for entry in entries  # noqa: E731
                                         if entry["domain"] == "compound"
                                         and all(entry["run"][name] == value for name, value in fields.items())]
        runs = {budget: own(campaign_entries, stage="fine-tune", budget=str(budget)) for budget in self.ladder["compound"]}
        runs[1738] = own(campaign_entries, stage="fine-tune", budget="full")
        _, curves = bootstrap_recovery_measured_top(own(campaign_entries, stage="source")[0],
                                                    own(revision_entries, stage="fine-tune"), runs,
                                                    seed=5 + E12_SEED_OFFSET, min_denominator=0.01)
        low, high = (float(value) for value in np.percentile(curves[:, -1], [2.5, 97.5]))
        self.assertAlmostEqual(float(compound["ci_low"]), low, places=12)
        self.assertAlmostEqual(float(compound["ci_high"]), high, places=12)
        self.assertTrue(0.74 < low < 0.86 and 1.07 < high < 1.22, (low, high))  # from 0.035 / 0.047 up to 0.045 / 0.037
        compound_budgets = [str(budget) for budget in (*self.ladder["compound"], 1738)]
        self.assertEqual([key[2] for key in curve if key[:2] == ("compound", "full-pool-e12")], compound_budgets)
        for budget in compound_budgets:  # the compound pool is the matched pool: one curve under two names
            full, same = curve[("compound", "full-pool-e12", budget)], curve[("compound", "matched-pool-e12", budget)]
            self.assertEqual({**full, "analysis": ""}, {**same, "analysis": ""})
        # Weather, full pool: an E12 gain of 0.008 does not exceed the floor, so recovery is undefined for every class key.
        undefined = [row for row in tables["recovery_points"] if (row["domain"], row["analysis"]) == ("weather", "full-pool-e12")]
        self.assertEqual([(row["budget"], row["recovery"], row["note"]) for row in undefined],
                         [("", "", "undefined: ceiling gain 0.0080 <= 0.01")] * 6)

        status = {(row["domain"], row["analysis"], row["threshold"]): (row["point"], row["status"])
                  for row in reliable("thresholds")}
        expected = {("lighting", "full-pool-e12"): [("100", "determined"), ("18015", "determined"), ("", "not reached")],
                    ("lighting", "full-pool"): [("100", "determined"), ("2000", "determined"),
                                                ("18015", "not determined within pool")],
                    ("lighting", "matched-pool-e12"): [("100", "determined"), ("1000", "determined"), ("1738", "determined")],
                    ("lighting", "matched-pool"): [("100", "determined")] + [("1738", "not determined within pool")] * 2,
                    ("weather", "matched-pool-e12"): [("100", "determined"), ("", "not reached"), ("", "not reached")],
                    ("compound", "full-pool-e12"): [("100", "determined"), ("1738", "determined"), ("1738", "determined")],
                    ("compound", "matched-pool-e12"): [("100", "determined"), ("1738", "determined"), ("1738", "determined")]}
        for (domain, analysis), rows in expected.items():
            self.assertEqual([status[(domain, analysis, label)] for label in ("N_BE", "N75", "N90")], rows, (domain, analysis))
        self.assertNotIn(("weather", "full-pool-e12", "N_BE"), status)
        self.assertEqual({row["status"] for row in tables["thresholds"] if row["analysis"].endswith("-e12")},
                         {"determined", "not reached"})
        # The replicate counts of the band travel with the "-e12" rows; the 3-epoch rows keep their columns empty there.
        self.assertEqual(list(tables["thresholds"][0])[-3:],
                         ["fit_inside_ladder_bracket", "band_replicates", "band_undefined_replicates"])
        counts = {(row["analysis"], row["band_replicates"], row["band_undefined_replicates"]) for row in tables["thresholds"]}
        self.assertEqual(counts, {("full-pool", "", ""), ("matched-pool", "", ""),
                                  ("full-pool-e12", "80", "0"), ("matched-pool-e12", "80", "0")})

        fit = {(row["domain"], row["analysis"], row["region_rule"]): row for row in reliable("fits")}
        region = lambda *key: tuple(fit[key][name] for name in ("rungs", "region_start", "region_end"))  # noqa: E731
        self.assertEqual(region("lighting", "full-pool-e12", "ci"), ("6", "100", "18015"))  # up to the measured top rung
        self.assertEqual(region("lighting", "full-pool", "ci"), ("5", "100", "2000"))
        self.assertEqual(region("lighting", "matched-pool-e12", "ci"), ("5", "100", "1738"))
        self.assertEqual(region("compound", "full-pool-e12", "ci"), ("5", "100", "1738"))
        self.assertEqual(region("compound", "full-pool", "ci"), ("4", "100", "1000"))
        self.assertNotIn(("weather", "full-pool-e12", "ci"), fit)
        holdout = [row for row in reliable("fit_holdout") if (row["domain"], row["region_rule"]) == ("lighting", "ci")]
        self.assertEqual([(row["budget"], row["is_ceiling_rung"]) for row in holdout if row["analysis"] == "full-pool-e12"],
                         [("1000", "False"), ("2000", "False"), ("18015", "False")])
        self.assertEqual([(row["budget"], row["is_ceiling_rung"]) for row in holdout if row["analysis"] == "full-pool"],
                         [("1000", "False"), ("2000", "False"), ("18015", "True")])
        self.assertEqual({row["is_ceiling_rung"] for row in tables["fit_holdout"] if row["analysis"].endswith("-e12")}, {"False"})

        denominators = {(row["domain"], row["analysis"]): row for row in reliable("recovery_denominator")}
        self.assertEqual(len(tables["recovery_denominator"]), 3 * 6 * 4)
        for key, (value, ci_low, ci_high, ratio, defined, at_floor) in {
                ("lighting", "full-pool-e12"): (0.0625, 0.0625, 0.0625, 6.25, "True", "0"),
                ("lighting", "matched-pool-e12"): (0.04, 0.04, 0.04, 4.0, "True", "0"),
                ("lighting", "matched-pool"): (0.05, 0.0475, 0.0525, 5.0, "True", "0"),
                ("weather", "full-pool-e12"): (0.008, 0.008, 0.008, 0.8, "False", "80"),
                ("weather", "matched-pool-e12"): (0.02, 0.02, 0.02, 2.0, "True", "0"),
                ("compound", "full-pool-e12"): (0.042, 0.037, 0.047, 4.2, "True", "0"),
                ("compound", "matched-pool-e12"): (0.042, 0.037, 0.047, 4.2, "True", "0")}.items():
            row = denominators[key]
            for name, number in (("denominator", value), ("ci_low", ci_low), ("ci_high", ci_high), ("ratio_to_floor", ratio)):
                self.assertAlmostEqual(float(row[name]), number, places=7, msg=(key, name))
            self.assertEqual((row["defined"], row["replicates_at_or_below_floor"], row["replicates"], row["ceiling_seeds"]),
                             (defined, at_floor, "80", "2"), key)

        comparison = {(row["domain"], row["pool"]): row for row in reliable("ceiling_e12_vs_e3")}
        self.assertEqual(len(tables["ceiling_e12_vs_e3"]), 3 * 6 * 2)  # shift x class key x pool
        for key, (difference, ci_low, ci_high, higher, lower) in {
                ("lighting", "full"): (0.0125, 0.0075, 0.0175, "True", "False"),
                ("lighting", "matched"): (-0.01, -0.0125, -0.0075, "False", "True"),
                ("weather", "full"): (-0.006, -0.011, -0.001, "False", "True"),
                ("weather", "matched"): (0.006, 0.0053, 0.0067, "True", "False"),
                ("compound", "full"): (0.002, 0.002, 0.002, "False", "False"),
                ("compound", "matched"): (0.002, 0.002, 0.002, "False", "False")}.items():
            row = comparison[key]
            for name, number in (("difference", difference), ("ci_low", ci_low), ("ci_high", ci_high)):
                self.assertAlmostEqual(float(row[name]), number, places=9, msg=(key, name))
            self.assertEqual((row["materially_higher"], row["materially_lower"]), (higher, lower), key)
            self.assertEqual((row["seeds_e3"], row["seeds_e12"], row["paired_seeds"], row["replicates"], row["delta"]),
                             ("0 1", "0 1", "0 1", "80", "0.005"), key)
        self.assertEqual([comparison[(domain, pool)]["pool_images"] for domain in TARGETS for pool in ("full", "matched")],
                         ["1738", "1738", "1985", "1738", "18015", "1738"])
        lighting = comparison[("lighting", "full")]
        self.assertAlmostEqual(float(lighting["mAP_c_e12"]) - float(lighting["mAP_c_e3"]), 0.0125, places=9)
        self.assertAlmostEqual(float(lighting["residual_gap_e3"]), 0.35 - 0.29, places=9)  # source model on its own test set
        self.assertAlmostEqual(float(lighting["residual_gap_e12"]), 0.35 - 0.3025, places=9)
        self.assertAlmostEqual(float(lighting["mAP_source"]) - float(lighting["mAP_c_e12"]), 0.0475, places=9)
        self.assertEqual({**comparison[("compound", "full")], "pool": ""}, {**comparison[("compound", "matched")], "pool": ""})

        # forgetting_e12.csv: every fine-tune run of the revision on the source test set, and no adaptation run.
        self.assertEqual([(row["domain"], row["budget"], row["seed"], row["epochs"]) for row in tables["forgetting_e12"]],
                         [("compound", "1738", "0", "12"), ("compound", "1738", "1", "12"),
                          ("weather", "1985", "0", "12"), ("weather", "1985", "1", "12"), ("weather", "1738", "0", "12"),
                          ("weather", "1738", "1", "12"), ("weather", "1738", "1", "3"),
                          ("lighting", "18015", "0", "12"), ("lighting", "18015", "1", "12"), ("lighting", "1738", "0", "12"),
                          ("lighting", "1738", "1", "12"), ("lighting", "1738", "1", "3")])
        for row in tables["forgetting_e12"]:
            self.assertAlmostEqual(float(row["change_vs_source_model"]), -0.04 if row["epochs"] == "12" else -0.02, places=9)
        self.assertEqual(len(tables["forgetting"]), 1)  # forgetting.csv keeps the campaign runs only

        self.assertEqual(list(summary), ["min_denominator", "interaction", "thresholds", "fits", "recovery_denominator",
                                         "thresholds_e12", "fits_e12", "recovery_denominator_e12", "ceiling_e12_vs_e3"])
        e3, e12 = ("full-pool", "matched-pool"), ("full-pool-e12", "matched-pool-e12")
        self.assertEqual({row["analysis"] for row in summary["thresholds"] + summary["fits"]}, {"full-pool"})
        self.assertEqual([(row["domain"], row["analysis"]) for row in summary["recovery_denominator"]],
                         [(domain, analysis) for domain in TARGETS for analysis in e3])
        self.assertEqual([(row["domain"], row["analysis"]) for row in summary["recovery_denominator_e12"]],
                         [(domain, analysis) for domain in TARGETS for analysis in e12])
        defined = [(domain, analysis) for domain in TARGETS for analysis in e12
                   if (domain, analysis) != ("weather", "full-pool-e12")]
        self.assertEqual([(row["domain"], row["analysis"], row["threshold"]) for row in summary["thresholds_e12"]],
                         [(*key, label) for key in defined for label in ("N_BE", "N75", "N90")])
        self.assertEqual([(row["domain"], row["analysis"], row["region_rule"]) for row in summary["fits_e12"]],
                         [(*key, rule) for key in defined for rule in ("ci", "point")])
        self.assertEqual([(row["domain"], row["pool"], row["materially_higher"], row["materially_lower"])
                          for row in summary["ceiling_e12_vs_e3"]],
                         [("compound", "full", False, False), ("compound", "matched", False, False),
                          ("weather", "full", False, True), ("weather", "matched", True, False),
                          ("lighting", "full", True, False), ("lighting", "matched", False, True)])
        self.assertEqual({row["class"] for key in summary if key.endswith("e12") or key == "ceiling_e12_vs_e3"
                          for row in summary[key]}, {"reliable"})
        self.assertEqual(summary["thresholds_e12"][3]["status"], "determined")
        self.assertEqual((summary["thresholds_e12"][7]["point"], summary["thresholds_e12"][7]["status"]), (None, "not reached"))
        self.assertEqual({(row["band_replicates"], row["band_undefined_replicates"]) for row in summary["thresholds_e12"]},
                         {(80, 0)})
        self.assertEqual([name for row in summary["thresholds"] for name in row if name.startswith("band_")],
                         ["band_from", "band_to"] * 9)

    def test_without_the_revision_no_new_code_is_reached_and_nothing_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            scored, revision, plain, both = (Path(folder) / name for name in ("scored", "revision", "plain", "both"))
            self.build(scored)
            self.build_revision(revision)  # on disk next to the campaign, but not passed
            reached = mock.Mock(side_effect=AssertionError("code of the revision reached without --revision-scored"))
            with mock.patch.multiple("analyze_campaign", read_revision=reached, e12_ceilings=reached, ceiling_comparison=reached,
                                     bootstrap_recovery_measured_top=reached, require_shared_resamples=reached,
                                     resample_signatures=reached, require_ceiling_seeds=reached):
                self.run_main(scored, plain, "--seed", "5")
            reached.assert_not_called()
            self.run_main(scored, both, "--seed", "5", "--revision-scored", str(revision))
            self.assertEqual(sorted(path.name for path in plain.iterdir()), self.e3_files)
            for name in ("rq1_drop", "forgetting", "controls"):
                self.assertEqual((plain / f"{name}.csv").read_bytes(), (both / f"{name}.csv").read_bytes(), name)
            # Rows that the revision must not move: every full-pool row, and the compound rows of matched-pool.
            kept = lambda row: (row["analysis"] == "full-pool"  # noqa: E731
                                or (row["analysis"], row["domain"]) == ("matched-pool", "compound"))
            for name in self.curve_tables:
                before, after = self.table(plain / f"{name}.csv"), self.table(both / f"{name}.csv")
                self.assertEqual({row["analysis"] for row in before}, {"full-pool", "matched-pool"}, name)
                self.assertEqual({row["analysis"] for row in after},
                                 {"full-pool", "matched-pool", "full-pool-e12", "matched-pool-e12"}, name)
                # Every earlier column in its place. recovery_points.csv gains `note`, which the synthetic campaign does
                # not need and the undefined weather curve against the 12-epoch ceiling does; thresholds.csv gains the
                # replicate counts of the band, which only the rows against a 12-epoch ceiling fill.
                columns = list(before[0])
                gained = {"recovery_points": ["note"], "thresholds": ["band_replicates", "band_undefined_replicates"]}
                self.assertEqual(list(after[0]), columns + gained.get(name, []), name)
                self.assertEqual([{column: row[column] for column in columns} for row in after if kept(row)],
                                 [row for row in before if kept(row)], name)
                self.assertGreaterEqual(len([row for row in before if kept(row)]), 4, name)
            summaries = [json.loads((directory / "summary.json").read_text(encoding="utf-8")) for directory in (plain, both)]
        self.assertEqual(list(summaries[0]), ["min_denominator", "interaction", "thresholds", "fits", "recovery_denominator"])
        for key in ("min_denominator", "interaction", "thresholds", "fits"):
            self.assertEqual(summaries[0][key], summaries[1][key], key)
        # recovery_denominator keeps its six rows; the two that changed are the matched-pool ceilings with a second seed.
        self.assertEqual(len(summaries[1]["recovery_denominator"]), 6)
        self.assertEqual([(row["domain"], row["analysis"]) for row, other in
                          zip(summaries[0]["recovery_denominator"], summaries[1]["recovery_denominator"]) if row != other],
                         [("weather", "matched-pool"), ("lighting", "matched-pool")])
        self.assertEqual([row["ceiling_seeds"] for row in summaries[0]["recovery_denominator"]], [2, 2, 2, 1, 2, 1])
        self.assertEqual([row["ceiling_seeds"] for row in summaries[1]["recovery_denominator"]], [2] * 6)

    def test_revision_runs_that_cannot_be_placed_stop_the_analysis(self):
        full = self.revision_runs()
        without = lambda *run: [other for other in full if other[:4] != run]  # noqa: E731
        e3_matched = (r"{}: the 3-epoch matched-pool ceiling \(N = 1738; seed 0 from the campaign, seed 1 from the revision\) "
                      "needs seeds 0 and 1, found {}$")
        # name: (fine-tune runs of the revision, expected message, seeds of the campaign's runs at 1,738 if not the usual)
        cases = {
            "a further 3-epoch full-pool seed": (full + [("weather", "full", 2, 3, 0.33)],
                                                 "no known kind: domain weather, budget full, epochs 3, seed 2"),
            "a ladder rung at 12 epochs": (full + [("lighting", 500, 0, 12, 0.3)],
                                           "no known kind: domain lighting, budget 500, epochs 12, seed 0"),
            "the campaign's own matched-pool run": (full + [("lighting", MATCHED_POOL, 0, 3, 0.29)],
                                                    "no known kind: domain lighting, budget 1738, epochs 3, seed 0"),
            "a third 12-epoch full-pool seed": (full + [("lighting", "full", 2, 12, 0.3)],
                                                "no known kind: domain lighting, budget full, epochs 12, seed 2"),
            "a third 3-epoch matched-pool seed": (full + [("lighting", MATCHED_POOL, 2, 3, 0.29)],
                                                  "no known kind: domain lighting, budget 1738, epochs 3, seed 2"),
            "one run in two folders": (full + [("weather", "full", 1, 12, 0.33)],
                                       "scored more than once.*: weather N = 1985 E = 12 seed 1$"),
            "a matched-pool seed that the campaign scored too": (
                full, "scored more than once.*: lighting N = 1738 E = 3 seed 1$", {"lighting": (0, 1), "weather": (0,)}),
            "no 12-epoch matched-pool ceiling": ([run for run in full if run[:2] != ("lighting", MATCHED_POOL) or run[3] != 12],
                                                 r"lighting: no 12-epoch matched-pool ceiling run \(N = 1738\)"),
            "no 12-epoch full-pool ceiling": ([run for run in full if run[:2] != ("compound", "full")],
                                              r"compound: no 12-epoch full-pool ceiling run \(N = 1738\)"),
            # One 12-epoch seed would be a one-seed ceiling with "materially higher" read off a single pair.
            "one seed of a 12-epoch full-pool ceiling": (
                without("lighting", "full", 1, 12),
                r"lighting: the 12-epoch full-pool ceiling \(N = 18015\) needs seeds 0 and 1, found 0$"),
            "one seed of a 12-epoch matched-pool ceiling": (
                without("weather", MATCHED_POOL, 0, 12),
                r"weather: the 12-epoch matched-pool ceiling \(N = 1738\) needs seeds 0 and 1, found 1$"),
            "no second 3-epoch matched-pool seed": (without("weather", MATCHED_POOL, 1, 3), e3_matched.format("weather", "0")),
            # Without a 3-epoch run at 1,738 the full-pool runs (18,015 images) must not be written as N = 1738.
            "no 3-epoch run at the matched pool in either directory": (
                without("lighting", MATCHED_POOL, 1, 3), e3_matched.format("lighting", "none"), {"weather": (0,)}),
            "no 3-epoch run at the matched pool in the campaign": (full, e3_matched.format("lighting", "1"), {"weather": (0,)}),
        }
        for name, (runs, message, *matched) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as folder:
                scored, revision, output = (Path(folder) / part for part in ("scored", "revision", "analysis"))
                self.build(scored, matched=matched[0] if matched else None)
                self.build_revision(revision, runs)
                with self.assertRaisesRegex(ValueError, message):
                    self.run_main(scored, output, "--revision-scored", str(revision))
                self.assertFalse(output.exists())

    def test_a_revision_scored_on_other_test_resamples_stops_the_analysis(self):
        listed = r"do not share one \(bootstrap seed, resamples, test images, rows of bootstrap_class_ap.npy\): "
        cases = {"another bootstrap seed": ({"bootstrap_seed": 1}, r"\(1, 80, 756, 80\), \(20261030, 80, 756, 80\)\. "),
                 "fewer resamples": ({"replicates": 40}, r"\(20261030, 40, 756, 40\), \(20261030, 80, 756, 80\)\. "),
                 "more resamples": ({"replicates": 160}, r"\(20261030, 160, 756, 160\), \(20261030, 80, 756, 80\)\. ")}
        for name, (options, signatures) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as folder:
                scored, revision, output = (Path(folder) / part for part in ("scored", "revision", "analysis"))
                self.build(scored)
                self.build_revision(revision, **options)
                with self.assertRaisesRegex(ValueError, "Test set compound: .*" + listed + signatures):
                    self.run_main(scored, output, "--revision-scored", str(revision))
                self.assertFalse(output.exists())
                self.run_main(scored, output)  # the campaign alone is read as before
                self.assertEqual(sorted(path.name for path in output.iterdir()), self.e3_files)

    def test_campaign_and_revision_runs_of_two_schedules_are_not_pooled_at_the_matched_pool(self):
        # A campaign whose runs trained for 2 epochs has a seed 0 at 1,738 that is no partner for the revision's 3-epoch
        # seed 1: pooled, the "3-epoch" matched-pool ceiling would be the mean of an E = 2 and an E = 3 run.
        with tempfile.TemporaryDirectory() as folder:
            scored, revision, output = (Path(folder) / part for part in ("scored", "revision", "analysis"))
            self.build(scored, epochs=2)
            self.build_revision(revision)
            with self.assertRaisesRegex(ValueError, "weather: the fine-tune runs at N = 1738 mix epoch counts 2, 3"):
                self.run_main(scored, output, "--revision-scored", str(revision))
            self.assertFalse(output.exists())
            self.run_main(scored, output)  # one schedule throughout: nothing to pool across
            self.assertEqual(sorted(path.name for path in output.iterdir()), self.e3_files)

    def test_a_rerun_without_the_revision_removes_its_tables(self):
        with tempfile.TemporaryDirectory() as folder:
            scored, revision, output = (Path(folder) / name for name in ("scored", "revision", "analysis"))
            self.build(scored)
            self.build_revision(revision)
            self.run_main(scored, output, "--revision-scored", str(revision))
            self.assertTrue((output / "ceiling_e12_vs_e3.csv").exists() and (output / "forgetting_e12.csv").exists())
            self.run_main(scored, output)
            self.assertEqual(sorted(path.name for path in output.iterdir()), self.e3_files)
            self.assertEqual(len(json.loads((output / "summary.json").read_text(encoding="utf-8"))), 5)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
    def test_e12_figures_are_written_next_to_unchanged_e3_figures(self):
        with tempfile.TemporaryDirectory() as folder:
            scored, revision, plain, both = (Path(folder) / name for name in ("scored", "revision", "plain", "both"))
            self.build(scored)
            self.build_revision(revision)
            self.run_main(scored, plain, plots=True)
            self.run_main(scored, both, "--revision-scored", str(revision), plots=True)
            e3_figures = [f"recovery_{domain}.png" for domain in sorted(TARGETS)]
            self.assertEqual(sorted(path.name for path in plain.glob("*.png")), e3_figures)
            # Weather has no defined curve against its 12-epoch full-pool ceiling (gain 0.008), so it gets no figure.
            self.assertEqual(sorted(path.name for path in both.glob("*.png")),
                             sorted(e3_figures + ["recovery_compound_e12.png", "recovery_lighting_e12.png"]))
            for name in e3_figures:
                self.assertEqual((plain / name).read_bytes(), (both / name).read_bytes(), name)
            self.assertGreater((both / "recovery_lighting_e12.png").stat().st_size, 10_000)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
    def test_a_rerun_removes_the_e12_figures_that_it_does_not_draw(self):
        figures = lambda directory: sorted(path.name for path in directory.glob("*.png"))  # noqa: E731
        with tempfile.TemporaryDirectory() as folder:
            scored, revision, flat, output = (Path(folder) / name for name in ("scored", "revision", "flat", "analysis"))
            self.build(scored)
            self.build_revision(revision)
            e3_figures = [f"recovery_{domain}.png" for domain in sorted(TARGETS)]
            both_e12 = ["recovery_compound_e12.png", "recovery_lighting_e12.png"]
            self.run_main(scored, output, "--revision-scored", str(revision), plots=True)
            self.assertEqual(figures(output), sorted(e3_figures + both_e12))
            # A revision whose lighting gain at 12 epochs (0.008) does not exceed the floor: that curve is undefined now,
            # and its figure of the earlier run must not stay next to tables that no longer hold the curve.
            runs = [(*run[:4], self.zero["lighting"] + 0.008) if run[:2] == ("lighting", "full") else run
                    for run in self.revision_runs()]
            self.build_revision(flat, runs)
            self.run_main(scored, output, "--revision-scored", str(flat), plots=True)
            self.assertEqual(figures(output), sorted(e3_figures + ["recovery_compound_e12.png"]))
            self.run_main(scored, output, "--revision-scored", str(revision), plots=True)
            self.assertEqual(figures(output), sorted(e3_figures + both_e12))
            # Without figures no figure is touched; with them and without the revision, the "-e12" figures go like its tables.
            self.run_main(scored, output)
            self.assertEqual(figures(output), sorted(e3_figures + both_e12))
            self.run_main(scored, output, plots=True)
            self.assertEqual(figures(output), e3_figures)
            self.assertEqual(sorted(path.name for path in output.iterdir()), sorted(self.e3_files + e3_figures))


if __name__ == "__main__":
    unittest.main()
