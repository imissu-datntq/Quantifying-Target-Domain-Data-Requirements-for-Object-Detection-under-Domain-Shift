"""Seed spread against N (direction 1.2): chi-square quantiles and SD intervals, digamma/trigamma at half-integers, the
power-law and variance-model fits (with the weighted refinement and its clip), the scale of the fit line in the figure,
the training-noise floor, the choice of rungs, and main() on a scored tree (with the required full-pool seeds).

Synthetic values and synthetic scored trees in a temporary directory only; nothing here reads runs/ or data/.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from statistics import mean

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_seed_spread import (CLIP_SHARE, EULER, KEYS, SIMULATIONS, chi2_cdf, chi2_quantile,  # noqa: E402
                                 digamma_trigamma, draw_panel, fine_tune_runs, main, pooled_floor, power_law_fit,
                                 sd_row, seed_spread, spread_rows, variance_interval, variance_model)
from analyze_seed_spread import VARIANCE_SEED_OFFSET  # noqa: E402
from analyze_rq5 import row_rng  # noqa: E402
from bdd_pipeline.campaign import EXTRA_RUNGS, LADDER, MATCHED_POOL, TARGETS  # noqa: E402
from bdd_pipeline.data import CLASSES, EXPECTED_DOMAINS, RELIABLE_CLASSES  # noqa: E402

LIGHTING_RUNGS = list(LADDER) + list(EXTRA_RUNGS["lighting"])


def run(stage: str, domain: str, budget, seed: int, epochs: int = 3) -> dict:
    return {"stage": stage, "domain": domain, "budget": str(budget), "seed": seed, "epochs": epochs, "total_updates": 500}


def entry(stage: str, domain: str, budget, seed: int, value: float | None, eval_domain: str | None = None,
          epochs: int = 3) -> dict:
    """An in-memory scored entry with the same value for every key."""
    return {"run": run(stage, domain, budget, seed, epochs), "domain": eval_domain or domain,
            "points": dict.fromkeys(KEYS, value), "boots": {}}


def rows(k: float, budgets: list[int], nu: int = 2, domain: str = "lighting", exponent: float = -0.5) -> list[dict]:
    """Spread rows whose SD is exactly k N^exponent."""
    return [{"domain": domain, "budget": n, "sd": k * n ** exponent, "nu": nu} for n in budgets]


class ChiSquareTest(unittest.TestCase):
    def test_one_degree_of_freedom_is_exact(self):
        self.assertAlmostEqual(chi2_quantile(0.025, 1), 0.000982, places=6)
        self.assertAlmostEqual(chi2_quantile(0.975, 1), 5.0239, places=4)

    def test_two_degrees_of_freedom_are_exact(self):
        self.assertAlmostEqual(chi2_quantile(0.975, 2), 7.3778, places=4)
        self.assertAlmostEqual(chi2_quantile(0.025, 2), -2 * math.log(0.975), places=12)

    def test_three_and_more_degrees_of_freedom_are_exact_in_both_tails(self):
        known = {(0.025, 3): 0.2157953, (0.975, 3): 9.3484036, (0.025, 4): 0.4844186, (0.975, 4): 11.1432868,
                 (0.025, 10): 3.2469728, (0.975, 10): 20.4831774}
        for (q, nu), value in known.items():
            with self.subTest(q=q, nu=nu):
                self.assertAlmostEqual(chi2_quantile(q, nu), value, places=6)
        self.assertAlmostEqual(sd_row(0.01, 3)["sd_ci_high"], 0.01 * math.sqrt(3 / 0.2157953), places=8)

    def test_cdf_against_closed_forms(self):
        for x in (0.05, 0.6, 2.5, 9.0):
            with self.subTest(x=x):
                self.assertAlmostEqual(chi2_cdf(x, 1), math.erf(math.sqrt(x / 2)), places=14)
                self.assertAlmostEqual(chi2_cdf(x, 2), 1 - math.exp(-x / 2), places=14)
                self.assertAlmostEqual(chi2_cdf(x, 3), math.erf(math.sqrt(x / 2))
                                       - math.sqrt(2 * x / math.pi) * math.exp(-x / 2), places=14)
                self.assertAlmostEqual(chi2_cdf(x, 4), 1 - math.exp(-x / 2) * (1 + x / 2), places=14)

    def test_no_quantile_below_one_degree_of_freedom(self):
        with self.assertRaises(ValueError):
            chi2_quantile(0.5, 0)

    def test_sd_interval(self):
        row = sd_row(0.01, 2)
        self.assertEqual((row["sd"], row["nu"]), (0.01, 2))
        self.assertAlmostEqual(row["sd_ci_low"], 0.01 * math.sqrt(2 / (-2 * math.log(0.025))), places=15)
        self.assertAlmostEqual(row["sd_ci_high"], 0.01 * math.sqrt(2 / (-2 * math.log(0.975))), places=15)
        for nu in range(1, 6):
            row = sd_row(0.01, nu)
            self.assertLess(row["sd_ci_low"], 0.01)
            self.assertGreater(row["sd_ci_high"], 0.01)


class PolygammaTest(unittest.TestCase):
    known = {0.5: (-1.9635100260214235, 4.934802200544679), 1.0: (-0.5772156649015329, 1.6449340668482264),
             1.5: (0.03648997397857652, 0.9348022005446793), 2.0: (0.42278433509846713, 0.6449340668482264),
             2.5: (0.7031566406452432, 0.4903577561002349)}

    def test_known_values(self):
        for x, (psi, trigamma) in self.known.items():
            with self.subTest(x=x):
                found = digamma_trigamma(x)
                self.assertAlmostEqual(found[0], psi, places=13)
                self.assertAlmostEqual(found[1], trigamma, places=13)

    def test_only_positive_multiples_of_one_half(self):
        for x in (0, -0.5, 0.3, 1.25):
            with self.subTest(x=x), self.assertRaises(ValueError):
                digamma_trigamma(x)


class PowerLawTest(unittest.TestCase):
    def test_exact_inverse_square_root_spread_from_seed_values_gives_slope_minus_half(self):
        k = 0.12
        ladder = {n: {seed: entry("fine-tune", "lighting", n, seed, 0.3 + (seed - 1) * k / math.sqrt(n))
                      for seed in (0, 1, 2)} for n in LIGHTING_RUNGS}
        spread = spread_rows("lighting", "reliable", ladder, 0.1, 0.01)
        for row in spread:
            self.assertAlmostEqual(row["sd"], k / math.sqrt(row["budget"]), places=14)
        fit = power_law_fit(spread)
        self.assertAlmostEqual(fit["slope"], -0.5, places=9)
        self.assertAlmostEqual(fit["intercept"], math.log(k) + EULER / 2, places=8)  # bias of ln s at nu = 2: -gamma / 2
        self.assertAlmostEqual(fit["chi2_df"], 0.0, places=12)
        x = np.log(LIGHTING_RUNGS)
        self.assertAlmostEqual(fit["slope_se"], math.sqrt(math.pi ** 2 / 24 / ((x - x.mean()) ** 2).sum()), places=12)
        self.assertEqual((fit["rungs"], fit["rungs_zero_sd"], fit["parameters"]), (len(LIGHTING_RUNGS), 0, 2))
        self.assertIs(fit["slope_covers_minus_half"], True)
        self.assertIs(fit["slope_below_zero"], True)

    def test_constant_spread_gives_slope_zero(self):
        fit = power_law_fit(rows(0.01, LIGHTING_RUNGS, exponent=0.0))
        self.assertAlmostEqual(fit["slope"], 0.0, places=12)
        self.assertIs(fit["slope_covers_minus_half"], False)
        self.assertIs(fit["slope_below_zero"], False)

    def test_bias_of_ln_s_is_removed_per_rung(self):
        # SD equal to the expected ln-SD bias of its own nu on an exact 1/sqrt(N) line: the corrected response is exact.
        spread = []
        for n, nu in zip(LADDER, (1, 2, 3, 4, 2, 5, 1)):
            psi = digamma_trigamma(nu / 2)[0]
            spread.append({"domain": "weather", "budget": n, "nu": nu,
                           "sd": 0.1 / math.sqrt(n) * math.exp((psi - math.log(nu / 2)) / 2)})
        fit = power_law_fit(spread)
        self.assertAlmostEqual(fit["slope"], -0.5, places=12)
        self.assertAlmostEqual(fit["intercept"], math.log(0.1), places=12)
        self.assertAlmostEqual(fit["chi2_df"], 0.0, places=12)

    def test_weights_and_covariance_match_a_weighted_polyfit(self):
        rng = np.random.default_rng(3)
        nus = [2, 2, 1, 3, 2, 4, 2]
        spread = [{"domain": "compound", "budget": n, "nu": nu, "sd": 0.05 * n ** -0.4 * math.exp(rng.normal(0, 0.3))}
                  for n, nu in zip(LADDER, nus)]
        fit = power_law_fit(spread)
        terms = [digamma_trigamma(nu / 2) for nu in nus]
        y = [math.log(row["sd"]) - (psi - math.log(row["nu"] / 2)) / 2 for row, (psi, _) in zip(spread, terms)]
        w = np.array([4 / trigamma for _, trigamma in terms])
        x = np.log(LADDER)
        (slope, intercept), covariance = np.polyfit(x, y, 1, w=np.sqrt(w), cov="unscaled")
        self.assertAlmostEqual(fit["slope"], slope, places=10)
        self.assertAlmostEqual(fit["intercept"], intercept, places=10)
        self.assertAlmostEqual(fit["slope_se"], math.sqrt(covariance[0, 0]), places=10)
        residual = np.array(y) - (intercept + slope * x)
        self.assertAlmostEqual(fit["chi2_df"], float(w @ residual ** 2) / (len(LADDER) - 2), places=9)
        self.assertAlmostEqual(fit["slope_ci_high"] - fit["slope"], 1.959963984540054 * fit["slope_se"], places=12)

    def test_pooled_fit_has_shift_intercepts_and_one_slope(self):
        pooled = power_law_fit(rows(0.1, list(LADDER), domain="compound") + rows(0.3, list(LADDER), domain="weather"))
        self.assertAlmostEqual(pooled["slope"], -0.5, places=12)
        self.assertEqual((pooled["parameters"], pooled["rungs"]), (3, 2 * len(LADDER)))
        self.assertIsNone(pooled["intercept"])
        # Same rungs and weights in both shifts: the common slope is the mean of the two slopes.
        mixed = power_law_fit(rows(0.1, list(LADDER), domain="compound", exponent=-0.3)
                              + rows(0.3, list(LADDER), domain="weather", exponent=-0.7))
        self.assertAlmostEqual(mixed["slope"], -0.5, places=12)
        self.assertGreater(mixed["chi2_df"], 0)

    def test_finite_pool_correction_gives_slope_minus_half_for_pure_random_sampling(self):
        pool = EXPECTED_DOMAINS["lighting"]["train"]
        sampled = [{"domain": "lighting", "budget": n, "nu": 2, "fpc": (pool - n) / (pool - 1),
                    "sd": 0.1 * math.sqrt((pool - n) / (pool - 1) / n)} for n in LIGHTING_RUNGS]
        self.assertAlmostEqual(power_law_fit(sampled, finite_pool=True)["slope"], -0.5, places=12)
        self.assertLess(power_law_fit(sampled)["slope"], -0.52)  # uncorrected: steeper than -1/2 near the pool

    def test_minimum_rungs_and_zero_spread(self):
        self.assertIsNone(power_law_fit(rows(0.1, [10, 100])))
        self.assertIsNone(power_law_fit([]))
        fit = power_law_fit(rows(0.1, [10, 100, 1000]) + [{"domain": "lighting", "budget": 5000, "sd": 0.0, "nu": 2}])
        self.assertEqual((fit["rungs"], fit["rungs_zero_sd"]), (3, 1))
        self.assertAlmostEqual(fit["slope"], -0.5, places=12)
        two_by_two = rows(0.1, [10, 100], domain="compound") + rows(0.2, [10, 100], domain="weather")
        self.assertIsNone(power_law_fit(two_by_two))  # 4 rungs, 3 parameters: not more than parameters + 1
        self.assertIsNotNone(power_law_fit(two_by_two + rows(0.2, [1000], domain="weather")))


class VarianceModelTest(unittest.TestCase):
    pool = EXPECTED_DOMAINS["lighting"]["train"]

    def spread(self, sigma2: float, c: float) -> list[dict]:
        return [{"domain": "lighting", "budget": n, "nu": 2, "sd": math.sqrt(sigma2 + c * (1 / n - 1 / self.pool))}
                for n in LIGHTING_RUNGS]

    def test_recovers_floor_and_slope_from_exact_variances(self):
        sigma2, c = 2e-5, 0.04
        model = variance_model(self.spread(sigma2, c), self.pool)
        self.assertAlmostEqual(model["sigma2"] / sigma2, 1.0, places=8)
        self.assertAlmostEqual(model["c"] / c, 1.0, places=8)
        self.assertAlmostEqual(model["sigma_floor"], math.sqrt(sigma2), places=10)
        self.assertAlmostEqual(model["n_star"] / (1 / (sigma2 / c + 1 / self.pool)), 1.0, places=6)
        self.assertEqual(model["rungs"], len(LIGHTING_RUNGS))

    def test_pure_sampling_variance_puts_n_star_at_half_the_pool(self):
        # s^2 = c / N is sigma2 = c / M plus c (1/N - 1/M): both parts are equal at N = M / 2.
        model = variance_model([{"domain": "lighting", "budget": n, "nu": 2, "sd": math.sqrt(0.04 / n)}
                                for n in LIGHTING_RUNGS], self.pool)
        self.assertAlmostEqual(model["n_star"] / (self.pool / 2), 1.0, places=6)

    def test_negative_floor_has_no_sigma_floor_and_no_n_star(self):
        model = variance_model(self.spread(-1e-6, 0.04), self.pool)
        self.assertAlmostEqual(model["sigma2"] / -1e-6, 1.0, places=6)
        self.assertEqual(model["sigma_floor"], 0.0)
        self.assertIsNone(model["n_star"])

    def test_needs_three_rungs(self):
        self.assertIsNone(variance_model(self.spread(2e-5, 0.04)[:2], self.pool))

    def test_parametric_interval_is_reproducible_and_covers_the_truth(self):
        sigma2, c = 2e-5, 0.04
        spread = self.spread(sigma2, c)
        model = variance_model(spread, self.pool)
        first = variance_interval(spread, self.pool, model, np.random.default_rng(7))
        self.assertEqual(first, variance_interval(spread, self.pool, model, np.random.default_rng(7)))
        self.assertEqual(first["simulations"], SIMULATIONS)
        self.assertLess(first["sigma_floor_ci_low"], math.sqrt(sigma2))
        self.assertGreater(first["sigma_floor_ci_high"], math.sqrt(sigma2))
        self.assertLess(first["c_ci_low"], c)
        self.assertGreater(first["c_ci_high"], c)
        self.assertLessEqual(first["n_star_ci_low"], first["n_star_ci_high"])
        self.assertTrue(0 <= first["n_star_undefined"] < SIMULATIONS)

    def test_parametric_interval_redraws_around_the_fitted_variances(self):
        # Each simulated s^2 is f chi2_nu / nu around the fitted f: with a stub generator whose chi-square draw equals
        # nu, every redraw is the fit itself, so the interval collapses on the point estimates.
        class Exact:
            def chisquare(self, nu):
                return np.asarray(nu, dtype=float)
        spread = self.spread(2e-5, 0.04)
        model = variance_model(spread, self.pool)
        bounds = variance_interval(spread, self.pool, model, Exact())
        self.assertAlmostEqual(bounds["sigma_floor_ci_low"] / model["sigma_floor"], 1.0, places=8)
        self.assertAlmostEqual(bounds["sigma_floor_ci_high"] / model["sigma_floor"], 1.0, places=8)
        self.assertAlmostEqual(bounds["c_ci_low"] / model["c"], 1.0, places=8)
        self.assertEqual(bounds["n_star_undefined"], 0)

    def by_hand(self, spread: list[dict], clip: bool = True) -> tuple[float, float]:
        """OLS, fitted s^2 clipped below at CLIP_SHARE of the mean s^2, then WLS with weights nu / f^2 (the factor 1/2
        of nu / (2 f^2) leaves the coefficients unchanged), written with np.polyfit."""
        x = np.array([1 / row["budget"] - 1 / self.pool for row in spread])
        y = np.array([row["sd"] ** 2 for row in spread])
        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        if clip:
            fitted = np.maximum(fitted, CLIP_SHARE * y.mean())
        c, sigma2 = np.polyfit(x, y, 1, w=np.sqrt([row["nu"] for row in spread]) / np.abs(fitted))
        return sigma2, c

    def test_one_weighted_refinement_on_noisy_variances_with_mixed_degrees_of_freedom(self):
        factors = (1.3, 0.7, 1.1, 0.8, 1.5, 0.6, 1.2, 0.9, 1.4, 0.75)
        spread = [dict(row, sd=row["sd"] * math.sqrt(factor), nu=nu)
                  for row, factor, nu in zip(self.spread(2e-5, 0.04), factors, (2, 1, 2, 2, 1, 2, 1, 2, 2, 1))]
        model = variance_model(spread, self.pool)
        sigma2, c = self.by_hand(spread)
        self.assertAlmostEqual(model["sigma2"] / sigma2, 1.0, places=9)
        self.assertAlmostEqual(model["c"] / c, 1.0, places=9)
        ordinary = np.polyfit([1 / row["budget"] - 1 / self.pool for row in spread], [row["sd"] ** 2 for row in spread], 1)
        self.assertGreater(abs(model["sigma2"] / ordinary[1] - 1), 0.01)  # the refinement moves the estimate

    def test_fitted_variances_are_clipped_before_weighting(self):
        spread = [{"domain": "lighting", "budget": n, "nu": nu, "sd": math.sqrt(s2)}
                  for n, nu, s2 in ((10, 2, 4e-3), (100, 2, 1e-5), (1000, 1, 1e-5), (5000, 2, 1e-5))]
        model = variance_model(spread, self.pool)
        sigma2, c = self.by_hand(spread)
        self.assertAlmostEqual(model["sigma2"] / sigma2, 1.0, places=9)
        self.assertAlmostEqual(model["c"] / c, 1.0, places=9)
        unclipped = self.by_hand(spread, clip=False)  # OLS fits s^2 < 0 at N = 1000 and 5000
        self.assertGreater(abs(model["sigma2"] / unclipped[0] - 1), 0.01)


class FigureTest(unittest.TestCase):
    def test_fit_line_is_drawn_on_the_scale_of_the_points(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k = 0.12
        ladder = {n: {seed: entry("fine-tune", "lighting", n, seed, 0.3 + (seed - 1) * k / math.sqrt(n))
                      for seed in (0, 1, 2)} for n in LIGHTING_RUNGS}
        spread = spread_rows("lighting", "reliable", ladder, 0.1, 0.01)
        fits = [{"domain": "lighting", "class": "reliable", "model": "power", **power_law_fit(spread)}]
        fig, ax = plt.subplots()
        try:
            draw_panel(ax, "lighting", "reliable", spread, [], fits)
            lines = {line.get_label(): line for line in ax.get_lines()}
            fit = next(line for label, line in lines.items() if label.startswith("fit:"))
            # An exact k / sqrt(N) spread: the fit and the -1/2 reference through the points coincide.
            np.testing.assert_allclose(fit.get_ydata(), lines["slope -1/2"].get_ydata(), rtol=1e-9)
            np.testing.assert_allclose(fit.get_ydata(), k / np.sqrt(fit.get_xdata()), rtol=1e-9)
        finally:
            plt.close(fig)


class FloorAndRungTest(unittest.TestCase):
    def test_floor_from_two_full_pool_seeds(self):
        floor = seed_spread({0: entry("fine-tune", "weather", "full", 0, 0.30),
                             1: entry("fine-tune", "weather", "full", 1, 0.304)}, "reliable")
        sd = 0.004 / math.sqrt(2)
        self.assertAlmostEqual(floor["sd"], sd, places=14)
        self.assertEqual((floor["nu"], floor["seeds"], floor["seed_list"]), (1, 2, "0 1"))
        self.assertAlmostEqual(floor["mean_AP"], 0.302, places=14)
        self.assertAlmostEqual(floor["sd_ci_low"], sd / math.sqrt(chi2_quantile(0.975, 1)), places=14)
        self.assertAlmostEqual(floor["sd_ci_high"], sd / math.sqrt(chi2_quantile(0.025, 1)), places=12)

    def test_pooled_floor_is_the_mean_variance_with_three_degrees_of_freedom(self):
        floors = [{"domain": domain, "sd": sd, "nu": 1} for domain, sd in zip(TARGETS, (0.002, 0.003, 0.004))]
        pooled = pooled_floor(floors)
        self.assertAlmostEqual(pooled["sd"], math.sqrt((0.002 ** 2 + 0.003 ** 2 + 0.004 ** 2) / 3), places=15)
        self.assertEqual((pooled["nu"], pooled["shifts"]), (3, " ".join(TARGETS)))
        self.assertAlmostEqual(pooled["sd_ci_low"], sd_row(pooled["sd"], 3)["sd_ci_low"], places=15)

    def test_rungs_are_e3_ladder_runs_on_the_own_test_set_without_the_matched_pool(self):
        entries = [entry("fine-tune", "lighting", n, seed, 0.3) for n in (10, 5000) for seed in (0, 1, 2)]
        entries += [entry("fine-tune", "lighting", "full", seed, 0.4) for seed in (0, 1)]
        entries += [entry("fine-tune", "lighting", MATCHED_POOL, seed, 0.35) for seed in (0, 1, 2)]  # matched pool
        entries += [entry("fine-tune", "lighting", "full", seed, 0.5, epochs=12) for seed in (0, 1)]  # E = 12 ceiling
        entries += [entry("fine-tune", "lighting", 10, 0, 0.2, eval_domain="source"),  # source test set
                    entry("control", "lighting", 10, 0, 0.1), entry("fine-tune", "weather", 10, 0, 0.3),
                    entry("fine-tune", "lighting", 777, 0, 0.3)]  # not a rung
        ladder, full = fine_tune_runs(entries, "lighting")
        self.assertEqual(sorted(ladder), [10, 5000])
        self.assertEqual({n: sorted(runs) for n, runs in ladder.items()}, {10: [0, 1, 2], 5000: [0, 1, 2]})
        self.assertEqual(sorted(full), [0, 1])
        self.assertTrue(all(e["points"]["reliable"] == 0.4 for e in full.values()))
        weather, _ = fine_tune_runs(entries, "weather")
        self.assertEqual(sorted(weather), [10])

    def test_a_seed_twice_stops(self):
        with self.assertRaises(ValueError):
            fine_tune_runs([entry("fine-tune", "compound", 10, 0, 0.3), entry("fine-tune", "compound", 10, 0, 0.31)],
                           "compound")

    def test_rungs_with_fewer_than_two_seeds_with_a_value_are_skipped(self):
        ladder = {10: {0: entry("fine-tune", "weather", 10, 0, 0.3)},
                  25: {0: entry("fine-tune", "weather", 25, 0, 0.3), 1: entry("fine-tune", "weather", 25, 1, None)},
                  50: {seed: entry("fine-tune", "weather", 50, seed, value) for seed, value in enumerate((0.3, None, 0.31))},
                  100: {seed: entry("fine-tune", "weather", 100, seed, 0.3 + 0.01 * seed) for seed in (0, 1, 2)}}
        spread = spread_rows("weather", "car", ladder, 0.05, 0.01)
        self.assertEqual([row["budget"] for row in spread], [50, 100])
        self.assertEqual((spread[0]["seeds"], spread[0]["seed_list"], spread[0]["nu"]), (2, "0 2", 1))
        self.assertAlmostEqual(spread[1]["sd"], 0.01, places=14)
        self.assertAlmostEqual(spread[1]["sd_recovery"], 0.2, places=12)
        pool = EXPECTED_DOMAINS["weather"]["train"]
        self.assertAlmostEqual(spread[1]["fpc"], (pool - 100) / (pool - 1), places=15)

    def test_sd_in_recovery_units_needs_the_minimum_denominator(self):
        ladder = {100: {seed: entry("fine-tune", "weather", 100, seed, 0.3 + 0.01 * seed) for seed in (0, 1, 2)}}
        for denominator in (0.009, None, -0.02):
            row = spread_rows("weather", "reliable", ladder, denominator, 0.01)[0]
            self.assertIsNone(row["sd_recovery"])
            self.assertEqual(row["recovery_denominator"], denominator)
        # A denominator equal to the floor is undefined, as in bdd_pipeline.recovery; above it, it is defined.
        self.assertIsNone(spread_rows("weather", "reliable", ladder, 0.01, 0.01)[0]["sd_recovery"])
        self.assertAlmostEqual(spread_rows("weather", "reliable", ladder, 0.02, 0.01)[0]["sd_recovery"], 0.5, places=12)


class MainTest(unittest.TestCase):
    """main() on a synthetic campaign tree.

    Rung N of a shift scores zero + 0.03 + (seed - 1) k / sqrt(N) for seeds 0, 1, 2, so every key has SD k / sqrt(N) and
    s^2 = k^2 / N, which the variance model reads as sigma2 = k^2 / M, c = k^2 and N* = M / 2. The E = 3 full pool scores
    zero + gain -+ f (floor SD sqrt(2) f). The tree also holds what must not be read: lighting and weather runs at the
    matched pool, E = 12 full-pool runs, a control run with seed 0, the scores of every fine-tune run on the source test
    set, and at compound N = 25 a single seed. Set folders are numbered and carry no run name.
    """
    images = {"source": 2650, "compound": 756, "weather": 933, "lighting": 4869}
    zero = {"compound": 0.22, "weather": 0.32, "lighting": 0.24}
    k = {"compound": 0.1, "weather": 0.15, "lighting": 0.2}
    gain = {"compound": 0.05, "weather": 0.005, "lighting": 0.08}  # weather: below the minimum denominator
    f = {"compound": 0.002, "weather": 0.003, "lighting": 0.001}

    def write(self, tree: Path, run_fields: dict, test_domain: str, level: float) -> None:
        folder = tree / f"{len(list(tree.glob('*'))):03d}" / test_domain
        folder.mkdir(parents=True)
        per_class = [level + 0.001 * index for index in range(len(CLASSES))]
        metrics = {"run": run_fields, "domain": test_domain, "images": self.images[test_domain],
                   "AP50_95": mean(per_class),
                   "reliable_AP50_95": mean(per_class[CLASSES.index(name)] for name in RELIABLE_CLASSES),
                   "per_class": [{"class": name, "AP50_95": value} for name, value in zip(CLASSES, per_class)],
                   "bootstrap": {"class_order": list(CLASSES), "seed": 20261030, "resamples": 4}}
        (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        np.save(folder / "bootstrap_class_ap.npy", np.tile(per_class, (4, 1)))

    def rungs(self, domain: str) -> list[int]:
        return list(LADDER) + list(EXTRA_RUNGS.get(domain, ()))

    def build(self, tree: Path) -> None:
        source = run("source", "source", "full", 0, epochs=12)
        for domain in self.images:
            self.write(tree, source, domain, self.zero.get(domain, 0.35))
        for domain in TARGETS:
            for n in self.rungs(domain):
                for seed in ((0,) if (domain, n) == ("compound", 25) else (0, 1, 2)):
                    fields = run("fine-tune", domain, n, seed)
                    self.write(tree, fields, domain, self.zero[domain] + 0.03 + (seed - 1) * self.k[domain] / math.sqrt(n))
                    self.write(tree, fields, "source", 0.3 + 0.01 * seed)
            for seed, sign in ((0, -1), (1, 1)):
                self.write(tree, run("fine-tune", domain, "full", seed), domain,
                           self.zero[domain] + self.gain[domain] + sign * self.f[domain])
                self.write(tree, run("fine-tune", domain, "full", seed, epochs=12), domain, self.zero[domain] + 0.3 + seed)
            if domain != "compound":
                for seed in (0, 1, 2):
                    self.write(tree, run("fine-tune", domain, MATCHED_POOL, seed), domain, self.zero[domain] + 0.05 * seed)
            self.write(tree, run("control", domain, 100, 0), domain, 0.01)

    def run_main(self, folder: Path, *options: str) -> Path:
        tree, output = folder / "scored", folder / "out"
        tree.mkdir()
        self.build(tree)
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--scored", str(tree), "--output-dir", str(output), *options])
        return output

    def table(self, path: Path) -> list[dict]:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_seed_drives_the_variance_bootstrap_per_shift_and_key(self):
        def intervals(*options: str) -> list[dict]:
            with tempfile.TemporaryDirectory() as folder:
                output = self.run_main(Path(folder), "--no-plots", *options)
                return [row for row in self.table(output / "seed_spread_fit.csv") if row["model"] == "variance"]
        columns = ("sigma_floor_ci_low", "sigma_floor_ci_high", "c_ci_low", "c_ci_high")
        first, again, other = intervals("--seed", "0"), intervals("--seed", "0"), intervals("--seed", "1")
        self.assertEqual([[row[c] for c in columns] for row in first], [[row[c] for c in columns] for row in again])
        self.assertNotEqual([[row[c] for c in columns] for row in first], [[row[c] for c in columns] for row in other])
        # Each row's interval is variance_interval with the generator of its own (shift, key).
        with tempfile.TemporaryDirectory() as folder:
            output = self.run_main(Path(folder), "--no-plots", "--seed", "3")
            spread = self.table(output / "seed_spread.csv")
            fits = [row for row in self.table(output / "seed_spread_fit.csv") if row["model"] == "variance"]
        for row in fits:
            domain, key = row["domain"], row["class"]
            rungs = [{"domain": domain, "budget": int(r["budget"]), "nu": int(r["nu"]), "sd": float(r["sd"])}
                     for r in spread if r["domain"] == domain and r["class"] == key]
            pool = EXPECTED_DOMAINS[domain]["train"]
            expected = variance_interval(rungs, pool, variance_model(rungs, pool),
                                         row_rng(3 + VARIANCE_SEED_OFFSET, TARGETS.index(domain), KEYS.index(key)))
            self.assertAlmostEqual(float(row["sigma_floor_ci_low"]), expected["sigma_floor_ci_low"], places=12)
            self.assertAlmostEqual(float(row["c_ci_high"]), expected["c_ci_high"], places=12)

    def test_tables_and_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            output = self.run_main(Path(folder), "--no-plots")
            self.assertEqual(sorted(path.name for path in output.iterdir()),
                             ["seed_spread.csv", "seed_spread_fit.csv", "seed_spread_floor.csv", "seed_spread_summary.json"])
            spread = self.table(output / "seed_spread.csv")
            floors = self.table(output / "seed_spread_floor.csv")
            fits = self.table(output / "seed_spread_fit.csv")
            summary = json.loads((output / "seed_spread_summary.json").read_text(encoding="utf-8"))
        budgets = {domain: sorted({int(row["budget"]) for row in spread if row["domain"] == domain}) for domain in TARGETS}
        self.assertEqual(budgets, {"compound": [n for n in LADDER if n != 25], "weather": list(LADDER),
                                   "lighting": self.rungs("lighting")})
        self.assertEqual(len(spread), (6 + 7 + 10) * len(KEYS))
        for row in spread:
            domain, n = row["domain"], int(row["budget"])
            self.assertAlmostEqual(float(row["sd"]), self.k[domain] / math.sqrt(n), places=10)
            self.assertEqual((row["seeds"], row["nu"]), ("3", "2"))
            pool = EXPECTED_DOMAINS[domain]["train"]
            self.assertAlmostEqual(float(row["fpc"]), (pool - n) / (pool - 1), places=12)
            self.assertAlmostEqual(float(row["recovery_denominator"]), self.gain[domain], places=10)
            if domain == "weather":
                self.assertEqual(row["sd_recovery"], "")
            else:
                self.assertAlmostEqual(float(row["sd_recovery"]), self.k[domain] / math.sqrt(n) / self.gain[domain], places=8)

        self.assertEqual(len(floors), 4 * len(KEYS))
        for row in floors:
            if row["domain"] == "pooled":
                expected = math.sqrt(mean(2 * value ** 2 for value in self.f.values()))
                self.assertEqual((row["nu"], row["shifts"], row["budget"]), ("3", " ".join(TARGETS), ""))
            else:
                expected = math.sqrt(2) * self.f[row["domain"]]
                self.assertEqual((row["nu"], row["seed_list"], row["budget"]),
                                 ("1", "0 1", str(EXPECTED_DOMAINS[row["domain"]]["train"])))
            self.assertAlmostEqual(float(row["sd"]), expected, places=10)

        power = [row for row in fits if row["model"] == "power"]
        corrected = [row for row in fits if row["model"] == "power_fpc"]
        variance = [row for row in fits if row["model"] == "variance"]
        self.assertEqual((len(power), len(corrected), len(variance)), (4 * len(KEYS), 4 * len(KEYS), 3 * len(KEYS)))
        for row in corrected:  # the synthetic SD is exactly k / sqrt(N), so the corrected slope is flatter than -1/2
            self.assertGreater(float(row["slope"]), -0.5)
            self.assertEqual(row["parameters"], "4" if row["domain"] == "pooled" else "2")
        for row in power:
            self.assertAlmostEqual(float(row["slope"]), -0.5, places=8)
            self.assertEqual(row["parameters"], "4" if row["domain"] == "pooled" else "2")
            self.assertEqual(row["slope_below_zero"], "True")
        for row in variance:
            domain = row["domain"]
            pool, k2 = EXPECTED_DOMAINS[domain]["train"], self.k[domain] ** 2
            self.assertAlmostEqual(float(row["sigma2"]) / (k2 / pool), 1.0, places=6)
            self.assertAlmostEqual(float(row["c"]) / k2, 1.0, places=6)
            self.assertAlmostEqual(float(row["n_star"]) / (pool / 2), 1.0, places=6)
            self.assertAlmostEqual(float(row["floor_ratio_shift"]),
                                   math.sqrt(k2 / pool) / (math.sqrt(2) * self.f[domain]), places=6)
            pooled = math.sqrt(mean(2 * value ** 2 for value in self.f.values()))
            self.assertAlmostEqual(float(row["floor_ratio_pooled"]), math.sqrt(k2 / pool) / pooled, places=6)
            self.assertLessEqual(float(row["sigma_floor_ci_low"]), float(row["sigma_floor_ci_high"]))
            pooled_high = next(float(floor["sd_ci_high"]) for floor in floors
                               if floor["domain"] == "pooled" and floor["class"] == row["class"])
            self.assertEqual(row["floor_above_training_noise"], str(float(row["sigma_floor_ci_low"]) > pooled_high))

        self.assertEqual(sorted(summary),
                         ["assumption", "fits", "floor", "limit", "min_denominator", "reading", "rules", "seed"])
        self.assertEqual(len(summary["floor"]), 4)
        self.assertEqual(len(summary["fits"]), 11)  # power and power_fpc for 3 shifts + pooled, variance for 3 shifts
        self.assertTrue(all(row["class"] == "reliable" for row in summary["floor"] + summary["fits"]))

    def test_figures(self):
        with tempfile.TemporaryDirectory() as folder:
            output = self.run_main(Path(folder))
            for name in ("seed_spread_reliable.png", "seed_spread_classes.png"):
                self.assertGreater((output / name).stat().st_size, 0)

    def test_the_e3_full_pool_needs_seeds_0_and_1(self):
        with tempfile.TemporaryDirectory() as folder:
            tree = Path(folder) / "scored"
            tree.mkdir()
            self.build(tree)
            for path in list(tree.glob("*/*/metrics.json")):
                if json.loads(path.read_text(encoding="utf-8"))["run"] == run("fine-tune", "weather", "full", 1):
                    shutil.rmtree(path.parents[1])
            with self.assertRaisesRegex(ValueError, r"^weather: the 3-epoch full pool .* needs seeds 0 and 1, found 0$"):
                main(["--scored", str(tree), "--output-dir", str(Path(folder) / "out"), "--no-plots"])

    def test_source_model_on_every_target_is_required(self):
        with tempfile.TemporaryDirectory() as folder:
            tree = Path(folder) / "scored"
            tree.mkdir()
            self.write(tree, run("source", "source", "full", 0, epochs=12), "source", 0.35)
            self.write(tree, run("fine-tune", "compound", 10, 0), "compound", 0.2)
            with self.assertRaises(ValueError):
                main(["--scored", str(tree), "--output-dir", str(Path(folder) / "out"), "--no-plots"])


if __name__ == "__main__":
    unittest.main()
