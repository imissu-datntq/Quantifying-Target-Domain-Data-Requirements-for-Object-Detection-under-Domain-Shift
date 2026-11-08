"""RQ5 analysis helpers: crossover, label-equivalent budget and the seed-paired Delta bootstrap."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_rq5 import crossover, equivalent_budget, paired_delta  # noqa: E402


def entry(point: float, boot: np.ndarray) -> dict:
    return {"points": {"reliable": point}, "boots": {"reliable": boot}}


class CrossoverTest(unittest.TestCase):
    def rows(self, flags):
        return [{"budget": n, "meaningful": f} for n, f in zip((0, 10, 100, 1000), flags)]

    def test_bracket_after_last_meaningful_rung(self):
        self.assertEqual(crossover(self.rows([True, True, False, False]))["N_x"], 100)
        self.assertEqual(crossover(self.rows([True, True, False, False]))["bracket"], "(10, 100]")

    def test_isolated_late_advantage_moves_crossover(self):
        self.assertEqual(crossover(self.rows([True, False, True, False]))["N_x"], 1000)

    def test_beyond_and_none(self):
        self.assertIsNone(crossover(self.rows([False, False, False, True]))["N_x"])
        self.assertEqual(crossover(self.rows([False] * 4))["N_x"], 0)


class EquivalentBudgetTest(unittest.TestCase):
    budgets = [10, 100, 1000, 2000]

    def test_log_linear_interpolation(self):
        value, _ = equivalent_budget(0.25, self.budgets, [0.1, 0.2, 0.3, 0.35])
        self.assertAlmostEqual(value, math.sqrt(100 * 1000))

    def test_sustained_crossing_ignores_early_dip(self):
        # FT dips below the target at 100 after being above it at 10: the last upward crossing counts.
        value, _ = equivalent_budget(0.20, self.budgets, [0.21, 0.15, 0.25, 0.3])
        self.assertTrue(100 < value < 1000)

    def test_edges(self):
        self.assertEqual(equivalent_budget(0.4, self.budgets, [0.1, 0.2, 0.3, 0.35])[1], "> full pool (2000)")
        self.assertEqual(equivalent_budget(0.05, self.budgets, [0.1, 0.2, 0.3, 0.35]), (10.0, "<= 10"))


class PairedDeltaTest(unittest.TestCase):
    def test_pairs_by_seed_and_shares_resamples(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.01, (2, 500))  # per-seed test-resample noise, shared by DA and FT
        ft = {s: entry(0.20 + 0.05 * s, 0.20 + 0.05 * s + noise[s]) for s in (0, 1)}
        da = {s: entry(0.21 + 0.05 * s, 0.21 + 0.05 * s + noise[s]) for s in (0, 1)}
        result = paired_delta(da, ft, "reliable", np.random.default_rng(1))
        self.assertAlmostEqual(result["estimate"], 0.01)
        self.assertAlmostEqual(result["ci_low"], 0.01)  # pairing cancels both seed and test noise
        self.assertAlmostEqual(result["ci_high"], 0.01)

    def test_zero_budget_pairs_every_seed_with_source_model(self):
        boot = np.full(100, 0.2)
        result = paired_delta({0: entry(0.21, boot + 0.01), 1: entry(0.23, boot + 0.03)}, {}, "reliable",
                              np.random.default_rng(0), zero=entry(0.2, boot))
        self.assertAlmostEqual(result["estimate"], 0.02)
        self.assertEqual(result["seeds"], 2)


if __name__ == "__main__":
    unittest.main()
