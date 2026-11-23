"""RQ5 analysis helpers: crossover, label-equivalent budget, the seed-paired Delta bootstrap, the replay share, the
headline rule, the comparison against no fine-tuning, the per-row generators and the shared-resample check; and the
revision runs: their classification by run fields, the merged replay arm, the matched-pass comparisons with their
pre-registered reading, and main() with and without --revision-scored.

Synthetic arrays and synthetic scored trees in a temporary directory only; nothing here reads runs/.
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
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_rq5 import (COLORS, HEADLINE_BUDGETS, MATCHED_BUDGET, MATCHED_COMPARISONS, MATCHED_UPDATES,  # noqa: E402
                         ORIGINAL_UPDATES, REPLAY_SEED_BUDGETS, STREAM_ARMS, arm_runs, crossover, equivalent_budget,
                         expected_undefined_fraction, headline_check, main, matched_pass_reading, matched_pass_rows,
                         matched_share_rows, merge_replay_seeds, paired_delta, plot, quotable, readings_by_shift,
                         replay_share, revision_kind, row_rng, shared_resamples, source_test_rows, split_revision,
                         stale_tables, versus_zero_shot)
from bdd_pipeline.data import CLASSES, RELIABLE_CLASSES  # noqa: E402

STANDS = "FT ahead from N = 1,000 stands"
WITHDRAWN = "crossover withdrawn: no meaningful difference at N = 1,000"
NOT_EVALUATED = "not evaluated: no adapt_matched and FT run share a sampling seed at N = 1,000"


def entry(point: float, boot: np.ndarray) -> dict:
    return {"points": {"reliable": point}, "boots": {"reliable": boot}}


def scored(stage: str, domain: str, budget, seed: int, updates: int | None, point: float = 0.25,
           test_domain: str | None = None, replicates: int = 200) -> dict:
    """One scored entry as the loader returns it, with the run fields that tell the revision runs apart."""
    return {"run": {"stage": stage, "domain": domain, "budget": str(budget), "seed": seed, "total_updates": updates},
            "domain": test_domain or domain, **entry(point, np.full(replicates, point))}


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


def group(values: dict[int, float], replicates: int = 200, noise: np.ndarray | None = None) -> dict[int, dict]:
    """Seed -> entry with a flat bootstrap vector at the point value, plus optional per-seed noise [seed index, replicate]."""
    return {s: entry(v, np.full(replicates, v) + (0.0 if noise is None else noise[i])) for i, (s, v) in enumerate(values.items())}


class ReplayShareTest(unittest.TestCase):
    def test_replay_reproduces_a_known_fraction_of_the_gain(self):
        noise = np.random.default_rng(0).normal(0, 0.01, (2, 500))  # shared by the three arms, so it cancels
        ft = group({0: 0.20, 1: 0.25}, 500, noise)
        adapt = group({0: 0.24, 1: 0.29}, 500, noise)
        replay = group({0: 0.23, 1: 0.28}, 500, noise)
        result = replay_share(adapt, replay, ft, np.random.default_rng(1))
        self.assertEqual((result["seeds"], result["seed_list"]), (2, [0, 1]))
        self.assertAlmostEqual(result["gain"], 0.04)
        self.assertAlmostEqual(result["replay_part"], 0.03)
        self.assertAlmostEqual(result["distill_part"], 0.01)
        self.assertAlmostEqual(result["share"], 0.75)
        self.assertAlmostEqual(result["share_ci_low"], 0.75)
        self.assertAlmostEqual(result["share_ci_high"], 0.75)
        self.assertAlmostEqual(result["gain_ci_low"], 0.04)
        self.assertAlmostEqual(result["distill_part_ci_high"], 0.01)
        self.assertEqual((result["undefined_replicates"], result["replicates"]), (0, 500))
        self.assertEqual(result["direction"], "advantage")
        self.assertTrue(result["majority_not_adaptation"])

    def test_gain_below_delta_gives_no_share(self):
        result = replay_share(group({0: 0.204}, 100), group({0: 0.203}, 100), group({0: 0.200}, 100), np.random.default_rng(0))
        self.assertAlmostEqual(result["gain"], 0.004)
        self.assertAlmostEqual(result["replay_part"], 0.003)
        self.assertAlmostEqual(result["replay_part_ci_low"], 0.003)
        self.assertIsNone(result["share"])
        self.assertIsNone(result["share_ci_low"])
        self.assertIsNone(result["share_ci_high"])
        self.assertEqual((result["undefined_replicates"], result["replicates"]), (100, 100))
        self.assertFalse(result["majority_not_adaptation"])
        # The same numbers pass once delta is lowered below the gain.
        relaxed = replay_share(group({0: 0.204}, 100), group({0: 0.203}, 100), group({0: 0.200}, 100),
                               np.random.default_rng(0), delta=0.001)
        self.assertAlmostEqual(relaxed["share"], 0.75)
        self.assertEqual(relaxed["undefined_replicates"], 0)

    def test_no_interval_when_the_point_share_is_undefined(self):
        # Point gain 0.004 (< delta) although every test resample has a gain of 0.04: no share, hence no interval.
        result = replay_share({0: entry(0.204, np.full(100, 0.24))}, {0: entry(0.203, np.full(100, 0.23))}, group({0: 0.20}, 100),
                              np.random.default_rng(0))
        self.assertIsNone(result["share"])
        self.assertIsNone(result["share_ci_low"])
        self.assertIsNone(result["share_ci_high"])
        self.assertEqual((result["undefined_replicates"], result["replicates"]), (0, 100))
        self.assertAlmostEqual(result["gain_ci_low"], 0.04)
        self.assertFalse(result["majority_not_adaptation"])

    def test_undefined_replicates_are_counted_and_left_out_of_the_interval(self):
        # 40 replicates have a gain of 0.002 (< delta): their ratio would be 15 and must not reach the interval.
        adapt_boot = np.concatenate([np.full(40, 0.202), np.full(60, 0.240)])
        result = replay_share({0: entry(0.24, adapt_boot)}, group({0: 0.23}, 100), group({0: 0.20}, 100),
                              np.random.default_rng(0))
        self.assertAlmostEqual(result["share"], 0.75)
        self.assertEqual((result["undefined_replicates"], result["replicates"]), (40, 100))
        self.assertAlmostEqual(result["share_ci_low"], 0.75)
        self.assertAlmostEqual(result["share_ci_high"], 0.75)
        self.assertAlmostEqual(result["gain_ci_low"], 0.002)  # the gain interval still sees every replicate
        self.assertTrue(result["majority_not_adaptation"])

    def test_no_interval_when_every_replicate_is_undefined(self):
        result = replay_share({0: entry(0.24, np.full(50, 0.201))}, group({0: 0.23}, 50), group({0: 0.20}, 50),
                              np.random.default_rng(0))
        self.assertAlmostEqual(result["share"], 0.75)
        self.assertIsNone(result["share_ci_low"])
        self.assertIsNone(result["share_ci_high"])
        self.assertEqual((result["undefined_replicates"], result["replicates"]), (50, 50))
        self.assertFalse(result["majority_not_adaptation"])

    def test_only_seeds_common_to_all_arms_are_used(self):
        adapt = group({0: 0.24, 1: 0.30, 2: 0.22})
        replay = group({0: 0.22, 2: 0.22, 3: 0.30})
        ft = group({0: 0.20, 1: 0.20, 2: 0.20, 3: 0.20})
        result = replay_share(adapt, replay, ft, np.random.default_rng(0))
        self.assertEqual((result["seeds"], result["seed_list"]), (2, [0, 2]))
        self.assertAlmostEqual(result["gain"], 0.03)  # seeds 0 and 2 only: (0.04 + 0.02) / 2
        self.assertAlmostEqual(result["replay_part"], 0.02)
        self.assertAlmostEqual(result["share"], 2 / 3)
        self.assertIsNone(replay_share(group({0: 0.24}), group({1: 0.22}), group({0: 0.20, 1: 0.20}), np.random.default_rng(0)))
        self.assertIsNone(replay_share(group({0: 0.24}), group({0: 0.22}), group({1: 0.20}), np.random.default_rng(0)))

    def test_missing_point_value_gives_none(self):
        missing = {0: {"points": {"reliable": None}, "boots": {"reliable": np.full(200, np.nan)}}}
        self.assertIsNone(replay_share(group({0: 0.24}), missing, group({0: 0.20}), np.random.default_rng(0)))
        self.assertIsNone(replay_share(group({0: 0.24}), group({0: 0.22}), {}, np.random.default_rng(0), zero=missing[0]))

    def test_zero_budget_pairs_every_common_seed_with_the_source_model(self):
        zero = entry(0.20, np.full(200, 0.20))
        adapt = group({0: 0.21, 1: 0.23})
        one_seed = replay_share(adapt, group({0: 0.204}), {}, np.random.default_rng(0), zero=zero)
        self.assertEqual((one_seed["seeds"], one_seed["seed_list"]), (1, [0]))  # adapt seed 1 has no replay partner
        self.assertAlmostEqual(one_seed["gain"], 0.01)
        self.assertAlmostEqual(one_seed["replay_part"], 0.004)
        self.assertAlmostEqual(one_seed["distill_part"], 0.006)
        self.assertAlmostEqual(one_seed["share"], 0.4)
        self.assertAlmostEqual(one_seed["share_ci_high"], 0.4)
        self.assertFalse(one_seed["majority_not_adaptation"])
        two_seeds = replay_share(adapt, group({0: 0.204, 1: 0.212}), {}, np.random.default_rng(0), zero=zero)
        self.assertEqual(two_seeds["seed_list"], [0, 1])
        self.assertAlmostEqual(two_seeds["gain"], 0.02)
        self.assertAlmostEqual(two_seeds["replay_part"], 0.008)
        self.assertAlmostEqual(two_seeds["share"], 0.4)

    def test_deficit_keeps_the_share_and_reports_the_direction(self):
        result = replay_share(group({0: 0.288, 1: 0.268}), group({0: 0.291, 1: 0.271}), group({0: 0.300, 1: 0.280}),
                              np.random.default_rng(0))
        self.assertAlmostEqual(result["gain"], -0.012)
        self.assertAlmostEqual(result["replay_part"], -0.009)
        self.assertAlmostEqual(result["distill_part"], -0.003)
        self.assertAlmostEqual(result["share"], 0.75)
        self.assertAlmostEqual(result["share_ci_low"], 0.75)
        self.assertEqual(result["direction"], "deficit")
        self.assertEqual(result["undefined_replicates"], 0)
        self.assertTrue(result["majority_not_adaptation"])

    def test_majority_rule_needs_the_lower_bound_above_one_half(self):
        ft, adapt = group({0: 0.20}, 100), group({0: 0.24}, 100)
        clear = replay_share(adapt, group({0: 0.23}, 100), ft, np.random.default_rng(0))
        self.assertAlmostEqual(clear["share_ci_low"], 0.75)
        self.assertTrue(clear["majority_not_adaptation"])
        minority = replay_share(adapt, group({0: 0.216}, 100), ft, np.random.default_rng(0))
        self.assertAlmostEqual(minority["share"], 0.4)
        self.assertFalse(minority["majority_not_adaptation"])
        # Share 0.6, but half of the test resamples put it at 0.4: the lower bound is below 0.5.
        replay_boot = np.concatenate([np.full(50, 0.216), np.full(50, 0.236)])
        uncertain = replay_share(adapt, {0: entry(0.224, replay_boot)}, ft, np.random.default_rng(0))
        self.assertAlmostEqual(uncertain["share"], 0.6)
        self.assertAlmostEqual(uncertain["share_ci_low"], 0.4)
        self.assertAlmostEqual(uncertain["share_ci_high"], 0.9)
        self.assertFalse(uncertain["majority_not_adaptation"])
        # Point share 0.4 although every test resample puts it at 0.75: the point value must pass as well.
        low_point = replay_share(adapt, {0: entry(0.216, np.full(100, 0.23))}, ft, np.random.default_rng(0))
        self.assertAlmostEqual(low_point["share"], 0.4)
        self.assertAlmostEqual(low_point["share_ci_low"], 0.75)
        self.assertFalse(low_point["majority_not_adaptation"])

    def test_seeds_are_redrawn_as_triples(self):
        # Seed levels differ by 0.10. Drawing the arms separately would mix levels into the gain; triples keep it at 0.04.
        ft = group({0: 0.20, 1: 0.30}, 400)
        adapt = group({0: 0.24, 1: 0.34}, 400)
        replay = group({0: 0.21, 1: 0.34}, 400)  # seed shares 0.25 and 1.0
        result = replay_share(adapt, replay, ft, np.random.default_rng(0))
        self.assertAlmostEqual(result["gain_ci_low"], 0.04)
        self.assertAlmostEqual(result["gain_ci_high"], 0.04)
        self.assertAlmostEqual(result["share"], 0.625)
        self.assertAlmostEqual(result["share_ci_low"], 0.25)  # both draws on seed 0
        self.assertAlmostEqual(result["share_ci_high"], 1.0)  # both draws on seed 1
        self.assertFalse(result["majority_not_adaptation"])

    def test_one_seed_draw_serves_the_gain_and_the_replay_part(self):
        # Seed gains 0.02 and 0.08, seed shares both 0.5. A separate draw per component would pair the replay part of one
        # seed with the gain of the other (0.01 / 0.08 or 0.04 / 0.02) and move the share off 0.5.
        result = replay_share(group({0: 0.22, 1: 0.28}, 400), group({0: 0.21, 1: 0.24}, 400), group({0: 0.20, 1: 0.20}, 400),
                              np.random.default_rng(0))
        self.assertAlmostEqual(result["gain"], 0.05)
        self.assertAlmostEqual(result["gain_ci_low"], 0.02)
        self.assertAlmostEqual(result["gain_ci_high"], 0.08)
        self.assertAlmostEqual(result["share"], 0.5)
        self.assertAlmostEqual(result["share_ci_low"], 0.5)
        self.assertAlmostEqual(result["share_ci_high"], 0.5)

    def test_a_replicate_is_the_ratio_of_means_over_as_many_draws_as_seeds(self):
        # Eight seeds; seed 0 has gain 0.16 and share 1, the others gain 0.04 and share 0.5. A replicate draws eight seeds,
        # k of them seed 0, with k ~ Binomial(8, 1/8): P(k <= 2) = 0.933 and P(k <= 3) = 0.989, so the 97.5th percentile
        # sits at k = 3 and the 2.5th at k = 0 (P = 0.34). One draw per replicate would give 0.16 and 1.0 at the top;
        # the mean of the seed ratios would give (3 * 1 + 5 * 0.5) / 8 = 0.6875.
        others = range(1, 8)
        adapt = group({0: 0.36, **{s: 0.24 for s in others}}, 2000)
        replay = group({0: 0.36, **{s: 0.22 for s in others}}, 2000)
        ft = group({s: 0.20 for s in range(8)}, 2000)
        result = replay_share(adapt, replay, ft, np.random.default_rng(0))
        self.assertEqual(result["seeds"], 8)
        self.assertAlmostEqual(result["gain"], 0.055)
        self.assertAlmostEqual(result["share"], 0.30 / 0.44)
        self.assertAlmostEqual(result["gain_ci_low"], 0.04)
        self.assertAlmostEqual(result["gain_ci_high"], (3 * 0.16 + 5 * 0.04) / 8)  # 0.085
        self.assertAlmostEqual(result["replay_part_ci_high"], (3 * 0.16 + 5 * 0.02) / 8)  # 0.0725
        self.assertAlmostEqual(result["share_ci_low"], 0.5)
        self.assertAlmostEqual(result["share_ci_high"], 0.58 / 0.68)  # ratio of the seed means: 0.8529

    def test_intervals_are_the_2_5_and_97_5_percentiles(self):
        # One seed; the replay resamples are spread evenly over 0.23 +/- 0.01, so the 2.5% and 97.5% points are +/- 0.0095.
        spread = np.linspace(-0.01, 0.01, 401)
        result = replay_share(group({0: 0.24}, 401), {0: entry(0.23, 0.23 + spread)}, group({0: 0.20}, 401),
                              np.random.default_rng(0))
        self.assertAlmostEqual(result["replay_part_ci_low"], 0.0205)
        self.assertAlmostEqual(result["replay_part_ci_high"], 0.0395)
        self.assertAlmostEqual(result["distill_part_ci_low"], 0.0005)
        self.assertAlmostEqual(result["distill_part_ci_high"], 0.0195)
        self.assertAlmostEqual(result["gain_ci_low"], 0.04)
        self.assertAlmostEqual(result["gain_ci_high"], 0.04)
        self.assertAlmostEqual(result["share"], 0.75)
        self.assertAlmostEqual(result["share_ci_low"], 0.5125)  # 0.0205 / 0.04
        self.assertAlmostEqual(result["share_ci_high"], 0.9875)  # 0.0395 / 0.04
        self.assertTrue(result["majority_not_adaptation"])

    def test_majority_threshold_is_strictly_one_half(self):
        # Exactly representable values: gain 0.5, replay part 0.28125 (share 0.5625) or 0.25 (share exactly 0.5).
        ft, adapt = group({0: 0.25}, 100), group({0: 0.75}, 100)
        above = replay_share(adapt, group({0: 0.53125}, 100), ft, np.random.default_rng(0))
        self.assertEqual((above["share"], above["share_ci_low"]), (0.5625, 0.5625))
        self.assertTrue(above["majority_not_adaptation"])
        at = replay_share(adapt, group({0: 0.5}, 100), ft, np.random.default_rng(0))
        self.assertEqual((at["share"], at["share_ci_low"]), (0.5, 0.5))
        self.assertFalse(at["majority_not_adaptation"])
        # Each condition is strict on its own: the bound at exactly 0.5, then the point at exactly 0.5.
        bound_at = replay_share(adapt, {0: entry(0.53125, np.full(100, 0.5))}, ft, np.random.default_rng(0))
        self.assertEqual((bound_at["share"], bound_at["share_ci_low"]), (0.5625, 0.5))
        self.assertFalse(bound_at["majority_not_adaptation"])
        point_at = replay_share(adapt, {0: entry(0.5, np.full(100, 0.53125))}, ft, np.random.default_rng(0))
        self.assertEqual((point_at["share"], point_at["share_ci_low"]), (0.5, 0.5625))
        self.assertFalse(point_at["majority_not_adaptation"])

    def test_delta_boundary_is_inclusive(self):
        # |gain| == delta exactly (0.5, exactly representable) is defined, for the point and for every replicate.
        low, middle, high = group({0: 0.25}, 100), group({0: 0.5}, 100), group({0: 0.75}, 100)
        at = replay_share(high, middle, low, np.random.default_rng(0), delta=0.5)
        self.assertEqual((at["gain"], at["share"], at["undefined_replicates"]), (0.5, 0.5, 0))
        self.assertEqual((at["share_ci_low"], at["share_ci_high"]), (0.5, 0.5))
        negative = replay_share(low, middle, high, np.random.default_rng(0), delta=0.5)
        self.assertEqual((negative["gain"], negative["share"], negative["undefined_replicates"]), (-0.5, 0.5, 0))
        below = replay_share(group({0: 0.625}, 100), middle, low, np.random.default_rng(0), delta=0.5)
        self.assertEqual((below["gain"], below["share"], below["undefined_replicates"]), (0.375, None, 100))

    def test_direction_follows_the_sign_of_the_gain_only(self):
        # Replay below FT while adapt is above it: a negative share, still an advantage, never a majority.
        negative_share = replay_share(group({0: 0.24}), group({0: 0.19}), group({0: 0.20}), np.random.default_rng(0))
        self.assertAlmostEqual(negative_share["share"], -0.25)
        self.assertEqual(negative_share["direction"], "advantage")
        self.assertFalse(negative_share["majority_not_adaptation"])
        # A deficit smaller than delta has no share and is still a deficit.
        small_deficit = replay_share(group({0: 0.197}), group({0: 0.20}), group({0: 0.20}), np.random.default_rng(0))
        self.assertAlmostEqual(small_deficit["gain"], -0.003)
        self.assertIsNone(small_deficit["share"])
        self.assertEqual(small_deficit["direction"], "deficit")
        # The rule is "advantage if gain > 0": a gain of exactly 0 is not an advantage.
        none = replay_share(group({0: 0.25}), group({0: 0.25}), group({0: 0.25}), np.random.default_rng(0))
        self.assertEqual((none["gain"], none["share"], none["direction"]), (0.0, None, "deficit"))

    def test_non_finite_bootstrap_replicates_are_rejected(self):
        # A NaN resample is not a replicate with |gain| < delta and must not be counted as one.
        boot = np.full(100, 0.24)
        boot[:30] = np.nan
        ft, replay = group({0: 0.20}, 100), group({0: 0.23}, 100)
        with self.assertRaisesRegex(ValueError, r"Non-finite bootstrap replicates of 'reliable' in \['adapt'\]"):
            replay_share({0: entry(0.24, boot)}, replay, ft, np.random.default_rng(0))
        with self.assertRaisesRegex(ValueError, r"in \['replay', 'reference'\]"):
            replay_share(group({0: 0.24}, 100), {0: entry(0.23, boot)}, {}, np.random.default_rng(0), zero=entry(0.20, boot))

    def test_delta_must_be_positive(self):
        arms = group({0: 0.20}), group({0: 0.19}), group({0: 0.20})
        for delta in (0.0, -1.0, float("nan")):
            with self.assertRaisesRegex(ValueError, "delta must be positive"):
                replay_share(*arms, np.random.default_rng(0), delta=delta)


class HeadlineCheckTest(unittest.TestCase):
    def rows(self, flags, seeds, directions=("advantage",) * 3, undefined=(0, 0, 0)):
        return [{"domain": d, "budget": 10, "majority_not_adaptation": f, "seeds": s, "direction": way,
                 "undefined_replicates": u, "replicates": 1000}
                for d, f, s, way, u in zip(("compound", "weather", "lighting"), flags, seeds, directions, undefined)]

    def test_needs_every_shift_and_three_replay_seeds(self):
        gain = {"direction": "advantage", "largest_undefined_fraction": 0.0}
        self.assertEqual(headline_check(self.rows([True] * 3, [3, 3, 3]), 10),
                         {"share_majority_all_shifts": True, "replay_seeds": 3, "supported": True, **gain})
        self.assertEqual(headline_check(self.rows([True] * 3, [3, 1, 3]), 10),
                         {"share_majority_all_shifts": True, "replay_seeds": 1, "supported": False, **gain})
        self.assertEqual(headline_check(self.rows([True, False, True], [3, 3, 3]), 10),
                         {"share_majority_all_shifts": False, "replay_seeds": 3, "supported": False, **gain})

    def test_missing_shift_or_budget_is_not_supported(self):
        missing = {"share_majority_all_shifts": False, "replay_seeds": 0, "supported": False, "direction": None,
                   "largest_undefined_fraction": None}
        self.assertEqual(headline_check(self.rows([True] * 3, [3, 3, 3])[:2], 10), missing)
        self.assertEqual(headline_check(self.rows([True] * 3, [3, 3, 3]), 100), missing)

    def test_a_share_of_a_deficit_never_supports_the_headline(self):
        # Addendum 2: the headline needs a gain to share out (adapt - FT > 0) in every shift.
        deficit = headline_check(self.rows([True] * 3, [3, 3, 3], ("deficit",) * 3), 10)
        self.assertEqual((deficit["supported"], deficit["share_majority_all_shifts"], deficit["direction"]),
                         (False, True, "deficit"))
        mixed = headline_check(self.rows([True] * 3, [3, 3, 3], ("advantage", "deficit", "deficit")), 10)
        self.assertEqual((mixed["supported"], mixed["direction"]), (False, "mixed"))

    def test_more_than_five_percent_undefined_replicates_blocks_the_headline(self):
        result = headline_check(self.rows([True] * 3, [3, 3, 3], undefined=(0, 926, 13)), 10)
        self.assertEqual(result["largest_undefined_fraction"], 0.926)
        self.assertFalse(result["supported"])
        at_limit = headline_check(self.rows([True] * 3, [3, 3, 3], undefined=(0, 50, 13)), 10)
        self.assertEqual((at_limit["largest_undefined_fraction"], at_limit["supported"]), (0.05, True))
        self.assertFalse(headline_check(self.rows([True] * 3, [3, 3, 3], undefined=(0, 51, 13)), 10)["supported"])

    def test_headline_is_evaluated_at_small_labeled_budgets_only(self):
        self.assertEqual(HEADLINE_BUDGETS, (10, 100))

    def test_quotable_needs_a_defined_share_and_at_most_five_percent_undefined(self):
        self.assertTrue(quotable({"share": 0.8, "undefined_replicates": 50, "replicates": 1000}))
        self.assertFalse(quotable({"share": 0.8, "undefined_replicates": 51, "replicates": 1000}))
        self.assertFalse(quotable({"share": None, "undefined_replicates": 0, "replicates": 1000}))

    def test_rows_of_the_replay_share_feed_the_headline(self):
        # Three shifts where FT is ahead of both arms in all three seeds: share 0.75 of a deficit of 0.012.
        adapt, replay = group({0: 0.288, 1: 0.268, 2: 0.278}), group({0: 0.291, 1: 0.271, 2: 0.281})
        ft = group({0: 0.300, 1: 0.280, 2: 0.290})
        rows = [{"domain": d, "budget": 1000, **replay_share(adapt, replay, ft, np.random.default_rng(0))}
                for d in ("compound", "weather", "lighting")]
        self.assertEqual([(r["share_ci_low"] > 0.5, r["direction"], r["majority_not_adaptation"]) for r in rows],
                         [(True, "deficit", True)] * 3)
        self.assertEqual(headline_check(rows, 1000), {"share_majority_all_shifts": True, "replay_seeds": 3, "supported": False,
                                                      "direction": "deficit", "largest_undefined_fraction": 0.0})


class VersusZeroShotTest(unittest.TestCase):
    def test_gain_over_the_source_model_per_arm_and_budget(self):
        zero = entry(0.20, np.full(200, 0.20))
        arms = {"FT": {10: group({0: 0.15, 1: 0.17}), 100: group({0: 0.21, 1: 0.21})},
                "adapt": {0: group({0: 0.21, 1: 0.21}), 10: group({0: 0.21, 1: 0.23}), 100: group({0: 0.25, 1: 0.25})},
                "replay": {10: group({0: 0.205})}}
        rows = versus_zero_shot(zero, arms, [0, 10, 100], (3, 0))
        self.assertEqual([(r["arm"], r["budget"], r["seeds"]) for r in rows],
                         [("FT", 10, 2), ("FT", 100, 2), ("adapt", 0, 2), ("adapt", 10, 2), ("adapt", 100, 2), ("replay", 10, 1)])
        ft10, adapt10, replay10 = rows[0], rows[3], rows[5]
        self.assertAlmostEqual(ft10["mAP"], 0.16)
        self.assertAlmostEqual(ft10["mAP_0"], 0.20)
        self.assertAlmostEqual(ft10["gain_vs_zero_shot"], -0.04)
        self.assertGreaterEqual(ft10["ci_low"], -0.05 - 1e-12)  # seed redraws stay between the two seeds
        self.assertLessEqual(ft10["ci_high"], -0.03 + 1e-12)
        self.assertAlmostEqual(adapt10["gain_vs_zero_shot"], 0.02)
        self.assertAlmostEqual(replay10["gain_vs_zero_shot"], 0.005)
        self.assertAlmostEqual(replay10["ci_low"], 0.005)
        self.assertAlmostEqual(replay10["ci_high"], 0.005)

    def test_interval_is_the_2_5_and_97_5_percentiles_of_the_requested_key(self):
        # One seed; resamples spread evenly over +/- 0.01 around the point, so the bounds are the gain -/+ 0.0095.
        spread = np.linspace(-0.01, 0.01, 401)
        zero = {"points": {"reliable": 0.20, "car": 0.40}, "boots": {"reliable": np.full(401, 0.20), "car": np.full(401, 0.40)}}
        run = {"points": {"reliable": 0.25, "car": 0.43}, "boots": {"reliable": 0.25 + spread, "car": 0.43 + 2 * spread}}
        (row,) = versus_zero_shot(zero, {"adapt": {10: {0: run}}}, [10], (3, 0))
        self.assertAlmostEqual(row["gain_vs_zero_shot"], 0.05)
        self.assertAlmostEqual(row["ci_low"], 0.0405)
        self.assertAlmostEqual(row["ci_high"], 0.0595)
        (car,) = versus_zero_shot(zero, {"adapt": {10: {0: run}}}, [10], (3, 0), key="car")
        self.assertAlmostEqual(car["mAP"], 0.43)
        self.assertAlmostEqual(car["mAP_0"], 0.40)
        self.assertAlmostEqual(car["gain_vs_zero_shot"], 0.03)
        self.assertAlmostEqual(car["ci_low"], 0.011)
        self.assertAlmostEqual(car["ci_high"], 0.049)

    def test_a_row_does_not_move_when_another_arm_gains_seeds(self):
        # Six FT seeds with gains 0.00 ... 0.05: the interval depends on which seeds are drawn, so a generator shared with
        # the replay rows would shift it as soon as replay has more than one seed.
        zero = entry(0.20, np.full(300, 0.20))
        ft = {10: group({s: 0.20 + 0.01 * s for s in range(6)}, 300)}
        one_seed, three_seeds = {10: group({0: 0.21}, 300)}, {10: group({0: 0.21, 1: 0.22, 2: 0.23}, 300)}
        alone = versus_zero_shot(zero, {"FT": ft}, [10], (3, 0))[0]
        after_one = versus_zero_shot(zero, {"replay": one_seed, "FT": ft}, [10], (3, 0))[1]
        after_three = versus_zero_shot(zero, {"replay": three_seeds, "FT": ft}, [10], (3, 0))[1]
        self.assertEqual(alone["arm"], "FT")
        self.assertEqual(after_one, alone)
        self.assertEqual(after_three, alone)
        self.assertLess(alone["ci_low"], alone["gain_vs_zero_shot"])  # the seeds were redrawn: the interval is not a point
        self.assertLess(alone["gain_vs_zero_shot"], alone["ci_high"])
        other_shift = versus_zero_shot(zero, {"FT": ft}, [10], (3, 1))[0]  # another shift is another stream
        self.assertNotEqual((other_shift["ci_low"], other_shift["ci_high"]), (alone["ci_low"], alone["ci_high"]))


class RowGeneratorTest(unittest.TestCase):
    def test_each_row_is_a_child_of_the_table_seed(self):
        first = row_rng(2, 1, 10).integers(0, 1000, 8).tolist()
        self.assertEqual(first, np.random.default_rng(np.random.SeedSequence(2, spawn_key=(1, 10))).integers(0, 1000, 8).tolist())
        self.assertEqual(row_rng(2, 1, 10).integers(0, 1000, 8).tolist(), first)  # reproducible, whatever was drawn elsewhere
        for other in (row_rng(2, 1, 100), row_rng(2, 0, 10), row_rng(3, 1, 10), np.random.default_rng(2)):
            self.assertNotEqual(other.integers(0, 1000, 8).tolist(), first)


class SharedResamplesTest(unittest.TestCase):
    def metrics(self, domain="compound", images=756, seed=20261030, resamples=1000):
        return {"domain": domain, "images": images, "bootstrap": {"seed": seed, "resamples": resamples}}

    def test_one_seed_count_and_image_list_per_domain(self):
        sets = [self.metrics(), self.metrics(), self.metrics("source", 2650), self.metrics("source", 2650)]
        self.assertEqual(shared_resamples(sets), {"seed": 20261030, "resamples": 1000, "scored_sets": 4,
                                                  "images": {"compound": 756, "source": 2650}})

    def test_unpaired_sets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, r"\(999, 1000\)"):
            shared_resamples([self.metrics(), self.metrics(seed=999)])
        with self.assertRaisesRegex(ValueError, r"\(20261030, 500\)"):
            shared_resamples([self.metrics(), self.metrics("source", 2650, resamples=500)])
        with self.assertRaisesRegex(ValueError, r"\{'compound': \[755, 756\]\}"):
            shared_resamples([self.metrics(), self.metrics(images=755), self.metrics("source", 2650)])


class RevisionKindTest(unittest.TestCase):
    def fields(self, stage: str, budget, updates: int | None, domain: str = "lighting", seed: int = 1) -> dict:
        return {"stage": stage, "domain": domain, "budget": budget, "seed": seed, "epochs": None, "total_updates": updates}

    def test_kind_is_read_from_the_run_fields(self):
        self.assertEqual((REPLAY_SEED_BUDGETS, ORIGINAL_UPDATES, MATCHED_BUDGET, MATCHED_UPDATES), ((10, 100), 2000, 1000, 3000))
        for budget in ("10", "100", 10, 100):
            self.assertEqual(revision_kind(self.fields("replay", budget, 2000)), "replay_seeds")
        for stage in ("adapt", "replay"):
            for domain in ("compound", "weather", "lighting"):
                self.assertEqual(revision_kind(self.fields(stage, "1000", 3000, domain)), "matched")
        # Fine-tune runs are not read here, whatever their numbers: the 12-epoch ceilings, the 3-epoch matched-pool run,
        # and a fine-tune run with the budget and the length of a matched-pass run or of a replay seed.
        for budget, updates in (("full", 108096), ("1738", 10428), ("1738", 2607), ("1000", 3000), ("10", 2000)):
            self.assertIsNone(revision_kind(self.fields("fine-tune", budget, updates)))
        for stage in ("source", "control"):
            self.assertIsNone(revision_kind(self.fields(stage, "full", 29412, "source")))

    def test_an_adapt_or_replay_run_of_neither_kind_is_rejected(self):
        cases = [("adapt", "10", 2000), ("adapt", "100", 2000),  # only replay gained seeds at N = 10 and 100
                 ("replay", "1000", 2000), ("adapt", "1000", 2000),  # N = 1,000 at the original length is not a revision run
                 ("replay", "10", 3000), ("replay", "100", 3000),  # a replay seed at the matched length
                 ("replay", "50", 2000), ("replay", "0", 2000), ("adapt", "5000", 10000), ("adapt", "1000", 3001),
                 ("adapt", "full", 3000), ("replay", "1000", None)]
        for stage, budget, updates in cases:
            with self.subTest(stage=stage, budget=budget, updates=updates):
                message = f"Revision {stage} run is neither a replay seed .* nor a matched-pass run"
                with self.assertRaisesRegex(ValueError, message):
                    revision_kind(self.fields(stage, budget, updates))
        # A target shift is required, and the message names the run by its fields.
        with self.assertRaisesRegex(ValueError, r"domain 'source', budget '10', seed 1, total_updates 2000"):
            revision_kind(self.fields("replay", "10", 2000, domain="source"))
        with self.assertRaisesRegex(ValueError, r"domain 'source', budget '1000', seed 2, total_updates 3000"):
            revision_kind(self.fields("adapt", "1000", 3000, domain="source", seed=2))
        without_length = self.fields("replay", "10", 2000)
        del without_length["total_updates"]
        with self.assertRaisesRegex(ValueError, "total_updates None"):
            revision_kind(without_length)


class SplitRevisionTest(unittest.TestCase):
    def test_kinds_are_separated_and_other_stages_left_out(self):
        seeds = [scored("replay", "lighting", n, s, 2000, test_domain=d)
                 for n in (10, 100) for s in (1, 2) for d in ("lighting", "source")]
        matched = [scored(stage, "lighting", 1000, s, 3000, test_domain=d)
                   for stage in ("adapt", "replay") for s in (0, 1, 2) for d in ("lighting", "source")]
        ceilings = [scored("fine-tune", "lighting", "full", s, 108096) for s in (0, 1)]
        ceilings.append(scored("fine-tune", "lighting", 1738, 1, 2607))
        kinds = split_revision(ceilings[:2] + matched + ceilings[2:] + seeds)
        self.assertEqual(sorted(kinds), ["matched", "replay_seeds"])
        self.assertEqual((len(kinds["replay_seeds"]), len(kinds["matched"])), (8, 12))
        self.assertTrue(all(left is right for left, right in zip(kinds["replay_seeds"], seeds)))
        self.assertTrue(all(left is right for left, right in zip(kinds["matched"], matched)))
        self.assertEqual({(e["run"]["stage"], e["run"]["budget"], e["run"]["total_updates"]) for e in kinds["matched"]},
                         {("adapt", "1000", 3000), ("replay", "1000", 3000)})
        self.assertEqual(split_revision(ceilings), {"replay_seeds": [], "matched": []})

    def test_a_repeated_run_on_one_test_domain_is_rejected(self):
        first = scored("replay", "weather", 100, 2, 2000, point=0.31)
        message = (r"Duplicate revision run \(stage, domain, budget, total_updates, seed\) scored on 'weather': "
                   r"\('replay', 'weather', 100, 2000, 2\)")
        with self.assertRaisesRegex(ValueError, message):
            split_revision([first, scored("replay", "weather", 100, 2, 2000, point=0.33)])
        with self.assertRaisesRegex(ValueError, r"scored on 'source': \('adapt', 'compound', 1000, 3000, 0\)"):
            split_revision([scored("adapt", "compound", 1000, 0, 3000, test_domain="source"),
                            scored("adapt", "compound", 1000, 1, 3000, test_domain="source"),
                            scored("adapt", "compound", 1000, 0, 3000, test_domain="source")])
        # One run scored on its two test domains is one run; another seed, shift or arm is another run.
        kinds = split_revision([first, scored("replay", "weather", 100, 2, 2000, test_domain="source"),
                                scored("replay", "weather", 100, 1, 2000), scored("replay", "compound", 100, 2, 2000),
                                scored("replay", "weather", 1000, 2, 3000), scored("adapt", "weather", 1000, 2, 3000)])
        self.assertEqual((len(kinds["replay_seeds"]), len(kinds["matched"])), (4, 2))

    def test_a_run_of_neither_kind_stops_the_split(self):
        with self.assertRaisesRegex(ValueError, "Revision adapt run is neither"):
            split_revision([scored("replay", "weather", 100, 2, 2000), scored("adapt", "weather", 100, 2, 2000)])

    def test_a_revision_run_without_a_reliable_value_is_rejected_by_name(self):
        undefined = scored("replay", "weather", 100, 2, 2000, test_domain="source")
        undefined["points"]["reliable"] = None  # a reliable class without a value: the seed means would fail on it
        message = r"Revision run without a reliable mAP on 'source': \('replay', 'weather', 100, 2000, 2\)"
        with self.assertRaisesRegex(ValueError, message):
            split_revision([scored("replay", "weather", 100, 1, 2000), undefined])
        matched = scored("adapt", "compound", 1000, 0, 3000)
        matched["points"]["reliable"] = None
        with self.assertRaisesRegex(ValueError, r"on 'compound': \('adapt', 'compound', 1000, 3000, 0\)"):
            split_revision([matched])
        ceiling = scored("fine-tune", "weather", "full", 0, 11916)  # not read here, so not checked here
        ceiling["points"]["reliable"] = None
        self.assertEqual(split_revision([ceiling]), {"replay_seeds": [], "matched": []})


class MergeReplaySeedsTest(unittest.TestCase):
    def original(self) -> dict[int, dict[int, dict]]:
        arm = {n: {0: scored("replay", "lighting", n, 0, 2000, point=point)}
               for n, point in ((10, 0.2443), (100, 0.2577), (1000, 0.2616))}
        arm[0] = {0: scored("replay", "source", 0, 0, 2000, point=0.2476, test_domain="lighting")}
        return arm

    def test_further_seeds_join_the_original_arm_in_seed_order(self):
        original = self.original()
        extra = {10: {2: scored("replay", "lighting", 10, 2, 2000, point=0.2432),
                      1: scored("replay", "lighting", 10, 1, 2000, point=0.2452)},
                 100: {1: scored("replay", "lighting", 100, 1, 2000, point=0.2577)}}
        merged = merge_replay_seeds(original, extra)
        self.assertEqual({n: list(runs) for n, runs in merged.items()}, {10: [0, 1, 2], 100: [0, 1], 1000: [0], 0: [0]})
        self.assertEqual([merged[10][s]["points"]["reliable"] for s in (0, 1, 2)], [0.2443, 0.2452, 0.2432])
        self.assertIs(merged[10][0], original[10][0])  # seed 0 is still the run of the original grid
        self.assertEqual({n: list(runs) for n, runs in original.items()}, {10: [0], 100: [0], 1000: [0], 0: [0]})  # not mutated
        self.assertEqual({n: list(runs) for n, runs in merge_replay_seeds(original, {}).items()},
                         {10: [0], 100: [0], 1000: [0], 0: [0]})

    def test_repeated_seed_and_mixed_run_length_are_rejected(self):
        with self.assertRaisesRegex(ValueError, r"Replay seeds \[0\] at N = 10 are in the original arm and in the revision runs"):
            merge_replay_seeds(self.original(), {10: {0: scored("replay", "lighting", 10, 0, 2000),
                                                      1: scored("replay", "lighting", 10, 1, 2000)}})
        with self.assertRaisesRegex(ValueError, r"Replay runs at N = 100 differ in total_updates .*: \[2000, 3000\]"):
            merge_replay_seeds(self.original(), {100: {1: scored("replay", "lighting", 100, 1, 3000)}})

    def test_merged_arm_gives_the_share_three_seeds_and_leaves_seed_0_alone(self):
        ft, adapt = group({0: 0.20, 1: 0.22, 2: 0.24}), group({0: 0.26, 1: 0.28, 2: 0.30})  # a gain of 0.06 at every seed
        original = {10: {0: scored("replay", "lighting", 10, 0, 2000, point=0.248)}}  # replay part 0.048: share 0.8
        extra = {10: {1: scored("replay", "lighting", 10, 1, 2000, point=0.271),
                      2: scored("replay", "lighting", 10, 2, 2000, point=0.294)}}
        merged = merge_replay_seeds(original, extra)  # replay parts 0.048, 0.051, 0.054
        one = replay_share(adapt, original[10], ft, row_rng(2, 0, 10))
        three = replay_share(adapt, merged[10], ft, row_rng(2, 0, 10))
        self.assertEqual((one["seeds"], one["seed_list"]), (1, [0]))
        self.assertAlmostEqual(one["share"], 0.8)
        self.assertEqual((three["seeds"], three["seed_list"]), (3, [0, 1, 2]))
        self.assertAlmostEqual(three["gain"], 0.06)
        self.assertAlmostEqual(three["replay_part"], 0.051)
        self.assertAlmostEqual(three["share"], 0.85)
        self.assertGreaterEqual(three["share_ci_low"], 0.8 - 1e-9)  # seed shares 0.8, 0.85, 0.9
        self.assertLessEqual(three["share_ci_high"], 0.9 + 1e-9)
        self.assertLess(three["share_ci_low"], three["share_ci_high"])
        # The headline needs three replay seeds in every shift: the merged arm has them, the original arm does not.
        rows = {name: [{"domain": d, "budget": 10, **share} for d in ("compound", "weather", "lighting")]
                for name, share in (("one", one), ("three", three))}
        checks = {name: headline_check(rows[name], 10) for name in rows}
        self.assertEqual((checks["one"]["replay_seeds"], checks["one"]["supported"]), (1, False))
        self.assertEqual((checks["three"]["replay_seeds"], checks["three"]["supported"]), (3, True))
        # The attribution reads seed 0 of the original arm, which the merge neither replaces nor extends.
        self.assertEqual(list(original[10]), [0])
        self.assertEqual(original[10][0]["points"]["reliable"], 0.248)

    def test_matched_runs_never_join_the_replay_arm(self):
        # The matched-pass replay runs here have seeds 1 and 2 only, so pooling them with the original seed 0 at N = 1,000
        # would raise no duplicate: only the run length keeps them out. Their level (0.9) would wreck the share.
        original = {1000: {0: scored("replay", "lighting", 1000, 0, 2000, point=0.291)}}
        revision = ([scored("replay", "lighting", 1000, s, 3000, point=0.9) for s in (1, 2)]
                    + [scored("adapt", "lighting", 1000, s, 3000, point=0.9) for s in (1, 2)]
                    + [scored("replay", "lighting", 10, s, 2000, point=0.25) for s in (1, 2)])
        kinds = split_revision(revision)
        self.assertEqual([(e["run"]["budget"], e["run"]["total_updates"]) for e in kinds["replay_seeds"]], [("10", 2000)] * 2)
        merged = merge_replay_seeds(original, arm_runs(kinds["replay_seeds"], "replay", "lighting", "lighting"))
        self.assertEqual({n: list(runs) for n, runs in merged.items()}, {1000: [0], 10: [1, 2]})
        self.assertIs(merged[1000][0], original[1000][0])
        matched = arm_runs(kinds["matched"], "replay", "lighting", "lighting")
        self.assertEqual({n: list(runs) for n, runs in matched.items()}, {1000: [1, 2]})
        self.assertEqual(list(arm_runs(kinds["matched"], "adapt", "lighting", "lighting")[1000]), [1, 2])
        share = replay_share(group({0: 0.288, 1: 0.268, 2: 0.278}), merged[1000], group({0: 0.300, 1: 0.280, 2: 0.290}),
                             np.random.default_rng(0))
        self.assertEqual((share["seeds"], share["seed_list"]), (1, [0]))
        self.assertAlmostEqual(share["replay_part"], -0.009)
        self.assertAlmostEqual(share["share"], 0.75)


class MatchedPassReadingTest(unittest.TestCase):
    def test_ft_ahead_needs_the_estimate_at_most_minus_delta_and_the_upper_bound_below_zero(self):
        self.assertEqual(matched_pass_reading(-0.008, -0.004), {"ft_ahead_stands": True, "reading": STANDS})
        self.assertEqual(matched_pass_reading(-0.002, -0.001), {"ft_ahead_stands": False, "reading": WITHDRAWN})
        # The estimate exactly at -delta passes (<=); one step above it does not.
        self.assertEqual(matched_pass_reading(-0.005, -0.001), {"ft_ahead_stands": True, "reading": STANDS})
        self.assertEqual(matched_pass_reading(math.nextafter(-0.005, 0.0), -0.001),
                         {"ft_ahead_stands": False, "reading": WITHDRAWN})
        # The upper bound exactly at 0 fails (<); one step below it passes.
        self.assertEqual(matched_pass_reading(-0.01, 0.0), {"ft_ahead_stands": False, "reading": WITHDRAWN})
        self.assertEqual(matched_pass_reading(-0.01, math.nextafter(0.0, -1.0)), {"ft_ahead_stands": True, "reading": STANDS})
        self.assertEqual(matched_pass_reading(-0.01, 0.002), {"ft_ahead_stands": False, "reading": WITHDRAWN})
        # A positive difference, even with its whole interval above 0, withdraws the claim: the rule has two readings.
        self.assertEqual(matched_pass_reading(0.02, 0.03), {"ft_ahead_stands": False, "reading": WITHDRAWN})
        self.assertEqual(matched_pass_reading(0.02, -0.01), {"ft_ahead_stands": False, "reading": WITHDRAWN})
        # Another delta, in exactly representable values.
        self.assertTrue(matched_pass_reading(-0.5, -0.25, delta=0.5)["ft_ahead_stands"])
        self.assertFalse(matched_pass_reading(-0.4375, -0.25, delta=0.5)["ft_ahead_stands"])


class MatchedPassRowsTest(unittest.TestCase):
    names = ["adapt_matched - FT", "replay_matched - FT", "adapt_matched - replay_matched", "adapt_matched - adapt_original",
             "replay_matched - replay_original", "adapt_original - FT"]

    def arms(self, replicates: int = 400) -> dict[str, dict[int, dict]]:
        """FT at 0.30, 0.28, 0.29 by seed; every arm is FT plus a constant, with noise shared by all arms of a seed."""
        noise = np.random.default_rng(0).normal(0, 0.01, (3, replicates))
        ft = {0: 0.30, 1: 0.28, 2: 0.29}
        return {"FT": group(ft, replicates, noise),
                "adapt_matched": group({s: v - 0.008 for s, v in ft.items()}, replicates, noise),
                "replay_matched": group({s: v - 0.006 for s, v in ft.items()}, replicates, noise),
                "adapt_original": group({s: v - 0.012 for s, v in ft.items()}, replicates, noise),
                "replay_original": group({0: ft[0] - 0.011}, replicates, noise)}  # seed 0 only, as in the original grid

    def test_six_comparisons_with_a_known_paired_difference_and_interval(self):
        self.assertEqual([f"{left} - {right}" for left, right in MATCHED_COMPARISONS], self.names)
        rows = matched_pass_rows(self.arms(), (5, 0), reading=True)
        self.assertEqual([r["comparison"] for r in rows], self.names)
        self.assertEqual([(r["seeds"], r["seed_list"]) for r in rows],
                         [(3, [0, 1, 2])] * 4 + [(1, [0])] + [(3, [0, 1, 2])])
        expected = [(0.282, 0.29, -0.008), (0.284, 0.29, -0.006), (0.282, 0.284, -0.002), (0.282, 0.278, 0.004),
                    (0.294, 0.289, 0.005), (0.278, 0.29, -0.012)]  # replay_original pairs seed 0 only: 0.294 - 0.289
        for row, (left, right, difference) in zip(rows, expected):
            with self.subTest(comparison=row["comparison"]):
                self.assertAlmostEqual(row["left"], left)
                self.assertAlmostEqual(row["right"], right)
                self.assertAlmostEqual(row["difference"], difference)
                self.assertAlmostEqual(row["ci_low"], difference)  # pairing cancels the seed level and the test noise
                self.assertAlmostEqual(row["ci_high"], difference)
        self.assertEqual((rows[0]["ft_ahead_stands"], rows[0]["reading"]), (True, STANDS))
        self.assertTrue(all("reading" not in row and "ft_ahead_stands" not in row for row in rows[1:]))
        self.assertEqual(list(rows[1]), ["comparison", "seeds", "seed_list", "left", "right", "difference", "ci_low", "ci_high"])
        # Without the reading (the per-class table) no row carries it.
        self.assertTrue(all("reading" not in row for row in matched_pass_rows(self.arms(), (5, 0))))

    def test_interval_is_the_2_5_and_97_5_percentiles_and_feeds_the_reading(self):
        # One seed; adapt_matched resamples spread evenly over 0.28 +/- 0.01, FT flat: the bounds are -0.02 -/+ 0.0095.
        spread = np.linspace(-0.01, 0.01, 401)
        arms = {"adapt_matched": {0: entry(0.28, 0.28 + spread)}, "FT": group({0: 0.30}, 401), "replay_matched": {},
                "adapt_original": {}, "replay_original": {}}
        (row,) = matched_pass_rows(arms, (5, 0), reading=True)
        self.assertEqual((row["comparison"], row["seeds"], row["seed_list"]), ("adapt_matched - FT", 1, [0]))
        self.assertAlmostEqual(row["difference"], -0.02)
        self.assertAlmostEqual(row["ci_low"], -0.0295)
        self.assertAlmostEqual(row["ci_high"], -0.0105)
        self.assertEqual((row["ft_ahead_stands"], row["reading"]), (True, STANDS))
        # Three times the spread puts the upper bound at +0.0085: the estimate alone (-0.02 <= -delta) does not carry it.
        arms["adapt_matched"] = {0: entry(0.28, 0.28 + 3 * spread)}
        (wide,) = matched_pass_rows(arms, (5, 0), reading=True)
        self.assertAlmostEqual(wide["difference"], -0.02)
        self.assertAlmostEqual(wide["ci_high"], 0.0085)
        self.assertEqual((wide["ft_ahead_stands"], wide["reading"]), (False, WITHDRAWN))

    def test_seeds_are_redrawn_as_pairs_and_an_upper_bound_of_exactly_zero_withdraws(self):
        # FT levels differ by 0.10 between the seeds; adapt_matched is 0.02 below FT at seed 0 and equal at seed 1.
        # Paired redraws give -0.02 (both draws seed 0), -0.01 or 0.0 (both seed 1); arms drawn apart would reach +/- 0.1.
        arms = {"adapt_matched": group({0: 0.28, 1: 0.20}, 400), "FT": group({0: 0.30, 1: 0.20}, 400), "replay_matched": {},
                "adapt_original": {}, "replay_original": {}}
        (row,) = matched_pass_rows(arms, (5, 0), reading=True)
        self.assertAlmostEqual(row["difference"], -0.01)
        self.assertAlmostEqual(row["ci_low"], -0.02)
        self.assertEqual(row["ci_high"], 0.0)
        self.assertEqual((row["ft_ahead_stands"], row["reading"]), (False, WITHDRAWN))  # -0.01 <= -delta, but 0.0 < 0 fails

    def test_reading_at_the_estimate_boundary_and_for_a_positive_difference(self):
        rest = {"replay_matched": {}, "adapt_original": {}, "replay_original": {}}
        # Exactly representable: 0.25 - 0.75 = -0.5 = -delta passes, 0.3125 - 0.75 = -0.4375 does not.
        (at,) = matched_pass_rows({"adapt_matched": group({0: 0.25}), "FT": group({0: 0.75}), **rest}, (5, 0),
                                  reading=True, delta=0.5)
        self.assertEqual((at["difference"], at["ci_high"], at["ft_ahead_stands"], at["reading"]), (-0.5, -0.5, True, STANDS))
        (above,) = matched_pass_rows({"adapt_matched": group({0: 0.3125}), "FT": group({0: 0.75}), **rest}, (5, 0),
                                     reading=True, delta=0.5)
        self.assertEqual((above["difference"], above["ft_ahead_stands"], above["reading"]), (-0.4375, False, WITHDRAWN))
        (positive,) = matched_pass_rows({"adapt_matched": group({0: 0.32}), "FT": group({0: 0.30}), **rest}, (5, 0), reading=True)
        self.assertAlmostEqual(positive["difference"], 0.02)
        self.assertAlmostEqual(positive["ci_low"], 0.02)
        self.assertEqual((positive["ft_ahead_stands"], positive["reading"]), (False, WITHDRAWN))

    def test_matched_and_original_arms_stay_two_arms(self):
        # Pooled into one arm of six runs the two would average to FT - 0.010 and their difference would vanish.
        rows = {r["comparison"]: r for r in matched_pass_rows(self.arms(), (5, 0))}
        self.assertAlmostEqual(rows["adapt_matched - adapt_original"]["difference"], 0.004)
        self.assertAlmostEqual(rows["adapt_matched - FT"]["difference"], -0.008)
        self.assertAlmostEqual(rows["adapt_original - FT"]["difference"], -0.012)
        self.assertEqual((rows["adapt_matched - FT"]["seeds"], rows["adapt_original - FT"]["seeds"]), (3, 3))
        self.assertAlmostEqual(rows["adapt_matched - FT"]["left"], 0.282)
        self.assertAlmostEqual(rows["adapt_original - FT"]["left"], 0.278)

    def test_only_seeds_in_both_arms_are_paired_and_the_requested_key_is_used(self):
        def classes(reliable: float, car: float) -> dict:
            return {"points": {"reliable": reliable, "car": car},
                    "boots": {"reliable": np.full(100, reliable), "car": np.full(100, car)}}
        arms = {"adapt_matched": {0: classes(0.29, 0.50), 1: classes(0.27, 0.46), 2: classes(0.40, 0.90)},
                "FT": {0: classes(0.30, 0.52), 1: classes(0.28, 0.50)}, "replay_matched": {}, "adapt_original": {},
                "replay_original": {}}
        (row,) = matched_pass_rows(arms, (6, 0, 1), key="car")
        self.assertEqual((row["seeds"], row["seed_list"]), (2, [0, 1]))  # adapt_matched seed 2 has no FT partner
        self.assertAlmostEqual(row["left"], 0.48)
        self.assertAlmostEqual(row["right"], 0.51)
        self.assertAlmostEqual(row["difference"], -0.03)
        self.assertNotIn("reading", row)
        (reliable,) = matched_pass_rows(arms, (5, 0))
        self.assertAlmostEqual(reliable["difference"], -0.01)
        self.assertEqual(matched_pass_rows({**arms, "FT": {3: classes(0.30, 0.52)}}, (5, 0)), [])  # no common seed, no row
        # The same with the unpaired seed in the right arm: FT seed 2 (0.90) has no adapt_matched partner and stays out of
        # `right`, which would be 0.4933 over all three FT seeds.
        mirrored = {**arms, "adapt_matched": {0: classes(0.29, 0.50), 1: classes(0.27, 0.46)},
                    "FT": {0: classes(0.30, 0.52), 1: classes(0.28, 0.50), 2: classes(0.90, 0.95)}}
        (row,) = matched_pass_rows(mirrored, (5, 0), reading=True)
        self.assertEqual((row["seeds"], row["seed_list"]), (2, [0, 1]))
        self.assertAlmostEqual(row["left"], 0.28)
        self.assertAlmostEqual(row["right"], 0.29)
        self.assertAlmostEqual(row["difference"], -0.01)
        self.assertAlmostEqual(row["left"] - row["right"], row["difference"])

    def test_the_reading_needs_every_test_resample(self):
        # 150 of 200 resamples of adapt_matched are NaN. Percentiles over the finite quarter would still give an interval
        # and "FT ahead"; the row that carries the pre-registered reading must stop instead.
        boot = np.full(200, 0.28)
        boot[:150] = np.nan
        rest = {"replay_matched": {}, "adapt_original": {}, "replay_original": {}}
        arms = {"adapt_matched": {0: entry(0.28, boot)}, "FT": group({0: 0.30}), **rest}
        message = r"Non-finite bootstrap replicates of 'reliable' in \['adapt_matched'\]: the matched-pass reading needs"
        with self.assertRaisesRegex(ValueError, message):
            matched_pass_rows(arms, (5, 0), reading=True)
        with self.assertRaisesRegex(ValueError, r"in \['FT'\]"):
            matched_pass_rows({"adapt_matched": group({0: 0.28}), "FT": {0: entry(0.30, boot + 0.02)}, **rest}, (5, 0),
                              reading=True)
        # A seed that is not paired is not read: FT seed 1 has no adapt_matched partner.
        (row,) = matched_pass_rows({"adapt_matched": group({0: 0.28}), "FT": {**group({0: 0.30}), 1: entry(0.30, boot)},
                                    **rest}, (5, 0), reading=True)
        self.assertEqual((row["seed_list"], row["reading"]), ([0], STANDS))
        # Only that row is guarded: the tables without the reading, and the other comparisons, keep the interval of
        # paired_delta over the finite resamples.
        (plain,) = matched_pass_rows(arms, (5, 0))
        self.assertAlmostEqual(plain["ci_high"], -0.02)
        others = {"adapt_matched": group({0: 0.28}), "FT": group({0: 0.30}), "replay_matched": {0: entry(0.29, boot + 0.01)},
                  "adapt_original": {}, "replay_original": {}}
        rows = matched_pass_rows(others, (5, 0), reading=True)
        self.assertEqual([r["comparison"] for r in rows],
                         ["adapt_matched - FT", "replay_matched - FT", "adapt_matched - replay_matched"])
        self.assertEqual(rows[0]["reading"], STANDS)
        self.assertAlmostEqual(rows[1]["ci_low"], -0.01)

    def test_a_row_does_not_move_when_another_arm_gains_seeds(self):
        # Six seeds whose adapt_matched - FT differences run from -0.010 to 0.000: the interval depends on the seeds drawn,
        # so a generator shared with the replay rows would move it as soon as a replay arm has more seeds.
        ft = group({s: 0.30 for s in range(6)}, 300)
        adapt_matched = group({s: 0.290 + 0.002 * s for s in range(6)}, 300)
        adapt_original = group({s: 0.285 + 0.001 * s for s in range(6)}, 300)

        def rows(replay_matched: dict, replay_original: dict, stream: tuple[int, ...] = (5, 0)) -> dict[str, dict]:
            arms = {"adapt_matched": adapt_matched, "replay_matched": replay_matched, "adapt_original": adapt_original,
                    "replay_original": replay_original, "FT": ft}
            return {r["comparison"]: r for r in matched_pass_rows(arms, stream, reading=True)}

        few = rows(group({0: 0.292}, 300), group({0: 0.288}, 300))
        many = rows(group({s: 0.292 + 0.001 * s for s in range(6)}, 300), group({s: 0.288 + 0.001 * s for s in range(3)}, 300))
        self.assertEqual((few["replay_matched - FT"]["seeds"], many["replay_matched - FT"]["seeds"]), (1, 6))
        self.assertEqual(few["replay_matched - replay_original"]["seeds"], 1)
        self.assertEqual(many["replay_matched - replay_original"]["seeds"], 3)
        for name in ("adapt_matched - FT", "adapt_matched - adapt_original", "adapt_original - FT"):
            self.assertEqual(many[name], few[name], name)
        pre_registered = few["adapt_matched - FT"]
        self.assertAlmostEqual(pre_registered["difference"], -0.005)
        self.assertLess(pre_registered["ci_low"], pre_registered["difference"])  # the seeds were redrawn
        self.assertLess(pre_registered["difference"], pre_registered["ci_high"])
        # Each comparison takes the child (comparison index) of the stream, whatever the other comparisons drew.
        for index, (left, right) in ((0, (adapt_matched, ft)), (3, (adapt_matched, adapt_original)), (5, (adapt_original, ft))):
            expected = paired_delta(left, right, "reliable", row_rng(5, 0, index))
            row = few[self.names[index]]
            self.assertEqual((row["ci_low"], row["ci_high"]), (expected["ci_low"], expected["ci_high"]))
        other_shift = rows(group({0: 0.292}, 300), group({0: 0.288}, 300), stream=(5, 1))["adapt_matched - FT"]
        self.assertNotEqual((other_shift["ci_low"], other_shift["ci_high"]),
                            (pre_registered["ci_low"], pre_registered["ci_high"]))


class MatchedShareRowsTest(unittest.TestCase):
    def test_share_against_fine_tuning_and_against_the_source_model(self):
        # Seed gains -0.010, -0.008, -0.006 and replay parts -0.0075, -0.006, -0.0045: share 0.75 at every seed.
        ft = group({0: 0.30, 1: 0.28, 2: 0.29})
        adapt, replay = group({0: 0.290, 1: 0.272, 2: 0.284}), group({0: 0.2925, 1: 0.274, 2: 0.2855})
        zero = entry(0.25, np.full(200, 0.25))
        against_ft, against_source = matched_share_rows(adapt, replay, ft, zero, (7, 0))
        self.assertEqual([against_ft[key] for key in ("budget", "labeled_budget", "reference", "seeds", "seed_list")],
                         [1000, True, "FT", 3, [0, 1, 2]])
        self.assertAlmostEqual(against_ft["gain"], -0.008)
        self.assertAlmostEqual(against_ft["replay_part"], -0.006)
        self.assertAlmostEqual(against_ft["distill_part"], -0.002)
        self.assertAlmostEqual(against_ft["share"], 0.75)
        self.assertAlmostEqual(against_ft["share_ci_low"], 0.75)
        self.assertAlmostEqual(against_ft["share_ci_high"], 0.75)
        flags = ("direction", "majority_not_adaptation", "quotable", "undefined_replicates")
        self.assertEqual([against_ft[key] for key in flags], ["deficit", True, True, 0])
        self.assertEqual((against_source["reference"], against_source["seeds"], against_source["direction"]),
                         ("source model", 3, "advantage"))
        self.assertAlmostEqual(against_source["gain"], 0.032)  # seed mean 0.282 against mAP(0) = 0.25
        self.assertAlmostEqual(against_source["replay_part"], 0.034)
        self.assertAlmostEqual(against_source["share"], 1.0625)
        # Each reference takes its own child of the stream: 0 for FT, 1 for the source model.
        for row, expected in ((against_ft, replay_share(adapt, replay, ft, row_rng(7, 0, 0))),
                              (against_source, replay_share(adapt, replay, {}, row_rng(7, 0, 1), zero=zero))):
            self.assertEqual((row["gain_ci_low"], row["gain_ci_high"]), (expected["gain_ci_low"], expected["gain_ci_high"]))
        self.assertLess(against_ft["gain_ci_low"], against_ft["gain_ci_high"])  # the gain differs by seed, so the draw shows

    def test_a_gain_below_delta_is_not_quotable(self):
        ft = group({0: 0.30, 1: 0.28})
        (against_ft, against_source) = matched_share_rows(group({0: 0.298, 1: 0.278}), group({0: 0.2985, 1: 0.2785}), ft,
                                                          entry(0.25, np.full(200, 0.25)), (7, 0))
        self.assertAlmostEqual(against_ft["gain"], -0.002)
        self.assertEqual((against_ft["share"], against_ft["quotable"], against_ft["undefined_replicates"]), (None, False, 200))
        self.assertEqual((against_source["quotable"], against_source["undefined_replicates"]), (True, 0))
        self.assertEqual((against_ft["expected_undefined_fraction"], against_source["expected_undefined_fraction"]), (1.0, 0.0))
        self.assertEqual(list(against_ft)[-2:], ["quotable", "expected_undefined_fraction"])
        self.assertEqual(matched_share_rows({}, {}, ft, entry(0.25, np.full(200, 0.25)), (7, 0)), [])  # no matched run, no row

    def test_quotable_near_the_limit_changes_with_the_stream_and_the_expected_fraction_does_not(self):
        # adapt_matched is 0.012 below FT, except seed 2 in the first 77 of 400 test resamples, where it is 0.001 below.
        # There a redraw with seed 2 drawn twice (gain -0.00467) or three times (-0.001) is undefined: 7 of the 27 ordered
        # redraws. Expected undefined fraction 77 / 400 * 7 / 27 = 0.0499, on the 5% limit of 20 replicates, so the count
        # of one random redraw per resample falls on either side of the limit, depending on the stream.
        replicates, sensitive = 400, 77
        ft = group({0: 0.30, 1: 0.30, 2: 0.30}, replicates)
        boot = np.full(replicates, 0.288)
        boot[:sensitive] = 0.299
        adapt = {**group({0: 0.288, 1: 0.288}, replicates), 2: entry(0.288, boot)}
        replay = group({0: 0.291, 1: 0.291, 2: 0.291}, replicates)
        zero = entry(0.25, np.full(replicates, 0.25))
        counts = {}
        for stream in range(7, 19):
            against_ft, against_source = matched_share_rows(adapt, replay, ft, zero, (stream, 0))
            draws = row_rng(stream, 0, 0).integers(0, 3, (replicates, 3))  # the redraw of replay_share for the FT reference
            counts[stream] = int(((draws[:sensitive] == 2).sum(axis=1) >= 2).sum())
            self.assertEqual(against_ft["undefined_replicates"], counts[stream], stream)
            self.assertEqual(against_ft["quotable"], counts[stream] <= 20, stream)  # still the flag on the drawn replicates
            self.assertAlmostEqual(against_ft["expected_undefined_fraction"], 77 * 7 / (400 * 27), places=12)
            self.assertAlmostEqual(against_ft["share"], 0.75)
            self.assertEqual((against_source["expected_undefined_fraction"], against_source["quotable"]), (0.0, True))
        self.assertGreater(min(counts.values()), 0)
        self.assertEqual({count <= 20 for count in counts.values()}, {True, False})  # the flag is not a property of the data


class ExpectedUndefinedFractionTest(unittest.TestCase):
    def test_mean_over_the_test_resamples_and_every_ordered_redraw_of_the_seeds(self):
        # Seed gains -0.012, -0.012, -0.001. Of the 27 ordered redraws of three seeds, 6 draw the third seed twice
        # (gain -0.00467) and 1 draws it three times (-0.001): 7 are within delta = 0.005 of 0.
        gains = np.array([[-0.012] * 4, [-0.012] * 4, [-0.001] * 4])
        self.assertAlmostEqual(expected_undefined_fraction(gains), 7 / 27, places=12)
        gains[:, :2] = -0.02  # half of the test resamples have a clear deficit at every seed
        self.assertAlmostEqual(expected_undefined_fraction(gains), 7 / 54, places=12)
        self.assertEqual(expected_undefined_fraction(np.full((3, 4), -0.02)), 0.0)
        self.assertEqual(expected_undefined_fraction(np.full((2, 4), 0.001)), 1.0)
        # Two seeds at +0.02 and -0.02: the two mixed redraws of four cancel to 0.
        self.assertEqual(expected_undefined_fraction(np.array([[0.02] * 3, [-0.02] * 3])), 0.5)
        # One seed: nothing to redraw, the fraction of test resamples within delta.
        self.assertEqual(expected_undefined_fraction(np.array([[0.001, 0.02, -0.004, -0.006]])), 0.5)
        # |gain| equal to delta is defined, as in replay_share (exactly representable values).
        self.assertEqual(expected_undefined_fraction(np.array([[0.5, -0.5, 0.25, -0.375]]), delta=0.5), 0.5)


class ReadingsByShiftTest(unittest.TestCase):
    def row(self, domain: str, comparison: str = "adapt_matched - FT", **values) -> dict:
        return {"domain": domain, "comparison": comparison, "seeds": 3, "seed_list": [0, 1, 2], "left": 0.245, "right": 0.252,
                "difference": -0.007, "ci_low": -0.009, "ci_high": -0.005, **values}

    def test_every_target_shift_has_an_entry_with_its_seeds(self):
        rows = [self.row("compound", ft_ahead_stands=True, reading=STANDS),
                self.row("compound", "replay_matched - FT", difference=-0.5),  # never read: not the pre-registered row
                self.row("lighting", seeds=1, seed_list=[0], difference=-0.002, ci_high=0.001, ft_ahead_stands=False,
                         reading=WITHDRAWN),
                self.row("weather", "adapt_original - FT")]
        readings = readings_by_shift(rows)
        self.assertEqual(list(readings), ["compound", "weather", "lighting"])
        self.assertEqual(readings["compound"], {"difference": -0.007, "ci_low": -0.009, "ci_high": -0.005, "seeds": 3,
                                                "seed_list": [0, 1, 2], "ft_ahead_stands": True, "reading": STANDS})
        self.assertEqual(readings["lighting"], {"difference": -0.002, "ci_low": -0.009, "ci_high": 0.001, "seeds": 1,
                                                "seed_list": [0], "ft_ahead_stands": False, "reading": WITHDRAWN})
        # Weather has no adapt_matched - FT row: it keeps its place, and it is neither "stands" nor "withdrawn".
        self.assertEqual(readings["weather"], {"difference": None, "ci_low": None, "ci_high": None, "seeds": 0, "seed_list": [],
                                               "ft_ahead_stands": None, "reading": NOT_EVALUATED})
        empty = readings_by_shift([])
        self.assertEqual({shift: found["reading"] for shift, found in empty.items()},
                         dict.fromkeys(("compound", "weather", "lighting"), NOT_EVALUATED))
        empty["weather"]["seed_list"].append(9)  # the missing entries do not share one list
        self.assertEqual(empty["compound"]["seed_list"], [])


class StaleTablesTest(unittest.TestCase):
    def test_tables_without_rows_that_an_earlier_run_left_in_the_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            tables = {"rq5_delta": [{"domain": "weather"}], "rq5_matched_passes": [], "rq5_replay_share_matched": []}
            self.assertEqual(stale_tables(directory, tables), [])
            self.assertEqual(stale_tables(directory / "not_created_yet", tables), [])
            for name in ("rq5_delta.csv", "rq5_replay_share_matched.csv", "rq5_weather.png"):
                (directory / name).write_text("left by an earlier run", encoding="utf-8")
            self.assertEqual(stale_tables(directory, tables), ["rq5_replay_share_matched.csv"])  # rq5_delta is rewritten
            (directory / "rq5_matched_passes.csv").write_text("left by an earlier run", encoding="utf-8")
            self.assertEqual(stale_tables(directory, tables), ["rq5_matched_passes.csv", "rq5_replay_share_matched.csv"])
            self.assertEqual(stale_tables(directory, {name: [{"domain": "weather"}] for name in tables}), [])


class SourceTestRowsTest(unittest.TestCase):
    def test_seed_mean_on_the_source_test_set_and_its_change(self):
        groups = [("replay", 10, group({0: 0.347, 1: 0.346, 2: 0.345})), ("replay", 100, {}),
                  ("adapt_matched", 1000, group({0: 0.344, 1: 0.348}))]
        rows = source_test_rows(groups, 0.35)
        self.assertEqual([(r["stage"], r["budget"], r["seeds"]) for r in rows], [("replay", 10, 3), ("adapt_matched", 1000, 2)])
        self.assertEqual(list(rows[0]), ["stage", "budget", "seeds", "source_reliable_mean", "change_vs_source_model"])
        self.assertAlmostEqual(rows[0]["source_reliable_mean"], 0.346)
        self.assertAlmostEqual(rows[0]["change_vs_source_model"], -0.004)
        self.assertAlmostEqual(rows[1]["source_reliable_mean"], 0.346)
        self.assertAlmostEqual(rows[1]["change_vs_source_model"], -0.004)


class MatchedArmsVersusZeroShotTest(unittest.TestCase):
    def test_matched_arms_take_new_children_and_leave_the_earlier_rows_alone(self):
        self.assertEqual(STREAM_ARMS, ("FT", "adapt", "replay", "adapt_matched", "replay_matched"))
        zero = entry(0.20, np.full(300, 0.20))
        ft = {1000: group({s: 0.20 + 0.01 * s for s in range(6)}, 300)}
        matched = {"adapt_matched": {1000: group({s: 0.22 + 0.004 * s for s in range(3)}, 300)},
                   "replay_matched": {1000: group({s: 0.23 for s in range(3)}, 300)}}
        alone = versus_zero_shot(zero, {"FT": ft}, [1000], (3, 0))[0]
        rows = versus_zero_shot(zero, {**matched, "FT": ft}, [1000], (3, 0))
        self.assertEqual([(r["arm"], r["budget"], r["seeds"]) for r in rows],
                         [("adapt_matched", 1000, 3), ("replay_matched", 1000, 3), ("FT", 1000, 6)])
        self.assertEqual(rows[2], alone)
        self.assertAlmostEqual(rows[0]["mAP"], 0.224)
        self.assertAlmostEqual(rows[0]["gain_vs_zero_shot"], 0.024)
        self.assertAlmostEqual(rows[1]["gain_vs_zero_shot"], 0.03)
        self.assertAlmostEqual(rows[1]["ci_low"], 0.03)
        expected = paired_delta(matched["adapt_matched"][1000], {}, "reliable", row_rng(3, 0, 3, 1000), zero=zero)
        self.assertEqual((rows[0]["ci_low"], rows[0]["ci_high"]), (expected["ci_low"], expected["ci_high"]))
        self.assertLess(rows[0]["ci_low"], rows[0]["ci_high"])


class PlotTest(unittest.TestCase):
    def curves(self) -> dict:
        return {"FT": [(0, 0.22, 0.21, 0.23), (10, 0.17, 0.16, 0.18), (1000, 0.25, 0.24, 0.26)],
                "adapt": [(0, 0.23, 0.22, 0.24), (10, 0.23, 0.22, 0.24), (1000, 0.24, 0.23, 0.25)],
                "replay": [(0, 0.221, None, None), (10, 0.22, None, None), (1000, 0.24, None, None)], "ceiling": 0.257}

    def drawn(self, curves: dict) -> tuple[list[str], dict[str, dict]]:
        """Legend labels in order, and per label the x and y data and the marker colours of its line."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_hex
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(plt, "close") as close:
            plot({"compound": curves}, Path(folder))
            self.assertEqual([path.name for path in Path(folder).iterdir()], ["rq5_compound.png"])
            figure = close.call_args.args[0]
        handles, labels = figure.axes[0].get_legend_handles_labels()
        lines = {label: {"x": list(handle.get_xdata()), "y": list(handle.get_ydata()), "linestyle": handle.get_linestyle(),
                         "face": to_hex(handle.get_markerfacecolor()), "edge": to_hex(handle.get_markeredgecolor())}
                 for handle, label in zip(handles, labels)}
        plt.close(figure)
        return labels, lines

    def test_without_revision_runs_the_three_original_lines(self):
        labels, lines = self.drawn(self.curves())
        self.assertEqual(labels, ["fine-tune (FT)", "ALDI++ adapt", "replay control (seed 0)"])
        self.assertEqual(lines["replay control (seed 0)"]["y"], [0.221, 0.22, 0.24])

    def test_matched_arms_are_hollow_markers_of_the_arm_colour_at_n_1000(self):
        merged = "replay control (seed 0; seed mean at N = 10)"
        labels, lines = self.drawn({**self.curves(), "replay_label": merged,
                                    "matched": {"adapt_matched": 0.245, "replay_matched": 0.246}})
        adapt, replay = "ALDI++ adapt, matched passes (3,000 updates)", "replay control, matched passes (3,000 updates)"
        self.assertEqual(labels, ["fine-tune (FT)", "ALDI++ adapt", merged, adapt, replay])
        self.assertEqual((lines[adapt]["x"], lines[adapt]["y"]), ([1000], [0.245]))
        self.assertEqual((lines[replay]["x"], lines[replay]["y"]), ([1000], [0.246]))
        for label, arm in ((adapt, "adapt"), (replay, "replay")):
            self.assertEqual((lines[label]["face"], lines[label]["edge"], lines[label]["linestyle"]),
                             ("#ffffff", COLORS[arm], "None"))
        self.assertEqual(lines["ALDI++ adapt"]["face"], COLORS["adapt"])  # the original markers stay filled


TARGETS = ("compound", "weather", "lighting")
FROZEN_TABLES = ["rq5_attribution.csv", "rq5_delta.csv", "rq5_delta_per_class.csv", "rq5_equivalent_budget.csv",
                 "rq5_forgetting.csv"]  # the 2026-11-03 tables: original runs only
ORIGINAL_FILES = FROZEN_TABLES + ["rq5_replay_share.csv", "rq5_replay_share_vs_source.csv", "rq5_vs_zero_shot.csv",
                                  "summary.json"]
REVISION_FILES = ["rq5_forgetting_revision.csv", "rq5_matched_passes.csv", "rq5_matched_passes_per_class.csv",
                  "rq5_replay_share_matched.csv"]
SUMMARY_KEYS = ["rules", "delta_threshold", "domains", "attribution", "replay_share", "replay_share_vs_source_model",
                "claim_rule", "headline_supported_all_shifts", "share_majority_all_shifts", "replay_seeds_available",
                "replay_seeds_required", "direction_all_shifts", "largest_undefined_fraction", "test_resamples"]


class MainTest(unittest.TestCase):
    """main() on three synthetic scored trees: the wiring between the tables that the pure functions cannot see.

    Every class of a run scores a level plus the offset of the class, and every bootstrap vector is that score plus
    test-resample noise shared by all runs scored on one test domain, in all three trees, so the noise cancels in every
    paired difference. Per shift, FT at N and seed s scores zero + ft_gain[N] + 0.002 s, adapt is FT + adapt_delta[N],
    replay is FT + replay_part[N][s] (seed 0 in the RQ5 tree, seeds 1 and 2 at N = 10 and 100 in the revision tree) and
    the matched-pass arms are FT + matched_delta[shift][stage] at 3,000 updates. Set folders are numbered and carry no
    run name, and the revision tree repeats (stage, domain, budget, seed) of RQ5 runs: only the run fields tell them apart.
    """
    images = {"source": 2650, "compound": 756, "weather": 933, "lighting": 4869}
    offset = [0.001 * index for index in range(len(CLASSES))]
    reliable_offset = sum(0.001 * CLASSES.index(name) for name in RELIABLE_CLASSES) / len(RELIABLE_CLASSES)
    source_level = 0.35
    zero = {"compound": 0.22, "weather": 0.32, "lighting": 0.24}
    ft_gain = {10: -0.05, 100: 0.0, 1000: 0.03}
    adapt_delta = {10: 0.06, 100: 0.016, 1000: -0.012}
    replay_part = {10: (0.048, 0.051, 0.054), 100: (0.012, 0.014, 0.016), 1000: (-0.009,)}
    matched_delta = {"compound": {"adapt": -0.008, "replay": -0.006}, "weather": {"adapt": -0.002, "replay": -0.0015},
                     "lighting": {"adapt": 0.003, "replay": 0.001}}
    matched_source = {"adapt": -0.005, "replay": -0.001}

    def ft(self, domain: str, budget: int, seed: int) -> float:
        return self.zero[domain] + self.ft_gain[budget] + 0.002 * seed

    def write(self, tree: Path, noise: dict, stage: str, run_domain: str, budget, seed: int, test_domain: str, level: float,
              updates: int = 2000, epochs: int | None = None, bootstrap_seed: int = 20261030) -> None:
        folder = tree / f"{len(list(tree.glob('*'))):03d}" / test_domain
        folder.mkdir(parents=True)
        per_class = [level + value for value in self.offset]
        metrics = {"run": {"stage": stage, "domain": run_domain, "budget": str(budget), "seed": seed, "epochs": epochs,
                           "total_updates": updates},
                   "domain": test_domain, "images": self.images[test_domain], "AP50_95": float(np.mean(per_class)),
                   "reliable_AP50_95": float(np.mean([per_class[CLASSES.index(name)] for name in RELIABLE_CLASSES])),
                   "per_class": [{"class": name, "AP50_95": value} for name, value in zip(CLASSES, per_class)],
                   "bootstrap": {"class_order": list(CLASSES), "seed": bootstrap_seed, "resamples": len(noise[test_domain])}}
        (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        np.save(folder / "bootstrap_class_ap.npy", np.asarray(per_class) + noise[test_domain])

    def build(self, root: Path, replicates: int = 80) -> tuple[dict[str, Path], dict]:
        rng = np.random.default_rng(42)
        noise = {domain: rng.normal(0, 0.004, (replicates, len(CLASSES))) for domain in self.images}
        trees = {name: root / name for name in ("campaign", "rq5", "revision")}
        for tree in trees.values():
            tree.mkdir(parents=True)
        campaign, rq5, revision = trees["campaign"], trees["rq5"], trees["revision"]
        for domain in self.images:
            self.write(campaign, noise, "source", "source", "full", 0, domain, self.zero.get(domain, self.source_level),
                       updates=29412, epochs=12)
        for domain in TARGETS:
            zero = self.zero[domain]
            for seed, shift in enumerate((-0.005, 0.005)):
                self.write(campaign, noise, "fine-tune", domain, "full", seed, domain, zero + 0.04 + shift, updates=2607,
                           epochs=3)
                # A 12-epoch ceiling of the revision: same stage, domain, budget and seed as the campaign run, not read here.
                self.write(revision, noise, "fine-tune", domain, "full", seed, domain, zero + 0.06, updates=10428, epochs=12)
                self.write(rq5, noise, "adapt", domain, 0, seed, domain, zero + 0.01)
                self.write(rq5, noise, "adapt", domain, 0, seed, "source", self.source_level)
            self.write(rq5, noise, "replay", "source", 0, 0, domain, zero + 0.004)
            for budget in (10, 100, 1000):
                for seed in (0, 1, 2):
                    level = self.ft(domain, budget, seed)
                    length = max(500, 3 * budget // 2)
                    self.write(campaign, noise, "fine-tune", domain, budget, seed, domain, level, updates=length, epochs=3)
                    self.write(campaign, noise, "fine-tune", domain, budget, seed, "source", self.source_level - 0.02,
                               updates=length, epochs=3)
                    self.write(rq5, noise, "adapt", domain, budget, seed, domain, level + self.adapt_delta[budget])
                    self.write(rq5, noise, "adapt", domain, budget, seed, "source", self.source_level - 0.002)
                for seed, part in enumerate(self.replay_part[budget]):  # seed 0: the RQ5 grid; seeds 1 and 2: the revision
                    tree = rq5 if seed == 0 else revision
                    self.write(tree, noise, "replay", domain, budget, seed, domain, self.ft(domain, budget, seed) + part)
                    self.write(tree, noise, "replay", domain, budget, seed, "source", self.source_level - 0.003 - 0.001 * seed)
            for stage in ("adapt", "replay"):  # matched passes: stage, domain, budget and seed of an RQ5 run, 3,000 updates
                for seed in (0, 1, 2):
                    self.write(revision, noise, stage, domain, 1000, seed, domain,
                               self.ft(domain, 1000, seed) + self.matched_delta[domain][stage], updates=3000)
                    self.write(revision, noise, stage, domain, 1000, seed, "source",
                               self.source_level + self.matched_source[stage], updates=3000)
        return trees, noise

    def run_main(self, trees: dict[str, Path], output: Path, *options: str, revision: bool = False, plots: bool = False) -> str:
        arguments = ["--ft-scored", str(trees["campaign"]), "--da-scored", str(trees["rq5"]), "--output-dir", str(output),
                     *options]
        if not plots:
            arguments.append("--no-plots")
        if revision:
            arguments += ["--revision-scored", str(trees["revision"])]
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            main(arguments)
        return printed.getvalue()

    def drop(self, tree: Path, unwanted) -> int:
        """Remove from a scored tree every set whose run fields satisfy unwanted; the number of sets removed."""
        removed = 0
        for path in sorted(tree.glob("*/*/metrics.json")):
            if unwanted(json.loads(path.read_text(encoding="utf-8"))["run"]):
                shutil.rmtree(path.parent.parent)
                removed += 1
        return removed

    def table(self, path: Path) -> list[dict]:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def outputs(self, directory: Path) -> dict:
        tables = {path.name: self.table(path) for path in directory.glob("*.csv")}
        return {**tables, "bytes": {path.name: path.read_bytes() for path in directory.iterdir()},
                "summary": json.loads((directory / "summary.json").read_text(encoding="utf-8"))}

    def both(self) -> tuple[dict, dict, str, str]:
        """Outputs and printed text of main() without and with --revision-scored on the same trees."""
        with tempfile.TemporaryDirectory() as folder:
            trees, _ = self.build(Path(folder))
            printed_before = self.run_main(trees, Path(folder) / "before")
            printed_after = self.run_main(trees, Path(folder) / "after", revision=True)
            return self.outputs(Path(folder) / "before"), self.outputs(Path(folder) / "after"), printed_before, printed_after

    def test_file_list_and_the_2026_11_03_tables_with_and_without_revision_runs(self):
        before, after, printed_before, _ = self.both()
        self.assertEqual(sorted(before["bytes"]), sorted(ORIGINAL_FILES))
        self.assertEqual(sorted(after["bytes"]), sorted(ORIGINAL_FILES + REVISION_FILES))
        for name in FROZEN_TABLES:
            self.assertEqual(after["bytes"][name], before["bytes"][name], name)
            self.assertGreater(len(before[name]), 0, name)
        # The attribution is seed 0 of the original arms in both runs: replay - FT at N = 10 is 0.048, not the 0.051 of
        # the three-seed mean, and the original adapt arm at N = 1,000 is 0.012 below FT, not the matched-pass value.
        for rows in (before["rq5_attribution.csv"], after["rq5_attribution.csv"]):
            self.assertEqual({row["seed"] for row in rows}, {"0"})
            parts = {(row["domain"], row["budget"], row["comparison"]): float(row["difference"]) for row in rows}
            for domain in TARGETS:
                self.assertAlmostEqual(parts[(domain, "10", "replay - FT")], 0.048, places=9)
                self.assertAlmostEqual(parts[(domain, "100", "replay - FT")], 0.012, places=9)
                self.assertAlmostEqual(parts[(domain, "1000", "adapt - FT")], -0.012, places=9)
                self.assertAlmostEqual(parts[(domain, "1000", "replay - FT")], -0.009, places=9)
        deltas = {(row["domain"], row["budget"]): row for row in after["rq5_delta.csv"]}
        for domain in TARGETS:  # three original seeds at N = 1,000, none of the matched-pass runs
            self.assertEqual(deltas[(domain, "1000")]["seeds"], "3")
            self.assertAlmostEqual(float(deltas[(domain, "1000")]["delta"]), -0.012, places=9)
            self.assertAlmostEqual(float(deltas[(domain, "1000")]["mAP_DA"]),
                                   self.zero[domain] + 0.03 + 0.002 - 0.012 + self.reliable_offset, places=9)
        forgetting = {(row["domain"], row["stage"], row["budget"]): row for row in after["rq5_forgetting.csv"]}
        self.assertEqual(forgetting[("lighting", "replay", "10")]["seeds"], "1")  # the original replay arm only
        self.assertEqual(list(before["summary"]), SUMMARY_KEYS)
        self.assertEqual(list(after["summary"]), SUMMARY_KEYS + ["matched_pass_rule", "matched_passes", "matched_pass_reading",
                                                                 "replay_share_matched", "revision_scored_entries",
                                                                 "ceiling_reference"])
        for key in ("rules", "delta_threshold", "domains", "attribution", "claim_rule", "replay_seeds_required"):
            self.assertEqual(after["summary"][key], before["summary"][key], key)
        # "ceiling" stays the 3-epoch full-pool mean (0.04 above mAP(0) here), not the 12-epoch runs of the revision tree
        # (0.06 above), and the summary says which of the two it is.
        for domain in TARGETS:
            self.assertAlmostEqual(after["summary"]["domains"][domain]["recovery_denominator"], 0.04, places=9)
        for words in ("seed mean of the 3-epoch full-pool fine-tune", "not mAP_c of DECISIONS 2026-11-21, 'Ceiling'",
                      "does not read the 12-epoch runs"):
            self.assertIn(words, after["summary"]["ceiling_reference"])
        self.assertEqual(before["summary"]["test_resamples"]["scored_sets"], 151)
        self.assertEqual(after["summary"]["test_resamples"]["scored_sets"], 217)
        self.assertEqual(after["summary"]["revision_scored_entries"], {"replay_seeds": 24, "matched": 36})
        self.assertNotIn("Matched passes", printed_before)
        self.assertTrue(printed_before.rstrip().splitlines()[-1].startswith("  headline at N=100:"))

    def test_merged_replay_arm_feeds_the_shares_the_headline_and_the_gain_over_zero_shot(self):
        before, after, _, printed_after = self.both()
        for name in ("rq5_replay_share.csv", "rq5_replay_share_vs_source.csv"):
            old = {(row["domain"], row["budget"]): row for row in before[name]}
            new = {(row["domain"], row["budget"]): row for row in after[name]}
            self.assertEqual(list(new), list(old))
            for key, row in new.items():  # N = 0 and N = 1,000 keep their original runs; N = 10 and 100 gain two seeds
                if key[1] in ("10", "100"):
                    self.assertEqual((old[key]["seeds"], row["seeds"], row["seed_list"]), ("1", "3", "[0, 1, 2]"), key)
                else:
                    self.assertEqual(row, old[key], key)
                    self.assertEqual((row["seeds"], row["seed_list"]), ("1", "[0]"), key)
        shares = {(row["domain"], row["budget"]): row for row in after["rq5_replay_share.csv"]}
        old_shares = {(row["domain"], row["budget"]): row for row in before["rq5_replay_share.csv"]}
        for domain in TARGETS:
            for budget, gain, part, one_seed in (("10", 0.06, 0.051, 0.8), ("100", 0.016, 0.014, 0.75)):
                row = shares[(domain, budget)]
                self.assertAlmostEqual(float(row["gain"]), gain, places=9)
                self.assertAlmostEqual(float(row["replay_part"]), part, places=9)
                self.assertAlmostEqual(float(row["share"]), part / gain, places=9)  # 0.85 and 0.875
                self.assertAlmostEqual(float(old_shares[(domain, budget)]["share"]), one_seed, places=9)
                self.assertEqual((row["direction"], row["majority_not_adaptation"], row["quotable"]),
                                 ("advantage", "True", "True"))
            self.assertAlmostEqual(float(shares[(domain, "1000")]["share"]), 0.75, places=9)  # seed 0: -0.009 / -0.012
            self.assertAlmostEqual(float(shares[(domain, "0")]["share"]), 0.4, places=9)
        for summary, seeds, supported in ((before["summary"], 1, False), (after["summary"], 3, True)):
            self.assertEqual(summary["replay_seeds_available"], {"10": seeds, "100": seeds})
            self.assertEqual(summary["headline_supported_all_shifts"], {"10": supported, "100": supported})
            self.assertEqual(summary["share_majority_all_shifts"], {"10": True, "100": True})
            self.assertEqual(summary["direction_all_shifts"], {"10": "advantage", "100": "advantage"})
        self.assertEqual([row["seeds"] for row in after["summary"]["replay_share"] if row["budget"] in (10, 100)], [3] * 6)
        self.assertEqual(printed_after.rstrip().splitlines()[-1],
                         "Headline ('most of the gain over fine-tuning is not adaptation') with the merged replay arm: "
                         "N=10 supported: True (replay seeds 3); N=100 supported: True (replay seeds 3)")
        # Against no fine-tuning: the FT and adapt rows are untouched, replay at N = 10 and 100 has three seeds, and the
        # matched arms follow the three original arms of their shift.
        old = {(row["domain"], row["arm"], row["budget"]): row for row in before["rq5_vs_zero_shot.csv"]}
        new = {(row["domain"], row["arm"], row["budget"]): row for row in after["rq5_vs_zero_shot.csv"]}
        self.assertEqual([key for key in new if key in old], list(old))
        self.assertEqual([key for key in new if key not in old],
                         [(domain, arm, "1000") for domain in TARGETS for arm in ("adapt_matched", "replay_matched")])
        self.assertEqual([row["arm"] for row in after["rq5_vs_zero_shot.csv"] if row["domain"] == "weather"],
                         ["FT"] * 3 + ["adapt"] * 4 + ["replay"] * 4 + ["adapt_matched", "replay_matched"])
        for key, row in old.items():
            if key[1] == "replay" and key[2] in ("10", "100"):
                self.assertEqual((row["seeds"], new[key]["seeds"]), ("1", "3"), key)
            else:
                self.assertEqual(new[key], row, key)
        self.assertLess(float(old[("lighting", "FT", "10")]["ci_low"]), float(old[("lighting", "FT", "10")]["ci_high"]))
        for domain in TARGETS:
            self.assertAlmostEqual(float(old[(domain, "replay", "10")]["gain_vs_zero_shot"]), -0.002, places=9)  # seed 0
            self.assertAlmostEqual(float(new[(domain, "replay", "10")]["gain_vs_zero_shot"]), 0.003, places=9)  # three seeds
            self.assertAlmostEqual(float(new[(domain, "replay", "100")]["gain_vs_zero_shot"]), 0.016, places=9)
            self.assertAlmostEqual(float(new[(domain, "replay", "1000")]["gain_vs_zero_shot"]), 0.021, places=9)  # seed 0
            for arm, stage in (("adapt_matched", "adapt"), ("replay_matched", "replay")):
                row = new[(domain, arm, "1000")]
                self.assertEqual(row["seeds"], "3")
                self.assertAlmostEqual(float(row["gain_vs_zero_shot"]), 0.032 + self.matched_delta[domain][stage], places=9)
                self.assertAlmostEqual(float(row["mAP_0"]), self.zero[domain] + self.reliable_offset, places=9)

    def test_matched_pass_tables_and_their_reading_per_shift(self):
        _, after, _, printed = self.both()
        rows = {(row["domain"], row["comparison"]): row for row in after["rq5_matched_passes.csv"]}
        self.assertEqual(list(after["rq5_matched_passes.csv"][0]),
                         ["domain", "comparison", "seeds", "seed_list", "left", "right", "difference", "ci_low", "ci_high",
                          "ft_ahead_stands", "reading"])
        self.assertEqual(list(rows), [(domain, f"{left} - {right}") for domain in TARGETS for left, right in MATCHED_COMPARISONS])
        readings = {"compound": ("True", STANDS), "weather": ("False", WITHDRAWN), "lighting": ("False", WITHDRAWN)}
        for domain in TARGETS:
            adapt, replay = self.matched_delta[domain]["adapt"], self.matched_delta[domain]["replay"]
            expected = {"adapt_matched - FT": adapt, "replay_matched - FT": replay,
                        "adapt_matched - replay_matched": adapt - replay, "adapt_matched - adapt_original": adapt + 0.012,
                        "replay_matched - replay_original": replay + 0.009, "adapt_original - FT": -0.012}
            for comparison, difference in expected.items():
                row = rows[(domain, comparison)]
                one_seed = comparison == "replay_matched - replay_original"  # the original replay at N = 1,000 is seed 0 only
                self.assertEqual((row["seeds"], row["seed_list"]), ("1", "[0]") if one_seed else ("3", "[0, 1, 2]"), comparison)
                for name in ("difference", "ci_low", "ci_high"):  # constant over seeds and resamples: the interval is the point
                    self.assertAlmostEqual(float(row[name]), difference, places=9, msg=f"{domain} {comparison} {name}")
                self.assertAlmostEqual(float(row["left"]) - float(row["right"]), difference, places=9)
                if comparison != "adapt_matched - FT":
                    self.assertEqual((row["ft_ahead_stands"], row["reading"]), ("", ""))
            pre_registered = rows[(domain, "adapt_matched - FT")]
            self.assertEqual((pre_registered["ft_ahead_stands"], pre_registered["reading"]), readings[domain])
            ft_mean = self.zero[domain] + 0.032 + self.reliable_offset  # seed mean of FT at N = 1,000
            self.assertAlmostEqual(float(pre_registered["left"]), ft_mean + adapt, places=9)
            self.assertAlmostEqual(float(pre_registered["right"]), ft_mean, places=9)
            summary = after["summary"]["matched_pass_reading"][domain]
            self.assertEqual(list(summary),
                             ["difference", "ci_low", "ci_high", "seeds", "seed_list", "ft_ahead_stands", "reading"])
            self.assertEqual((str(summary["ft_ahead_stands"]), summary["reading"]), readings[domain])
            self.assertEqual((summary["seeds"], summary["seed_list"]), (3, [0, 1, 2]))
            self.assertAlmostEqual(summary["difference"], adapt, places=9)
        self.assertEqual(list(after["summary"]["matched_pass_reading"]), list(TARGETS))
        self.assertNotIn("not evaluated", printed)
        self.assertEqual([(row["domain"], row["comparison"]) for row in after["summary"]["matched_passes"]], list(rows))
        self.assertIn("upper 95% bound is below 0", after["summary"]["matched_pass_rule"])
        self.assertIn("0.9996^3000 = 0.30", after["summary"]["matched_pass_rule"])
        self.assertIn("  compound adapt_matched - FT: -0.0080 [-0.0080, -0.0080]; seeds [0, 1, 2]; " + STANDS, printed)
        self.assertIn("  weather adapt_matched - FT: -0.0020 [-0.0020, -0.0020]; seeds [0, 1, 2]; " + WITHDRAWN, printed)
        self.assertIn("  lighting adapt_matched - FT: +0.0030 [+0.0030, +0.0030]; seeds [0, 1, 2]; " + WITHDRAWN, printed)
        # Per class: the same six comparisons for each reliable class, no reading.
        per_class = after["rq5_matched_passes_per_class.csv"]
        self.assertEqual(list(per_class[0]), ["domain", "class", "comparison", "seeds", "seed_list", "left", "right",
                                              "difference", "ci_low", "ci_high"])
        self.assertEqual([(row["domain"], row["class"], row["comparison"]) for row in per_class],
                         [(domain, name, f"{left} - {right}") for domain in TARGETS for name in RELIABLE_CLASSES
                          for left, right in MATCHED_COMPARISONS])
        car = {(row["domain"], row["comparison"]): row for row in per_class if row["class"] == "car"}
        self.assertAlmostEqual(float(car[("compound", "adapt_matched - FT")]["difference"]), -0.008, places=9)
        self.assertAlmostEqual(float(car[("compound", "adapt_matched - FT")]["left"]),
                               0.22 + 0.032 - 0.008 + 0.001 * CLASSES.index("car"), places=9)
        self.assertEqual(car[("lighting", "replay_matched - replay_original")]["seeds"], "1")

    def test_matched_pass_share_and_source_test_rows(self):
        before, after, _, printed = self.both()
        shares = {(row["domain"], row["reference"]): row for row in after["rq5_replay_share_matched.csv"]}
        self.assertEqual(list(shares), [(domain, reference) for domain in TARGETS for reference in ("FT", "source model")])
        self.assertEqual(list(after["rq5_replay_share_matched.csv"][0]),
                         ["domain", "budget", "labeled_budget", "reference"] + list(before["rq5_replay_share.csv"][0])[3:]
                         + ["expected_undefined_fraction"])
        for row in shares.values():
            self.assertEqual((row["budget"], row["labeled_budget"], row["seeds"], row["seed_list"]),
                             ("1000", "True", "3", "[0, 1, 2]"))
        # The gain is the same at every seed and test resample here, so every redraw is on one side of delta: -0.008 for
        # compound against FT (never undefined), -0.002 and +0.003 for weather and lighting (always), and at least 0.024
        # against the source model (never).
        self.assertEqual({key: float(row["expected_undefined_fraction"]) for key, row in shares.items()},
                         {("compound", "FT"): 0.0, ("weather", "FT"): 1.0, ("lighting", "FT"): 1.0,
                          **{(domain, "source model"): 0.0 for domain in TARGETS}})
        compound = shares[("compound", "FT")]  # a deficit of 0.008, of which the replay arm has 0.006
        self.assertAlmostEqual(float(compound["gain"]), -0.008, places=9)
        self.assertAlmostEqual(float(compound["replay_part"]), -0.006, places=9)
        self.assertAlmostEqual(float(compound["share"]), 0.75, places=9)
        self.assertEqual((compound["direction"], compound["quotable"], compound["undefined_replicates"]),
                         ("deficit", "True", "0"))
        source = shares[("compound", "source model")]  # gains 0.024 and 0.026 over mAP(0)
        self.assertAlmostEqual(float(source["gain"]), 0.024, places=9)
        self.assertAlmostEqual(float(source["share"]), 0.026 / 0.024, places=9)
        self.assertEqual((source["direction"], source["quotable"]), ("advantage", "True"))
        for domain in ("weather", "lighting"):  # |adapt_matched - FT| = 0.002 and 0.003: below delta, no share
            row = shares[(domain, "FT")]
            self.assertEqual((row["share"], row["share_ci_low"], row["quotable"], row["undefined_replicates"], row["replicates"]),
                             ("", "", "False", "80", "80"))
        self.assertEqual(len(after["summary"]["replay_share_matched"]), 6)
        self.assertIn("  weather against FT: share undefined; deficit -0.0020; 80/80 undefined (expected 1.000); "
                      "seeds [0, 1, 2] ~", printed)
        self.assertIn("  compound against FT: share 0.750 [0.750, 0.750]; deficit -0.0080; 0/80 undefined (expected 0.000); "
                      "seeds [0, 1, 2]\n", printed)
        # Source test set: the merged replay arm at N = 10 and 100, then the two matched arms, in the rq5_forgetting format.
        rows = after["rq5_forgetting_revision.csv"]
        self.assertEqual(list(rows[0]), list(before["rq5_forgetting.csv"][0]))
        self.assertEqual([(row["domain"], row["stage"], row["budget"], row["seeds"]) for row in rows],
                         [(domain, stage, budget, "3") for domain in TARGETS
                          for stage, budget in (("replay", "10"), ("replay", "100"), ("adapt_matched", "1000"),
                                                ("replay_matched", "1000"))])
        for row in rows:
            change = {"replay": -0.004, "adapt_matched": -0.005, "replay_matched": -0.001}[row["stage"]]
            self.assertAlmostEqual(float(row["change_vs_source_model"]), change, places=9)
            self.assertAlmostEqual(float(row["source_reliable_mean"]), self.source_level + self.reliable_offset + change,
                                   places=9)

    def test_revision_tables_draw_from_their_own_streams(self):
        with tempfile.TemporaryDirectory() as folder:
            trees, _ = self.build(Path(folder))
            with (mock.patch("analyze_rq5.matched_pass_rows", wraps=matched_pass_rows) as passes,
                  mock.patch("analyze_rq5.matched_share_rows", wraps=matched_share_rows) as shares,
                  mock.patch("analyze_rq5.versus_zero_shot", wraps=versus_zero_shot) as zero_shot):
                self.run_main(trees, Path(folder) / "analysis", "--seed", "11", revision=True)
        reliable = [call for call in passes.call_args_list if call.kwargs.get("reading")]
        per_class = [call for call in passes.call_args_list if not call.kwargs.get("reading")]
        self.assertEqual([call.args[1] for call in reliable], [(16, 0), (16, 1), (16, 2)])  # seed + 5, shift
        self.assertEqual([call.args[1] for call in per_class], [(17, shift, index) for shift in range(3) for index in range(4)])
        self.assertEqual([call.kwargs["key"] for call in per_class], list(RELIABLE_CLASSES) * 3)
        self.assertEqual([call.args[4] for call in shares.call_args_list], [(18, 0), (18, 1), (18, 2)])  # seed + 7, shift
        # Against no fine-tuning: per shift one call for the three original arms and one for the two matched arms, both on
        # the stream (seed + 3, shift) of that shift, where the arm index and N pick the child.
        self.assertEqual([(list(call.args[1]), call.args[2], call.args[3]) for call in zero_shot.call_args_list],
                         [(arms, budgets, (14, shift)) for shift in range(3)
                          for arms, budgets in ((["FT", "adapt", "replay"], [0, 10, 100, 1000]),
                                                (["adapt_matched", "replay_matched"], [1000]))])

    def test_curves_handed_to_the_plot_with_and_without_revision_runs(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch("analyze_rq5.plot") as plotted:
            trees, _ = self.build(Path(folder))
            self.run_main(trees, Path(folder) / "before", plots=True)
            self.run_main(trees, Path(folder) / "after", revision=True, plots=True)
            self.run_main(trees, Path(folder) / "unplotted", revision=True)  # --no-plots: no third call
            self.assertEqual([call.args[1] for call in plotted.call_args_list], [Path(folder) / "before", Path(folder) / "after"])
        before, after = (call.args[0] for call in plotted.call_args_list)
        self.assertEqual((list(before), list(after)), (list(TARGETS), list(TARGETS)))
        for domain in TARGETS:
            old, new, level = before[domain], after[domain], self.zero[domain] + self.reliable_offset
            self.assertEqual(list(old), ["FT", "adapt", "replay", "ceiling"])
            self.assertEqual(list(new), ["FT", "adapt", "replay", "ceiling", "replay_label", "ceiling_label", "matched"])
            for key in ("FT", "adapt", "ceiling"):  # points and bands of the original arms: the same draws in both runs
                self.assertEqual(new[key], old[key], f"{domain} {key}")
            self.assertEqual([n for n, *_ in old["FT"]], [0, 10, 100, 1000])
            self.assertAlmostEqual(old["ceiling"], level + 0.04, places=9)
            # The replay line: seed 0 without the revision runs; with them the three-seed mean at N = 10 and 100, where
            # FT's seed mean is 0.002 above its seed 0 and the replay parts average 0.051 and 0.014 (seed 0: 0.048, 0.012).
            self.assertEqual([(n, low, high) for n, _, low, high in old["replay"]], [(n, None, None) for n in (0, 10, 100, 1000)])
            self.assertEqual([(n, low, high) for n, _, low, high in new["replay"]], [(n, None, None) for n in (0, 10, 100, 1000)])
            for rows, gains in ((old["replay"], (0.004, -0.002, 0.012, 0.021)), (new["replay"], (0.004, 0.003, 0.016, 0.021))):
                for (n, value, _, _), gain in zip(rows, gains):
                    self.assertAlmostEqual(value, level + gain, places=9, msg=f"{domain} replay N={n}")
            self.assertEqual(new["replay_label"], "replay control (seed 0; seed mean at N = 10, 100)")
            self.assertEqual(new["ceiling_label"], "FT full pool, 3 epochs")  # the dashed line is no longer the ceiling
            # The hollow markers are the 3,000-update arms (FT + matched_delta), not the original arms at N = 1,000, which
            # sit at FT - 0.012 (adapt, seed mean) and FT - 0.009 (replay, seed 0).
            self.assertEqual(list(new["matched"]), ["adapt_matched", "replay_matched"])
            for arm, stage in (("adapt_matched", "adapt"), ("replay_matched", "replay")):
                self.assertAlmostEqual(new["matched"][arm], level + 0.032 + self.matched_delta[domain][stage], places=9)
            self.assertAlmostEqual(dict((n, value) for n, value, _, _ in new["adapt"])[1000], level + 0.032 - 0.012, places=9)
            self.assertGreater(abs(new["matched"]["adapt_matched"] - (level + 0.020)), 0.003)
            self.assertGreater(abs(new["matched"]["replay_matched"] - (level + 0.021)), 0.003)

    def test_a_shift_without_matched_runs_is_reported_as_not_evaluated_and_the_seeds_go_with_the_reading(self):
        with tempfile.TemporaryDirectory() as folder:
            trees, _ = self.build(Path(folder))
            # Weather loses its matched-pass runs; compound keeps seed 0 only of adapt_matched (replay_matched keeps three).
            weather = self.drop(trees["revision"], lambda run: run["domain"] == "weather" and run["total_updates"] == 3000)
            compound = self.drop(trees["revision"], lambda run: (run["stage"], run["domain"], run["total_updates"])
                                 == ("adapt", "compound", 3000) and run["seed"] != 0)
            self.assertEqual((weather, compound), (12, 4))  # scored sets: two test domains per run
            printed = self.run_main(trees, Path(folder) / "analysis", revision=True)
            after = self.outputs(Path(folder) / "analysis")
        readings = after["summary"]["matched_pass_reading"]
        self.assertEqual(list(readings), list(TARGETS))
        self.assertEqual(readings["weather"], {"difference": None, "ci_low": None, "ci_high": None, "seeds": 0, "seed_list": [],
                                               "ft_ahead_stands": None, "reading": NOT_EVALUATED})
        self.assertIn(f"  weather adapt_matched - FT: {NOT_EVALUATED}\n", printed)
        self.assertEqual(printed.count("not evaluated"), 1)
        # The rule names no seed count: compound is read on its one paired seed, and says so.
        self.assertEqual([readings["compound"][key] for key in ("seeds", "seed_list", "ft_ahead_stands", "reading")],
                         [1, [0], True, STANDS])
        self.assertAlmostEqual(readings["compound"]["difference"], -0.008, places=9)
        self.assertIn("  compound adapt_matched - FT: -0.0080 [-0.0080, -0.0080]; seeds [0]; " + STANDS, printed)
        self.assertEqual([readings["lighting"][key] for key in ("seeds", "seed_list", "ft_ahead_stands", "reading")],
                         [3, [0, 1, 2], False, WITHDRAWN])
        rows = {(row["domain"], row["comparison"]): row for row in after["rq5_matched_passes.csv"]}
        self.assertEqual([comparison for domain, comparison in rows if domain == "weather"], ["adapt_original - FT"])
        self.assertEqual((rows[("compound", "adapt_matched - FT")]["seeds"], rows[("compound", "replay_matched - FT")]["seeds"]),
                         ("1", "3"))
        shares = {(row["domain"], row["reference"]): row["seeds"] for row in after["rq5_replay_share_matched.csv"]}
        self.assertEqual(shares, {("compound", "FT"): "1", ("compound", "source model"): "1", ("lighting", "FT"): "3",
                                  ("lighting", "source model"): "3"})
        self.assertEqual({row["domain"] for row in after["rq5_vs_zero_shot.csv"] if row["arm"] == "adapt_matched"},
                         {"compound", "lighting"})
        self.assertTrue(printed.rstrip().splitlines()[-1].startswith("Headline ('most of the gain over fine-tuning"))

    def test_tables_of_an_earlier_run_that_would_stay_in_the_output_directory_stop_the_script(self):
        with tempfile.TemporaryDirectory() as folder:
            trees, _ = self.build(Path(folder))
            output = Path(folder) / "analysis"
            self.run_main(trees, output, revision=True)
            first = {path.name: path.read_bytes() for path in output.iterdir()}
            self.assertEqual(sorted(first), sorted(ORIGINAL_FILES + REVISION_FILES))
            # Without the option the four revision tables have no rows and would stay next to a summary without them.
            message = (r"analysis holds tables of an earlier run that this run does not write: \['rq5_matched_passes.csv', "
                       r"'rq5_matched_passes_per_class.csv', 'rq5_replay_share_matched.csv', 'rq5_forgetting_revision.csv'\]")
            with self.assertRaisesRegex(ValueError, message):
                self.run_main(trees, output)
            self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, first)  # nothing was written
            # The same run again is no conflict and writes the same bytes.
            self.run_main(trees, output, revision=True)
            self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, first)
            # A revision directory without matched-pass runs has no matched share: that one table would be left over.
            self.assertEqual(self.drop(trees["revision"], lambda run: run["total_updates"] == 3000), 36)
            with self.assertRaisesRegex(ValueError, r"does not write: \['rq5_replay_share_matched.csv'\]"):
                self.run_main(trees, output, revision=True)
            self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, first)
            self.run_main(trees, Path(folder) / "fresh", revision=True)  # an empty directory takes any run
            self.assertEqual(sorted(path.name for path in (Path(folder) / "fresh").iterdir()),
                             sorted(set(ORIGINAL_FILES + REVISION_FILES) - {"rq5_replay_share_matched.csv"}))

    def test_revision_directory_that_cannot_be_read_as_revision_runs_stops_the_script(self):
        with tempfile.TemporaryDirectory() as folder:
            trees, noise = self.build(Path(folder))
            # An adapt run at N = 10 fits neither kind.
            self.write(trees["revision"], noise, "adapt", "weather", 10, 1, "weather", 0.3)
            misfit = "Revision adapt run is neither a replay seed .* budget '10', seed 1, total_updates 2000"
            with self.assertRaisesRegex(ValueError, misfit):
                self.run_main(trees, Path(folder) / "misfit", revision=True)
            # Only fine-tune runs: nothing to read.
            ceilings = Path(folder) / "ceilings"
            ceilings.mkdir()
            self.write(ceilings, noise, "fine-tune", "weather", "full", 0, "weather", 0.34, updates=11916, epochs=12)
            with self.assertRaisesRegex(ValueError, "No adapt or replay run among the scored sets under"):
                self.run_main({**trees, "revision": ceilings}, Path(folder) / "empty", revision=True)
            # Scored with another bootstrap seed: replicate t is not the same test resample.
            unpaired = Path(folder) / "unpaired"
            unpaired.mkdir()
            self.write(unpaired, noise, "replay", "weather", 10, 1, "weather", 0.3, bootstrap_seed=7)
            with self.assertRaisesRegex(ValueError, r"do not share one bootstrap \(seed, resamples\)"):
                self.run_main({**trees, "revision": unpaired}, Path(folder) / "unpaired_out", revision=True)
            # A second copy of a matched-pass run under another folder: the run fields are equal, so it is a duplicate.
            duplicate = Path(folder) / "duplicate"
            duplicate.mkdir()
            for level in (0.30, 0.31):
                self.write(duplicate, noise, "replay", "weather", 1000, 2, "weather", level, updates=3000)
            repeated = r"Duplicate revision run .* scored on 'weather': \('replay', 'weather', 1000, 3000, 2\)"
            with self.assertRaisesRegex(ValueError, repeated):
                self.run_main({**trees, "revision": duplicate}, Path(folder) / "duplicate_out", revision=True)


if __name__ == "__main__":
    unittest.main()
