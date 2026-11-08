from __future__ import annotations

import unittest

import bdd100k_model_inputs as inputs
from bdd100k_research_split import SplitError


class BoxTests(unittest.TestCase):
    def test_normalizes_and_clips_box(self) -> None:
        result = inputs.normalized_box(
            {"x1": -10, "y1": 20, "x2": 110, "y2": 80}, 100, 100
        )
        self.assertEqual(result, (0.5, 0.5, 1.0, 0.6, True))

    def test_rejects_degenerate_box_after_clipping(self) -> None:
        with self.assertRaisesRegex(SplitError, "Degenerate"):
            inputs.normalized_box(
                {"x1": 120, "y1": 20, "x2": 130, "y2": 80}, 100, 100
            )


class BudgetTests(unittest.TestCase):
    def test_nested_order_is_reproducible(self) -> None:
        names = [f"{index}.jpg" for index in range(100)]
        self.assertEqual(inputs.nested_order(names, 42), inputs.nested_order(reversed(names), 42))
        self.assertNotEqual(inputs.nested_order(names, 42), inputs.nested_order(names, 43))

    def test_lighting_budget_contains_matched_and_extended_rungs(self) -> None:
        values = dict(inputs.budget_values("lighting", 18_015))
        self.assertEqual(values["1738"], 1738)
        self.assertEqual(values["10000"], 10_000)
        self.assertEqual(values["full"], 18_015)

    def test_budget_values_do_not_overflow(self) -> None:
        values = inputs.budget_values("compound", 1_738)
        self.assertTrue(all(value <= 1_738 for _label, value in values))
        self.assertEqual(values[-1], ("full", 1_738))


if __name__ == "__main__":
    unittest.main()
