"""analyze_instances.py: instance counts of the training subsets, the transitions along each seed's chain, the
regression specifications with their shared bootstrap, and main() end to end.

Synthetic manifests, a synthetic train annotation file and synthetic scored trees in a temporary directory only;
nothing here reads runs/ or data/.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path
from statistics import mean
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_instances import (POOLED, READINGS, ROWS, TRAIN_ANNOTATION, count_instances, estimation_sample,  # noqa: E402
                               fit_row, instance_reading, main, out_of_shift_rmse, regression_rows,
                               replicate_observations, run_subset, seed_draws, shift_chains, tally, transitions)
from bdd_pipeline.campaign import TARGETS  # noqa: E402
from bdd_pipeline.data import CLASSES, domain_rows, nested_rows  # noqa: E402
from prepare_revised_bdd100k import DOMAINS, RELIABLE_CLASSES  # noqa: E402

POOL = 50  # images per synthetic train pool: steps 10 -> 25 -> full add 15 and 25 images
ZERO = {"compound": 0.20, "weather": 0.30, "lighting": 0.25}
INSTANCE_SLOPE, IMAGE_SLOPE = 0.0005, 0.0001  # AP of a synthetic run = mAP(0) + 0.0005 I_c + 0.0001 N, exactly


def true_counts(name: str) -> dict[str, int]:
    """Kept boxes per reliable class of a synthetic image; the source pool has two more pedestrians per image."""
    domain, index = name.split("_")[0], int(name.split("_")[1][:3])
    return {"pedestrian": index % 4 + (2 if domain == "source" else 0), "car": 1 + index % 5,
            "traffic light": int(index % 7 == 0), "traffic sign": (index * 7) % 3}


def synthetic_manifests() -> dict:
    """Train rows of the four domain pools, plus one image of a stratum that belongs to no pool."""
    rows = [{"name": f"{domain}_{index:03d}.jpg", "timeofday": time, "weather": weather, "split": "train"}
            for domain, (time, weather) in DOMAINS.items() for index in range(POOL)]
    rows.append({"name": "dawn_000.jpg", "timeofday": "dawn/dusk", "weather": "clear", "split": "train"})
    return {"train": sorted(rows, key=lambda row: row["name"])}


def write_annotation(path: Path, names: list[str], unlabeled: tuple[str, ...] = ()) -> None:
    """A train annotation file: the true boxes (every car partly outside the image, so clipped and kept), plus a rider
    box and three boxes that sanitize_labels drops. The records of `unlabeled` have no "labels" key."""
    records = []
    for name in names:
        labels = []
        for key, count in (true_counts(name).items() if name.split("_")[0] in DOMAINS else ()):
            box = {"x1": 1200.0, "y1": 10.0, "x2": 1350.0, "y2": 80.0} if key == "car" else \
                  {"x1": 10.0, "y1": 20.0, "x2": 60.0, "y2": 90.0}
            labels += [{"category": key, "box2d": dict(box)} for _ in range(count)]
        labels += [{"category": "rider", "box2d": {"x1": 1.0, "y1": 1.0, "x2": 9.0, "y2": 9.0}},
                   {"category": "pedestrian", "box2d": {"x1": 50.0, "y1": 50.0, "x2": 40.0, "y2": 60.0}},
                   {"category": "car", "box2d": {"x1": 1300.0, "y1": 10.0, "x2": 1400.0, "y2": 50.0}},
                   {"category": "traffic sign"}]
        records.append({"name": name, "attributes": {}, **({} if name in unlabeled else {"labels": labels})})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records), encoding="utf-8")


def counts_of(manifests: dict) -> dict[str, dict[str, int]]:
    return {row["name"]: true_counts(row["name"]) for domain in DOMAINS for row in domain_rows(manifests, "train", domain)}


def run_points(manifests: dict, shift: str, budget: str, seed: int) -> dict[str, float]:
    """Per-class AP of a synthetic fine-tune run: an exact function of the instances and images it trained on."""
    subset = nested_rows(domain_rows(manifests, "train", shift), budget, seed)
    return {key: ZERO[shift] + 0.01 * index + INSTANCE_SLOPE * sum(true_counts(row["name"])[key] for row in subset)
            + IMAGE_SLOPE * len(subset) for index, key in enumerate(RELIABLE_CLASSES)}


def zero_points(shift: str) -> dict[str, float]:
    return {key: ZERO[shift] + 0.01 * index for index, key in enumerate(RELIABLE_CLASSES)}


def entry(stage: str, domain: str, budget: str, seed: int, test_domain: str, points: dict[str, float],
          noise: np.ndarray, epochs: int = 3, updates: int = 500) -> dict:
    """One scored entry as analyze_campaign.load returns it; noise is [replicate] and added to every class."""
    return {"run": {"stage": stage, "domain": domain, "budget": budget, "seed": seed, "epochs": epochs,
                    "total_updates": updates}, "domain": test_domain, "points": dict(points),
            "boots": {key: value + noise for key, value in points.items()}}


def synthetic_entries(manifests: dict, replicates: int = 20) -> list[dict]:
    """Source model on every target; per shift rungs 10 and 25 (seeds 0, 1, 2) and the full pool (seeds 0, 1).
    Every run carries its own test-resample noise, so replicates differ from one another and from the point values."""
    rng = np.random.default_rng(3)
    entries = [entry("source", "source", "full", 0, shift, zero_points(shift), rng.normal(0, 0.01, replicates),
                     epochs=12, updates=30000) for shift in TARGETS]
    for shift in TARGETS:
        for budget, seeds in (("10", (0, 1, 2)), ("25", (0, 1, 2)), ("full", (0, 1))):
            entries += [entry("fine-tune", shift, budget, seed, shift, run_points(manifests, shift, budget, seed),
                              rng.normal(0, 0.01, replicates), updates=600 if budget == "full" else 500)
                        for seed in seeds]
    return entries


def observation(shift: str, seed: int, key: str, budgets: tuple[str, str], images: tuple[int, int],
                instances: tuple[int, int], d_ap: float) -> dict:
    """A hand-made observation with the fields the regressions read."""
    return {"shift": shift, "seed": seed, "class": key, "budget_from": budgets[0], "budget_to": budgets[1],
            "instances_from": instances[0], "from_zero": images[0] == 0, "d_ap": d_ap,
            "d_images": images[1] - images[0], "d_instances": instances[1] - instances[0],
            "dln_images": math.log(images[1] / images[0]) if images[0] else None,
            "dln_instances": math.log(instances[1] / instances[0]) if instances[0] else None}


STEPS = ((("10", "25"), (10, 25)), (("25", "50"), (25, 50)), (("50", "100"), (50, 100)), (("100", "250"), (100, 250)))


def hand_observations(outcome, seeds: int = 3) -> list[dict]:
    """Observations of every shift, seed, class and image step with random instance steps (I_from >= 1);
    outcome(observation without d_ap, class index, step index) gives d_ap."""
    rng = np.random.default_rng(11)
    rows = []
    for shift in TARGETS:
        for seed in range(seeds):
            for index, key in enumerate(RELIABLE_CLASSES):
                for step, (budgets, images) in enumerate(STEPS):
                    start = int(rng.integers(1, 20))
                    row = observation(shift, seed, key, budgets, images, (start, start + int(rng.integers(0, 40))), 0.0)
                    row["d_ap"] = outcome(row, index, step)
                    rows.append(row)
    return rows


class CountTest(unittest.TestCase):
    def test_counts_are_the_boxes_training_keeps(self):
        manifests = synthetic_manifests()
        names = [row["name"] for row in manifests["train"]]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / TRAIN_ANNOTATION
            write_annotation(path, names, unlabeled=(names[0],))
            counts = count_instances(manifests, path)
        first = names[0]  # its record has no "labels" key: no boxes
        self.assertEqual(counts.pop(first), dict.fromkeys(RELIABLE_CLASSES, 0))
        self.assertEqual(counts, {name: true_counts(name) for name in names[1:] if not name.startswith("dawn")})

    def test_missing_record_stops(self):
        manifests = synthetic_manifests()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / TRAIN_ANNOTATION
            write_annotation(path, [row["name"] for row in manifests["train"]][:-3])
            with self.assertRaisesRegex(ValueError, "no annotation record"):
                count_instances(manifests, path)

    def test_duplicate_record_stops(self):
        manifests = synthetic_manifests()
        names = [row["name"] for row in manifests["train"]]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / TRAIN_ANNOTATION
            write_annotation(path, names + [names[5]])
            with self.assertRaisesRegex(ValueError, "Duplicate annotation"):
                count_instances(manifests, path)

    def test_only_the_train_annotation_file_is_read(self):
        with tempfile.TemporaryDirectory() as folder:
            for split in ("test", "val"):
                path = Path(folder) / "annotations" / f"bdd100k_labels_images_{split}.json"
                with mock.patch("analyze_instances.stream_records") as stream:
                    with self.assertRaisesRegex(ValueError, "Only the train annotation file"):
                        count_instances(synthetic_manifests(), path)
                stream.assert_not_called()


class SubsetTest(unittest.TestCase):
    def test_run_subset_is_the_training_call(self):
        manifests = synthetic_manifests()
        pool = domain_rows(manifests, "train", "lighting")
        subset = run_subset(manifests, {"domain": "lighting", "budget": "10", "seed": 1})
        self.assertEqual(subset, nested_rows(pool, "10", 1))
        shuffled = sorted(pool, key=lambda row: row["name"])  # the training code's rule, written out independently
        random.Random(1).shuffle(shuffled)
        self.assertEqual([row["name"] for row in subset], [row["name"] for row in shuffled[:10]])
        self.assertNotEqual(subset, run_subset(manifests, {"domain": "lighting", "budget": "10", "seed": 2}))
        self.assertEqual(len(run_subset(manifests, {"domain": "lighting", "budget": "full", "seed": 0})), POOL)
        counted = tally(subset, counts_of(manifests))
        self.assertEqual(counted["images"], 10)
        for key in RELIABLE_CLASSES:
            self.assertEqual(counted["instances"][key], sum(true_counts(row["name"])[key] for row in subset))
            self.assertEqual(counted["positive"][key], sum(true_counts(row["name"])[key] > 0 for row in subset))

    def test_chain_rungs_are_the_nested_subsets_and_steps_are_nonnegative(self):
        manifests = synthetic_manifests()
        counts, entries = counts_of(manifests), synthetic_entries(manifests)
        for shift in TARGETS:
            chains = shift_chains(entries, manifests, counts, shift)
            self.assertEqual({seed: [rung["budget"] for rung in chain] for seed, chain in chains.items()},
                             {0: ["0", "10", "25", "full"], 1: ["0", "10", "25", "full"], 2: ["0", "10", "25"]})
            for seed, chain in chains.items():
                for rung in chain[1:]:
                    rows = nested_rows(domain_rows(manifests, "train", shift), rung["budget"], seed)
                    self.assertEqual({k: rung[k] for k in ("images", "instances", "positive")}, tally(rows, counts))
                self.assertEqual((chain[0]["images"], chain[0]["updates"]), (0, 0))
                for row in transitions(shift, seed, chain):
                    self.assertGreater(row["d_images"], 0)
                    self.assertGreaterEqual(row["d_instances"], 0)

    def test_matched_pool_runs_are_not_rungs_and_schedules_do_not_mix(self):
        manifests = synthetic_manifests()
        counts, entries = counts_of(manifests), synthetic_entries(manifests)
        noise = np.zeros(20)
        # 1,738 is above the synthetic pool: rebuilding its subset would fail, so it must be skipped.
        matched = entry("fine-tune", "lighting", "1738", 0, "lighting", zero_points("lighting"), noise)
        chains = shift_chains(entries + [matched], manifests, counts, "lighting")
        self.assertNotIn("1738", {rung["budget"] for chain in chains.values() for rung in chain})
        long = entry("fine-tune", "weather", "full", 2, "weather", zero_points("weather"), noise, epochs=12)
        with self.assertRaisesRegex(ValueError, "mix epoch counts"):
            shift_chains(entries + [long], manifests, counts, "weather")

    def test_one_source_model_and_an_e3_ladder_with_the_full_pool_are_required(self):
        manifests = synthetic_manifests()
        counts, entries = counts_of(manifests), synthetic_entries(manifests)
        source = next(e for e in entries if e["run"]["stage"] == "source" and e["domain"] == "weather")
        full = {id(e) for e in entries if e["run"]["domain"] == "weather" and e["run"]["budget"] == "full"}
        long = [{**e, "run": {**e["run"], "epochs": 12}} if id(e) in full else e for e in entries]  # no group mixes
        for broken, message in (([e for e in entries if e is not source], "expected one source model .* found 0"),
                                (entries + [source], "expected one source model .* found 2"),
                                (long, "the ladder needs"), ([e for e in entries if id(e) not in full], "the ladder needs")):
            with self.assertRaisesRegex(ValueError, message):
                shift_chains(broken, manifests, counts, "weather")


class TransitionTest(unittest.TestCase):
    def test_zero_transition_is_tabulated_but_never_regressed(self):
        manifests = synthetic_manifests()
        counts, entries = counts_of(manifests), synthetic_entries(manifests)
        chains = {shift: shift_chains(entries, manifests, counts, shift) for shift in TARGETS}
        observations = [o for shift, by_seed in chains.items() for seed, chain in by_seed.items()
                        for o in transitions(shift, seed, chain)]
        zero = [o for o in observations if o["from_zero"]]
        self.assertEqual(len(zero), 3 * 3 * len(RELIABLE_CLASSES))  # shift x seed x class
        self.assertTrue(all(o["budget_from"] == "0" and o["updates_from"] == 0 and o["dln_images"] is None for o in zero))
        perturbed = [{**o, "d_ap": 1e6} if o["from_zero"] else o for o in observations]
        for row in ROWS:
            self.assertFalse(any(o["from_zero"] for o in estimation_sample(observations, *row)[0]), row)
            self.assertEqual(fit_row(observations, *row), fit_row(perturbed, *row), row)
            self.assertEqual(out_of_shift_rmse(observations, *row), out_of_shift_rmse(perturbed, *row), row)


class OlsTest(unittest.TestCase):
    def test_log_form_recovers_known_coefficients(self):
        rows = hand_observations(lambda o, c, s: 0.1 + 0.02 * o["dln_instances"] + 0.03 * o["dln_images"])
        fit = fit_row(rows, "both", "log", POOLED)
        self.assertAlmostEqual(fit["coef_I"], 0.02, places=10)
        self.assertAlmostEqual(fit["coef_N"], 0.03, places=10)
        self.assertAlmostEqual(fit["r2"], 1.0, places=10)
        x_i = np.array([o["dln_instances"] for o in rows])
        x_n = np.array([o["dln_images"] for o in rows])
        r = np.corrcoef(x_i, x_n)[0, 1]
        self.assertAlmostEqual(fit["corr_IN"], r, places=12)
        self.assertAlmostEqual(fit["vif"], 1 / (1 - r * r), places=10)
        y = 0.02 * x_i + 0.03 * x_n
        self.assertAlmostEqual(fit["std_coef_I"], 0.02 * x_i.std() / y.std(), places=10)
        self.assertAlmostEqual(out_of_shift_rmse(rows, "both", "log", POOLED), 0.0, places=10)
        for key in RELIABLE_CLASSES:
            self.assertAlmostEqual(fit_row(rows, "both", "log", key)["coef_I"], 0.02, places=10)
        self.assertIsNone(fit_row(rows, "instances", "log", POOLED)["coef_N"])
        self.assertIsNone(fit_row(rows, "images", "log", POOLED)["vif"])

    def test_raw_form_and_class_intercepts(self):
        rows = hand_observations(lambda o, c, s: 0.01 * c + 0.0005 * o["d_instances"] + 0.0002 * o["d_images"])
        fit = fit_row(rows, "both+class", "raw", POOLED)
        self.assertAlmostEqual(fit["coef_I"], 0.0005, places=12)
        self.assertAlmostEqual(fit["coef_N"], 0.0002, places=12)
        self.assertAlmostEqual(fit["r2"], 1.0, places=10)
        self.assertAlmostEqual(out_of_shift_rmse(rows, "both+class", "raw", POOLED), 0.0, places=10)
        self.assertLess(fit_row(rows, "both", "raw", POOLED)["r2"], 1.0)  # one intercept cannot absorb the class levels

    def test_class_intercepts_are_partialled_out_of_corr_in(self):
        # Instances = class density x images + noise: the density differences between classes, which the class
        # intercepts absorb, lower the pooled correlation of the raw regressors.
        rng, density = np.random.default_rng(4), (1, 10, 3, 4)
        rows = [observation(shift, seed, key, budgets, images,
                            tuple(density[c] * n + int(rng.integers(0, 20)) for n in images), 0.0)
                for shift in TARGETS for seed in range(3) for c, key in enumerate(RELIABLE_CLASSES)
                for budgets, images in STEPS]
        for o in rows:
            o["d_ap"] = 0.001 * o["d_instances"] + float(rng.normal(0, 0.01))
        x_i = np.array([o["d_instances"] for o in rows], dtype=float)
        x_n = np.array([o["d_images"] for o in rows], dtype=float)
        labels = np.array([o["class"] for o in rows])
        for key in RELIABLE_CLASSES:
            x_i[labels == key] -= x_i[labels == key].mean()
            x_n[labels == key] -= x_n[labels == key].mean()
        within = float(np.corrcoef(x_i, x_n)[0, 1])
        pooled, with_class = fit_row(rows, "both", "raw", POOLED), fit_row(rows, "both+class", "raw", POOLED)
        self.assertAlmostEqual(with_class["corr_IN"], within, places=10)
        self.assertAlmostEqual(with_class["vif"], 1 / (1 - within * within), places=8)
        self.assertGreater(with_class["vif"], pooled["vif"] + 0.5)

    def test_out_of_shift_rmse_holds_each_shift_out(self):
        slopes = dict(zip(TARGETS, (0.01, 0.02, 0.03)))
        rows = hand_observations(lambda o, c, s: slopes[o["shift"]] * o["d_instances"])
        y = np.array([o["d_ap"] for o in rows])
        x = np.column_stack([np.ones(len(rows)), [o["d_instances"] for o in rows]])
        shifts = np.array([o["shift"] for o in rows])
        errors = []
        for shift in TARGETS:
            held = shifts == shift
            errors.append(y[held] - x[held] @ np.linalg.lstsq(x[~held], y[~held], rcond=None)[0])
        expected = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
        in_sample = float(np.sqrt(np.mean((y - x @ np.linalg.lstsq(x, y, rcond=None)[0]) ** 2)))
        self.assertAlmostEqual(out_of_shift_rmse(rows, "instances", "raw", POOLED), expected, places=12)
        self.assertGreater(expected, 1.2 * in_sample)

    def test_collinear_regressors_leave_the_fit_undefined(self):
        # Constant density: I = 2 N at every rung, so ln(I_to / I_from) = ln(N_to / N_from).
        rows = [observation(shift, 0, key, budgets, images, (2 * images[0], 2 * images[1]), 0.01 * step)
                for shift in TARGETS for key in RELIABLE_CLASSES for step, (budgets, images) in enumerate(STEPS)]
        self.assertIsNone(fit_row(rows, "both", "log", POOLED))
        self.assertIsNotNone(fit_row(rows, "instances", "log", POOLED))
        self.assertIsNone(out_of_shift_rmse(rows, "both", "log", POOLED))


class WithinTest(unittest.TestCase):
    def test_pure_image_effect_is_absorbed(self):
        rows = hand_observations(lambda o, c, s: 0.01 + 0.05 * o["dln_images"] + 0.002 * c)
        for form in ("log", "raw"):
            fit = fit_row(rows, "within", form, POOLED)
            self.assertAlmostEqual(fit["coef_I"], 0.0, places=12)
            self.assertIsNone(fit["coef_N"])
        self.assertIsNone(out_of_shift_rmse(rows, "within", "log", POOLED))

    def test_pure_instance_effect_is_recovered(self):
        rows = hand_observations(lambda o, c, s: 0.04 * o["dln_instances"] + 0.003 * s + 0.001 * c + 0.02 * o["dln_images"])
        fit = fit_row(rows, "within", "log", POOLED)
        self.assertAlmostEqual(fit["coef_I"], 0.04, places=12)
        self.assertAlmostEqual(fit["r2"], 1.0, places=10)

    def test_singleton_cells_are_left_out(self):
        rows = hand_observations(lambda o, c, s: 0.04 * o["dln_instances"])
        lone = observation("weather", 0, "car", ("250", "500"), (250, 500), (40, 90), 0.5)
        kept, _ = estimation_sample(rows + [lone], "within", "log", POOLED)
        self.assertEqual(len(kept), len(rows))
        self.assertEqual(len(estimation_sample(rows + [lone], "both", "log", POOLED)[0]), len(rows) + 1)


class LogFormDropTest(unittest.TestCase):
    def test_zero_instances_from_are_dropped_from_the_log_form_and_counted(self):
        rows = hand_observations(lambda o, c, s: 0.01 * o["d_instances"])
        empty = [observation("compound", seed, "traffic light", budgets, images, (0, 3), 0.02)
                 for seed in range(3) for budgets, images in STEPS[:2]]
        kept, dropped = estimation_sample(rows + empty, "both", "log", POOLED)
        self.assertEqual((len(kept), dropped), (len(rows), 6))
        self.assertEqual(estimation_sample(rows + empty, "both", "log", "traffic light")[1], 6)
        self.assertEqual(estimation_sample(rows + empty, "both", "log", "car")[1], 0)
        self.assertEqual(estimation_sample(rows + empty, "both", "raw", POOLED), (rows + empty, 0))


class ReadingTest(unittest.TestCase):
    def test_interval_maps_to_the_plan_readings(self):
        for (low, high), key in (((0.001, 0.02), "above"), ((-0.01, 0.02), "covers"), ((0.0, 0.2), "covers"),
                                 ((-0.2, -0.1), "below"), ((None, None), None), ((None, 0.1), None)):
            self.assertEqual(instance_reading({"coef_I_ci_low": low, "coef_I_ci_high": high}), READINGS[key])


class BootstrapTest(unittest.TestCase):
    replicates = 12

    def setUp(self):
        manifests = synthetic_manifests()
        counts, entries = counts_of(manifests), synthetic_entries(manifests, self.replicates)
        self.chains = {shift: shift_chains(entries, manifests, counts, shift) for shift in TARGETS}

    def test_draws_come_from_one_generator_in_target_order(self):
        draws = seed_draws(self.chains, self.replicates, 7)
        rng = np.random.default_rng(7)
        for shift in TARGETS:
            np.testing.assert_array_equal(draws[shift], rng.integers(0, 3, (self.replicates, 3)))

    def test_replicate_takes_resample_t_of_the_drawn_chains(self):
        draws = {shift: np.array([[2, 2, 0]] * self.replicates) for shift in TARGETS}
        sample = replicate_observations(self.chains, draws, 5)
        for shift in TARGETS:
            seeds = [o["seed"] for o in sample if o["shift"] == shift and o["class"] == "car"]
            self.assertEqual(seeds, [2, 2, 2, 2, 0, 0, 0])  # seed 2 twice (two transitions each), then seed 0 (three)
        o = next(o for o in sample if o["shift"] == "weather" and o["budget_to"] == "25" and o["class"] == "car")
        rung = next(r for r in self.chains["weather"][2] if r["budget"] == "25")
        self.assertEqual(o["ap_to"], float(rung["entry"]["boots"]["car"][5]))
        self.assertNotEqual(o["ap_to"], rung["entry"]["points"]["car"])

    def test_every_row_is_refitted_on_the_same_replicate(self):
        draws = seed_draws(self.chains, self.replicates, 1)
        observations = [o for shift, by_seed in self.chains.items() for seed, chain in by_seed.items()
                        for o in transitions(shift, seed, chain)]
        seen = []

        def spy(sample, *row):
            seen.append((sample, row))
            return fit_row(sample, *row)

        with mock.patch("analyze_instances.fit_row", side_effect=spy):
            rows = regression_rows(observations, self.chains, draws)
        replicate_calls = seen[:self.replicates * len(ROWS)]
        for t in range(self.replicates):
            calls = replicate_calls[t * len(ROWS):(t + 1) * len(ROWS)]
            self.assertEqual([row for _, row in calls], ROWS)
            self.assertTrue(all(sample is calls[0][0] for sample, _ in calls))
            self.assertEqual(calls[0][0], replicate_observations(self.chains, draws, t))
        self.assertTrue(all(sample is observations for sample, _ in seen[self.replicates * len(ROWS):]))
        self.assertEqual([(r["spec"], r["form"], r["class"]) for r in rows], ROWS)
        for row in rows:
            self.assertEqual(row["replicates"], self.replicates)
            if row["coef_I"] is not None and row["undefined_replicates"] < self.replicates:
                self.assertLessEqual(row["coef_I_ci_low"], row["coef_I_ci_high"])

    def observations(self) -> list[dict]:
        return [o for shift, by_seed in self.chains.items() for seed, chain in by_seed.items()
                for o in transitions(shift, seed, chain)]

    def test_a_nan_resample_leaves_that_replicate_undefined_for_the_rows_that_use_it(self):
        draws = {shift: np.tile([0, 1, 2], (self.replicates, 1)) for shift in TARGETS}  # every seed in every replicate
        before = {(r["spec"], r["form"], r["class"]): r for r in regression_rows(self.observations(), self.chains, draws)}
        next(r for r in self.chains["weather"][0] if r["budget"] == "10")["entry"]["boots"]["car"][4] = np.nan
        for row in regression_rows(self.observations(), self.chains, draws):
            key = (row["spec"], row["form"], row["class"])
            defined = before[key]["undefined_replicates"] < self.replicates
            self.assertEqual(row["undefined_replicates"],
                             before[key]["undefined_replicates"] + (defined and row["class"] in (POOLED, "car")), key)
            for column in ("coef_I_ci_low", "coef_I_ci_high", "coef_N_ci_low", "coef_N_ci_high"):
                self.assertTrue(row[column] is None or math.isfinite(row[column]), (key, column))
        self.assertEqual(before[("within", "log", POOLED)]["undefined_replicates"], 0)

    def test_a_point_without_ap_is_left_out_of_the_point_fit_and_every_replicate(self):
        draws = seed_draws(self.chains, self.replicates, 2)
        before = {(r["spec"], r["form"], r["class"]): r for r in regression_rows(self.observations(), self.chains, draws)}
        rung = next(r for r in self.chains["compound"][0] if r["budget"] == "25")
        self.assertTrue(np.isfinite(rung["entry"]["boots"]["traffic sign"]).all())
        rung["entry"]["points"]["traffic sign"] = None

        def touched(sample):
            return [o for o in sample if (o["shift"], o["seed"], o["class"]) == ("compound", 0, "traffic sign")
                    and "25" in (o["budget_from"], o["budget_to"])]

        self.assertEqual([o["d_ap"] for o in touched(self.observations())], [None, None])  # 10 -> 25 and 25 -> full
        drawn = [o for t in range(self.replicates) for o in touched(replicate_observations(self.chains, draws, t))]
        self.assertTrue(drawn and all(o["d_ap"] is None for o in drawn))  # though its resamples are finite
        after = {(r["spec"], r["form"], r["class"]): r for r in regression_rows(self.observations(), self.chains, draws)}
        for scope, fewer in ((POOLED, 2), ("traffic sign", 2), ("car", 0)):
            self.assertEqual(after[("both", "raw", scope)]["n_obs"], before[("both", "raw", scope)]["n_obs"] - fewer)


class MainTest(unittest.TestCase):
    replicates = 30

    def write(self, scored: Path, noise: dict, run: dict, test_domain: str, points: dict[str, float]) -> None:
        folder = scored / f"{len(list(scored.iterdir())):03d}_{run['stage']}_{run['domain']}_N{run['budget']}" / test_domain
        folder.mkdir(parents=True)
        per_class = [points.get(name, 0.3) for name in CLASSES]
        metrics = {"run": run, "domain": test_domain, "images": 50, "AP50_95": mean(per_class),
                   "reliable_AP50_95": mean(points[name] for name in RELIABLE_CLASSES),
                   "per_class": [{"class": name, "AP50_95": value} for name, value in zip(CLASSES, per_class)],
                   "bootstrap": {"seed": 20261030, "resamples": self.replicates, "class_order": list(CLASSES)}}
        (folder / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        np.save(folder / "bootstrap_class_ap.npy", np.asarray(per_class) + noise[test_domain])

    def build(self, scored: Path, manifests: dict) -> None:
        """Every run's AP is an exact function of its subset, and every bootstrap vector is that AP plus test-resample
        noise shared by all sets of one test domain, so every replicate gives the point differences exactly."""
        rng = np.random.default_rng(5)
        noise = {domain: rng.normal(0, 0.005, (self.replicates, len(CLASSES))) for domain in DOMAINS}
        scored.mkdir()
        source = {"stage": "source", "domain": "source", "budget": "full", "seed": 0, "epochs": 12, "total_updates": 30000}
        self.write(scored, noise, source, "source", dict.fromkeys(RELIABLE_CLASSES, 0.35))
        for shift in TARGETS:
            self.write(scored, noise, source, shift, zero_points(shift))
            for budget, seeds in (("10", (0, 1, 2)), ("25", (0, 1, 2)), ("full", (0, 1))):
                for seed in seeds:
                    run = {"stage": "fine-tune", "domain": shift, "budget": budget, "seed": seed, "epochs": 3,
                           "total_updates": 600 if budget == "full" else 500}
                    self.write(scored, noise, run, shift, run_points(manifests, shift, budget, seed))
        # Not rungs: the matched-pool run (its budget exceeds the synthetic pool), a run scored on the source test set
        # and a control run.
        self.write(scored, noise, {"stage": "fine-tune", "domain": "lighting", "budget": "1738", "seed": 0, "epochs": 3,
                                   "total_updates": 2607}, "lighting", zero_points("lighting"))
        self.write(scored, noise, {"stage": "fine-tune", "domain": "lighting", "budget": "10", "seed": 0, "epochs": 3,
                                   "total_updates": 500}, "source", dict.fromkeys(RELIABLE_CLASSES, 0.33))
        self.write(scored, noise, {"stage": "control", "domain": "weather", "budget": "10", "seed": 0, "epochs": 3,
                                   "total_updates": 500}, "weather", dict.fromkeys(RELIABLE_CLASSES, 0.1))

    def test_unshared_resamples_stop(self):
        manifests = synthetic_manifests()
        for broken, message in (("seed", "do not share one bootstrap"), ("length", "different lengths")):
            with tempfile.TemporaryDirectory() as folder:
                root, scored = Path(folder) / "dataset", Path(folder) / "scored"
                write_annotation(root / TRAIN_ANNOTATION, [row["name"] for row in manifests["train"]])
                self.build(scored, manifests)
                folder_of_run = next(scored.glob("*_fine-tune_weather_N25/weather"))
                if broken == "seed":
                    path = folder_of_run / "metrics.json"
                    metrics = json.loads(path.read_text(encoding="utf-8"))
                    metrics["bootstrap"]["seed"] = 1
                    path.write_text(json.dumps(metrics), encoding="utf-8")
                else:  # metrics.json still says 30 resamples: only the length check sees it
                    path = folder_of_run / "bootstrap_class_ap.npy"
                    np.save(path, np.load(path)[:-1])
                with (mock.patch("analyze_instances.read_manifests", return_value=({}, manifests)),
                      self.assertRaisesRegex(ValueError, message)):
                    main(["--scored", str(scored), "--dataset-root", str(root), "--output-dir", str(Path(folder) / "out"),
                          "--no-plots"])

    def table(self, path: Path) -> list[dict]:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_a_rung_without_ap_for_a_class_is_flagged_out_of_the_regressions(self):
        manifests = synthetic_manifests()
        with tempfile.TemporaryDirectory() as folder:
            root, scored, output = Path(folder) / "dataset", Path(folder) / "scored", Path(folder) / "out"
            write_annotation(root / TRAIN_ANNOTATION, [row["name"] for row in manifests["train"]])
            self.build(scored, manifests)
            for path in scored.glob("*_fine-tune_compound_N25/compound/metrics.json"):
                metrics = json.loads(path.read_text(encoding="utf-8"))
                if metrics["run"]["seed"] == 0:  # no traffic sign AP, though its resamples stay finite
                    metrics["per_class"][CLASSES.index("traffic sign")]["AP50_95"] = None
                    path.write_text(json.dumps(metrics), encoding="utf-8")
            with (mock.patch("analyze_instances.read_manifests", return_value=({}, manifests)),
                  contextlib.redirect_stdout(io.StringIO())):
                main(["--scored", str(scored), "--dataset-root", str(root), "--output-dir", str(output), "--no-plots"])
            table = self.table(output / "instance_transitions.csv")
            regression = next(r for r in self.table(output / "instance_regression.csv")
                              if (r["spec"], r["form"], r["class"]) == ("both", "raw", POOLED))
        touched = [r for r in table if (r["shift"], r["seed"], r["class"]) == ("compound", "0", "traffic sign")
                   and "25" in (r["budget_from"], r["budget_to"])]
        self.assertEqual([(r["budget_from"], r["budget_to"]) for r in touched], [("10", "25"), ("25", "full")])
        self.assertTrue(all((r["d_ap"], r["in_regressions"], r["in_log_form"]) == ("", "False", "False") for r in touched))
        self.assertEqual(sum(r["in_regressions"] == "True" for r in table), 3 * 5 * len(RELIABLE_CLASSES) - 2)
        self.assertEqual(int(regression["n_obs"]), 3 * 5 * len(RELIABLE_CLASSES) - 2)

    def test_end_to_end(self):
        manifests = synthetic_manifests()
        with tempfile.TemporaryDirectory() as folder:
            root, scored, output = Path(folder) / "dataset", Path(folder) / "scored", Path(folder) / "out"
            write_annotation(root / TRAIN_ANNOTATION, [row["name"] for row in manifests["train"]])
            self.build(scored, manifests)
            printed = io.StringIO()
            with (mock.patch("analyze_instances.read_manifests", return_value=({}, manifests)) as read,
                  mock.patch("analyze_instances.nested_rows", wraps=nested_rows) as nested,
                  contextlib.redirect_stdout(printed)):
                main(["--scored", str(scored), "--dataset-root", str(root), "--output-dir", str(output), "--seed", "3"])
            read.assert_called_once_with(root)
            files = sorted(path.name for path in output.iterdir())
            tables = {name: self.table(output / f"{name}.csv") for name in
                      ("instance_pool_density", "instance_counts", "instance_transitions", "instance_regression",
                       "instance_recovery")}
            summary = json.loads((output / "instance_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(files, ["instance_counts.csv", "instance_pool_density.csv", "instance_recovery.csv",
                                 "instance_recovery.png", "instance_regression.csv", "instance_summary.json",
                                 "instance_transitions.csv"])

        # Every rung's subset was rebuilt with the training call on its own pool, and nothing else was.
        calls = {(rows[0]["name"].split("_")[0], budget, seed) for (rows, budget, seed), _ in nested.call_args_list}
        self.assertEqual(len(nested.call_args_list), 3 * 8)
        self.assertEqual(calls, {(shift, budget, seed) for shift in TARGETS
                                 for budget, seeds in (("10", (0, 1, 2)), ("25", (0, 1, 2)), ("full", (0, 1)))
                                 for seed in seeds})
        counts = counts_of(manifests)
        row = next(r for r in tables["instance_counts"] if (r["shift"], r["seed"], r["budget"], r["class"])
                   == ("lighting", "1", "25", "pedestrian"))
        expected = tally(nested_rows(domain_rows(manifests, "train", "lighting"), "25", 1), counts)
        self.assertEqual((int(row["images"]), int(row["instances"]), int(row["positive_images"])),
                         (25, expected["instances"]["pedestrian"], expected["positive"]["pedestrian"]))
        self.assertEqual(len(tables["instance_counts"]), 3 * 8 * len(RELIABLE_CLASSES))

        density = {(r["domain"], r["class"]): r for r in tables["instance_pool_density"]}
        self.assertEqual(len(density), 4 * len(RELIABLE_CLASSES))
        self.assertEqual((int(density[("source", "pedestrian")]["instances"]), int(density[("source", "pedestrian")]["images"])),
                         (173, POOL))
        self.assertIn("pedestrian, source train pool: 173 / 50 = 3.46 per image (lecturer: 13,170 / 9,804 = 1.34)",
                      printed.getvalue())
        self.assertIn("lecturer: 12,265 / 18,015 = 0.68", printed.getvalue())

        transitions_table = tables["instance_transitions"]
        self.assertEqual(len(transitions_table), 3 * 8 * len(RELIABLE_CLASSES))
        zero = [r for r in transitions_table if r["from_zero"] == "True"]
        self.assertEqual(len(zero), 3 * 3 * len(RELIABLE_CLASSES))
        self.assertTrue(all(r["in_regressions"] == "False" and r["updates_from"] == "0" and r["updates_to"] == "500"
                            for r in zero))
        self.assertTrue(all(int(r["d_instances"]) >= 0 for r in transitions_table))

        regressions = {(r["spec"], r["form"], r["class"]): r for r in tables["instance_regression"]}
        self.assertEqual(list(regressions), ROWS)
        self.assertEqual(int(regressions[("both", "raw", POOLED)]["n_obs"]), 3 * 5 * len(RELIABLE_CLASSES))
        # The synthetic AP is exact in instances and images and the resample noise cancels inside a test domain.
        for key in ("within", "raw", POOLED), ("both", "raw", POOLED), ("both+class", "raw", POOLED):
            for column in ("coef_I", "coef_I_ci_low", "coef_I_ci_high"):
                self.assertAlmostEqual(float(regressions[key][column]), INSTANCE_SLOPE, places=9, msg=(key, column))
            self.assertEqual(regressions[key]["undefined_replicates"], "0")
        self.assertAlmostEqual(float(regressions[("both", "raw", POOLED)]["coef_N"]), IMAGE_SLOPE, places=9)
        log_both = regressions[("both", "log", POOLED)]
        self.assertEqual(int(log_both["n_obs"]) + int(log_both["dropped_zero_instances"]), 3 * 5 * len(RELIABLE_CLASSES))

        # Traffic light gains 0.0005 * 8 + 0.0001 * 50 = 0.009 <= 0.01 at the full pool: no recovery curve.
        recovery = tables["instance_recovery"]
        self.assertEqual({r["shift"] for r in recovery if r["class"] == "traffic light" and r["note"]}, set(TARGETS))
        full = [r for r in recovery if r["budget"] == "full" and r["class"] == "pedestrian"]
        self.assertEqual(len(full), 3)
        self.assertTrue(all(math.isclose(float(r["recovery"]), 1.0) for r in full))

        # At a fixed image step the synthetic AP rises with the instances drawn, so the within/log interval is above 0.
        self.assertEqual(summary["reading"], READINGS["above"])
        self.assertEqual((summary["within_log"]["spec"], summary["within_log"]["form"], summary["within_log"]["class"]),
                         ("within", "log", POOLED))
        self.assertGreater(summary["within_log"]["coef_I_ci_low"], 0)
        self.assertAlmostEqual(summary["within_log"]["coef_I"], float(regressions[("within", "log", POOLED)]["coef_I"]))
        # Regressed transitions per class: seeds 0 and 1 have 10 -> 25 -> full, seed 2 has 10 -> 25, on three shifts.
        self.assertEqual(summary["observations"]["transitions"], 3 * 5)
        self.assertEqual(summary["observations"]["raw_form"], 3 * 5 * len(RELIABLE_CLASSES))
        self.assertEqual(summary["observations"]["from_zero_transitions"], 9)
        self.assertIn("15 transitions, 60 observations in the raw form", printed.getvalue())
        self.assertEqual(set(summary["out_of_shift_rmse"]), {"log", "raw"})
        self.assertEqual(summary["pedestrian_density"]["source"]["instances"], 173)


if __name__ == "__main__":
    unittest.main()
