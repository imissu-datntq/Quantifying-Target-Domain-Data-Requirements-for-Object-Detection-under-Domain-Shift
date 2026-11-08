"""RQ5 port: augmentation, ALDI++ distillation losses, EMA, deterministic streams and the grid."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bdd_adapt.augment import generator_for, mic_mask, strong  # noqa: E402
from bdd_adapt.distill import distillation_losses, ema_update  # noqa: E402
from bdd_adapt.grid import POLICY, grid, require_locked, train_arguments  # noqa: E402
from bdd_adapt.train import Stream, labeled_plan, loader, run_updates  # noqa: E402
from bdd_pipeline.model import freeze_batch_norm  # noqa: E402


def small_detector(seed: int = 0):
    torch.manual_seed(seed)
    model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=11, min_size=128, max_size=128)
    freeze_batch_norm(model.backbone)
    return model


class Toy(Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        image = torch.full((3, 4, 6), float(index))
        return image, {"boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0]]), "labels": torch.tensor([1])}


class AugmentationTests(unittest.TestCase):
    def test_strong_is_deterministic_bounded_and_keeps_shape(self):
        image = torch.rand(3, 72, 128)
        a = strong(image, generator_for(0, "source", 7), erase=True, mic=True)
        b = strong(image, generator_for(0, "source", 7), erase=True, mic=True)
        c = strong(image, generator_for(0, "source", 8), erase=True, mic=True)
        self.assertEqual(a.shape, image.shape)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))
        self.assertGreaterEqual(float(a.min()), 0.0)
        self.assertLessEqual(float(a.max()), 1.0)

    def test_mic_masks_about_half_in_whole_blocks(self):
        masked = mic_mask(torch.ones(3, 720, 1280), generator_for("mic"))
        self.assertAlmostEqual(float(masked.mean()), 0.5, delta=0.1)
        block = masked[0, :32, :32]  # block-aligned near the origin: all kept or all masked
        self.assertIn(float(block.mean()), (0.0, 1.0))


class DistillationTests(unittest.TestCase):
    def setUp(self):
        self.student = small_detector()
        self.teacher = copy.deepcopy(self.student).eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.student.train()
        torch.manual_seed(1)
        self.images = [torch.rand(3, 128, 128) for _ in range(2)]

    def test_identical_models_and_views_give_zero_regression_and_kl(self):
        losses, diagnostics = distillation_losses(self.student, self.teacher, self.images, self.images, threshold=0.0)
        self.assertGreater(diagnostics["pseudo_boxes"], 0)
        self.assertGreater(diagnostics["rpn_positive"], 0)
        self.assertAlmostEqual(float(losses["loss_rpn_l1"]), 0.0, places=5)
        self.assertAlmostEqual(float(losses["loss_roih_l1"]), 0.0, places=5)
        self.assertGreater(float(losses["loss_cls_ce"]), 0.0)  # = teacher entropy when predictions agree

    def test_gradients_reach_student_only(self):
        losses, _ = distillation_losses(self.student, self.teacher, self.images,
                                        [image.flip(0) for image in self.images], threshold=0.0)
        sum(losses.values()).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.student.parameters()))
        self.assertTrue(all(p.grad is None for p in self.teacher.parameters()))
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))

    def test_no_pseudo_labels_still_finite(self):
        losses, diagnostics = distillation_losses(self.student, self.teacher, self.images, self.images, threshold=1.01)
        self.assertEqual(diagnostics["pseudo_boxes"], 0)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))

    def test_requires_modes(self):
        self.student.eval()
        with self.assertRaises(ValueError):
            distillation_losses(self.student, self.teacher, self.images, self.images)

    def test_ema_update(self):
        teacher, student = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        before = teacher.weight.detach().clone()
        ema_update(teacher, student, 0.9)
        self.assertTrue(torch.allclose(teacher.weight, 0.9 * before + 0.1 * student.weight))


class StreamTests(unittest.TestCase):
    def test_stream_is_a_pure_function_of_index(self):
        a, b = Stream(Toy(5), "source", 3, True), Stream(Toy(5), "source", 3, True)
        first = [a.index(k) for k in range(10)]
        self.assertEqual(first, [b.index(k) for k in range(10)])
        self.assertEqual(sorted(first[:5]), list(range(5)))
        self.assertEqual(sorted(first[5:]), list(range(5)))
        self.assertNotEqual(first, [Stream(Toy(5), "source", 4, True).index(k) for k in range(10)])

    def test_resumed_loader_continues_the_same_batches(self):
        stream = Stream(Toy(7), "target", 0, True)
        full = [next_batch[2] for next_batch in loader(stream, 2, 0, 6, 0, False)]
        resumed = [next_batch[2] for next_batch in loader(Stream(Toy(7), "target", 0, True), 2, 4, 6, 0, False)]
        self.assertEqual(full[4:], resumed)

    def test_loader_creation_leaves_global_rng_untouched(self):
        torch.manual_seed(5)
        expected = torch.rand(3)
        torch.manual_seed(5)
        loader(Stream(Toy(4), "source", 0, True), 2, 3, 5, 0, False)
        self.assertTrue(torch.equal(torch.rand(3), expected))

    def test_flip_moves_boxes(self):
        stream = Stream(Toy(3), "source", 0, True)
        flipped = [k for k in range(40) if stream[k][1]["boxes"][0, 0] > 0]
        self.assertTrue(flipped)
        self.assertTrue(all(stream[k][1]["boxes"][0].tolist() == [4.0, 0.0, 6.0, 2.0] for k in flipped))


class GridTests(unittest.TestCase):
    def test_schedule_and_labeled_stream(self):
        self.assertEqual(run_updates(3000, 0), 3000)
        self.assertEqual(run_updates(3000, 1000), 3000)
        self.assertEqual(run_updates(3000, 5000), 10000)
        self.assertEqual(labeled_plan(0), [("source", 2)])
        self.assertEqual(labeled_plan(10), [("source", 1), ("target", 1)])

    def test_grid_rows(self):
        rows = grid(POLICY)
        self.assertEqual(len(rows), 46)
        stages = [row["stage"] for row in rows]
        self.assertEqual(stages.count("replay"), 10)
        self.assertEqual(sum(row["budget"] == 0 and row["stage"] == "adapt" for row in rows), 6)
        self.assertEqual([row["budget"] for row in rows[-3:]], [5000] * 3)
        self.assertEqual(rows[0]["domain"], "compound")

    def test_locking(self):
        with self.assertRaises(ValueError):
            require_locked(dict(POLICY, floor_updates=None))
        self.assertEqual((POLICY["floor_updates"], POLICY["ema_alpha"]), (2000, 0.9996))  # pilot lock, 2026-11-04
        locked = dict(POLICY, floor_updates=3000, ema_alpha=0.9996)
        arguments = train_arguments(grid(locked)[0], locked, Path("d"), Path("o"), Path("s.pt"))
        self.assertIn("--amp", arguments)
        self.assertEqual(arguments[arguments.index("--floor-updates") + 1], "3000")


class PilotLockTests(unittest.TestCase):
    def test_rule(self):
        from rq5_pilot_lock import lock
        steps = range(0, 3001, 500)
        rising = dict(zip(steps, [0.25, 0.26, 0.27, 0.275, 0.278, 0.279, 0.279]))  # within 0.005 of peak from 1000
        late = dict(zip(steps, [0.25, 0.252, 0.255, 0.26, 0.265, 0.27, 0.272]))    # from 2500
        curves = {(0, 0.9996): rising, (100, 0.9996): late, (0, 0.999): rising, (100, 0.999): late}
        result = lock(curves)
        self.assertEqual(result["alpha"], 0.9996)  # tie -> published default
        self.assertEqual(result["T"], 2500)
        self.assertTrue(result["gate"]["passed"])
        worse = {**curves, (0, 0.999): {s: v + 0.01 for s, v in rising.items()},
                 (100, 0.999): {s: v + 0.01 for s, v in late.items()}}
        self.assertEqual(lock(worse)["alpha"], 0.999)
        # N=0 peaks at step 0 then falls; its last point is within delta of a late peak elsewhere
        falling = dict(zip(steps, [0.25, 0.24, 0.23, 0.23, 0.23, 0.23, 0.231]))
        dropping = {**curves, (0, 0.9996): falling, (0, 0.999): falling}
        collapsed = lock(dropping)  # never sustained within delta of its step-0 peak
        self.assertIsNone(collapsed["T"])
        self.assertFalse(collapsed["gate"]["passed"])
        sagging = {**curves, (0, 0.9996): dict(zip(steps, [0.26, 0.25, 0.25, 0.25, 0.25, 0.25, 0.258])),
                   (0, 0.999): dict(zip(steps, [0.26, 0.25, 0.25, 0.25, 0.25, 0.25, 0.258]))}
        self.assertTrue(lock(sagging)["gate"]["passed"])  # recovers to within delta of step 0 by T


if __name__ == "__main__":
    unittest.main()
