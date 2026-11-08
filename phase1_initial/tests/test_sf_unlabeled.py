"""Run with the SF-YOLO environment; optional dependencies skip elsewhere."""
import csv
import importlib.util
import tempfile
import unittest
import os
import subprocess
import sys
import itertools
from pathlib import Path
from unittest.mock import patch

try:
    import cv2
    import numpy as np
    import torch
except ImportError:
    cv2 = None


@unittest.skipIf(cv2 is None, 'SF environment required')
class UnlabeledTargetTests(unittest.TestCase):
    def test_auxiliary_sampler_seed_is_reproducible_and_local(self):
        source = Path(__file__).resolve().parents[1] / 'external/sf-yolo/TargetAugment_train/sampler.py'
        spec = importlib.util.spec_from_file_location('sf_sampler', source)
        sampler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sampler)
        np.random.seed(987)
        expected = np.random.rand()
        np.random.seed(987)
        first = list(itertools.islice(sampler.InfiniteSampler(8, seed=42), 40))
        second = list(itertools.islice(sampler.InfiniteSampler(8, seed=42), 40))
        self.assertEqual(first, second)
        self.assertEqual(np.random.rand(), expected)

    def setUp(self):
        source = Path(__file__).resolve().parents[1] / 'external/sf-yolo/utils/unlabeled_target.py'
        spec = importlib.util.spec_from_file_location('sf_unlabeled', source)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_native_pixels_empty_labels_and_no_annotation_access(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / 'images/train/example.png'
            image.parent.mkdir(parents=True)
            pixels = np.full((720, 1280, 3), (10, 20, 30), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image), pixels))
            manifest = root / 'target.csv'
            with manifest.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['name', 'image', 'coco_image_id'])
                writer.writeheader()
                writer.writerow(dict(name=image.name, image='images/train/example.png', coco_image_id=1))
            real_open = open

            def guarded_open(path, *args, **kwargs):
                self.assertEqual(Path(path), manifest, 'Loader tried to open annotations/cache')
                return real_open(path, *args, **kwargs)

            with patch('builtins.open', guarded_open):
                loader, dataset = self.module.create_unlabeled_dataloader(manifest, root, 1)
                batch, targets, _, _ = next(iter(loader))
            self.assertEqual(tuple(batch.shape), (1, 3, 736, 1280))
            self.assertEqual(tuple(targets.shape), (0, 6))
            self.assertTrue(np.array_equal(batch[0, :, 8:728].permute(1, 2, 0).numpy(), pixels[:, :, ::-1]))
            self.assertEqual(dataset.labels[0].shape, (0, 5))
            empty = self.module.concatenate_pseudo_labels([], torch.device('cpu'))
            self.assertEqual(tuple(empty.shape), (0, 6))

    def test_upstream_loss_accepts_empty_pseudo_labels(self):
        source = Path(__file__).resolve().parents[1] / 'external/sf-yolo'
        # Separate process avoids importing another candidate's `utils` package.
        script = '''
import torch
from models.yolo import Model
from utils.loss import ComputeLoss
torch.set_num_threads(2)
model = Model('models/yolov5l.yaml', nc=10)
model.hyp = dict(cls_pw=1., obj_pw=1., fl_gamma=0., box=.05,
                 obj=1., cls=.5, anchor_t=4.)
criterion = ComputeLoss(model)
predictions = [torch.zeros(1, 3, h, w, 15, requires_grad=True)
               for h, w in [(92, 160), (46, 80), (23, 40)]]
loss, components = criterion(predictions, torch.zeros(0, 6))
assert torch.isfinite(loss).all()
assert components[0] == 0 and components[2] == 0
loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all()
           for p in predictions)
print('Empty pseudo-label loss/backward passed; this is not a full graph smoke.')
'''
        env = dict(os.environ, YOLO_AUTOINSTALL='false', YOLOv5_AUTOINSTALL='false',
                   PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, '-c', script], cwd=source, env=env,
                                timeout=120, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
