import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import cv2
    import numpy as np
    import torch
except ImportError:
    cv2 = None


@unittest.skipIf(cv2 is None, 'SSDA environment required')
class PairedTargetTests(unittest.TestCase):
    def test_pairs_preserve_pixels_without_annotation_access(self):
        path = Path(__file__).resolve().parents[1] / 'external/SSDA-YOLO/utils/unlabeled_target.py'
        spec = importlib.util.spec_from_file_location('ssda_unlabeled', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            real = root / 'images/train/a.png'
            real.parent.mkdir(parents=True)
            fake = root / 'translated/a.png'
            fake.parent.mkdir()
            for image, color in [(real, 30), (fake, 100)]:
                cv2.imwrite(str(image), np.full((720, 1280, 3), color, dtype=np.uint8))
            manifest = root / 'target.csv'
            with manifest.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['name', 'image'])
                writer.writeheader()
                writer.writerow(dict(name='a.png', image='images/train/a.png'))
            real_open = open

            def guarded_open(path, *args, **kwargs):
                self.assertEqual(Path(path), manifest, 'Unexpected label/cache read')
                return real_open(path, *args, **kwargs)

            with patch('builtins.open', guarded_open):
                loader, dataset = module.create_unlabeled_pairs(manifest, root, fake.parent, 1)
                images, translated, targets, _, _, _ = next(iter(loader))
            self.assertEqual(tuple(images.shape), (1, 3, 736, 1280))
            self.assertTrue(torch.all(images[:, :, 8:728] == 30))
            self.assertTrue(torch.all(translated[:, :, 8:728] == 100))
            self.assertEqual(tuple(targets.shape), (0, 6))
            with self.assertRaises(ValueError):
                module.UnlabeledPairs(manifest, root, real.parent)


if __name__ == '__main__':
    unittest.main()
