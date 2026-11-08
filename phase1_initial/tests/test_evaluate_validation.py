import importlib.util
import unittest
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from evaluate_validation import evaluate, verify_coverage


class CoverageTests(unittest.TestCase):
    def test_requires_all_images_including_zero_detection_images(self):
        coverage = dict(status='completed', split='val', evaluated_image_ids=[1, 2],
                        prediction_sha256='p', checkpoint_sha256='a' * 64)
        verify_coverage(coverage, [1, 2], 'p')
        for ids in ([1], [1, 1, 2], [1, 2, 3]):
            with self.assertRaises(ValueError):
                verify_coverage(dict(coverage, evaluated_image_ids=ids), [1, 2], 'p')
        with self.assertRaises(ValueError):
            verify_coverage(dict(coverage, split='test'), [1, 2], 'p')
        with self.assertRaises(ValueError):
            verify_coverage(coverage, [1, 2], 'different')
        with self.assertRaises(ValueError):
            verify_coverage(dict(coverage, status='failed'), [1, 2], 'p')


@unittest.skipUnless(importlib.util.find_spec('pycocotools'), 'COCO environment required')
class CommonEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.gt = dict(images=[dict(id=1, width=1280, height=720)],
                       categories=[dict(id=1, name='car'), dict(id=2, name='absent')],
                       annotations=[dict(id=1, image_id=1, category_id=1,
                                         bbox=[100, 100, 80, 60], area=4800, iscrowd=0)],
                       info={})

    def test_perfect_predictions_and_absent_class(self):
        result = evaluate(self.gt, [dict(image_id=1, category_id=1,
                                        bbox=[100, 100, 80, 60], score=.9)], [1])
        self.assertAlmostEqual(result['map_50_95'], 100.)
        self.assertIsNone(result['per_class_ap']['2'])

    def test_large_canonical_image_id_is_not_rounded(self):
        image_id = 7201176584005273989
        self.gt['images'][0]['id'] = image_id
        self.gt['annotations'][0]['image_id'] = image_id
        result = evaluate(self.gt, [dict(image_id=image_id, category_id=1,
                                        bbox=[100, 100, 80, 60], score=.9)], [image_id])
        self.assertAlmostEqual(result['map_50_95'], 100.)

    def test_empty_and_incorrect_predictions(self):
        for predictions in [[], [dict(image_id=1, category_id=1,
                                      bbox=[900, 500, 10, 10], score=.9)]]:
            self.assertEqual(evaluate(self.gt, predictions, [1])['map_50_95'], 0.)

    def test_rejects_foreign_image_ids(self):
        with self.assertRaises(ValueError):
            evaluate(self.gt, [dict(image_id=2, category_id=1,
                                   bbox=[1, 1, 10, 10], score=.9)], [1])

    def test_cli_validation_paths_and_collision_guard(self):
        script = Path(__file__).resolve().parents[1] / 'evaluate_validation.py'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'coco').mkdir()
            manifest = root / 'manifests/domains/source/val.csv'
            manifest.parent.mkdir(parents=True)
            (root / 'coco/instances_val.json').write_text(json.dumps(self.gt))
            with manifest.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['image', 'coco_image_id'])
                writer.writeheader()
                writer.writerow(dict(image='images/val/example.jpg', coco_image_id=1))
            predictions = root / 'predictions.json'
            predictions.write_text('[]')
            coverage = root / 'coverage.json'
            coverage.write_text(json.dumps(dict(status='completed', split='val', evaluated_image_ids=[1],
                prediction_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest(),
                checkpoint_sha256='a' * 64)))
            output = root / 'metrics.json'
            command = [sys.executable, str(script), '--inputs', str(root), '--domain',
                       'source', '--predictions', str(predictions), '--coverage',
                       str(coverage), '--output', str(output)]
            first = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(first.returncode, 0, first.stderr)
            original = output.read_bytes()
            self.assertEqual(json.loads(original)['metrics']['map_50_95'], 0.)
            second = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(output.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
