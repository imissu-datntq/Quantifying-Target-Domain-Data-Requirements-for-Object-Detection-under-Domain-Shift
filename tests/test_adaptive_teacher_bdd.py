import csv
import json
import tempfile
import unittest
from pathlib import Path

from adaptive_teacher_bdd import dataset_specifications, load_records


class LabelPolicyTests(unittest.TestCase):
    def test_pilot_limits_are_configurable(self):
        specs = dataset_specifications(
            domain="lighting", budget=1000, seed=42,
            train_limit=1000, validation_limit=1000)
        self.assertEqual(specs["bdd_source_train"][-1], 1000)
        self.assertEqual(specs["bdd_target_unlabeled"][-1], 1000)
        self.assertEqual(specs["bdd_target_val"][-1], 1000)
        self.assertEqual(specs["bdd_finetune_train"][-1], 1000)

    def test_unlabeled_loading_needs_no_annotation_store(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / 'images/train/example.jpg'
            image.parent.mkdir(parents=True)
            image.touch()
            manifest = root / 'unlabeled.csv'
            with manifest.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['name', 'image', 'coco_image_id'])
                writer.writeheader()
                writer.writerow(dict(name='example.jpg', image='images/train/example.jpg', coco_image_id=7))
            records = load_records(root, manifest, 'train', False)
            self.assertEqual(records[0]['annotations'], [])
            self.assertEqual(records[0]['image_id'], 7)
            self.assertFalse((root / 'coco').exists())
            with self.assertRaises(ValueError):
                load_records(root, manifest, 'val', False)
            with self.assertRaises(ValueError):
                load_records(root, manifest, 'test', True)
            with self.assertRaises(ValueError):
                load_records(root, manifest, 'val', True)
            (root / 'coco').mkdir()
            coco = dict(categories=[dict(id=i, name=str(i)) for i in range(1, 11)],
                        images=[dict(id=7, file_name='example.jpg', width=1280, height=720)],
                        annotations=[dict(image_id=7, category_id=10, bbox=[1, 2, 3, 4])])
            annotation_file = root / 'coco/instances_train.json'
            annotation_file.write_text(json.dumps(coco))
            labeled = load_records(root, manifest, 'train', True)
            self.assertEqual(labeled[0]['annotations'][0]['category_id'], 9)
            # Presence of real labels must not change the unlabeled result.
            self.assertEqual(load_records(root, manifest, 'train', False)[0]['annotations'], [])
            coco['images'][0]['file_name'] = 'another.jpg'
            annotation_file.write_text(json.dumps(coco))
            with self.assertRaises(ValueError):
                load_records(root, manifest, 'train', True)


if __name__ == '__main__':
    unittest.main()
