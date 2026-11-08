"""MS export boundary: head order, original pixels, complete coverage."""
import copy
import importlib.util
import json
import io
import struct
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / 'export_ms_validation.py').is_file(),
                        'MS canonical exporter is not implemented')
        spec = importlib.util.spec_from_file_location('ms_export', ROOT / 'export_ms_validation.py')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.ids = [9007199254740993, 9007199254740995]
        self.classes = [{'yolo_id': i, 'coco_id': i + 1} for i in reversed(range(10))]
        self.records = [dict(index=0, width=1280, height=720,
                             detections=[[0, .9, -10, -20, 1300, 740],
                                         [9, .8, 10, 20, 30, 50]]),
                        dict(index=1, width=1280, height=720, detections=[])]

    def convert(self, records=None):
        return self.module.convert_records(self.records if records is None else records,
                                           self.ids, self.classes)

    def test_head_mapping_clipping_large_ids_and_empty_coverage(self):
        predictions, covered = self.convert()
        self.assertEqual(covered, [9007199254740993, 9007199254740995])
        self.assertEqual(predictions, [
            dict(image_id=9007199254740993, category_id=1,
                 score=.9, bbox=[0, 0, 1280, 720]),
            dict(image_id=9007199254740993, category_id=10,
                 score=.8, bbox=[10, 20, 20, 30])])

    def test_reject_partial_duplicate_foreign_and_wrong_geometry(self):
        for records in [self.records[:1], self.records * 2, self.records[::-1],
                        [dict(self.records[0], index=2), self.records[1]],
                        [dict(self.records[0], height=736), self.records[1]]]:
            with self.subTest(records=records), self.assertRaises(ValueError):
                self.convert(records)

    def test_reject_nonfinite_and_invalid_class_score(self):
        for position, value in [(0, -1), (0, 10), (0, .5), (0, True),
                                (1, -1), (1, 1.1), (1, float('nan')),
                                (2, float('inf')), (3, '20')]:
            records = copy.deepcopy(self.records)
            records[0]['detections'][0][position] = value
            with self.subTest(position=position, value=value), self.assertRaises(ValueError):
                self.convert(records)

    def test_top100_after_filtering_degenerate_boxes(self):
        records = copy.deepcopy(self.records)
        records[0]['detections'] = [[0, i / 200, 0, 0, 20, 20] for i in range(150)]
        records[0]['detections'].append([0, 1, 1300, 0, 1400, 20])
        predictions, _ = self.convert(records)
        self.assertEqual(len(predictions), 100)
        self.assertEqual(predictions[0]['score'], .745)
        self.assertEqual(predictions[-1]['score'], .25)

    def test_reject_invalid_class_metadata_and_duplicate_ids(self):
        self.classes[0]['yolo_id'] = 0
        with self.assertRaises(ValueError):
            self.convert()

    def test_weights_audit_rejects_truncation_nonfinite_and_wrong_header(self):
        self.assertTrue(hasattr(self.module, 'audit_weights'), 'Missing strict weight audit')
        valid = struct.pack('<iiiQ3f', 0, 2, 5, 480, 1, 2, 3)
        self.assertEqual(self.module.audit_weights(io.BytesIO(valid), 3, 480), 3)
        for payload in [valid[:20], valid[:-1], valid + b'xxxx',
                        struct.pack('<iiiQ3f', 0, 2, 5, 480, 1, float('nan'), 3),
                        struct.pack('<iiiQ3f', 0, 1, 5, 480, 1, 2, 3),
                        struct.pack('<iiiQ3f', 0, 2, 5, 0, 1, 2, 3)]:
            with self.subTest(length=len(payload)), self.assertRaises(ValueError):
                self.module.audit_weights(io.BytesIO(payload), 3, 480)
        self.classes = [{'yolo_id': i, 'coco_id': i + 1} for i in range(10)]
        self.ids[1] = self.ids[0]
        with self.assertRaises(ValueError):
            self.convert()


if __name__ == '__main__':
    unittest.main()
