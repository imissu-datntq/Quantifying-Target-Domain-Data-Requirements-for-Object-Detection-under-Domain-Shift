"""Static regression guards; runtime compilation/gradient audit remain separate."""
import re
import unittest
from pathlib import Path
from run_ms_dayolo_smoke import adapt_config


ROOT = Path(__file__).resolve().parents[1] / 'external/MS-DAYOLO'


@unittest.skipUnless(ROOT.is_dir(), 'Pinned MS-DAYOLO checkout required')
class TargetPolicyTests(unittest.TestCase):
    def test_native_ten_class_smoke_config(self):
        cfg = adapt_config((ROOT / 'cfg/ms-dayolo.cfg').read_text(), 6, 2)
        self.assertEqual(cfg.count('classes=10'), 3)
        self.assertEqual(cfg.count('filters=45'), 3)
        for setting in ('width=1280', 'height=736', 'batch=4', 'subdivisions=2',
                        'letter_box=1', 'max_batches=6', 'optimized_memory=2'):
            self.assertIn(setting, cfg)

    def test_both_loader_implementations_guard_annotation_reads(self):
        text = (ROOT / 'src/data.c').read_text()
        branches = text.split('data load_data_detection(')[1:]
        self.assertEqual(len(branches), 2)
        for branch in branches:
            branch = branch.split('\nvoid ')[0]
            calls = [line.strip() for line in branch.splitlines()
                     if 'fill_truth_detection(filename' in line]
            self.assertEqual(len(calls), 1)
            self.assertTrue('unlabeled ? 0 :' in calls[0]
                            or 'if (!unlabeled)' in calls[0])
        detector = (ROOT / 'src/detector.c').read_text()
        self.assertIn('args_target.unlabeled = 1;', detector)

    def test_detection_heads_exclude_target_and_batch_uses_network_size(self):
        cfg = (ROOT / 'cfg/ms-dayolo.cfg').read_text()
        sections = re.split(r'\[([^\]]+)\]', cfg)
        heads = [sections[i + 1] for i in range(1, len(sections), 2)
                 if sections[i] == 'yolo']
        self.assertEqual(len(heads), 3)
        for head in heads:
            self.assertRegex(head, r'use_target\s*=\s*0')
        parser = (ROOT / 'src/parser.c').read_text()
        self.assertIn('l.use_target==0 && net.batch!=1', parser)
        self.assertNotIn('l.use_target==0 && params.batch!=1', parser)


if __name__ == '__main__':
    unittest.main()
