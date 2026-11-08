"""Metadata contract for the bounded SF adaptation input view."""
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'prepare_sf_adapt', ROOT / 'prepare_sf_adapt_smoke.py')


class SFPreparationTests(unittest.TestCase):
    def test_report_explicitly_excludes_holdout(self):
        module = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(module)
        report = module.preparation_report(
            train_images=1000, validation_images=1000,
            source_smoke=ROOT / 'runs/source')
        self.assertIs(report['holdout_access'], False)
        self.assertEqual(report['source_train_images'], 1000)
        self.assertEqual(report['target_train_images'], 1000)


if __name__ == '__main__':
    unittest.main()
