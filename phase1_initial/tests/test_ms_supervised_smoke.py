"""Static guards for the MS supervised source/fine-tune adapter."""

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT / "external/MS-DAYOLO").is_dir(),
                     "Pinned MS-DAYOLO checkout required")
class SupervisedSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "ms_supervised", ROOT / "run_ms_supervised_smoke.py")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.upstream = ROOT / "external/MS-DAYOLO"

    def test_native_supervised_config_is_not_da(self):
        config = self.module.adapt_config(
            (self.upstream / "cfg/ms-dayolo.cfg").read_text(), 6, 4096)
        self.assertEqual(config.count("classes=10"), 3)
        self.assertEqual(config.count("filters=45"), 3)
        for setting in ("width=1280", "height=736", "batch=2",
                        "subdivisions=2", "max_batches=6",
                        "optimized_memory=2", "pinned_memory_MB=4096"):
            self.assertIn(setting, config)

    def test_supervised_path_has_no_da_switch_or_target_truth(self):
        source = (ROOT / "run_ms_supervised_smoke.py").read_text()
        self.assertIn('adaptation=False', source)
        self.assertNotIn('"-da"', source)
        self.assertIn('row["yolo_label"]', source)
        self.assertIn('holdout_access=False', source)


if __name__ == "__main__":
    unittest.main()
