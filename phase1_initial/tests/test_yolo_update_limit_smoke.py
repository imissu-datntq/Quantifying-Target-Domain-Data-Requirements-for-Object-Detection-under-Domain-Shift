"""Exact-update runtime audit command contracts."""

import tempfile
import unittest
from pathlib import Path

import run_yolo_update_limit_smoke as smoke


class YoloUpdateLimitSmokeTests(unittest.TestCase):
    def test_both_commands_use_the_same_exact_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            for candidate in ("ssda-yolo", "sf-yolo"):
                command = smoke.build_command(candidate, Path(folder) / candidate)
                self.assertEqual(
                    command[command.index("--max-optimizer-updates") + 1], "6"
                )
                self.assertIn("--unlabeled-target", command)
                self.assertNotIn("holdout", " ".join(command).lower())


if __name__ == "__main__":
    unittest.main()
