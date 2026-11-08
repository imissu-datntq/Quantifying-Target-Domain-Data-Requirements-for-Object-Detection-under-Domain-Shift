"""Tests for the bounded SSDA adaptation launcher."""
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'ssda_benchmark', ROOT / 'run_ssda_adaptation_benchmark.py')


class SSDABenchmarkTests(unittest.TestCase):
    def setUp(self):
        module = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(module)
        self.module = module

    def test_command_is_train_only_and_fixed_geometry(self):
        command = self.module.build_command(
            python=Path('/env/bin/python'), upstream=ROOT / 'external/SSDA-YOLO',
            output=ROOT / 'runs/ssda-adapt-151-001',
            source_checkpoint=ROOT / 'weights/source.pt',
            data=ROOT / 'target.yaml', hyp=ROOT / 'hyp.yaml', epochs=5,
            max_optimizer_updates=151)
        self.assertIn('--unlabeled-target', command)
        self.assertIn('--consistency_loss', command)
        self.assertIn('--noautoanchor', command)
        self.assertIn('--notest', command)
        self.assertEqual(command[command.index('--max-optimizer-updates') + 1], '151')
        self.assertEqual(self.module.DEFAULT_PILOT_UPDATES, 151)
        self.assertEqual(command[command.index('--img-size') + 1:command.index('--img-size') + 3],
                         ['1280', '1280'])
        self.assertNotIn('holdout', ' '.join(command).lower())

    def test_status_writer_is_exclusive(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'status.json'
            self.module.write_status(path, {'status': 'running'})
            with self.assertRaises(FileExistsError):
                self.module.write_status(path, {'status': 'running'})


if __name__ == '__main__':
    unittest.main()
