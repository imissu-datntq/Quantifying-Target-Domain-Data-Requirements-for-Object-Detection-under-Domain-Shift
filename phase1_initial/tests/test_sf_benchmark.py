"""SF bounded-pilot launcher: fixed geometry and label-free target."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('sf_benchmark', ROOT / 'run_sf_adaptation_benchmark.py')


class SFBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / 'run_sf_adaptation_benchmark.py').is_file())
        module = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(module)
        self.module = module

    def test_command_has_no_holdout_and_enforces_research_geometry(self):
        command = self.module.build_command(
            python=Path('/env/bin/python'), upstream=ROOT / 'external/sf-yolo',
            output=ROOT / 'runs/sf-adapt-151-001',
            source_checkpoint=ROOT / 'runs/source.pt', data=ROOT / 'target.yaml',
            hyp=ROOT / 'hyp.yaml', decoder=ROOT / 'decoder.pth',
            encoder=ROOT / 'encoder.pth', fc1=ROOT / 'fc1.pth', fc2=ROOT / 'fc2.pth',
            epochs=8, max_optimizer_updates=151)
        joined = ' '.join(map(str, command))
        self.assertIn('--unlabeled-target', command)
        self.assertIn('--noautoanchor', command)
        self.assertIn('--noplots', command)
        self.assertIn('--imgsz', command)
        self.assertEqual(command[command.index('--max-optimizer-updates') + 1], '151')
        self.assertEqual(self.module.DEFAULT_PILOT_UPDATES, 151)
        self.assertEqual(command[command.index('--imgsz') + 1], '1280')
        self.assertNotIn('test', joined.lower())
        self.assertNotIn('holdout', joined.lower())
        self.assertEqual(command[command.index('--epochs') + 1], '8')

    def test_status_writer_is_exclusive(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'status.json'
            self.module.write_status(path, {'status': 'running'})
            with self.assertRaises(FileExistsError):
                self.module.write_status(path, {'status': 'running'})


if __name__ == '__main__':
    unittest.main()
