"""Contract tests for the bounded 1,000-image SF-YOLO source run."""
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'sf_source_pilot', ROOT / 'run_sf_source_pilot.py')


class SFSourcePilotTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / 'run_sf_source_pilot.py').is_file())
        module = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(module)
        self.module = module

    def test_command_is_one_epoch_fixed_geometry_without_holdout(self):
        command = self.module.build_command(
            python=Path('/env/bin/python'),
            upstream=ROOT / 'external/sf-yolo',
            output=ROOT / 'runs/pilot1000-sf-source',
            weights=ROOT / 'weights/yolov5l-v7.0.pt',
            data=ROOT / 'runs/pilot1000-sf-source-inputs/data.yaml',
            hyp=ROOT / 'runs/pilot1000-sf-source-inputs/hyp.yaml')
        joined = ' '.join(map(str, command)).lower()
        self.assertEqual(command[command.index('--epochs') + 1], '1')
        self.assertEqual(command[command.index('--batch-size') + 1], '1')
        self.assertEqual(command[command.index('--imgsz') + 1], '1280')
        self.assertIn('--rect', command)
        self.assertIn('--technical-no-amp', command)
        self.assertIn('--noautoanchor', command)
        self.assertIn('--noplots', command)
        self.assertNotIn('test', joined)
        self.assertNotIn('holdout', joined)

    def test_status_writer_is_exclusive(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'status.json'
            self.module.write_status(path, {'status': 'running'})
            with self.assertRaises(FileExistsError):
                self.module.write_status(path, {'status': 'running'})


if __name__ == '__main__':
    unittest.main()
