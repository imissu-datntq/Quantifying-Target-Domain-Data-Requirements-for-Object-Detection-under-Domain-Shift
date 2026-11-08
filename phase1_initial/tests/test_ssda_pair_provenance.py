import json
import tempfile
import unittest
from pathlib import Path

from evaluate_validation import digest
from prepare_ssda_adapt_smoke import verify_pairs


class PairProvenanceTests(unittest.TestCase):
    def test_upstream_source_loader_uses_normalized_path_groups(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / 'external/SSDA-YOLO/utils/datasets.py').read_text()
        self.assertIn('zip(self.path, self.path_fake)', source)
        self.assertNotIn('zip(path, path_fake)', source)

    def test_membership_direction_and_content_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / 'images/train/a.jpg'
            original.parent.mkdir(parents=True)
            original.write_bytes(b'original')
            pairs = root / 'translated'
            pairs.mkdir()
            translated = pairs / 'a.jpg'
            translated.write_bytes(b'translated')
            rows = [dict(name='a.jpg', image='images/train/a.jpg')]
            report = dict(status='completed', direction='AtoB', split='train',
                          production_asset=False, images=[dict(name='a.jpg',
                          input_sha256=digest(original), output_sha256=digest(translated))])
            (pairs / 'provenance.json').write_text(json.dumps(report))
            verify_pairs(pairs, rows, 'AtoB', root)
            with self.assertRaises(ValueError):
                verify_pairs(pairs, rows, 'BtoA', root)
            with self.assertRaises(ValueError):
                verify_pairs(pairs, [], 'AtoB', root)
            translated.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify_pairs(pairs, rows, 'AtoB', root)


if __name__ == '__main__':
    unittest.main()
