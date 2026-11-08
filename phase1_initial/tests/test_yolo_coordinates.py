import unittest
from export_yolo_validation import unpad_box


class CoordinatesTests(unittest.TestCase):
    def test_native_round_trip(self):
        self.assertEqual(unpad_box([100, 208, 180, 268]), [100, 200, 80, 60])

    def test_padding_and_clipping(self):
        self.assertEqual(unpad_box([-5, 0, 1290, 736]), [0, 0, 1280, 720])
        self.assertIsNone(unpad_box([10, 0, 20, 8]))
        with self.assertRaises(ValueError):
            unpad_box([0, 0, float('nan'), 20])


if __name__ == '__main__':
    unittest.main()
