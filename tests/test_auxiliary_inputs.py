import unittest
from prepare_auxiliary_inputs import validate_domain


class AuxiliaryInputTests(unittest.TestCase):
    def setUp(self):
        self.canonical = {'a.jpg': dict(domains='source'), 'b.jpg': dict(domains='lighting')}
        self.row = dict(name='a.jpg', image='images/train/a.jpg')

    def test_exact_domain(self):
        self.assertEqual(validate_domain([self.row], self.canonical, 'source', 1), [self.row])

    def test_rejects_validation_paths_wrong_domains_and_duplicates(self):
        for rows in ([dict(self.row, image='images/val/a.jpg')],
                     [dict(name='b.jpg', image='images/train/b.jpg')],
                     [self.row, self.row]):
            with self.assertRaises(ValueError):
                validate_domain(rows, self.canonical, 'source', 1)


if __name__ == '__main__':
    unittest.main()
