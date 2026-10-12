from pathlib import Path
import tempfile
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inventory as i
import compare_execution as c


class Comparison(unittest.TestCase):
    def test_equal_arrays(self):
        x = np.arange(12.).reshape(3, 4)
        self.assertTrue(c.numerical_difference(x, x.copy())['array_equal'])

    def test_changed_array_reported(self):
        x = np.arange(12.).reshape(3, 4); y = x.copy(); y[0, 0] = .01
        self.assertFalse(c.numerical_difference(x, y)['array_equal'])
        self.assertEqual(c.numerical_difference(x, y)['max_absolute_difference'], .01)

    def test_dtype_difference_rejected(self):
        with self.assertRaises(ValueError): c.numerical_difference(np.zeros(2), np.zeros(2, dtype='float32'))

    def test_nonfinite_pattern_rejected(self):
        with self.assertRaises(ValueError): c.numerical_difference(np.array([np.nan]), np.array([0.]))

    def test_unknown_result_json_rejected(self):
        with self.assertRaises(ValueError): c.json_check('unknown.json', None, None, None, None)

    def test_scientific_field_change_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); a = root / 'new.json'; b = root / 'old.json'
            i.write(a, {'status': 'PASS', 'n_cells': 10}); i.write(b, {'status': 'PASS', 'n_cells': 11})
            with self.assertRaises(ValueError): c.json_check(i.VALUE + '/summary/VERIFICATION.json', a, b, root, root)

    def test_additional_field_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); a = root / 'new.json'; b = root / 'old.json'
            i.write(a, {'status': 'PASS', 'extra': 0}); i.write(b, {'status': 'PASS'})
            with self.assertRaises(ValueError): c.json_check(i.VALUE + '/summary/VERIFICATION.json', a, b, root, root)

    def test_time_only_permitted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); a = root / 'new.json'; b = root / 'old.json'
            i.write(a, {'status': 'PASS', 'runtime_seconds': 2, 'input_sha256': {}})
            i.write(b, {'status': 'PASS', 'runtime_seconds': 1, 'input_sha256': {}})
            result = c.json_check(i.PERCENTILE + '/computation_manifest.json', a, b, root, root)
            self.assertTrue(result['exact_other_fields'])


if __name__ == '__main__': unittest.main()
