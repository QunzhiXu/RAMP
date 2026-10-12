from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import write, HERE
from compare import array_comparison, tree_differences, json_compare, required_payload
from prepare import verify, LOCK_SHA


class Comparison(unittest.TestCase):
    def test_identical_array(self):
        a = np.arange(12, dtype=np.float32).reshape(4, 3)
        self.assertTrue(array_comparison(a, a.copy())['equal'])

    def test_dtype_change(self):
        a = np.arange(3, dtype=np.float32)
        self.assertFalse(array_comparison(a, a.astype(float))['equal'])

    def test_one_value_change(self):
        a = np.arange(3, dtype=float); b = a.copy(); b[1] += .75
        result = array_comparison(a, b)
        self.assertFalse(result['equal']); self.assertEqual(result['max_abs_difference'], .75)

    def test_shape_change(self):
        self.assertFalse(array_comparison(np.zeros((4, 3)), np.zeros((3, 4)))['equal'])

    def test_bool_arrays(self):
        self.assertTrue(array_comparison(np.array([True, False]), np.array([True, False]))['equal'])

    def test_tree_missing(self):
        self.assertEqual(tree_differences({'a': 1}, {})[0]['field'], '/a')

    def test_tree_nested_difference(self):
        self.assertEqual(tree_differences({'a': [1, 2]}, {'a': [1, 3]})[0]['field'], '/a/1')

    def test_json_only_declared_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); new = root / 'new'; old = root / 'old'
            write(new, {'path': '/newtree/input', 'seconds': 2., 'weights': [.5, .5], 'hash': 'a' * 64})
            write(old, {'path': '/oldtree/input', 'seconds': 3., 'weights': [.5, .5], 'hash': 'b' * 64})
            r = json_compare(new, old, Path('/newtree'), Path('/oldtree'), {('a' * 64, 'b' * 64)})
            self.assertTrue(r['equal_after_declared_normalization'])
            self.assertEqual(len(r['provenance_differences']), 2)

    def test_scientific_parameter_not_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); new = root / 'new'; old = root / 'old'
            write(new, {'seed': 43}); write(old, {'seed': 42})
            r = json_compare(new, old, Path('/new'), Path('/old'), set())
            self.assertFalse(r['equal_after_declared_normalization'])

    def test_unrecognized_hash_not_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); new = root / 'new'; old = root / 'old'
            write(new, {'hash': 'a' * 64}); write(old, {'hash': 'b' * 64})
            self.assertFalse(json_compare(new, old, Path('/new'), Path('/old'), set())['equal_after_declared_normalization'])

    def test_required_prediction_grid(self):
        required = required_payload()
        self.assertEqual(sum(p.endswith('_prediction.npz') for p in required), 15)
        self.assertEqual(sum(p.endswith('.pt') for p in required), 15)
        self.assertIn('story_cd19_prediction/umap/data/umap_coordinates.npz', required)

    def test_actual_lock(self):
        lock = verify(HERE / 'frozen_20261011')
        self.assertEqual(len(lock['files']), 254)

    def test_unknown_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(Path(tmp) / 'LOCK.json', {})
            with self.assertRaisesRegex(ValueError, 'Unrecognized'):
                verify(tmp)


if __name__ == '__main__':
    unittest.main()
