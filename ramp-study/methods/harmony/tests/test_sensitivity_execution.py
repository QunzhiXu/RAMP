import unittest

import numpy as np
import pandas as pd
from fixtures import study
from validate_sensitivity_execution import branch, audit_transform, CASES, PROFILE


class SensitivityEvidence(unittest.TestCase):
    def pair(self):
        source = pd.DataFrame({'cell_id': ['a', 'b', 'c'], 'dataset': ['A', 'A', 'B'],
                               'unit_key': ['u', 'u', 'v'], 'M': [-8.0, 0.0, 16.0]})
        fixed = source.copy(); fixed['M'] = np.arcsinh(1 + source['M'] / 5)
        return source, fixed

    def test_four_fixed_contracts(self):
        for case, counts in CASES.items():
            spec, files = branch(case)
            self.assertEqual(spec['profile'], PROFILE)
            self.assertEqual((spec['cells'], len(spec['markers']), spec['units']), counts)
            self.assertEqual(len(files), 21)

    def test_story_and_unknown_rejected(self):
        for case in ('hnscc_story', 'melanoma_story', 'lung_story', 'unknown'):
            with self.subTest(case=case), self.assertRaises(ValueError): branch(case)

    def test_single_transform_accepts_signed_values(self):
        a, b = self.pair(); before = b.copy(deep=True)
        self.assertEqual(audit_transform(a, b, ['M'])['maximum_absolute_difference'], 0)
        pd.testing.assert_frame_equal(b, before)

    def test_missing_transform_rejected(self):
        a, _ = self.pair()
        with self.assertRaises(ValueError): audit_transform(a, a.copy(), ['M'])

    def test_double_transform_rejected(self):
        a, b = self.pair(); b['M'] = np.arcsinh(1 + b['M'] / 5)
        with self.assertRaises(ValueError): audit_transform(a, b, ['M'])

    def test_row_reorder_rejected(self):
        a, b = self.pair()
        with self.assertRaises(ValueError): audit_transform(a, b.iloc[::-1].reset_index(drop=True), ['M'])

    def test_marker_order_rejected(self):
        a, b = self.pair()
        with self.assertRaises(ValueError): audit_transform(a, b[['M', 'cell_id', 'dataset', 'unit_key']], ['M'])

    def test_nonfinite_rejected(self):
        a, b = self.pair(); b.loc[0, 'M'] = np.nan
        with self.assertRaises(ValueError): audit_transform(a, b, ['M'])

    def test_unit_change_rejected(self):
        a, b = self.pair(); b.loc[0, 'unit_key'] = 'changed'
        with self.assertRaises(ValueError): audit_transform(a, b, ['M'])

    def test_other_formula_rejected(self):
        a, b = self.pair(); b['M'] = np.arcsinh(a['M'] / 5)
        with self.assertRaises(ValueError): audit_transform(a, b, ['M'])


if __name__ == '__main__': unittest.main()
