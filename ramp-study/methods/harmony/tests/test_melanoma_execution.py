import copy
import unittest
from unittest.mock import patch

from fixtures import study
from validate_melanoma_execution import branch


class MelanomaEvidence(unittest.TestCase):
    def test_story_raw_contract(self):
        spec, files = branch()
        self.assertEqual(spec['input'], 'figure_melanoma/Comparison/input/raw.csv.gz')
        self.assertEqual((spec['cells'], spec['units'], len(spec['markers'])), (165686, 38, 15))
        self.assertEqual(len(files), 23)

    def test_only_two_platforms(self):
        spec, _ = branch()
        self.assertEqual(spec['datasets'], ['Moldoveanu IMC', 'Engelbrecht CITE-seq'])

    def test_other_cases_rejected(self):
        for case in ('melanoma', 'hnscc_story', 'lung_story', 'bone_marrow', 'unknown'):
            with self.subTest(case=case), self.assertRaises(ValueError): branch(case)

    def test_bundle_excludes_corrected_outputs_and_analysis_ready(self):
        _, files = branch()
        self.assertFalse(any(row['path'].endswith(('values.npy', 'analysis_ready.csv.gz', 'fit.rds',
                                                  'corrected_expression.csv.gz')) for row in files))
        self.assertEqual(sum(row['origin'] == 'source' for row in files), 6)
        self.assertEqual(sum(row['path'].startswith('adapter/') for row in files), 16)
        self.assertEqual(sum(row['path'] == 'scripts/run_r.R' for row in files), 1)

    def rejected_change(self, key, value):
        spec, files = branch(); spec = copy.deepcopy(spec); spec[key] = value
        with patch.object(study, 'plan', return_value=(spec, files)), self.assertRaises(ValueError): branch()

    def test_analysis_ready_contract_rejected(self):
        self.rejected_change('input', 'figure_melanoma/Comparison/input/analysis_ready.csv.gz')

    def test_wrong_cohort_rejected(self):
        self.rejected_change('cells', 60000)

    def test_wrong_units_rejected(self):
        self.rejected_change('units', 46)

    def test_wrong_marker_panel_rejected(self):
        spec, _ = branch(); self.rejected_change('markers', spec['markers'][:12])

    def test_technical_kind_rejected(self):
        self.rejected_change('kind', 'technical')

    def test_platform_order_rejected(self):
        self.rejected_change('datasets', ['Engelbrecht CITE-seq', 'Moldoveanu IMC'])


if __name__ == '__main__': unittest.main()
