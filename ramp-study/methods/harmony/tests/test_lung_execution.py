import copy
import unittest
from unittest.mock import patch

from fixtures import study
import validate_lung_execution as lung


class LungEvidence(unittest.TestCase):
    def test_current_lung_story_contract(self):
        spec, files = lung.branch()
        self.assertEqual(spec['input'], 'story_lung_HLA_ABC/Comparison_senosain140k_20260921/input/raw.csv.gz')
        self.assertEqual((spec['cells'], spec['units'], len(spec['markers'])), (84000, 84, 9))
        self.assertEqual(len(files), 23)

    def test_three_platforms_include_zheng_not_monkman(self):
        spec, _ = lung.branch()
        self.assertEqual(spec['datasets'], ['Senosain CyTOF', 'Cords IMC', 'Zheng mIF'])

    def test_other_cases_rejected(self):
        for case in ('lung_cancer', 'hnscc_story', 'melanoma_story', 'bone_marrow', 'unknown'):
            with self.subTest(case=case), self.assertRaises(ValueError): lung.branch(case)

    def test_bundle_excludes_corrected_outputs_and_analysis_ready(self):
        _, files = lung.branch()
        self.assertFalse(any(row['path'].endswith(('values.npy', 'analysis_ready.csv.gz', 'fit.rds',
                                                  'corrected_expression.csv.gz')) for row in files))
        self.assertEqual(sum(row['origin'] == 'source' for row in files), 6)
        self.assertEqual(sum(row['path'].startswith('adapter/') for row in files), 16)
        self.assertEqual(sum(row['path'] == 'scripts/run_r.R' for row in files), 1)

    def rejected_change(self, key, value):
        spec, files = lung.branch(); spec = copy.deepcopy(spec); spec[key] = value
        with patch.object(study, 'plan', return_value=(spec, files)), self.assertRaises(ValueError): lung.branch()

    def test_analysis_ready_rejected(self):
        self.rejected_change('input', lung.BASE + '/input/analysis_ready.csv.gz')

    def test_earlier_comparison_rejected(self):
        self.rejected_change('base', 'story_lung_HLA_ABC/Comparison')

    def test_wrong_cohort_rejected(self):
        self.rejected_change('cells', 60000)

    def test_wrong_units_rejected(self):
        self.rejected_change('units', 30)

    def test_wrong_panel_rejected(self):
        spec, _ = lung.branch(); self.rejected_change('markers', spec['markers'][:8])

    def test_technical_kind_rejected(self):
        self.rejected_change('kind', 'technical')

    def test_monkman_substitution_rejected(self):
        self.rejected_change('datasets', ['Senosain CyTOF', 'Cords IMC', 'Monkman CODEX'])

    def test_platform_order_rejected(self):
        self.rejected_change('datasets', ['Zheng mIF', 'Cords IMC', 'Senosain CyTOF'])

    def test_legacy_message_relabel_retains_rejection_and_cause(self):
        original = ValueError('Non-identical core HNSCC result')
        with patch.object(lung, 'shared_story_comparison', side_effect=original):
            with self.assertRaisesRegex(ValueError, '^Non-identical core lung_story result$') as caught:
                lung.story_comparison(None, None, {'case': 'lung_story'})
        self.assertIs(caught.exception.__cause__, original)

    def test_other_errors_propagate_unchanged(self):
        original = ValueError('Scientific parameter differs')
        with patch.object(lung, 'shared_story_comparison', side_effect=original):
            with self.assertRaises(ValueError) as caught:
                lung.story_comparison(None, None, {'case': 'lung_story'})
        self.assertIs(caught.exception, original)

    def test_success_evidence_is_not_modified(self):
        result = {'core_byte_identical': 7, 'npy_maximum_absolute_difference': 0.0}
        with patch.object(lung, 'shared_story_comparison', return_value=result):
            self.assertIs(lung.story_comparison(None, None, {'case': 'lung_story'}), result)


if __name__ == '__main__': unittest.main()
