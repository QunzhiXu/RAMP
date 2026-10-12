import copy
import unittest
from unittest.mock import patch

from fixtures import study
from validate_hnscc_execution import branch, compare_completion, compare_sessions, normalize_elapsed


class HNSCCEvidence(unittest.TestCase):
    def metadata(self):
        return {'method': 'harmony', 'elapsed_seconds': 12.5, 'input_sha256': 'input',
                'parameters': {'seed': 42}, 'result': {'cells': 94488}, 'output_sha256': 'output'}

    def test_hnscc_raw_contract(self):
        spec, files = branch()
        self.assertEqual(spec['input'], 'figure_HNSCC/Comparison/input/raw.csv.gz')
        self.assertEqual((spec['cells'], spec['units'], len(spec['markers'])), (94488, 77, 22))
        self.assertEqual(len(files), 23)
        self.assertFalse(any(row['path'].endswith(('values.npy', 'analysis_ready.csv.gz', 'fit.rds')) for row in files))

    def test_other_cases_rejected(self):
        for case in ('bone_marrow', 'hnscc', 'melanoma_story', 'lung_story'):
            with self.subTest(case=case), self.assertRaises(ValueError): branch(case)

    def test_analysis_ready_contract_rejected(self):
        spec, files = branch(); spec = copy.deepcopy(spec)
        spec['input'] = 'figure_HNSCC/Comparison/input/analysis_ready.csv.gz'
        with patch.object(study, 'plan', return_value=(spec, files)), self.assertRaises(ValueError): branch()

    def test_elapsed_only_accepted_without_mutation(self):
        old = self.metadata(); new = copy.deepcopy(old); new['elapsed_seconds'] = 99.5
        before = copy.deepcopy(new); compare_completion(old, new)
        self.assertEqual(new, before)

    def test_nested_elapsed_only_accepted(self):
        old = {'values_sha256': 'a', 'runtime': self.metadata()}; new = copy.deepcopy(old)
        new['runtime']['elapsed_seconds'] = 77
        self.assertEqual(compare_completion(old, new, nested=True)['only_allowed_difference'], 'runtime.elapsed_seconds')

    def test_parameter_change_rejected(self):
        old = self.metadata(); new = copy.deepcopy(old); new['parameters']['seed'] = 840
        with self.assertRaises(ValueError): compare_completion(old, new)

    def test_hash_change_rejected(self):
        old = self.metadata(); new = copy.deepcopy(old); new['output_sha256'] = 'changed'
        with self.assertRaises(ValueError): compare_completion(old, new)

    def test_nested_hash_change_rejected(self):
        old = {'values_sha256': 'a', 'runtime': self.metadata()}; new = copy.deepcopy(old)
        new['values_sha256'] = 'changed'
        with self.assertRaises(ValueError): compare_completion(old, new, nested=True)

    def test_extra_field_rejected(self):
        old = self.metadata(); new = copy.deepcopy(old); new['unknown'] = 'extra'
        with self.assertRaises(ValueError): compare_completion(old, new)

    def test_invalid_elapsed_rejected(self):
        for value in (-1, float('inf'), float('nan'), True, '12'):
            old = self.metadata(); old['elapsed_seconds'] = value
            with self.subTest(value=value), self.assertRaises(ValueError): normalize_elapsed(old)

    def test_missing_elapsed_rejected(self):
        old = self.metadata(); del old['elapsed_seconds']
        with self.assertRaises(KeyError): normalize_elapsed(old)

    def test_macos_release_only_accepted(self):
        result = compare_sessions('R 4.3.2\nRunning under: macOS 26.6.1\nharmony 1.2.3',
                                  'R 4.3.2\nRunning under: macOS 26.7.1\nharmony 1.2.3')
        self.assertTrue(result['equal_except_Running_under_line'])

    def test_package_change_rejected(self):
        with self.assertRaises(ValueError):
            compare_sessions('Running under: macOS 26.6.1\nharmony 1.2.3', 'Running under: macOS 26.7.1\nharmony 1.2.4')

    def test_non_macos_or_missing_or_duplicate_os_rejected(self):
        old = 'Running under: macOS 26.6.1\nlocale C.UTF-8'
        for new in ('Running under: Linux\nlocale C.UTF-8', 'locale C.UTF-8', old + '\nRunning under: macOS 26.7.1'):
            with self.subTest(new=new), self.assertRaises(ValueError): compare_sessions(old, new)

    def test_locale_change_rejected(self):
        with self.assertRaises(ValueError):
            compare_sessions('Running under: macOS 26.6.1\nlocale C.UTF-8', 'Running under: macOS 26.7.1\nlocale C')


if __name__ == '__main__': unittest.main()
