import unittest

from fixtures import study
from validate_technical_execution import branch, changed_payload_hashes, CASES


class TechnicalExecutionEvidence(unittest.TestCase):
    def test_three_source_scale_contracts(self):
        for case, counts in CASES.items():
            spec, files = branch(case)
            self.assertEqual((spec['cells'], len(spec['markers']), spec['units']), counts)
            self.assertEqual(spec['profile'], 'source_scale')
            self.assertEqual(len(files), 21)

    def test_other_rounds_rejected(self):
        for case in ('bone_marrow', 'hnscc_story', 'melanoma_story', 'lung_story', 'unknown'):
            with self.subTest(case=case), self.assertRaises(ValueError): branch(case)

    def test_lung_is_monkman_not_zheng(self):
        spec, _ = branch('lung_cancer')
        self.assertEqual(spec['datasets'], ['Senosain CyTOF', 'Cords IMC', 'Monkman CODEX'])

    def test_two_named_provenance_pairs(self):
        result = changed_payload_hashes({'payloads': {
            'sessionInfo.txt': {'bytes_equal': False, 'historical': {'sha256': 'a'}, 'current': {'sha256': 'b'}},
            'resolved_config.json': {'bytes_equal': False, 'historical': {'sha256': 'c'}, 'current': {'sha256': 'd'}},
            'corrected.csv': {'bytes_equal': True}}})
        self.assertEqual(result, {'sessionInfo.txt': {'historical_sha256': 'a', 'current_sha256': 'b'},
                                  'resolved_config.json': {'historical_sha256': 'c', 'current_sha256': 'd'}})

    def test_scientific_payload_difference_rejected(self):
        with self.assertRaises(ValueError):
            changed_payload_hashes({'payloads': {'marker_rank_audit.csv': {
                'bytes_equal': False, 'historical': {'sha256': 'a'}, 'current': {'sha256': 'b'}}}})

    def test_inconsistent_hash_pair_rejected(self):
        with self.assertRaises(ValueError):
            changed_payload_hashes({'payloads': {'sessionInfo.txt': {
                'bytes_equal': False, 'historical': {'sha256': 'a'}, 'current': {'sha256': 'a'}}}})

    def test_identical_payloads_not_normalized(self):
        self.assertEqual(changed_payload_hashes({'payloads': {'sessionInfo.txt': {'bytes_equal': True}}}), {})


if __name__ == '__main__': unittest.main()
