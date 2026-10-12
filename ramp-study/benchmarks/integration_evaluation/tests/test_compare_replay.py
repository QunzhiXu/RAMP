import unittest
import copy
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import compare_replay as comparison


class NumericalComparison(unittest.TestCase):
    def test_identical(self):
        result = comparison.compare_values([0., .5, 1.], [0., .5, 1.])
        self.assertEqual(result['exact_values'], 3)
        self.assertEqual(result['maximum_absolute_difference'], 0.)

    def test_single_ulp_reported_not_hidden(self):
        result = comparison.compare_values([np.nextafter(.5, 1.)], [.5])
        self.assertEqual(result['exact_values'], 0)
        self.assertGreater(result['maximum_absolute_difference'], 0.)

    def test_scientific_difference_rejected(self):
        with self.assertRaises(ValueError): comparison.compare_values([.501], [.5])

    def test_shape_rejected(self):
        with self.assertRaises(ValueError): comparison.compare_values([1., 2.], [1.])

    def test_mismatch_missing_rejected(self):
        with self.assertRaises(ValueError): comparison.compare_values([np.nan], [0.])

    def test_matching_missing_explicit(self):
        self.assertEqual(comparison.compare_values([np.nan], [np.nan])['missing_values'], 1)

    def test_infinite_rejected(self):
        with self.assertRaises(ValueError): comparison.compare_values([np.inf], [np.inf])


class BatchBinding(unittest.TestCase):
    def setUp(self):
        self.manifest = {'cases': [{'case': 'example', 'variants': [{'id': 'v00'}, {'id': 'v01'}]}]}
        self.report = {'code_records': {'test': 'locked'}, 'branches': [
            {'case': 'example', 'variant': 'v00'}, {'case': 'example', 'variant': 'v01'}]}

    def test_grid_accepted(self):
        with patch.object(comparison.r, 'code_records', return_value={'test': 'locked'}):
            self.assertEqual(len(comparison.branch_records(self.manifest, self.report)), 2)

    def test_duplicate_rejected(self):
        self.report['branches'][1] = self.report['branches'][0]
        with self.assertRaisesRegex(ValueError, 'grid'): comparison.branch_records(self.manifest, self.report)

    def test_missing_rejected(self):
        self.report['branches'].pop()
        with self.assertRaisesRegex(ValueError, 'grid'): comparison.branch_records(self.manifest, self.report)

    def test_foreign_case_rejected(self):
        self.report['branches'][0]['case'] = 'foreign'
        with self.assertRaisesRegex(ValueError, 'grid'): comparison.branch_records(self.manifest, self.report)

    def test_unknown_controller_rejected(self):
        with patch.object(comparison.r, 'code_records', return_value={'test': 'other'}):
            with self.assertRaisesRegex(ValueError, 'controller'): comparison.branch_records(self.manifest, self.report)

    def record_fixture(self, root):
        run = root / 'example/v00'; run.mkdir(parents=True)
        parent = {'code_records': {'test': 'locked'}, 'python': '/recorded/python', 'bundle': '/recorded/bundle'}
        request = {**parent, 'case': 'example', 'variant': 'v00'}
        comparison.r.write(run / 'REQUEST.json', request)
        comparison.r.write(run / 'RUN_MANIFEST.json', {'files': {}})
        command = [parent['python'], '-B', str(root / 'controller/replay_worker.py'), '--run', str(run)]
        comparison.r.write(run / 'COMMAND.json', {'argv': command})
        entry = {'case': 'example', 'variant': 'v00', 'manifest': comparison.r.record(run / 'RUN_MANIFEST.json')}
        return run, entry, parent

    def test_matching_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); run, entry, parent = self.record_fixture(root)
            self.assertEqual(comparison.check_run_record(root, run, entry, parent)['case'], 'example')

    def test_changed_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); run, entry, parent = self.record_fixture(root)
            entry['manifest']['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'Changed file'): comparison.check_run_record(root, run, entry, parent)

    def test_interchanged_record_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); run, entry, parent = self.record_fixture(root)
            entry['variant'] = 'v01'
            with self.assertRaisesRegex(ValueError, 'identity'): comparison.check_run_record(root, run, entry, parent)

    def test_changed_parent_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); run, entry, parent = self.record_fixture(root)
            parent['bundle'] = '/other/bundle'
            with self.assertRaisesRegex(ValueError, 'request'): comparison.check_run_record(root, run, entry, parent)

    def test_foreign_command_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); run, entry, parent = self.record_fixture(root)
            with patch.object(comparison.r, 'read', side_effect=[
                {**parent, 'case': 'example', 'variant': 'v00'}, {'argv': ['foreign.py']} ]):
                with self.assertRaisesRegex(ValueError, 'command'): comparison.check_run_record(root, run, entry, parent)


if __name__ == '__main__': unittest.main()
