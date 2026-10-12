from pathlib import Path
import copy
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import inventory as i
import contract as c


def compete(path, barrier, queue):
    barrier.wait()
    try: i.reserve(path); queue.put('owned')
    except FileExistsError: queue.put('refused')


class Safety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name).resolve()
        self.source = self.root / 'source'; self.source.mkdir()
        (self.source / 'data.txt').write_text('frozen')
        self.catalog = {'sources': {'data.txt': i.record(self.source / 'data.txt')}, 'copies': ['data.txt'], 'hash_links': []}

    def tearDown(self): self.tmp.cleanup()

    def freeze(self):
        with patch.object(i, 'collect', return_value=copy.deepcopy(self.catalog)):
            return i.freeze(self.source, self.root / 'snapshot')

    def test_create_only(self):
        self.freeze()
        with self.assertRaises(FileExistsError): self.freeze()

    def test_moved_snapshot(self):
        result = self.freeze(); shutil.move(self.root / 'snapshot', self.root / 'with spaces')
        shutil.rmtree(self.source)
        self.assertEqual(i.verify(self.root / 'with spaces', result['catalog_sha256']), result)

    def test_partial_rejected(self):
        result = self.freeze(); path = self.root / 'snapshot/COMPLETE.json'
        data = i.read(path); data['status'] = 'partial'; path.unlink(); i.write(path, data)
        with self.assertRaises(ValueError): i.verify(self.root / 'snapshot', result['catalog_sha256'])

    def test_missing_complete_rejected(self):
        result = self.freeze(); (self.root / 'snapshot/COMPLETE.json').unlink()
        with self.assertRaises(ValueError): i.verify(self.root / 'snapshot', result['catalog_sha256'])

    def test_extra_file_rejected(self):
        result = self.freeze(); (self.root / 'snapshot/extra').touch()
        with self.assertRaises(ValueError): i.verify(self.root / 'snapshot', result['catalog_sha256'])

    def test_payload_change_rejected(self):
        result = self.freeze(); (self.root / 'snapshot/sources/data.txt').write_text('changed')
        with self.assertRaises(ValueError): i.verify(self.root / 'snapshot', result['catalog_sha256'])

    def test_wrong_anchor_rejected(self):
        self.freeze()
        with self.assertRaises(ValueError): i.verify(self.root / 'snapshot', '0' * 64)

    def test_symlink_rejected(self):
        result = self.freeze(); target = self.root / 'snapshot/sources/data.txt'; target.unlink()
        target.symlink_to(self.source / 'data.txt')
        with self.assertRaises(ValueError): i.verify(self.root / 'snapshot', result['catalog_sha256'])

    def test_overlap_before_writes(self):
        with self.assertRaises(ValueError): i.reserve(self.source / 'new/child', [self.source])
        self.assertFalse((self.source / 'new').exists())

    def test_parent_overlap(self):
        with self.assertRaises(ValueError): i.reserve(self.root / 'outside', [self.root / 'outside/source'])
        self.assertFalse((self.root / 'outside').exists())

    def test_dangling_output(self):
        link = self.root / 'link'; link.symlink_to(self.root / 'missing')
        with self.assertRaises(FileExistsError): i.reserve(link)

    def test_parallel_claim(self):
        ctx = mp.get_context('spawn'); barrier = ctx.Barrier(2); queue = ctx.Queue()
        workers = [ctx.Process(target=compete, args=(self.root / 'race', barrier, queue)) for _ in range(2)]
        for worker in workers: worker.start()
        for worker in workers: worker.join(30); self.assertEqual(worker.exitcode, 0)
        self.assertEqual(sorted([queue.get(timeout=2), queue.get(timeout=2)]), ['owned', 'refused'])

    def test_copy_mutation_no_complete(self):
        original = i.shutil.copyfileobj
        def copy_mutate(src, dst):
            original(src, dst); (self.source / 'data.txt').write_text('changed')
        with patch.object(i.shutil, 'copyfileobj', side_effect=copy_mutate), self.assertRaises(ValueError): self.freeze()
        self.assertFalse((self.root / 'snapshot/COMPLETE.json').exists())
        self.assertTrue((self.root / 'snapshot/FAILED.json').exists())

    def test_completion_is_last(self):
        operations = []; original_write = i.write; original_verify = i.verify
        def write(path, data): operations.append(Path(path).name); return original_write(path, data)
        def verify(*args, **kwargs): operations.append('verify'); return original_verify(*args, **kwargs)
        with patch.object(i, 'write', side_effect=write), patch.object(i, 'verify', side_effect=verify): self.freeze()
        self.assertEqual(operations[-1], 'COMPLETE.json')

    def test_validation_failure_no_complete(self):
        with patch.object(i, 'verify', side_effect=ValueError('payload')), self.assertRaises(ValueError): self.freeze()
        self.assertFalse((self.root / 'snapshot/COMPLETE.json').exists())

    def test_exact_json_field(self):
        owner = self.source / 'owner.json'
        i.write(owner, {'raw': 'wrong', 'notes': self.catalog['sources']['data.txt']['sha256']})
        cat = copy.deepcopy(self.catalog); cat['sources']['owner.json'] = i.record(owner)
        cat['hash_links'] = [{'owner': 'owner.json', 'fields': ['raw'], 'target': 'data.txt',
                             'sha256': cat['sources']['data.txt']['sha256']}]
        with self.assertRaisesRegex(ValueError, 'field'): i.verify_sources(cat, self.source)

    def test_unsafe_paths(self):
        for name in ('../data', '/etc/passwd', 'x/../../data'):
            with self.subTest(name=name), self.assertRaises(ValueError): i.bind(self.source, name)

    def test_historical_root_rebinding(self):
        self.assertEqual(i.bind(self.source, '/old/final_xu/data.txt'), self.source / 'data.txt')


class NumericalContracts(unittest.TestCase):
    def setUp(self):
        self.source = Path(__file__).resolve().parents[1] / 'frozen_20261011_r2/sources'

    def test_extracted_percentiles(self):
        funcs = c.pure_functions(self.source / i.PERCENTILE / 'compute_celltype_shifts.py',
                                ['percentile_matrix', 'quartiles'], {'np': np, 'rankdata': c.rankdata})
        a = np.array([[1, 2], [1, 4], [3, 1.]])
        np.testing.assert_array_equal(funcs['percentile_matrix'](a), [[37.5, 50], [37.5, 75], [75, 25]])
        np.testing.assert_array_equal(funcs['quartiles'](a), np.quantile(a, [.25, .5, .75], axis=0, method='linear'))

    def test_source_gate(self):
        funcs = c.pure_functions(self.source / i.VALUE / 'evaluate_values.py', ['fit_gate', 'score'],
                                {'np': np, 'roc_curve': c.roc_curve, 'roc_auc_score': c.roc_auc_score})
        values = np.array([0., 1., 2., 3.]); positive = np.array([False, False, True, True])
        gate = funcs['fit_gate'](values, positive, np.array(['a', 'a', 'b', 'b']))
        self.assertEqual(gate['threshold'], 2); self.assertEqual(gate['source_balanced_accuracy'], 1)
        self.assertEqual(funcs['score'](values, positive, gate['threshold'])['balanced_accuracy'], 1)

    def test_top_level_not_executed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'danger.py'
            path.write_text("raise RuntimeError('top level')\ndef safe(): return 42\n")
            self.assertEqual(c.pure_functions(path, ['safe'], {})['safe'](), 42)

    def test_missing_function(self):
        with self.assertRaises(ValueError): c.pure_functions(self.source / i.VALUE / 'evaluate_values.py', ['missing'], {})

    def test_duplicate_summary_rejected(self):
        df = pd.DataFrame({'key': ['a', 'a'], 'value': [1., 2.]})
        with self.assertRaises(ValueError): c.table_difference(df, df, ['key'], ['value'])

    def test_missing_summary_rejected(self):
        df = pd.DataFrame({'key': ['a', 'b'], 'value': [1., 2.]})
        with self.assertRaises(ValueError): c.table_difference(df, df.iloc[:1], ['key'], ['value'])

    def test_nonfinite_summary_rejected(self):
        df = pd.DataFrame({'key': ['a'], 'value': [np.nan]})
        with self.assertRaises(ValueError): c.table_difference(df, df, ['key'], ['value'])

    def test_changed_summary_rejected(self):
        df = pd.DataFrame({'key': ['a'], 'value': [1.]}); old = df.copy(); old['value'] = 2.
        with self.assertRaises(ValueError): c.table_difference(df, old, ['key'], ['value'])

    def test_method_sets_distinct(self):
        self.assertNotIn('Raw', c.FINAL_D); self.assertIn('Raw', c.FINAL_F)
        self.assertNotIn('Harmony-direct[fixed_arcsinh]', c.FINAL_D)
        self.assertIn('fastMNN reconstructed', c.FINAL_D)


if __name__ == '__main__': unittest.main()
