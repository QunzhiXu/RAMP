import ast
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contract as c
import inventory as inv
import numpy as np
import pandas as pd


def compete(path, barrier, queue):
    barrier.wait()
    try:
        inv.reserve(path, [])
        queue.put('won')
    except FileExistsError:
        queue.put('refused')


class OutputSafety(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def test_existing_directory_refused(self):
        with self.assertRaises(FileExistsError): inv.reserve(self.root, [])

    def test_existing_file_refused(self):
        p = self.root / 'file'; p.write_text('retain')
        with self.assertRaises(FileExistsError): inv.reserve(p, [])
        self.assertEqual(p.read_text(), 'retain')

    def test_dangling_symlink_refused(self):
        p = self.root / 'link'; p.symlink_to(self.root / 'absent')
        with self.assertRaises(FileExistsError): inv.reserve(p, [])

    def test_protected_child_refused_before_mkdir(self):
        p = self.root / 'protected'
        with self.assertRaises(ValueError): inv.reserve(p / 'nested' / 'out', [p])
        self.assertFalse(p.exists())

    def test_protected_ancestor_refused(self):
        with self.assertRaises(ValueError): inv.reserve(self.root / 'out', [self.root / 'out' / 'input'])

    def test_symlink_parent_to_protected_refused(self):
        real = self.root / 'protected'; real.mkdir()
        alias = self.root / 'alias'; alias.symlink_to(real)
        with self.assertRaises(ValueError): inv.reserve(alias / 'out', [real])

    def test_real_two_process_claim(self):
        ctx = mp.get_context('spawn'); barrier = ctx.Barrier(2); queue = ctx.Queue()
        jobs = [ctx.Process(target=compete, args=(self.root / 'out', barrier, queue)) for _ in range(2)]
        for job in jobs: job.start()
        for job in jobs:
            job.join(20); self.assertFalse(job.is_alive()); self.assertEqual(job.exitcode, 0)
        self.assertEqual(sorted([queue.get(timeout=5), queue.get(timeout=5)]), ['refused', 'won'])

    def test_child_rejects_escape(self):
        for name in ('../escape', '/tmp/escape', ''):
            with self.assertRaises(ValueError): inv.child(self.root, name)

    def test_child_rejects_symlink(self):
        (self.root / 'link').symlink_to(self.root)
        with self.assertRaises(ValueError): inv.child(self.root, 'link/child')

    def test_exclusive_json_write(self):
        p = self.root / 'record'; inv.write(p, {'first': True})
        with self.assertRaises(FileExistsError): inv.write(p, {'second': True})

    def test_file_tamper_rejected(self):
        p = self.root / 'input'; p.write_text('good'); old = c.record(p); p.write_text('bad')
        with self.assertRaises(ValueError): inv.check(p, old)

    def test_sources_outside_root_rejected(self):
        (self.root / 'source').mkdir()
        src = c.Sources(self.root / 'source')
        p = self.root / 'outside'; p.write_text('data')
        with self.assertRaises(ValueError): src.add(p)

    def test_historical_field_exact(self):
        p = self.root / 'input'; p.write_text('data'); digest = c.record(p)['sha256']
        meta = self.root / 'meta.json'; inv.write(meta, {'raw_sha': 'wrong', 'note': digest})
        with self.assertRaises(ValueError): c.Sources(self.root).bind(meta, ['raw_sha'], p)

    def test_source_changes_during_collection_rejected(self):
        p = self.root / 'input'; p.write_text('old'); src = c.Sources(self.root); src.add(p)
        p.write_text('new')
        with self.assertRaises(ValueError): src.add(p)

    def fixture(self):
        source = self.root / 'source'; source.mkdir()
        (source / 'data.txt').write_text('payload')
        registry = self.root / 'registry.json'; registry.write_text('{"test": true}')
        digest = c.record(registry)['sha256']
        doc = {'sources': {'data.txt': c.record(source / 'data.txt')}, 'copies': ['data.txt'],
               'registry_sha256': digest, 'historical_hash_links': [], 'cases': []}
        return source, registry, digest, doc

    def fake_freeze(self, source, registry, digest, doc, mutation=None):
        original = inv.verify_payload
        def verify(root, document, complete=False):
            self.assertFalse((root / 'COMPLETE.json').exists())
            if mutation: mutation(root)
            return original(root, document, complete)
        with patch.object(c, 'HERE', self.root), patch.object(c, 'REGISTRY', registry), \
             patch.object(c, 'REGISTRY_SHA', digest), patch.object(c, 'collect', return_value=doc), \
             patch.object(inv, 'verify_payload', side_effect=verify):
            return inv.freeze(source, self.root / 'snapshot')

    def test_complete_is_last(self):
        args = self.fixture(); self.fake_freeze(*args)
        self.assertEqual(c.read(self.root / 'snapshot/COMPLETE.json')['status'], 'complete')

    def test_bad_final_payload_leaves_failed_only(self):
        args = self.fixture()
        with self.assertRaises(ValueError):
            self.fake_freeze(*args, mutation=lambda root: (root / 'sources/data.txt').write_text('bad'))
        self.assertTrue((self.root / 'snapshot/FAILED.json').exists())
        self.assertFalse((self.root / 'snapshot/COMPLETE.json').exists())

    def test_registry_change_leaves_failed_only(self):
        source, registry, digest, doc = self.fixture()
        def changed(*args):
            registry.write_text('changed'); return doc
        with patch.object(c, 'HERE', self.root), patch.object(c, 'REGISTRY', registry), \
             patch.object(c, 'REGISTRY_SHA', digest), patch.object(c, 'collect', side_effect=changed):
            with self.assertRaises(ValueError): inv.freeze(source, self.root / 'snapshot')
        self.assertFalse((self.root / 'snapshot/COMPLETE.json').exists())

    def test_relocated_snapshot_without_source(self):
        import shutil
        source, registry, digest, doc = self.fixture(); self.fake_freeze(source, registry, digest, doc)
        dest = self.root / 'moved path'; (self.root / 'snapshot').rename(dest)
        shutil.rmtree(source)  # temporary fixture only
        with patch.object(c, 'REGISTRY_SHA', digest): self.assertEqual(inv.verify(dest)['copies'], 1)

    def test_extra_file_rejected(self):
        source, registry, digest, doc = self.fixture(); self.fake_freeze(source, registry, digest, doc)
        (self.root / 'snapshot/extra').write_text('extra')
        with patch.object(c, 'REGISTRY_SHA', digest):
            with self.assertRaises(ValueError): inv.verify(self.root / 'snapshot')


class Contracts(unittest.TestCase):
    def test_sha256_json_direct_call(self):
        path = c.HERE / 'frozen_20261011_r2/sources' / c.EVALUATOR
        helper = c.pure_helpers(path)
        payload = {'b': [1, 2], 'a': 'value'}
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(helper['sha256_json'](payload), expected)

    def ranks(self):
        names = ['m' + str(i) for i in range(31)]
        families = ['biology'] * 5 + ['mixing'] * 14 + ['transfer'] * 2 + ['visual'] * 2 + ['geometry'] * 8
        methods = ['RAMP', 'MaxFuse', 'cyCombine', 'Harmony-direct', 'fastMNN', 'CytoNorm']
        rows = [dict(benchmark='case' + str(case), method=method, metric=metric, family=families[i],
                     rank=float(rank + 1)) for case in range(7) for rank, method in enumerate(methods)
                for i, metric in enumerate(names)]
        all31 = pd.DataFrame(rows)
        main23 = all31[all31.metric.isin(names[:23])].copy()
        policy = dict(included_metrics=names[:23], primary_family_weights=dict(biology=.25, mixing=.25, transfer=.25, visual=.25))
        ref = pd.DataFrame({'method': methods, 'mean_rank': [1., 2., 3., 4., 5., 6.]})
        return all31, main23, policy, ref

    def test_rank_aggregation(self):
        result = c.check_rank_tables(*self.ranks())
        self.assertEqual(result['max_aggregate_difference'], 0.)
        self.assertEqual(result['main23_rows'], 966)

    def test_changed_weights_rejected(self):
        a, b, p, r = self.ranks(); p['primary_family_weights']['mixing'] = .5
        with self.assertRaisesRegex(ValueError, 'weights'): c.check_rank_tables(a, b, p, r)

    def test_changed_main23_rejected(self):
        a, b, p, r = self.ranks(); b.loc[b.index[0], 'rank'] = 2.
        with self.assertRaises(AssertionError): c.check_rank_tables(a, b, p, r)

    def test_changed_aggregate_rejected(self):
        a, b, p, r = self.ranks(); r.loc[0, 'mean_rank'] = 2.
        with self.assertRaisesRegex(ValueError, 'aggregate'): c.check_rank_tables(a, b, p, r)

    def test_missing_task_rejected(self):
        a, b, p, r = self.ranks(); a = a[a.benchmark.ne('case0')]
        with self.assertRaises(ValueError): c.check_rank_tables(a, b, p, r)

    def test_missing_method_rejected(self):
        a, b, p, r = self.ranks(); a = a[a.method.ne('MaxFuse')]
        with self.assertRaises(ValueError): c.check_rank_tables(a, b, p, r)

    def test_complete_grid(self):
        c.check_grid(pd.DataFrame({'method': ['A', 'B'], 'sampling_seed': [1, 1], 'm': [0., 1.]}), ['A', 'B'], [1], ['m'])

    def test_missing_grid_rejected(self):
        with self.assertRaises(ValueError):
            c.check_grid(pd.DataFrame({'method': ['A'], 'sampling_seed': [1], 'm': [0.]}), ['A', 'B'], [1], ['m'])

    def test_duplicate_grid_rejected(self):
        with self.assertRaises(ValueError):
            c.check_grid(pd.DataFrame({'method': ['A', 'A'], 'sampling_seed': [1, 1], 'm': [0., 1.]}), ['A'], [1], ['m'])

    def test_missing_metric_rejected(self):
        with self.assertRaises(ValueError):
            c.check_grid(pd.DataFrame({'method': ['A'], 'sampling_seed': [1]}), ['A'], [1], ['m'])

    def test_no_original_script_import(self):
        tree = ast.parse((c.HERE / 'contract.py').read_text())
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr in ('exec_module', 'spec_from_file_location', 'fit_transform')
                             for n in ast.walk(tree)))

    def test_evaluator_ast_only_helpers(self):
        # Missing required functions are rejected without executing a top-level trap.
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'script.py'; p.write_text("raise RuntimeError('must not run')\n")
            with self.assertRaisesRegex(ValueError, 'Missing pure'): c.pure_helpers(p)

    def test_raw_signed_values_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'x.npy'; x = np.array([[-1., 0.], [1., 2.]])
            np.save(p, x)
            actual = c.load_matrix(p, 'RAMP', pd.DataFrame({'cell_id': ['a', 'b']}), ['A', 'B'])
            np.testing.assert_array_equal(actual, x)

    def test_matrix_row_identity_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'x.csv'; p.write_text('cell_id,A,B\nb,1,2\na,3,4\n')
            with self.assertRaisesRegex(ValueError, 'row IDs'):
                c.load_matrix(p, 'RAMP', pd.DataFrame({'cell_id': ['a', 'b']}), ['A', 'B'])

    def test_marker_order_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'x.csv'; p.write_text('cell_id,B,A\na,1,2\nb,3,4\n')
            with self.assertRaisesRegex(ValueError, 'Marker order'):
                c.load_matrix(p, 'RAMP', pd.DataFrame({'cell_id': ['a', 'b']}), ['A', 'B'])


if __name__ == '__main__': unittest.main()
