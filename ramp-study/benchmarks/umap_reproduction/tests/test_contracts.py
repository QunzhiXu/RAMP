import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import study as s
from bootstrap import r
import worker as w


class Contracts(unittest.TestCase):
    def setUp(self):
        catalog = r.read(s.HERE / 'support/LOCK.json')
        self.assertEqual(set(catalog['files']), {'replay.py', 'replay_worker.py', 'contract.py', 'inventory.py', 'compare_replay.py'})
        counts = [('bone_marrow', 10), ('tnbc', 10), ('melanoma', 10), ('lung_cancer', 10),
                  ('lung_story', 9), ('melanoma_story', 7), ('hnscc_story', 7)]
        self.manifest = {'cases': [{'case': case, 'variants': [{'id': f'v{i:02d}', 'method': 'RAMP' if i == 1 else str(i)}
                                                              for i in range(n)]} for case, n in counts]}
        self.entries = [{'case': c['case'], 'variant': v['id']} for c in self.manifest['cases'] for v in c['variants']]

    def test_full_grid(self): self.assertEqual(len(s.grid(self.manifest, self.entries, 'all')), 63)

    def test_permuted_grid(self): self.assertEqual(len(s.grid(self.manifest, list(reversed(self.entries)), 'all')), 63)

    def test_duplicate_63(self):
        with self.assertRaises(ValueError): s.grid(self.manifest, [self.entries[0]] * 63, 'all')

    def test_single_duplicate(self):
        self.entries[-1] = self.entries[0]
        with self.assertRaises(ValueError): s.grid(self.manifest, self.entries, 'all')

    def test_missing(self):
        with self.assertRaises(ValueError): s.grid(self.manifest, self.entries[:-1], 'all')

    def test_extra(self):
        with self.assertRaises(ValueError): s.grid(self.manifest, self.entries + [{'case': 'other', 'variant': 'v00'}], 'all')

    def test_foreign_case(self):
        self.entries[0]['case'] = '../../wrong'
        with self.assertRaises(ValueError): s.grid(self.manifest, self.entries, 'all')

    def test_foreign_variant(self):
        self.entries[0]['variant'] = 'v99'
        with self.assertRaises(ValueError): s.grid(self.manifest, self.entries, 'all')

    def test_malformed(self):
        for entries in (None, {}, [None], [{}]):
            with self.subTest(entries=entries), self.assertRaises(ValueError): s.grid(self.manifest, entries, 'all')

    def test_pilot(self):
        self.assertEqual([(c['case'], v['method']) for c, v in s.branches(self.manifest, 'pilot')],
                         [('tnbc', 'RAMP'), ('hnscc_story', 'RAMP')])

    def test_partial_not_full(self):
        selected = [{'case': c['case'], 'variant': v['id']} for c, v in s.branches(self.manifest, 'pilot')]
        with self.assertRaises(ValueError): s.grid(self.manifest, selected, 'all')

    def test_unknown_scope(self):
        with self.assertRaises(ValueError): s.branches(self.manifest, 'anything')

    def test_empty_fit_directory_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); e = Mock(); e.fit_or_load_umap.return_value = 'coordinates'
            self.assertEqual(w.fresh_fit(e, 'values', 'method', root), 'coordinates')
            e.fit_or_load_umap.assert_called_once_with('values', 'method', root, allow_fit=True)

    def test_preexisting_cache_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'umap').mkdir(); e = Mock()
            with self.assertRaisesRegex(ValueError, 'empty'): w.fresh_fit(e, 'values', 'method', root)
            e.fit_or_load_umap.assert_not_called()

    def test_missing_fit_directory_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError): w.fresh_fit(Mock(), None, 'method', Path(temp) / 'missing')

    def test_output_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            with self.assertRaises(ValueError): r.reserve(path / 'input/new', [path / 'input'])
            self.assertFalse((path / 'input').exists())

    def test_create_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'done').mkdir()
            with self.assertRaises(FileExistsError): r.reserve(root / 'done')

    def test_unknown_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); r.write(root / 'MANIFEST.json', {})
            with self.assertRaisesRegex(ValueError, 'accepted'): s.inputs(root)

    def test_worker_refuses_unreserved_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); r.write(root / 'REQUEST.json', {'case': 'tnbc', 'variant': 'v01'})
            with self.assertRaisesRegex(ValueError, 'reserved'): w.run(root)
            self.assertEqual([p.name for p in root.iterdir()], ['REQUEST.json'])

    def test_runtime_requires_single_thread(self):
        import os
        with patch.dict(os.environ, {k: '2' for k in r.THREADS}), patch('importlib.metadata.version', side_effect=r.PACKAGES.get):
            with self.assertRaisesRegex(ValueError, 'thread'): w.w.runtime()

    def test_coordinate_comparison_exact(self):
        import numpy as np
        a = np.arange(10, dtype='float32').reshape(-1, 2)
        report = w.compare_coordinates(a, a.copy(), lambda v: 'digest')
        self.assertTrue(report['array_equal']); self.assertEqual(report['changed_elements'], 0)

    def test_coordinate_difference_recorded(self):
        import numpy as np
        a = np.zeros((4, 2), dtype='float32'); b = a.copy(); b[0, 0] = 1
        report = w.compare_coordinates(a, b, lambda v: 'digest')
        self.assertFalse(report['array_equal']); self.assertEqual(report['maximum_absolute_difference'], 1)

    def test_bad_coordinate_rejected(self):
        import numpy as np
        a = np.zeros((4, 2), dtype='float32')
        for b in (a.astype(float), a[:-1], np.full_like(a, np.nan)):
            with self.assertRaises(ValueError): w.compare_coordinates(b, a, lambda v: 'digest')

    def test_protocol_exact(self):
        self.assertEqual((s.UMAP_PROTOCOL['n_neighbors'], s.UMAP_PROTOCOL['min_dist'], s.UMAP_PROTOCOL['random_state']), (30, .15, 42))
        self.assertEqual(s.UMAP_PROTOCOL['preprocessing'], 'pooled dimension-wise StandardScaler; no additional PCA')


if __name__ == '__main__': unittest.main()
