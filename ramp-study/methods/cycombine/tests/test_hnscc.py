"""HNSCC contract, identity, conversion and execution-control tests; R is mocked."""
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import inventory as inv
import hnscc
import study


def make_fixture(root):
    source = root / 'source'; base = source / hnscc.BASE; (base / 'input').mkdir(parents=True)
    (base / 'scripts').mkdir()
    spec = {'case': hnscc.CASE, 'input': str(hnscc.BASE / 'input/analysis_ready.csv.gz'),
        'metadata': str(hnscc.BASE / 'input/fit_metadata.csv.gz'), 'config': str(hnscc.BASE / 'run_config.json'),
        'marker_file': str(hnscc.BASE / 'input/markers.txt'), 'cells': 4, 'markers': ['CD3', 'CD8'],
        'datasets': ['ADT', 'CODEX'], 'parameters': inv.PARAMETERS,
        'preprocessing': [{'dataset': 'ADT', 'cells': 2, 'units': 1}, {'dataset': 'CODEX', 'cells': 2, 'units': 1}]}
    import pandas as pd
    frame = pd.DataFrame({'cell_id': ['a','b','c','d'], 'dataset': ['ADT','ADT','CODEX','CODEX'],
                          'unit_key': ['a1','a1','c1','c1'], 'CD3': [1.,2.,-1.,4.], 'CD8': [5.,6.,7.,8.]})
    frame.to_csv(base / 'input/analysis_ready.csv.gz', index=False)
    frame.iloc[:, :3].to_csv(base / 'input/fit_metadata.csv.gz', index=False)
    (base / 'input/markers.txt').write_text('CD3\nCD8\n')
    np.save(base / 'input/cell_ids.npy', frame.cell_id.to_numpy(dtype=str))
    inv.create_json(base / 'input/manifest.json', {'fixture': True})
    inv.create_json(base / 'run_config.json', {'input_manifest_sha256': inv.sha(base / 'input/manifest.json'),
                                             'datasets': spec['datasets'], 'markers': spec['markers']})
    (base / 'scripts/cycombine_repro.R').write_text('fixture; never sourced')
    files = []
    for path in sorted(base.rglob('*')):
        if path.is_file():
            relative = path.relative_to(source).as_posix()
            dest = 'scripts/cycombine_repro.R' if path.name == 'cycombine_repro.R' else 'inputs/' + relative
            files.append({'path': dest, 'source': relative, 'origin': 'source', **inv.record(path)})
    return source, spec, files


def race_prepare(source, spec, files, output, barrier, queue):
    barrier.wait(20)
    try:
        with patch.object(hnscc, 'plan', return_value=(spec, files)): hnscc.prepare(source, output)
        queue.put('won')
    except FileExistsError: queue.put('refused')


class HNSCC(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='hnscc_cycombine_')
        self.root = Path(self.temp.name).resolve(); self.addCleanup(self.temp.cleanup)
        self.source, self.spec, self.files = make_fixture(self.root)
        self.bundle = self.root / 'bundle'; self.run = self.root / 'run'
        self.library = self.root / 'library'; self.library.mkdir()
        self.rscript = self.root / 'Rscript'; self.rscript.write_text('fixture executable; never called')

    def prepare(self):
        with patch.object(hnscc, 'plan', return_value=(self.spec, self.files)): hnscc.prepare(self.source, self.bundle)

    def verify(self):
        with patch.object(hnscc, 'plan', return_value=(self.spec, self.files)): return hnscc.verify_bundle(self.bundle)

    def fake_command(self, argv, output, name, env):
        (output / (name + '.stdout.log')).write_text('environment stable\n')
        (output / (name + '.stderr.log')).write_text('')
        if name == 'correction':
            import pandas as pd
            result = output / hnscc.RESULT; result.mkdir(parents=True)
            inp = pd.read_csv(self.bundle / 'inputs' / self.spec['input'])
            inp[['cell_id', *self.spec['markers']]].to_csv(result / 'corrected_expression.csv.gz', index=False)
            (result / 'fit.rds').write_bytes(b'fake fit; control-layer test only')
            inv.create_json(result / 'R_COMPLETE.json', {'method': 'cycombine', 'parameters': self.spec['parameters'],
                'result': {'cells': 4, 'dimensions': 2, 'row_identity_verified': True, 'output_type': 'marker_values'},
                'input_sha256': inv.sha(self.bundle / 'inputs' / self.spec['input']),
                'helper_sha256': inv.sha(self.bundle / 'scripts/cycombine_repro.R'),
                'output_sha256': inv.sha(result / 'corrected_expression.csv.gz'), 'elapsed_seconds': 0.1})

    def fit(self, command=None):
        reserve = inv.reserve
        def fixture_reserve(*a):
            out = reserve(*a); (out / 'runtime/tree/algorithm/cycombine').mkdir(parents=True); return out
        with patch.object(hnscc, 'plan', return_value=(self.spec, self.files)), \
             patch.object(inv, 'reserve', side_effect=fixture_reserve), \
             patch.object(study, 'run_command', side_effect=command or self.fake_command):
            return hnscc.fit(self.bundle, self.library, self.rscript, self.run, self.root)

    def assert_failed(self, path):
        self.assertTrue((path / 'FAILED.json').exists()); self.assertFalse((path / 'COMPLETE.json').exists())

    def test_import_is_side_effect_free(self):
        with patch.object(Path, 'open', side_effect=AssertionError), patch.object(study.subprocess, 'run', side_effect=AssertionError):
            spec = importlib.util.spec_from_file_location('hnscc_import_test', HERE / 'hnscc.py')
            spec.loader.exec_module(importlib.util.module_from_spec(spec))

    def test_real_plan(self):
        spec, files = hnscc.plan()
        self.assertEqual((spec['cells'], len(spec['markers'])), (94488, 22))
        self.assertEqual(len(files), 24)
        self.assertEqual(sum(r['origin'] == 'source' for r in files), 6)
        self.assertEqual(sum(r['path'].startswith('adapter/') for r in files), 14)
        self.assertFalse(any(r['source'].endswith(('values.npy','raw.npy')) or '/methods/cycombine/' in r['source'] for r in files))
        self.assertEqual([p['operator'] for p in spec['preprocessing']], ['log1p', 'identity'])

    def test_other_cases_refused(self):
        for name in ('bone_marrow','lung_story','melanoma_story'):
            with self.subTest(case=name), self.assertRaises(ValueError): hnscc.plan(name)

    def test_technical_entry_still_refuses_hnscc(self):
        with self.assertRaises(ValueError): study.plan('hnscc_story')

    def test_prepare_verifies_negative_values_unchanged(self):
        self.prepare(); self.verify()
        data = inv.table(self.bundle / 'inputs' / self.spec['input'])
        self.assertEqual(float(data.CD3.iloc[2]), -1.)

    def test_move_without_source(self):
        self.prepare(); self.source.rename(self.root / 'hidden')
        moved = self.root / 'moved'; self.bundle.rename(moved); self.bundle = moved
        self.verify()

    def test_overlap_refused(self):
        self.bundle = self.source / 'new'
        with self.assertRaises(ValueError): self.prepare()

    def test_existing_and_dangling_refused(self):
        self.bundle.symlink_to(self.root / 'missing')
        with self.assertRaises(FileExistsError): self.prepare()

    def test_no_repeat_prepare(self):
        self.prepare()
        with self.assertRaises(FileExistsError): self.prepare()

    def test_changed_source_after_copy(self):
        original = inv.copy_new
        def change(src, dst, expected):
            original(src, dst, expected)
            if src.name == 'markers.txt': src.write_text('altered')
        with patch.object(inv, 'copy_new', side_effect=change):
            with self.assertRaises(ValueError): self.prepare()
        self.assert_failed(self.bundle)

    def test_false_or_missing_status(self):
        self.prepare(); p = self.bundle / 'COMPLETE.json'; data = inv.read_json(p)
        for state in ('partial', None):
            if state: data['status'] = state
            else: data.pop('status')
            p.write_text(json.dumps(data))
            with self.assertRaises(ValueError): self.verify()

    def test_extra_file(self):
        self.prepare(); (self.bundle / 'extra').write_text('x')
        with self.assertRaises(ValueError): self.verify()

    def test_false_self_report_hash(self):
        self.prepare(); p = self.bundle / 'inputs' / self.spec['marker_file']; p.write_text('wrong')
        manifestpath = self.bundle / 'MANIFEST.json'; d = inv.read_json(manifestpath)
        for r in d['files']:
            if r['path'] == str(p.relative_to(self.bundle)): r.update(inv.record(p))
        manifestpath.write_text(json.dumps(d))
        (self.bundle / 'COMPLETE.json').write_text(json.dumps({'status': 'complete', 'manifest_sha256': inv.sha(manifestpath)}))
        with self.assertRaises(ValueError): self.verify()

    def test_real_prepare_race(self):
        ctx = mp.get_context('spawn'); barrier = ctx.Barrier(2); queue = ctx.Queue()
        workers = [ctx.Process(target=race_prepare, args=(self.source,self.spec,self.files,self.bundle,barrier,queue)) for _ in range(2)]
        for worker in workers: worker.start()
        outcomes = [queue.get(timeout=30) for _ in workers]
        for worker in workers: worker.join(30); self.assertEqual(worker.exitcode, 0)
        self.assertEqual(sorted(outcomes), ['refused','won']); self.verify()

    def test_input_identity_check_independent_of_hash(self):
        self.prepare(); p = self.bundle / 'inputs' / hnscc.BASE / 'input/cell_ids.npy'
        np.save(p, np.array(['b','a','c','d']))
        with self.assertRaises(ValueError): hnscc.validate_inputs(self.bundle,self.spec)

    def test_input_marker_order_check(self):
        self.prepare(); (self.bundle / 'inputs' / self.spec['marker_file']).write_text('CD8\nCD3\n')
        with self.assertRaises(ValueError): hnscc.validate_inputs(self.bundle,self.spec)

    def test_success_and_exact_historical_conversion(self):
        import pandas as pd
        self.prepare(); self.fit()
        result = self.run / hnscc.RESULT
        expected = pd.read_csv(result/'corrected_expression.csv.gz',low_memory=False).iloc[:,1:].to_numpy(float)
        self.assertTrue(np.array_equal(expected,np.load(result/'values.npy')))
        with patch.object(hnscc,'plan',return_value=(self.spec,self.files)): hnscc.verify_run(self.run,self.bundle)

    def test_repeat_fit_refused(self):
        self.prepare(); self.fit(); before = inv.record(self.run / 'COMPLETE.json')
        with self.assertRaises(FileExistsError): self.fit()
        self.assertEqual(before, inv.record(self.run / 'COMPLETE.json'))

    def test_R_failure_keeps_partial(self):
        self.prepare()
        def fail(*a): self.fake_command(*a); raise RuntimeError('interrupted')
        with self.assertRaises(RuntimeError): self.fit(fail)
        self.assert_failed(self.run)

    def test_environment_change(self):
        self.prepare()
        def change(*a):
            self.fake_command(*a)
            if a[2]=='story_runtime_after': (self.run/'story_runtime_after.stdout.log').write_text('changed')
        with self.assertRaises(ValueError): self.fit(change)
        self.assert_failed(self.run)

    def test_bundle_change_during_fit(self):
        self.prepare()
        def change(*a):
            self.fake_command(*a)
            if a[2]=='correction': (self.bundle/'MANIFEST.json').write_text('{}')
        with self.assertRaises((ValueError,KeyError)): self.fit(change)
        self.assert_failed(self.run)
        self.assertTrue((self.run/hnscc.RESULT/'R_COMPLETE.json').exists())

    def test_runtime_copy_change(self):
        self.prepare()
        def change(*a):
            self.fake_command(*a)
            if a[2]=='correction': (self.run/'runtime/tree'/hnscc.BASE/'scripts/cycombine_repro.R').write_text('changed')
        with self.assertRaises(ValueError): self.fit(change)
        self.assert_failed(self.run)

    def test_wrong_R_parameters(self):
        self.prepare()
        def change(*a):
            self.fake_command(*a)
            if a[2]=='correction':
                path=self.run/hnscc.RESULT/'R_COMPLETE.json';d=inv.read_json(path);d['parameters']['seed']=1;path.write_text(json.dumps(d))
        with self.assertRaises(ValueError): self.fit(change)
        self.assert_failed(self.run)

    def test_values_refuse_overwrite(self):
        self.prepare(); self.fit(); result=self.run/hnscc.RESULT
        before=inv.record(result/'values.npy')
        with self.assertRaises(FileExistsError): hnscc.finish_values(result,self.bundle,self.spec)
        self.assertEqual(before,inv.record(result/'values.npy'))

    def test_missing_outer_complete_rejected(self):
        self.prepare(); self.fit(); (self.run/'COMPLETE.json').unlink()
        with patch.object(hnscc,'plan',return_value=(self.spec,self.files)), self.assertRaises(FileNotFoundError): hnscc.verify_run(self.run,self.bundle)


if __name__ == '__main__': unittest.main()
