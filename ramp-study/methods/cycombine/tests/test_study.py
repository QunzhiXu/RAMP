import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import inventory as inv
import study


def synthetic_plan(source):
    spec = {'case': 'bone_marrow', 'input': 'ready.csv', 'marker_file': 'markers.txt',
            'config': 'config.json', 'parameters': inv.PARAMETERS}
    files = []
    for name, origin in (('ready.csv', 'source'), ('markers.txt', 'source'), ('config.json', 'source')):
        files.append({'path': 'inputs/' + name, 'source': name, 'origin': origin, **inv.record(source / name)})
    return spec, files


def prepare_racer(source, output, barrier, queue):
    source = Path(source)
    frozen_plan = synthetic_plan(source)
    barrier.wait(20)
    try:
        with patch.object(study, 'plan', return_value=frozen_plan):
            study.prepare(source, output)
        queue.put('won')
    except FileExistsError:
        queue.put('refused')


class Study(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='cycombine_study_')
        self.root = Path(self.tmp.name).resolve()
        self.addCleanup(self.tmp.cleanup)
        self.source = self.root / 'source'; self.source.mkdir()
        for name, content in (('ready.csv', 'cell_id,x\na,2\n'), ('markers.txt', 'x\n'), ('config.json', '{}')):
            (self.source / name).write_text(content)
        self.spec, self.files = synthetic_plan(self.source)
        self.bundle = self.root / 'bundle'

    def prepare(self):
        with patch.object(study, 'plan', return_value=(self.spec, self.files)):
            study.prepare(self.source, self.bundle)

    def verify(self):
        with patch.object(study, 'plan', return_value=(self.spec, self.files)):
            return study.verify_bundle(self.bundle)

    def assert_failed(self, output):
        self.assertTrue((output / 'FAILED.json').is_file())
        self.assertFalse((output / 'COMPLETE.json').exists())

    def test_import_does_not_write_or_start_process(self):
        with patch.object(Path, 'open', side_effect=AssertionError), patch.object(study.subprocess, 'run', side_effect=AssertionError):
            spec = importlib.util.spec_from_file_location('study_import_probe', HERE / 'study.py')
            spec.loader.exec_module(importlib.util.module_from_spec(spec))

    def test_real_plan_is_bm_and_method_ready_not_raw(self):
        spec, files = study.plan()
        self.assertEqual(spec['cells'], 88031)
        self.assertEqual(len(spec['markers']), 14)
        self.assertEqual(len(files), 25)
        self.assertTrue(spec['input'].endswith('combined_analysis_ready.csv.gz'))
        self.assertFalse(any(r['source'].endswith('raw.npy') for r in files))
        self.assertFalse(any('corrected_expression' in r['source'] for r in files))

    def test_unsupported_task_refused(self):
        with self.assertRaises(ValueError): study.plan('hnscc_story')

    def test_reference_paths_confined(self):
        self.assertEqual(study.relative_reference('../p/m.json', 'a/b'), 'a/p/m.json')
        for value in ('/tmp/x', '../../../x'):
            with self.assertRaises(ValueError): study.relative_reference(value, 'a/b')

    def test_prepare_and_verify(self):
        self.prepare(); self.verify()
        self.assertEqual((self.bundle / 'inputs/ready.csv').read_bytes(), (self.source / 'ready.csv').read_bytes())

    def test_moved_bundle_without_source(self):
        self.prepare()
        self.source.rename(self.root / 'hidden_source')
        moved = self.root / 'moved'; self.bundle.rename(moved); self.bundle = moved
        self.verify()

    def test_prepare_refuses_existing(self):
        self.prepare()
        with self.assertRaises(FileExistsError): self.prepare()

    def test_prepare_refuses_dangling_symlink(self):
        self.bundle.symlink_to(self.root / 'missing')
        with self.assertRaises(FileExistsError): self.prepare()

    def test_prepare_refuses_source_overlap(self):
        self.bundle = self.source / 'new'
        with self.assertRaises(ValueError): self.prepare()

    def test_prepare_changed_source_before_start(self):
        (self.source / 'ready.csv').write_text('changed')
        with self.assertRaises(ValueError): self.prepare()
        self.assertFalse(self.bundle.exists())

    def test_changed_source_during_copy_retains_failure(self):
        original = inv.copy_new
        def mutate(src, dst, expected):
            original(src, dst, expected)
            if src == self.source / 'ready.csv': src.write_text('changed')
        with patch.object(inv, 'copy_new', side_effect=mutate):
            with self.assertRaises(ValueError): self.prepare()
        self.assert_failed(self.bundle)

    def test_interrupted_copy_cannot_reuse(self):
        with patch.object(inv, 'copy_new', side_effect=RuntimeError('interrupted')):
            with self.assertRaises(RuntimeError): self.prepare()
        self.assert_failed(self.bundle)
        with self.assertRaises(FileExistsError): self.prepare()

    def test_complete_written_only_after_payload_validation(self):
        original = study.payload
        def fail(*args):
            self.assertFalse((self.bundle / 'COMPLETE.json').exists())
            original(*args)
            raise ValueError('final check failed')
        with patch.object(study, 'payload', side_effect=fail):
            with self.assertRaises(ValueError): self.prepare()
        self.assert_failed(self.bundle)

    def test_bundle_status_required(self):
        for status in ('partial', None):
            if self.bundle.exists(): self.bundle = self.root / 'other'
            self.prepare()
            path = self.bundle / 'COMPLETE.json'; data = inv.read_json(path)
            if status is None: del data['status']
            else: data['status'] = status
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError): self.verify()

    def test_bundle_hash_tampering(self):
        self.prepare(); (self.bundle / 'inputs/ready.csv').write_text('changed')
        with self.assertRaises(ValueError): self.verify()

    def test_bundle_cannot_self_report_new_hash(self):
        self.prepare()
        data = self.bundle / 'inputs/ready.csv'; data.write_text('changed')
        path = self.bundle / 'MANIFEST.json'; manifest = inv.read_json(path)
        manifest['files'][0].update(inv.record(data)); path.write_text(json.dumps(manifest))
        (self.bundle / 'COMPLETE.json').write_text(json.dumps({'status': 'complete', 'manifest_sha256': inv.sha(path)}))
        with self.assertRaises(ValueError): self.verify()

    def test_bundle_extra_file_rejected(self):
        self.prepare(); (self.bundle / 'extra').write_text('x')
        with self.assertRaises(ValueError): self.verify()

    def test_bundle_symlink_rejected(self):
        self.prepare(); (self.bundle / 'alias').symlink_to(self.source / 'ready.csv')
        with self.assertRaises(ValueError): self.verify()

    def test_true_process_prepare_race(self):
        ctx = mp.get_context('spawn')
        barrier = ctx.Barrier(2); queue = ctx.Queue()
        workers = [ctx.Process(target=prepare_racer, args=(str(self.source), str(self.bundle), barrier, queue)) for _ in range(2)]
        for p in workers: p.start()
        outcomes = [queue.get(timeout=30) for _ in workers]
        for p in workers: p.join(30); self.assertEqual(p.exitcode, 0)
        self.assertEqual(sorted(outcomes), ['refused', 'won'])
        self.verify()

    def fit_fixture(self):
        self.prepare()
        self.library = self.root / 'library'; self.library.mkdir()
        self.rscript = self.root / 'Rscript'; self.rscript.write_text('fake executable; never invoked')
        self.run = self.root / 'run'

    def fake_command(self, argv, output, name, env):
        (output / (name + '.stdout.log')).write_text('same environment\n')
        (output / (name + '.stderr.log')).write_text('')
        if name == 'correction':
            result = output / 'result'; result.mkdir()
            inv.create_json(result / 'COMPLETE.json', {'status': 'complete'})

    def fit(self, command=None, check=None):
        # Fake process/check only exercises the control layer, not numerical parity.
        (self.run.parent / 'unused').mkdir(exist_ok=True)
        def verify(*a): return self.spec, self.files
        original_copy = inv.copy_new
        # Empty fixture adapter; production uses 14 fixed byte-identical files.
        original_reserve = inv.reserve
        def reserve(*a):
            output = original_reserve(*a)
            (output / 'runtime/adapter').mkdir(parents=True)
            return output
        with patch.object(study, 'verify_bundle', side_effect=verify), patch.object(inv, 'reserve', side_effect=reserve), \
             patch.object(study, 'run_command', side_effect=command or self.fake_command), \
             patch.object(study, 'check_output', side_effect=check or (lambda *a: {})):
            return study.fit(self.bundle, self.library, self.rscript, self.run, self.root)

    def test_fit_controller_success(self):
        self.fit_fixture(); self.fit()
        self.assertEqual(inv.read_json(self.run / 'COMPLETE.json')['status'], 'complete')
        self.assertFalse(inv.read_json(self.run / 'RUN_MANIFEST.json')['history_used_by_fit'])

    def test_second_fit_cannot_overwrite(self):
        self.fit_fixture(); self.fit()
        before = inv.record(self.run / 'COMPLETE.json')
        with self.assertRaises(FileExistsError): self.fit()
        self.assertEqual(inv.record(self.run / 'COMPLETE.json'), before)

    def test_subprocess_failure_keeps_logs_and_no_complete(self):
        self.fit_fixture()
        def fail(*a): self.fake_command(*a); raise RuntimeError('R failed')
        with self.assertRaises(RuntimeError): self.fit(command=fail)
        self.assert_failed(self.run)
        self.assertTrue((self.run / 'environment_before.stdout.log').is_file())

    def test_nested_r_complete_not_enough(self):
        self.fit_fixture()
        with self.assertRaises(ValueError): self.fit(check=lambda *a: (_ for _ in ()).throw(ValueError('bad output')))
        self.assert_failed(self.run)
        self.assertTrue((self.run / 'result/COMPLETE.json').is_file())

    def test_executable_change_rejected_after_fit(self):
        self.fit_fixture()
        def change(*a):
            self.fake_command(*a)
            if a[2] == 'correction': self.rscript.write_text('changed')
        with self.assertRaises(ValueError): self.fit(command=change)
        self.assert_failed(self.run)

    def test_environment_change_rejected(self):
        self.fit_fixture()
        def change(*a):
            self.fake_command(*a)
            if a[2] == 'environment_after': (self.run / 'environment_after.stdout.log').write_text('changed')
        with self.assertRaises(ValueError): self.fit(command=change)
        self.assert_failed(self.run)

    def test_space_in_r_scratch_rejected(self):
        self.fit_fixture(); bad = self.root / 'bad scratch'; bad.mkdir()
        with self.assertRaises(ValueError): study.fit(self.bundle, self.library, self.rscript, self.run, bad)
        self.assertFalse(self.run.exists())

    def test_bundle_changed_during_fit(self):
        self.fit_fixture()
        def change(*a):
            self.fake_command(*a)
            if a[2] == 'correction': (self.bundle / 'MANIFEST.json').write_text('{}')
        with self.assertRaises(ValueError): self.fit(command=change)
        self.assert_failed(self.run)


if __name__ == '__main__':
    unittest.main()
