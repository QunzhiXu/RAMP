import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fixtures import fixture, io, inv, study
from validate import compare_baselines


class FilesystemSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='harmony test ')
        self.root = Path(self.tmp.name).resolve()

    def tearDown(self): self.tmp.cleanup()

    def test_existing_output_unchanged(self):
        out = self.root / 'out'; out.mkdir(); (out / 'old').write_text('kept')
        with self.assertRaises(FileExistsError): io.reserve(out)
        self.assertEqual((out / 'old').read_text(), 'kept')

    def test_broken_symlink_output_rejected(self):
        out = self.root / 'out'; out.symlink_to(self.root / 'missing')
        with self.assertRaises(FileExistsError): io.reserve(out)

    def test_symlink_parent_protected(self):
        protected = self.root / 'source'; protected.mkdir(); link = self.root / 'link'; link.symlink_to(protected, target_is_directory=True)
        with self.assertRaises(ValueError): io.reserve(link / 'out', [protected])
        self.assertFalse((protected / 'out').exists())

    def test_ancestor_protected_rejected(self):
        with self.assertRaises(ValueError): io.reserve(self.root / 'future', [self.root / 'future/input'])

    def test_hardlink_not_overwritten(self):
        src = self.root / 'input'; src.write_text('old'); dest = self.root / 'target'; os.link(src, dest)
        with self.assertRaises(FileExistsError): io.copy_new(src, dest, io.record(src))
        self.assertEqual(src.read_text(), 'old')

    def test_escaping_references_rejected(self):
        f = self.root / 'a'; f.write_text('x')
        for name in ('../a', str(f)):
            with self.subTest(name=name), self.assertRaises(ValueError): io.within(self.root, name)

    def test_real_two_process_reservation(self):
        # Both children reach a barrier before competing for the same mkdir.
        code = """import sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import harmony_io as io
root=Path(sys.argv[2]); name=sys.argv[3]
(root/('ready'+name)).write_text('ready')
while not (root/'go').exists(): time.sleep(.01)
try: out=io.reserve(root/'out')
except FileExistsError: sys.exit(17)
(out/'winner').write_text(name)
"""
        ps = [subprocess.Popen([sys.executable, '-B', '-c', code, str(inv.HERE), str(self.root), str(i)]) for i in range(2)]
        import time
        try:
            deadline = time.monotonic() + 15
            while not all((self.root / ('ready'+str(i))).exists() for i in range(2)):
                if time.monotonic() > deadline: self.fail('Child barrier timed out')
                time.sleep(.01)
            (self.root / 'go').write_text('go')
            self.assertEqual(sorted(p.wait(timeout=15) for p in ps), [0, 17])
            self.assertIn((self.root / 'out/winner').read_text(), ('0', '1'))
        finally:
            for p in ps:
                if p.poll() is None: p.kill(); p.wait()


class SnapshotContract(unittest.TestCase):
    def test_all_eleven_branches(self):
        self.assertEqual(len(inv.verify_local()['branches']), 11)
        for case in inv.TECHNICAL:
            for profile in inv.PROFILES:
                spec, files = study.plan(case, profile)
                self.assertEqual(spec['parameters'], {**inv.PARAMETERS, 'resolved_nclust': 100})
                self.assertFalse(any('evaluation_metadata' in r['path'] for r in files))
        for case in inv.STORIES:
            spec, files = study.plan(case)
            self.assertTrue(spec['input'].endswith('/raw.csv.gz'))
            self.assertFalse(any('analysis_ready.csv.gz' in r['path'] for r in files))

    def test_unsupported_story_profile(self):
        with self.assertRaises(ValueError): study.plan('lung_story', 'adtnorm_fixed_arcsinh')

    def test_unknown_case(self):
        with self.assertRaises(ValueError): study.plan('missing')

    def test_rehashed_catalog_cannot_replace_anchor(self):
        with tempfile.TemporaryDirectory() as name:
            dst = Path(name) / 'snap'; shutil.copytree(inv.SNAPSHOT, dst)
            p = dst / 'catalog.json'; c = io.read_json(p); c['branches'][0]['cells'] = 1; p.write_text(json.dumps(c))
            (dst / 'COMPLETE.json').write_text(json.dumps({'status': 'complete', 'manifest_sha256': io.sha(p)}))
            with self.assertRaisesRegex(ValueError, 'anchor'): inv.verify_local(dst)

    def test_registry_pin_not_self_authenticated(self):
        with tempfile.TemporaryDirectory() as name:
            dst = Path(name) / 'snap'; shutil.copytree(inv.SNAPSHOT, dst); (dst / 'registry.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'registry'): inv.verify_local(dst)

    def test_hash_link_checks_specific_field(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); (root / 'a').write_text('abc')
            io.create_json(root / 'm.json', {'wrong': '0'*64, 'unrelated': io.sha(root / 'a')})
            b = inv.Builder(root)
            with self.assertRaisesRegex(ValueError, 'hash differs'): b.link('m.json', ['wrong'], 'a')


class EntryContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='harmony test '); self.root = Path(self.tmp.name).resolve()
        self.context = fixture(self.root, 'technical', 3); self.source, self.spec = self.context.__enter__()
        self.bundle = self.root / 'bundle'

    def tearDown(self):
        self.context.__exit__(None, None, None); self.tmp.cleanup()

    def prepare(self): return study.prepare(self.source, self.bundle, self.spec['case'])

    def rewrite_manifest(self, edit):
        m = io.read_json(self.bundle / 'MANIFEST.json'); edit(m)
        (self.bundle / 'MANIFEST.json').write_text(json.dumps(m))
        (self.bundle / 'COMPLETE.json').write_text(json.dumps({'status': 'complete', 'manifest_sha256': io.sha(self.bundle / 'MANIFEST.json')}))

    def test_prepare_and_moved_bundle(self):
        self.prepare(); moved = self.root / 'moved'; self.bundle.rename(moved)
        self.assertEqual(study.verify_bundle(moved)[0]['cells'], 360)

    def test_explicit_symlink_temporary_parent(self):
        alias = self.root / 'alias'; alias.symlink_to(self.root, target_is_directory=True)
        study.prepare(self.source, alias / 'bundle', self.spec['case']); study.verify_bundle(self.bundle)

    def test_changed_input_rejected_before_output(self):
        (self.source / self.spec['input']).write_bytes(b'changed')
        with self.assertRaises(ValueError): self.prepare()
        self.assertFalse(self.bundle.exists())

    def test_copy_failure_leaves_no_complete(self):
        with patch.object(io, 'copy_new', side_effect=OSError('injected disk failure')):
            with self.assertRaises(OSError): self.prepare()
        self.assertTrue((self.bundle / 'FAILED.json').is_file()); self.assertFalse((self.bundle / 'COMPLETE.json').exists())

    def test_source_mutation_during_copy_fails(self):
        original = io.copy_new; changed = False
        def copy_then_change(src, dst, expected):
            nonlocal changed
            original(src, dst, expected)
            if src == (self.source / self.spec['input']).resolve() and not changed:
                src.write_bytes(b'changed'); changed = True
        with patch.object(io, 'copy_new', side_effect=copy_then_change):
            with self.assertRaises(ValueError): self.prepare()
        self.assertFalse((self.bundle / 'COMPLETE.json').exists())

    def test_existing_bundle_unchanged(self):
        self.prepare(); before = io.sha(self.bundle / 'MANIFEST.json')
        with self.assertRaises(FileExistsError): self.prepare()
        self.assertEqual(io.sha(self.bundle / 'MANIFEST.json'), before)

    def test_partial_completion_rejected(self):
        self.prepare(); p = self.bundle / 'COMPLETE.json'; d = io.read_json(p); d['status'] = 'partial'; p.write_text(json.dumps(d))
        with self.assertRaises(ValueError): study.verify_bundle(self.bundle)

    def test_missing_completion_status_rejected(self):
        self.prepare(); p = self.bundle / 'COMPLETE.json'; d = io.read_json(p); del d['status']; p.write_text(json.dumps(d))
        with self.assertRaises(ValueError): study.verify_bundle(self.bundle)

    def test_rehashed_partial_manifest_rejected(self):
        self.prepare(); self.rewrite_manifest(lambda m: m.update(status='partial'))
        with self.assertRaises(ValueError): study.verify_bundle(self.bundle)

    def test_rehashed_input_cannot_replace_contract(self):
        self.prepare(); name = 'inputs/' + self.spec['input']; (self.bundle / name).write_bytes(b'bad')
        self.rewrite_manifest(lambda m: [r.update(io.record(self.bundle / name)) for r in m['files'] if r['path'] == name])
        with self.assertRaisesRegex(ValueError, 'frozen contract'): study.verify_bundle(self.bundle)

    def test_extra_file_rejected(self):
        self.prepare(); (self.bundle / 'extra').write_text('x')
        with self.assertRaisesRegex(ValueError, 'inventory'): study.verify_bundle(self.bundle)

    def test_payload_symlink_rejected(self):
        self.prepare(); p = self.bundle / 'adapter/README.md'; p.unlink(); p.symlink_to(inv.SNAPSHOT / 'adapter/README.md')
        with self.assertRaisesRegex(ValueError, 'inventory'): study.verify_bundle(self.bundle)

    def test_relocation_changes_only_path(self):
        self.prepare(); orig = io.read_json(self.bundle / 'inputs' / self.spec['config'])
        derived = study.relocated_config(self.bundle, self.root / 'run', self.spec)
        derived['input_scale']['preprocessing_record_file'] = orig['input_scale']['preprocessing_record_file']
        self.assertEqual(orig, derived)

    def test_fit_failure_has_no_complete(self):
        self.prepare(); lib = self.root / 'lib'; lib.mkdir(); run = self.root / 'run'
        with patch.object(study, 'run_command', side_effect=RuntimeError('simulated R failure')):
            with self.assertRaises(RuntimeError): study.fit(self.bundle, lib, '/usr/bin/false', run, '/private/tmp')
        self.assertTrue((run / 'FAILED.json').exists()); self.assertFalse((run / 'COMPLETE.json').exists())

    def test_wrong_scratch_rejected_before_run(self):
        self.prepare(); lib = self.root / 'lib'; lib.mkdir(); run = self.root / 'run'
        with self.assertRaisesRegex(ValueError, 'space-free'): study.fit(self.bundle, lib, '/usr/bin/false', run, self.root)
        self.assertFalse(run.exists())

    def test_environment_change_rejects_completion(self):
        self.prepare(); lib = self.root / 'lib'; lib.mkdir(); run = self.root / 'run'
        def changing_environment(argv, out, name, env):
            (out / (name + '.stdout.log')).write_text(name)
            (out / (name + '.stderr.log')).write_text('')
        with patch.object(study, 'run_command', side_effect=changing_environment):
            with self.assertRaisesRegex(ValueError, 'Environment changed'):
                study.fit(self.bundle, lib, '/usr/bin/false', run, '/private/tmp')
        self.assertTrue((run / 'FAILED.json').exists()); self.assertFalse((run / 'COMPLETE.json').exists())

    def test_success_exit_without_outputs_rejected(self):
        self.prepare(); lib = self.root / 'lib'; lib.mkdir(); run = self.root / 'run'
        def empty_success(argv, out, name, env):
            (out / (name + '.stdout.log')).write_text('same')
            (out / (name + '.stderr.log')).write_text('')
        with patch.object(study, 'run_command', side_effect=empty_success):
            with self.assertRaises(FileNotFoundError): study.fit(self.bundle, lib, '/usr/bin/false', run, '/private/tmp')
        self.assertFalse((run / 'COMPLETE.json').exists())

    def test_output_inside_source_rejected(self):
        with self.assertRaises(ValueError): study.prepare(self.source, self.source / 'out', self.spec['case'])


class StoryContract(unittest.TestCase):
    def test_two_platform_bundle(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            with fixture(root, 'story', 2) as (source, spec):
                bundle = study.prepare(source, root / 'bundle', spec['case'])
                _, files = study.verify_bundle(bundle)
                self.assertEqual(study.validate_inputs(bundle, spec), {'cells': 240, 'markers': 5, 'units': 6})
                self.assertEqual(str(study.result_path(spec)), 'runtime/tree/' + spec['base'] + '/methods/harmony')
                self.assertTrue(any(r['path'] == 'scripts/run_r.R' for r in files))


class BaselineComparison(unittest.TestCase):
    def test_observation_time_may_change(self):
        a = {'BM_RAMP': {'checked_at': '2026-10-10T01:00:00+00:00', 'all_reference_checks_passed': True}, 'sha256': 'same'}
        b = copy.deepcopy(a); b['BM_RAMP']['checked_at'] = '2026-10-10T02:00:00+00:00'
        compare_baselines(a, b)
        self.assertNotEqual(a['BM_RAMP']['checked_at'], b['BM_RAMP']['checked_at'])

    def test_evidence_changes_still_rejected(self):
        a = {'BM_RAMP': {'checked_at': '2026-10-10T01:00:00+00:00', 'all_reference_checks_passed': True}, 'sha256': 'same'}
        for mutate in (lambda b: b.update(sha256='changed'), lambda b: b['BM_RAMP'].update(all_reference_checks_passed=False)):
            b = copy.deepcopy(a); mutate(b)
            with self.assertRaisesRegex(ValueError, 'verification changed'): compare_baselines(a, b)


class RLocaleContract(unittest.TestCase):
    def test_unset_parent_is_pinned_without_mutation(self):
        parent = {'PATH': '/example', 'TMPDIR': '/private/tmp', 'R_LIBS': '/untrusted'}
        before = parent.copy(); env = study.r_environment(parent)
        self.assertEqual(parent, before)
        self.assertEqual({k: env[k] for k in ('LC_ALL', 'LANG')}, {'LC_ALL': 'C.UTF-8', 'LANG': 'C.UTF-8'})
        self.assertEqual(env['PATH'], '/example'); self.assertEqual(env['TMPDIR'], '/private/tmp')
        self.assertNotIn('R_LIBS', env)

    def test_C_parent_cannot_override_collation(self):
        with patch.dict(os.environ, {'LC_ALL': 'C', 'LANG': 'C', 'LC_COLLATE': 'C'}, clear=True):
            env = study.r_environment()
            self.assertEqual(env['LC_ALL'], 'C.UTF-8'); self.assertEqual(env['LANG'], 'C.UTF-8')
            self.assertEqual(os.environ['LC_ALL'], 'C')

    def test_fit_records_and_passes_locale_before_R(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name).resolve()
            with fixture(root, 'technical', 3) as (source, spec):
                bundle = study.prepare(source, root / 'bundle', spec['case']); lib = root / 'lib'; lib.mkdir()
                run = root / 'run'; observed = []
                def probe(argv, out, stage, env):
                    observed.append((stage, env.copy(), io.read_json(out / 'REQUEST.json')))
                    raise RuntimeError('stop after environment capture')
                with patch.dict(os.environ, {'LC_ALL': 'C', 'LANG': 'C'}), patch.object(study, 'run_command', side_effect=probe):
                    with self.assertRaisesRegex(RuntimeError, 'environment capture'):
                        study.fit(bundle, lib, '/usr/bin/false', run, '/private/tmp')
                stage, env, request = observed[0]
                self.assertEqual(stage, 'environment_before')
                self.assertEqual({k: env[k] for k in ('LC_ALL', 'LANG')}, study.R_LOCALE_ENV)
                self.assertEqual(request['r_locale_environment'], study.R_LOCALE_ENV)
                self.assertFalse((run / 'COMPLETE.json').exists())


if __name__ == '__main__': unittest.main()
