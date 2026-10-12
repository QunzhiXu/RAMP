import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replay as r
import replay_worker as w


class ReplayContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog, cls.prior = r.anchored()
        cls.registry = r.read(r.SNAPSHOT / 'registry.json')

    def test_all_63_routes(self):
        cases = r.plan(self.catalog, self.prior, self.registry)
        bindings = [v['binding'] for c in cases for v in c['variants']]
        self.assertEqual(len(bindings), 63)
        self.assertEqual(sum(b['root'] == 'study' for b in bindings), 46)
        self.assertEqual(sum(b['root'] == 'original' for b in bindings), 17)

    def test_no_abandoned_batch(self):
        cases = r.plan(self.catalog, self.prior, self.registry)
        for c in cases:
            for v in c['variants']:
                path = v['binding']['path']
                self.assertNotIn('/six_fit_20261011/', path)
                self.assertNotIn('/seven_fit_20261010/', path)

    def test_full_panel_story_maxfuse(self):
        for c in r.plan(self.catalog, self.prior, self.registry):
            if c['case'].endswith('_story'):
                for v in c['variants']:
                    if v['method'].startswith('MaxFuse'):
                        self.assertIn('/maxfuse_full_panel/', v['binding']['path'])

    def test_missing_rerun_refused(self):
        case = self.catalog['cases'][0]
        variant = next(v for v in case['variants'] if v['method'] == 'RAMP')
        with self.assertRaises(ValueError): r.source_binding(case, variant, self.catalog, {'outside_files': {}})

    def test_duplicate_rerun_refused(self):
        case = self.catalog['cases'][0]; variant = next(v for v in case['variants'] if v['method'] == 'RAMP')
        prior = copy.deepcopy(self.prior)
        b = r.source_binding(case, variant, self.catalog, prior)
        prior['outside_files'][str(Path(b['path']).parent / 'extra.npy')] = {k: b[k] for k in ('bytes', 'sha256')}
        with self.assertRaises(ValueError): r.source_binding(case, variant, self.catalog, prior)

    def test_case_substitution_refused(self):
        case = copy.deepcopy(self.catalog['cases'][0]); case['case'] = 'lung_story'
        v = next(v for v in case['variants'] if v['method'] == 'RAMP')
        with self.assertRaises(ValueError): r.source_binding(case, v, self.catalog, self.prior)

    def test_input_expected_hash_not_self_report(self):
        c = copy.deepcopy(self.catalog['cases'][0]); v = next(v for v in c['variants'] if v['method'] == 'RAMP')
        catalog = copy.deepcopy(self.catalog); catalog['sources'][v['matrix']]['sha256'] = '0' * 64
        with self.assertRaises(ValueError): r.source_binding(c, v, catalog, self.prior)

    def test_thread_env(self):
        with patch.dict(os.environ, {'PYTHONPATH': '/untrusted', 'OMP_NUM_THREADS': '14'}):
            env = r.environment(Path('/tmp/example'))
        self.assertTrue(all(env[k] == '1' for k in r.THREADS)); self.assertNotIn('PYTHONPATH', env)

    def test_runtime_wrong_threads_refused(self):
        with patch.dict(os.environ, {k: '2' for k in r.THREADS}), patch('importlib.metadata.version', side_effect=lambda k: r.PACKAGES[k]):
            with self.assertRaisesRegex(ValueError, 'thread'): w.runtime()

    def test_runtime_wrong_version_refused(self):
        with patch('importlib.metadata.version', return_value='wrong'):
            with self.assertRaisesRegex(ValueError, 'dependencies'): w.runtime()

    def test_output_outside_module_refused(self):
        with self.assertRaises(ValueError): r.local_output(r.STUDY / 'methods/maxfuse/new-output', 'runs')

    def test_reserved_area_itself_refused(self):
        with self.assertRaises(ValueError): r.local_output(r.HERE / 'runs', 'runs')

    def test_preflight_does_not_change_numerical_evaluator(self):
        self.assertEqual(self.catalog['sources']['benchmark_results/comparison_evaluation/scripts/evaluate_benchmark.py']['sha256'],
                         'b99fac68a9e3698d5dad369d4d74ad3aca6d6cfafc589e4055e47c4d2722bca6')

    def test_worker_rejects_unreserved_path_before_write(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp); r.write(p / 'REQUEST.json', {'case': 'bone_marrow', 'variant': 'v00'})
            with self.assertRaises(ValueError): w.run(p)
            self.assertEqual([x.name for x in p.iterdir()], ['REQUEST.json'])

    def test_create_only_and_failed_output_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'run'; r.reserve(p); r.write(p / 'FAILED.json', {'error': 'retained'})
            with self.assertRaises(FileExistsError): r.reserve(p)
            self.assertTrue((p / 'FAILED.json').exists())


if __name__ == '__main__': unittest.main()
