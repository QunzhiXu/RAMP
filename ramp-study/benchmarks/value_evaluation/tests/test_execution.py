from pathlib import Path
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import execution as e
import inventory as i


class Execution(unittest.TestCase):
    def test_eight_stages(self):
        commands = e.commands(Path('/mirror'), Path('/python'))
        self.assertEqual(len(commands), 8)
        self.assertEqual([x[-1] for x in commands[2:6]], list(i.CASES))

    def test_no_historical_output_command(self):
        for argv in e.commands(Path('/mirror'), Path('/python')):
            self.assertTrue(argv[2].startswith('/mirror/Main Figure/'))
            self.assertNotIn('final_xu', ' '.join(argv))

    def test_threads_for_child(self):
        with patch.dict(os.environ, {'OMP_NUM_THREADS': '12', 'PYTHONPATH': '/wrong'}):
            env = e.environment(Path('/output'))
        self.assertTrue(all(env[k] == '1' for k in e.THREADS)); self.assertNotIn('PYTHONPATH', env)

    def test_wrong_runtime_rejected(self):
        with patch.dict(os.environ, {k: '2' for k in e.THREADS}), self.assertRaises(ValueError): e.runtime()

    def test_outside_output(self):
        with self.assertRaises(ValueError): e.local_output('/private/tmp/arbitrary')

    def test_root_output(self):
        with self.assertRaises(ValueError): e.local_output(e.HERE / 'runs')

    def test_binding_requires_accepted_run(self):
        name = 'benchmark_results/tnbc/ramp/alignment/data/ramp_aligned.npy'
        with self.assertRaisesRegex(ValueError, 'accepted'): e.binding(name, {'bytes': 1, 'sha256': 'abc'}, {})

    def test_rejected_failed_batch_binding(self):
        name = 'benchmark_results/tnbc/ramp/alignment/data/ramp_aligned.npy'; rec = {'bytes': 1, 'sha256': 'abc'}
        with self.assertRaises(ValueError): e.binding(name, rec,
            {'benchmarks/ramp_alignment/runs/six_fit_20261011/tnbc/result/ramp_aligned.npy': rec})

    def test_exact_rerun_binding(self):
        name = 'benchmark_results/tnbc/ramp/alignment/data/ramp_aligned.npy'; rec = {'bytes': 1, 'sha256': 'abc'}
        path = e.rerun_prefix('tnbc', 'ramp') + 'ramp_aligned.npy'
        self.assertEqual(e.binding(name, rec, {path: rec})['path'], path)

    def test_ambiguous_binding_rejected(self):
        name = 'benchmark_results/tnbc/ramp/alignment/data/ramp_aligned.npy'; rec = {'bytes': 1, 'sha256': 'abc'}
        prefix = e.rerun_prefix('tnbc', 'ramp')
        with self.assertRaises(ValueError): e.binding(name, rec, {prefix + 'a.npy': rec, prefix + 'b.npy': rec})

    def test_old_manifest_explicit_provenance(self):
        name = 'benchmark_results/tnbc/ramp/alignment/alignment_manifest.json'; rec = {'bytes': 1, 'sha256': 'abc'}
        self.assertEqual(e.binding(name, rec, {})['role'], 'frozen_method_input_or_provenance')

    def test_frozen_identity_binding(self):
        name = 'benchmark_results/tnbc/ramp/alignment/input/cell_metadata.csv.gz'; rec = {'bytes': 1, 'sha256': 'abc'}
        path = 'benchmarks/integration_inputs/runs/frozen_inputs_20261010/tnbc/evaluation_metadata.csv.gz'
        self.assertEqual(e.binding(name, rec, {path: rec})['path'], path)

    def test_frozen_identity_change_rejected(self):
        with self.assertRaises(ValueError): e.binding('benchmark_results/tnbc/ramp/alignment/input/raw.npy',
                                                     {'bytes': 1, 'sha256': 'abc'}, {})

    def test_harmony_profiles_distinct(self):
        self.assertNotEqual(e.rerun_prefix('tnbc', 'harmony', 'source_scale'), e.rerun_prefix('tnbc', 'harmony', 'adtnorm_fixed_arcsinh'))

    def test_relocation_only_input_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); files = {}; originals = {}
            for case in i.CASES:
                relative = f'benchmark_results/{case}/fastmnn/preprocessing/data/input.csv'
                source = root / relative; source.parent.mkdir(parents=True); source.write_text('1,2\n')
                files[relative] = i.record(source)
                name = f'benchmark_results/{case}/fastmnn/alignment/data/manifest.json'; target = root / name
                target.parent.mkdir(parents=True); originals[name] = {'input': {'path': '/old/final_xu/' + relative,
                    'sha256': i.record(source)['sha256']}, 'other': {'numerical_parameter': 42}}
                i.write(target, originals[name])
            changes = e.relocate(root, {'files': files}); self.assertEqual(len(changes), 4)
            for row in changes:
                data = i.read(root / row['path']); data['input']['path'] = row['old_value']
                self.assertEqual(data, originals[row['path']]); self.assertEqual(row['field'], ['input', 'path'])

    def test_unregistered_relocation_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); target = root / 'benchmark_results/bone_marrow/fastmnn/alignment/data/manifest.json'
            target.parent.mkdir(parents=True); i.write(target, {'input': {'path': '/old/final_xu/outside.csv', 'sha256': 'abc'}})
            before = target.read_bytes()
            with self.assertRaises(ValueError): e.relocate(root, {'files': {}})
            self.assertEqual(before, target.read_bytes())

    def test_guarded_scripts_unique(self): self.assertEqual(len(e.SCRIPTS), len(set(e.SCRIPTS)))

    def test_no_prior_gate_outputs_in_plan_names(self):
        names = list(e.SCRIPTS) + list(e.EXTRA)
        self.assertFalse(any('source_gates.csv' in p or 'gate_target_unit_scores' in p for p in names))


if __name__ == '__main__': unittest.main()
