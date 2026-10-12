"""Grid/control tests use synthetic payloads; the handoff also runs unmocked real CLI checks."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replay as r


class GridContract(unittest.TestCase):
    def setUp(self):
        self.manifest = {'cases': [{'case': 'example', 'variants': [{'id': 'v00'}, {'id': 'v01'}]}]}
        self.report = {'code_records': r.code_records(), 'branches': [
            {'case': 'example', 'variant': 'v00'}, {'case': 'example', 'variant': 'v01'}]}

    def test_complete_grid(self):
        self.assertEqual(set(r.branch_records(self.manifest, self.report)), {('example', 'v00'), ('example', 'v01')})

    def test_order_not_material(self):
        self.report['branches'].reverse()
        self.assertEqual(len(r.branch_records(self.manifest, self.report)), 2)

    def test_same_count_duplicate(self):
        self.report['branches'][1] = self.report['branches'][0]
        with self.assertRaisesRegex(ValueError, 'grid.*duplicate'): r.branch_records(self.manifest, self.report)

    def test_missing(self):
        self.report['branches'].pop()
        with self.assertRaisesRegex(ValueError, 'grid'): r.branch_records(self.manifest, self.report)

    def test_extra(self):
        self.report['branches'].append({'case': 'extra', 'variant': 'v00'})
        with self.assertRaisesRegex(ValueError, 'grid'): r.branch_records(self.manifest, self.report)

    def test_unknown_case(self):
        self.report['branches'][0]['case'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'grid'): r.branch_records(self.manifest, self.report)

    def test_unknown_variant(self):
        self.report['branches'][0]['variant'] = 'v02'
        with self.assertRaisesRegex(ValueError, 'grid'): r.branch_records(self.manifest, self.report)

    def test_unsafe_path(self):
        self.report['branches'][0]['case'] = '../../outside'
        with self.assertRaisesRegex(ValueError, 'grid'): r.branch_records(self.manifest, self.report)

    def test_malformed(self):
        for entries in (None, {}, [None], [{}], [{'case': [], 'variant': 'v00'}]):
            with self.subTest(entries=entries):
                self.report['branches'] = entries
                with self.assertRaisesRegex(ValueError, 'grid'): r.branch_records(self.manifest, self.report)

    def test_current_controller(self):
        r.check_controller_records(r.code_records())

    def test_pinned_controller(self):
        r.check_controller_records(copy.deepcopy(r.ACCEPTED_R3_CODE))

    def test_unknown_controller(self):
        changed = copy.deepcopy(r.ACCEPTED_R3_CODE)
        changed['replay.py']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'controller'): r.check_controller_records(changed)

    def test_missing_controller_component(self):
        changed = copy.deepcopy(r.ACCEPTED_R3_CODE); changed.pop('contract.py')
        with self.assertRaisesRegex(ValueError, 'controller'): r.check_controller_records(changed)

    def test_mixed_controllers(self):
        changed = copy.deepcopy(r.ACCEPTED_R3_CODE)
        changed['contract.py'] = {'sha256': '0' * 64, 'bytes': 123}
        with self.assertRaisesRegex(ValueError, 'controller'): r.check_controller_records(changed)


class BatchVerifier(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='grid-unit-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.bundle = self.root / 'bundle'; self.bundle.mkdir()
        self.batch = self.root / 'batch'; self.batch.mkdir()
        self.manifest = {'cases': [{'case': 'example', 'variants': [{'id': 'v00'}]}]}
        r.write(self.bundle / 'MANIFEST.json', self.manifest)
        self.codes = r.code_records()
        for name, expected in self.codes.items(): r.copy(r.HERE / name, self.batch / 'controller' / name, expected)
        self.parent = {'code_records': self.codes, 'bundle_manifest': r.record(self.bundle / 'MANIFEST.json'),
                       'bundle': str(self.bundle), 'jobs': 1}
        self.request = {k: v for k, v in self.parent.items() if k != 'jobs'}
        self.request.update(case='example', variant='v00')
        self.run = self.batch / 'example/v00'; self.run.mkdir(parents=True)
        self.result = {'method': 'Raw', 'resamples': 20, 'metric_values': 620}
        r.write(self.run / 'RUN_MANIFEST.json', {'files': {}})
        self.report = {'code_records': self.codes, 'branches': [{'case': 'example', 'variant': 'v00',
                       'manifest': r.record(self.run / 'RUN_MANIFEST.json'), 'result': self.result}]}

    def publish(self):
        r.write(self.batch / 'REQUEST.json', self.parent)
        r.write(self.run / 'REQUEST.json', self.request)
        r.write(self.batch / 'BATCH_MANIFEST.json', self.report)
        r.write(self.batch / 'COMPLETE.json', {'status': 'complete',
                                             'manifest': r.record(self.batch / 'BATCH_MANIFEST.json')})

    def verify(self):
        with patch.object(r, 'verify_bundle', return_value=self.manifest), \
             patch.object(r, 'verify_branch', return_value=self.result):
            return r.verify_run(self.batch, self.bundle)

    def test_valid_batch_no_writes(self):
        self.publish(); before = r.tree(self.root)
        self.assertEqual(self.verify(), {'branches': 1, 'status': 'passed'})
        self.assertEqual(before, r.tree(self.root))

    def test_index_request_swap(self):
        self.request['variant'] = 'v01'; self.publish()
        with self.assertRaisesRegex(ValueError, 'identity'): self.verify()

    def test_parent_worker_controller_mismatch(self):
        self.request['code_records'] = r.ACCEPTED_R3_CODE; self.publish()
        with self.assertRaisesRegex(ValueError, 'request'): self.verify()

    def test_parent_report_mismatch(self):
        self.parent['code_records'] = r.ACCEPTED_R3_CODE; self.publish()
        with self.assertRaisesRegex(ValueError, 'controller'): self.verify()

    def test_missing_worker_field(self):
        self.request.pop('bundle'); self.publish()
        with self.assertRaisesRegex(ValueError, 'request'): self.verify()

    def test_changed_bundle_binding(self):
        self.parent['bundle_manifest'] = {'sha256': '0' * 64, 'bytes': 1}; self.publish()
        with self.assertRaisesRegex(ValueError, 'Changed file'): self.verify()

    def test_changed_manifest_binding(self):
        self.report['branches'][0]['manifest']['sha256'] = '0' * 64; self.publish()
        with self.assertRaisesRegex(ValueError, 'Changed file'): self.verify()

    def test_changed_summary(self):
        self.report['branches'][0]['result'] = {**self.result, 'resamples': 19}; self.publish()
        with self.assertRaisesRegex(ValueError, 'summary'): self.verify()

    def test_changed_runtime_copy(self):
        self.publish()
        (self.batch / 'controller/replay.py').write_text('not the recorded controller')
        with self.assertRaisesRegex(ValueError, 'Changed file'): self.verify()

    def test_incomplete_diagnostic(self):
        with self.assertRaisesRegex(ValueError, 'Batch incomplete'): r.verify_run(self.batch, self.bundle)
        self.assertFalse((self.batch / 'COMPLETE.json').exists())

    def test_duplicate_rejected_before_branch_access(self):
        self.report['branches'] *= 63; self.publish()
        with patch.object(r, 'verify_bundle', return_value=self.manifest), patch.object(r, 'verify_branch') as branch:
            with self.assertRaisesRegex(ValueError, 'grid'): r.verify_run(self.batch, self.bundle)
            branch.assert_not_called()

    def cli(self):
        # Stub only the large input-payload verifier. This invokes the actual
        # argument parser and grid checks in a new Python process, not a mock CLI.
        script = ('import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv.pop(1));'
                  'import replay;r=replay;'
                  'r.verify_bundle=lambda p: json.loads((Path(p)/"MANIFEST.json").read_text());r.main()')
        return subprocess.run([sys.executable, '-B', '-c', script, str(r.HERE), 'verify-run',
                               '--output', str(self.batch), '--bundle', str(self.bundle)],
                              text=True, capture_output=True)

    def test_cli_63_duplicates(self):
        self.report['branches'] *= 63; self.publish(); before = r.tree(self.root)
        result = self.cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Batch branch grid differs: duplicate', result.stderr)
        self.assertEqual(before, r.tree(self.root))

    def test_cli_missing_branch(self):
        self.report['branches'] = []; self.publish()
        result = self.cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Batch branch grid differs', result.stderr)

    def test_cli_foreign_variant(self):
        self.report['branches'][0]['variant'] = 'v99'; self.publish()
        result = self.cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Batch branch grid differs', result.stderr)


if __name__ == '__main__': unittest.main()
