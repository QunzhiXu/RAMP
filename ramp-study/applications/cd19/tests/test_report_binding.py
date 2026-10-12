import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import validate
from figure_metrics import SUPPLEMENT_SHA


class ReportBinding(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.lock = self.root / 'explicit lock'; self.lock.mkdir()
        self.run = self.root / 'run'; self.run.mkdir()
        self.comp = self.root / 'comparison'; self.comp.mkdir()
        self.fig = self.root / 'figure'; self.fig.mkdir()
        common.write(self.lock / 'LOCK.json', {'source_root': str(self.root / 'source')})
        common.write(self.run / 'RUN_MANIFEST.json', {'files': 'synthetic; no real training'})
        self.reports = [
            {'status': 'completed_with_differences', 'differing_files': ['qc.csv'],
             'run_manifest_sha256': common.sha(self.run / 'RUN_MANIFEST.json'),
             'lock_sha256': common.sha(self.lock / 'LOCK.json')},
            {'status': 'exact', 'base_run_manifest_sha256': common.sha(self.run / 'RUN_MANIFEST.json'),
             'supplementary_lock_sha256': SUPPLEMENT_SHA}]
        self.seal()

    def tearDown(self):
        self.temp.cleanup()

    def seal(self):
        # Only synthetic fixtures are overwritten/resealed for the regression.
        for directory, report in zip((self.comp, self.fig), self.reports):
            (directory / 'report.json').write_text(json.dumps(report))
            (directory / 'COMPLETE.json').write_text(json.dumps(
                {'status': 'complete', 'report_sha256': common.sha(directory / 'report.json')}))

    def check(self):
        return validate.report_bindings(self.lock, self.run, self.comp, self.fig)

    def reject(self, side, field, missing=False):
        if missing:
            del self.reports[side][field]
        else:
            self.reports[side][field] = '0' * 64
        self.seal()
        before = common.files(self.root)
        with self.assertRaisesRegex(ValueError, 'report binding differs: ' + field):
            self.check()
        self.assertEqual(common.files(self.root), before)

    def test_correct_bindings_keep_known_differences(self):
        a, b, bindings = self.check()
        self.assertEqual([a, b], self.reports)
        self.assertEqual(a['status'], 'completed_with_differences')
        self.assertEqual(bindings['comparison_report'], common.record(self.comp / 'report.json'))
        self.assertEqual(bindings['figure_metrics_report'], common.record(self.fig / 'report.json'))

    def test_comparison_wrong_run(self):
        self.reject(0, 'run_manifest_sha256')

    def test_comparison_missing_run(self):
        self.reject(0, 'run_manifest_sha256', True)

    def test_comparison_wrong_lock(self):
        self.reject(0, 'lock_sha256')

    def test_comparison_missing_lock(self):
        self.reject(0, 'lock_sha256', True)

    def test_figure_wrong_run(self):
        self.reject(1, 'base_run_manifest_sha256')

    def test_figure_missing_run(self):
        self.reject(1, 'base_run_manifest_sha256', True)

    def test_figure_wrong_supplement(self):
        self.reject(1, 'supplementary_lock_sha256')

    def test_figure_missing_supplement(self):
        self.reject(1, 'supplementary_lock_sha256', True)

    def test_two_reports_agree_with_each_other_but_not_run(self):
        self.reports[0]['run_manifest_sha256'] = '0' * 64
        self.reports[1]['base_run_manifest_sha256'] = '0' * 64
        self.seal()
        with self.assertRaisesRegex(ValueError, 'Comparison report binding differs'):
            self.check()

    def test_changed_actual_run_manifest(self):
        (self.run / 'RUN_MANIFEST.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'run_manifest_sha256'):
            self.check()

    def test_changed_actual_lock(self):
        (self.lock / 'LOCK.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'lock_sha256'):
            self.check()

    def test_moved_explicit_lock_with_no_default(self):
        moved = self.root / 'other place'; self.lock.rename(moved); self.lock = moved
        self.assertFalse((self.root / 'frozen_20261011').exists())
        self.check()

    def test_unsealed_report(self):
        (self.comp / 'report.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Unsealed comparison'):
            self.check()

    def test_incomplete_report(self):
        (self.fig / 'COMPLETE.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.check()

    def test_failed_report(self):
        common.write(self.fig / 'FAILED.json', {'error': 'fixture'})
        with self.assertRaisesRegex(ValueError, 'Failed comparison'):
            self.check()

    def test_inflight_report_change(self):
        real_record = validate.record
        def changed(path):
            rec = real_record(path)
            if Path(path) == self.comp / 'report.json':
                self.reports[0]['differing_files'] = ['changed.csv']
                self.seal()
            return rec
        with patch.object(validate, 'record', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'Unsealed comparison'):
                self.check()

    def test_validate_rejects_before_outputs_or_expensive_checks(self):
        self.reports[1]['base_run_manifest_sha256'] = '0' * 64
        self.seal()
        out = self.root / 'out'
        with patch.object(validate, 'verify_run', return_value={'status': 'verified'}), \
             patch.object(validate, 'cohorts') as cohorts, patch.object(validate, 'reserve') as reserve:
            with self.assertRaisesRegex(ValueError, 'Figure metrics report binding differs'):
                validate.validate(self.lock, self.run, self.comp, self.fig, out)
            reserve.assert_not_called(); cohorts.assert_not_called()
        self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
