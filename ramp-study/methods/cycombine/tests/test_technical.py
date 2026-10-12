"""Extension routing tests; no R, extraction or numerical reimplementation."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import inventory as inv
import study


class Technical(unittest.TestCase):
    def check_case(self, case, cells, markers):
        spec, files = study.plan(case)
        self.assertEqual(spec['case'], case)
        self.assertEqual(spec['cells'], cells)
        self.assertEqual(len(spec['markers']), markers)
        self.assertEqual(spec['parameters'], inv.PARAMETERS)
        self.assertEqual(len(files), 25)
        sources = [f for f in files if f['origin'] == 'source']
        self.assertEqual(len(sources), 10)
        for f in sources:
            self.assertTrue(f['source'].startswith(('benchmark_results/' + case + '/', 'benchmark_data/' + case + '/')))
            self.assertFalse('corrected_expression' in f['source'])
            self.assertFalse(f['source'].endswith('raw.npy'))
        self.assertEqual(sum(f['path'].startswith('adapter/') for f in files), 14)

    def test_tnbc_plan(self): self.check_case('tnbc', 62701, 11)
    def test_melanoma_plan(self): self.check_case('melanoma', 60000, 12)
    def test_lung_technical_plan(self): self.check_case('lung_cancer', 60000, 8)

    def test_lung_uses_monkman_not_zheng(self):
        spec, _ = study.plan('lung_cancer')
        self.assertEqual(spec['datasets'], ['Senosain CyTOF', 'Cords IMC', 'Monkman CODEX'])
        self.assertIn('c_counts_asinh', spec['preprocessing'][1]['assay_or_layer'])

    def test_story_routes_remain_disabled(self):
        for name in ('lung_story', 'melanoma_story', 'hnscc_story', 'unknown'):
            with self.subTest(case=name), self.assertRaises(ValueError): study.plan(name)

    def test_case_cli_dispatch(self):
        for name in ('tnbc', 'melanoma', 'lung_cancer'):
            with self.subTest(case=name), patch.object(study, 'prepare', return_value='prepared') as call, contextlib.redirect_stdout(io.StringIO()):
                study.main(['prepare', '--case', name, '--source-root', '/source', '--output', '/output'])
                call.assert_called_once_with(Path('/source'), Path('/output'), name)

    def test_default_still_bm(self):
        with patch.object(study, 'prepare', return_value='prepared') as call, contextlib.redirect_stdout(io.StringIO()):
            study.main(['prepare', '--source-root', '/source', '--output', '/output'])
            call.assert_called_once_with(Path('/source'), Path('/output'), 'bone_marrow')

    def test_cli_refuses_story_before_preparation(self):
        with patch.object(study, 'prepare') as call, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                study.main(['prepare', '--case', 'lung_story', '--source-root', '/source', '--output', '/output'])
            call.assert_not_called()

    def test_catalog_anchor_still_required(self):
        with patch.object(study, 'CATALOG_SHA', '0' * 64):
            with self.assertRaises(ValueError): study.plan('tnbc')

    def test_tnbc_uses_updated_meyer_cohort(self):
        spec, _ = study.plan('tnbc')
        meyer = spec['preprocessing'][1]
        self.assertEqual(meyer['dataset'], 'Meyer IMC')
        self.assertIn('22,701', meyer['processing_steps'][0])
        self.assertIn('20260960', meyer['processing_steps'][0])

    def test_all_plans_use_identical_adapter(self):
        base = [r for r in study.plan()[1] if r['origin'] == 'snapshot']
        for name in ('tnbc', 'melanoma', 'lung_cancer'):
            with self.subTest(case=name):
                self.assertEqual(base, [r for r in study.plan(name)[1] if r['origin'] == 'snapshot'])


if __name__ == '__main__': unittest.main()
