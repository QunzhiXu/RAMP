import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import specs
import study


class Controls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def test_grid(self):
        self.assertEqual(len(specs.MARKERS), 14)
        self.assertEqual(len(specs.STAGES), 27)
        self.assertEqual(len(set(specs.STAGES)), len(specs.STAGES))
        for stage in specs.STAGES:
            rel, _, _ = specs.action(stage)
            self.assertIn(rel, specs.CODES)

    def test_unknown_marker(self):
        for marker in ('CD19', '../CD3', '', 'CD03'):
            with self.subTest(marker=marker), self.assertRaises(ValueError):
                specs.marker_dir(marker)

    def test_reference_not_input(self):
        self.assertEqual(len(specs.INPUTS), 7)
        self.assertFalse(any('prediction' in Path(p).name or 'output/' in p for p in specs.INPUTS))

    def test_no_output_overlap(self):
        protected = self.root / 'protected'; protected.mkdir()
        for path in (protected, protected / 'sub', self.root):
            with self.subTest(path=path), self.assertRaises(ValueError):
                common.reserve(path, [protected])
        self.assertEqual(list(protected.iterdir()), [])

    def test_existing_directory(self):
        out = self.root / 'out'; out.mkdir()
        with self.assertRaises(FileExistsError):
            common.reserve(out, [])

    def test_dangling_link(self):
        out = self.root / 'out'; out.symlink_to(self.root / 'absent')
        with self.assertRaises(ValueError):
            common.reserve(out, [])

    def test_parent_alias(self):
        real = self.root / 'real'; real.mkdir()
        alias = self.root / 'alias'; alias.symlink_to(real)
        with self.assertRaises(ValueError):
            common.reserve(alias / 'out', [])

    def test_hardlink(self):
        a = self.root / 'a'; a.write_bytes(b'a')
        os.link(a, self.root / 'b')
        with self.assertRaises(ValueError):
            common.record(a)

    def test_exclusive_record(self):
        p = self.root / 'record.json'; common.write(p, {'a': 1})
        with self.assertRaises(FileExistsError):
            common.write(p, {'a': 2})
        self.assertEqual(common.read(p), {'a': 1})

    def test_resource_limits(self):
        good = {'study_allocated_bytes': 12_000_000_000, 'filesystem_free_bytes': 20_000_000_000}
        self.assertEqual(common.check_resources(good), good)
        for changed in ({'study_allocated_bytes': 60_000_000_000}, {'filesystem_free_bytes': 19_999_999_999}):
            with self.assertRaises(RuntimeError):
                common.check_resources(good | changed)

    def test_clean_environment(self):
        with patch.dict(os.environ, {'PYTHONPATH': '/bad', 'OMP_NUM_THREADS': '14'}):
            env = common.environment(self.root)
        self.assertNotIn('PYTHONPATH', env)
        for key in common.THREADS:
            self.assertEqual(env[key], '1')
        self.assertTrue(Path(env['NUMBA_CACHE_DIR']).is_relative_to(self.root))

    def test_mirror_is_small(self):
        lockdir = self.root / 'lock'; source = self.root / 'source'; run = self.root / 'run'
        for rel in specs.CODES:
            p = lockdir / 'code' / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('# code')
        for rel in specs.INPUTS:
            p = source / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'input')
        study.mirror(run, lockdir, {'source_root': str(source)})
        self.assertEqual(len(common.links(run)), len(specs.INPUTS))
        self.assertEqual(len(common.files(run)), len(specs.CODES))

    def test_guard_real_subprocess(self):
        run = self.root / 'run'; run.mkdir()
        source = self.root / 'source'; source.write_text('original')
        (run / 'input').symlink_to(source)
        code = """import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from common import write_guard
r=Path(sys.argv[2]); write_guard(r, [r/'code.py'])
for p in [r/'input',r/'code.py',r.parent/'other']:
    try: p.write_text('bad')
    except PermissionError: pass
    else: raise AssertionError(p)
(r/'allowed').write_text('ok')
"""
        result = subprocess.run([sys.executable, '-B', '-c', code, str(common.HERE), str(run)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(source.read_text(), 'original')

    def test_unknown_stage(self):
        with self.assertRaises(KeyError):
            specs.action('rogue')

    def test_worker_refuses_missing_request(self):
        run = self.root / 'run'; run.mkdir()
        result = subprocess.run([sys.executable, '-B', str(common.HERE / 'worker.py'), '--run', str(run), '--stage', 'main_prediction'], capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(run.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
