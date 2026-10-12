from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate import check_empty_evaluation_cache


class EvaluationCache(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / 'evaluation'
        self.root.mkdir()
        for name in ('matplotlib', 'numba', 'xdg'):
            (self.root / name).mkdir()

    def test_frozen_evaluator_empty_directories_accepted(self):
        self.assertEqual(check_empty_evaluation_cache(self.root),
                         {'empty_directories': ['matplotlib', 'numba', 'xdg'], 'payload_files': 0})

    def test_legacy_rglob_guard_reproduced(self):
        self.assertTrue(any(self.root.rglob('*')))
        self.assertEqual(check_empty_evaluation_cache(self.root)['payload_files'], 0)

    def test_payload_file_refused(self):
        (self.root / 'numba' / 'result.npy').write_bytes(b'not a genuine result')
        with self.assertRaisesRegex(ValueError, 'payload'):
            check_empty_evaluation_cache(self.root)

    def test_nested_empty_directory_refused(self):
        (self.root / 'xdg' / 'unexpected').mkdir()
        with self.assertRaisesRegex(ValueError, 'payload'):
            check_empty_evaluation_cache(self.root)

    def test_extra_top_level_refused(self):
        (self.root / 'unexpected').mkdir()
        with self.assertRaisesRegex(ValueError, 'payload'):
            check_empty_evaluation_cache(self.root)

    def test_missing_top_level_refused(self):
        (self.root / 'numba').rmdir()
        with self.assertRaisesRegex(ValueError, 'payload'):
            check_empty_evaluation_cache(self.root)

    def test_child_symlink_refused(self):
        (self.root / 'numba').rmdir()
        (self.root / 'numba').symlink_to(self.root / 'xdg', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'payload'):
            check_empty_evaluation_cache(self.root)

    def test_root_symlink_refused(self):
        alias = self.root.parent / 'alias'
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'directory'):
            check_empty_evaluation_cache(alias)

    def test_top_level_file_refused(self):
        (self.root / 'numba').rmdir()
        (self.root / 'numba').write_bytes(b'payload')
        with self.assertRaisesRegex(ValueError, 'payload'):
            check_empty_evaluation_cache(self.root)


if __name__ == '__main__':
    unittest.main()
