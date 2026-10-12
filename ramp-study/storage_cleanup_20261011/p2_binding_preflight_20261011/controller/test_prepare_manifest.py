import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import prepare_manifest as p


class ManifestSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/private/tmp')
        self.root = Path(self.tmp.name).resolve()
        self.file = self.root / 'payload'
        self.file.write_bytes(b'example')

    def tearDown(self):
        self.tmp.cleanup()

    def test_hash_content_and_stat(self):
        rec, st, data = p.hash_file(self.file, capture=True)
        self.assertEqual(rec, dict(bytes=7, sha256=hashlib.sha256(b'example').hexdigest()))
        self.assertEqual(data, b'example')
        self.assertEqual(st, p.regular(self.file))

    def test_expected_hash_refuses_tampering(self):
        with self.assertRaisesRegex(ValueError, 'sealed record'):
            p.hash_file(self.file, dict(bytes=7, sha256='0'*64))

    def test_expected_size_refuses_tampering(self):
        rec = p.hash_file(self.file)[0] | {'bytes': 8}
        with self.assertRaises(ValueError):
            p.hash_file(self.file, rec)

    def test_leaf_symlink_refused(self):
        alias = self.root / 'alias'
        alias.symlink_to(self.file)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            p.hash_file(alias)

    def test_parent_symlink_refused(self):
        real = self.root / 'real'; real.mkdir()
        (real / 'x').write_bytes(b'x')
        alias = self.root / 'alias'; alias.symlink_to(real)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            p.hash_file(alias / 'x')

    def test_dangling_symlink_refused(self):
        alias = self.root / 'alias'; alias.symlink_to(self.root / 'missing')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            p.member(self.root, 'alias')

    def test_hardlink_refused(self):
        os.link(self.file, self.root / 'linked')
        with self.assertRaisesRegex(ValueError, 'single-link'):
            p.hash_file(self.file)

    def test_directory_not_payload(self):
        with self.assertRaises(ValueError):
            p.hash_file(self.root)

    def test_traversal_refused(self):
        with self.assertRaises(ValueError):
            p.member(self.root, '../x')

    def test_absolute_member_refused(self):
        with self.assertRaises(ValueError):
            p.member(self.root, str(self.file))

    def test_empty_member_refused(self):
        with self.assertRaises(ValueError):
            p.member(self.root, '')

    def test_valid_member(self):
        self.assertEqual(p.member(self.root, 'payload'), self.file)

    def test_changed_during_hash_refused(self):
        original = os.fstat
        calls = []
        def fstat(fd):
            calls.append(fd)
            if len(calls) == 2:
                self.file.write_bytes(b'changed!')
            return original(fd)
        with patch.object(p.os, 'fstat', side_effect=fstat):
            with self.assertRaisesRegex(ValueError, 'during read'):
                p.hash_file(self.file)

    def test_new_json_never_overwrites(self):
        with self.assertRaises(FileExistsError):
            p.write_new(self.file, {})
        self.assertEqual(self.file.read_bytes(), b'example')

    def test_compressed_file_exclusive_and_roundtrip(self):
        import gzip
        path = self.root / 'records.gz'
        with p.compressed_lines(path) as emit:
            emit({'x': 1}); emit({'x': 2})
        with gzip.open(path, 'rt') as f:
            self.assertEqual([json.loads(s) for s in f], [{'x': 1}, {'x': 2}])
        with self.assertRaises(FileExistsError):
            with p.compressed_lines(path): pass

    def test_reserve_output_exclusive(self):
        with patch.object(p, 'HERE', self.root):
            path = self.root / 'manifest_test'
            self.assertEqual(p.reserve_output(path), path)
            with self.assertRaises(FileExistsError): p.reserve_output(path)

    def test_reserve_output_protected_path_rejected_before_write(self):
        with patch.object(p, 'HERE', self.root):
            for path in (self.root / 'runs', self.root / 'nested/manifest_x'):
                with self.assertRaises(ValueError): p.reserve_output(path)
                self.assertFalse(path.exists())

    def test_composite_key_distinguishes_method_and_task(self):
        task = dict.fromkeys(p.TASK_FIELDS, 'x') | {'task_type': 'J2'}
        self.assertNotEqual(p.digest(p.key_for(task)), p.digest(p.key_for(task | {'method': 'y'})))
        self.assertNotEqual(p.digest(p.key_for(task)), p.digest(p.key_for(task | {'task': 'y'})))

    def test_incomplete_identity_refused(self):
        with self.assertRaises(ValueError): p.key_for({'id': 'same-across-methods'})

    def test_j1_missing_marker_has_explicit_null_in_manifest(self):
        task = {k: 'x' for k in p.TASK_FIELDS if k != 'marker'} | {'task_type': 'J1'}
        self.assertIsNone(p.key_for(task)['marker'])
        self.assertEqual(p.key_for(task), p.key_for(task | {'marker': None}))

    def test_j2_marker_is_required(self):
        task = {k: 'x' for k in p.TASK_FIELDS if k != 'marker'} | {'task_type': 'J2'}
        for value in (task, task | {'marker': None}, task | {'marker': ''}):
            with self.assertRaises(ValueError): p.key_for(value)

    def test_j1_cannot_acquire_masked_marker(self):
        task = dict.fromkeys(p.TASK_FIELDS, 'x') | {'task_type': 'J1'}
        with self.assertRaises(ValueError): p.key_for(task)

    def test_categories(self):
        self.assertEqual(p.category('original', 'ledger.json'), 'B1')
        self.assertEqual(p.category('continuation', 'predictions.npz'), 'B2')
        self.assertEqual(p.category('failed', 'RESULT.json'), 'A_failed')
        self.assertEqual(p.category('smoke', 'predictions.json'), 'A_smoke')

    def test_accepted_results_not_candidates(self):
        for name in ('RESULT.json', 'predictions.json', 'WORKER_DONE.json'):
            with self.assertRaises(ValueError): p.category('original', name)

    def test_unknown_origin_refused(self):
        with self.assertRaises(ValueError): p.category('something', 'ledger.json')

    def test_counts_exact(self):
        p.check_counts(p.EXPECTED_COUNTS)
        for values in ({}, p.EXPECTED_COUNTS | {'B1': 32369}, p.EXPECTED_COUNTS | {'unknown': 1}):
            with self.assertRaises(ValueError): p.check_counts(values)

    def test_census_does_not_follow_symlink_directories(self):
        (self.root / 'alias').symlink_to(self.root)
        with self.assertRaises(ValueError): p.census(self.root)

    def test_read_json_hashes_same_bytes(self):
        path = self.root / 'record.json'; path.write_text('{"ok":true}')
        value, rec = p.read_json(path)
        self.assertTrue(value['ok'])
        self.assertEqual(rec, p.hash_file(path)[0])


if __name__ == '__main__':
    unittest.main()
