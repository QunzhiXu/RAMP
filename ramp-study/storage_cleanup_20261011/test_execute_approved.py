"""Small-file safety tests; never touch research payloads."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import execute_approved as e
import prepare_manifest as p


class ExecutionSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.module = self.root / 'module'; self.module.mkdir()
        self.source = self.root / 'source'; self.source.mkdir()
        self.target = self.module / 'ledger.json'; self.target.write_bytes(b'tiny payload')
        self.canonical = self.source / 'ledger.json'; self.canonical.write_bytes(b'tiny payload')
        r, st, _ = p.hash_file(self.target); c, cs, _ = p.hash_file(self.canonical)
        self.row = dict(path=str(self.target), relative_to_module='ledger.json', payload='ledger.json',
            category='B1', origin='original', stat=st, **r,
            canonical=dict(path=str(self.canonical), stat=cs, **c), current_bytes_equal_canonical=True)

    def live(self):
        return e.live_pair(self.row, self.module, self.source)

    def remove(self, emit):
        # Scope binding itself is independently audited and tested in test_candidate_binding.
        with patch.object(e.v, 'check_candidate_binding'):
            e.remove_one(self.row, 3, {}, emit, self.module, self.source)

    def test_live_pair(self):
        self.assertEqual(self.live()[0], self.target)

    def test_unlink_only_target_with_intent_then_receipt(self):
        events = []
        self.remove(events.append)
        self.assertFalse(self.target.exists())
        self.assertEqual(self.canonical.read_bytes(), b'tiny payload')
        self.assertEqual([x['event'] for x in events], ['intent', 'deleted'])
        self.assertEqual(events[0]['row_sha256'], p.digest(self.row))
        self.assertTrue(self.module.exists())

    def test_target_change_refused(self):
        self.target.write_bytes(b'changed')
        with self.assertRaises(ValueError): self.live()

    def test_source_change_refused(self):
        self.canonical.write_bytes(b'changed')
        with self.assertRaises(ValueError): self.live()
        self.assertTrue(self.target.exists())

    def test_path_mismatch_refused(self):
        self.row['path'] = str(self.canonical)
        with self.assertRaises(ValueError): self.live()

    def test_traversal_refused(self):
        self.row['relative_to_module'] = '../source/ledger.json'
        with self.assertRaises(ValueError): self.live()

    def test_wrong_role_refused(self):
        self.row['category'] = 'B2'
        with self.assertRaises(ValueError): self.live()

    def test_wrong_hash_refused_even_matching_stat(self):
        self.row['sha256'] = '0' * 64
        with self.assertRaises(ValueError): self.live()

    def test_symlink_refused(self):
        self.target.unlink(); self.target.symlink_to(self.canonical)
        with self.assertRaises(ValueError): self.live()

    def test_hardlink_refused(self):
        os.link(self.target, self.module / 'alias')
        with self.assertRaises(ValueError): self.live()

    def test_failed_intent_write_prevents_delete(self):
        def fail(_): raise OSError('journal unavailable')
        with self.assertRaises(OSError): self.remove(fail)
        self.assertTrue(self.target.exists())

    def test_change_during_intent_prevents_delete(self):
        def change(_): self.canonical.write_bytes(b'changed')
        with self.assertRaises(ValueError): self.remove(change)
        self.assertTrue(self.target.exists())

    def test_binding_refusal_precedes_any_delete(self):
        with self.assertRaises(ValueError):
            e.remove_one(self.row, 0, {}, lambda _: None, self.module, self.source)
        self.assertTrue(self.target.exists())

    def test_a_metadata_can_differ_only_as_recorded(self):
        self.target.rename(self.module / 'RESULT.json'); self.target = self.module / 'RESULT.json'
        self.target.write_bytes(b'run metadata')
        r, st, _ = p.hash_file(self.target)
        self.row.update(path=str(self.target), relative_to_module='RESULT.json', payload='RESULT.json',
                        origin='failed', category='A_failed', stat=st, current_bytes_equal_canonical=False, **r)
        self.live()
        self.row['current_bytes_equal_canonical'] = True
        with self.assertRaises(ValueError): self.live()


if __name__ == '__main__': unittest.main()
