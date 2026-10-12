import multiprocessing as mp
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import replay as r


def compete(directory, barrier, queue):
    target = Path(directory) / 'output'
    barrier.wait(timeout=20)
    try:
        claimed = r.reserve(target)
        r.write(claimed / 'OWNER.json', {'owned': True})
        queue.put('won')
    except FileExistsError:
        queue.put('refused')


class OutputSafety(unittest.TestCase):
    def test_two_process_directory_claim(self):
        context = mp.get_context('spawn')
        for _ in range(3):
            with tempfile.TemporaryDirectory() as temp:
                barrier = context.Barrier(2); queue = context.Queue()
                children = [context.Process(target=compete, args=(temp, barrier, queue)) for _ in range(2)]
                for child in children: child.start()
                for child in children:
                    child.join(timeout=30)
                    self.assertFalse(child.is_alive())
                    self.assertEqual(child.exitcode, 0)
                self.assertEqual(sorted([queue.get(timeout=5), queue.get(timeout=5)]), ['refused', 'won'])
                self.assertEqual(r.read(Path(temp) / 'output/OWNER.json'), {'owned': True})
                queue.close(); queue.join_thread()

    def test_existing_directory_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'existing'; p.mkdir(); r.write(p / 'KEEP.json', {'original': True})
            before = r.tree(p)
            with self.assertRaises(FileExistsError): r.reserve(p)
            self.assertEqual(r.tree(p), before)

    def test_dangling_link_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'alias'; p.symlink_to(Path(temp) / 'absent')
            with self.assertRaises(FileExistsError): r.reserve(p)
            self.assertTrue(p.is_symlink()); self.assertFalse((Path(temp) / 'absent').exists())

    def test_overlap_rejected_before_parent_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); protected = root / 'source'; protected.mkdir()
            for destination in (protected, protected / 'missing/new', root):
                with self.assertRaises((ValueError, FileExistsError)): r.reserve(destination, [protected])
            self.assertEqual(list(protected.iterdir()), [])

    def test_area_symlink_escape_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve(); module = root / 'module'; module.mkdir()
            (module / 'runs').mkdir(); outside = root / 'outside'; outside.mkdir()
            (module / 'runs/alias').symlink_to(outside)
            with patch.object(r, 'HERE', module):
                with self.assertRaises(ValueError): r.local_output(module / 'runs/alias/new', 'runs')
            self.assertEqual(list(outside.iterdir()), [])

    def test_copy_refuses_hardlinked_target_without_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'source.json'; r.write(source, {'original': True})
            target = root / 'target.json'; target.hardlink_to(source); before = r.record(source)
            with self.assertRaises(FileExistsError): r.copy(source, target, before)
            self.assertEqual(r.record(source), before); self.assertEqual(r.record(target), before)

    def test_wrong_hash_leaves_partial_output_not_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'source.json'; r.write(source, {'original': True})
            output = r.reserve(root / 'output'); expected = r.record(source); expected['sha256'] = '0' * 64
            with self.assertRaises(ValueError): r.copy(source, output / 'partial.json', expected)
            self.assertTrue((output / 'partial.json').exists())
            self.assertFalse((output / 'COMPLETE.json').exists())
            with self.assertRaises(FileExistsError): r.reserve(output)

    def test_worker_count_bounds_before_io(self):
        for count in (0, 13):
            with self.assertRaisesRegex(ValueError, 'jobs'):
                r.batch(Path('/absent'), Path('/absent-output'), Path('/absent-python'), count)

    def test_relative_traversal_and_symlink_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'alias').symlink_to(root / 'elsewhere')
            for name in ('../escape', '/absolute', 'alias/file'):
                with self.assertRaises(ValueError): r.child(root, name)


if __name__ == '__main__': unittest.main()
