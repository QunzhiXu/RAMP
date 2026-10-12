import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import study as s
from bootstrap import r


def claim(path, barrier, queue):
    barrier.wait()
    try: r.reserve(path); queue.put('owned')
    except FileExistsError: queue.put('refused')


class Safety(unittest.TestCase):
    def test_real_process_race(self):
        with tempfile.TemporaryDirectory() as temp:
            ctx = mp.get_context('spawn'); b = ctx.Barrier(2); q = ctx.Queue()
            workers = [ctx.Process(target=claim, args=(Path(temp) / 'new', b, q)) for _ in range(2)]
            for worker in workers: worker.start()
            for worker in workers: worker.join(30); self.assertEqual(worker.exitcode, 0)
            self.assertEqual(sorted([q.get(timeout=2), q.get(timeout=2)]), ['owned', 'refused'])

    def test_dangling_output_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'alias'; path.symlink_to(Path(temp) / 'missing')
            with self.assertRaises(FileExistsError): r.reserve(path)

    def test_symlink_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'source'; source.mkdir(); (root / 'alias').symlink_to(source)
            with self.assertRaises(ValueError): r.reserve(root / 'alias/new', [source])
            self.assertEqual(list(source.iterdir()), [])

    def test_hardlink_exclusive_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / 'source'; source.write_text('original')
            target = root / 'target'; os.link(source, target)
            with self.assertRaises(FileExistsError): r.write(target, {'overwrite': 'no'})
            self.assertEqual(source.read_text(), 'original')

    def test_symlinked_payload_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'alias').symlink_to(root / 'missing')
            with self.assertRaises(ValueError): r.tree(root)

    def test_child_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            for name in ('../x', '/tmp/x', 'a/../../x'):
                with self.subTest(name=name), self.assertRaises(ValueError): r.child(Path(temp), name)

    def test_finish_checks_code_before_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError): s.finish_branch(Path(temp), {})
            self.assertFalse((Path(temp) / 'COMPLETE.json').exists())

    def test_finish_writes_completion_last(self):
        with tempfile.TemporaryDirectory() as temp:
            names = []; original = r.write
            def write(path, data): names.append(path.name); return original(path, data)
            with patch.object(r, 'write', side_effect=write): s.finish_branch(Path(temp), s.codes())
            self.assertEqual(names, ['RUN_MANIFEST.json', 'COMPLETE.json'])

    def test_worker_repeat_start_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); r.write(root / 'STARTED.json', {'first': True})
            with self.assertRaises(FileExistsError): r.write(root / 'STARTED.json', {'first': False})

    def test_invalid_jobs_rejected_before_input_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for jobs in (0, 5):
                with self.assertRaisesRegex(ValueError, 'jobs'): s.batch(root / 'missing', root / 'output', Path(sys.executable), jobs, 'all')
            self.assertFalse((root / 'output').exists())

    def test_cli_unknown_bundle_refused_without_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); r.write(root / 'MANIFEST.json', {})
            before = r.tree(root)
            result = subprocess.run([sys.executable, '-B', str(s.HERE / 'study.py'), 'verify-input', '--bundle', str(root)],
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0); self.assertIn('accepted', result.stderr)
            self.assertEqual(before, r.tree(root))


if __name__ == '__main__': unittest.main()
