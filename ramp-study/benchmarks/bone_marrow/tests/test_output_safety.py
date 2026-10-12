"""Concurrency/output regression tests; synthetic files in temporary directories only."""

import hashlib
import importlib.util
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import paths
import build_benchmark as builder
import build_relabel as relabel


def fake_stage(root, value):
    root.mkdir()
    for name in ["bm_codex", "bm_cytof", "triana_abseq"]:
        directory = root / name
        directory.mkdir()
        (directory / "data.csv").write_text(value)
    for name in ["selected_cells.csv.gz", "donor_cell_counts.csv", "marker_manifest.csv",
                 "dataset_manifest.csv", "manifest.json"]:
        (root / name).write_text(value)
    return root


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def competing_task(output, stage, barrier, queue):
    # Force both workers through the existence preflight before either mkdir.
    original_check = paths.check_new_output

    def synchronized_check(*args, **kwargs):
        result = original_check(*args, **kwargs)
        barrier.wait(timeout=20)
        return result

    try:
        with patch.object(paths, "check_new_output", synchronized_check):
            claimed = paths.new_output(output)
        builder.publish(Path(stage), claimed)
        (claimed / "winner_extra.txt").write_text(str(stage))
        queue.put(("published", str(stage)))
    except FileExistsError:
        queue.put(("rejected", str(stage)))
    except Exception as error:
        queue.put(("unexpected", repr(error)))


def competing_publish(output, stage, barrier, queue):
    try:
        barrier.wait(timeout=20)
        builder.publish(Path(stage), Path(output))
        queue.put(("published", str(stage)))
    except FileExistsError:
        queue.put(("rejected", str(stage)))
    except Exception as error:
        queue.put(("unexpected", repr(error)))


class OutputSafetyTests(unittest.TestCase):
    def run_competitors(self, worker, output, stages):
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(2)
        queue = context.Queue()
        children = [context.Process(target=worker, args=(str(output), str(stage), barrier, queue))
                    for stage in stages]
        try:
            for child in children:
                child.start()
            for child in children:
                child.join(timeout=30)
                self.assertFalse(child.is_alive(), "Concurrency worker timed out")
                self.assertEqual(child.exitcode, 0)
            results = [queue.get(timeout=5) for _ in children]
            self.assertEqual(sorted(status for status, _ in results), ["published", "rejected"], results)
            return Path(next(stage for status, stage in results if status == "published"))
        finally:
            for child in children:
                if child.is_alive():
                    child.terminate()
                    child.join(timeout=5)
            queue.close()
            queue.join_thread()

    def test_preflight_is_read_only_and_claim_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "missing_parent" / "output"
            self.assertEqual(paths.check_new_output(target), target.resolve())
            self.assertFalse(target.parent.exists())
            self.assertEqual(paths.new_output(target), target.resolve())
            self.assertTrue(target.is_dir())
            with self.assertRaises(FileExistsError):
                paths.new_output(target)

    def test_two_tasks_after_same_preflight_only_one_can_publish(self):
        for repeat in range(3):
            with self.subTest(repeat=repeat), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                stages = [fake_stage(root / "first", "first"), fake_stage(root / "second", "second")]
                target = root / "output"
                winner = self.run_competitors(competing_task, target, stages)
                self.assertEqual((target / "winner_extra.txt").read_text(), str(winner))
                actual = hashes(target)
                actual.pop("winner_extra.txt")
                self.assertEqual(actual, hashes(winner))

    def test_concurrent_publish_into_empty_claim_never_replaces_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stages = [fake_stage(root / "first", "first"), fake_stage(root / "second", "second")]
            target = paths.new_output(root / "output")
            winner = self.run_competitors(competing_publish, target, stages)
            self.assertEqual(hashes(target), hashes(winner))

    def test_second_publish_preserves_existing_extra_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = fake_stage(root / "first", "first")
            second = fake_stage(root / "second", "second")
            target = paths.new_output(root / "output")
            builder.publish(first, target)
            (target / "bm_codex/extra.txt").write_text("must remain")
            before = hashes(target)
            with self.assertRaises(FileExistsError):
                builder.publish(second, target)
            self.assertEqual(hashes(target), before)
            self.assertFalse(list(root.glob("**/*.backup")))
            self.assertFalse(list(root.glob("**/*.incoming")))

    def test_file_copy_rejects_existing_file_hardlink_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, protected = root / "new", root / "protected"
            source.write_text("new")
            protected.write_text("unchanged")
            (root / "hard").hardlink_to(protected)
            (root / "sym").symlink_to(protected)
            for target in [protected, root / "hard", root / "sym"]:
                with self.subTest(target=target.name), self.assertRaises(FileExistsError):
                    builder.copy_new_file(source, target)
            self.assertEqual(protected.read_text(), "unchanged")

    def test_copy_conflict_after_preflight_does_not_truncate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fake_stage(root / "source", "new")
            target = paths.new_output(root / "output")
            original = builder.copy_new_file

            def conflicting_copy(src, dst):
                dst = Path(dst)
                if dst.name == "selected_cells.csv.gz":
                    dst.write_text("other writer")
                original(src, dst)

            with patch.object(builder, "copy_new_file", conflicting_copy):
                with self.assertRaises(FileExistsError):
                    builder.publish(source, target)
            self.assertEqual((target / "selected_cells.csv.gz").read_text(), "other writer")
            self.assertFalse((target / "manifest.json").exists())

    def test_interrupted_publish_keeps_partial_output_and_no_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = fake_stage(root / "source", "new")
            target = paths.new_output(root / "output")
            original = builder.copy_new_directory

            def interrupted_copy(src, dst):
                if Path(dst).name == "bm_cytof":
                    raise OSError("simulated interrupted copy")
                original(src, dst)

            with patch.object(builder, "copy_new_directory", interrupted_copy):
                with self.assertRaises(OSError):
                    builder.publish(source, target)
            self.assertTrue((target / "bm_codex/data.csv").is_file())
            self.assertFalse((target / "manifest.json").exists())
            with self.assertRaises(FileExistsError):
                paths.new_output(target)
            with self.assertRaises(FileExistsError):
                builder.publish(source, target)

    def test_run_all_does_not_preclaim_child_stages(self):
        spec = importlib.util.spec_from_file_location("bm_run_for_test", HERE / "run.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            target = root / "run"
            stages = []

            def fake_subprocess(command, **kwargs):
                output = Path(command[command.index("--output-dir") + 1])
                self.assertFalse(output.exists())
                stages.append(output.name)
                paths.new_output(output)

            argv = [str(HERE / "run.py"), "--raw-root", str(raw), "--run-dir", str(target)]
            with patch.object(sys, "argv", argv), patch.object(runner.subprocess, "run", fake_subprocess):
                runner.main()
                with self.assertRaises(FileExistsError):
                    runner.main()
            self.assertEqual(stages, ["data", "relabel", "alignment", "umap"])

    def test_relabel_reaches_input_read_after_claim_without_second_mkdir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "data"
            inputs.mkdir()
            output = root / "relabel"
            argv = ["build_relabel.py", "--benchmark-root", str(inputs), "--output-dir", str(output)]
            with patch.object(sys, "argv", argv), patch.object(relabel.pd, "read_csv", side_effect=RuntimeError("input read reached")):
                with self.assertRaisesRegex(RuntimeError, "input read reached"):
                    relabel.main()
            self.assertTrue(output.is_dir())
            with self.assertRaises(FileExistsError):
                paths.new_output(output)

    def test_data_rejects_claimed_output_before_source_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            target = paths.new_output(root / "output")
            argv = ["build_benchmark.py", "--raw-root", str(raw), "--output-dir", str(target)]
            with patch.object(sys, "argv", argv), patch.object(builder, "build_codex") as load:
                with self.assertRaises(FileExistsError):
                    builder.main()
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
