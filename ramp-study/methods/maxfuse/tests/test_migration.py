"""Control-flow/provenance tests; no cohort fitting or upstream source writes."""

import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))
from migrate import Collector, check_cell_order, field, sha
from run import command, launch, reserve
from verify import within, verify_local


def race(output, protected, barrier, queue):
    barrier.wait(timeout=20)
    try:
        reserve(output, (protected,))
        queue.put("won")
    except FileExistsError:
        queue.put("lost")


class MigrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.input = self.source / "input.npz"
        self.input.write_bytes(b"synthetic input; never fitted")
        self.config = self.source / "config.json"
        self.config.write_text("{}")

    def test_local_snapshot_byte_equivalence(self):
        result = verify_local(MODULE / "frozen_20261010")
        self.assertEqual(result["copied_text_files"], 173)

    def test_exact_hash_link_not_note(self):
        p = self.source / "manifest.json"
        p.write_text(json.dumps({"raw": "bad", "note": sha(self.input)}))
        with self.assertRaises(ValueError):
            Collector(self.source).attest(p, ["raw"], self.input, "input")

    def test_exact_hash_link(self):
        p = self.source / "manifest.json"
        p.write_text(json.dumps({"outputs": {"raw": sha(self.input)}}))
        collector = Collector(self.source)
        collector.attest(p, ["outputs", "raw"], self.input, "input")
        self.assertEqual(len(collector.links), 1)

    def test_array_copy_prohibited(self):
        with self.assertRaises(ValueError):
            Collector(self.source).add(self.input, "input", copy=True)

    def test_reject_traversal_and_absolute_paths(self):
        for name in ("../source/input.npz", str(self.input)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                within(self.source, name)

    def test_symlink_escape(self):
        (self.source / "escape").symlink_to(self.root)
        with self.assertRaises(ValueError):
            within(self.source, "escape/source/escape")

    def test_output_cannot_overlap_inputs(self):
        with self.assertRaises(ValueError):
            reserve(self.input / "nested", (self.input,))
        with self.assertRaises(FileExistsError):
            reserve(self.source, (self.input,))

    def test_existing_output_and_sentinel_preserved(self):
        output = self.root / "output"
        output.mkdir()
        (output / "keep").write_text("keep")
        with self.assertRaises(FileExistsError):
            reserve(output, (self.input,))
        self.assertEqual((output / "keep").read_text(), "keep")

    def test_dangling_output_symlink(self):
        output = self.root / "output"
        output.symlink_to(self.root / "absent")
        with self.assertRaises(FileExistsError):
            reserve(output, (self.input,))

    def test_actual_two_process_output_claim(self):
        ctx = multiprocessing.get_context("spawn")
        barrier, queue = ctx.Barrier(2), ctx.Queue()
        jobs = [ctx.Process(target=race, args=(self.root / "output", self.input, barrier, queue)) for _ in range(2)]
        try:
            for p in jobs: p.start()
            for p in jobs:
                p.join(timeout=25)
                self.assertEqual(p.exitcode, 0)
            self.assertCountEqual([queue.get(timeout=5) for _ in jobs], ["won", "lost"])
        finally:
            for p in jobs:
                if p.is_alive(): p.terminate(); p.join()
            queue.close()

    def test_command_preserves_explicit_parameters(self):
        extra = ["--views", "a", "b", "c", "--hub", "a", "--svd-components", "none", "none", "none", "--gcca-components", "15", "--outer-reuse-policy", "allow"]
        cmd = command("/env/bin/python", MODULE, "three-view", self.input, self.config, self.root / "result", extra)
        self.assertEqual(cmd[3:7], ["--input", str(self.input), "--pair-config-map", str(self.config)])
        self.assertEqual(cmd[7:-2], extra)
        self.assertEqual(cmd[-2:], ["--output", str(self.root / "result")])

    def test_reject_override_and_argparse_abbreviations(self):
        for option in ["--output", "--output=x", "--out", "--inp=x", "--con", "--pair-conf", "--o=x"]:
            with self.subTest(option=option), self.assertRaises(ValueError):
                command("python", MODULE, "pairwise", self.input, self.config, self.root / "result", [option])

    def test_launcher_failure_retains_partial(self):
        output = self.root / "output"
        with patch("run.subprocess.run") as child:
            child.return_value.returncode = 2
            with self.assertRaises(RuntimeError):
                launch(sys.executable, "pairwise", self.input, self.config, output)
        self.assertTrue((output / "REQUEST.json").exists())
        self.assertTrue((output / "FAILED.json").exists())
        self.assertFalse((output / "COMPLETE.json").exists())
        with self.assertRaises(FileExistsError):
            launch(sys.executable, "pairwise", self.input, self.config, output)

    def test_launcher_does_not_resolve_venv_python_symlink(self):
        python = self.root / "venv/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        with patch("run.subprocess.run") as child:
            child.return_value.returncode = 2
            with self.assertRaises(RuntimeError):
                launch(python, "pairwise", self.input, self.config, self.root / "out")
        self.assertEqual(child.call_args.args[0][0], str(python))

    def test_success_manifest_is_checked(self):
        output = self.root / "output"
        def fake(cmd, **kw):
            folder = Path(cmd[-1]); folder.mkdir()
            (folder / "manifest.json").write_text(json.dumps({"status": "complete", "input_sha256": sha(self.input), "config_file_sha256": sha(self.config)}))
            return type("Result", (), {"returncode": 0})()
        with patch("run.subprocess.run", side_effect=fake):
            launch(sys.executable, "pairwise", self.input, self.config, output)
        self.assertTrue((output / "COMPLETE.json").exists())
        self.assertEqual(self.config.read_text(), "{}")

    def test_input_changes_during_run_no_completion(self):
        output = self.root / "output"
        def fake(*args, **kw):
            self.input.write_bytes(b"changed")
            return type("Result", (), {"returncode": 0})()
        with patch("run.subprocess.run", side_effect=fake):
            with self.assertRaises(RuntimeError):
                launch(sys.executable, "pairwise", self.input, self.config, output)
        self.assertFalse((output / "COMPLETE.json").exists())


if __name__ == "__main__":
    unittest.main()
