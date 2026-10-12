"""Synthetic create-only/identity tests; never read or execute research code."""

import csv
import gzip
import json
import multiprocessing
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))
import inputs
from freeze_registry import check_historical_raw, freeze


def concurrent_stage(registry, source, output, barrier, queue):
    try:
        barrier.wait(timeout=20)
        inputs.stage(registry, source, output)
        queue.put("success")
    except FileExistsError:
        queue.put("exists")
    except Exception as exc:
        queue.put(f"error: {type(exc).__name__}: {exc}")


class InputsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "source"
        self.data = self.source / "cohort"
        self.data.mkdir(parents=True)
        self.raw = np.array([[0., 1.], [-2., 1.], [0., 4.], [3., 0.]])
        np.save(self.data / "raw.npy", self.raw)
        (self.data / "markers.txt").write_text("A\nB\n")
        self.fit = [
            {"dataset": "D1", "unit_key": "D1|01"},
            {"dataset": "D1", "unit_key": "D1|01"},
            {"dataset": "D2", "unit_key": "D2|01"},
            {"dataset": "D2", "unit_key": "D2|01"},
        ]
        self.evaluation = [
            {"cell_id": f"000{i}", **r, "celltype": "T" if i < 2 else "B", "joint_evaluation": "True"}
            for i, r in enumerate(self.fit)
        ]
        self.write_gz(self.data / "fit_metadata.csv.gz", self.fit)
        self.write_gz(self.data / "evaluation_metadata.csv.gz", self.evaluation)
        (self.data / "manifest.json").write_text(json.dumps({"inputs": {"raw.npy": inputs.sha256(self.data / "raw.npy")}}))
        # Verifying lineage must not import/execute this referenced script.
        self.reference = self.source / "producer.py"
        self.reference.write_text("raise RuntimeError('Never execute me')\n")
        self.spec = {
            "case": "fixture", "kind": "technical", "input_directory": "cohort",
            "evaluation_filename": "evaluation_metadata.csv.gz", "source_manifest": "cohort/manifest.json",
            "expected_shape": [4, 2], "expected_units": 2, "label_column": "celltype",
            "fit_columns": ["dataset", "unit_key"], "references": ["producer.py"],
        }
        self.specs_path = self.base / "specs.json"
        self.specs_path.write_text(json.dumps({"scope": "fixture", "global_references": [], "cases": [self.spec]}))
        self.registry_path = self.base / "registry.json"
        self.registry = freeze(self.specs_path, self.source, self.registry_path)
        self.output = self.base / "output"

    @staticmethod
    def write_gz(path, rows):
        with gzip.open(path, "wt", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def inspect(self, spec=None):
        return inputs.inspect_cohort({k: self.source / p for k, p in inputs.input_paths(self.spec).items()}, spec or self.spec)

    def make_stage(self):
        return inputs.stage(self.registry_path, self.source, self.output)

    def test_source_and_staged_bytes_exact(self):
        before = {p: inputs.sha256(p) for p in self.source.rglob("*") if p.is_file()}
        self.assertEqual(inputs.verify_source(self.registry, self.source)["input_files_checked"], 5)
        self.make_stage()
        self.assertEqual(inputs.verify_staged(self.output, self.registry_path)["status"], "passed")
        for role, filename in inputs.FILENAMES.items():
            source = self.source / self.registry["cases"][0]["files"][role]["path"]
            dest = self.output / "fixture" / filename
            self.assertEqual(source.read_bytes(), dest.read_bytes())
            self.assertFalse(source.samefile(dest))
        self.assertTrue(np.array_equal(np.load(self.output / "fixture/raw.npy"), self.raw))
        self.assertEqual(before, {p: inputs.sha256(p) for p in before})

    def test_staged_verification_after_move_without_source(self):
        self.make_stage()
        self.source.rename(self.base / "unavailable-source")
        self.output.rename(self.base / "moved")
        result = inputs.verify_staged(self.base / "moved", self.registry_path)
        self.assertFalse(result["requires_original_source_tree"])

    def test_reject_existing_output_preserves_all_files(self):
        self.make_stage()
        sentinel = self.output / "user-data"
        sentinel.write_text("KEEP")
        with self.assertRaises(FileExistsError):
            self.make_stage()
        self.assertEqual(sentinel.read_text(), "KEEP")

    def test_reject_existing_empty_output(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.make_stage()

    def test_reject_output_symlinks(self):
        self.output.symlink_to(self.base / "missing")
        with self.assertRaises(FileExistsError):
            self.make_stage()
        self.assertFalse((self.base / "missing").exists())

    def test_reject_source_overlap(self):
        with self.assertRaises(ValueError):
            inputs.stage(self.registry_path, self.source, self.source / "new-output")
        self.assertFalse((self.source / "new-output").exists())

    def test_exclusive_file_copy_rejects_existing_hardlink(self):
        target = self.base / "alias.npy"
        target.hardlink_to(self.data / "raw.npy")
        before = inputs.sha256(target)
        with self.assertRaises(FileExistsError):
            inputs.copy_checked(self.data / "raw.npy", target, self.registry["cases"][0]["files"]["raw"])
        self.assertEqual(inputs.sha256(target), before)

    def test_reject_changed_source_before_output_reservation(self):
        (self.data / "markers.txt").write_text("B\nA\n")
        with self.assertRaises(ValueError):
            self.make_stage()
        self.assertFalse(self.output.exists())

    def test_reject_changed_code_reference(self):
        self.reference.write_text("changed\n")
        with self.assertRaises(ValueError):
            inputs.verify_source(self.registry, self.source)

    def test_interrupted_stage_preserves_partial_directory(self):
        with mock.patch.object(inputs, "copy_checked", side_effect=RuntimeError("interrupt")):
            with self.assertRaises(RuntimeError):
                self.make_stage()
        self.assertTrue(self.output.is_dir())
        self.assertFalse((self.output / "COMPLETE.json").exists())
        with self.assertRaises(FileExistsError):
            self.make_stage()

    def test_source_changes_during_copy_no_complete_manifest(self):
        copy = inputs.copy_checked
        def change_reference(source, target, record):
            copy(source, target, record)
            if Path(target).name == "markers.txt":
                self.reference.write_text("changed during copy\n")
        with mock.patch.object(inputs, "copy_checked", side_effect=change_reference):
            with self.assertRaises(ValueError):
                self.make_stage()
        self.assertTrue(self.output.is_dir())
        self.assertFalse((self.output / "COMPLETE.json").exists())

    def test_concurrent_processes_only_one_publishes(self):
        context = multiprocessing.get_context("spawn")
        for repeat in range(3):
            with self.subTest(repeat=repeat):
                output = self.base / f"race-{repeat}"
                barrier, queue = context.Barrier(2), context.Queue()
                processes = [context.Process(target=concurrent_stage, args=(
                    self.registry_path, self.source, output, barrier, queue)) for _ in range(2)]
                try:
                    for process in processes:
                        process.start()
                    for process in processes:
                        process.join(timeout=25)
                        self.assertEqual(process.exitcode, 0)
                    self.assertCountEqual([queue.get(timeout=5) for _ in range(2)], ["success", "exists"])
                    self.assertEqual(inputs.verify_staged(output, self.registry_path)["status"], "passed")
                finally:
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                            process.join()
                    queue.close()

    def test_freeze_snapshot_refuses_overwrite(self):
        before = inputs.sha256(self.registry_path)
        with self.assertRaises(FileExistsError):
            freeze(self.specs_path, self.source, self.registry_path)
        self.assertEqual(inputs.sha256(self.registry_path), before)

    def test_freeze_cannot_write_inside_source(self):
        with self.assertRaises(ValueError):
            freeze(self.specs_path, self.source, self.source / "new-registry.json")

    def test_freeze_requires_historical_raw_hash(self):
        (self.data / "manifest.json").write_text("{}")
        with self.assertRaises(ValueError):
            freeze(self.specs_path, self.source, self.base / "new-registry.json")
        self.assertFalse((self.base / "new-registry.json").exists())

    def test_manifest_correct_hash_only_in_note_is_rejected(self):
        correct = inputs.sha256(self.data / "raw.npy")
        for kind, field in [("technical", "inputs"), ("application_alignment", "output_sha256")]:
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                check_historical_raw({field: {"raw.npy": "incorrect"}, "note": correct}, kind, correct)

    def test_manifest_exact_fields_for_both_kinds(self):
        correct = inputs.sha256(self.data / "raw.npy")
        check_historical_raw({"inputs": {"raw.npy": correct}}, "technical", correct)
        check_historical_raw({"output_sha256": {"raw.npy": correct}}, "application_alignment", correct)

    def test_manifest_no_cross_schema_fallback(self):
        correct = inputs.sha256(self.data / "raw.npy")
        for kind, field in [("technical", "output_sha256"), ("application_alignment", "inputs")]:
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                check_historical_raw({field: {"raw.npy": correct}}, kind, correct)

    def test_manifest_missing_malformed_or_unknown_kind(self):
        for value in [None, [], {}, {"inputs": "raw.npy"}, {"inputs": {"raw.npy": None}}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                check_historical_raw(value, "technical", "abc")
        with self.assertRaises(ValueError):
            check_historical_raw({}, "unknown", "abc")

    def test_traversal_absolute_and_symlink_escape(self):
        (self.source / "escape").symlink_to(self.base)
        for name in ("../specs.json", str(self.registry_path), "escape/specs.json"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                inputs.child(self.source, name)

    def test_scalar_matrix(self):
        np.save(self.data / "raw.npy", np.array(1.))
        with self.assertRaisesRegex(ValueError, "shape"):
            self.inspect()

    def test_non_finite_matrix(self):
        self.raw[0, 0] = np.nan
        np.save(self.data / "raw.npy", self.raw)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.inspect()

    def test_duplicate_marker(self):
        (self.data / "markers.txt").write_text("A\nA\n")
        with self.assertRaises(ValueError):
            self.inspect()

    def test_duplicate_cell_id(self):
        self.evaluation[1]["cell_id"] = self.evaluation[0]["cell_id"]
        self.write_gz(self.data / "evaluation_metadata.csv.gz", self.evaluation)
        with self.assertRaisesRegex(ValueError, "duplicate cell_id"):
            self.inspect()

    def test_label_leak_in_fit(self):
        self.write_gz(self.data / "fit_metadata.csv.gz", self.evaluation)
        with self.assertRaisesRegex(ValueError, "labels leaked"):
            self.inspect()

    def test_fit_eval_dataset_order_mismatch(self):
        self.write_gz(self.data / "fit_metadata.csv.gz", list(reversed(self.fit)))
        with self.assertRaisesRegex(ValueError, "row order"):
            self.inspect()

    def test_fit_eval_cell_id_order_mismatch(self):
        rows = [{"cell_id": str(3-i), **r} for i, r in enumerate(self.fit)]
        self.write_gz(self.data / "fit_metadata.csv.gz", rows)
        spec = {**self.spec, "fit_columns": ["cell_id", "dataset", "unit_key"]}
        with self.assertRaisesRegex(ValueError, "cell_id row order"):
            self.inspect(spec)

    def test_metadata_length_mismatch(self):
        self.write_gz(self.data / "evaluation_metadata.csv.gz", self.evaluation[:-1])
        with self.assertRaisesRegex(ValueError, "row counts"):
            self.inspect()

    def test_joint_flag_is_not_coerced(self):
        self.evaluation[0]["joint_evaluation"] = "not false"
        self.write_gz(self.data / "evaluation_metadata.csv.gz", self.evaluation)
        with self.assertRaisesRegex(ValueError, "joint_evaluation"):
            self.inspect()

    def test_incomplete_output_cannot_verify(self):
        self.output.mkdir()
        with self.assertRaises(FileNotFoundError):
            inputs.verify_staged(self.output, self.registry_path)

    def test_extra_or_changed_staged_file_rejected(self):
        self.make_stage()
        (self.output / "fixture/markers.txt").write_text("B\nA\n")
        with self.assertRaises(ValueError):
            inputs.verify_staged(self.output, self.registry_path)

    def test_extra_staged_file_rejected(self):
        self.make_stage()
        (self.output / "fixture/unexpected").write_text("extra")
        with self.assertRaises(ValueError):
            inputs.verify_staged(self.output, self.registry_path)

    def test_staged_registry_not_self_certifying(self):
        self.make_stage()
        (self.output / "registry.json").write_text("{}")
        with self.assertRaises(ValueError):
            inputs.verify_staged(self.output, self.registry_path)

    def test_help_and_import_are_non_mutating(self):
        before = {p: inputs.sha256(p) for p in self.base.rglob("*") if p.is_file()}
        for script in ("inputs.py", "freeze_registry.py"):
            subprocess.run([sys.executable, "-B", str(MODULE / script), "--help"],
                           cwd=self.base, check=True, capture_output=True)
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from unittest.mock import patch; "
            "import numpy; "
            "guards=[patch('pathlib.Path.open',side_effect=AssertionError('opened file')), "
            "patch('subprocess.Popen',side_effect=AssertionError('subprocess'))]; "
            "[g.start() for g in guards]; import inputs, freeze_registry"
        )
        subprocess.run([sys.executable, "-B", "-c", code, str(MODULE)], cwd=self.base,
                       check=True, capture_output=True)
        self.assertEqual(before, {p: inputs.sha256(p) for p in self.base.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
