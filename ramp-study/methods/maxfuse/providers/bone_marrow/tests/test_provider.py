"""BM provider port tests; synthetic fixtures only, no real provider extraction."""

import ast
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import build_inputs as builder
import support


def race_claim(output, raw, benchmark, barrier, queue):
    barrier.wait(timeout=20)
    try:
        support.claim_output(Path(output), Path(raw), Path(benchmark))
        queue.put("won")
    except FileExistsError:
        queue.put("lost")


def race_publish(stage, output, manifest, barrier, queue):
    with patch("support.verify_reference", return_value=manifest):
        barrier.wait(timeout=20)
        try:
            support.publish_new(Path(stage), Path(output))
            queue.put("won")
        except FileExistsError:
            queue.put("lost")


class ProviderTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.benchmark = self.root / "benchmark"
        self.benchmark.mkdir()
        (self.benchmark / "manifest.json").write_text("{}")

    def test_original_source_and_profiles_match_historical_hashes(self):
        support.check_code()

    def test_all_21_nonentry_functions_are_ast_identical(self):
        def functions(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
        original = functions(HERE / "source/build_inputs.py")
        port = functions(HERE / "build_inputs.py")
        self.assertEqual(set(original), set(port))
        self.assertEqual(len(original), 23)
        self.assertEqual([n for n in original if original[n] != port[n]], ["parse_args", "build"])

    def test_constants_and_embedded_r_exporter_identical(self):
        original = ast.parse((HERE / "source/build_inputs.py").read_text())
        assignments = {n.targets[0].id: ast.literal_eval(n.value) for n in original.body
                       if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                       and n.targets[0].id in ("ID_COLUMNS", "HELDOUT", "SHARED_MARKERS", "EXPECTED", "MARKER_ALIASES", "CODEX_EXPORT_R")}
        for name, value in assignments.items():
            self.assertEqual(getattr(builder, name), value)
        self.assertEqual(len(assignments), 6)

    def test_scientific_dataset_loop_identical(self):
        def loop(path):
            tree = ast.parse(path.read_text())
            build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build")
            loop = next(n for n in ast.walk(build) if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple)
                        and [x.id for x in n.target.elts if isinstance(x, ast.Name)] == ["slug", "spec"])
            return ast.dump(loop, include_attributes=False)
        self.assertEqual(loop(HERE / "source/build_inputs.py"), loop(HERE / "build_inputs.py"))

    def test_clr_and_sample_sd_are_original_definitions(self):
        x = np.array([[0., 1., 8.], [4., 2., 1.], [5., 3., 0.]])
        clr = builder.seurat_clr_per_cell(x)
        expected = np.log1p(x / np.exp(np.log1p(x).mean(axis=1))[:, None])
        self.assertTrue(np.array_equal(clr, expected))
        z, _ = builder.marker_sample_zscore(clr, "synthetic", list("abc"), "CLR")
        self.assertTrue(np.array_equal(z, (clr - clr.mean(axis=0)) / clr.std(axis=0, ddof=1)))

    def test_heldout_absent_from_shared_and_active_panels(self):
        self.assertNotIn("CD19", builder.SHARED_MARKERS)
        for d in support.history()["datasets"].values():
            self.assertNotIn("CD19", d["active_markers"])

    def test_contract_rejects_self_reported_changed_row_lock(self):
        target = self.benchmark / "rows.csv"
        target.write_text("wrong")
        m = {"package": "bone_marrow_three_platform_benchmark_v1", "n_total_cells": 88031,
             "source_files_sha256": {}, "files_sha256": {"rows.csv": support.sha(target)}}
        (self.benchmark / "manifest.json").write_text(json.dumps(m))
        with patch("support.locked_records", return_value={"rows.csv": "0" * 64}), patch("support.provider_records", return_value={}):
            with self.assertRaises(ValueError): support.frozen_contract(self.benchmark)

    def test_contract_accepts_only_pinned_rows_and_provider_dictionary(self):
        target = self.benchmark / "rows.csv"
        target.write_text("id\n1\n")
        expected = {"rows.csv": support.sha(target)}
        m = {"package": "bone_marrow_three_platform_benchmark_v1", "n_total_cells": 88031,
             "source_files_sha256": {}, "files_sha256": expected}
        (self.benchmark / "manifest.json").write_text(json.dumps(m))
        with patch("support.locked_records", return_value=expected), patch("support.provider_records", return_value={}):
            self.assertEqual(support.frozen_contract(self.benchmark), m)

    def test_output_refuses_existing_and_source_overlap(self):
        with self.assertRaises(FileExistsError): support.claim_output(self.raw, self.raw, self.benchmark)
        with self.assertRaises(ValueError): support.claim_output(self.raw / "new", self.raw, self.benchmark)

    def test_output_refuses_dangling_symlink(self):
        output = self.root / "link"
        output.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError): support.claim_output(output, self.raw, self.benchmark)

    def run_race(self, function, args):
        ctx = multiprocessing.get_context("spawn")
        barrier, queue = ctx.Barrier(2), ctx.Queue()
        children = [ctx.Process(target=function, args=(*args, barrier, queue)) for _ in range(2)]
        try:
            for child in children: child.start()
            for child in children:
                child.join(timeout=25)
                self.assertEqual(child.exitcode, 0)
            self.assertCountEqual([queue.get(timeout=5) for _ in children], ["won", "lost"])
        finally:
            for child in children:
                if child.is_alive(): child.terminate(); child.join()
            queue.close()

    def test_real_two_process_output_claim(self):
        self.run_race(race_claim, (self.root / "out", self.raw, self.benchmark))

    def stage(self):
        output = support.claim_output(self.root / "out", self.raw, self.benchmark)
        stage = output / "_work" / "processed"
        stage.mkdir(parents=True)
        (stage / "data.txt").write_text("synthetic input")
        m = {"files_sha256": {"data.txt": support.sha(stage / "data.txt")}, "datasets": {}}
        (stage / "manifest.json").write_text(json.dumps(m))
        return output, stage, m

    def test_real_two_process_publication(self):
        output, stage, m = self.stage()
        self.run_race(race_publish, (stage, output, m))
        self.assertEqual((output / "data.txt").read_text(), "synthetic input")

    def test_publish_twice_refused(self):
        output, stage, m = self.stage()
        with patch("support.verify_reference", return_value=m):
            support.publish_new(stage, output)
            with self.assertRaises(FileExistsError): support.publish_new(stage, output)

    def test_publish_never_overwrites_existing_file(self):
        output, stage, m = self.stage()
        (output / "data.txt").write_text("keep")
        with patch("support.verify_reference", return_value=m):
            with self.assertRaises(FileExistsError): support.publish_new(stage, output)
        self.assertEqual((output / "data.txt").read_text(), "keep")
        self.assertFalse((output / "COMPLETE.json").exists())

    def test_incomplete_staging_is_retained(self):
        output = support.claim_output(self.root / "out", self.raw, self.benchmark)
        with self.assertRaises(RuntimeError):
            with support.owned_work(output) as tmp:
                (Path(tmp) / "partial").write_text("diagnostic")
                raise RuntimeError("interrupted")
        self.assertTrue((output / "_work/partial").is_file())
        self.assertTrue((output / "FAILED.json").is_file())
        self.assertFalse((output / "COMPLETE.json").exists())

    def test_reference_rejects_changed_outputs(self):
        output, stage, m = self.stage()
        with patch("support.history", return_value=m):
            support.verify_reference(stage)
            (stage / "data.txt").write_text("changed")
            with self.assertRaises(ValueError): support.verify_reference(stage)


if __name__ == "__main__":
    unittest.main()
