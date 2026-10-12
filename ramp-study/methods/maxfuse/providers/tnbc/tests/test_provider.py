"""Synthetic TNBC scientific-equivalence and exclusive-publication tests."""

import ast
import copy
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import build_inputs as builder
import tnbc_support as support


def claim(output, raw, lock):
    h = {"source_files": {"canonical_cell_lock": {"sha256": support.sha(lock)}}}
    with patch("tnbc_support.check_code", return_value=h):
        return support.claim_output(output, raw, lock)


def race_claim(output, raw, lock, barrier, queue):
    barrier.wait(timeout=20)
    try:
        claim(Path(output), Path(raw), Path(lock))
        queue.put("won")
    except FileExistsError:
        queue.put("lost")


def race_publish(stage, root, manifest, barrier, queue):
    with patch("tnbc_support.verify_reference", return_value=manifest):
        barrier.wait(timeout=20)
        try:
            support.publish_new(Path(stage), Path(root))
            queue.put("won")
        except FileExistsError:
            queue.put("lost")


class ProviderTest(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.root = Path(t.name).resolve()
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.lock = self.root / "lock.csv"
        self.lock.write_text("id\n1\n")

    def test_source_snapshots_and_r_recipe_pinned(self):
        support.check_code()
        self.assertEqual(support.sha(HERE / "extract_provider_rows.R"), support.history()["source_files"]["provider_extraction_recipe"]["sha256"])

    def test_15_nonentry_functions_numerically_identical(self):
        def functions(path):
            nodes = {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
            for n in nodes.values():
                if isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str):
                    n.body.pop(0)
            return {k: ast.dump(v, include_attributes=False) for k,v in nodes.items()}
        old, new = [functions(HERE / p) for p in ("source/build_inputs.py", "build_inputs.py")]
        self.assertEqual(set(old) - set(new), {"verify_existing_output", "install_staged_output"})
        names = set(new) - {"main"}
        self.assertEqual(len(names), 15)
        for name in names:
            self.assertEqual(old[name], new[name], name)

    def test_all_15_scientific_constants_identical(self):
        tree = ast.parse((HERE / "source/build_inputs.py").read_text())
        names = list(vars(builder))
        checked = []
        started = False
        for n in tree.body:
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
                name = n.targets[0].id
                if name == "SCHEMA": started = True
                if started:
                    self.assertEqual(getattr(builder, name), ast.literal_eval(n.value), name)
                    checked.append(name)
        self.assertEqual(len(checked), 15)

    def test_scientific_main_loops_identical(self):
        def loops(path):
            tree = ast.parse(path.read_text())
            main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
            return [ast.dump(n, include_attributes=False) for n in ast.walk(main)
                    if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "dataset"]
        self.assertEqual(loops(HERE / "source/build_inputs.py"), loops(HERE / "build_inputs.py"))
        self.assertEqual(len(loops(HERE / "build_inputs.py")), 2)

    def test_asinh1_sample_sd_unchanged(self):
        x = np.array([[0.,1.],[3.,2.],[4.,5.]])
        with patch.dict(builder.EXPECTED_CELLS, {"synthetic": 3}):
            with patch.dict(builder.DATASET_SLUGS, {"synthetic": "synthetic"}):
                z, rows = builder.transform_dataset("synthetic", x, ["a","b"], {"a":"a","b":"b"})
                a = np.arcsinh(x)
                np.testing.assert_array_equal(z, (a-a.mean(axis=0))/a.std(axis=0, ddof=1))
                self.assertEqual({r["zscore_ddof"] for r in rows}, {1})
                with self.assertRaises(AssertionError):
                    builder.transform_dataset("synthetic", -x, ["a","b"], {"a":"a","b":"b"})

    def test_meyer_retains_biological_histone_states(self):
        markers = builder.EXPECTED_H5_SOURCE_MARKERS["Meyer IMC"]
        _, active, _ = builder.active_from_source("Meyer IMC", np.ones((2, len(markers))), markers)
        self.assertEqual(len(active), 36)
        self.assertIn("H3K27ME3", active)
        self.assertIn("PH3", active)
        self.assertNotIn("HISTONEH3", active)

    def test_shiao_compartment_mean_and_original_row_order(self):
        table = {builder.SHIAO_ID_COLUMN: [f"s{i}" for i in range(6)]}
        for k, root in enumerate(builder.SHIAO_SOURCE_ROOTS):
            table[root + "_nuclei_mean"] = np.arange(6)+k
            table[root + "_membrane_mean"] = np.arange(6)+k+4
        path = self.raw / "source.csv"
        pd.DataFrame(table).to_csv(path, index=False)
        meta = pd.DataFrame({"source_index_zero_based":[4,1], "source_cell_id":["s4","s1"]})
        with patch.object(builder, "SHIAO_COUNTS", path), patch.object(builder, "EXPECTED_SHIAO_SOURCE_ROWS", 6):
            values, markers, _, audit = builder.load_shiao_dataset(meta)
        np.testing.assert_array_equal(values[:,0], [7,4])
        self.assertEqual(markers[0], "CD8")
        self.assertFalse(audit["maxfuse_only_compartment_override"])

    def test_wrong_cell_lock_refused_before_claim(self):
        h = {"source_files":{"canonical_cell_lock":{"sha256":"0"*64}}}
        with patch("tnbc_support.check_code", return_value=h):
            with self.assertRaises(ValueError): support.claim_output(self.root / "new", self.raw, self.lock)
        self.assertFalse((self.root / "new").exists())

    def test_existing_output_and_raw_overlap_refused(self):
        with self.assertRaises(FileExistsError): claim(self.raw, self.raw, self.lock)
        with self.assertRaises(ValueError): claim(self.raw / "new", self.raw, self.lock)

    def test_dangling_output_link_refused(self):
        p = self.root / "alias"
        p.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError): claim(p, self.raw, self.lock)

    def race(self, function, args):
        ctx = multiprocessing.get_context("spawn")
        barrier, queue = ctx.Barrier(2), ctx.Queue()
        children = [ctx.Process(target=function, args=(*args,barrier,queue)) for _ in range(2)]
        try:
            for p in children: p.start()
            for p in children:
                p.join(25)
                self.assertEqual(p.exitcode, 0)
            self.assertCountEqual([queue.get(timeout=5) for _ in children], ["won","lost"])
        finally:
            for p in children:
                if p.is_alive(): p.terminate(); p.join()
            queue.close()

    def stage(self):
        out = claim(self.root / "out", self.raw, self.lock)
        stage = out / "_work/processed"
        stage.mkdir(parents=True)
        (stage / "value.txt").write_text("new")
        m = {"output_sha256":{"value.txt":support.sha(stage / "value.txt")}}
        (stage / "input_manifest.json").write_text(json.dumps(m))
        return out, stage, m

    def test_real_two_process_claim(self):
        self.race(race_claim, (self.root / "out", self.raw, self.lock))

    def test_real_two_process_publish(self):
        out, stage, m = self.stage()
        self.race(race_publish, (stage,out,m))
        self.assertTrue((out / "COMPLETE.json").exists())

    def test_duplicate_publish_refused(self):
        out, stage, m = self.stage()
        with patch("tnbc_support.verify_reference", return_value=m):
            support.publish_new(stage,out)
            with self.assertRaises(FileExistsError): support.publish_new(stage,out)
        self.assertNotEqual((stage / "value.txt").stat().st_ino, (out / "value.txt").stat().st_ino)

    def test_existing_file_preserved_on_failure(self):
        out, stage, m = self.stage()
        (out / "value.txt").write_text("existing")
        with patch("tnbc_support.verify_reference", return_value=m):
            with self.assertRaises(FileExistsError): support.publish_new(stage,out)
        self.assertEqual((out / "value.txt").read_text(), "existing")
        self.assertFalse((out / "COMPLETE.json").exists())

    def test_failure_record_retains_staging(self):
        out, stage, _ = self.stage()
        support.fail(out, RuntimeError("interrupted"))
        self.assertTrue((stage / "value.txt").exists())
        self.assertTrue((out / "FAILED.json").exists())
        self.assertFalse((out / "COMPLETE.json").exists())

    def test_source_tamper_not_accepted_by_self_report(self):
        expected = {"source_files":{"row":{"sha256":"0"*64}}}
        records = {"row":{"sha256":support.sha(self.lock), "path":str(self.lock)}}
        with patch("tnbc_support.check_code", return_value=expected):
            with self.assertRaises(ValueError): support.verify_sources_unchanged(records)

    def test_changed_generated_file_rejected(self):
        out, stage, m = self.stage()
        (stage / "value.txt").write_text("tampered")
        with patch("tnbc_support.history", return_value=m):
            with self.assertRaises(ValueError): support.verify_reference(stage)


if __name__ == "__main__":
    unittest.main()
