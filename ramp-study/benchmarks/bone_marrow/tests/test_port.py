"""Lightweight port checks; no provider data or historical source tree needed."""

import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
STUDY = HERE.parents[1]
sys.path.insert(0, str(HERE / "scripts"))
from paths import new_output
from build_benchmark import exact_uniform_indices, MARKERS14
from build_relabel import relabel_codex, relabel_cytof


class SourceTests(unittest.TestCase):
    def test_vendor_and_r_export_byte_identical(self):
        records = json.loads((HERE / "provenance/original_sources.json").read_text())["files"]
        for record in records:
            if "/vendor/" not in record["destination"] and not record["destination"].endswith(".R"):
                continue
            relative = Path(record["destination"]).relative_to("ramp-study")
            with self.subTest(file=str(relative)):
                self.assertEqual(hashlib.sha256((STUDY / relative).read_bytes()).hexdigest(), record["sha256"])

    def test_unmodified_helper_function_ast(self):
        records = json.loads((HERE / "provenance/original_sources.json").read_text())["files"]
        for record in records:
            if not record["destination"].endswith(".py") or "/vendor/" in record["destination"]:
                continue
            path = STUDY / Path(record["destination"]).relative_to("ramp-study")
            functions = {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
            for name, expected in record["function_ast_sha256"].items():
                if name in {"main", "parse_args"}:  # CLI/path wiring changed; inspect separately.
                    continue
                if path.name == "build_benchmark.py" and name in {
                    "atomic_copy_file", "atomic_replace_directory", "publish"
                }:  # Explicit output-safety fix, covered by concurrent publication tests.
                    continue
                actual = hashlib.sha256(ast.dump(functions[name], include_attributes=False).encode()).hexdigest()
                with self.subTest(file=path.name, function=name):
                    self.assertEqual(actual, expected)

    def test_no_external_research_tree_runtime_paths(self):
        for path in (HERE / "scripts").glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and "\n" not in node.value:
                    self.assertNotIn("/Users/", node.value)
                    self.assertNotIn("test0827", node.value)


class SamplingAndLabelsTests(unittest.TestCase):
    def test_cytof_rng_and_order(self):
        groups = np.repeat(["H2", "H1"], 50)
        expected = []
        for index, name in enumerate(["H1", "H2"]):
            pool = np.flatnonzero(groups == name)
            expected.extend(np.random.default_rng(20260821 + 104729 * (index + 1)).choice(pool, 12, replace=False))
        actual = exact_uniform_indices(groups, 12, 20260821)
        np.testing.assert_array_equal(actual, np.sort(expected))
        self.assertEqual(len(np.unique(actual)), 24)

    def test_insufficient_pool_rejected(self):
        with self.assertRaises(ValueError):
            exact_uniform_indices(np.array(["H1"]), 2, 42)

    def test_cd19_excluded(self):
        self.assertEqual(len(MARKERS14), 14)
        self.assertNotIn("CD19", MARKERS14)

    def test_macrophage_requires_both_provider_labels(self):
        frame = pd.DataFrame({"cluster_anno_l1": ["Myeloid", "Macrophages", "Macrophages", "unknown"],
                              "cluster_anno_l2": ["Macrophages", "Macrophages", "other", "other"]})
        self.assertEqual(relabel_codex(frame).tolist(), ["Other myeloid", "Macrophage", "Other myeloid", "Unclassified / technical"])

    def test_nk_merge(self):
        self.assertEqual(relabel_cytof(np.array(["CD16+ NK cells", "CD16- NK cells"])).tolist(), ["NK", "NK"])


class PathTests(unittest.TestCase):
    def test_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                new_output(directory)

    def test_nested_input_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                new_output(root / "output", [root])

    def test_siblings_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input").mkdir()
            self.assertEqual(new_output(root / "output", [root / "input"]), (root / "output").resolve())

    def test_symlink_and_hardlink_targets_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_text("unchanged")
            for name, create in [("sym", lambda p: p.symlink_to(source)), ("hard", lambda p: p.hardlink_to(source))]:
                target = root / name
                create(target)
                with self.assertRaises(FileExistsError):
                    new_output(target, [source])
            self.assertEqual(source.read_text(), "unchanged")

    def test_broken_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "broken"
            target.symlink_to(Path(directory) / "missing")
            with self.assertRaises(FileExistsError):
                new_output(target)


if __name__ == "__main__":
    unittest.main()
