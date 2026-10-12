"""Portable entry tests: synthetic data only; never fit a real benchmark."""

import ast
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))
from migrate import create_json, sha
from study import fit
from study_inputs import (SNAPSHOT, Sources, branch_input, historical_functions,
                          npz_equal, output_records, validate_views, verify_bundle)
from study_worker import (execute, fresh_options, full_output, load_core, replay,
                          validate_triplets)


class StudyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # macOS /var may alias /private/var. Match the production launcher's
        # canonical paths when this fixture constructs a worker request.
        self.root = Path(self.tmp.name).resolve()
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        rng = np.random.default_rng(42)
        self.views = []
        for i, name in enumerate(("a", "b", "c")):
            self.views.append({"name": name, "dataset": name, "active": rng.normal(size=(45, 8)),
                               "shared": rng.normal(size=(45, 4)), "features": [f"p{j}" for j in range(8)],
                               "rows": np.arange(i * 45, (i + 1) * 45), "fit_local": np.arange(40),
                               "ids": np.asarray([f"{name}{j}" for j in range(45)])})
        self.ids = np.concatenate([v["ids"] for v in self.views])
        self.branch = {"name": "hub_a", "mode": "three-view", "order": ["a", "b", "c"],
                       "extension": {"svd_components": {n: None for n in "abc"}, "gcca_components": 3,
                                     "gcca_initialization_max_iter": 100, "gcca_refinement_max_iter": 100,
                                     "gcca_tolerance": 0.001, "outer_reuse_policy": "allow"},
                       "cache": {"triplet_index_order": ["a", "c", "b"],
                                 "audit": {"outer_reuse_policy": "allow"}, "wrapper_version": "1.1.0"}}
        self.arrays = {v["name"]: {"active": v["active"][:40], "cell_ids": v["ids"][:40]} for v in self.views}
        self.triplets = {}
        for n, key in zip("acb", ("hub", "outer_left", "outer_right")):
            inds = np.arange(30)
            self.triplets[key] = inds
            self.triplets[key + "_cell_id"] = self.arrays[n]["cell_ids"][inds]
        self.triplets["left_score"] = np.linspace(0.1, 0.9, 30)
        self.triplets["right_score"] = np.linspace(0.2, 0.8, 30)

    def materialize(self):
        p = self.bundle / "branches/hub_a"
        p.mkdir(parents=True)
        np.savez_compressed(p / "fit_input.npz", **branch_input(self.views, list("abcd"), [0, 1, 2], False))
        np.savez_compressed(p / "triplet_matching.npz", **self.triplets)
        create_json(p / "config.json", {})
        np.save(self.bundle / "cell_ids.npy", self.ids)
        for v in self.views:
            directory = self.bundle / "views" / v["name"]
            directory.mkdir(parents=True)
            for key in ("active", "rows", "ids", "fit_local"):
                np.save(directory / (key + ".npy"), v[key])
        m = {"schema": "ramp-study.maxfuse.fit_bundle.v1", "status": "complete", "case": "synthetic",
             "catalog_sha256": sha(SNAPSHOT / "catalog.json"), "branches": [self.branch],
             "views": [{k: v[k] for k in ("name", "dataset", "features")} for v in self.views],
             "outputs": output_records(self.bundle)}
        create_json(self.bundle / "MANIFEST.json", m)
        return m

    def test_named_npz_comparison_accepts_historical_outer_permutation(self):
        a, b = self.root / "a.npz", self.root / "b.npz"
        np.savez(a, **branch_input(self.views, list("abcd"), [0, 1, 2], False))
        np.savez(b, **branch_input(self.views, list("abcd"), [0, 2, 1], False))
        self.assertTrue(npz_equal(a, b, ["a", "b", "c"], ["a", "c", "b"]))
        with self.assertRaises(ValueError):
            npz_equal(a, b)

    def test_npz_comparison_rejects_one_changed_value(self):
        a, b = self.root / "a.npz", self.root / "b.npz"
        data = branch_input(self.views, list("abcd"), [0, 1, 2], False)
        np.savez(a, **data)
        data["view2_active"][0, 0] += 0.001
        np.savez(b, **data)
        with self.assertRaises(ValueError):
            npz_equal(a, b, list("abc"), list("abc"))

    def test_pairwise_npz_schema(self):
        data = branch_input(self.views[:2], list("abcd"), [0, 1], True)
        self.assertEqual(set(data), {f"{prefix}{i}{suffix}" for i in (1, 2)
                         for prefix, suffix in [("shared", ""), ("active", ""), ("cell_ids", ""),
                                                ("shared", "_feature_names"), ("active", "_feature_names")]})

    def test_bundle_relocates_without_original_tree(self):
        self.materialize()
        moved = self.root / "moved"
        self.bundle.rename(moved)
        with patch("study_inputs.Sources", side_effect=AssertionError("Original tree must not be consulted")):
            self.assertEqual(verify_bundle(moved)["case"], "synthetic")

    def test_bundle_rejects_missing_extra_and_changed_files(self):
        self.materialize()
        extra = self.bundle / "unexpected"
        extra.write_text("x")
        with self.assertRaises(ValueError): verify_bundle(self.bundle)
        extra.unlink()
        target = self.bundle / "cell_ids.npy"
        target.write_bytes(b"bad")
        with self.assertRaises(ValueError): verify_bundle(self.bundle)
        target.unlink()
        with self.assertRaises(ValueError): verify_bundle(self.bundle)

    def test_bundle_rejects_external_symlink(self):
        self.materialize()
        target = self.bundle / "cell_ids.npy"
        outside = self.root / "outside.npy"
        target.rename(outside)
        target.symlink_to(outside)
        with self.assertRaises(ValueError): verify_bundle(self.bundle)

    def test_bundle_rejects_unsafe_branch_name(self):
        self.branch["name"] = "../../outside"
        self.materialize()
        with self.assertRaises(ValueError): verify_bundle(self.bundle)

    def test_bundle_rejects_duplicate_order(self):
        self.branch["order"] = ["a", "b", "b"]
        self.materialize()
        with self.assertRaises(ValueError): verify_bundle(self.bundle)

    def test_views_partition_and_id_checks(self):
        validate_views(self.views, self.ids, list("abcd"))
        self.views[1]["rows"][0] = 0
        with self.assertRaises(ValueError): validate_views(self.views, self.ids, list("abcd"))

    def test_historical_functions_do_not_execute_top_level(self):
        with patch("pathlib.Path.mkdir", side_effect=AssertionError("Unexpected top-level I/O")):
            functions = historical_functions()
        self.assertTrue(callable(functions["select_fit_positions"]))
        self.assertNotIn("main", functions)

    def test_source_remaps_old_host_but_rejects_traversal(self):
        obj = object.__new__(Sources)
        self.assertEqual(obj.relative("/different/machine/final_xu/benchmark/a.npy"), "benchmark/a.npy")
        for value in ("../x", "/etc/a", "/tmp/final_xu/../../x"):
            with self.assertRaises(ValueError): obj.relative(value)

    def test_triplets_map_by_name(self):
        mapped = validate_triplets(self.branch, self.arrays, self.triplets)
        self.assertEqual(set(mapped), set("abc"))

    def test_triplets_reject_wrong_ids(self):
        self.triplets["outer_left_cell_id"][0] = "BAD"
        with self.assertRaises(ValueError): validate_triplets(self.branch, self.arrays, self.triplets)

    def test_triplets_reject_out_of_bounds(self):
        self.triplets["hub"][0] = 10000
        with self.assertRaises(ValueError): validate_triplets(self.branch, self.arrays, self.triplets)

    def test_triplets_reject_different_policy(self):
        self.branch["extension"]["outer_reuse_policy"] = "drop_reused"
        with self.assertRaises(ValueError): validate_triplets(self.branch, self.arrays, self.triplets)

    def test_triplets_reject_nonfinite_scores(self):
        self.triplets["right_score"][0] = np.nan
        with self.assertRaises(ValueError): validate_triplets(self.branch, self.arrays, self.triplets)

    def test_fresh_options_preserve_archive_parameters(self):
        flags = fresh_options(self.branch)
        self.assertEqual(flags[:6], ["--views", "a", "b", "c", "--hub", "a"])
        self.assertEqual(flags[-2:], ["--outer-reuse-policy", "allow"])
        self.assertEqual(flags[flags.index("--gcca-components") + 1], "3")
        self.assertEqual(fresh_options({"mode": "pairwise"}), [])

    def test_no_silent_fresh_fallback(self):
        self.branch["cache"] = None
        self.materialize()
        with patch("study.subprocess.run") as child:
            with self.assertRaises(ValueError): fit(sys.executable, self.bundle, self.root / "out", "replay")
        child.assert_not_called()
        self.assertFalse((self.root / "out").exists())

    def test_fit_failure_retains_directory_and_refuses_repeat(self):
        self.materialize()
        output = self.root / "out"
        with patch("study.subprocess.run") as child:
            child.return_value.returncode = 9
            with self.assertRaises(RuntimeError): fit(sys.executable, self.bundle, output, "fresh")
        self.assertTrue((output / "FAILED.json").exists())
        self.assertFalse((output / "COMPLETE.json").exists())
        with self.assertRaises(FileExistsError): fit(sys.executable, self.bundle, output, "fresh")

    def test_manifest_change_after_runtime_probe_prevents_fit(self):
        self.materialize()
        from types import SimpleNamespace
        def change_manifest(*args, **kwargs):
            p = self.bundle / "MANIFEST.json"
            value = json.loads(p.read_text())
            value["changed_during_launch"] = True
            p.write_text(json.dumps(value))
            return SimpleNamespace(returncode=0)
        with patch("study.subprocess.run", side_effect=change_manifest) as child:
            with self.assertRaisesRegex(RuntimeError, "manifest changed"):
                fit(sys.executable, self.bundle, self.root / "out", "fresh")
            self.assertEqual(child.call_count, 1)
        self.assertFalse((self.root / "out/hub_a").exists())

    def test_worker_requires_owned_request(self):
        self.materialize()
        output = self.root / "out"
        output.mkdir()
        with self.assertRaises(FileNotFoundError): execute(self.bundle, "hub_a", "replay", output)

    def test_successful_fresh_control_flow_with_mocked_fitting(self):
        self.materialize()
        from types import SimpleNamespace
        called = []
        def fake_launch(python, mode, input_path, config_path, output, extra):
            called.append((mode, input_path, config_path, extra))
            output.mkdir()
            replay(self.branch, input_path, input_path.parent / "triplet_matching.npz", output / "result")
            return output
        def fake_process(command, **kwargs):
            if "probe_runtime.py" in command[2]:
                return SimpleNamespace(returncode=0)
            execute(Path(command[command.index("--bundle") + 1]),
                    command[command.index("--branch") + 1], command[command.index("--route") + 1],
                    Path(command[command.index("--output") + 1]))
            return SimpleNamespace(returncode=0)
        with patch("study.subprocess.run", side_effect=fake_process), patch("study_worker.launch", side_effect=fake_launch):
            out = fit(sys.executable, self.bundle, self.root / "fresh", "fresh")
        self.assertEqual(called[0][0], "three-view")
        self.assertEqual(called[0][3], fresh_options(self.branch))
        self.assertEqual(json.loads((out / "COMPLETE.json").read_text())["route"], "fresh")
        self.assertEqual(np.load(out / "hub_a/values.npy").shape, (135, 3))

    def test_worker_exclusive_start_refuses_second_invocation(self):
        self.materialize()
        out = self.root / "out"
        out.mkdir()
        create_json(out / "REQUEST.json", {"bundle": str(self.bundle), "branch": "hub_a", "route": "replay",
                                           "bundle_manifest_sha256": sha(self.bundle / "MANIFEST.json")})
        execute(self.bundle, "hub_a", "replay", out)
        before = sha(out / "values.npy")
        with self.assertRaises(FileExistsError): execute(self.bundle, "hub_a", "replay", out)
        self.assertEqual(sha(out / "values.npy"), before)

    def test_worker_fixture_under_explicit_symlink_tmpdir(self):
        real = self.root / "real_tmp"
        real.mkdir()
        alias = self.root / "linked_tmp"
        alias.symlink_to(real, target_is_directory=True)
        with patch("tempfile.tempdir", str(alias)):
            fixture = StudyTest("test_worker_exclusive_start_refuses_second_invocation")
            fixture.setUp()
            try:
                self.assertEqual(Path(fixture.tmp.name).parent, alias)
                self.assertEqual(fixture.root.parent, real)
                fixture.test_worker_exclusive_start_refuses_second_invocation()
            finally:
                fixture.doCleanups()

    def test_synthetic_replay_matches_original_numerical_block(self):
        manifest = self.materialize()
        p = self.bundle / "branches/hub_a"
        actual = replay(self.branch, p / "fit_input.npz", p / "triplet_matching.npz", self.root / "replay")
        core = load_core()
        from maxfuse_repro.gcca import fit_archive_gcca, matched_component_correlations
        from maxfuse_repro.projection import build_projection_model
        # Execute the original replay runner's entire numerical block verbatim,
        # not a retyped reference, on synthetic data with permuted old outer order.
        source = SNAPSHOT / "study_sources/benchmark_results/_maxfuse_archive_replay.py"
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "replay_one_hub")
        block = next(n for n in function.body if isinstance(n, ast.Try)).body
        def assignment(n, name):
            return isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name
        start = next(i for i, n in enumerate(block) if assignment(n, "preprocessing"))
        stop = next(i for i, n in enumerate(block) if assignment(n, "outer_names"))
        ctx = {"np": np, "archive_svd_scores": core.archive_svd_scores, "fit_archive_gcca": fit_archive_gcca,
               "build_projection_model": build_projection_model, "matched_component_correlations": matched_component_correlations,
               "view_order": tuple("abc"), "extension": self.branch["extension"],
               "fit_active": {v["name"]: v["active"][:40] for v in self.views},
               "indices_by_view": validate_triplets(self.branch, self.arrays, self.triplets),
               "feature_names": {v["name"]: tuple(v["features"]) for v in self.views}}
        exec(compile(ast.Module(body=block[start:stop], type_ignores=[]), str(source), "exec"), ctx)
        for i, name in enumerate("abc"):
            self.assertTrue(np.array_equal(np.load(actual / f"view{i}_gcca_embedding.npy"), ctx["fit_embeddings"][name]))
        out = self.root / "full"
        out.mkdir()
        self.assertEqual(full_output(self.bundle, manifest, self.branch, actual, out), [135, 3])
        values = np.load(out / "values.npy")
        for v in self.views:
            expected = ctx["model"].transform(v["name"], v["active"], v["features"])
            self.assertTrue(np.array_equal(values[v["rows"]], expected))

    def test_pairwise_full_output_restores_cohort_order(self):
        manifest = self.materialize()
        formal = self.root / "formal"
        formal.mkdir()
        order = np.arange(len(self.ids))[::-1]
        for i, rows in enumerate(np.array_split(order, 2), 1):
            np.save(formal / f"cca_embedding{i}.npy", rows[:, None].astype(float))
            np.save(formal / f"cca_embedding{i}_cell_ids.npy", self.ids[rows])
        out = self.root / "full"
        out.mkdir()
        full_output(self.bundle, manifest, {"mode": "pairwise"}, formal, out)
        self.assertTrue(np.array_equal(np.load(out / "values.npy")[:, 0], np.arange(len(self.ids))))


if __name__ == "__main__":
    unittest.main()
