"""Create-only evidence for the portable fit-input/replay layer, no cohort fit."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from migrate import HERE, create_json, sha
from run import reserve
from study_inputs import SNAPSHOT, npz_equal, verify_bundle
from verify import matches, verify_sources, within


def check_real_bundles(bundles, source_root):
    catalog = json.loads((SNAPSHOT / "catalog.json").read_text())
    record = []
    for case in catalog["cases"]:
        bundle = bundles / case["case"]
        m = verify_bundle(bundle)
        for path, r in m["source_files"].items():
            matches(within(source_root, path), r)
        for link in m["historical_field_links"]:
            obj = json.loads(within(source_root, link["manifest"]).read_text())
            for key in link["keys"]:
                obj = obj[key]
            if obj != link["sha256"] or obj != sha(within(source_root, link["target"])):
                raise ValueError("Native-input exact hash link failed")
        ids = np.load(bundle / "cell_ids.npy", allow_pickle=False)
        original = pd.read_csv(within(source_root, case["common_metadata"]), usecols=["cell_id"], dtype=str).cell_id.to_numpy()
        if not np.array_equal(ids, original):
            raise ValueError("Bundle/common cohort IDs differ")
        branches = []
        for branch, old in zip(m["branches"], case["runs"], strict=True):
            old_manifest = json.loads(within(source_root, old["manifest"]).read_text())
            if "legacy_pairwise_source" in old_manifest:
                value = old_manifest["legacy_pairwise_source"]["alignment"]
                formal = value.split("/final_xu/", 1)[1] + "/formal_fit/manifest.json"
            elif "/alignment/" in old["manifest"]:
                formal = str(Path(old["manifest"]).parent / "formal_fit/manifest.json")
            else:
                formal = old["manifest"]
            fm = json.loads(within(source_root, formal).read_text())
            old_input = within(source_root, fm["input"].split("/final_xu/", 1)[1])
            pairwise = branch["mode"] == "pairwise"
            npz_equal(bundle / "branches" / branch["name"] / "fit_input.npz", old_input,
                      None if pairwise else branch["order"], None if pairwise else fm["views"])
            key = "config_file" if pairwise else "pair_config_map"
            historical_cfg = json.loads(within(source_root, fm[key].split("/final_xu/", 1)[1]).read_text())
            actual_cfg = json.loads((bundle / "branches" / branch["name"] / "config.json").read_text())
            if actual_cfg != historical_cfg:
                raise ValueError("Resolved parameters differ")
            branches.append({"name": branch["name"], "fit_arrays_equal_by_named_view": True,
                             "resolved_pairwise_config_equal": True, "replay_cache": branch["cache"] is not None})
        record.append({"case": case["case"], "cells": len(ids), "branches": branches,
                       "files": len(m["outputs"]), "bytes": sum(r["bytes"] for r in m["outputs"].values()),
                       "bundle_manifest_sha256": sha(bundle / "MANIFEST.json"),
                       "source_records": len(m["source_files"]), "historical_field_links": len(m["historical_field_links"])})
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--bundles", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if not args.output.resolve().is_relative_to(HERE / "validation"):
        p.error("Choose a new module validation directory")
    out = reserve(args.output, (args.source_root, args.bundles, SNAPSHOT))
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
    env.pop("PYTHONPATH", None)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    for key, leaf in [("TMPDIR", "tmp"), ("MPLCONFIGDIR", "cache/matplotlib"),
                      ("NUMBA_CACHE_DIR", "cache/numba"), ("XDG_CACHE_HOME", "cache/xdg")]:
        folder = out / leaf
        folder.mkdir(parents=True)
        env[key] = str(folder)
    report = {"scope": "Native-input-to-fit assembly and synthetic replay/control flow only; no real pairwise or cohort GCCA refit", "commands": []}
    def command(cmd):
        result = subprocess.run(cmd, cwd=out, env=env, text=True, capture_output=True)
        report["commands"].append({"command": list(map(str, cmd)), "returncode": result.returncode,
                                   "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError(f"Validation command failed: {cmd}")
        print("PASS", str(cmd[2]), flush=True)
    python = str(args.python.absolute())
    try:
        command([python, "-B", "-m", "unittest", "discover", "-s", str(HERE / "tests"), "-v"])
        command([python, "-B", str(HERE / "study.py"), "list"])
        command([python, "-B", str(HERE / "study.py"), "--help"])
        report["real_bundles"] = check_real_bundles(args.bundles.resolve(), args.source_root.resolve())
        # Actual subprocess launch of synthetic replay (not mocked, no pairwise).
        sys.path.insert(0, str(HERE / "tests"))
        from test_study import StudyTest
        fixture = StudyTest("test_worker_requires_owned_request")
        fixture.setUp()
        try:
            fixture.materialize()
            moved = fixture.root / "moved_bundle"
            fixture.bundle.rename(moved)
            target = fixture.root / "cli_result"
            command([python, "-B", str(HERE / "study.py"), "fit", "--python", python,
                     "--bundle", str(moved), "--output", str(target), "--route", "replay"])
            done = json.loads((target / "hub_a/COMPLETE.json").read_text())
            report["actual_synthetic_replay_cli"] = {**done, "moved_input_bundle": True,
                "environment": json.loads((target / "runtime.json").read_text())}
        finally:
            fixture.doCleanups()
        report["source_checks"] = verify_sources(SNAPSHOT, args.source_root)
        project, study = args.source_root.resolve().parent, HERE.parents[1]
        baselines = [(study / "inventory_20261010/evidence/source_census.json", project),
                     (study / "inventory_20261010/evidence/curated_sources.json", project),
                     (study / "benchmarks/bone_marrow/validation/code_snapshot_output_safety_20261010.json", study)]
        report["unchanged_baselines"] = []
        for snapshot, base in baselines:
            data = json.loads(snapshot.read_text())
            checked = 0
            for item in data["files"]:
                if item.get("sha256"):
                    matches(within(base, item["path"]), item)
                    checked += 1
            report["unchanged_baselines"].append({"snapshot": str(snapshot), "sha256": sha(snapshot), "checked": checked})
        report["study_readme_sha256"] = sha(study / "README.md")
        report["module_sha256"] = {str(path.relative_to(HERE)): sha(path)
                                    for path in sorted(HERE.glob("*")) if path.is_file()}
        report["module_sha256"].update({str(path.relative_to(HERE)): sha(path) for path in sorted((HERE / "tests").glob("*.py"))})
        report["status"] = "passed"
        create_json(out / "report.json", report)
    except BaseException as error:
        create_json(out / "FAILED.json", {**report, "error": repr(error)})
        raise
    print(out / "report.json")


if __name__ == "__main__":
    main()
