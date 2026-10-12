"""Create-only BM provider-port evidence; never regenerate inputs or fit MaxFuse."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import support
from support import HERE, MAXFUSE, create_json, sha
from run import reserve
from study_inputs import npz_equal, verify_bundle
from verify import matches, verify_sources, within


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "raw-root", "benchmark-root", "rebuilt", "bundle", "provider-python", "maxfuse-python", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.output.resolve().is_relative_to(HERE / "validation"):
        parser.error("Choose a new directory under this provider module's validation/")
    out = reserve(args.output, (args.source_root, args.raw_root, args.benchmark_root,
                               args.rebuilt, args.bundle, HERE / "source"))
    report = {"scope": "BM provider extraction/preprocessing equivalence and 3-hub input assembly; no MaxFuse cohort fit",
              "captured_utc": datetime.now(timezone.utc).isoformat(), "commands": []}
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTHONPATH", None)
    # Retain inherited default TMPDIR, including macOS aliases.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    for key, leaf in (("MPLCONFIGDIR", "matplotlib"), ("NUMBA_CACHE_DIR", "numba"), ("XDG_CACHE_HOME", "xdg")):
        folder = out / "cache" / leaf
        folder.mkdir(parents=True)
        env[key] = str(folder)

    def command(python, tests):
        # absolute(), not resolve(): preserve a virtualenv interpreter symlink.
        cmd = [str(python.absolute()), "-B", "-m", "unittest", "discover", "-s", str(tests), "-v"]
        result = subprocess.run(cmd, cwd=out, env=env, text=True, capture_output=True)
        report["commands"].append({"command": cmd, "returncode": result.returncode,
                                    "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError("Unit tests failed")
        print("PASS tests", tests, flush=True)

    try:
        command(args.provider_python, HERE / "tests")
        command(args.maxfuse_python, MAXFUSE / "tests")
        old = support.check_code()
        support.verify_sources_unchanged(args.raw_root, args.benchmark_root)
        actual = support.verify_reference(args.rebuilt)
        changed = sorted(k for k in old.keys() | actual.keys() if old.get(k) != actual.get(k))
        if changed != ["recipe_sha256", "source_records"]:
            raise ValueError("Unexpected manifest differences")
        request = json.loads((args.rebuilt / "REQUEST.json").read_text())
        complete = json.loads((args.rebuilt / "COMPLETE.json").read_text())
        for name, field in (("build_inputs.py", "builder_sha256"), ("support.py", "support_sha256"),
                            ("source/historical_manifest.json", "historical_manifest_sha256")):
            if sha(HERE / name) != request[field]:
                raise ValueError("Provider run code differs from current code")
        if request["benchmark_manifest_sha256"] != sha(args.benchmark_root / "manifest.json"):
            raise ValueError("BM row-lock manifest changed")
        if complete["status"] != "complete" or complete["manifest_sha256"] != sha(args.rebuilt / "manifest.json"):
            raise ValueError("Incomplete provider run")
        reference = args.source_root / "benchmark_results/bone_marrow/maxfuse_v2"
        snapshot_files = {"source/build_inputs.py": "build_inputs.py", "config.json": "config.json",
                          "pairwise_config.json": "pairwise_config.json", "source/historical_manifest.json": "processed/manifest.json"}
        for copy, original in snapshot_files.items():
            if sha(HERE / copy) != sha(reference / original):
                raise ValueError("Historical snapshot differs from original")
        for name, digest in actual["files_sha256"].items():
            if sha(reference / "processed" / name) != digest:
                raise ValueError("Historical processed file changed")
        report["provider_run"] = {"root": str(args.rebuilt.resolve()), "cells": 88031,
            "provider_hashes": support.provider_records(), "row_lock_hashes": support.locked_records(),
            "byte_identical_files": actual["files_sha256"], "manifest_differences": changed,
            "software_versions": actual["software_versions"], "dataset_records_equal": True,
            "request_sha256": sha(args.rebuilt / "REQUEST.json"), "completion_sha256": sha(args.rebuilt / "COMPLETE.json")}
        print("PASS provider sources, row locks, 10 outputs and manifests", flush=True)

        prior_report = MAXFUSE / "validation/study_handoff_20261010_r2/report.json"
        prior = json.loads(prior_report.read_text())
        report["existing_bundles"] = []
        for record in prior["real_bundles"]:
            bundle = MAXFUSE / "runs/study_inputs_20261010_r2" / record["case"]
            if sha(bundle / "MANIFEST.json") != record["bundle_manifest_sha256"]:
                raise ValueError("Old bundle manifest changed")
            m = verify_bundle(bundle)
            report["existing_bundles"].append({"case": record["case"], "files": len(m["outputs"]),
                "manifest_sha256": record["bundle_manifest_sha256"]})
        bm = MAXFUSE / "runs/study_inputs_20261010_r2/bone_marrow"
        old_bundle, new_bundle = verify_bundle(bm), verify_bundle(args.bundle)
        if set(old_bundle["outputs"]) != set(new_bundle["outputs"]):
            raise ValueError("BM bundle file inventory changed")
        # Same cyclic view order: every dtype/value, ID and marker string in 45 arrays.
        for branch in new_bundle["branches"]:
            relative = Path("branches") / branch["name"] / "fit_input.npz"
            npz_equal(args.bundle / relative, bm / relative)
        for name, record in old_bundle["outputs"].items():
            if sha(args.bundle / name) != record["sha256"]:
                raise ValueError(f"BM bundle payload not byte-identical: {name}")
        origins = {k: v["read_from_provider_rebuild"] for k, v in new_bundle["source_files"].items()
                   if "read_from_provider_rebuild" in v}
        routing = new_bundle["native_input_override"]
        if len(origins) != 6 or set(origins) != set(routing["file_mapping"]):
            raise ValueError("Did not consume exactly six regenerated CSVs")
        for name, path in origins.items():
            expected_path = within(args.rebuilt, routing["file_mapping"][name])
            if Path(path).resolve(strict=True) != expected_path:
                raise ValueError("Override did not read this provider run")
            matches(expected_path, new_bundle["source_files"][name])
        if routing["manifest_sha256"] != sha(args.rebuilt / "manifest.json") or routing["completion_sha256"] != sha(args.rebuilt / "COMPLETE.json"):
            raise ValueError("Bridge provider-manifest hash differs")
        report["bridge"] = {"bundle": str(args.bundle.resolve()), "manifest_sha256": sha(args.bundle / "MANIFEST.json"),
            "cells": new_bundle["cells"], "branches": [b["name"] for b in new_bundle["branches"]],
            "fit_arrays_equal": 45, "all_payload_files_byte_identical": len(new_bundle["outputs"]),
            "read_from_provider_rebuild": origins, "actual_fit_performed": False}
        print("PASS three-hub bridge and unchanged seven bundles", flush=True)

        # Original code/profile/result files and field links remain untouched.
        report["historical_source_checks"] = verify_sources(MAXFUSE / "frozen_20261010", args.source_root)
        study, workspace = MAXFUSE.parents[1], args.source_root.resolve().parent
        report["unchanged_baselines"] = []
        for snapshot, base in ((study / "inventory_20261010/evidence/source_census.json", workspace),
                               (study / "inventory_20261010/evidence/curated_sources.json", workspace),
                               (study / "benchmarks/bone_marrow/validation/code_snapshot_output_safety_20261010.json", study)):
            count = 0
            for record in json.loads(snapshot.read_text())["files"]:
                if record.get("sha256"):
                    matches(within(base, record["path"]), record)
                    count += 1
            report["unchanged_baselines"].append({"snapshot": str(snapshot), "sha256": sha(snapshot), "checked": count})
        allowed = {"README.md", "STUDY_WORKFLOW.md", "study.py", "study_inputs.py", "tests/test_study.py"}
        for name, expected in prior["module_sha256"].items():
            if name not in allowed and sha(MAXFUSE / name) != expected:
                raise ValueError("Unexpected preexisting module change: " + name)
        # The previous fixture fix, unlike this round's optional routing, must stay untouched.
        fixture_report = MAXFUSE / "validation/tmp_path_fix_20261010/report.json"
        for name, expected in json.loads(fixture_report.read_text())["current_files"].items():
            if sha(MAXFUSE / name) != expected:
                raise ValueError("Previously audited fixture fix changed")
        report["previous_report_sha256"] = {str(p.relative_to(MAXFUSE)): sha(p) for p in (prior_report, fixture_report)}
        files = [p for p in HERE.rglob("*") if p.is_file() and not ({"runs", "validation", "__pycache__"} & set(p.relative_to(HERE).parts))]
        files += [MAXFUSE / n for n in ("study.py", "study_inputs.py", "tests/test_provider_bridge.py", "README.md", "STUDY_WORKFLOW.md")]
        files += [study / "README.md"]
        report["current_code_sha256"] = {str(p.relative_to(study)): sha(p) for p in sorted(files)}
        report["status"] = "passed"
        create_json(out / "report.json", report)
    except BaseException as error:
        create_json(out / "FAILED.json", {**report, "error": repr(error)})
        raise
    print(out / "report.json", flush=True)


if __name__ == "__main__":
    main()
