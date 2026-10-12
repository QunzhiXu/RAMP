"""Explicit local audit report; output is create-only under this module's validation/.

Runs synthetic tests and read-only checks, not research analyses. Unlike
inputs.py verify commands, this maintenance command writes a new evidence folder.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from inputs import HERE, check_record, child, record, reserve_directory, sha256, verify_source, verify_staged, write_new_json
from freeze_registry import check_historical_raw


def check_previous_snapshot(snapshot, root):
    data = json.loads(snapshot.read_text(encoding="utf-8"))
    checked = 0
    skipped = []
    for item in data["files"]:
        if item.get("sha256"):
            check_record(child(root, item["path"]), item)
            checked += 1
        else:
            skipped.append(item["path"])
    return {"snapshot_sha256": sha256(snapshot), "checked": checked,
            "not_hashed_in_original_snapshot": skipped, "all_checked_unchanged": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.output.resolve().is_relative_to(HERE / "validation"):
        parser.error("Evidence output must be a new subdirectory of this module's validation/")
    registry_path = HERE / "registry_20261010.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    project = args.project_root.resolve(strict=True)
    output = reserve_directory(args.output, (args.input_root, registry_path, project / "final_xu"))
    command = [sys.executable, "-B", "-m", "unittest", "discover", "-s", str(HERE / "tests"), "-v"]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    tests = subprocess.run(command, cwd=HERE, env=env, capture_output=True, text=True)
    report = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "numpy": np.__version__,
        "registry_sha256": sha256(registry_path),
        "scope": "Seven frozen input cohorts; no preprocessing, method fitting, metrics, UMAP or biological analysis run",
        "tests": {"command": command, "returncode": tests.returncode,
                  "stdout": tests.stdout, "stderr": tests.stderr},
    }
    if tests.returncode:
        write_new_json(output / "FAILED.json", report)
        raise RuntimeError("Tests failed; see FAILED.json")
    report["original_inputs_and_listed_references"] = verify_source(registry, project / "final_xu")
    report["staged_inputs"] = verify_staged(args.input_root, registry_path)
    for case in registry["cases"]:
        historical = json.loads((project / "final_xu" / case["source_manifest"]).read_text())
        check_historical_raw(historical, case["kind"], case["files"]["raw"]["sha256"])
    report["exact_historical_raw_fields_checked"] = len(registry["cases"])
    report["input_bytes"] = sum(f["bytes"] for c in registry["cases"] for f in c["files"].values())
    report["selected_reference_records"] = len(registry["global_references"]) + sum(len(c["references"]) for c in registry["cases"])
    units = {}
    for case in registry["cases"]:
        units[case["case"]] = {
            (dataset, key.rsplit("|", 1)[1] if dataset in {"Senosain CyTOF", "Cords IMC"} else key)
            for dataset, counts in case["observed"]["unit_cells"].items() for key in counts
        }
    report["unit_scope"] = {
        "task_specific_occurrences": sum(len(u) for u in units.values()),
        "distinct_dataset_specific_units": len(set.union(*units.values())),
        "lung_overlap": sorted(units["lung_cancer"] & units["lung_story"]),
        "identity_normalization": "Only for counting: for Senosain/Cords use suffix after |; input metadata not edited",
    }
    inventory = project / "ramp-study/inventory_20261010/evidence"
    report["previous_census"] = check_previous_snapshot(inventory / "source_census.json", project)
    report["previous_curated"] = check_previous_snapshot(inventory / "curated_sources.json", project)
    report["accepted_bm_and_vendor"] = check_previous_snapshot(
        project / "ramp-study/benchmarks/bone_marrow/validation/code_snapshot_output_safety_20261010.json",
        project / "ramp-study",
    )
    # Excludes generated runs/evidence; includes this validator and documentation.
    paths = [p.relative_to(HERE).as_posix() for p in HERE.iterdir() if p.is_file()]
    paths += [p.relative_to(HERE).as_posix() for p in (HERE / "tests").glob("*.py")]
    report["module_files"] = [record(HERE, p) for p in sorted(paths)]
    report["study_readme"] = record(project / "ramp-study", "README.md")
    report["status"] = "passed"
    write_new_json(output / "report.json", report)
    print(json.dumps({"status": "passed", "report": str(output / "report.json"),
                      "input_files": 35, "selected_references": report["selected_reference_records"]}))


if __name__ == "__main__":
    main()
