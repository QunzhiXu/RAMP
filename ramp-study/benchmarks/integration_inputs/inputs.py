"""Verify/stage frozen integration cohorts; never fit, transform or resample.

All outputs are create-only. Source files are read, not imported or executed.
The registry is the trust anchor, not a certificate of raw-data reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
FILENAMES = {
    "raw": "raw.npy",
    "fit_metadata": "fit_metadata.csv.gz",
    "evaluation_metadata": "evaluation_metadata.csv.gz",
    "markers": "markers.txt",
    "source_manifest": "source_manifest.json",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def child(root, relative):
    """Reject absolute/traversing paths, including symlink escapes."""
    part = Path(relative)
    if part.is_absolute() or not part.parts or ".." in part.parts:
        raise ValueError(f"Not a relative child path: {relative}")
    root = Path(root).resolve()
    result = (root / part).resolve()
    if result == root or root not in result.parents:
        raise ValueError(f"Path escapes root: {relative}")
    return result


def record(root, relative):
    path = child(root, relative)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}


def check_record(path, expected):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
        raise ValueError(f"Frozen file changed: {path}")


def reserve_directory(output, protected=()):
    """Atomically claim a new output; preserve partial output after failure.

    Scope: cooperating tasks, not hostile replacement of directory ancestors.
    This uses the accepted BM mkdir(exist_ok=False) ownership rule.
    """
    path = Path(output).expanduser()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Output exists; choose a new directory: {path}")
    path = path.resolve()
    for item in protected:
        source = Path(item).resolve(strict=True)
        if path == source or path in source.parents or source in path.parents:
            raise ValueError(f"Output overlaps protected input: {source}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path


def write_new_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def copy_checked(source, target, expected):
    """Exclusive byte-copy, not a hard link; fail on changed source bytes."""
    digest = hashlib.sha256()
    length = 0
    with Path(source).open("rb") as src, Path(target).open("xb") as dst:
        for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
            dst.write(chunk)
            digest.update(chunk)
            length += len(chunk)
    if length != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"Source changed during copy; partial output retained: {source}")


def input_paths(spec):
    base = Path(spec["input_directory"])
    result = {role: str(base / name) for role, name in FILENAMES.items()}
    result["evaluation_metadata"] = str(base / spec["evaluation_filename"])
    result["source_manifest"] = spec["source_manifest"]
    return result


def inspect_cohort(paths, spec):
    """Check row identities without changing ID spelling or metadata types.

    Only dataset/unit_key (plus bookkeeping cell_id where historically present)
    are permitted in fitting metadata. Evaluation labels are never fit inputs.
    """
    raw = np.load(paths["raw"], mmap_mode="r", allow_pickle=False)
    if raw.ndim != 2 or list(raw.shape) != spec["expected_shape"]:
        raise ValueError(f"Wrong raw matrix shape: {raw.shape}")
    if raw.dtype.kind not in "fiu":
        raise ValueError("Raw matrix must be a real numeric matrix")
    for start in range(0, len(raw), 10000):
        if not np.isfinite(raw[start : start + 10000]).all():
            raise ValueError("Non-finite raw matrix")
    markers = Path(paths["markers"]).read_text(encoding="utf-8").splitlines()
    if len(markers) != raw.shape[1] or len(set(markers)) != len(markers):
        raise ValueError("Marker count or uniqueness mismatch")
    if any(not value or value.strip() != value for value in markers):
        raise ValueError("Blank/padded marker name")
    cells, units, label_counts, joint_flags = {}, {}, Counter(), Counter()
    seen_ids = set()
    ordered_ids = hashlib.sha256()
    ordered_rows = hashlib.sha256()
    n = 0
    with gzip.open(paths["fit_metadata"], "rt", encoding="utf-8", newline="") as fit_stream, gzip.open(
        paths["evaluation_metadata"], "rt", encoding="utf-8", newline=""
    ) as eval_stream:
        fit = csv.DictReader(fit_stream)
        evaluation = csv.DictReader(eval_stream)
        if fit.fieldnames != spec["fit_columns"] or set(fit.fieldnames) - {"cell_id", "dataset", "unit_key"}:
            raise ValueError("Fitting metadata columns changed / labels leaked")
        required = {"cell_id", "dataset", "unit_key", spec["label_column"]}
        if not required.issubset(evaluation.fieldnames or []):
            raise ValueError("Missing evaluation metadata columns")
        if len(set(evaluation.fieldnames)) != len(evaluation.fieldnames):
            raise ValueError("Duplicate evaluation metadata columns")
        for f, e in itertools.zip_longest(fit, evaluation):
            if f is None or e is None or None in f or None in e or None in f.values() or None in e.values():
                raise ValueError("Malformed metadata or different row counts")
            for key in ("dataset", "unit_key"):
                if not e[key] or e[key].strip() != e[key] or e[key] != f[key]:
                    raise ValueError(f"Fit/evaluation {key} row order mismatch")
            cell_id = e["cell_id"]
            if not cell_id or cell_id in seen_ids or cell_id.strip() != cell_id:
                raise ValueError("Blank, padded or duplicate cell_id")
            if "cell_id" in f and f["cell_id"] != cell_id:
                raise ValueError("Fit/evaluation cell_id row order mismatch")
            seen_ids.add(cell_id)
            ordered_ids.update((json.dumps(cell_id, ensure_ascii=False) + "\n").encode())
            ordered_rows.update((json.dumps([cell_id, e["dataset"], e["unit_key"]], ensure_ascii=False) + "\n").encode())
            cells[e["dataset"]] = cells.get(e["dataset"], 0) + 1
            units.setdefault(e["dataset"], Counter())[e["unit_key"]] += 1
            label_counts[e[spec["label_column"]]] += 1
            flag = e.get("joint_evaluation", "absent")
            if flag not in {"True", "False", "absent"}:
                raise ValueError(f"Unexpected joint_evaluation value: {flag}")
            joint_flags[flag] += 1
            n += 1
    if n != raw.shape[0] or sum(len(v) for v in units.values()) != spec["expected_units"]:
        raise ValueError("Metadata cell/unit counts differ from specification")
    return {
        "shape": list(raw.shape), "dtype": str(raw.dtype), "markers": markers,
        "fit_columns": fit.fieldnames, "evaluation_columns": evaluation.fieldnames,
        "label_column": spec["label_column"], "dataset_order": list(cells),
        "dataset_cells": cells, "dataset_units": {k: len(v) for k, v in units.items()},
        "unit_cells": {k: dict(sorted(v.items())) for k, v in units.items()},
        "label_counts": dict(sorted(label_counts.items())), "joint_evaluation_counts": dict(joint_flags),
        "ordered_cell_ids_sha256": ordered_ids.hexdigest(),
        "ordered_cell_dataset_unit_sha256": ordered_rows.hexdigest(),
        "fit_cell_ids_present": "cell_id" in fit.fieldnames,
    }


def selected_cases(registry, case):
    cases = registry["cases"]
    names = [c["case"] for c in cases]
    if len(set(names)) != len(names) or any(Path(n).name != n or n in {"", ".", "..", "all"} for n in names):
        raise ValueError("Unsafe/duplicate case identifiers")
    selected = cases if case == "all" else [c for c in cases if c["case"] == case]
    if not selected:
        raise ValueError(f"Unknown case: {case}")
    return selected


def verify_source(registry, root, case="all"):
    cases = selected_cases(registry, case)
    for ref in registry["global_references"]:
        check_record(child(root, ref["path"]), ref)
    for item in cases:
        paths = {}
        for role in FILENAMES:
            info = item["files"][role]
            paths[role] = child(root, info["path"])
            check_record(paths[role], info)
        if inspect_cohort(paths, item) != item["observed"]:
            raise ValueError(f"Cohort structure changed: {item['case']}")
        for ref in item["references"]:
            check_record(child(root, ref["path"]), ref)
    return {"status": "passed", "cases": [c["case"] for c in cases],
            "cells": sum(c["observed"]["shape"][0] for c in cases),
            "input_files_checked": len(cases) * len(FILENAMES),
            "scope": "Frozen input bytes, structure and listed source references; no analysis run"}


def stage(registry_path, source_root, output, case="all"):
    registry_path = Path(registry_path).resolve(strict=True)
    registry_hash = sha256(registry_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    cases = selected_cases(registry, case)
    verify_source(registry, source_root, case)
    dest = reserve_directory(output, (source_root, registry_path))
    copy_checked(registry_path, dest / "registry.json", {"bytes": registry_path.stat().st_size, "sha256": registry_hash})
    for item in cases:
        folder = dest / item["case"]
        folder.mkdir(exist_ok=False)
        for role, name in FILENAMES.items():
            info = item["files"][role]
            copy_checked(child(source_root, info["path"]), folder / name, info)
        if inspect_cohort({k: folder / v for k, v in FILENAMES.items()}, item) != item["observed"]:
            raise ValueError(f"Staged cohort differs: {item['case']}")
    # Recheck after copying, including source code/configuration references.
    verify_source(registry, source_root, case)
    check_record(registry_path, {"bytes": (dest / 'registry.json').stat().st_size, "sha256": registry_hash})
    write_new_json(dest / "COMPLETE.json", {
        "schema_version": 1, "cases": [c["case"] for c in cases],
        "registry_sha256": registry_hash, "mode": "frozen_input_byte_copy",
        "scientific_transforms": [], "resampling_performed": False,
        "method_fitting_performed": False, "provider_raw_data_copied": False,
    })
    return dest


def verify_staged(root, expected_registry_path):
    """Read-only verification; uses a trusted registry, never original sources."""
    root = Path(root).resolve(strict=True)
    trusted = Path(expected_registry_path).resolve(strict=True)
    expected_hash = sha256(trusted)
    check_record(child(root, "registry.json"), {"bytes": trusted.stat().st_size, "sha256": expected_hash})
    registry = json.loads(trusted.read_text(encoding="utf-8"))
    complete = json.loads(child(root, "COMPLETE.json").read_text(encoding="utf-8"))
    if complete.get("registry_sha256") != expected_hash or complete.get("mode") != "frozen_input_byte_copy":
        raise ValueError("Invalid completion manifest")
    names = complete.get("cases", [])
    if not names or len(names) != len(set(names)):
        raise ValueError("Invalid completed case list")
    cases = [selected_cases(registry, n)[0] for n in names if n != "all"]
    if len(cases) != len(names):
        raise ValueError("Invalid completed case name")
    expected_entries = {"registry.json", "COMPLETE.json", *names}
    if {p.name for p in root.iterdir()} != expected_entries:
        raise ValueError("Unexpected/missing entries in staged directory")
    for item in cases:
        folder = child(root, item["case"])
        if {p.name for p in folder.iterdir()} != set(FILENAMES.values()):
            raise ValueError("Unexpected/missing files in staged cohort")
        paths = {role: child(folder, filename) for role, filename in FILENAMES.items()}
        for role, path in paths.items():
            check_record(path, item["files"][role])
        if inspect_cohort(paths, item) != item["observed"]:
            raise ValueError(f"Staged structure changed: {item['case']}")
    return {"status": "passed", "cases": names, "input_files_checked": len(cases) * len(FILENAMES),
            "cells": sum(c["observed"]["shape"][0] for c in cases),
            "requires_original_source_tree": False, "analysis_rerun": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=HERE / "registry_20261010.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List frozen cohort sizes; no writes")
    verify = sub.add_parser("verify-source", help="Read-only original input/config checks")
    verify.add_argument("--source-root", type=Path, required=True)
    verify.add_argument("--case", default="all")
    copy = sub.add_parser("stage", help="Byte-copy into an exclusively new output directory")
    copy.add_argument("--source-root", type=Path, required=True)
    copy.add_argument("--output", type=Path, required=True)
    copy.add_argument("--case", default="all")
    check = sub.add_parser("verify-staged", help="Verify copied inputs independently of source tree")
    check.add_argument("--input-root", type=Path, required=True)
    args = parser.parse_args(argv)
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if args.command == "list":
        result = [{"case": c["case"], "shape": c["observed"]["shape"],
                   "units": sum(c["observed"]["dataset_units"].values()),
                   "datasets": c["observed"]["dataset_cells"]} for c in registry["cases"]]
    elif args.command == "verify-source":
        result = verify_source(registry, args.source_root, args.case)
    elif args.command == "stage":
        dest = stage(args.registry, args.source_root, args.output, args.case)
        result = {"output": str(dest), **verify_staged(dest, args.registry)}
    else:
        result = verify_staged(args.input_root, args.registry)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
