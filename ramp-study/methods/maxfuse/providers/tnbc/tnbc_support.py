"""Create-only storage/provenance for the TNBC provider port; no numeric changes."""

import argparse
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
MAXFUSE = HERE.parents[1]
sys.path.insert(0, str(MAXFUSE))
from migrate import create_json, sha
from run import reserve
from verify import within, verify_local

PREFIX = "benchmark_results/tnbc/maxfuse_archive/"
SNAPSHOTS = {"source/build_inputs.py": "build_inputs.py",
             "source/historical_manifest.json": "input/input_manifest.json",
             "extract_provider_rows.R": "extract_provider_rows.R",
             "config.json": "config.json", "pairwise_config.json": "pairwise_config.json"}


def history():
    return json.loads((HERE / "source/historical_manifest.json").read_text())


def check_code():
    snapshot = MAXFUSE / "frozen_20261010"
    verify_local(snapshot)
    catalog = json.loads((snapshot / "catalog.json").read_text())
    expected = {r["source"]: r["sha256"] for r in catalog["files"]}
    # These two upstream files were outside the earlier seven-case catalog.
    # Add explicit immutable fingerprints; do not mutate that accepted catalog.
    expected.update({
        PREFIX + "input/input_manifest.json": "922bf93d38f6cd5c4e4732e7e4cde24806e5a54b61e346bcec7005579f707048",
        PREFIX + "extract_provider_rows.R": "69cf3983b731bb418dfca700a77f4eed5f0d98e4d9743882c06985c2164af851",
    })
    for local, original in SNAPSHOTS.items():
        if sha(HERE / local) != expected[PREFIX + original]:
            raise ValueError("Frozen TNBC source/profile changed: " + local)
    return history()


def parse_inputs():
    p = argparse.ArgumentParser(description="Rebuild frozen TNBC native panels from providers; no resampling or fitting")
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--cell-lock", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def claim_output(output, raw_root, cell_lock):
    h = check_code()
    if sha(cell_lock) != h["source_files"]["canonical_cell_lock"]["sha256"]:
        raise ValueError("Canonical TNBC row lock changed")
    root = reserve(output, (raw_root, cell_lock, HERE / "source", HERE / "extract_provider_rows.R"))
    create_json(root / "REQUEST.json", {"case": "tnbc", "raw_root": str(raw_root.resolve()),
        "cell_lock": str(cell_lock.resolve()), "cell_lock_sha256": sha(cell_lock),
        "builder_sha256": sha(HERE / "build_inputs.py"), "support_sha256": sha(Path(__file__)),
        "source_sha256": {p: sha(HERE / p) for p in SNAPSHOTS},
        "Rscript": shutil.which("Rscript"), "cell_resampling": False})
    return root


def verify_sources_unchanged(records):
    expected = check_code()["source_files"]
    if set(records) != set(expected):
        raise ValueError("TNBC provider/row-lock source inventory changed")
    for name, row in records.items():
        if row["sha256"] != expected[name]["sha256"] or sha(Path(row["path"])) != expected[name]["sha256"]:
            raise ValueError("Provider/row-lock/recipe changed: " + name)


def verify_reference(root):
    h = history()
    manifest = json.loads(within(root, "input_manifest.json").read_text())
    if manifest.get("output_sha256") != h["output_sha256"]:
        raise ValueError("TNBC native output inventory or hashes changed")
    for name, digest in h["output_sha256"].items():
        if sha(within(root, name)) != digest:
            raise ValueError("Regenerated TNBC file differs: " + name)
    # Only locations/code hashes and truthful staging-retention prose can change.
    def scientific(m):
        m = json.loads(json.dumps(m))
        for key in ("builder", "builder_sha256", "writes_confined_to", "source_files"):
            m.pop(key, None)
        m["input_protocol"].pop("provider_extraction")
        return m
    if scientific(manifest) != scientific(h):
        raise ValueError("TNBC scientific manifest/values/panels/environment changed")
    verify_sources_unchanged(manifest["source_files"])
    return manifest


def publish_new(stage, root):
    stage, root = stage.resolve(strict=True), root.resolve(strict=True)
    if not stage.is_relative_to(root / "_work"):
        raise ValueError("Only owned staging may be published")
    request = json.loads((root / "REQUEST.json").read_text())
    for name, key in (("build_inputs.py", "builder_sha256"), ("tnbc_support.py", "support_sha256")):
        if sha(HERE / name) != request[key]:
            raise ValueError("Builder changed during extraction")
    for name, digest in request["source_sha256"].items():
        if sha(HERE / name) != digest:
            raise ValueError("Frozen source changed during extraction")
    manifest = verify_reference(stage)
    create_json(root / "PUBLISH_STARTED.json", {"manifest_sha256": sha(stage / "input_manifest.json")})
    for relative in [*sorted(manifest["output_sha256"]), "input_manifest.json"]:
        src, dst = within(stage, relative), root / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as source, dst.open("xb") as target:
            shutil.copyfileobj(source, target)
    verify_reference(root)
    create_json(root / "COMPLETE.json", {"case": "tnbc", "status": "complete", "cells": 62701,
        "manifest_sha256": sha(root / "input_manifest.json"), "byte_identical_outputs": manifest["output_sha256"],
        "provider_files": 3, "row_lock_files": 1, "new_maxfuse_fit_performed": False})


def fail(root, error):
    create_json(root / "FAILED.json", {"error": repr(error), "partial_output_retained": True})
