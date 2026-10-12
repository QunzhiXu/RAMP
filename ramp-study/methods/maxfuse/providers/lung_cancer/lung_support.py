"""Pinned inputs and create-only output for lung_cancer's two upstream stages."""

import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
MAXFUSE = HERE.parents[1]
sys.path.insert(0, str(MAXFUSE))
from migrate import create_json, sha
from run import reserve
from verify import matches, within

SOURCE_HASHES = {
    "source/rebuild_source_active.py": "c2d3f24147e32d81c79522e80578f86d267a17f92cf10966cf07894e8214299d",
    "source/build_inputs.py": "502e72f52b95391eb89741138fea9a85006693e04ee6a139aebeb8cbc552c9e7",
    "source/historical_source_manifest.json": "5675e949e0b238b1d137712824fc22fc5124f56a96f1a1785c496b3ffeae064d",
    "source/historical_method_manifest.json": "727e496a86d69246eeda9994965fccf94d8443018c97df2270902490c775474c",
    "config.json": "dbc37599edfcb04ded0b3bfc86847e0299787bdf5ea718779b00a61686de27ec",
    "pairwise_config.json": "56a912a8ccf996a6adc8402efd2c5ffab66f4de576c8ef5bc1642bfd0d39ef32"
}
CODE = ("rebuild_source_active.py", "build_inputs.py", "lung_support.py")


def history(stage):
    if stage not in ("source", "method"):
        raise ValueError("Unknown provider stage")
    return json.loads((HERE / f"source/historical_{stage}_manifest.json").read_text())


def check_code():
    for name, digest in SOURCE_HASHES.items():
        if sha(HERE / name) != digest:
            raise ValueError("Frozen lung_cancer source/profile changed: " + name)


def input_records():
    h = history("source")
    canonical = h["canonical_inputs"]
    return ([record for d in h["datasets"] for record in d["source_files"].values()],
            [canonical[k] for k in ("selected_cells", "raw_shared", "shared_markers")]
            + list(canonical["inspectable_row_locks"].values()))


def check_inputs(raw_root, benchmark_root):
    check_code()
    providers, locks = input_records()
    groups = [(benchmark_root, "benchmark_data/lung_cancer/", locks)]
    if raw_root is not None:
        groups.append((raw_root, "raw_data/lung_cancer/", providers))
    for root, separator, records in groups:
        for record in records:
            relative = record["path"].split(separator, 1)[1]
            matches(within(root, relative), record)


def claim(output, stage, benchmark_root, raw_root=None, source_active=None):
    check_code()
    protected = [benchmark_root, HERE / "source"]
    protected += [p for p in (raw_root, source_active) if p is not None]
    root = reserve(output, protected)
    request = {"case": "lung_cancer", "stage": stage, "benchmark_root": str(benchmark_root.resolve()),
        "raw_root": str(raw_root.resolve()) if raw_root else None,
        "source_active": str(source_active.resolve()) if source_active else None,
        "code_sha256": {p: sha(HERE / p) for p in CODE}, "source_sha256": SOURCE_HASHES,
        "cell_resampling": False}
    if source_active:
        request["source_manifest_sha256"] = sha(source_active / "manifest.json")
        request["source_completion_sha256"] = sha(source_active / "COMPLETE.json")
    create_json(root / "REQUEST.json", request)
    return root


def verify_reference(root, stage):
    expected = history(stage)
    actual = json.loads(within(root, "manifest.json").read_text())
    if actual.get("outputs") != expected["outputs"]:
        raise ValueError("Lung output inventory/hashes differ")
    for name, digest in expected["outputs"].items():
        if sha(within(root, name)) != digest:
            raise ValueError("Rebuilt lung_cancer file differs: " + name)
    def numeric(m):
        m = json.loads(json.dumps(m))
        for key in ("recipe", "recipe_sha256", "source_active_manifest", "source_active_manifest_sha256",
                    "source_active_recipe", "source_active_recipe_sha256"):
            m.pop(key, None)
        def strip_paths(value):
            if isinstance(value, dict):
                if "sha256" in value and "path" in value:
                    value.pop("path")
                for child in value.values(): strip_paths(child)
            elif isinstance(value, list):
                for child in value: strip_paths(child)
        strip_paths(m)
        return m
    if numeric(actual) != numeric(expected):
        raise ValueError("Lung scientific manifest differs")
    return actual


def publish_new(stage_dir, output, stage):
    stage_dir, output = stage_dir.resolve(strict=True), output.resolve(strict=True)
    if stage_dir != output / "_work":
        raise ValueError("Publication requires this run's owned staging")
    request = json.loads((output / "REQUEST.json").read_text())
    if request["stage"] != stage:
        raise ValueError("Wrong requested stage")
    check_code()
    for name, digest in request["code_sha256"].items():
        if sha(HERE / name) != digest:
            raise ValueError("Builder changed during run")
    if request["source_active"]:
        src = Path(request["source_active"])
        if sha(src / "manifest.json") != request["source_manifest_sha256"] or sha(src / "COMPLETE.json") != request["source_completion_sha256"]:
            raise ValueError("Source-active provenance changed during scaling")
    manifest = verify_reference(stage_dir, stage)
    create_json(output / "PUBLISH_STARTED.json", {"manifest_sha256": sha(stage_dir / "manifest.json")})
    for name in [*sorted(manifest["outputs"]), "manifest.json"]:
        src, dst = within(stage_dir, name), output / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as source, dst.open("xb") as target:
            shutil.copyfileobj(source, target)
    verify_reference(output, stage)
    create_json(output / "COMPLETE.json", {"status": "complete", "case": "lung_cancer", "stage": stage,
        "cells": 60000, "manifest_sha256": sha(output / "manifest.json"),
        "byte_identical_outputs": manifest["outputs"], "new_maxfuse_fit_performed": False})


def fail(root, error):
    create_json(root / "FAILED.json", {"error": repr(error), "partial_output_retained": True})

