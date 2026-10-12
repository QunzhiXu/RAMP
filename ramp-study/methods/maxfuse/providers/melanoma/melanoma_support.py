"""Pinned inputs and create-only output for melanoma's two upstream stages."""

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
    "source/rebuild_source_active.py": "b211c50034e638223faad6e1a5b1743000c3bb24fe611e88c0deab3c3003d752",
    "source/build_inputs.py": "937a95dda55168b893c37002da38a5015dcd649e08d36547ba54498cb160ca03",
    "source/historical_source_manifest.json": "f0a12e207d7925bde5b566e8519e17d281f5a1d3bf6661f1f21d881dc7332f4a",
    "source/historical_method_manifest.json": "15168c1be7676ac6211da5338d257d5d4ddf25f5bd7dcb9dfd00906cf3efa91c",
    "config.json": "c7f93b62399d22cbbd7d69b28ec43cadc6c7d545d9fea950263c8b422929d14e",
    "pairwise_config.json": "507f2ab3df9fd8a34494e064714915f0867956993f9d165bd093c16cf0d02a61",
}
CODE = ("rebuild_source_active.py", "build_inputs.py", "melanoma_support.py")


def history(stage):
    if stage not in ("source", "method"):
        raise ValueError("Unknown provider stage")
    return json.loads((HERE / f"source/historical_{stage}_manifest.json").read_text())


def check_code():
    for name, digest in SOURCE_HASHES.items():
        if sha(HERE / name) != digest:
            raise ValueError("Frozen melanoma source/profile changed: " + name)


def input_records():
    h = history("source")
    canonical = h["canonical_inputs"]
    return ([d["source_file"] for d in h["datasets"]],
            [canonical[k] for k in ("selected_cells", "raw_shared", "shared_markers")]
            + list(canonical["inspectable_row_locks"].values()))


def check_inputs(raw_root, benchmark_root):
    check_code()
    providers, locks = input_records()
    groups = [(benchmark_root, "benchmark_data/melanoma/", locks)]
    if raw_root is not None:
        groups.append((raw_root, "raw_data/melanoma/", providers))
    for root, separator, records in groups:
        for record in records:
            relative = record["path"].split(separator, 1)[1]
            matches(within(root, relative), record)


def claim(output, stage, benchmark_root, raw_root=None, source_active=None):
    check_code()
    protected = [benchmark_root, HERE / "source"]
    protected += [p for p in (raw_root, source_active) if p is not None]
    root = reserve(output, protected)
    request = {"case": "melanoma", "stage": stage, "benchmark_root": str(benchmark_root.resolve()),
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
        raise ValueError("Melanoma output inventory/hashes differ")
    for name, digest in expected["outputs"].items():
        if sha(within(root, name)) != digest:
            raise ValueError("Rebuilt melanoma file differs: " + name)
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
        raise ValueError("Melanoma scientific manifest differs")
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
    create_json(output / "COMPLETE.json", {"status": "complete", "case": "melanoma", "stage": stage,
        "cells": 60000, "manifest_sha256": sha(output / "manifest.json"),
        "byte_identical_outputs": manifest["outputs"], "new_maxfuse_fit_performed": False})


def fail(root, error):
    create_json(root / "FAILED.json", {"error": repr(error), "partial_output_retained": True})
