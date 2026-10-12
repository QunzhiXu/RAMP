"""Storage/provenance boundary for the faithful BM provider-input port."""

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
MAXFUSE = HERE.parents[1]
sys.path.insert(0, str(MAXFUSE))
from migrate import create_json, sha
from run import reserve
from verify import within


def history():
    return json.loads((HERE / "source/historical_manifest.json").read_text())


def check_code():
    h = history()
    for file, key in (("source/build_inputs.py", "recipe_sha256"),
                      ("config.json", "config_sha256"), ("pairwise_config.json", "pairwise_config_sha256")):
        if sha(HERE / file) != h[key]:
            raise ValueError(f"Historical code/profile snapshot changed: {file}")
    return h


def locked_records():
    return {row["path"].split("benchmark_data/bone_marrow/", 1)[1]: row["sha256"]
            for row in history()["source_records"] if row["role"].endswith("row_lock")}


def provider_records():
    return {row["role"].removeprefix("provider_source:"): row["sha256"]
            for row in history()["source_records"] if row["role"].startswith("provider_source:")}


def frozen_contract(benchmark_root):
    """Accept a rebuilt manifest, but only if pinned row/provider identities agree."""
    check_code()
    manifest = json.loads(within(benchmark_root, "manifest.json").read_text())
    if manifest.get("package") != "bone_marrow_three_platform_benchmark_v1" or manifest.get("n_total_cells") != 88031:
        raise ValueError("Wrong BM cohort")
    if manifest["source_files_sha256"] != provider_records():
        raise ValueError("Provider identity dictionary changed")
    for name, expected in locked_records().items():
        if manifest["files_sha256"].get(name) != expected or sha(within(benchmark_root, name)) != expected:
            raise ValueError(f"Frozen row lock changed: {name}")
    return manifest


def verify_sources_unchanged(raw_root, benchmark_root):
    frozen_contract(benchmark_root)
    for name, expected in provider_records().items():
        if sha(within(raw_root, name)) != expected:
            raise ValueError(f"Provider changed: {name}")


def claim_output(output, raw_root, benchmark_root):
    root = reserve(output, (raw_root, benchmark_root, HERE / "source", HERE / "config.json"))
    create_json(root / "REQUEST.json", {
        "raw_root": str(raw_root.resolve()), "benchmark_root": str(benchmark_root.resolve()),
        "benchmark_manifest_sha256": sha(benchmark_root / "manifest.json"),
        "builder_sha256": sha(HERE / "build_inputs.py"), "support_sha256": sha(Path(__file__)),
        "historical_manifest_sha256": sha(HERE / "source/historical_manifest.json"),
        "Rscript": shutil.which("Rscript"), "cell_resampling": False})
    return root


@contextmanager
def owned_work(output):
    """Retain staging, selected-cell export and diagnostics even on interruption."""
    work = output / "_work"
    work.mkdir(exist_ok=False)
    try:
        yield str(work)
    except BaseException as error:
        create_json(output / "FAILED.json", {"error": repr(error), "partial_output_retained": True})
        raise


def verify_reference(stage):
    expected = history()
    manifest = json.loads(within(stage, "manifest.json").read_text())
    for name, digest in expected["files_sha256"].items():
        if manifest["files_sha256"].get(name) != digest or sha(within(stage, name)) != digest:
            raise ValueError(f"Regenerated native input/audit differs: {name}")
    if manifest["files_sha256"] != expected["files_sha256"]:
        raise ValueError("Native output inventory changed")
    # Includes raw, pre-z and post-z buffers, order hashes and feature inventories.
    if manifest["datasets"] != expected["datasets"]:
        raise ValueError("Per-platform values, identities or panel descriptions changed")
    return manifest


def publish_new(stage, output):
    stage, output = stage.resolve(strict=True), output.resolve(strict=True)
    if not stage.is_relative_to(output / "_work"):
        raise ValueError("Only this run's owned staging tree may be published")
    request = json.loads((output / "REQUEST.json").read_text())
    if sha(HERE / "build_inputs.py") != request["builder_sha256"] or sha(Path(__file__)) != request["support_sha256"]:
        raise ValueError("Builder changed during extraction")
    if sha(Path(request["benchmark_root"]) / "manifest.json") != request["benchmark_manifest_sha256"]:
        raise ValueError("Benchmark manifest changed during extraction")
    manifest = verify_reference(stage)
    # Independent arbitration even if publish is accidentally invoked twice.
    create_json(output / "PUBLISH_STARTED.json", {"stage_manifest_sha256": sha(stage / "manifest.json")})
    for relative in [*sorted(manifest["files_sha256"]), "manifest.json"]:
        src, dst = within(stage, relative), output / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as source, dst.open("xb") as target:
            shutil.copyfileobj(source, target)
    verify_reference(output)
    create_json(output / "COMPLETE.json", {"status": "complete", "cells": 88031,
                "source_provider_files": 21, "row_lock_files": 4,
                "manifest_sha256": sha(output / "manifest.json"),
                "byte_identical_outputs": manifest["files_sha256"],
                "historical_dataset_records_equal": True,
                "new_maxfuse_fit_performed": False})
