#!/usr/bin/env python3
"""Validate real frozen inputs, label isolation, negative contracts, and outputs."""
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import benchmark as b

output = b.configure(b.DEFAULT_OUTPUT)
import numpy as np
import pandas as pd

design = json.loads((HERE / "config.json").read_text())
original_read = pd.read_csv
reads = []


def guarded_read(path, *args, **kwargs):
    p = Path(path).resolve()
    if "locked_metadata" in p.name or "relabel" in p.parts:
        raise AssertionError("Fitting adapter opened post-hoc metadata")
    reads.append(str(p))
    return original_read(path, *args, **kwargs)


pd.read_csv = guarded_read
checks = []
for case, cfg in design["cases"].items():
    source, markers, records = b.load_source(case, cfg)
    transformed, audit = b.transformed(source, markers, cfg)
    align = output / case / "alignment"
    b.verify_complete(align)
    corrected = pd.read_csv(align / "corrected_expression.csv.gz", dtype={"cell_id": str}, float_precision="round_trip")
    assert corrected.cell_id.tolist() == source.cell_id.tolist()
    assert list(corrected) == ["cell_id", *markers]
    assert np.isfinite(corrected[markers].to_numpy()).all()
    assert not any(row["adjacent_rank_inversions"] for row in audit)
    integrity = json.loads((align / "integrity.json").read_text())
    assert integrity["spline_replay_max_absolute_error"] == 0
    assert integrity["source_labels_used"] is False
    for r in records:
        assert b.sha(r["path"]) == r["sha256"]
    checks.append({"case": case, "cells": len(source), "markers": len(markers),
                   "units": source.unit_key.nunique(), "official_spline_replay_exact": True,
                   "row_identity_exact": True, "nonfinite": 0, "sources_unchanged": True})
pd.read_csv = original_read

# Hash corruption must fail before fitting and before publishing any output.
testdir = output / "contract_checks"
testdir.mkdir(exist_ok=True)
cfg = json.loads((output / "bone_marrow/run_config.json").read_text())
cfg["input_sha256"] = "0" * 64
b.write_json(testdir / "corrupt_config.json", cfg)
destination = testdir / "must_not_exist"
import os
link = Path("/private/tmp" if Path("/private/tmp").exists() else "/tmp") / f"cytonorm-test-{os.getpid()}"
link.symlink_to(output / "tmp", target_is_directory=True)
try:
    completed = subprocess.run(["Rscript", "--vanilla", str(HERE / "run_cytonorm.R"),
        "--input", str(output / "bone_marrow/prepared/analysis_ready.csv.gz"),
        "--markers", str(output / "bone_marrow/prepared/markers.txt"),
        "--config", str(testdir / "corrupt_config.json"), "--output", str(destination)],
        env=dict(os.environ, TMPDIR=str(link)), capture_output=True, text=True)
finally:
    link.unlink()
assert completed.returncode != 0 and "Prepared input hash mismatch" in completed.stderr
assert not destination.exists() and not list(testdir.glob("must_not_exist.staging-*"))
b.write_json(testdir / "VERIFICATION.json", {"cases": checks, "fitting_labels_read": False,
    "corrupt_hash_rejected_before_fit": True, "no_failed_output_published": True,
    "adapter_reads": sorted(set(reads))})
print(json.dumps(checks, indent=2))
