#!/usr/bin/env python3
"""Generic command-line runner for the canonical RAMP implementation.

The input matrix must contain raw shared-marker measurements in the same row
order as the metadata table.  Metadata may contain any columns, but RAMP reads
only ``dataset`` and ``unit_key``.  Cell labels are neither required nor read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd

# Direct execution places this file's directory ahead of the package parent,
# which would otherwise resolve ``import ramp`` to the legacy sibling
# ``ramp.py`` module.  Insert the algorithm root first so the package public
# API in ``ramp/__init__.py`` is always used.
ALGORITHM_ROOT = Path(__file__).resolve().parents[1]
if str(ALGORITHM_ROOT) not in sys.path:
    sys.path.insert(0, str(ALGORITHM_ROOT))

from ramp import SEED, __version__, fit_transform
from ramp import ramp_ver2 as canonical_impl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_metadata(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run label-free rank-preserving RAMP on two or three datasets."
    )
    parser.add_argument("--raw", type=Path, required=True, help="Cells × markers .npy matrix")
    parser.add_argument("--metadata", type=Path, required=True, help="CSV/TSV/Parquet metadata")
    parser.add_argument("--markers", type=Path, required=True, help="One marker name per line")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", default="ramp_run", help="Stable run identifier")
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Explicit order of the two or three datasets; default is metadata order",
    )
    args = parser.parse_args()

    raw = np.load(args.raw)
    metadata = read_metadata(args.metadata)
    markers = [line.strip() for line in args.markers.read_text().splitlines() if line.strip()]
    observed = list(pd.unique(metadata["dataset"].astype(str)))
    datasets = observed if args.datasets is None else list(map(str, args.datasets))

    result = fit_transform(
        raw, metadata, markers, datasets=datasets, family=args.family
    )

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "kmeanslow_global.npy", result.baseline.astype(np.float32))
    np.save(args.output / "ramp_aligned.npy", result.aligned.astype(np.float32))
    result.coupling_audit.to_csv(args.output / "coupling_audit.csv", index=False)
    result.strength_audit.to_csv(args.output / "strength_audit.csv", index=False)
    result.evidence_audit.to_csv(args.output / "evidence_audit.csv", index=False)
    result.reference_weight_audit.to_csv(
        args.output / "reference_weight_audit.csv", index=False
    )
    result.profile_audit.to_csv(args.output / "profile_audit.csv", index=False)
    result.algorithm_audit.to_csv(args.output / "algorithm_audit.csv", index=False)
    manifest = {
        "algorithm": "RAMP",
        "algorithm_version": __version__,
        "algorithm_seed": SEED,
        "algorithm_source": {
            "file": str(Path(canonical_impl.__file__).resolve()),
            "sha256": sha256_file(Path(canonical_impl.__file__)),
        },
        "algorithm_helper_source": {
            "file": str(Path(__file__).with_name("ramp.py").resolve()),
            "sha256": sha256_file(Path(__file__).with_name("ramp.py")),
        },
        "runner_source": {
            "file": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "scipy": version("scipy"),
            "scikit_learn": version("scikit-learn"),
        },
        "family": args.family,
        "datasets": datasets,
        "dataset_count": len(datasets),
        "dataset_cells": metadata.dataset.astype(str).value_counts(sort=False).to_dict(),
        "cells": int(raw.shape[0]),
        "markers": markers,
        "marker_count": len(markers),
        "strict_rank_inversions": int(result.strict_rank_inversions),
        "metadata_columns_read_for_fitting": ["dataset", "unit_key"],
        "celltype_labels_read_for_fitting": False,
        "inputs": {
            "raw": str(args.raw.resolve()),
            "raw_sha256": sha256_file(args.raw),
            "metadata": str(args.metadata.resolve()),
            "metadata_sha256": sha256_file(args.metadata),
            "markers": str(args.markers.resolve()),
            "markers_sha256": sha256_file(args.markers),
        },
        "outputs": {
            "baseline": "kmeanslow_global.npy",
            "aligned": "ramp_aligned.npy",
            "coupling_audit": "coupling_audit.csv",
            "strength_audit": "strength_audit.csv",
            "evidence_audit": "evidence_audit.csv",
            "reference_weight_audit": "reference_weight_audit.csv",
            "profile_audit": "profile_audit.csv",
            "algorithm_audit": "algorithm_audit.csv",
        },
    }
    (args.output / "ramp_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"RAMP {__version__}: wrote {raw.shape[0]:,} cells × {raw.shape[1]} markers "
        f"for {len(datasets)} datasets; strict rank inversions = "
        f"{result.strict_rank_inversions}"
    )


if __name__ == "__main__":
    main()
