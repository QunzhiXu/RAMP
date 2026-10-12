"""File input, validated metadata handling, and reproducible RAMP outputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import ALGORITHM_VERSION, __version__
from . import _algorithm
from . import _helpers
from ._algorithm import RampResult
from ._validation import validate_identity_values


OUTPUT_FILES = {
    "baseline": "kmeanslow_global.npy",
    "aligned": "ramp_aligned.npy",
    "metadata": "cell_metadata.csv.gz",
    "markers": "markers.txt",
    "coupling_audit": "coupling_audit.csv",
    "strength_audit": "strength_audit.csv",
    "evidence_audit": "evidence_audit.csv",
    "reference_weight_audit": "reference_weight_audit.csv",
    "profile_audit": "profile_audit.csv",
    "algorithm_audit": "algorithm_audit.csv",
    "manifest": "ramp_manifest.json",
}

EMPTY_AUDIT_COLUMNS = {
    "strength_audit": [
        "query_dataset",
        "reference_dataset",
        "unit_key",
        "marker_index",
        "calibration_queries",
        "dataset_alpha",
        "donor_alpha",
        "adaptive_alpha",
        "dataset_gain",
        "donor_gain",
        "mapping_operator",
        "maximum_displacement",
    ],
    "evidence_audit": [
        "query_dataset",
        "reference_dataset",
        "unit_key",
        "marker_index",
        "dataset_z",
        "donor_z",
        "hierarchical_z",
        "reference_log_evidence",
        "dataset_mean_error_reduction",
        "dataset_unit_standard_error",
        "provider_units",
        "provider_evidence_unit_aggregated",
        "donor_queries",
        "donor_effective_queries",
        "donor_shrinkage",
    ],
    "reference_weight_audit": [
        "dataset",
        "unit_key",
        "marker_index",
        "identity_weight",
        "alignment_mass",
        "maximum_displacement",
        "final_mapping_operator",
    ],
    "profile_audit": [
        "query_dataset",
        "reference_dataset",
        "unit_key",
        "marker_index",
        "curve_cells",
        "curve_bins_requested",
        "curve_equal_count_bins",
        "curve_minimum_bin_cells",
        "curve_profile_fitted",
        "curve_bins_used",
        "curve_unique_bin_centres",
        "curve_smallest_bin",
        "curve_largest_bin",
    ],
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_metadata(path: str | Path) -> pd.DataFrame:
    """Read row-aligned cell metadata from CSV, TSV, or Parquet."""
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except ImportError as error:
            raise ImportError(
                "Parquet metadata requires the optional dependency: "
                "pip install 'ramp-proteomics[parquet]'"
            ) from error
    if suffixes.endswith(".tsv") or suffixes.endswith(".tsv.gz"):
        return pd.read_csv(
            path,
            sep="\t",
            low_memory=False,
            dtype="string",
            keep_default_na=False,
        )
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(
            path,
            low_memory=False,
            dtype="string",
            keep_default_na=False,
        )
    raise ValueError("metadata must be CSV, CSV.GZ, TSV, TSV.GZ, or Parquet")


def read_markers(path: str | Path) -> list[str]:
    """Read one non-empty marker name per line."""
    markers = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not markers:
        raise ValueError("marker file is empty")
    return markers


def validate_cell_metadata(metadata: pd.DataFrame, n_cells: int) -> pd.DataFrame:
    """Validate identity columns while preserving all user metadata.

    RAMP fits with only ``dataset`` and ``unit_key``. ``cell_id`` and
    ``donor_id`` are identity/provenance fields and are never supplied to the
    algorithm. If ``cell_id`` is absent, deterministic row IDs are generated.
    """
    if len(metadata) != int(n_cells):
        raise ValueError(
            f"metadata has {len(metadata):,} rows but expression has {n_cells:,} cells"
        )
    required = ["dataset", "unit_key"]
    missing = [column for column in required if column not in metadata.columns]
    if missing:
        raise ValueError(f"metadata is missing required columns: {', '.join(missing)}")
    answer = metadata.copy().reset_index(drop=True)
    for column in required:
        answer[column] = validate_identity_values(answer[column], column)
    if "cell_id" not in answer.columns:
        width = max(7, len(str(max(0, n_cells - 1))))
        answer.insert(
            0,
            "cell_id",
            [f"cell_{index:0{width}d}" for index in range(n_cells)],
        )
    if answer["cell_id"].isna().any():
        raise ValueError("cell_id must be non-missing and unique")
    cell_ids = answer["cell_id"].astype(str)
    if cell_ids.str.strip().eq("").any():
        raise ValueError("cell_id cannot contain blank values")
    if cell_ids.ne(cell_ids.str.strip()).any():
        raise ValueError("cell_id cannot contain leading or trailing whitespace")
    if cell_ids.duplicated().any():
        raise ValueError("cell_id must be non-missing and unique")
    answer["cell_id"] = cell_ids
    if "donor_id" in answer.columns:
        if answer["donor_id"].isna().any():
            raise ValueError("donor_id cannot contain missing values when provided")
        donor_ids = answer["donor_id"].astype(str)
        if donor_ids.str.strip().eq("").any():
            raise ValueError("donor_id cannot contain blank values when provided")
        if donor_ids.ne(donor_ids.str.strip()).any():
            raise ValueError(
                "donor_id cannot contain leading or trailing whitespace"
            )
        answer["donor_id"] = donor_ids

    unit_dataset_counts = answer.groupby("unit_key", sort=False)["dataset"].nunique()
    if int(unit_dataset_counts.max()) != 1:
        raise ValueError(
            "unit_key must be globally unique to one dataset; use values such as "
            "'CODEX::P01' and retain the cross-platform patient identifier in donor_id"
        )
    return answer


def _installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _source_record(module: Any) -> dict[str, str]:
    path = Path(module.__file__).resolve()
    return {
        "module": str(module.__name__),
        "file": path.name,
        "sha256": sha256_file(path),
    }


def capture_input_records(input_paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    """Capture input hashes without exposing local paths or filenames."""
    return {
        name: {"sha256": sha256_file(Path(path).resolve())}
        for name, path in input_paths.items()
    }


def capture_input_signatures(
    input_paths: dict[str, Path],
) -> dict[str, tuple[int, int, int, int]]:
    """Capture local file identity used to detect mutation during input loading."""
    signatures: dict[str, tuple[int, int, int, int]] = {}
    for name, path in input_paths.items():
        stat = Path(path).resolve().stat()
        signatures[name] = (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        )
    return signatures


def validate_input_signatures(
    input_paths: dict[str, Path],
    expected: dict[str, tuple[int, int, int, int]],
) -> None:
    """Fail if an input was replaced or modified while RAMP was reading it."""
    observed = capture_input_signatures(input_paths)
    changed = [name for name in input_paths if observed.get(name) != expected.get(name)]
    if changed:
        raise RuntimeError(
            "input file changed while it was being read: " + ", ".join(changed)
        )


def validate_input_output_paths(
    input_paths: dict[str, Path], output: str | Path
) -> None:
    """Reject direct, symbolic-link, and hard-link aliases of an input."""
    output_path = Path(output).resolve()
    planned_outputs = {
        (output_path / filename).resolve() for filename in OUTPUT_FILES.values()
    }
    for name, input_path in input_paths.items():
        resolved = Path(input_path).resolve()
        if resolved in planned_outputs:
            raise ValueError(
                f"input {name!r} resolves to a planned RAMP output path: {resolved.name}"
            )
        for planned in planned_outputs:
            if planned.exists() and resolved.samefile(planned):
                raise ValueError(
                    f"input {name!r} shares file identity with a planned RAMP "
                    f"output path: {planned.name}"
                )


def prepare_output_directory(output: str | Path, *, overwrite: bool = False) -> Path:
    """Create an output directory without deleting unrelated user files."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    validate_output_directory(output, overwrite=overwrite)
    return output


def _write_audit_table(frame: pd.DataFrame, path: Path, name: str) -> None:
    if len(frame.columns) == 0 and name in EMPTY_AUDIT_COLUMNS:
        frame = pd.DataFrame(columns=EMPTY_AUDIT_COLUMNS[name])
    frame.to_csv(path, index=False)


def validate_output_directory(
    output: str | Path, *, overwrite: bool = False
) -> None:
    """Fail before fitting when known outputs already exist."""
    output = Path(output)
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"RAMP output path is not a directory: {output}")
    existing = [
        output / name
        for name in OUTPUT_FILES.values()
        if (output / name).exists() or (output / name).is_symlink()
    ]
    for path in existing:
        if path.is_dir():
            raise IsADirectoryError(f"RAMP output file path is a directory: {path}")
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing[:4])
        raise FileExistsError(
            f"RAMP outputs already exist in {output}: {names}; pass --overwrite to replace them"
        )
    writable_parent = output if output.exists() else output.parent
    while not writable_parent.exists() and writable_parent != writable_parent.parent:
        writable_parent = writable_parent.parent
    if not writable_parent.is_dir():
        raise NotADirectoryError(
            f"RAMP output parent is not a directory: {writable_parent}"
        )
    if not os.access(writable_parent, os.W_OK):
        raise PermissionError(f"RAMP output location is not writable: {writable_parent}")


def write_result_directory(
    result: RampResult,
    metadata: pd.DataFrame,
    markers: list[str],
    output: str | Path,
    *,
    family: str,
    datasets: list[str],
    overwrite: bool = False,
    output_dtype: str = "float64",
    input_paths: dict[str, Path] | None = None,
    input_records: dict[str, dict[str, str]] | None = None,
    cell_id_generated: bool | None = None,
) -> dict[str, Any]:
    """Stage complete files, then publish without truncating existing inodes.

    Each overwrite is atomic at the file level. The output directory as a
    whole is not a transaction; its manifest is published last.
    """
    output_path = Path(output).resolve()
    recorded_inputs = {} if input_records is None else dict(input_records)
    if input_paths:
        validate_input_output_paths(input_paths, output_path)
        if not recorded_inputs:
            recorded_inputs = capture_input_records(input_paths)
    dtype = np.dtype(output_dtype)
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("output_dtype must be float32 or float64")
    output = prepare_output_directory(output_path, overwrite=overwrite)
    with tempfile.TemporaryDirectory(prefix=".ramp-stage-", dir=output) as temporary:
        staged = Path(temporary)
        manifest = _write_result_files(
            result,
            metadata,
            markers,
            staged,
            family=family,
            datasets=datasets,
            dtype=dtype,
            recorded_inputs=recorded_inputs,
            cell_id_generated=cell_id_generated,
        )
        # Repeat the checks after serialization, immediately before publishing.
        if input_paths:
            validate_input_output_paths(input_paths, output)
        validate_output_directory(output, overwrite=overwrite)
        filenames = [
            name for key, name in OUTPUT_FILES.items() if key != "manifest"
        ] + [OUTPUT_FILES["manifest"]]
        for filename in filenames:
            source, destination = staged / filename, output / filename
            if overwrite:
                os.replace(source, destination)
            else:
                # Linking a completed temporary file fails if the destination
                # appeared concurrently, and never follows a destination link.
                os.link(source, destination)
        return manifest


def _write_result_files(
    result: RampResult,
    metadata: pd.DataFrame,
    markers: list[str],
    output: Path,
    *,
    family: str,
    datasets: list[str],
    dtype: np.dtype,
    recorded_inputs: dict[str, dict[str, str]],
    cell_id_generated: bool | None,
) -> dict[str, Any]:
    """Serialize files into a new private directory before publication."""

    np.save(output / OUTPUT_FILES["baseline"], result.baseline.astype(dtype, copy=False))
    np.save(output / OUTPUT_FILES["aligned"], result.aligned.astype(dtype, copy=False))
    metadata.to_csv(
        output / OUTPUT_FILES["metadata"],
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    (output / OUTPUT_FILES["markers"]).write_text(
        "\n".join(map(str, markers)) + "\n", encoding="utf-8"
    )
    for audit_name in (
        "coupling_audit",
        "strength_audit",
        "evidence_audit",
        "reference_weight_audit",
        "profile_audit",
        "algorithm_audit",
    ):
        _write_audit_table(
            getattr(result, audit_name), output / OUTPUT_FILES[audit_name], audit_name
        )

    dataset_values = metadata["dataset"].astype(str)
    displacement = np.abs(result.aligned - result.baseline)
    corrected_values = int(np.count_nonzero(displacement > 1e-12))
    if (
        not result.profile_audit.empty
        and "curve_profile_fitted" in result.profile_audit.columns
    ):
        fitted_profile_rows = int(
            result.profile_audit["curve_profile_fitted"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
    else:
        fitted_profile_rows = 0
    donor_present = "donor_id" in metadata.columns
    if donor_present:
        donor_pairs = metadata[["dataset", "donor_id"]].drop_duplicates()
        donor_counts_by_dataset = {
            str(dataset): int(group["donor_id"].nunique())
            for dataset, group in donor_pairs.groupby("dataset", sort=False)
        }
    else:
        donor_pairs = None
        donor_counts_by_dataset = None
    manifest: dict[str, Any] = {
        "software": "ramp-proteomics",
        "software_version": __version__,
        "algorithm": "RAMP",
        "algorithm_version": ALGORITHM_VERSION,
        "algorithm_seed": int(_algorithm.SEED),
        "family": str(family),
        "datasets": list(map(str, datasets)),
        "dataset_count": len(datasets),
        "dataset_cells": dataset_values.value_counts(sort=False).to_dict(),
        "cells": int(result.aligned.shape[0]),
        "markers": list(map(str, markers)),
        "marker_count": len(markers),
        "output_dtype": dtype.name,
        "strict_rank_inversions": int(result.strict_rank_inversions),
        "profile_audit_rows": int(len(result.profile_audit)),
        "fitted_profile_rows": fitted_profile_rows,
        "corrected_cell_marker_values": corrected_values,
        "identity_only_alignment": corrected_values == 0,
        "metadata_columns_read_for_fitting": ["dataset", "unit_key"],
        "metadata_columns_preserved": list(map(str, metadata.columns)),
        "cell_id_generated": cell_id_generated,
        "celltype_labels_read_for_fitting": False,
        "donor_id_present": donor_present,
        "donor_count": (
            int(metadata["donor_id"].astype(str).nunique()) if donor_present else None
        ),
        "dataset_donor_pair_count": (
            int(len(donor_pairs)) if donor_pairs is not None else None
        ),
        "donor_counts_by_dataset": donor_counts_by_dataset,
        "unit_count": int(metadata["unit_key"].astype(str).nunique()),
        "source": {
            "algorithm": _source_record(_algorithm),
            "helpers": _source_record(_helpers),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": _installed_version("numpy"),
            "pandas": _installed_version("pandas"),
            "scipy": _installed_version("scipy"),
            "scikit_learn": _installed_version("scikit-learn"),
        },
        "outputs": OUTPUT_FILES,
    }
    manifest["output_sha256"] = {
        name: sha256_file(output / filename)
        for name, filename in OUTPUT_FILES.items()
        if name != "manifest"
    }
    if recorded_inputs:
        manifest["inputs"] = recorded_inputs
    (output / OUTPUT_FILES["manifest"]).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
