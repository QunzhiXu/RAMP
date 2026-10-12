"""Command-line interface for RAMP."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from . import __version__, fit_transform
from ._validation import validate_raw_matrix
from .io import (
    capture_input_records,
    capture_input_signatures,
    read_markers,
    read_metadata,
    validate_cell_metadata,
    validate_input_output_paths,
    validate_input_signatures,
    validate_output_directory,
    write_result_directory,
)


def _align(args: argparse.Namespace) -> int:
    input_paths = {
        "raw": args.raw,
        "metadata": args.metadata,
        "markers": args.markers,
    }
    validate_input_output_paths(input_paths, args.output)
    validate_output_directory(args.output, overwrite=args.overwrite)
    input_signatures = capture_input_signatures(input_paths)
    input_records = capture_input_records(input_paths)
    raw = validate_raw_matrix(np.load(args.raw, allow_pickle=False))
    input_metadata = read_metadata(args.metadata)
    cell_id_generated = "cell_id" not in input_metadata.columns
    metadata = validate_cell_metadata(input_metadata, len(raw))
    markers = read_markers(args.markers)
    validate_input_signatures(input_paths, input_signatures)
    datasets = list(map(str, args.datasets))
    result = fit_transform(
        raw,
        metadata,
        markers,
        datasets=datasets,
        family=args.family,
    )
    write_result_directory(
        result,
        metadata,
        markers,
        args.output,
        family=args.family,
        datasets=datasets,
        overwrite=args.overwrite,
        output_dtype=args.output_dtype,
        input_paths=input_paths,
        input_records=input_records,
        cell_id_generated=cell_id_generated,
    )
    print(
        f"RAMP {__version__}: wrote {raw.shape[0]:,} cells x {raw.shape[1]} markers "
        f"for {len(datasets)} datasets; corrected cell-marker values = "
        f"{np.count_nonzero(np.abs(result.aligned - result.baseline) > 1e-12):,}; "
        f"strict rank inversions = {result.strict_rank_inversions}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ramp-proteomics",
        description="Rank-preserving alignment of two or three protein datasets.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    align = subparsers.add_parser("align", help="align a shared-marker matrix")
    align.add_argument("--raw", type=Path, required=True, help="cells x markers NumPy .npy")
    align.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="row-aligned CSV/TSV/Parquet containing dataset and unit_key",
    )
    align.add_argument("--markers", type=Path, required=True, help="one marker per line")
    align.add_argument("--output", type=Path, required=True, help="result directory")
    align.add_argument(
        "--family",
        required=True,
        help="stable run identifier used in deterministic role assignment",
    )
    align.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="explicit reproducible order of the two or three datasets",
    )
    align.add_argument(
        "--output-dtype",
        choices=("float32", "float64"),
        default="float64",
        help=(
            "matrix storage dtype; input conversion and final mapping use float64; "
            "some internal steps use float32"
        ),
    )
    align.add_argument(
        "--overwrite",
        action="store_true",
        help="replace known RAMP files but never delete unrelated files",
    )
    align.set_defaults(func=_align)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
