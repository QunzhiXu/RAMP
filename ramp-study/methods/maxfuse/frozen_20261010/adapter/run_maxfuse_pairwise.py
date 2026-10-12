#!/usr/bin/env python3
"""Generic file runner for the frozen pairwise MaxFuse public-API workflow."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import tempfile

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

from maxfuse_repro import (
    PairwiseConfig,
    ViewData,
    __version__,
    algorithm_source_hashes,
    fit_pairwise,
    output_hashes,
    runtime_manifest,
    sha256_file,
    strict_string_vector,
)


def write_result(
    output: Path,
    input_path: Path,
    input_sha256: str,
    config_path: Path | None,
    config_sha256: str | None,
    config: PairwiseConfig,
    result: object,
    first: ViewData,
    second: ViewData,
    source_sha256: dict[str, str],
) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        np.savez_compressed(
            temporary / "pivot_matching.npz",
            first=result.pivot_matching.first,
            second=result.pivot_matching.second,
            score=result.pivot_matching.score,
            first_cell_id=np.asarray(first.cell_ids)[result.pivot_matching.first],
            second_cell_id=np.asarray(second.cell_ids)[result.pivot_matching.second],
        )
        np.save(temporary / "cca_embedding1.npy", result.embedding1)
        np.save(temporary / "cca_embedding2.npy", result.embedding2)
        np.save(
            temporary / "cca_embedding1_cell_ids.npy",
            np.asarray(first.cell_ids, dtype=np.str_),
        )
        np.save(
            temporary / "cca_embedding2_cell_ids.npy",
            np.asarray(second.cell_ids, dtype=np.str_),
        )
        if result.full_matching is not None:
            np.savez_compressed(
                temporary / "full_matching.npz",
                first=result.full_matching.first,
                second=result.full_matching.second,
                score=result.full_matching.score,
                first_cell_id=np.asarray(first.cell_ids)[result.full_matching.first],
                second_cell_id=np.asarray(second.cell_ids)[result.full_matching.second],
            )
        artifacts = output_hashes(temporary)
        manifest = {
            "manifest_schema": "maxfuse_repro.pairwise_manifest.v2",
            "status": "complete",
            "method": "official MaxFuse pairwise public-API workflow",
            "wrapper_version": __version__,
            "input": str(input_path.resolve()),
            "input_sha256": input_sha256,
            "config_file": str(config_path.resolve()) if config_path is not None else None,
            "config_file_sha256": config_sha256,
            "configuration": config.to_dict(),
            "view_schema": {
                "first": {
                    "cell_count": len(first.cell_ids),
                    "shared_feature_names": list(first.shared_feature_names),
                    "active_feature_names": list(first.active_feature_names),
                },
                "second": {
                    "cell_count": len(second.cell_ids),
                    "shared_feature_names": list(second.shared_feature_names),
                    "active_feature_names": list(second.active_feature_names),
                },
            },
            "labels_supplied_to_maxfuse": False,
            "pivot_records": int(len(result.pivot_matching.first)),
            "full_matching_records": (
                int(len(result.full_matching.first))
                if result.full_matching is not None
                else None
            ),
            "upstream": asdict(result.upstream),
            "runtime": runtime_manifest(),
            "algorithm_source_sha256": source_sha256,
            "output_sha256": artifacts,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        if algorithm_source_hashes((Path(__file__).resolve(),)) != source_sha256:
            raise RuntimeError("algorithm source changed during result serialization")
        if output.exists():
            raise FileExistsError(f"output appeared during fitting: {output}")
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--config", type=Path, help="PairwiseConfig JSON")
    choice.add_argument(
        "--official-citeseq-profile",
        action="store_true",
        help="use the official CITE-seq PBMC tutorial parameters",
    )
    parser.add_argument(
        "--seed", type=int, default=argparse.SUPPRESS,
        help=(
            "override the deterministic official-profile seed "
            "(default: 20260830)"
        ),
    )
    parser.add_argument(
        "--no-propagation", action="store_true", default=argparse.SUPPRESS,
        help="skip the official propagation/full-matching stage",
    )
    args = parser.parse_args()

    runner = Path(__file__).resolve()
    source_sha256 = algorithm_source_hashes((runner,))
    input_sha256 = sha256_file(args.input)
    config_sha256: str | None = None
    if args.config is not None:
        if hasattr(args, "seed") or hasattr(args, "no_propagation"):
            raise ValueError("--seed/--no-propagation apply only to --official-citeseq-profile")
        config_sha256 = sha256_file(args.config)
        config = PairwiseConfig.from_dict(json.loads(args.config.read_text()))
    else:
        config = PairwiseConfig.citeseq_pbmc_tutorial(
            deterministic_seed=getattr(args, "seed", 20260830),
            include_propagation=not getattr(args, "no_propagation", False),
        )
    with np.load(args.input, allow_pickle=False) as archive:
        required = {
            "shared1", "shared2", "active1", "active2",
            "shared1_feature_names", "shared2_feature_names",
            "active1_feature_names", "active2_feature_names",
            "cell_ids1", "cell_ids2",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"input NPZ lacks arrays: {sorted(missing)}")
        shared_names1 = strict_string_vector(
            archive["shared1_feature_names"], "shared1_feature_names"
        )
        shared_names2 = strict_string_vector(
            archive["shared2_feature_names"], "shared2_feature_names"
        )
        active_names1 = strict_string_vector(
            archive["active1_feature_names"], "active1_feature_names"
        )
        active_names2 = strict_string_vector(
            archive["active2_feature_names"], "active2_feature_names"
        )
        cell_ids1 = strict_string_vector(archive["cell_ids1"], "cell_ids1")
        cell_ids2 = strict_string_vector(archive["cell_ids2"], "cell_ids2")
        first = ViewData(
            archive["shared1"].copy(), archive["active1"].copy(),
            cell_ids=cell_ids1,
            active_feature_names=active_names1,
            shared_feature_names=shared_names1,
        )
        second = ViewData(
            archive["shared2"].copy(), archive["active2"].copy(),
            cell_ids=cell_ids2,
            active_feature_names=active_names2,
            shared_feature_names=shared_names2,
        )
    result = fit_pairwise(first, second, config)
    if sha256_file(args.input) != input_sha256:
        raise RuntimeError("input NPZ changed during fitting")
    if args.config is not None and sha256_file(args.config) != config_sha256:
        raise RuntimeError("configuration JSON changed during fitting")
    if algorithm_source_hashes((runner,)) != source_sha256:
        raise RuntimeError("algorithm source changed during fitting")
    write_result(
        args.output,
        args.input,
        input_sha256,
        args.config,
        config_sha256,
        config,
        result,
        first,
        second,
        source_sha256,
    )
    print(
        f"MaxFuse pairwise complete: {len(result.pivot_matching.first):,} pivot records"
    )


if __name__ == "__main__":
    main()
