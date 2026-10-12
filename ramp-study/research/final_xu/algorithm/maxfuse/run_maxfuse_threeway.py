#!/usr/bin/env python3
"""Run the official-Archive three-view MaxFuse workflow."""

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
    THREE_VIEW_EXTENSION_METHOD,
    ViewData,
    __version__,
    algorithm_source_hashes,
    fit_three_view_extension,
    output_hashes,
    runtime_manifest,
    sha256_file,
    strict_string_vector,
)


def _optional_positive_int(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"none", "null"}:
        return None
    try:
        parsed = int(normalized)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "SVD components must be a positive integer or 'none'"
        ) from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("SVD components must be positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--views", nargs=3, required=True, metavar=("VIEW0", "VIEW1", "VIEW2"))
    parser.add_argument("--hub", required=True)
    parser.add_argument(
        "--pair-config-map",
        type=Path,
        required=True,
        help="JSON object mapping each non-hub view name to a PairwiseConfig object",
    )
    parser.add_argument(
        "--svd-components",
        nargs=3,
        type=_optional_positive_int,
        required=True,
        metavar=("VIEW0_SVD", "VIEW1_SVD", "VIEW2_SVD"),
        help="per-view Archive SVD widths; use 'none' to skip SVD",
    )
    parser.add_argument("--gcca-components", type=int, default=15)
    parser.add_argument("--gcca-initialization-max-iter", type=int, default=2000)
    parser.add_argument("--gcca-refinement-max-iter", type=int, default=2000)
    parser.add_argument("--gcca-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--outer-reuse-policy",
        choices=("allow", "drop_reused"),
        required=True,
        help=(
            "allow hub-chained outer endpoint reuse, or use the explicitly "
            "non-official drop_reused sensitivity rule"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    names = tuple(map(str, args.views))
    if any(not name for name in names) or len(set(names)) != 3 or names[0] != args.hub:
        raise ValueError("--views must be unique and place --hub first")
    svd_components = {
        name: value for name, value in zip(names, args.svd_components)
    }
    runner = Path(__file__).resolve()
    packaged_upstream_lock = runner.parent / "maxfuse_repro" / "upstream_lock.json"
    upstream_lock_payload = json.loads(packaged_upstream_lock.read_text())
    source_sha256 = algorithm_source_hashes((runner,))
    input_sha256 = sha256_file(args.input)
    config_sha256 = sha256_file(args.pair_config_map)
    with np.load(args.input, allow_pickle=False) as archive:
        views: dict[str, ViewData] = {}
        for position, name in enumerate(names):
            shared_key = f"view{position}_shared"
            active_key = f"view{position}_active"
            shared_feature_key = f"view{position}_shared_feature_names"
            active_feature_key = f"view{position}_active_feature_names"
            cell_id_key = f"view{position}_cell_ids"
            missing = {
                key for key in (
                    shared_key, active_key, shared_feature_key,
                    active_feature_key, cell_id_key,
                )
                if key not in archive.files
            }
            if missing:
                raise ValueError(f"input NPZ lacks arrays: {sorted(missing)}")
            shared_features = strict_string_vector(
                archive[shared_feature_key], shared_feature_key
            )
            active_features = strict_string_vector(
                archive[active_feature_key], active_feature_key
            )
            cell_ids = strict_string_vector(archive[cell_id_key], cell_id_key)
            views[name] = ViewData(
                archive[shared_key].copy(),
                archive[active_key].copy(),
                cell_ids=cell_ids,
                active_feature_names=active_features,
                shared_feature_names=shared_features,
            )
    config_payload = json.loads(args.pair_config_map.read_text())
    if not isinstance(config_payload, dict):
        raise TypeError("pair-config map must be a JSON object")
    outer_names = {name for name in names if name != args.hub}
    if set(config_payload) != outer_names:
        raise ValueError("pair-config map names do not match the two non-hub views")
    configs = {
        name: PairwiseConfig.from_dict(config_payload[name]) for name in outer_names
    }
    result = fit_three_view_extension(
        views,
        hub=args.hub,
        view_order=names,
        pairwise_configs=configs,
        svd_components=svd_components,
        gcca_components=args.gcca_components,
        gcca_initialization_max_iter=args.gcca_initialization_max_iter,
        gcca_refinement_max_iter=args.gcca_refinement_max_iter,
        gcca_tolerance=args.gcca_tolerance,
        outer_reuse_policy=args.outer_reuse_policy,
    )
    if sha256_file(args.input) != input_sha256:
        raise RuntimeError("input NPZ changed during fitting")
    if sha256_file(args.pair_config_map) != config_sha256:
        raise RuntimeError("pair-config map changed during fitting")
    if algorithm_source_hashes((runner,)) != source_sha256:
        raise RuntimeError("algorithm source changed during fitting")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        for position, name in enumerate(names):
            np.save(temporary / f"view{position}_gcca_embedding.npy", result.embeddings[name])
            np.save(
                temporary / f"view{position}_gcca_cell_ids.npy",
                np.asarray(views[name].cell_ids, dtype=np.str_),
            )
        np.savez_compressed(
            temporary / "triplet_matching.npz",
            hub=result.triplets.hub,
            outer_left=result.triplets.outer_left,
            outer_right=result.triplets.outer_right,
            left_score=result.triplets.left_score,
            right_score=result.triplets.right_score,
            hub_cell_id=np.asarray(views[result.audit.view_order[0]].cell_ids)[result.triplets.hub],
            outer_left_cell_id=np.asarray(views[result.audit.view_order[1]].cell_ids)[result.triplets.outer_left],
            outer_right_cell_id=np.asarray(views[result.audit.view_order[2]].cell_ids)[result.triplets.outer_right],
        )
        pair_dir = temporary / "pairwise_pivots"
        pair_dir.mkdir()
        pairwise_files: dict[str, str] = {}
        for position, outer in enumerate(sorted(outer_names)):
            matching = result.pairwise[outer].pivot_matching
            relative = f"pairwise_pivots/pair{position}.npz"
            pairwise_files[outer] = relative
            np.savez_compressed(
                temporary / relative,
                first=matching.first,
                second=matching.second,
                score=matching.score,
                hub_cell_id=np.asarray(views[args.hub].cell_ids)[matching.first],
                outer_cell_id=np.asarray(views[outer].cell_ids)[matching.second],
            )
        result.projection_model.save(temporary / "projection_model.npz")
        artifacts = output_hashes(temporary)
        manifest = {
            "manifest_schema": "maxfuse_repro.three_view_extension_manifest.v2",
            "status": "complete",
            "method": THREE_VIEW_EXTENSION_METHOD,
            "wrapper_version": __version__,
            "input": str(args.input.resolve()),
            "input_sha256": input_sha256,
            "pair_config_map": str(args.pair_config_map.resolve()),
            "pair_config_map_sha256": config_sha256,
            "views": list(names),
            "hub": args.hub,
            "view_schema": {
                name: {
                    "position": position,
                    "cell_count": len(views[name].cell_ids),
                    "shared_feature_names": list(views[name].shared_feature_names),
                    "active_feature_names": list(views[name].active_feature_names),
                    "embedding_file": f"view{position}_gcca_embedding.npy",
                    "embedding_cell_ids_file": f"view{position}_gcca_cell_ids.npy",
                }
                for position, name in enumerate(names)
            },
            "pairwise_output_files": pairwise_files,
            "triplet_index_order": list(result.audit.view_order),
            "configuration": {
                "pairwise": {name: configs[name].to_dict() for name in sorted(configs)},
                "view_order": list(names),
                "svd_components": {
                    name: svd_components[name] for name in names
                },
                "gcca_components": args.gcca_components,
                "gcca_initialization_max_iter": args.gcca_initialization_max_iter,
                "gcca_refinement_max_iter": args.gcca_refinement_max_iter,
                "gcca_tolerance": args.gcca_tolerance,
                "outer_reuse_policy": args.outer_reuse_policy,
            },
            "labels_supplied_to_maxfuse": False,
            "audit": asdict(result.audit),
            "upstream": {
                name: asdict(result.pairwise[name].upstream) for name in sorted(outer_names)
            },
            "archive_reference": {
                "repository": upstream_lock_payload["repository"],
                "reviewed_repository_commit": upstream_lock_payload[
                    "reviewed_repository_commit"
                ],
                "notebooks": upstream_lock_payload["archive_reference_notebooks"],
                "upstream_lock_sha256": sha256_file(packaged_upstream_lock),
            },
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
    print(
        f"{THREE_VIEW_EXTENSION_METHOD} complete: {result.audit.triplets:,} triplets, "
        f"{result.audit.gcca_components} GCCA components"
    )


if __name__ == "__main__":
    main()
