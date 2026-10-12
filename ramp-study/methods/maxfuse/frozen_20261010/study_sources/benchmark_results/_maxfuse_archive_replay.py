#!/usr/bin/env python3
"""Replay Archive GCCA from verified MaxFuse-v2 pairwise triplets.

This runner never calls pairwise MaxFuse.  It reuses the immutable v2 fitted
inputs and hub-chained triplets, applies the provenance-locked Archive GCCA
workflow, projects the model to every cell, and writes a separate v3 alignment.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FINAL_XU = HERE.parent
ALGORITHM_DIR = FINAL_XU / "algorithm" / "maxfuse"
if str(ALGORITHM_DIR) not in sys.path:
    sys.path.insert(0, str(ALGORITHM_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _maxfuse_benchmark as benchmark
from maxfuse_repro import (
    THREE_VIEW_EXTENSION_METHOD,
    __version__,
    algorithm_source_hashes,
    archive_svd_scores,
    output_hashes,
    runtime_manifest,
    sha256_file,
    strict_string_vector,
)
from maxfuse_repro.gcca import fit_archive_gcca, matched_component_correlations
from maxfuse_repro.projection import build_projection_model


REPLAY_SCHEMA = "final_xu.maxfuse_archive_replay.v1"
FIT_METADATA_COLUMNS = ("cell_id", "dataset", "donor", "unit_key")


def maximum_multiplicity(indices: np.ndarray) -> int:
    if len(indices) == 0:
        return 0
    return int(np.max(np.unique(indices, return_counts=True)[1]))


def reused_records(indices: np.ndarray) -> int:
    return int(len(indices) - len(np.unique(indices)))


def verify_legacy_hub(source: Path) -> tuple[dict, dict]:
    alignment_manifest_path = source / "alignment_manifest.json"
    formal_manifest_path = source / "formal_fit" / "manifest.json"
    if not alignment_manifest_path.is_file() or not formal_manifest_path.is_file():
        raise FileNotFoundError(f"legacy hub is incomplete: {source}")
    alignment_manifest = json.loads(alignment_manifest_path.read_text())
    formal_manifest = json.loads(formal_manifest_path.read_text())
    if alignment_manifest.get("status") != "complete" or formal_manifest.get("status") != "complete":
        raise RuntimeError(f"legacy hub is not complete: {source}")
    if formal_manifest.get("labels_supplied_to_maxfuse") is not False:
        raise RuntimeError(f"legacy pairwise fit was not explicitly label-free: {source}")
    if alignment_manifest.get("labels_used_for_fitting") is not False:
        raise RuntimeError(f"legacy alignment was not explicitly label-free: {source}")
    if alignment_manifest.get("labels_loaded_before_alignment_completed") is not False:
        raise RuntimeError(f"legacy labels were loaded before alignment completed: {source}")
    if not benchmark.verify_hash_map(source, alignment_manifest.get("outputs", {})):
        raise RuntimeError(f"legacy alignment output hash failure: {source}")
    if not benchmark.verify_hash_map(
        source / "formal_fit", formal_manifest.get("output_sha256", {})
    ):
        raise RuntimeError(f"legacy formal-fit output hash failure: {source}")
    fit_input = source / "input" / "fit_input.npz"
    if sha256_file(fit_input) != formal_manifest.get("input_sha256"):
        raise RuntimeError(f"legacy fit-input hash failure: {fit_input}")
    return alignment_manifest, formal_manifest


def replay_one_hub(config: dict, config_path: Path, hub_slug: str) -> Path:
    canonical_slugs = [row["slug"] for row in config["datasets"]]
    if hub_slug not in canonical_slugs:
        raise KeyError(hub_slug)
    hub_position = canonical_slugs.index(hub_slug)
    view_order = tuple(
        canonical_slugs[hub_position:] + canonical_slugs[:hub_position]
    )
    extension = config["three_view_extension"]
    if extension["method"] != THREE_VIEW_EXTENSION_METHOD:
        raise ValueError("config method does not match the installed Archive workflow")
    if extension["view_order_policy"] != "cyclic_from_dataset_config_order":
        raise ValueError("replay requires the cyclic view-order policy")

    legacy_root = HERE / config["benchmark_slug"] / "maxfuse_v2"
    legacy_alignment = legacy_root / f"hub_{hub_slug}" / "alignment"
    old_alignment_manifest, old_formal_manifest = verify_legacy_hub(legacy_alignment)
    if old_formal_manifest.get("hub") != hub_slug:
        raise AssertionError("legacy formal-fit hub does not match the requested hub")

    target = benchmark.output_root(config) / f"hub_{hub_slug}" / "alignment"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite Archive alignment: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".archive-replay.tmp-", dir=target.parent))
    started = time.time()
    try:
        fit_input_path = legacy_alignment / "input" / "fit_input.npz"
        fit_active: dict[str, np.ndarray] = {}
        fit_cell_ids: dict[str, np.ndarray] = {}
        feature_names: dict[str, tuple[str, ...]] = {}
        with np.load(fit_input_path, allow_pickle=False) as archive:
            expected_fit_input_fields = {
                f"view{int(schema['position'])}_{suffix}"
                for schema in old_formal_manifest["view_schema"].values()
                for suffix in (
                    "shared",
                    "active",
                    "shared_feature_names",
                    "active_feature_names",
                    "cell_ids",
                )
            }
            if set(archive.files) != expected_fit_input_fields:
                raise AssertionError(
                    "legacy formal fit input contains unexpected fields; "
                    "outcome-free provenance cannot be established"
                )
            for slug, schema in old_formal_manifest["view_schema"].items():
                position = int(schema["position"])
                fit_active[slug] = archive[f"view{position}_active"].copy()
                fit_cell_ids[slug] = np.asarray(
                    strict_string_vector(
                        archive[f"view{position}_cell_ids"],
                        f"view{position}_cell_ids",
                    ),
                    dtype=np.str_,
                )
                feature_names[slug] = strict_string_vector(
                    archive[f"view{position}_active_feature_names"],
                    f"view{position}_active_feature_names",
                )
                if feature_names[slug] != tuple(schema["active_feature_names"]):
                    raise AssertionError(f"{slug}: legacy active-feature order drift")

        triplet_path = legacy_alignment / "formal_fit" / "triplet_matching.npz"
        old_order = tuple(old_formal_manifest["triplet_index_order"])
        if len(old_order) != 3 or old_order[0] != hub_slug:
            raise AssertionError(
                "legacy triplet order does not place the requested hub first"
            )
        with np.load(triplet_path, allow_pickle=False) as archive:
            indices_by_view = {
                old_order[0]: archive["hub"].copy(),
                old_order[1]: archive["outer_left"].copy(),
                old_order[2]: archive["outer_right"].copy(),
            }
            score_by_view = {
                old_order[1]: archive["left_score"].copy(),
                old_order[2]: archive["right_score"].copy(),
            }
            triplet_ids_by_view = {
                old_order[0]: np.asarray(
                    strict_string_vector(archive["hub_cell_id"], "hub_cell_id"),
                    dtype=np.str_,
                ),
                old_order[1]: np.asarray(
                    strict_string_vector(
                        archive["outer_left_cell_id"], "outer_left_cell_id"
                    ),
                    dtype=np.str_,
                ),
                old_order[2]: np.asarray(
                    strict_string_vector(
                        archive["outer_right_cell_id"], "outer_right_cell_id"
                    ),
                    dtype=np.str_,
                ),
            }
        if set(indices_by_view) != set(view_order):
            raise AssertionError("legacy triplet views do not match the new ordered views")
        tuple_count = len(indices_by_view[hub_slug])
        if any(len(indices_by_view[slug]) != tuple_count for slug in view_order):
            raise AssertionError("legacy triplet index lengths differ")
        for slug in view_order:
            indices = indices_by_view[slug]
            if indices.ndim != 1 or indices.dtype.kind not in {"i", "u"}:
                raise AssertionError(
                    f"{slug}: legacy triplet indices are not an integer vector"
                )
            if np.any(indices < 0) or np.any(indices >= len(fit_cell_ids[slug])):
                raise AssertionError(f"{slug}: legacy triplet indices are out of bounds")
            if len(triplet_ids_by_view[slug]) != tuple_count:
                raise AssertionError(f"{slug}: legacy triplet cell-ID length differs")
            if not np.array_equal(
                fit_cell_ids[slug][indices].astype(str),
                triplet_ids_by_view[slug].astype(str),
            ):
                raise AssertionError(
                    f"{slug}: legacy triplet cell IDs do not match indices"
                )

        old_audit = old_formal_manifest["audit"]
        if old_audit.get("outer_reuse_policy") != extension["outer_reuse_policy"]:
            raise AssertionError("legacy and Archive outer-reuse policies differ")

        preprocessing = {
            slug: archive_svd_scores(
                fit_active[slug], extension["svd_components"][slug]
            )
            for slug in view_order
        }
        gcca = fit_archive_gcca(
            [preprocessing[slug].scores for slug in view_order],
            [indices_by_view[slug] for slug in view_order],
            requested_components=int(extension["gcca_components"]),
            initialization_max_iter=int(extension["gcca_initialization_max_iter"]),
            refinement_max_iter=int(extension["gcca_refinement_max_iter"]),
            tolerance=float(extension["gcca_tolerance"]),
        )
        fit_embeddings = {
            slug: value for slug, value in zip(view_order, gcca.embeddings)
        }
        model = build_projection_model(
            view_order, preprocessing, gcca, feature_names
        )
        roundtrip = {}
        for slug in view_order:
            projected = model.transform(slug, fit_active[slug], feature_names[slug])
            error = float(np.max(np.abs(projected - fit_embeddings[slug])))
            if error > 1e-9:
                raise AssertionError(f"{slug}: Archive projection roundtrip failed: {error}")
            roundtrip[slug] = error
        correlations = matched_component_correlations(
            [fit_embeddings[slug] for slug in view_order],
            [indices_by_view[slug] for slug in view_order],
        )
        if not np.isfinite(correlations).all():
            raise RuntimeError("matched Archive component correlations are non-finite")

        outer_names = view_order[1:]
        audit = {
            "hub": hub_slug,
            "view_order": list(view_order),
            "view_order_policy": extension["view_order_policy"],
            "pairwise_pivot_records": old_audit["pairwise_pivot_records"],
            "pairwise_unique_hub_pivots": old_audit["pairwise_unique_hub_pivots"],
            "triplets": tuple_count,
            "outer_reuse_policy": extension["outer_reuse_policy"],
            "outer_reused_records": {
                slug: reused_records(indices_by_view[slug]) for slug in outer_names
            },
            "outer_maximum_multiplicity": {
                slug: maximum_multiplicity(indices_by_view[slug]) for slug in outer_names
            },
            "archive_svd_components": [
                extension["svd_components"][slug] for slug in view_order
            ],
            "gcca_components": gcca.n_components,
            "gcca_effective_ranks": list(gcca.effective_ranks),
            "gcca_iteration_counts": list(gcca.iteration_counts),
            "gcca_final_objectives": list(gcca.final_objectives),
            "gcca_all_components_converged": bool(all(gcca.converged)),
            "projection_roundtrip_max_abs_error": roundtrip,
            "mean_matched_component_correlation": float(np.mean(correlations)),
            "mean_matched_component_correlation_role": (
                "training-fit diagnostic on the GCCA triplets"
            ),
        }

        formal = temporary / "formal_fit"
        formal.mkdir()
        for position, slug in enumerate(view_order):
            np.save(formal / f"view{position}_gcca_embedding.npy", fit_embeddings[slug])
            np.save(formal / f"view{position}_gcca_cell_ids.npy", fit_cell_ids[slug])
        np.savez_compressed(
            formal / "triplet_matching.npz",
            hub=indices_by_view[view_order[0]],
            outer_left=indices_by_view[view_order[1]],
            outer_right=indices_by_view[view_order[2]],
            left_score=score_by_view[view_order[1]],
            right_score=score_by_view[view_order[2]],
            hub_cell_id=fit_cell_ids[view_order[0]][indices_by_view[view_order[0]]],
            outer_left_cell_id=fit_cell_ids[view_order[1]][indices_by_view[view_order[1]]],
            outer_right_cell_id=fit_cell_ids[view_order[2]][indices_by_view[view_order[2]]],
        )
        model.save(formal / "projection_model.npz")
        formal_outputs = output_hashes(formal)
        formal_manifest = {
            "manifest_schema": "maxfuse_repro.archive_replay_manifest.v1",
            "status": "complete",
            "method": THREE_VIEW_EXTENSION_METHOD,
            "wrapper_version": __version__,
            "views": list(view_order),
            "hub": hub_slug,
            "triplet_index_order": list(view_order),
            "view_schema": {
                slug: {
                    "position": position,
                    "cell_count": int(len(fit_active[slug])),
                    "active_feature_names": list(feature_names[slug]),
                    "embedding_file": f"view{position}_gcca_embedding.npy",
                    "embedding_cell_ids_file": f"view{position}_gcca_cell_ids.npy",
                }
                for position, slug in enumerate(view_order)
            },
            "configuration": extension,
            "pairwise_reused_without_refitting": True,
            "pairwise_subprocess_invoked": False,
            "legacy_ridge_embedding_or_model_used": False,
            "legacy_pairwise_source": {
                "alignment": str(legacy_alignment.resolve()),
                "alignment_manifest_sha256": sha256_file(
                    legacy_alignment / "alignment_manifest.json"
                ),
                "formal_manifest_sha256": sha256_file(
                    legacy_alignment / "formal_fit" / "manifest.json"
                ),
                "fit_input_sha256": sha256_file(fit_input_path),
                "triplet_matching_sha256": sha256_file(triplet_path),
                "legacy_triplet_index_order": list(old_order),
            },
            "labels_supplied_to_maxfuse_or_gcca": False,
            "outcomes_supplied_to_maxfuse_or_gcca": False,
            "formal_fit_input_fields": sorted(expected_fit_input_fields),
            "outcome_provenance": (
                "formal fit input is restricted to shared/active protein matrices, "
                "feature names, and cell IDs; no outcome field is present"
            ),
            "audit": audit,
            "runtime": runtime_manifest(),
            "algorithm_source_sha256": algorithm_source_hashes(
                (
                    Path(__file__).resolve(),
                    Path(benchmark.__file__).resolve(),
                )
            ),
            "output_sha256": formal_outputs,
        }
        (formal / "manifest.json").write_text(
            json.dumps(formal_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )

        prepared = legacy_root / "prepared"
        prepared_manifest_path = prepared / "manifest.json"
        prepared_manifest = json.loads(prepared_manifest_path.read_text())
        if not benchmark.verify_hash_map(prepared, prepared_manifest.get("outputs", {})):
            raise RuntimeError(f"legacy prepared hash failure: {prepared}")
        metadata = pd.read_csv(prepared / "fit_metadata.csv.gz", low_memory=False)
        if tuple(metadata.columns) != FIT_METADATA_COLUMNS:
            raise AssertionError(
                "frozen fit metadata contains unexpected columns; outcome-free "
                "provenance cannot be established"
            )
        dataset_name = {row["slug"]: row["name"] for row in config["datasets"]}
        dataset_values = metadata["dataset"].astype(str).to_numpy()
        full_embedding = np.empty((len(metadata), gcca.n_components), dtype=np.float64)
        projection_audit = []
        for slug in canonical_slugs:
            rows = np.flatnonzero(dataset_values == dataset_name[slug])
            active_path = prepared / prepared_manifest["active_input_files"][slug]
            full_active = np.load(active_path, allow_pickle=False)
            full_features = tuple(
                (prepared / prepared_manifest["active_marker_files"][slug])
                .read_text()
                .splitlines()
            )
            if full_features != feature_names[slug]:
                raise AssertionError(f"{slug}: full active-feature order drift")
            if full_active.shape[0] != len(rows):
                raise AssertionError(f"{slug}: full active row count mismatch")
            fit_indices = np.load(
                prepared / prepared_manifest["fit_index_files"][slug],
                allow_pickle=False,
            )
            local_rows = np.searchsorted(rows, fit_indices)
            if not np.array_equal(rows[local_rows], fit_indices):
                raise AssertionError(f"{slug}: invalid frozen fit indices")
            if not np.array_equal(
                metadata.iloc[fit_indices]["cell_id"].astype(str).to_numpy(),
                fit_cell_ids[slug].astype(str),
            ):
                raise AssertionError(f"{slug}: fit cell IDs drifted")
            if not np.array_equal(full_active[local_rows], fit_active[slug]):
                raise AssertionError(f"{slug}: fit active matrix is not the frozen full slice")
            projected = model.transform(slug, full_active, full_features)
            if projected.shape != (len(rows), gcca.n_components):
                raise AssertionError(f"{slug}: invalid full Archive projection shape")
            full_embedding[rows] = projected
            projection_audit.append({
                "dataset": dataset_name[slug],
                "slug": slug,
                "fit_cells": int(len(fit_active[slug])),
                "projected_full_cells": int(len(rows)),
            })
        if not np.isfinite(full_embedding).all():
            raise RuntimeError("full Archive embedding contains non-finite values")

        data = temporary / "data"
        data.mkdir()
        embedding_path = data / "maxfuse_embedding_float64.npy"
        ids_path = data / "cell_ids.npy"
        metadata_path = data / "cell_metadata_label_free.csv.gz"
        np.save(embedding_path, full_embedding)
        cell_ids = metadata["cell_id"].astype(str).to_numpy(dtype=np.str_)
        np.save(ids_path, cell_ids)
        shutil.copyfile(prepared / "fit_metadata.csv.gz", metadata_path)
        pd.DataFrame(projection_audit).to_csv(
            temporary / "full_projection_audit.csv", index=False
        )
        outputs = {
            "data/maxfuse_embedding_float64.npy": sha256_file(embedding_path),
            "data/cell_ids.npy": sha256_file(ids_path),
            "data/cell_metadata_label_free.csv.gz": sha256_file(metadata_path),
            "full_projection_audit.csv": sha256_file(
                temporary / "full_projection_audit.csv"
            ),
        }
        alignment_manifest = {
            "schema_version": benchmark.ALIGNMENT_SCHEMA,
            "replay_schema": REPLAY_SCHEMA,
            "status": "complete",
            "benchmark": config["benchmark_slug"],
            "method": THREE_VIEW_EXTENSION_METHOD,
            "hub_slug": hub_slug,
            "hub_dataset": dataset_name[hub_slug],
            "view_input_order": list(view_order),
            "view_order_policy": extension["view_order_policy"],
            "fit_cells": {
                slug: int(len(fit_active[slug])) for slug in canonical_slugs
            },
            "full_cells": int(len(metadata)),
            "active_markers": {
                slug: list(feature_names[slug]) for slug in canonical_slugs
            },
            "feature_protocol": config["feature_protocol"],
            "fit_sampling": config["fit_sampling"],
            "requested_archive_svd_components": extension["svd_components"],
            "requested_gcca_components": int(extension["gcca_components"]),
            "full_projection": projection_audit,
            "labels_used_for_fitting": False,
            "outcomes_used_for_fitting": False,
            "outcome_provenance": {
                "status": "not_present_in_any_fit_input",
                "formal_fit_input_fields": sorted(expected_fit_input_fields),
                "fit_metadata_columns": list(FIT_METADATA_COLUMNS),
            },
            "pairwise_reused_without_refitting": True,
            "pairwise_subprocess_invoked": False,
            "legacy_ridge_embedding_or_model_used": False,
            "formal_fit_audit": audit,
            "formal_fit_manifest": str((target / "formal_fit" / "manifest.json").resolve()),
            "formal_fit_manifest_sha256": sha256_file(formal / "manifest.json"),
            "legacy_pairwise_source": formal_manifest["legacy_pairwise_source"],
            "prepared_source": {
                "directory": str(prepared.resolve()),
                "manifest_sha256": sha256_file(prepared_manifest_path),
            },
            "config": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "replay_runner": str(Path(__file__).resolve()),
            "replay_runner_sha256": sha256_file(Path(__file__).resolve()),
            "algorithm_source_sha256": formal_manifest["algorithm_source_sha256"],
            "runtime_seconds": time.time() - started,
            "outputs": outputs,
        }
        (temporary / "alignment_manifest.json").write_text(
            json.dumps(alignment_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        temporary.replace(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        f"[{config['benchmark_slug']}/{hub_slug}] Archive GCCA replay complete: "
        f"{tuple_count:,} triplets, {gcca.n_components} components, "
        f"{time.time() - started:.1f}s",
        flush=True,
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hub", default="all")
    args = parser.parse_args()
    config, config_path = benchmark.load_config(args.config)
    slugs = [row["slug"] for row in config["datasets"]]
    selected = slugs if args.hub == "all" else [args.hub]
    if set(selected) - set(slugs):
        raise KeyError(f"unknown hub: {args.hub}")
    for slug in selected:
        replay_one_hub(config, config_path, slug)


if __name__ == "__main__":
    main()
