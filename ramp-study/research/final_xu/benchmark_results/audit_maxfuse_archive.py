#!/usr/bin/env python3
"""Audit all formal MaxFuse Archive alignments, UMAPs, and summaries."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ALGORITHM_DIR = HERE.parent / "algorithm" / "maxfuse"
for location in (HERE, ALGORITHM_DIR):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

import _maxfuse_benchmark as benchmark
from maxfuse_repro import algorithm_source_hashes, strict_string_vector


CANCERS = ("bone_marrow", "tnbc", "melanoma", "lung_cancer")
EXPECTED_CELLS = {
    "bone_marrow": 88_031,
    "tnbc": 62_701,
    "melanoma": 60_000,
    "lung_cancer": 60_000,
}
REPLAY_RUNNER = HERE / "_maxfuse_archive_replay.py"
BENCHMARK_HELPER = HERE / "_maxfuse_benchmark.py"
FIT_METADATA_COLUMNS = ("cell_id", "dataset", "donor", "unit_key")


def maximum_multiplicity(indices: np.ndarray) -> int:
    if len(indices) == 0:
        return 0
    return int(np.max(np.unique(indices, return_counts=True)[1]))


def reused_records(indices: np.ndarray) -> int:
    return int(len(indices) - len(np.unique(indices)))


def expected_fit_input_fields(view_schema: dict) -> set[str]:
    return {
        f"view{int(schema['position'])}_{suffix}"
        for schema in view_schema.values()
        for suffix in (
            "shared",
            "active",
            "shared_feature_names",
            "active_feature_names",
            "cell_ids",
        )
    }


def load_triplets(path: Path, order: tuple[str, ...]):
    with np.load(path, allow_pickle=False) as archive:
        indices = {
            order[0]: archive["hub"].copy(),
            order[1]: archive["outer_left"].copy(),
            order[2]: archive["outer_right"].copy(),
        }
        identifiers = {
            order[0]: np.asarray(
                strict_string_vector(archive["hub_cell_id"], "hub_cell_id"),
                dtype=np.str_,
            ),
            order[1]: np.asarray(
                strict_string_vector(
                    archive["outer_left_cell_id"], "outer_left_cell_id"
                ),
                dtype=np.str_,
            ),
            order[2]: np.asarray(
                strict_string_vector(
                    archive["outer_right_cell_id"], "outer_right_cell_id"
                ),
                dtype=np.str_,
            ),
        }
        scores = {
            order[1]: archive["left_score"].copy(),
            order[2]: archive["right_score"].copy(),
        }
    return indices, identifiers, scores


def audit_current_pairwise_fit(
    cancer: str,
    hub: str,
    expected_order: tuple[str, ...],
    config: dict,
    manifest: dict,
    alignment: Path,
    formal: Path,
    formal_manifest: dict,
) -> None:
    """Audit a current fit that ran official pairwise MaxFuse from its own inputs."""

    prefix = f"{cancer}/{hub}"
    if cancer != "tnbc":
        raise AssertionError(f"{prefix}: unexpected current-fit provenance mode")
    if manifest.get("legacy_pairwise_source") is not None:
        raise AssertionError(f"{prefix}: current fit unexpectedly names a legacy source")
    if formal_manifest.get("labels_supplied_to_maxfuse") is not False:
        raise AssertionError(f"{prefix}: labels entered current formal fitting")
    if tuple(formal_manifest["triplet_index_order"]) != expected_order:
        raise AssertionError(f"{prefix}: current formal order drift")
    if manifest.get("formal_fit_audit") != formal_manifest.get("audit"):
        raise AssertionError(f"{prefix}: formal/alignment audit payload drift")
    if formal_manifest.get("algorithm_source_sha256") != algorithm_source_hashes():
        raise AssertionError(f"{prefix}: current algorithm source provenance drift")

    fit_input = alignment / "input" / "fit_input.npz"
    if formal_manifest.get("input_sha256") != benchmark.sha256(fit_input):
        raise AssertionError(f"{prefix}: current formal input hash drift")
    expected_fields = expected_fit_input_fields(formal_manifest["view_schema"])
    fit_ids: dict[str, np.ndarray] = {}
    with np.load(fit_input, allow_pickle=False) as archive:
        if set(archive.files) != expected_fields:
            raise AssertionError(f"{prefix}: unexpected field in current fit input")
        for slug, schema in formal_manifest["view_schema"].items():
            position = int(schema["position"])
            fit_ids[slug] = np.asarray(
                strict_string_vector(
                    archive[f"view{position}_cell_ids"],
                    f"{prefix}/view{position}_cell_ids",
                ),
                dtype=np.str_,
            )

    indices, identifiers, scores = load_triplets(
        formal / "triplet_matching.npz", expected_order
    )
    audit = formal_manifest["audit"]
    if int(audit["triplets"]) != len(indices[hub]):
        raise AssertionError(f"{prefix}: current triplet count drift")
    policy = config["three_view_extension"]["outer_reuse_policy"]
    if audit.get("outer_reuse_policy") != policy:
        raise AssertionError(f"{prefix}: current outer-reuse policy drift")
    for slug in expected_order:
        selected = indices[slug]
        if selected.ndim != 1 or selected.dtype.kind not in {"i", "u"}:
            raise AssertionError(f"{prefix}/{slug}: invalid current triplet indices")
        if np.any(selected < 0) or np.any(selected >= len(fit_ids[slug])):
            raise AssertionError(f"{prefix}/{slug}: current triplet index out of bounds")
        expected_ids = fit_ids[slug][selected].astype(str)
        if not np.array_equal(identifiers[slug].astype(str), expected_ids):
            raise AssertionError(f"{prefix}/{slug}: current triplet ID drift")
    for slug in expected_order[1:]:
        if not np.isfinite(scores[slug]).all():
            raise AssertionError(f"{prefix}/{slug}: non-finite current pairwise score")
        reused = reused_records(indices[slug])
        multiplicity = maximum_multiplicity(indices[slug])
        if int(audit["outer_reused_records"][slug]) != reused:
            raise AssertionError(f"{prefix}/{slug}: current reused-record audit drift")
        if int(audit["outer_maximum_multiplicity"][slug]) != multiplicity:
            raise AssertionError(f"{prefix}/{slug}: current multiplicity audit drift")
        if policy == "drop_reused" and multiplicity > 1:
            raise AssertionError(f"{prefix}/{slug}: reuse violates configured policy")


def audit_legacy_triplet_replay(
    cancer: str,
    hub: str,
    expected_order: tuple[str, ...],
    config: dict,
    manifest: dict,
    formal: Path,
    formal_manifest: dict,
) -> None:
    prefix = f"{cancer}/{hub}"
    if manifest.get("legacy_pairwise_source") != formal_manifest.get(
        "legacy_pairwise_source"
    ):
        raise AssertionError(f"{prefix}: legacy source manifests disagree")
    source = formal_manifest["legacy_pairwise_source"]
    legacy_alignment = Path(source["alignment"])
    source_files = {
        "alignment_manifest_sha256": legacy_alignment / "alignment_manifest.json",
        "formal_manifest_sha256": legacy_alignment / "formal_fit" / "manifest.json",
        "fit_input_sha256": legacy_alignment / "input" / "fit_input.npz",
        "triplet_matching_sha256": (
            legacy_alignment / "formal_fit" / "triplet_matching.npz"
        ),
    }
    for key, path in source_files.items():
        if not path.is_file() or source.get(key) != benchmark.sha256(path):
            raise AssertionError(f"{prefix}: legacy source hash drift: {key}")

    legacy_manifest = json.loads(source_files["alignment_manifest_sha256"].read_text())
    legacy_formal = json.loads(source_files["formal_manifest_sha256"].read_text())
    if legacy_formal.get("labels_supplied_to_maxfuse") is not False:
        raise AssertionError(f"{prefix}: legacy formal fit was not label-free")
    if legacy_manifest.get("labels_used_for_fitting") is not False:
        raise AssertionError(f"{prefix}: legacy alignment was not label-free")
    if legacy_manifest.get("labels_loaded_before_alignment_completed") is not False:
        raise AssertionError(f"{prefix}: legacy labels were loaded before alignment")
    if not benchmark.verify_hash_map(legacy_alignment, legacy_manifest.get("outputs", {})):
        raise AssertionError(f"{prefix}: legacy alignment output hash failure")
    if not benchmark.verify_hash_map(
        legacy_alignment / "formal_fit", legacy_formal.get("output_sha256", {})
    ):
        raise AssertionError(f"{prefix}: legacy formal output hash failure")

    old_order = tuple(legacy_formal["triplet_index_order"])
    if old_order[0] != hub or set(old_order) != set(expected_order):
        raise AssertionError(f"{prefix}: legacy triplet order drift")
    if tuple(source.get("legacy_triplet_index_order", ())) != old_order:
        raise AssertionError(f"{prefix}: recorded legacy triplet order drift")
    expected_fields = expected_fit_input_fields(legacy_formal["view_schema"])
    if tuple(formal_manifest.get("formal_fit_input_fields", ())) != tuple(
        sorted(expected_fields)
    ):
        raise AssertionError(f"{prefix}: outcome-free fit-input allowlist drift")

    fit_ids: dict[str, np.ndarray] = {}
    with np.load(source_files["fit_input_sha256"], allow_pickle=False) as archive:
        if set(archive.files) != expected_fields:
            raise AssertionError(f"{prefix}: unexpected field in formal fit input")
        for slug, schema in legacy_formal["view_schema"].items():
            position = int(schema["position"])
            fit_ids[slug] = np.asarray(
                strict_string_vector(
                    archive[f"view{position}_cell_ids"],
                    f"{prefix}/view{position}_cell_ids",
                ),
                dtype=np.str_,
            )

    old_indices, old_ids, old_scores = load_triplets(
        source_files["triplet_matching_sha256"], old_order
    )
    new_indices, new_ids, new_scores = load_triplets(
        formal / "triplet_matching.npz", expected_order
    )
    audit = formal_manifest["audit"]
    if manifest.get("formal_fit_audit") != audit:
        raise AssertionError(f"{prefix}: formal/alignment audit payload drift")
    policy = config["three_view_extension"]["outer_reuse_policy"]
    if (
        policy != legacy_formal["audit"].get("outer_reuse_policy")
        or policy != audit.get("outer_reuse_policy")
    ):
        raise AssertionError(f"{prefix}: outer-reuse policy drift")
    tuple_count = len(old_indices[hub])
    if int(audit["triplets"]) != tuple_count:
        raise AssertionError(f"{prefix}: triplet count drift")
    for slug in expected_order:
        old = old_indices[slug]
        new = new_indices[slug]
        if old.ndim != 1 or old.dtype.kind not in {"i", "u"}:
            raise AssertionError(f"{prefix}/{slug}: invalid legacy triplet indices")
        if np.any(old < 0) or np.any(old >= len(fit_ids[slug])):
            raise AssertionError(f"{prefix}/{slug}: legacy triplet index out of bounds")
        if not np.array_equal(old, new):
            raise AssertionError(f"{prefix}/{slug}: triplet remap drift")
        expected_ids = fit_ids[slug][old].astype(str)
        if not np.array_equal(old_ids[slug].astype(str), expected_ids):
            raise AssertionError(f"{prefix}/{slug}: legacy triplet ID drift")
        if not np.array_equal(new_ids[slug].astype(str), expected_ids):
            raise AssertionError(f"{prefix}/{slug}: replay triplet ID drift")
    for slug in expected_order[1:]:
        if not np.array_equal(old_scores[slug], new_scores[slug]):
            raise AssertionError(f"{prefix}/{slug}: pairwise score replay drift")
        reused = reused_records(new_indices[slug])
        multiplicity = maximum_multiplicity(new_indices[slug])
        if int(audit["outer_reused_records"][slug]) != reused:
            raise AssertionError(f"{prefix}/{slug}: reused-record audit drift")
        if int(audit["outer_maximum_multiplicity"][slug]) != multiplicity:
            raise AssertionError(f"{prefix}/{slug}: multiplicity audit drift")
        if policy == "drop_reused" and multiplicity > 1:
            raise AssertionError(f"{prefix}/{slug}: reuse violates configured policy")


def main() -> None:
    rows: list[dict[str, object]] = []
    for cancer in CANCERS:
        config_path = HERE / cancer / "maxfuse_archive" / "config.json"
        config, _ = benchmark.load_config(config_path)
        if int(config["expected_cells"]) != EXPECTED_CELLS[cancer]:
            raise AssertionError(f"{cancer}: unexpected canonical cell count")
        root = benchmark.output_root(config)
        slugs = [item["slug"] for item in config["datasets"]]
        names = {item["slug"]: item["name"] for item in config["datasets"]}
        expected_ids: np.ndarray | None = None
        for hub_position, hub in enumerate(slugs):
            expected_order = tuple(slugs[hub_position:] + slugs[:hub_position])
            alignment = root / f"hub_{hub}" / "alignment"
            alignment_manifest_path = alignment / "alignment_manifest.json"
            manifest = json.loads(alignment_manifest_path.read_text())
            if manifest["schema_version"] != benchmark.ALIGNMENT_SCHEMA:
                raise AssertionError(f"{cancer}/{hub}: wrong alignment schema")
            if manifest.get("labels_used_for_fitting") is not False:
                raise AssertionError(f"{cancer}/{hub}: labels entered fitting")
            if tuple(manifest["view_input_order"]) != expected_order:
                raise AssertionError(f"{cancer}/{hub}: cyclic order drift")
            if manifest["config_sha256"] != benchmark.sha256(config_path):
                raise AssertionError(f"{cancer}/{hub}: config hash drift")
            if not benchmark.verify_hash_map(alignment, manifest["outputs"]):
                raise AssertionError(f"{cancer}/{hub}: alignment output hash failure")

            formal = alignment / "formal_fit"
            formal_manifest_path = formal / "manifest.json"
            formal_manifest = json.loads(formal_manifest_path.read_text())
            if tuple(formal_manifest["triplet_index_order"]) != expected_order:
                raise AssertionError(f"{cancer}/{hub}: formal order drift")
            if not benchmark.verify_hash_map(formal, formal_manifest["output_sha256"]):
                raise AssertionError(f"{cancer}/{hub}: formal output hash failure")
            if manifest["formal_fit_manifest_sha256"] != benchmark.sha256(
                formal_manifest_path
            ):
                raise AssertionError(f"{cancer}/{hub}: formal manifest link drift")
            if cancer == "tnbc":
                if manifest.get("algorithm_runner_sha256") != benchmark.sha256(
                    ALGORITHM_DIR / "run_maxfuse_three_view_extension.py"
                ):
                    raise AssertionError(f"{cancer}/{hub}: current runner hash drift")
                if manifest.get("benchmark_runner_sha256") != benchmark.sha256(
                    BENCHMARK_HELPER
                ):
                    raise AssertionError(f"{cancer}/{hub}: benchmark runner hash drift")
                audit_current_pairwise_fit(
                    cancer,
                    hub,
                    expected_order,
                    config,
                    manifest,
                    alignment,
                    formal,
                    formal_manifest,
                )
                pairwise_provenance = "fitted_from_current_canonical_inputs"
                formal_input_fields = sorted(
                    expected_fit_input_fields(formal_manifest["view_schema"])
                )
            else:
                if manifest.get("pairwise_reused_without_refitting") is not True:
                    raise AssertionError(f"{cancer}/{hub}: pairwise reuse is not explicit")
                if manifest.get("pairwise_subprocess_invoked") is not False:
                    raise AssertionError(f"{cancer}/{hub}: pairwise subprocess status is unsafe")
                if manifest.get("legacy_ridge_embedding_or_model_used") is not False:
                    raise AssertionError(f"{cancer}/{hub}: legacy ridge result entered Archive")
                if manifest.get("outcomes_used_for_fitting") is not False:
                    raise AssertionError(f"{cancer}/{hub}: outcomes entered fitting")
                if manifest["replay_runner_sha256"] != benchmark.sha256(REPLAY_RUNNER):
                    raise AssertionError(f"{cancer}/{hub}: replay runner hash drift")
                if formal_manifest.get("pairwise_reused_without_refitting") is not True:
                    raise AssertionError(f"{cancer}/{hub}: formal pairwise reuse missing")
                if formal_manifest.get("pairwise_subprocess_invoked") is not False:
                    raise AssertionError(f"{cancer}/{hub}: formal pairwise subprocess status unsafe")
                if formal_manifest.get("legacy_ridge_embedding_or_model_used") is not False:
                    raise AssertionError(f"{cancer}/{hub}: formal legacy ridge status unsafe")
                if formal_manifest.get("labels_supplied_to_maxfuse_or_gcca") is not False:
                    raise AssertionError(f"{cancer}/{hub}: formal labels entered fitting")
                if formal_manifest.get("outcomes_supplied_to_maxfuse_or_gcca") is not False:
                    raise AssertionError(f"{cancer}/{hub}: formal outcomes entered fitting")
                source_hashes = formal_manifest.get("algorithm_source_sha256", {})
                expected_sources = algorithm_source_hashes(
                    (REPLAY_RUNNER.resolve(), BENCHMARK_HELPER.resolve())
                )
                if source_hashes != expected_sources:
                    raise AssertionError(f"{cancer}/{hub}: algorithm source provenance drift")
                audit_legacy_triplet_replay(
                    cancer,
                    hub,
                    expected_order,
                    config,
                    manifest,
                    formal,
                    formal_manifest,
                )
                pairwise_provenance = "replayed_from_frozen_maxfuse_v2_triplets"
                formal_input_fields = formal_manifest["formal_fit_input_fields"]

            audit = manifest["formal_fit_audit"]
            iterations = np.asarray(audit["gcca_iteration_counts"], dtype=int)
            objectives = np.asarray(audit["gcca_final_objectives"], dtype=float)
            if (
                len(iterations) != 15
                or np.any(iterations < 1)
                or np.any(iterations > 2000)
                or not np.isfinite(objectives).all()
            ):
                raise AssertionError(f"{cancer}/{hub}: invalid GCCA audit")
            roundtrip_max = max(
                float(value)
                for value in audit["projection_roundtrip_max_abs_error"].values()
            )
            if roundtrip_max > 1e-9:
                raise AssertionError(f"{cancer}/{hub}: projection roundtrip failure")

            embedding = np.load(
                alignment / "data" / "maxfuse_embedding_float64.npy",
                allow_pickle=False,
            )
            cell_ids = np.load(alignment / "data" / "cell_ids.npy", allow_pickle=False)
            metadata = pd.read_csv(
                alignment / "data" / "cell_metadata_label_free.csv.gz",
                low_memory=False,
            )
            if tuple(metadata.columns) != FIT_METADATA_COLUMNS:
                raise AssertionError(f"{cancer}/{hub}: fit metadata contains outcomes")
            expected_outcome_provenance = {
                "status": "not_present_in_any_fit_input",
                "formal_fit_input_fields": formal_input_fields,
                "fit_metadata_columns": list(FIT_METADATA_COLUMNS),
            }
            if cancer != "tnbc" and manifest.get("outcome_provenance") != expected_outcome_provenance:
                raise AssertionError(f"{cancer}/{hub}: outcome provenance drift")
            if embedding.shape != (manifest["full_cells"], 15):
                raise AssertionError(f"{cancer}/{hub}: embedding shape drift")
            if not np.isfinite(embedding).all():
                raise AssertionError(f"{cancer}/{hub}: embedding is non-finite")
            if not np.array_equal(
                metadata["cell_id"].astype(str).to_numpy(), cell_ids.astype(str)
            ):
                raise AssertionError(f"{cancer}/{hub}: alignment row-ID drift")
            if expected_ids is None:
                expected_ids = cell_ids.copy()
            elif not np.array_equal(expected_ids, cell_ids):
                raise AssertionError(f"{cancer}/{hub}: cell IDs differ across hubs")
            dataset_values = metadata["dataset"].astype(str).to_numpy()
            maximum_mean = 0.0
            maximum_sd_error = 0.0
            for slug in slugs:
                block = embedding[dataset_values == names[slug]]
                maximum_mean = max(maximum_mean, float(np.max(np.abs(block.mean(axis=0)))))
                maximum_sd_error = max(
                    maximum_sd_error,
                    float(np.max(np.abs(block.std(axis=0, ddof=0) - 1.0))),
                )
            if maximum_mean > 1e-10 or maximum_sd_error > 1e-10:
                raise AssertionError(f"{cancer}/{hub}: final Archive scaling drift")

            umap_dir = root / f"hub_{hub}" / "umap"
            umap_manifest_path = umap_dir / "umap_manifest.json"
            umap_manifest = json.loads(umap_manifest_path.read_text())
            if umap_manifest["schema_version"] != benchmark.UMAP_SCHEMA:
                raise AssertionError(f"{cancer}/{hub}: wrong UMAP schema")
            if not benchmark.verify_hash_map(umap_dir, umap_manifest["outputs"]):
                raise AssertionError(f"{cancer}/{hub}: UMAP output hash failure")
            if umap_manifest["alignment_manifest_sha256"] != benchmark.sha256(
                alignment_manifest_path
            ):
                raise AssertionError(f"{cancer}/{hub}: UMAP alignment link drift")
            coordinates = np.load(
                umap_dir / "data" / "umap_coordinates.npy", allow_pickle=False
            )
            if coordinates.shape != (manifest["full_cells"], 2) or not np.isfinite(
                coordinates
            ).all():
                raise AssertionError(f"{cancer}/{hub}: invalid UMAP coordinates")

            rows.append({
                "benchmark": cancer,
                "hub_slug": hub,
                "hub_dataset": names[hub],
                "cells": int(manifest["full_cells"]),
                "triplets": int(audit["triplets"]),
                "pairwise_provenance": pairwise_provenance,
                "view_order": " -> ".join(expected_order),
                "gcca_all_components_converged": bool(
                    audit["gcca_all_components_converged"]
                ),
                "gcca_components_at_iteration_limit": int(
                    np.sum(iterations == 2000)
                ),
                "gcca_max_iterations": int(iterations.max()),
                "training_fit_mean_matched_component_correlation": float(
                    audit["mean_matched_component_correlation"]
                ),
                "projection_roundtrip_max_abs_error": roundtrip_max,
                "final_scaling_max_abs_mean": maximum_mean,
                "final_scaling_max_abs_sd_error": maximum_sd_error,
                "alignment_hashes": "PASS",
                "umap_hashes": "PASS",
            })

        summary = root / "summary"
        summary_manifest = json.loads((summary / "summary_manifest.json").read_text())
        if summary_manifest["schema_version"] != "final_xu.maxfuse_all_hubs_summary.v2":
            raise AssertionError(f"{cancer}: wrong summary schema")
        if not benchmark.verify_hash_map(summary, summary_manifest["outputs"]):
            raise AssertionError(f"{cancer}: summary output hash failure")

    frame = pd.DataFrame(rows)
    components_at_limit = int(frame["gcca_components_at_iteration_limit"].sum())
    converged_components = int(len(frame) * 15 - components_at_limit)
    csv_path = HERE / "MAXFUSE_ARCHIVE_CURRENT_AUDIT.csv"
    md_path = HERE / "MAXFUSE_ARCHIVE_CURRENT_AUDIT.md"
    frame.to_csv(csv_path, index=False)
    lines = [
        "# MaxFuse Archive current audit",
        "",
        "Status: **PASS**",
        "",
        "- 4 benchmark cohorts; 12 cyclic hub/order runs.",
        "- TNBC uses 62,701 canonical cells and refits official pairwise MaxFuse "
        "from the current provider-full inputs for every hub orientation.",
        "- Bone marrow, melanoma, and lung cancer retain provenance-locked, "
        "label/outcome-free replay of their frozen MaxFuse-v2 pairwise triplets.",
        "- No historical ridge-GCCA embedding/model enters any current Archive result.",
        "- Canonical algorithm entrypoints, configurations, inputs, triplet cell IDs, "
        "and outputs are provenance-locked under the applicable fit/replay route.",
        f"- Archive GCCA: {converged_components}/{len(frame) * 15} components met the "
        "tolerance; the remaining components returned at the official 2,000-iteration cap.",
        "- Alignment, projection, row identity, scaling, UMAP, and summary hashes passed.",
        "- Matched-component correlation is retained only as a training-fit diagnostic.",
        "",
        "```text",
        frame.to_string(index=False),
        "```",
        "",
    ]
    md_path.write_text("\n".join(lines))
    print(f"MAXFUSE_ARCHIVE_CURRENT_AUDIT_OK rows={len(frame)}")
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
