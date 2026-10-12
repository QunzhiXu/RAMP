#!/usr/bin/env python3
"""Exhaustive marker-value artifact audit across the four frozen benchmarks.

Primary summaries pool all donors within each dataset, as requested.  Values
are converted to within-dataset midrank percentiles before methods are compared,
while literal native-scale means and box statistics are retained separately.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

CACHE = Path(
    os.environ.get(
        "RAMP_ARTIFACT_CACHE",
        str(Path(tempfile.gettempdir()) / "ramp-exact-value-artifact-cache"),
    )
)
for variable, subdirectory in (
    ("MPLCONFIGDIR", "matplotlib"),
    ("XDG_CACHE_HOME", "xdg"),
):
    destination = CACHE / subdirectory
    destination.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(variable, str(destination))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
BENCHMARK_RESULTS = HERE.parents[2]
FINAL_XU = HERE.parents[3]
LEGACY_ARTIFACTS = FINAL_XU / "test0827" / "lineage_marker_artifacts"
EXPECTATIONS_PATH = ROOT / "config" / "canonical_expectations.csv"
SUMMARY = ROOT / "summary"
CASES = ("bone_marrow", "tnbc", "melanoma", "lung_cancer")
MIN_CELLS = 20
EXHAUSTIVE_MIN_CELLS = 50
INVALID_CELLTYPE = re.compile(
    r"non-joint|unclassified|technical|unannotated|unknown|ambiguous|\bmix\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class MethodData:
    name: str
    family: str
    variant: str
    role: str
    input_values: np.ndarray
    corrected_values: np.ndarray
    input_path: Path
    output_path: Path
    interpretation: str


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def read_markers(path: Path) -> list[str]:
    markers = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not markers or len(markers) != len(set(markers)):
        raise ValueError(f"invalid markers: {path}")
    return markers


def read_cell_id_csv(path: Path, ids: np.ndarray, markers: list[str]) -> np.ndarray:
    frame = pd.read_csv(path, low_memory=False)
    if "cell_id" not in frame:
        raise ValueError(f"{path}: missing cell_id")
    observed = frame["cell_id"].astype(str).to_numpy()
    if not np.array_equal(observed, ids):
        raise ValueError(f"{path}: cell order mismatch")
    if missing := [marker for marker in markers if marker not in frame]:
        raise ValueError(f"{path}: missing marker columns {missing}")
    values = frame[markers].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite marker values")
    return values


def read_row_id_csv(path: Path, n_cells: int, markers: list[str]) -> np.ndarray:
    frame = pd.read_csv(path, low_memory=False)
    if "row_id" not in frame:
        raise ValueError(f"{path}: missing row_id")
    frame = frame.sort_values("row_id", kind="stable")
    if not np.array_equal(frame["row_id"].to_numpy(), np.arange(n_cells)):
        raise ValueError(f"{path}: row_id mismatch")
    if missing := [marker for marker in markers if marker not in frame]:
        raise ValueError(f"{path}: missing marker columns {missing}")
    values = frame[markers].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite marker values")
    return values


def midrank_percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)
    return (ranks / (len(values) + 1.0)).astype(np.float32)


def rank_within_groups(matrix: np.ndarray, groups: np.ndarray) -> np.ndarray:
    ranked = np.full(matrix.shape, np.nan, dtype=np.float32)
    for group in pd.unique(groups):
        rows = np.flatnonzero(groups == group)
        for marker_index in range(matrix.shape[1]):
            ranked[rows, marker_index] = midrank_percentile(matrix[rows, marker_index])
    return ranked


def auc_positive_greater(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    if not len(positive) or not len(negative):
        return float("nan")
    pooled = np.concatenate([positive, negative])
    ranks = pd.Series(pooled).rank(method="average").to_numpy(dtype=float)
    n_positive = len(positive)
    rank_sum = float(ranks[:n_positive].sum())
    u_statistic = rank_sum - n_positive * (n_positive + 1) / 2.0
    return u_statistic / (n_positive * len(negative))


def split_labels(value: object) -> list[str]:
    return [item.strip() for item in str(value).split(";") if item.strip()]


def allowed_datasets(value: object, available: np.ndarray) -> list[str]:
    if pd.isna(value) or str(value).strip() in {"", "*"}:
        return [str(item) for item in pd.unique(available)]
    allowed = set(split_labels(value))
    return [str(item) for item in pd.unique(available) if str(item) in allowed]


def classify(row: dict[str, object]) -> str:
    raw_auc = float(row["raw_target_vs_nontarget_auc"])
    if raw_auc < 0.70:
        return "raw specificity insufficient"
    raw_target_difference = float(row["raw_target_minus_nontarget_mean_rank"])
    target_difference = float(row["corrected_target_minus_nontarget_mean_rank"])
    raw_non = float(row["raw_nontarget_mean_rank"])
    corrected_non = float(row["corrected_nontarget_mean_rank"])
    non_shift = float(row["nontarget_mean_rank_shift"])
    raw_target = float(row["raw_target_mean_rank"])
    corrected_target = float(row["corrected_target_mean_rank"])
    target_shift = float(row["target_mean_rank_shift"])
    if (
        raw_target_difference >= 0.10
        and target_difference <= -0.05
        and raw_target_difference - target_difference >= 0.20
    ):
        return "target/non-target mean-order reversal"
    if raw_non <= 0.40 and corrected_non >= 0.60 and non_shift >= 0.20:
        return "clear non-target low-to-high"
    if raw_target >= 0.60 and corrected_target <= 0.40 and target_shift <= -0.20:
        return "clear expected-target high-to-low"
    if raw_non <= 0.45 and corrected_non >= 0.55 and non_shift >= 0.15:
        return "strong non-target intrusion"
    if non_shift >= 0.10:
        return "non-target upward shift"
    if target_shift <= -0.10:
        return "expected-target downward shift"
    if target_difference < 0:
        return "weak mean-order crossing"
    return "no artifact hit"


def is_hit(classification: str) -> bool:
    return classification in {
        "target/non-target mean-order reversal",
        "clear non-target low-to-high",
        "clear expected-target high-to-low",
        "strong non-target intrusion",
    }


def strict_order_inversion(before: np.ndarray, after: np.ndarray) -> tuple[bool, float]:
    """Detect x_i < x_j but y_i > y_j, while ignoring ties in x.

    Midrank Spearman can move when an input tie is split, even if every strict
    unequal order is retained.  This monotonicity diagnostic therefore audits
    the precise strict-order guarantee separately from correlation summaries.
    """
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    order = np.argsort(before, kind="stable")
    before = before[order]
    after = after[order]
    previous_max = -np.inf
    maximum_margin = 0.0
    start = 0
    while start < len(before):
        stop = start + 1
        while stop < len(before) and before[stop] == before[start]:
            stop += 1
        if np.isfinite(previous_max):
            maximum_margin = max(maximum_margin, previous_max - float(np.min(after[start:stop])))
        previous_max = max(previous_max, float(np.max(after[start:stop])))
        start = stop
    return bool(maximum_margin > 1e-12), float(max(0.0, maximum_margin))


def require_sha256(path: Path, expected: str, context: str) -> None:
    observed = sha256_file(path)
    if not expected or observed != expected:
        raise ValueError(
            f"{context}: SHA256 mismatch for {path}; expected={expected}, observed={observed}"
        )


def load_case(case: str) -> tuple[pd.DataFrame, list[str], np.ndarray, list[MethodData], list[dict[str, object]]]:
    case_root = BENCHMARK_RESULTS / case
    input_root = case_root / "ramp" / "alignment" / "input"
    metadata_path = input_root / "cell_metadata.csv.gz"
    marker_path = input_root / "markers.txt"
    raw_path = input_root / "raw.npy"
    metadata = pd.read_csv(metadata_path, low_memory=False)
    ids = metadata["cell_id"].astype(str).to_numpy()
    if len(ids) != len(set(ids)):
        raise ValueError(f"{case}: duplicate cell IDs")
    markers = read_markers(marker_path)
    raw = np.asarray(np.load(raw_path, mmap_mode="r"), dtype=np.float64)
    if raw.shape != (len(metadata), len(markers)) or not np.isfinite(raw).all():
        raise ValueError(f"{case}: invalid raw matrix")

    methods: list[MethodData] = []
    provenance: list[dict[str, object]] = []

    def add(
        name: str,
        family: str,
        variant: str,
        role: str,
        input_values: np.ndarray,
        corrected_values: np.ndarray,
        input_path: Path,
        output_path: Path,
        interpretation: str,
        authentication: str,
    ) -> None:
        for label, values in (("input", input_values), ("output", corrected_values)):
            if values.shape != raw.shape or not np.isfinite(values).all():
                raise ValueError(f"{case}/{name}: invalid {label} matrix {values.shape}")
        methods.append(
            MethodData(
                name,
                family,
                variant,
                role,
                input_values,
                corrected_values,
                input_path,
                output_path,
                interpretation,
            )
        )
        provenance.append(
            {
                "benchmark": case,
                "method": name,
                "family": family,
                "variant": variant,
                "role": role,
                "input_path": str(input_path.resolve()),
                "input_sha256": sha256_file(input_path),
                "output_path": str(output_path.resolve()),
                "output_sha256": sha256_file(output_path),
                "n_cells": len(metadata),
                "n_markers": len(markers),
                "interpretation": interpretation,
                "authentication": authentication,
            }
        )

    ramp_data_root = case_root / "ramp" / "alignment" / "data"
    ramp_output = (
        ramp_data_root / "ramp_aligned_float64.npy"
        if (ramp_data_root / "ramp_aligned_float64.npy").exists()
        else ramp_data_root / "ramp_aligned.npy"
    )
    ramp = np.asarray(np.load(ramp_output, mmap_mode="r"), dtype=np.float64)
    ramp_input = (
        ramp_data_root / "kmeanslow_global_float64.npy"
        if (ramp_data_root / "kmeanslow_global_float64.npy").exists()
        else ramp_data_root / "kmeanslow_global.npy"
    )
    ramp_input_values = np.asarray(np.load(ramp_input, mmap_mode="r"), dtype=np.float64)
    ramp_run_manifest = json.loads(
        (case_root / "ramp" / "alignment" / "alignment_manifest.json").read_text()
    )
    ramp_outputs = ramp_run_manifest.get("outputs", {})
    require_sha256(ramp_input, ramp_outputs.get(ramp_input.name, ""), f"{case}/RAMP input")
    require_sha256(ramp_output, ramp_outputs.get(ramp_output.name, ""), f"{case}/RAMP output")
    if int(ramp_run_manifest.get("strict_rank_inversions", -1)) != 0:
        raise ValueError(f"{case}/RAMP formal manifest does not report zero strict inversions")
    add(
        "RAMP",
        "RAMP",
        "ver2",
        "rank-preserving control",
        ramp_input_values,
        ramp,
        ramp_input,
        ramp_output,
        "Formal rank-preserving aligned marker matrix.",
        "Input/output SHA256 and zero-strict-inversion status verified against alignment_manifest.json.",
    )

    cy_input_path = case_root / "cycombine" / "preprocessing" / "data" / "combined_analysis_ready.csv.gz"
    cy_output_path = case_root / "cycombine" / "alignment" / "data" / "corrected_expression.csv.gz"
    cy_manifest = json.loads(
        (case_root / "cycombine" / "alignment" / "data" / "manifest.json").read_text()
    )
    require_sha256(
        cy_input_path,
        cy_manifest.get("inputs", {}).get("expression", {}).get("sha256", ""),
        f"{case}/cyCombine input",
    )
    require_sha256(
        cy_output_path,
        cy_manifest.get("outputs", {}).get("corrected_expression_csv", {}).get("sha256", ""),
        f"{case}/cyCombine output",
    )
    add(
        "cyCombine",
        "cyCombine",
        "formal",
        "exact corrected marker values",
        read_cell_id_csv(cy_input_path, ids, markers),
        read_cell_id_csv(cy_output_path, ids, markers),
        cy_input_path,
        cy_output_path,
        "Formal cyCombine corrected expression output.",
        "Input/output SHA256 verified against formal cyCombine run manifest.",
    )

    for variant, name in (
        ("source_scale", "Harmony-direct[source_scale]"),
        ("adtnorm_fixed_arcsinh", "Harmony-direct[fixed_arcsinh]"),
    ):
        harmony_input = (
            case_root / "harmony" / variant / "preprocessing" / "data" / "combined_analysis_ready.csv.gz"
        )
        harmony_output = (
            case_root
            / "harmony"
            / variant
            / "alignment"
            / "data"
            / "harmony_adjusted_marker_coordinates.csv.gz"
        )
        harmony_manifest = json.loads(
            (case_root / "harmony" / variant / "alignment" / "data" / "manifest.json").read_text()
        )
        if harmony_manifest.get("status") != "complete":
            raise ValueError(f"{case}/{name}: formal Harmony manifest is not complete")
        require_sha256(
            harmony_input,
            harmony_manifest.get("input", {}).get("sha256", ""),
            f"{case}/{name} input",
        )
        harmony_outputs = {
            item.get("file"): item.get("sha256")
            for item in harmony_manifest.get("output_files", [])
        }
        require_sha256(
            harmony_output,
            harmony_outputs.get(harmony_output.name, ""),
            f"{case}/{name} output",
        )
        add(
            name,
            "Harmony-direct",
            variant,
            "exact adjusted marker-axis coordinates",
            read_cell_id_csv(harmony_input, ids, markers),
            read_cell_id_csv(harmony_output, ids, markers),
            harmony_input,
            harmony_output,
            "ADTnorm-style direct-marker Harmony without PCA; not canonical PCA-Harmony.",
            "Complete status and input/output SHA256 verified against formal Harmony-direct manifest.",
        )

    fast_input_path = LEGACY_ARTIFACTS / case / "methods" / "standardized_input.csv.gz"
    fast_output_path = LEGACY_ARTIFACTS / case / "methods" / "fastMNN.csv.gz"
    legacy_manifest_path = LEGACY_ARTIFACTS / case / "case_manifest.json"
    legacy_manifest = json.loads(legacy_manifest_path.read_text())
    if Path(legacy_manifest["raw_source"]).resolve() != raw_path.resolve():
        raise ValueError(f"{case}: legacy d=NA raw source is not the frozen benchmark raw matrix")
    if Path(legacy_manifest["metadata_source"]).resolve() != metadata_path.resolve():
        raise ValueError(f"{case}: legacy d=NA metadata source is not the frozen benchmark metadata")
    expected_hashes = legacy_manifest.get("sha256", {})
    current_hashes = {
        "raw": sha256_file(raw_path),
        "metadata": sha256_file(metadata_path),
        "markers": sha256_file(marker_path),
    }
    if any(expected_hashes.get(key) != value for key, value in current_hashes.items()):
        raise ValueError(f"{case}: legacy d=NA manifest hashes no longer match frozen inputs")
    if legacy_manifest.get("markers") != markers or legacy_manifest.get("cells") != len(metadata):
        raise ValueError(f"{case}: legacy d=NA manifest shape/marker contract mismatch")
    add(
        "fastMNN-marker[d=NA]",
        "fastMNN",
        "artifact_only_d_NA",
        "exact marker-axis stress-test output",
        read_row_id_csv(fast_input_path, len(metadata), markers),
        read_row_id_csv(fast_output_path, len(metadata), markers),
        fast_input_path,
        fast_output_path,
        "Non-default small-panel d=NA stress test; separate from the formal PCA-fastMNN benchmark embedding.",
        "Frozen raw/metadata/marker hashes and row order verified; stress-test output hash recorded here because the legacy run did not predeclare an output hash.",
    )

    provenance.extend(
        [
            {
                "benchmark": case,
                "method": "MaxFuse",
                "role": "excluded",
                "interpretation": "Latent embedding only; no exact marker-axis values.",
            },
            {
                "benchmark": case,
                "method": "fastMNN formal",
                "role": "excluded from marker audit",
                "interpretation": "Formal output is a PCA corrected embedding; marker_level_interpretation=not_applicable.",
            },
            {
                "benchmark": case,
                "method": "CytofRUV",
                "role": "NOT_APPLICABLE",
                "interpretation": "No true cross-batch replicated reference specimen.",
            },
            {
                "benchmark": case,
                "method": "ADTnorm",
                "role": "not run as a comparator",
                "interpretation": "The ADTnorm paper motivates the direct-marker artifact audit; ADTnorm itself was not part of the agreed cross-technology comparator set.",
            },
            {
                "benchmark": case,
                "method": "legacy HarmonyMarker",
                "role": "superseded/excluded",
                "interpretation": "Earlier dataset-z-score/no-PCA stress-test output used for the first CD68 figure; replaced here by the two formal Harmony-direct benchmark outputs.",
            },
        ]
    )
    return metadata, markers, raw, methods, provenance


def contrast_row(
    case: str,
    dataset: str,
    marker: str,
    target_label: str,
    non_target_label: str,
    target_mask: np.ndarray,
    non_target_mask: np.ndarray,
    raw: np.ndarray,
    raw_rank: np.ndarray,
    method: MethodData,
    input_rank: np.ndarray,
    corrected_rank: np.ndarray,
    marker_index: int,
    rule_id: str,
    tier: str,
    discovery_scope: str,
) -> dict[str, object]:
    raw_target_rank = float(raw_rank[target_mask, marker_index].mean())
    raw_non_rank = float(raw_rank[non_target_mask, marker_index].mean())
    input_target_rank = float(input_rank[target_mask, marker_index].mean())
    input_non_rank = float(input_rank[non_target_mask, marker_index].mean())
    corrected_target_rank = float(corrected_rank[target_mask, marker_index].mean())
    corrected_non_rank = float(corrected_rank[non_target_mask, marker_index].mean())
    row: dict[str, object] = {
        "case": case,
        "dataset": dataset,
        "discovery_scope": discovery_scope,
        "rule_id": rule_id,
        "tier": tier,
        "marker": marker,
        "expected_target": target_label,
        "non_target_celltype": non_target_label,
        "method": method.name,
        "method_family": method.family,
        "variant": method.variant,
        "n_target_cells": int(target_mask.sum()),
        "n_non_target_cells": int(non_target_mask.sum()),
        "raw_target_vs_nontarget_auc": auc_positive_greater(
            raw[target_mask, marker_index], raw[non_target_mask, marker_index]
        ),
        "input_target_vs_nontarget_auc": auc_positive_greater(
            method.input_values[target_mask, marker_index],
            method.input_values[non_target_mask, marker_index],
        ),
        "corrected_target_vs_nontarget_auc": auc_positive_greater(
            method.corrected_values[target_mask, marker_index],
            method.corrected_values[non_target_mask, marker_index],
        ),
        "raw_target_mean_rank": raw_target_rank,
        "input_target_mean_rank": input_target_rank,
        "corrected_target_mean_rank": corrected_target_rank,
        "target_preprocessing_rank_shift": input_target_rank - raw_target_rank,
        "target_correction_rank_shift": corrected_target_rank - input_target_rank,
        "target_mean_rank_shift": corrected_target_rank - raw_target_rank,
        "raw_nontarget_mean_rank": raw_non_rank,
        "input_nontarget_mean_rank": input_non_rank,
        "corrected_nontarget_mean_rank": corrected_non_rank,
        "nontarget_preprocessing_rank_shift": input_non_rank - raw_non_rank,
        "nontarget_correction_rank_shift": corrected_non_rank - input_non_rank,
        "nontarget_mean_rank_shift": corrected_non_rank - raw_non_rank,
        "raw_target_minus_nontarget_mean_rank": raw_target_rank - raw_non_rank,
        "input_target_minus_nontarget_mean_rank": input_target_rank - input_non_rank,
        "corrected_target_minus_nontarget_mean_rank": corrected_target_rank - corrected_non_rank,
        "raw_target_native_mean": float(raw[target_mask, marker_index].mean()),
        "raw_nontarget_native_mean": float(raw[non_target_mask, marker_index].mean()),
        "input_target_native_mean": float(method.input_values[target_mask, marker_index].mean()),
        "input_nontarget_native_mean": float(method.input_values[non_target_mask, marker_index].mean()),
        "corrected_target_native_mean": float(method.corrected_values[target_mask, marker_index].mean()),
        "corrected_nontarget_native_mean": float(method.corrected_values[non_target_mask, marker_index].mean()),
    }
    row["threshold_classification"] = classify(row)
    row["pooled_threshold_hit"] = is_hit(str(row["threshold_classification"]))
    row["is_rank_preserving_control"] = method.name == "RAMP"
    row["control_diagnostic_event"] = bool(
        row["is_rank_preserving_control"] and row["pooled_threshold_hit"]
    )
    row["classification"] = (
        "pooled control threshold event (not a within-unit inversion): "
        + str(row["threshold_classification"])
        if row["control_diagnostic_event"]
        else str(row["threshold_classification"])
    )
    row["artifact_hit"] = bool(
        row["pooled_threshold_hit"] and not row["is_rank_preserving_control"]
    )
    correction_magnitude = abs(float(row["nontarget_correction_rank_shift"])) + abs(
        float(row["target_correction_rank_shift"])
    )
    preprocessing_magnitude = abs(float(row["nontarget_preprocessing_rank_shift"])) + abs(
        float(row["target_preprocessing_rank_shift"])
    )
    row["dominant_stage"] = (
        "correction"
        if correction_magnitude > preprocessing_magnitude + 1e-12
        else "preprocessing"
        if preprocessing_magnitude > correction_magnitude + 1e-12
        else "tied/no-change"
    )
    row["artifact_severity"] = max(
        0.0,
        float(row["nontarget_mean_rank_shift"]),
        -float(row["target_mean_rank_shift"]),
        -float(row["corrected_target_minus_nontarget_mean_rank"]),
    )
    return row


def box_statistics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    lower_fence = float(q1 - 1.5 * iqr)
    upper_fence = float(q3 + 1.5 * iqr)
    low = float(np.min(values[values >= lower_fence]))
    high = float(np.max(values[values <= upper_fence]))
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "whisker_low": low,
        "whisker_high": high,
    }


def plot_box_grid(
    case: str,
    dataset: str,
    marker: str,
    target_labels: list[str],
    highlight_non_target: str,
    metadata: pd.DataFrame,
    raw: np.ndarray,
    methods: list[MethodData],
    markers: list[str],
    display_scale: str = "percentile",
    output_png: Path | None = None,
    pdf: PdfPages | None = None,
) -> list[dict[str, object]]:
    if display_scale not in {"percentile", "native"}:
        raise ValueError("display_scale must be percentile or native")
    marker_index = markers.index(marker)
    datasets = metadata["dataset"].astype(str).to_numpy()
    celltypes = metadata["celltype"].fillna("Missing").astype(str).to_numpy()
    mask = datasets == dataset
    counts = pd.Series(celltypes[mask]).value_counts()
    labels = [label for label, count in counts.items() if count >= MIN_CELLS]
    labels = sorted(
        labels,
        key=lambda label: (
            0 if label in target_labels else 1 if label == highlight_non_target else 2,
            label,
        ),
    )
    display_order = {
        "Harmony-direct[source_scale]": 0,
        "Harmony-direct[fixed_arcsinh]": 1,
        "fastMNN-marker[d=NA]": 2,
        "cyCombine": 3,
        "RAMP": 4,
    }
    methods = sorted(methods, key=lambda method: display_order.get(method.name, 99))
    panels = [("Raw", raw)] + [(method.name, method.corrected_values) for method in methods]
    if display_scale == "percentile":
        panels = [(label, rank_within_groups(values, datasets)) for label, values in panels]
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(17, 9.5),
        squeeze=False,
        sharey=display_scale == "percentile",
    )
    summary_rows: list[dict[str, object]] = []
    for ax, (panel, values) in zip(axes.ravel(), panels):
        stats = []
        means = []
        colors = []
        for label in labels:
            local = values[mask & (celltypes == label), marker_index]
            record = box_statistics(local)
            stats.append(
                {
                    "label": label,
                    "med": record["median"],
                    "q1": record["q1"],
                    "q3": record["q3"],
                    "whislo": record["whisker_low"],
                    "whishi": record["whisker_high"],
                    "fliers": [],
                }
            )
            means.append(record["mean"])
            colors.append(
                "#2563A6"
                if label in target_labels
                else "#D64545"
                if label == highlight_non_target
                else "#B8BDC5"
            )
            summary_rows.append(
                {
                    "case": case,
                    "dataset": dataset,
                    "marker": marker,
                    "output": panel,
                    "display_scale": display_scale,
                    "celltype": label,
                    **record,
                }
            )
        artists = ax.bxp(stats, showfliers=False, patch_artist=True, widths=0.68)
        for patch, color in zip(artists["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.82)
            patch.set_edgecolor("#4B5563")
        ax.scatter(
            np.arange(1, len(labels) + 1),
            means,
            marker="D",
            s=24,
            facecolor="white",
            edgecolor="#111827",
            linewidth=0.8,
            zorder=4,
        )
        ax.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=42, ha="right", fontsize=8)
        ax.set_title(panel, fontsize=11.5, fontweight="bold")
        ax.set_ylabel(
            f"{marker} within-dataset percentile rank"
            if display_scale == "percentile"
            else f"{marker} (native output scale)"
        )
        if display_scale == "percentile":
            ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")
    figure.suptitle(
        f"{case} · {dataset} · {marker}\n"
        f"blue: expected-high target; red: selected non-target; white diamond: pooled mean",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=[0, 0.02, 1, 0.95])
    if output_png is not None:
        figure.savefig(output_png, dpi=240, bbox_inches="tight")
    if pdf is not None:
        pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)
    return summary_rows


def make_heatmap(canonical: pd.DataFrame, output: Path, methods: list[str]) -> None:
    candidates = canonical[
        (canonical["method"] != "RAMP")
        & (canonical["raw_target_vs_nontarget_auc"] >= 0.70)
    ].copy()
    candidates["priority"] = np.where(candidates["artifact_hit"], 0, 1)
    keys = (
        candidates.sort_values(
            ["priority", "artifact_severity", "nontarget_mean_rank_shift"],
            ascending=[True, False, False],
        )
        .drop_duplicates(["case", "dataset", "marker", "expected_target", "non_target_celltype"])
        .head(30)[["case", "dataset", "marker", "expected_target", "non_target_celltype"]]
    )
    key_tuples = list(keys.itertuples(index=False, name=None))
    matrix = np.full((len(key_tuples), len(methods)), np.nan)
    annotations = np.full(matrix.shape, "", dtype=object)
    for row_index, key in enumerate(key_tuples):
        case, dataset, marker, target, non_target = key
        block = canonical[
            (canonical["case"] == case)
            & (canonical["dataset"] == dataset)
            & (canonical["marker"] == marker)
            & (canonical["expected_target"] == target)
            & (canonical["non_target_celltype"] == non_target)
        ]
        for column_index, method in enumerate(methods):
            match = block[block["method"] == method]
            if not len(match):
                continue
            row = match.iloc[0]
            matrix[row_index, column_index] = row["nontarget_mean_rank_shift"]
            reversal = " R" if bool(row["artifact_hit"]) and "reversal" in row["classification"] else ""
            annotations[row_index, column_index] = (
                f"{row['nontarget_mean_rank_shift']:+.2f}\n"
                f"→{row['corrected_nontarget_mean_rank']:.2f}{reversal}"
            )
    height = max(8.0, 0.48 * len(key_tuples) + 2.5)
    figure, ax = plt.subplots(figsize=(13.5, height))
    image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="RdBu_r", vmin=-0.4, vmax=0.4)
    ax.set_xticks(np.arange(len(methods)), methods, rotation=25, ha="right", fontsize=9)
    labels = [f"{case} | {dataset} | {marker} in {non_target}" for case, dataset, marker, _, non_target in key_tuples]
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if not annotations[row, column]:
                continue
            color = "white" if abs(matrix[row, column]) >= 0.22 else "#111827"
            ax.text(column, row, annotations[row, column], ha="center", va="center", fontsize=7.5, color=color)
    ax.tick_params(length=0)
    ax.set_title(
        "Canonical lineage-marker artifact scan\n"
        "cell text: pooled non-target mean-rank shift; arrow: corrected mean rank; R: order reversal",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    colorbar = figure.colorbar(image, ax=ax, fraction=0.028, pad=0.02)
    colorbar.set_label("Raw → corrected non-target pooled mean-rank shift")
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 20) -> str:
    local = frame.loc[:, columns].head(limit).copy()
    for column in local.select_dtypes(include=["float", "float64", "float32"]):
        local[column] = local[column].map(lambda value: f"{value:.3f}" if pd.notna(value) else "NA")
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in local.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def main() -> None:
    SUMMARY.mkdir(parents=True, exist_ok=True)
    completion_manifest = ROOT / "manifest.json"
    incomplete_marker = ROOT / "RUN_INCOMPLETE"
    completion_manifest.unlink(missing_ok=True)
    incomplete_marker.write_text(
        "Artifact audit is running or failed before the completion manifest was written.\n"
    )
    expectations = pd.read_csv(EXPECTATIONS_PATH)
    required_expectation_columns = {
        "cancer",
        "rule_id",
        "marker",
        "positive_labels",
        "negative_labels",
        "dataset_allowlist",
        "tier",
        "rationale",
    }
    if missing := sorted(required_expectation_columns - set(expectations.columns)):
        raise ValueError(f"canonical expectation file is missing columns: {missing}")
    if expectations["rule_id"].astype(str).duplicated().any():
        raise ValueError("canonical expectation file has duplicate rule_id values")
    if not set(expectations["cancer"].astype(str)).issubset(CASES):
        raise ValueError("canonical expectation file contains an unknown benchmark")
    canonical_rows: list[dict[str, object]] = []
    exhaustive_rows: list[dict[str, object]] = []
    integrity_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    resolution_rows: list[dict[str, object]] = []
    for case in CASES:
        print(f"[{case}] loading exact marker-value outputs", flush=True)
        metadata, markers, raw, methods, provenance = load_case(case)
        provenance_rows.extend(provenance)
        datasets = metadata["dataset"].astype(str).to_numpy()
        units = metadata["unit_key"].astype(str).to_numpy()
        celltypes = metadata["celltype"].fillna("Missing").astype(str).to_numpy()
        raw_dataset_rank = rank_within_groups(raw, datasets)
        marker_index = {marker: index for index, marker in enumerate(markers)}
        case_expectations = expectations[expectations["cancer"] == case]

        available_datasets = [str(value) for value in pd.unique(datasets)]
        for method_number, method in enumerate(methods):
            print(f"[{case}] auditing {method.name}", flush=True)
            input_dataset_rank = rank_within_groups(method.input_values, datasets)
            corrected_dataset_rank = rank_within_groups(method.corrected_values, datasets)
            input_unit_rank = rank_within_groups(method.input_values, units)
            corrected_unit_rank = rank_within_groups(method.corrected_values, units)
            for unit in pd.unique(units):
                unit_rows = np.flatnonzero(units == unit)
                for column, marker in enumerate(markers):
                    before = input_unit_rank[unit_rows, column]
                    after = corrected_unit_rank[unit_rows, column]
                    rho = spearmanr(before, after).statistic
                    shift = after - before
                    inversion_present, inversion_margin = strict_order_inversion(
                        method.input_values[unit_rows, column],
                        method.corrected_values[unit_rows, column],
                    )
                    integrity_rows.append(
                        {
                            "case": case,
                            "method": method.name,
                            "dataset": datasets[unit_rows[0]],
                            "unit_key": unit,
                            "marker": marker,
                            "n_cells": len(unit_rows),
                            "spearman_input_vs_output": rho,
                            "mean_absolute_rank_shift": float(np.mean(np.abs(shift))),
                            "maximum_absolute_rank_shift": float(np.max(np.abs(shift))),
                            "fraction_abs_rank_shift_ge_0_20": float(np.mean(np.abs(shift) >= 0.20)),
                            "strict_order_inversion_present": inversion_present,
                            "maximum_strict_order_inversion_margin": inversion_margin,
                        }
                    )

            for rule in case_expectations.itertuples(index=False):
                targets = split_labels(rule.positive_labels)
                non_targets = split_labels(rule.negative_labels)
                requested_datasets = (
                    available_datasets
                    if pd.isna(rule.dataset_allowlist)
                    or str(rule.dataset_allowlist).strip() in {"", "*"}
                    else split_labels(rule.dataset_allowlist)
                )
                if rule.marker not in marker_index:
                    if method_number == 0:
                        for dataset in requested_datasets:
                            for non_target in non_targets:
                                resolution_rows.append(
                                    {
                                        "case": case,
                                        "rule_id": rule.rule_id,
                                        "dataset": dataset,
                                        "marker": rule.marker,
                                        "expected_target": rule.positive_labels,
                                        "non_target_celltype": non_target,
                                        "status": "skipped",
                                        "reason": "marker_not_shared",
                                        "n_target_cells": 0,
                                        "n_non_target_cells": 0,
                                    }
                                )
                    continue
                column = marker_index[rule.marker]
                for dataset in requested_datasets:
                    if dataset not in available_datasets:
                        if method_number == 0:
                            for non_target in non_targets:
                                resolution_rows.append(
                                    {
                                        "case": case,
                                        "rule_id": rule.rule_id,
                                        "dataset": dataset,
                                        "marker": rule.marker,
                                        "expected_target": rule.positive_labels,
                                        "non_target_celltype": non_target,
                                        "status": "skipped",
                                        "reason": "dataset_not_present",
                                        "n_target_cells": 0,
                                        "n_non_target_cells": 0,
                                    }
                                )
                        continue
                    dataset_mask = datasets == dataset
                    target_mask = dataset_mask & np.isin(celltypes, targets)
                    n_target = int(target_mask.sum())
                    for non_target in non_targets:
                        non_target_mask = dataset_mask & (celltypes == non_target)
                        n_non_target = int(non_target_mask.sum())
                        if n_target < MIN_CELLS or n_non_target < MIN_CELLS:
                            if method_number == 0:
                                resolution_rows.append(
                                    {
                                        "case": case,
                                        "rule_id": rule.rule_id,
                                        "dataset": dataset,
                                        "marker": rule.marker,
                                        "expected_target": rule.positive_labels,
                                        "non_target_celltype": non_target,
                                        "status": "skipped",
                                        "reason": (
                                            "target_below_minimum"
                                            if n_target < MIN_CELLS
                                            else "non_target_below_minimum"
                                        ),
                                        "n_target_cells": n_target,
                                        "n_non_target_cells": n_non_target,
                                    }
                                )
                            continue
                        if method_number == 0:
                            resolution_rows.append(
                                {
                                    "case": case,
                                    "rule_id": rule.rule_id,
                                    "dataset": dataset,
                                    "marker": rule.marker,
                                    "expected_target": rule.positive_labels,
                                    "non_target_celltype": non_target,
                                    "status": "evaluated",
                                    "reason": "eligible",
                                    "n_target_cells": n_target,
                                    "n_non_target_cells": n_non_target,
                                }
                            )
                        canonical_rows.append(
                            contrast_row(
                                case,
                                dataset,
                                rule.marker,
                                rule.positive_labels,
                                non_target,
                                target_mask,
                                non_target_mask,
                                raw,
                                raw_dataset_rank,
                                method,
                                input_dataset_rank,
                                corrected_dataset_rank,
                                column,
                                rule.rule_id,
                                rule.tier,
                                "canonical_predeclared",
                            )
                        )

            # Exhaustive data-driven target discovery covers every shared marker.
            for dataset in pd.unique(datasets):
                dataset_mask = datasets == dataset
                eligible_labels = [
                    label
                    for label, count in pd.Series(celltypes[dataset_mask]).value_counts().items()
                    if count >= EXHAUSTIVE_MIN_CELLS and not INVALID_CELLTYPE.search(str(label))
                ]
                if len(eligible_labels) < 2:
                    continue
                for column, marker in enumerate(markers):
                    mean_by_label = {
                        label: float(
                            raw_dataset_rank[dataset_mask & (celltypes == label), column].mean()
                        )
                        for label in eligible_labels
                    }
                    target = max(mean_by_label, key=mean_by_label.get)
                    target_mask = dataset_mask & (celltypes == target)
                    for non_target in eligible_labels:
                        if non_target == target:
                            continue
                        non_target_mask = dataset_mask & (celltypes == non_target)
                        exhaustive_rows.append(
                            contrast_row(
                                case,
                                str(dataset),
                                marker,
                                target,
                                non_target,
                                target_mask,
                                non_target_mask,
                                raw,
                                raw_dataset_rank,
                                method,
                                input_dataset_rank,
                                corrected_dataset_rank,
                                column,
                                f"DATA_DRIVEN_{case}_{dataset}_{marker}",
                                "data-driven",
                                "exhaustive_all_shared_markers",
                            )
                        )

    canonical = pd.DataFrame(canonical_rows)
    exhaustive = pd.DataFrame(exhaustive_rows)
    integrity = pd.DataFrame(integrity_rows)
    provenance = pd.DataFrame(provenance_rows)
    resolution = pd.DataFrame(resolution_rows)
    if resolution.empty or resolution.duplicated(
        ["case", "rule_id", "dataset", "non_target_celltype"]
    ).any():
        raise RuntimeError("canonical rule-resolution audit is empty or duplicated")
    canonical = canonical.sort_values(
        ["artifact_hit", "artifact_severity", "raw_target_vs_nontarget_auc"],
        ascending=[False, False, False],
    )
    exhaustive = exhaustive.sort_values(
        ["artifact_hit", "artifact_severity", "raw_target_vs_nontarget_auc"],
        ascending=[False, False, False],
    )
    canonical.to_csv(SUMMARY / "canonical_all_comparisons.csv.gz", index=False, compression="gzip")
    exhaustive.to_csv(SUMMARY / "exhaustive_all_shared_marker_comparisons.csv.gz", index=False, compression="gzip")
    canonical_hits = canonical[canonical["artifact_hit"]].copy()
    exhaustive_hits = exhaustive[exhaustive["artifact_hit"]].copy()
    control_events = pd.concat(
        [
            canonical[canonical["control_diagnostic_event"]].assign(screen="canonical"),
            exhaustive[exhaustive["control_diagnostic_event"]].assign(screen="exhaustive"),
        ],
        ignore_index=True,
    )
    canonical_hits.to_csv(SUMMARY / "canonical_artifact_hits.csv", index=False)
    exhaustive_hits.to_csv(SUMMARY / "exhaustive_artifact_candidates.csv", index=False)
    control_events.to_csv(SUMMARY / "rank_preserving_control_threshold_events.csv", index=False)
    integrity.to_csv(SUMMARY / "rank_integrity_by_unit_marker.csv.gz", index=False, compression="gzip")
    integrity_summary = (
        integrity.groupby(["case", "method"], as_index=False)
        .agg(
            median_spearman=("spearman_input_vs_output", "median"),
            minimum_spearman=("spearman_input_vs_output", "min"),
            median_mean_absolute_rank_shift=("mean_absolute_rank_shift", "median"),
            maximum_absolute_rank_shift=("maximum_absolute_rank_shift", "max"),
            median_fraction_abs_rank_shift_ge_0_20=("fraction_abs_rank_shift_ge_0_20", "median"),
            n_unit_markers_with_strict_order_inversion=("strict_order_inversion_present", "sum"),
            maximum_strict_order_inversion_margin=("maximum_strict_order_inversion_margin", "max"),
            n_unit_markers=("marker", "size"),
        )
    )
    integrity_summary.to_csv(SUMMARY / "rank_integrity_summary.csv", index=False)
    provenance.to_csv(SUMMARY / "method_value_output_inventory.csv", index=False)
    resolution.to_csv(SUMMARY / "canonical_rule_resolution.csv", index=False)

    count_summary = (
        canonical.groupby(["method", "method_family", "variant"], as_index=False)
        .agg(
            n_raw_specific_comparisons=("raw_target_vs_nontarget_auc", lambda x: int(np.sum(np.asarray(x) >= 0.70))),
            n_pooled_threshold_hits=("pooled_threshold_hit", "sum"),
            n_artifact_hits=("artifact_hit", "sum"),
            n_order_reversals=("threshold_classification", lambda x: int(np.sum(np.asarray(x) == "target/non-target mean-order reversal"))),
            n_clear_low_to_high=("threshold_classification", lambda x: int(np.sum(np.asarray(x) == "clear non-target low-to-high"))),
            n_clear_high_to_low=("threshold_classification", lambda x: int(np.sum(np.asarray(x) == "clear expected-target high-to-low"))),
            n_strong_intrusions=("threshold_classification", lambda x: int(np.sum(np.asarray(x) == "strong non-target intrusion"))),
        )
    )
    count_summary.to_csv(SUMMARY / "canonical_artifact_counts_by_method.csv", index=False)

    replicated = (
        canonical.assign(hit=canonical["artifact_hit"].astype(int))
        .groupby(
            ["case", "method", "marker", "expected_target", "non_target_celltype"],
            as_index=False,
        )
        .agg(
            n_datasets_tested=("dataset", "nunique"),
            n_datasets_hit=("hit", "sum"),
            datasets_hit=("dataset", lambda x: "; ".join(pd.unique(x[canonical.loc[x.index, "artifact_hit"]]))),
            median_raw_auc=("raw_target_vs_nontarget_auc", "median"),
            median_nontarget_rank_shift=("nontarget_mean_rank_shift", "median"),
            minimum_corrected_target_minus_nontarget=("corrected_target_minus_nontarget_mean_rank", "min"),
        )
    )
    replicated = replicated[replicated["n_datasets_hit"] >= 2].sort_values(
        ["n_datasets_hit", "median_nontarget_rank_shift"], ascending=[False, False]
    )
    replicated.to_csv(SUMMARY / "canonical_replicated_artifacts.csv", index=False)

    plot_methods = [
        "Harmony-direct[source_scale]",
        "Harmony-direct[fixed_arcsinh]",
        "fastMNN-marker[d=NA]",
        "cyCombine",
        "RAMP",
    ]
    make_heatmap(canonical, SUMMARY / "canonical_artifact_heatmap", plot_methods)

    # Regenerate the remembered Tietscher-CD68 plot from the current formal
    # exact-value outputs plus the explicitly labeled fastMNN d=NA stress test.
    t_meta, t_markers, t_raw, t_methods, _ = load_case("tnbc")
    cd68_rows = plot_box_grid(
        "tnbc",
        "Tietscher IMC",
        "CD68",
        ["Myeloid"],
        "Endothelial",
        t_meta,
        t_raw,
        t_methods,
        t_markers,
        display_scale="percentile",
        output_png=SUMMARY / "tietscher_cd68_percentile_boxplots.png",
    )
    # Save a separate vector version.
    with PdfPages(SUMMARY / "tietscher_cd68_percentile_boxplots.pdf") as pdf:
        plot_box_grid(
            "tnbc",
            "Tietscher IMC",
            "CD68",
            ["Myeloid"],
            "Endothelial",
            t_meta,
            t_raw,
            t_methods,
            t_markers,
            display_scale="percentile",
            pdf=pdf,
        )
    pd.DataFrame(cd68_rows).to_csv(
        SUMMARY / "tietscher_cd68_percentile_boxplot_statistics.csv", index=False
    )
    native_cd68_rows = plot_box_grid(
        "tnbc",
        "Tietscher IMC",
        "CD68",
        ["Myeloid"],
        "Endothelial",
        t_meta,
        t_raw,
        t_methods,
        t_markers,
        display_scale="native",
        output_png=SUMMARY / "tietscher_cd68_native_value_boxplots.png",
    )
    with PdfPages(SUMMARY / "tietscher_cd68_native_value_boxplots.pdf") as pdf:
        plot_box_grid(
            "tnbc",
            "Tietscher IMC",
            "CD68",
            ["Myeloid"],
            "Endothelial",
            t_meta,
            t_raw,
            t_methods,
            t_markers,
            display_scale="native",
            pdf=pdf,
        )
    pd.DataFrame(native_cd68_rows).to_csv(
        SUMMARY / "tietscher_cd68_native_value_boxplot_statistics.csv", index=False
    )

    top = (
        canonical_hits[canonical_hits["method"] != "RAMP"]
        .sort_values(["artifact_severity", "raw_target_vs_nontarget_auc"], ascending=[False, False])
        .drop_duplicates(["case", "dataset", "marker", "expected_target", "non_target_celltype"])
        .head(12)
    )
    box_summary: list[dict[str, object]] = []
    with PdfPages(SUMMARY / "top_canonical_artifact_percentile_boxplots.pdf") as pdf:
        for row in top.itertuples(index=False):
            metadata, markers, raw, methods, _ = load_case(row.case)
            box_summary.extend(
                plot_box_grid(
                    row.case,
                    row.dataset,
                    row.marker,
                    split_labels(row.expected_target),
                    row.non_target_celltype,
                    metadata,
                    raw,
                    methods,
                    markers,
                    display_scale="percentile",
                    pdf=pdf,
                )
            )
    pd.DataFrame(box_summary).to_csv(
        SUMMARY / "top_canonical_artifact_percentile_boxplot_statistics.csv", index=False
    )

    result_lines = [
        "# Exact marker-value artifact audit",
        "",
        "This audit pools all donors within each dataset for the primary cell-type marker comparison.",
        "Only outputs with an interpretable one-to-one marker axis are included.",
        "",
        "## Method counts",
        "",
        markdown_table(
            count_summary,
            [
                "method",
                "n_raw_specific_comparisons",
                "n_pooled_threshold_hits",
                "n_artifact_hits",
                "n_order_reversals",
                "n_clear_low_to_high",
                "n_clear_high_to_low",
                "n_strong_intrusions",
            ],
            20,
        ),
        "",
        "## Strongest canonical hits",
        "",
        markdown_table(
            canonical_hits,
            [
                "case",
                "dataset",
                "method",
                "marker",
                "expected_target",
                "non_target_celltype",
                "raw_target_vs_nontarget_auc",
                "nontarget_mean_rank_shift",
                "corrected_target_minus_nontarget_mean_rank",
                "classification",
                "dominant_stage",
            ],
            30,
        ),
        "",
        "## Scope boundaries",
        "",
        "- MaxFuse is excluded because it returns a latent embedding, not marker-axis values.",
        "- Formal fastMNN is excluded because its primary output is a PCA corrected embedding; the separately labeled `fastMNN-marker[d=NA]` arm is a non-default marker-space stress test.",
        "- CytofRUV is not applicable to these four benchmarks because no true cross-batch replicated reference specimen is available.",
        "- Harmony-direct is the ADTnorm-style no-PCA marker-axis variant, not conventional PCA-Harmony.",
        "- Exhaustive all-marker results are discovery candidates; canonical predeclared rules are the biological evidence set.",
        "- Every exact marker-axis output, including RAMP, receives the same pooled threshold screen. RAMP threshold events are stored separately as composition/control diagnostics and are not called artifacts unless a strict within-unit order inversion is present.",
        "- Rank integrity is audited per donor/ROI and marker using both midrank summaries and an explicit strict-order inversion test; tie splitting can change midrank Spearman without reversing any unequal pair.",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(result_lines) + "\n")
    output_paths = sorted(
        [path for path in SUMMARY.rglob("*") if path.is_file()]
        + [ROOT / "RESULTS.md"],
        key=lambda path: str(path),
    )
    manifest = {
        "schema_version": "final_xu.exact_value_artifacts.v2",
        "audit_script_sha256": sha256_file(HERE),
        "benchmarks": list(CASES),
        "canonical_rule_count": len(expectations),
        "canonical_declared_contrasts": len(resolution),
        "canonical_evaluated_contrasts": int((resolution["status"] == "evaluated").sum()),
        "canonical_skipped_contrasts": int((resolution["status"] == "skipped").sum()),
        "primary_pooling": "all donors pooled within each dataset",
        "comparison_scale": "within-dataset midrank percentile, rank/(n+1)",
        "literal_native_values_retained": True,
        "minimum_cells_per_canonical_target_or_nontarget": MIN_CELLS,
        "minimum_cells_per_exhaustive_celltype": EXHAUSTIVE_MIN_CELLS,
        "raw_specificity_gate_auc": 0.70,
        "methods": plot_methods,
        "canonical_comparison_rows": len(canonical),
        "canonical_artifact_hits": len(canonical_hits),
        "exhaustive_comparison_rows": len(exhaustive),
        "exhaustive_artifact_candidates": len(exhaustive_hits),
        "rank_preserving_control_threshold_events": len(control_events),
        "labels_used_for_method_fitting": False,
        "labels_used_posthoc_for_audit": True,
        "expectation_file": str(EXPECTATIONS_PATH.resolve()),
        "expectation_sha256": sha256_file(EXPECTATIONS_PATH),
        "output_files": [
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in output_paths
        ],
        "python": sys.version,
    }
    temporary_manifest = completion_manifest.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(completion_manifest)
    incomplete_marker.unlink()
    print(
        f"complete: canonical hits={len(canonical_hits)}, "
        f"exhaustive candidates={len(exhaustive_hits)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
