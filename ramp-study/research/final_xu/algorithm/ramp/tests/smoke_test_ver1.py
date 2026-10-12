#!/usr/bin/env python3
"""Deterministic checks for all seven RAMP ver1 corrections."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from ramp import ramp_ver1


def synthetic_input() -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(20260827)
    matrices, rows = [], []
    markers = ["M1", "M2", "M3", "M4"]
    for dataset_index in range(3):
        dataset = f"D{dataset_index + 1}"
        for unit_index in range(2):
            n = 105 + 5 * unit_index
            latent = rng.normal(size=(n, 2))
            state = rng.integers(0, 3, size=n)
            signal = latent @ rng.normal(size=(2, len(markers)))
            signal += state[:, None] * np.linspace(0.3, 0.9, len(markers))
            shifted = signal - signal.min(axis=0) + 0.02
            raw = np.expm1(
                np.log1p(shifted) * (0.85 + 0.12 * dataset_index)
            )
            raw += rng.gamma(1.4, 0.015, size=raw.shape)
            matrices.append(raw)
            rows.extend(
                {
                    "dataset": dataset,
                    "unit_key": f"{dataset}|U{unit_index + 1}",
                }
                for _ in range(n)
            )
    return np.vstack(matrices), pd.DataFrame(rows), markers


def check_nonself_k() -> None:
    values = np.arange(40, dtype=float)[:, None]
    _, neighbors = ramp_ver1._nonself_knn(values, 15)
    assert neighbors.shape == (40, 15)
    assert not np.any(neighbors == np.arange(40)[:, None])
    transition = ramp_ver1.transition(values, 15)
    assert np.all(np.diff(transition.indptr) == 15)
    assert np.allclose(transition.diagonal(), 0)


def check_percentiles_and_bounded_mapper() -> None:
    values = np.asarray([1.0, 1.0, 2.0, 4.0, 9.0])
    u = ramp_ver1.within_unit_percentiles(values)
    assert u[0] == u[1]
    identity = ramp_ver1.empirical_profile(values)
    # Profiles are stored on the fixed 257-point grid, so a second interpolation
    # has only the expected sub-grid approximation error.
    np.testing.assert_allclose(
        np.interp(u, ramp_ver1.GRID, identity), values, atol=0.025
    )
    extreme = identity + 100.0
    mapped = ramp_ver1.bounded_profile_mapping(values, u, extreme)
    assert np.max(np.abs(mapped - values)) <= ramp_ver1.MAX_DISPLACEMENT + 1e-12


def check_equal_count_bins() -> None:
    rng = np.random.default_rng(8)
    u = np.sort(rng.uniform(size=103))
    y = u**2 + rng.normal(0, 0.01, len(u))
    profile, audit = ramp_ver1.fit_profile(u, y, np.ones(len(u)))
    assert profile is not None
    assert audit["curve_equal_count_bins"]
    assert audit["curve_bins_used"] == 20
    assert audit["curve_largest_bin"] - audit["curve_smallest_bin"] <= 1
    assert audit["curve_smallest_bin"] >= ramp_ver1.CURVE_MIN_BIN_CELLS


def check_pair_fit_maximum() -> None:
    rows = []
    for dataset in ("A", "B", "C"):
        for unit in range(120):
            rows.extend(
                {"dataset": dataset, "unit_key": f"{dataset}|{unit}"}
                for _ in range(100)
            )
    metadata = pd.DataFrame(rows)
    effective, audit = ramp_ver1.adaptive_pair_fit_size(
        metadata, ["A", "B", "C"]
    )
    assert effective == ramp_ver1.PAIR_FIT_MAXIMUM
    assert audit["pair_fit_requested_before_maximum"] == 12_000
    assert audit["pair_fit_limited_by_maximum"]
    assert not hasattr(ramp_ver1, "PAIR_FIT_CAP")


def check_full_pipeline() -> None:
    raw, metadata, markers = synthetic_input()
    datasets = list(pd.unique(metadata.dataset))
    result = ramp_ver1.fit_transform(
        raw,
        metadata,
        markers,
        datasets=datasets,
        family="ramp_ver1_smoke",
    )
    assert result.baseline.shape == raw.shape
    assert result.aligned.shape == raw.shape
    assert result.strict_rank_inversions == 0
    displacement = np.abs(result.aligned - result.baseline)
    assert displacement.max() <= ramp_ver1.MAX_DISPLACEMENT + 1e-10
    assert result.coupling_audit["graph_k_nonself"].eq(15).all()
    assert result.coupling_audit["transition_includes_self"].eq(False).all()
    assert result.reference_weight_audit[
        "compatibility_dataset_balanced"
    ].eq(True).all()
    assert result.reference_weight_audit[
        "compatibility_pca_fit_balanced"
    ].eq(True).all()
    if len(result.evidence_audit):
        assert result.evidence_audit[
            "provider_evidence_unit_aggregated"
        ].eq(True).all()
    audit = result.algorithm_audit.set_index("audit_key")["audit_value"]
    assert int(audit["cell_marker_values_over_maximum"]) == 0
    assert audit["percentile_definition"] == ramp_ver1.PERCENTILE_DEFINITION


def main() -> None:
    check_nonself_k()
    check_percentiles_and_bounded_mapper()
    check_equal_count_bins()
    check_pair_fit_maximum()
    check_full_pipeline()
    print(f"RAMP {ramp_ver1.__version__} VER1_SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
