#!/usr/bin/env python3
"""Small deterministic API checks for both supported dataset counts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

# Import the package from the new algorithm root when this file is executed
# directly, without depending on the caller's working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from ramp import __version__, fit_transform
from ramp.ramp_ver2 import (
    PAIR_FIT_MINIMUM,
    PAIR_FIT_PER_UNIT,
    adaptive_pair_fit_size,
    unit_balanced_fit_indices,
)


def synthetic_input(sizes: list[int], seed: int) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    matrices, rows = [], []
    markers = [f"M{i + 1}" for i in range(6)]
    for dataset_index, size in enumerate(sizes):
        dataset = f"Dataset {dataset_index + 1}"
        state = rng.integers(0, 3, size=size)
        latent = rng.normal(size=(size, 2))
        weights = rng.normal(size=(2, len(markers)))
        biological = latent @ weights + state[:, None] * np.linspace(0.35, 1.0, len(markers))
        # Dataset-specific monotone measurement distortion and dynamic range.
        shifted = biological - biological.min(axis=0) + 0.05
        raw = (np.expm1(np.log1p(shifted) * (0.85 + 0.18 * dataset_index)))
        raw += rng.gamma(1.5, 0.02, raw.shape)
        matrices.append(raw)
        donor = np.where(np.arange(size) % 2 == 0, "A", "B")
        rows.extend(
            {"dataset": dataset, "unit_key": f"{dataset}|{value}"}
            for value in donor
        )
    return np.vstack(matrices), pd.DataFrame(rows), markers


def check(sizes: list[int]) -> None:
    raw, metadata, markers = synthetic_input(sizes, 100 + len(sizes))
    result = fit_transform(
        raw, metadata, markers,
        datasets=list(pd.unique(metadata.dataset)),
        family=f"smoke_{len(sizes)}_datasets",
    )
    assert result.baseline.shape == raw.shape
    assert result.aligned.shape == raw.shape
    assert np.isfinite(result.baseline).all()
    assert np.isfinite(result.aligned).all()
    assert result.strict_rank_inversions == 0


def check_label_isolation() -> None:
    raw, metadata, markers = synthetic_input([140, 155], 917)
    datasets = list(pd.unique(metadata.dataset))
    plain = fit_transform(
        raw, metadata, markers, datasets=datasets, family="label_isolation"
    )
    augmented = metadata.copy()
    rng = np.random.default_rng(918)
    augmented["celltype_must_not_be_read"] = rng.choice(["T", "B", "Myeloid"], len(augmented))
    augmented["spatial_x_must_not_be_read"] = rng.normal(size=len(augmented))
    augmented["clinical_response_must_not_be_read"] = rng.choice(["R", "NR"], len(augmented))
    with_extra_metadata = fit_transform(
        raw, augmented, markers, datasets=datasets, family="label_isolation"
    )
    np.testing.assert_array_equal(plain.baseline, with_extra_metadata.baseline)
    np.testing.assert_array_equal(plain.aligned, with_extra_metadata.aligned)


def expect_value_error(raw, metadata, markers, message: str) -> None:
    try:
        fit_transform(raw, metadata, markers)
    except ValueError as error:
        assert message in str(error)
    else:
        raise AssertionError(f"expected ValueError containing: {message}")


def check_input_boundaries() -> None:
    raw, metadata, markers = synthetic_input([40, 45], 919)
    missing = metadata.copy()
    missing.loc[0, "unit_key"] = np.nan
    expect_value_error(raw, missing, markers, "cannot contain missing")
    expect_value_error(raw[:, :1], metadata, markers[:1], "at least two marker")

    singleton = metadata.copy()
    singleton.loc[0, "unit_key"] = "Dataset 1|singleton"
    expect_value_error(raw, singleton, markers, "at least two cells")


def scaling_metadata(unit_counts: list[int], cells_per_unit: int) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    datasets = [f"Scale dataset {i + 1}" for i in range(len(unit_counts))]
    for dataset, unit_count in zip(datasets, unit_counts):
        for unit in range(unit_count):
            rows.extend(
                {"dataset": dataset, "unit_key": f"{dataset}|U{unit:03d}"}
                for _ in range(cells_per_unit)
            )
    return pd.DataFrame(rows), datasets


def check_adaptive_pair_fit_size() -> None:
    small, datasets = scaling_metadata([20, 40, 8], 600)
    effective, audit = adaptive_pair_fit_size(small, datasets)
    assert effective == PAIR_FIT_MINIMUM == 4_000
    assert audit["pair_fit_requested"] == 4_000
    assert not audit["pair_fit_limited_by_smallest_dataset"]

    large, datasets = scaling_metadata([70, 30, 5], 150)
    effective, audit = adaptive_pair_fit_size(large, datasets)
    assert PAIR_FIT_PER_UNIT == 100
    assert audit["pair_fit_requested"] == 7_000
    assert effective == 750  # the five-unit dataset is the limiting side
    assert audit["pair_fit_limited_by_smallest_dataset"]

    supported, datasets = scaling_metadata([70, 50], 150)
    effective, audit = adaptive_pair_fit_size(supported, datasets)
    assert effective == 7_000
    for dataset in datasets:
        selected = unit_balanced_fit_indices(supported, dataset, effective)
        counts = supported.iloc[selected].groupby("unit_key").size()
        assert len(selected) == effective
        assert counts.min() >= 100


def main() -> None:
    # Unequal sizes explicitly test the common fitting-cap logic.
    check([420, 510])
    check([420, 475, 530])
    check_label_isolation()
    check_input_boundaries()
    check_adaptive_pair_fit_size()
    print(f"RAMP {__version__} SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
