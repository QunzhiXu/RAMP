"""Shared deterministic synthetic inputs for package tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_input(
    datasets: int = 3,
    *,
    cells_per_unit: int = 150,
    markers: int = 4,
    seed: int = 20260827,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    matrices: list[np.ndarray] = []
    rows: list[dict[str, str]] = []
    marker_names = [f"M{i + 1}" for i in range(markers)]
    for dataset_index in range(datasets):
        dataset = f"D{dataset_index + 1}"
        for unit_index in range(2):
            donor = f"P{unit_index + 1:02d}"
            n_cells = cells_per_unit + unit_index * 3
            latent = rng.normal(size=(n_cells, 2))
            state = rng.integers(0, 3, size=n_cells)
            signal = latent @ rng.normal(size=(2, markers))
            signal += state[:, None] * np.linspace(0.3, 0.9, markers)
            shifted = signal - signal.min(axis=0) + 0.02
            measured = np.expm1(
                np.log1p(shifted) * (0.85 + 0.12 * dataset_index)
            )
            measured += rng.gamma(1.4, 0.015, size=measured.shape)
            matrices.append(measured)
            rows.extend(
                {
                    "cell_id": f"{dataset}_{donor}_{cell:04d}",
                    "dataset": dataset,
                    "donor_id": donor,
                    "unit_key": f"{dataset}::{donor}",
                    "celltype_must_not_be_read": str(state[cell]),
                }
                for cell in range(n_cells)
            )
    return np.vstack(matrices), pd.DataFrame(rows), marker_names
