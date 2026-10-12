"""Minimal in-memory RAMP example using synthetic protein measurements."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ramp_proteomics import fit_transform


def synthetic_data() -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(7)
    matrices: list[np.ndarray] = []
    rows: list[dict[str, str]] = []
    markers = ["CD3", "CD4", "CD8", "HLA-DR"]
    for dataset_index, dataset in enumerate(("CODEX", "CyTOF")):
        for donor_index in range(2):
            donor = f"P{donor_index + 1:02d}"
            n_cells = 150
            state = rng.integers(0, 3, size=n_cells)
            latent = rng.normal(size=(n_cells, 2))
            signal = latent @ rng.normal(size=(2, len(markers)))
            signal += state[:, None] * np.linspace(0.25, 0.9, len(markers))
            shifted = signal - signal.min(axis=0) + 0.05
            measured = np.expm1(
                np.log1p(shifted) * (0.9 + 0.18 * dataset_index)
            )
            measured += rng.gamma(1.5, 0.02, size=measured.shape)
            matrices.append(measured)
            rows.extend(
                {
                    "cell_id": f"{dataset}_{donor}_{cell:04d}",
                    "dataset": dataset,
                    "donor_id": donor,
                    "unit_key": f"{dataset}::{donor}",
                }
                for cell in range(n_cells)
            )
    return np.vstack(matrices), pd.DataFrame(rows), markers


def main() -> None:
    raw, metadata, markers = synthetic_data()
    result = fit_transform(
        raw,
        metadata,
        markers,
        datasets=["CODEX", "CyTOF"],
        family="quickstart-v1",
    )

    print("raw:", raw.shape)
    print("aligned:", result.aligned.shape)
    print("strict rank inversions:", result.strict_rank_inversions)


if __name__ == "__main__":
    main()
