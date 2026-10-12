#!/usr/bin/env python3
"""Rebuild the HNSCC patient spatial-return feature using top-40% state CD8."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
CELL_PATH = PARENT / "output/haist74_pt_cd8_tigit_cd39_return.csv.gz"
TOP25_PATH = PARENT / "output/haist74_tigit_cd39_patient_features.csv"
OUT = HERE / "output"
OUT_FEATURES = OUT / "haist74_tigit_cd39_top40_patient_features.csv"
OUT_MANIFEST = OUT / "top40_feature_manifest.json"
THRESHOLD = 0.60


def main() -> None:
    cells = pd.read_csv(
        CELL_PATH,
        usecols=[
            "patient",
            "spatial_y",
            "in_caf_macrophage_rich_niche",
            "predicted_pair_rank_within_patient",
        ],
        dtype={"patient": str},
    )
    cells["high_state_top40"] = cells["predicted_pair_rank_within_patient"].ge(THRESHOLD)
    cells["in_niche"] = cells["spatial_y"].gt(0)

    rows: list[dict] = []
    for patient, frame in cells.groupby("patient", sort=True):
        high = frame["high_state_top40"].to_numpy(bool)
        niche = frame["in_niche"].to_numpy(bool)
        spatial = frame["spatial_y"].to_numpy(float)
        if not high.any():
            raise AssertionError(f"Patient {patient} has no top-40% cells")
        rows.append(
            {
                "patient": str(patient),
                "pt_cd8_cells": int(len(frame)),
                "high_state_cells": int(high.sum()),
                "niche_cells": int(niche.sum()),
                "high_state_niche_cells": int((high & niche).sum()),
                "high_state_niche_fraction": float(niche[high].mean()),
                "high_state_niche_recall": float((high & niche).sum() / niche.sum()),
                "high_pair_mean_spatial_y": float(spatial[high].mean()),
                "patient_spatial_niche_score": float(spatial.mean()),
            }
        )
    features = pd.DataFrame(rows).sort_values("patient", key=lambda x: x.astype(int))
    if len(features) != 74 or int(features.pt_cd8_cells.sum()) != 38_299:
        raise AssertionError("Expected 74 patients and 38,299 PT-CD8 cells")

    # Verify that only the high-state selection changed; the direct spatial
    # score and patient universe must remain identical to the top-25 analysis.
    old = pd.read_csv(TOP25_PATH, dtype={"patient": str})[
        ["patient", "pt_cd8_cells", "patient_spatial_niche_score"]
    ]
    check = features.merge(old, on="patient", suffixes=("_top40", "_top25"), validate="one_to_one")
    if not np.array_equal(check.pt_cd8_cells_top40, check.pt_cd8_cells_top25):
        raise AssertionError("Patient cell counts changed")
    if not np.allclose(
        check.patient_spatial_niche_score_top40,
        check.patient_spatial_niche_score_top25,
        rtol=0,
        atol=1e-12,
    ):
        raise AssertionError("The direct spatial score changed")

    OUT.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUT_FEATURES, index=False)
    high = cells.high_state_top40.to_numpy(bool)
    niche = cells.in_niche.to_numpy(bool)
    manifest = {
        "status": "test-only top-40% threshold sensitivity",
        "source_cell_table": str(CELL_PATH),
        "patients": int(features.patient.nunique()),
        "pt_cd8_cells": int(len(cells)),
        "high_state_definition": "predicted_pair_rank_within_patient >= 0.60",
        "patient_score_definition": (
            "mean continuous ROI-adjusted CAF/macrophage spatial_y among each patient's "
            "top 40% returned TIGIT/CD39-state PT-CD8 cells"
        ),
        "pooled_high_state_cells": int(high.sum()),
        "pooled_high_state_niche_cells": int((high & niche).sum()),
        "pooled_high_state_niche_fraction": float(niche[high].mean()),
        "pooled_niche_recall": float((high & niche).sum() / niche.sum()),
        "output": str(OUT_FEATURES),
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
