#!/usr/bin/env python3
"""Find display crops with clear all-CD8 held-out spatial recovery."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FIGURE_ROOT = HERE.parents[1]
FINAL_XU = FIGURE_ROOT.parent
PREDICTIONS = HERE / "outer_patient_predictions.csv.gz"
RAW_TABLE = (
    FINAL_XU
    / "raw_data/melanoma/moldoveanu_imc/Moldoveanu_2022_CyTOF_melanoma/Data/ICI_meanIntensity.tsv.gz"
)
OUT = HERE / "visual_roi_candidates.csv"

CALL_COLUMNS = {
    "top3": "called_top3",
    "top5": "called_top5",
    "top7.5": "called_top7p5",
    "top10": "called_top10",
    "top15": "called_top15",
    "top20": "called_top20",
    "top25": "called_top25",
    "top30": "called_top30",
}


def main() -> None:
    pred = pd.read_csv(PREDICTIONS)
    raw = pd.read_csv(
        RAW_TABLE,
        sep="\t",
        usecols=["Sample_ID", "Cell_ID", "Location_Center_X", "Location_Center_Y"],
        low_memory=False,
    )
    raw["cell_uid_ordered"] = raw["Sample_ID"].astype(str) + "::" + raw["Cell_ID"].astype(str)
    lookup = raw.set_index("cell_uid_ordered")[["Location_Center_X", "Location_Center_Y"]]

    rows: list[dict[str, object]] = []
    for patient, cells in pred.groupby("patient", sort=True):
        cells = cells.copy()
        coordinates = lookup.reindex(cells["cell_uid_ordered"])
        if coordinates.isna().any().any():
            raise AssertionError(f"Missing coordinates for {patient}")
        xy = coordinates.to_numpy(float)
        target = cells["target_30um"].to_numpy(bool)
        if target.sum() < 5:
            continue
        xmax = int(np.ceil(xy[:, 0].max()))
        ymax = int(np.ceil(xy[:, 1].max()))

        for width, height, step in (
            (250, 250, 25),
            (300, 300, 25),
            (350, 350, 25),
            (400, 400, 25),
            (500, 400, 25),
            (500, 500, 25),
            (600, 500, 25),
        ):
            if width > xmax or height > ymax:
                continue
            for x0 in range(0, max(1, xmax - width + 1), step):
                in_x = (xy[:, 0] >= x0) & (xy[:, 0] <= x0 + width)
                if in_x.sum() < 8:
                    continue
                for y0 in range(0, max(1, ymax - height + 1), step):
                    inside = in_x & (xy[:, 1] >= y0) & (xy[:, 1] <= y0 + height)
                    n_pool = int(inside.sum())
                    if n_pool < 8:
                        continue
                    n_target = int((inside & target).sum())
                    if not 6 <= n_target <= 35:
                        continue
                    for rule, column in CALL_COLUMNS.items():
                        called = cells[column].to_numpy(bool)
                        n_called = int((inside & called).sum())
                        if not 6 <= n_called <= 35:
                            continue
                        tp = int((inside & called & target).sum())
                        if tp < 5:
                            continue
                        fp = n_called - tp
                        fn = n_target - tp
                        precision = tp / n_called
                        recall = tp / n_target
                        f1 = 2 * precision * recall / (precision + recall)
                        balance = min(n_target, n_called) / max(n_target, n_called)
                        visual_score = min(precision, recall) + 0.08 * np.log1p(tp) + 0.05 * balance
                        rows.append({
                            "patient": str(patient),
                            "rule": rule,
                            "x0": x0,
                            "y0": y0,
                            "x1": x0 + width,
                            "y1": y0 + height,
                            "width": width,
                            "height": height,
                            "n_all_cd8_pool": n_pool,
                            "n_tcae_pool": int((inside & cells["is_tcae"].to_numpy(bool)).sum()),
                            "n_tcnaive_pool": int((inside & cells["is_tcnaive"].to_numpy(bool)).sum()),
                            "n_target": n_target,
                            "n_called": n_called,
                            "tp": tp,
                            "fp": fp,
                            "fn": fn,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "balance": balance,
                            "visual_score": visual_score,
                        })

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No visual candidates found")
    result = result.sort_values(["visual_score", "tp", "f1"], ascending=[False, False, False])
    result = result.drop_duplicates(
        ["patient", "x0", "y0", "x1", "y1", "n_called", "tp", "n_target"]
    )
    result.to_csv(OUT, index=False)
    print(result.head(100).to_string(index=False))


if __name__ == "__main__":
    main()
