#!/usr/bin/env python3
"""Post-hoc screen for five-patient Meyer draws with B/endothelial coverage.

This is explicitly a coverage-guided sensitivity experiment, not an unbiased
random benchmark draw.  It scans a fixed, recorded block of 100 sequential
seeds and selects the draw maximizing the smaller of its B-cell and endothelial
coverage folds relative to the current frozen Meyer five-patient sample,
subject to 18,000--27,000 total Meyer cells.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CASE = HERE.parent
FINAL_XU = HERE.parents[2]
MEYER_RDS = (
    FINAL_XU / "raw_data" / "tnbc" / "Meyer_2025"
    / "processed_r_objects" / "sce_ALL.rds"
)
EXPORTER = HERE / "export_meyer.R"
OUT = CASE / "coverage_seed_screen"
SEED_START = 20260903
SEED_STOP_INCLUSIVE = 20261002
MIN_TOTAL_CELLS = 18_000
MAX_TOTAL_CELLS = 27_000
N_DONORS = 5
CURRENT_B_CELLS = 1_694
CURRENT_ENDOTHELIAL_CELLS = 2_224


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["Rscript", str(EXPORTER), "candidates", str(MEYER_RDS), str(OUT)],
        check=True,
        cwd=HERE,
    )
    subprocess.run(
        ["Rscript", str(EXPORTER), "coverage", str(MEYER_RDS), str(OUT)],
        check=True,
        cwd=HERE,
    )
    candidates = pd.read_csv(
        OUT / "meyer_candidate_donor_counts.csv", dtype={"donor": str}
    ).sort_values("donor").reset_index(drop=True)
    coverage = pd.read_csv(
        OUT / "meyer_donor_celltype_counts.csv", dtype={"donor": str}
    )
    wide = coverage.pivot_table(
        index="donor", columns="provider_celltype", values="cells",
        aggfunc="sum", fill_value=0,
    ).reindex(candidates["donor"]).fillna(0)
    if len(candidates) != 215 or len(wide) != 215:
        raise AssertionError("Unexpected Meyer donor universe")

    donor_array = candidates["donor"].to_numpy(dtype=object)
    total_lookup = candidates.set_index("donor")["cells"]
    rows = []
    for seed in range(SEED_START, SEED_STOP_INCLUSIVE + 1):
        draw = np.random.default_rng(seed).choice(
            donor_array, N_DONORS, replace=False
        ).astype(str).tolist()
        b_cells = int(wide.loc[draw].get("B_cell", pd.Series(0, index=draw)).sum())
        endothelial = int(
            wide.loc[draw].get("Endothelial", pd.Series(0, index=draw)).sum()
        )
        total = int(total_lookup.loc[draw].sum())
        eligible = MIN_TOTAL_CELLS <= total <= MAX_TOTAL_CELLS
        b_fold = b_cells / CURRENT_B_CELLS
        endothelial_fold = endothelial / CURRENT_ENDOTHELIAL_CELLS
        score = float(min(b_fold, endothelial_fold)) if eligible else -1.0
        rows.append({
            "seed": seed,
            "draw_order": ";".join(draw),
            "selected_sorted": ";".join(sorted(draw)),
            "meyer_cells": total,
            "b_lineage_cells": b_cells,
            "endothelial_cells": endothelial,
            "b_fold_vs_current": b_fold,
            "endothelial_fold_vs_current": endothelial_fold,
            "balanced_min_fold_vs_current": score,
            "within_total_cell_window": eligible,
        })
    results = pd.DataFrame(rows).sort_values(
        ["balanced_min_fold_vs_current", "b_lineage_cells", "endothelial_cells"],
        ascending=False,
    ).reset_index(drop=True)
    if not results.loc[0, "within_total_cell_window"]:
        raise AssertionError("No eligible seed in fixed screening block")
    results["coverage_rank"] = np.arange(1, len(results) + 1)
    results.to_csv(OUT / "seed_screen_all_100.csv", index=False)
    results.head(20).to_csv(OUT / "seed_screen_top20.csv", index=False)

    winner = results.iloc[0].to_dict()
    manifest = {
        "analysis": "post-hoc Meyer B/endothelial coverage-guided seed screen",
        "not_an_unbiased_random_draw": True,
        "seed_range_inclusive": [SEED_START, SEED_STOP_INCLUSIVE],
        "seeds_scanned": int(len(results)),
        "selection_rule": (
            "maximize min(B_cell/1694, Endothelial/2224) using provider "
            "metacluster_combined counts; denominators are the current frozen "
            "Meyer five-patient counts"
        ),
        "total_meyer_cell_constraint": [MIN_TOTAL_CELLS, MAX_TOTAL_CELLS],
        "donors_per_draw": N_DONORS,
        "winner": winner,
    }
    (OUT / "seed_screen_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(winner, indent=2), flush=True)


if __name__ == "__main__":
    main()
