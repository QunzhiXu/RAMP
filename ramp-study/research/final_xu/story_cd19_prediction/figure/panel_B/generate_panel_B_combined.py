#!/usr/bin/env python3
"""Combine label-free QC with label-based post-hoc CD19 confirmation."""

from __future__ import annotations

import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
STORY = HERE.parents[1]
SAMPLE = STORY / "sampled_data"
DATA = HERE / "data"
CACHE = HERE / ".cache" / "combined"
for folder in [DATA, CACHE]:
    folder.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DATASETS = ["BM CODEX", "BM CyTOF", "Triana AbSeq"]
DATASET_COLORS = {
    "BM CODEX": "#3B6FB6",
    "BM CyTOF": "#4DA66A",
    "Triana AbSeq": "#E58A2B",
}
MARKER_COLORS = {
    "CD19": "#D81B60",
    "CD61": "#4477AA",
    "CD45RA": "#66A61E",
    "CD33": "#66A61E",
}


def label_masks(folder: str, metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Provider B and CD4/CD8 T labels used only for the right-hand audit."""

    if folder == "bm_codex":
        label = metadata["cluster_anno_l1"].fillna("").astype(str)
        b_cell = label.eq("B-Cells")
        t_cell = label.isin(["CD4+ T-Cell", "CD8+ T-Cell"])
    elif folder == "bm_cytof":
        label = metadata["provider_celltype"].fillna("").astype(str)
        b_cell = label.isin(["Mature B cells", "PreBcells"])
        t_cell = label.isin(["CD4 T cells", "CD8 T cells"])
    elif folder == "triana_abseq":
        label = metadata["provider_celltype"].fillna("").astype(str)
        b_names = {
            "CD11c+ memory B cells",
            "Class switched memory B cells",
            "Immature B cells",
            "Mature naive B cells",
            "Nonswitched memory B cells",
            "Pre-B cells",
            "Pre-pro-B cells",
            "Pro-B cells",
            "Small pre-B cell",
        }
        b_cell = label.isin(b_names)
        t_cell = label.str.startswith("CD4+") | label.str.startswith("CD8+")
    else:
        raise ValueError(folder)
    return b_cell.to_numpy(bool), t_cell.to_numpy(bool)


def compute_specificity() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for folder in ["bm_codex", "bm_cytof", "triana_abseq"]:
        expression = pd.read_csv(
            SAMPLE / folder / "source_input_15.csv.gz",
            usecols=["cell_id", "dataset", "donor", "CD19"],
        )
        metadata = pd.read_csv(SAMPLE / folder / "locked_metadata.csv.gz", low_memory=False)
        if not np.array_equal(
            expression["cell_id"].astype(str).to_numpy(),
            metadata["cell_id"].astype(str).to_numpy(),
        ):
            raise AssertionError(f"Cell-order mismatch in {folder}")
        b_cell, t_cell = label_masks(folder, metadata)
        donors = expression["donor"].astype(str).to_numpy()
        cd19 = expression["CD19"].to_numpy(float)
        dataset = str(expression["dataset"].iloc[0])
        for donor in sorted(np.unique(donors)):
            donor_mask = donors == donor
            selected = donor_mask & (b_cell | t_cell)
            outcome = b_cell[selected].astype(int)
            if len(np.unique(outcome)) != 2:
                continue
            records.append(
                {
                    "dataset": dataset,
                    "donor": donor,
                    "n_b_cells": int(np.sum(donor_mask & b_cell)),
                    "n_cd4_cd8_t_cells": int(np.sum(donor_mask & t_cell)),
                    "b_vs_t_cd19_auroc": float(roc_auc_score(outcome, cd19[selected])),
                }
            )
    return pd.DataFrame(records)


def draw_qc(axis: plt.Axes, scores: pd.DataFrame, summary: pd.DataFrame, order: list[str]) -> None:
    donors = sorted(scores["unit_key"].unique())
    offsets = np.linspace(-0.15, 0.15, len(donors))
    pivot = scores.pivot(index="unit_key", columns="marker", values="qc_alarm").loc[donors, order]
    for donor_index, donor in enumerate(donors):
        x = np.arange(len(order), dtype=float) + offsets[donor_index]
        axis.plot(
            x,
            pivot.loc[donor].to_numpy(float),
            color="#A6A6A6",
            alpha=0.23,
            lw=0.7,
            zorder=1,
        )
    for marker_index, marker in enumerate(order):
        values = pivot[marker].to_numpy(float)
        axis.scatter(
            marker_index + offsets,
            values,
            s=34,
            color=MARKER_COLORS.get(marker, "#777777"),
            edgecolor="white",
            linewidth=0.45,
            alpha=0.94,
            zorder=3,
        )
        row = summary.loc[summary["marker"].eq(marker)].iloc[0]
        axis.text(
            marker_index,
            1.035,
            f"{int(row.top1_donors)}/12 ranked #1",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold" if marker == "CD19" else "normal",
            color=MARKER_COLORS.get(marker, "#555555"),
        )
    axis.set(xlim=(-0.42, len(order) - 0.58), ylim=(0.50, 1.105))
    axis.set_xticks(range(len(order)), order, fontsize=10.5, fontweight="bold")
    axis.set_ylabel("Marker anomaly score", fontsize=9.6, fontweight="bold")
    axis.set_title("Label-free anomaly detection", loc="left", fontsize=10.8, fontweight="bold", pad=8)
    axis.grid(axis="y", color="#D9D9D9", lw=0.65, alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#555555")
    axis.tick_params(axis="y", labelsize=8.4)


def draw_specificity(axis: plt.Axes, results: pd.DataFrame) -> None:
    for dataset_index, dataset in enumerate(DATASETS):
        values = results.loc[
            results["dataset"].eq(dataset), "b_vs_t_cd19_auroc"
        ].to_numpy(float)
        offsets = np.linspace(-0.14, 0.14, len(values)) if len(values) > 1 else np.zeros(1)
        axis.scatter(
            dataset_index + offsets,
            values,
            s=46,
            color=DATASET_COLORS[dataset],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.96,
            zorder=3,
        )
        median = float(np.median(values))
        axis.plot(
            [dataset_index - 0.19, dataset_index + 0.19],
            [median, median],
            color="#222222",
            lw=2.0,
            solid_capstyle="round",
            zorder=4,
        )
        axis.text(
            dataset_index,
            min(1.035, max(values) + 0.045),
            f"median {median:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color=DATASET_COLORS[dataset],
        )
    axis.axhline(0.5, color="#666666", linestyle=(0, (4, 3)), lw=1.0, zorder=0)
    axis.set(xlim=(-0.48, 2.48), ylim=(0.33, 1.08))
    axis.set_xticks(range(3), ["CODEX", "CyTOF", "AbSeq"], fontsize=10.0, fontweight="bold")
    axis.set_ylabel("B-versus-T CD19 AUROC", fontsize=9.6, fontweight="bold")
    axis.set_title("Observed CD19 specificity", loc="left", fontsize=10.8, fontweight="bold", pad=8)
    axis.grid(axis="y", color="#D9D9D9", lw=0.65, alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#555555")
    axis.tick_params(axis="y", labelsize=8.4)


def main() -> None:
    qc_scores = pd.read_csv(DATA / "codex_top3_marker_donor_scores.csv")
    qc_summary = pd.read_csv(DATA / "codex_marker_qc_summary.csv")
    qc_manifest = json.loads((DATA / "manifest.json").read_text())
    specificity = compute_specificity()
    specificity.to_csv(DATA / "donor_cd19_specificity.csv", index=False)
    specificity_summary = (
        specificity.groupby("dataset", as_index=False)
        .agg(
            n_donors=("donor", "nunique"),
            median_auroc=("b_vs_t_cd19_auroc", "median"),
            min_auroc=("b_vs_t_cd19_auroc", "min"),
            max_auroc=("b_vs_t_cd19_auroc", "max"),
        )
    )
    specificity_summary.to_csv(DATA / "platform_cd19_specificity_summary.csv", index=False)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 3.55),
        dpi=200,
        gridspec_kw={"width_ratios": [1.08, 1.0]},
    )
    draw_qc(axes[0], qc_scores, qc_summary, list(qc_manifest["top3_markers"]))
    draw_specificity(axes[1], specificity)
    fig.suptitle(
        "B.  Label-free QC identifies failed CODEX CD19",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=15.0,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.10, wspace=0.31)
    fig.savefig(HERE / "panel_B_qc_and_cd19_specificity.png", dpi=300, facecolor="white")
    fig.savefig(HERE / "panel_B_qc_and_cd19_specificity.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
