#!/usr/bin/env python3
"""Build Panel F: survival associated with returned state-high CD8 exposure."""

from __future__ import annotations

import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
HNSCC_ROOT = HERE.parent.parent
RESULTS = HNSCC_ROOT / "spatial_return_survival" / "output"
KM_FILE = RESULTS / "top40_median_split_km_curve.csv"
RISK_FILE = RESULTS / "top40_median_split_number_at_risk.csv"
MODEL_FILE = RESULTS / "top40_survival_models.csv"
CACHE = HERE / ".cache"
for child in ("matplotlib", "xdg"):
    (CACHE / child).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE / "xdg"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


INK = "#182027"
BLUE = "#2B6CB0"
RED = "#C9433C"


def main() -> None:
    missing = [str(path) for path in (KM_FILE, RISK_FILE, MODEL_FILE) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing formal survival outputs:\n" + "\n".join(missing))

    km = pd.read_csv(KM_FILE)
    risk = pd.read_csv(RISK_FILE)
    models = pd.read_csv(MODEL_FILE)
    primary = models.loc[models.role.eq("primary")]
    if len(primary) != 1:
        raise AssertionError("Expected exactly one primary survival model")
    primary = primary.iloc[0]
    if int(primary.patients) != 74 or int(primary.deaths) != 24:
        raise AssertionError("Frozen complete-OS cohort counts have changed")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
    fig = plt.figure(figsize=(5.575, 5.6), facecolor="white")
    host = fig.add_axes([0, 0, 1, 1])
    host.set_axis_off()

    host.text(0.010, 0.975, "F", ha="left", va="top", fontsize=24,
              weight="bold", color=INK)
    host.text(
        0.110, 0.952,
        "STATE-HIGH CD8–CAF/MACROPHAGE PROXIMITY\nASSOCIATES WITH POORER SURVIVAL",
        ha="left", va="top", fontsize=11.1, weight="bold", color=INK,
        linespacing=1.05,
    )
    host.text(
        0.555, 0.835,
        f"CONTINUOUS COX: HR = {primary.hazard_ratio_per_sd:.2f} PER SD",
        ha="center", va="center", fontsize=9.0, weight="bold", color=INK,
    )
    host.text(
        0.555, 0.793,
        f"95% CI, {primary.ci95_low:.2f}–{primary.ci95_high:.2f};  P = {primary.wald_p:.3f}",
        ha="center", va="center", fontsize=8.3, weight="bold", color=INK,
    )

    axis = fig.add_axes([0.165, 0.315, 0.795, 0.425])
    colors = {"Lower": BLUE, "Higher": RED}
    labels = {
        "Lower": "Lower CAF/macrophage proximity (n=37)",
        "Higher": "Higher CAF/macrophage proximity (n=37)",
    }
    for group in ("Lower", "Higher"):
        data = km.loc[km.group.eq(group)].sort_values("time")
        axis.step(
            data.time, data.survival, where="post", color=colors[group],
            linewidth=2.5, label=labels[group],
        )

    axis.set_xlim(0, 120)
    axis.set_ylim(0, 1.05)
    axis.set_xticks([0, 24, 48, 72, 96, 120])
    axis.set_yticks(np.arange(0, 1.01, 0.2))
    axis.set_xlabel("Overall survival (months)", fontsize=9.4, color=INK, labelpad=5)
    axis.set_ylabel("Overall survival probability", fontsize=9.4, color=INK, labelpad=5)
    axis.tick_params(axis="both", labelsize=8.0, colors=INK, length=3)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(INK)
    axis.grid(False)
    axis.legend(
        handles=[
            Line2D([0], [0], color=BLUE, linewidth=2.5, label=labels["Lower"]),
            Line2D([0], [0], color=RED, linewidth=2.5, label=labels["Higher"]),
        ],
        loc="lower left", bbox_to_anchor=(0.01, 0.02), frameon=False,
        fontsize=7.7, handlelength=2.1,
    )

    host.text(
        0.070, 0.235, "NUMBER\nAT RISK", ha="center", va="center",
        fontsize=7.2, weight="bold", color=INK, linespacing=0.92,
    )
    x_positions = np.linspace(0.165, 0.960, 6)
    for group, y in (("Lower", 0.178), ("Higher", 0.108)):
        host.text(0.145, y, group, ha="right", va="center", fontsize=7.8,
                  weight="bold", color=colors[group])
        values = risk.loc[risk.group.eq(group)].set_index("time").n_risk
        for x, month in zip(x_positions, [0, 24, 48, 72, 96, 120]):
            host.text(x, y, str(int(values.loc[month])), ha="center", va="center",
                      fontsize=7.8, weight="bold", color=colors[group])

    png = HERE / "panel_F.png"
    pdf = HERE / "panel_F.pdf"
    fig.savefig(png, dpi=320, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
