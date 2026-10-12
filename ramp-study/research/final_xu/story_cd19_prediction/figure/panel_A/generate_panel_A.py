#!/usr/bin/env python3
"""Add the panel letter and title to the unchanged CODEX CD19 image."""

from __future__ import annotations

import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SOURCE = HERE / "panel_A_observed_codex_cd19.png"
OUTPUT = HERE / "panel_A_failed_codex_cd19"


def main() -> None:
    image = plt.imread(SOURCE)
    fig = plt.figure(figsize=(5.0, 4.15), dpi=180, facecolor="white")
    axis = fig.add_axes([0.025, 0.025, 0.95, 0.82])
    axis.imshow(image)
    axis.axis("off")

    fig.text(
        0.025,
        0.955,
        "A.  Failed CODEX CD19",
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#111111",
    )

    fig.savefig(f"{OUTPUT}.png", dpi=300, facecolor="white")
    fig.savefig(f"{OUTPUT}.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
