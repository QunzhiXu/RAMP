"""Editable schematic of the completed trainability experiment; no model fitting."""
from pathlib import Path
import hashlib
import json
import os
import shutil

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ROOT = PROJECT.parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.colors import to_rgb


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    source = ROOT / "icon/ramp_sign_no_pipes.png"
    icon = assets / source.name
    if icon.exists():
        assert sha(icon) == sha(source), "Existing figure asset differs; preserve it for review."
    else:
        shutil.copy2(source, icon)

    plt.rcParams.update({
        "font.family": "Arial", "font.size": 12,
        "text.color": "#202C32", "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    W, H = 7.80, 4.32
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(H, 0), aspect="equal")
    ax.axis("off")
    ink, edge, arrow_color = "#202C32", "#BBC3C8", "#65727A"
    blue, orange = "#347CAB", "#DE913D"

    def text(x, y, value, size=12, weight="normal", color=ink, ha="center", **kwargs):
        return ax.text(x, y, value, ha=ha, va="center", fontsize=size,
                       fontweight=weight, color=color, linespacing=1.35, zorder=8, **kwargs)

    def box(x, y, w, h):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0,rounding_size=0.075",
                     facecolor="white", edgecolor=edge, linewidth=.95, zorder=2))

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                     shrinkA=0, shrinkB=0, color=arrow_color, linewidth=1.15, zorder=5))

    def heatmap(x, y, color, values, cell=.112):
        base = np.array(to_rgb(color))
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                face = 1 - value * (1 - base)
                ax.add_patch(Rectangle((x+j*cell, y+i*cell), cell-.012, cell-.012,
                             edgecolor="none", facecolor=face, zorder=4))

    text(.18, .30, "A", size=20, weight="bold", ha="left")
    text(.56, .30, "Cross-platform learning", size=17, weight="bold", ha="left")

    # Expression matrices are schematic, not experimental heatmaps.
    for x, w, label in [(.18, 2.00, "Cross-platform data"),
                         (2.58, 1.82, "Preprocessing"),
                         (4.80, 2.81, "Model training")]:
        text(x+w/2, .88, label, size=12.6, weight="bold")
        box(x, 1.10, w, 1.56)

    text(.72, 1.35, "Source(s)", size=11.5, weight="bold", color=blue)
    text(1.64, 1.35, "Target", size=11.5, weight="bold", color=orange)
    vals = np.array([[.25, .68, .92, .32], [.80, .20, .52, .77],
                     [.62, .87, .23, .48], [.36, .52, .78, .92],
                     [.91, .33, .64, .21]])
    heatmap(.493, 1.61, blue, vals)
    heatmap(1.413, 1.61, orange, np.roll(vals, 1, axis=0) * .85)
    text(1.18, 2.43, "Unlabeled expression", size=10.8, weight="bold")

    logo = plt.imread(icon)
    x, y, w = 2.745, 1.33, 1.49
    h = w * logo.shape[0] / logo.shape[1]
    ax.imshow(logo, extent=(x, x+w, y+h, y), zorder=4)
    text(3.49, 2.43, "vs. 9 control pipelines", size=10.1, weight="bold")

    text(6.205, 1.56, "Linear  ·  kNN  ·  MLP", size=13, weight="bold")
    ax.plot([5.07, 7.34], [1.88, 1.88], color="#E0E5E7", linewidth=.8, zorder=4)
    text(6.205, 2.23, "Source supervision only", size=12, weight="bold", color=blue)
    arrow((2.19, 1.90), (2.56, 1.90))
    arrow((4.41, 1.90), (4.78, 1.90))

    # Both branches use the trained model on target representations. Target
    # labels / masked-marker measurements are accessed only by the scorer.
    ax.plot([6.205, 6.205, 2.105], [2.67, 3.02, 3.02],
            color=arrow_color, linewidth=1.15, zorder=4)
    text(4.04, 3.02, "Apply to target cells", size=11.3, weight="bold",
         bbox=dict(facecolor="white", edgecolor="none", pad=3))
    arrow((2.105, 3.02), (2.105, 3.31))
    arrow((6.205, 3.02), (6.205, 3.31))

    # Compact vector task icons; protocol details stay in the caption / Methods.
    box(.70, 3.33, 2.81, .80)
    box(4.80, 3.33, 2.81, .80)
    text(2.46, 3.73, "Cell-type\nannotation", size=12.5, weight="bold")
    text(6.59, 3.73, "Missing-marker\nprediction", size=12.5, weight="bold")

    # Cell categories are represented by color and small nuclei, not measured
    # expression or a claim of perfect classification.
    for j, color in enumerate([blue, "#A873A3", orange]):
        for i in range(2):
            cx, cy = 1.015 + j*.225, 3.605 + i*.25
            ax.add_patch(Circle((cx, cy), .088, facecolor=color,
                         edgecolor="none", zorder=4))
            ax.add_patch(Circle((cx+.012, cy-.007), .030,
                         facecolor="white", edgecolor="none", alpha=.70, zorder=5))

    # A missing matrix column becomes a predicted continuous-expression column.
    # All intensities are arbitrary schematic values, never an experimental ROI.
    heatmap(5.00, 3.555, "#8498A5",
            np.array([[.30, .70], [.65, .40], [.45, .80], [.80, .30]]), cell=.095)
    ax.add_patch(Rectangle((5.195, 3.551), .088, .374, facecolor="#F5F7F8",
                 edgecolor="#8C989F", linestyle=(0, (1.5, 1.5)), linewidth=.75, zorder=4))
    ax.add_patch(FancyArrowPatch((5.325, 3.738), (5.505, 3.738),
                 arrowstyle="-|>", mutation_scale=7, shrinkA=0, shrinkB=0,
                 color=arrow_color, linewidth=.9, zorder=5))
    heatmap(5.56, 3.555, "#007C78", np.array([[.20], [.43], [.69], [.95]]), cell=.095)

    outputs = []
    for ext in ("png", "pdf", "svg"):
        path = HERE / f"panel_a_cross_platform_learning.{ext}"
        fig.savefig(path, dpi=320, facecolor="white")
        outputs.append(path)
    plt.close(fig)
    records = [source, PROJECT / "MOTIVATION_AND_DESIGN.md",
               PROJECT / "reports/formal_v0_5_20260921/RESULTS_REVIEW_ZH.md"]
    provenance = {
        "panel": "A", "version": 3, "status": "compact task labels with vector icons", "canvas_inches": [W, H],
        "sources": [{"path": str(p), "sha256": sha(p)} for p in records],
        "icon_copied_without_modification": sha(source) == sha(icon),
        "scientific_scope": "Transductive preprocessing; source-only supervision; target-only scoring",
        "comparison_pipelines": 10,
        "visuals_are_schematic": True,
        "task_icons": ["Colored nucleated cells representing cell categories", "A masked matrix column becoming a predicted expression column"],
        "tasks_shown": ["J1", "J2", "J3 via one-or-two-source input"],
        "note": "J2 masks marker A in every platform before input reconstruction and refitting. A denotes a marker, not a source cohort.",
        "outputs": [{"path": str(p), "sha256": sha(p)} for p in outputs],
    }
    (HERE / "DESIGN.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
