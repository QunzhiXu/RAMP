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
    W, H = 6.66, 4.02
    fig = plt.figure(figsize=(W, H), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(H, 0), aspect="equal")
    ax.axis("off")
    ink, edge, arrow_color = "#202C32", "#BBC3C8", "#65727A"
    blue, orange = "#347CAB", "#DE913D"

    def X(value):
        # Compress connector gaps, preserving the logo's width and all type sizes.
        return np.interp(value, [0, .18, 2.18, 2.58, 4.40, 4.80, 7.61, 7.80],
                         [0, .18, 1.99, 2.15, 3.91, 4.07, 6.60, 6.66])

    def Y(value):
        # Remove vertical gaps while retaining the module and task icon heights.
        return np.interp(value, [0, .30, .88, 1.10, 2.66, 3.02, 3.33, 4.13, 4.32],
                         [0, .27, .68, .89, 2.45, 2.74, 3.03, 3.83, 4.02])

    def line(xs, ys, **kwargs):
        return ax.plot(X(xs), Y(ys), **kwargs)

    def text(x, y, value, size=12, weight="normal", color=ink, ha="center", **kwargs):
        return ax.text(X(x), Y(y), value, ha=ha, va="center", fontsize=size,
                       fontweight=weight, color=color, linespacing=kwargs.pop("linespacing", 1.35), zorder=8, **kwargs)

    def box(x, y, w, h):
        ax.add_patch(FancyBboxPatch((X(x), Y(y)), X(x+w)-X(x), Y(y+h)-Y(y),
                     boxstyle="round,pad=0,rounding_size=0.075",
                     facecolor="white", edgecolor=edge, linewidth=.95, zorder=2))

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch((X(start[0]), Y(start[1])), (X(end[0]), Y(end[1])), arrowstyle="-|>", mutation_scale=10,
                     shrinkA=0, shrinkB=0, color=arrow_color, linewidth=1.15, zorder=5))

    def heatmap(x, y, color, values, cell=.112):
        base = np.array(to_rgb(color))
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                face = 1 - value * (1 - base)
                left = x+j*cell
                ax.add_patch(Rectangle((X(left), Y(y+i*cell)), X(left+cell-.012)-X(left), cell-.012,
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
    h = (X(x+w)-X(x)) * logo.shape[0] / logo.shape[1]
    ax.imshow(logo, extent=(X(x), X(x+w), Y(y)+h, Y(y)), zorder=4)
    text(3.49, 2.43, "vs. 9 control pipelines", size=10.1, weight="bold")

    # Three independent prediction heads: line fit, nearest neighbors, network.
    # These drawings illustrate model families and encode no performance result.
    line([4.995, 4.995, 5.63], [1.36, 2.13, 2.13],
            color=arrow_color, linewidth=.9, zorder=4)
    line([5.065, 5.60], [2.035, 1.445], color=blue, linewidth=1.7, zorder=5)
    for cx, cy in [(5.08, 1.94), (5.17, 2.00), (5.27, 1.77),
                   (5.36, 1.81), (5.45, 1.57), (5.56, 1.52)]:
        ax.add_patch(Circle((X(cx), Y(cy)), .030, facecolor=blue,
                     edgecolor="white", linewidth=.45, zorder=6))
    text(5.32, 2.43, "Linear", size=12.2, weight="bold")

    query = (6.205, 1.765)
    neighbors = [(6.06, 1.69), (6.30, 1.66), (6.18, 1.92)]
    ax.add_patch(Circle((X(query[0]), Y(query[1])), .226, facecolor="#F3F7FA",
                 edgecolor=blue, linestyle=(0, (2, 2)), linewidth=.9, zorder=3))
    for cx, cy in neighbors:
        line([query[0], cx], [query[1], cy], color="#91AABB", lw=.8, zorder=4)
    for cx, cy in neighbors + [(5.97, 1.40), (6.43, 1.46),
                               (5.92, 1.99), (6.46, 2.13), (6.37, 1.99)]:
        ax.add_patch(Circle((X(cx), Y(cy)), .037, facecolor=blue,
                     edgecolor="white", linewidth=.4, zorder=5))
    ax.add_patch(Circle((X(query[0]), Y(query[1])), .052, facecolor=orange,
                 edgecolor="white", linewidth=.65, zorder=6))
    text(6.205, 2.43, "kNN", size=12.2, weight="bold")

    layers = [[(6.745, y) for y in [1.45, 1.78, 2.11]],
              [(7.055, y) for y in [1.36, 1.64, 1.92, 2.20]],
              [(7.365, y) for y in [1.57, 1.99]]]
    for left, right in zip(layers[:-1], layers[1:]):
        for p in left:
            for q in right:
                line([p[0], q[0]], [p[1], q[1]], color="#BCC8D1", lw=.7, zorder=3)
    for layer_idx, layer in enumerate(layers):
        for cx, cy in layer:
            ax.add_patch(Circle((X(cx), Y(cy)), .048,
                         facecolor=orange if layer_idx == 2 else blue,
                         edgecolor="white", linewidth=.65, zorder=5))
    text(7.055, 2.43, "Neural\nnetwork", size=10.8, weight="bold", linespacing=1.05)
    arrow((2.19, 1.90), (2.56, 1.90))
    arrow((4.41, 1.90), (4.78, 1.90))

    # Both branches use the trained model on target representations. Target
    # labels / masked-marker measurements are accessed only by the scorer.
    line([6.205, 6.205, 2.105], [2.67, 3.02, 3.02],
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
            ax.add_patch(Circle((X(cx), Y(cy)), .088, facecolor=color,
                         edgecolor="none", zorder=4))
            ax.add_patch(Circle((X(cx)+.012, Y(cy)-.007), .030,
                         facecolor="white", edgecolor="none", alpha=.70, zorder=5))

    # The same marker A is measured in the blue source and predicted in the
    # orange target. The two strips have unequal lengths and arbitrary values:
    # they do not denote paired cells, copied measurements or direct imputation.
    text(5.075, 3.49, "A", size=9.5, weight="bold", color=blue)
    text(5.565, 3.49, "A", size=9.5, weight="bold", color=orange)
    for i, value in enumerate([.25, .78, .45, .93]):
        face = 1 - value * (1 - np.array(to_rgb(blue)))
        ax.add_patch(Rectangle((X(5.00), Y(3.595+i*.083)), X(5.15)-X(5.00), .071,
                     facecolor=face, edgecolor="none", zorder=4))
    ax.add_patch(FancyArrowPatch((X(5.205), Y(3.753)), (X(5.40), Y(3.753)),
                 arrowstyle="-|>", mutation_scale=7, shrinkA=0, shrinkB=0,
                 color=arrow_color, linewidth=.9, zorder=5))
    for i, value in enumerate([.76, .25, .60, .91, .42]):
        face = 1 - value * (1 - np.array(to_rgb(orange)))
        ax.add_patch(Rectangle((X(5.49), Y(3.595+i*.066)), X(5.64)-X(5.49), .055,
                     facecolor=face, edgecolor="none", zorder=4))
    ax.add_patch(Rectangle((X(5.463), Y(3.568)), X(5.667)-X(5.463), .382,
                 facecolor="none", edgecolor=orange, linestyle=(0, (2, 2)),
                 linewidth=.8, zorder=5))

    outputs = []
    for ext in ("png", "pdf", "svg"):
        path = HERE / f"panel_a_cross_platform_learning.{ext}"
        fig.savefig(path, dpi=320, facecolor="white")
        outputs.append(path)
    plt.close(fig)
    records = [source, PROJECT / "MOTIVATION_AND_DESIGN.md",
               PROJECT / "reports/formal_v0_5_20260921/RESULTS_REVIEW_ZH.md"]
    provenance = {
        "panel": "A", "version": 8, "status": "trimmed right margin and compact layout with unchanged typography", "canvas_inches": [W, H],
        "sources": [{"path": str(p), "sha256": sha(p)} for p in records],
        "icon_copied_without_modification": sha(source) == sha(icon),
        "scientific_scope": "Transductive preprocessing; source-only supervision; target-only scoring",
        "comparison_pipelines": 10,
        "visuals_are_schematic": True,
        "task_icons": ["Colored nucleated cells representing cell categories", "Blue measured source marker A to orange predicted target marker A; unequal cell counts, no cell pairing"],
        "model_icons": ["Linear fit", "Nearest-neighbor query", "Multilayer perceptron"],
        "tasks_shown": ["J1", "J2", "J3 via one-or-two-source input"],
        "note": "J2 masks marker A in every platform before input reconstruction and refitting. A denotes a marker, not a source cohort.",
        "outputs": [{"path": str(p), "sha256": sha(p)} for p in outputs],
    }
    (HERE / "DESIGN.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
