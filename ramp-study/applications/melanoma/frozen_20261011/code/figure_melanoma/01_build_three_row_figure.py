#!/usr/bin/env python3
"""Build the compact three-row formal melanoma CD8 story figure.

All RAMP-dependent panels read the local clean15 RAMP 2.3.0 analysis under
``figure_melanoma/analysis``.  Version-independent observed inputs are frozen
under ``figure_melanoma/inputs``.  No test directory or copied raw matrix is
required.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
FINAL_XU = HERE.parent
ANALYSIS_ROOT = HERE / "analysis"
ALIGNMENT_ROOT = ANALYSIS_ROOT / "alignment/output"
ALL_CD8 = ANALYSIS_ROOT / "all_cd8_prediction"
INPUTS = HERE / "inputs"
PANELS = HERE / "panels"
CACHE = HERE / "cache"
OUTPUT = HERE / "output"
PDF_OUTPUT = OUTPUT / "pdf"
for directory in (PANELS, CACHE, OUTPUT, PDF_OUTPUT):
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".xdg-cache"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(HERE / ".numba-cache"))
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.preprocessing import StandardScaler


ALIGNMENT_VALUES = ALIGNMENT_ROOT / "ramp_aligned.npy"
ALIGNMENT_META = ALIGNMENT_ROOT / "cell_metadata.csv.gz"
ALIGNMENT_MANIFEST = ALIGNMENT_ROOT / "manifest.json"
DISPLAY_LABELS = INPUTS / "harmonized_display_labels.csv.gz"

PANEL_A_SOURCE = INPUTS / "panel_a_spatial_state_definition_30um.png"
SURVIVAL_FEATURES = INPUTS / "observed_patient_level_features.csv"
SURVIVAL_RESULTS = INPUTS / "observed_survival_results.csv"
SPATIAL_PAIR_SOURCE = (
    ALL_CD8 / "spatial_figure/22RD_allCD8_fixed30um_global_top20_recommended.png"
)
PROJECTION_TABLE = (
    ALL_CD8 / "citeseq_projection/engelbrecht_all_cd8_projection.csv.gz"
)
PATIENT_METRICS = ALL_CD8 / "patient_metrics.csv"
HELDOUT_EFFECTS = (
    ALL_CD8 / "biological_interpretation/all_heldout_adt_effects_by_patient.csv"
)
MODULES = ALL_CD8 / "biological_interpretation/candidate_biomarker_modules.csv"

ARIAL = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
ARIAL_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")

SEED = 20260820
COL_W = 1180
MID_W = 1305
RIGHT_W = 930
ROW_H = 1000
GAP_X = 20
GAP_Y = 22
PAIR_W = MID_W * 2 + GAP_X
TOTAL_W = COL_W + GAP_X + PAIR_W + GAP_X + RIGHT_W

DATASET_COLORS = {
    "Moldoveanu IMC": "#3F78B5",
    "Engelbrecht CITE-seq": "#E68A2E",
}
CELLTYPE_COLORS = {
    "Tumor": "#D94B4B",
    "CD8 T": "#155C8A",
    "Other T / CD8-": "#6E9ACB",
    "B lineage": "#E5B13C",
    "NK": "#13A89E",
    "Myeloid": "#D9781E",
    "Endothelial": "#4D9851",
    "Stroma": "#9A6DB0",
    "Non-joint / ambiguous": "#9A9A9A",
}
CELLTYPE_DRAW_ORDER = [
    "Non-joint / ambiguous",
    "Tumor",
    "Stroma",
    "Endothelial",
    "NK",
    "Myeloid",
    "B lineage",
    "Other T / CD8-",
    "CD8 T",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(ARIAL_BOLD if bold else ARIAL), size)


def contain(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )


def cover(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def image_panel(
    title: str,
    source: Path,
    destination: Path,
    width: int,
    height: int,
    crop: tuple[int, int, int, int] | None = None,
    mode: str = "contain",
    subtitle: str | None = None,
) -> Path:
    source_image = Image.open(source).convert("RGB")
    if crop is not None:
        source_image = source_image.crop(crop)
    header = 112 if subtitle is None else 150
    body_h = height - header
    fitted = (
        cover(source_image, width, body_h)
        if mode == "cover"
        else contain(source_image, width, body_h)
    )
    panel = Image.new("RGB", (width, height), "white")
    x = (width - fitted.width) // 2
    y = header + (body_h - fitted.height) // 2
    panel.paste(fitted, (x, y))
    draw = ImageDraw.Draw(panel)
    draw.text((18, 10), title, fill="#151515", font=font(64, True))
    if subtitle:
        draw.text((20, 83), subtitle, fill="#4D5962", font=font(28, False))
    panel.save(destination, dpi=(300, 300), optimize=True)
    return destination


def build_observed_panels() -> tuple[Path, Path, Path]:
    panel_a = image_panel(
        "A. Define the spatial CD8 state",
        PANEL_A_SOURCE,
        PANELS / "panel_a_spatial_definition.png",
        COL_W,
        ROW_H,
        crop=(0, 105, 1688, 1439),
        mode="contain",
    )
    panel_b = build_survival_panel()

    # Exact microscopy rectangles in the 2590 x 1470 test output.
    true_crop = (126, 220, 1207, 1283)
    pred_crop = (1478, 220, 2559, 1283)
    source = Image.open(SPATIAL_PAIR_SOURCE).convert("RGB")
    true_image = source.crop(true_crop)
    pred_image = source.crop(pred_crop)
    panel_c = Image.new("RGB", (RIGHT_W, ROW_H * 2 + GAP_Y), "white")
    draw = ImageDraw.Draw(panel_c)
    draw.text((18, 10), "C. Held-out spatial recovery", fill="#151515", font=font(64, True))
    draw.text((20, 78), "Fixed spatial ground truth", fill="#151515", font=font(43, True))
    draw.text(
        (20, 129),
        "Ki67+ CD8 T cells within 30 um of melanoma",
        fill="#4D5962",
        font=font(27, False),
    )
    # Both spatial maps use one exact display size.  Previously the top map was
    # height-constrained by its longer header and therefore appeared narrower.
    map_width = RIGHT_W - 20
    map_height = round(true_image.height * map_width / true_image.width)
    fitted_true = true_image.resize((map_width, map_height), Image.Resampling.LANCZOS)
    fitted_pred = pred_image.resize((map_width, map_height), Image.Resampling.LANCZOS)
    map_left = (RIGHT_W - map_width) // 2
    true_top = 160
    panel_c.paste(fitted_true, (map_left, true_top))

    second_top = true_top + map_height + 12
    draw.text(
        (20, second_top),
        "Patient-held-out prediction",
        fill="#151515",
        font=font(43, True),
    )
    panel_c.paste(
        fitted_pred,
        (map_left, second_top + 60),
    )
    panel_c_path = PANELS / "panel_c_spatial_recovery.png"
    panel_c.save(panel_c_path, dpi=(300, 300), optimize=True)
    return panel_a, panel_b, panel_c_path


def km_curve(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    survival = 1.0
    x = [0.0]
    y = [1.0]
    for point in np.sort(np.unique(time[event == 1])):
        at_risk = int(np.sum(time >= point))
        deaths = int(np.sum((time == point) & (event == 1)))
        x.extend([float(point), float(point)])
        y.extend([survival, survival * (1.0 - deaths / at_risk)])
        survival = y[-1]
    x.append(float(np.max(time)))
    y.append(survival)
    return np.asarray(x), np.asarray(y)


def survival_at(time: np.ndarray, event: np.ndarray, query: np.ndarray) -> np.ndarray:
    result = []
    event_times = np.sort(np.unique(time[event == 1]))
    for point in query:
        survival = 1.0
        for event_time in event_times[event_times <= point]:
            at_risk = int(np.sum(time >= event_time))
            deaths = int(np.sum((time == event_time) & (event == 1)))
            survival *= 1.0 - deaths / at_risk
        result.append(survival)
    return np.asarray(result)


def build_survival_panel() -> Path:
    feature = "observed_ki67_contact30_tcae_per_all"
    data = pd.read_csv(SURVIVAL_FEATURES)
    result = pd.read_csv(SURVIVAL_RESULTS).set_index("feature").loc[feature]
    keep = (
        pd.to_numeric(data[feature], errors="coerce").notna()
        & pd.to_numeric(data["Survival_from_Rx_Start"], errors="coerce").notna()
        & pd.to_numeric(data["Known_Deceased"], errors="coerce").notna()
    )
    frame = data.loc[keep].copy()
    burden = frame[feature].to_numpy(float)
    time = frame["Survival_from_Rx_Start"].to_numpy(float)
    event = frame["Known_Deceased"].to_numpy(int)
    high = burden >= float(np.median(burden))
    p_value = float(result["logrank_p"])

    colors = {"High": "#C92035", "Low": "#3B82C4"}
    fig = plt.figure(
        figsize=(PAIR_W / 300, ROW_H / 300),
        dpi=300,
        facecolor="white",
    )
    grid = fig.add_gridspec(2, 1, height_ratios=[4.4, 1.15], hspace=0.10)
    ax = fig.add_subplot(grid[0])
    risk_ax = fig.add_subplot(grid[1], sharex=ax)
    for label, mask in [("Low", ~high), ("High", high)]:
        x, y = km_curve(time[mask], event[mask])
        ax.step(
            x,
            y,
            where="post",
            color=colors[label],
            linewidth=2.6,
            label=f"{label} observed burden (n={int(mask.sum())})",
        )
        censor_time = time[mask & (event == 0)]
        if len(censor_time):
            ax.plot(
                censor_time,
                survival_at(time[mask], event[mask], censor_time),
                linestyle="none",
                marker="|",
                markersize=7,
                markeredgewidth=1.4,
                color=colors[label],
            )
    ax.set_xlim(0, 2100)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Overall survival probability", fontsize=9.0, fontweight="bold")
    ax.set_xticks(np.arange(0, 2001, 400))
    ax.tick_params(axis="x", labelbottom=False)
    ax.tick_params(axis="y", labelsize=8.0)
    ax.grid(axis="y", color="#D9DEE2", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=8.0, handlelength=2.2)
    ax.text(
        0.98,
        0.96,
        f"Median OS: 554 vs 158 days\nlog-rank P = {p_value:.3g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 2.0},
    )
    risk_times = np.arange(0, 2001, 400)
    risk_ax.set_ylim(-0.1, 2.55)
    risk_ax.set_yticks([1.4, 0.45], ["High", "Low"], fontsize=8.0, fontweight="bold")
    risk_ax.set_xlabel("Days from ICI initiation", fontsize=9.0, fontweight="bold", labelpad=1)
    risk_ax.set_xticks(risk_times)
    risk_ax.tick_params(axis="x", labelsize=7.8, length=2)
    risk_ax.tick_params(axis="y", length=0, pad=6)
    risk_ax.spines[["top", "right", "left"]].set_visible(False)
    risk_ax.text(0, 2.25, "Number at risk", ha="left", va="center", fontsize=7.8, fontweight="bold")
    for row_y, mask, color in [(1.4, high, colors["High"]), (0.45, ~high, colors["Low"])]:
        for point in risk_times:
            risk_ax.text(
                point,
                row_y,
                str(int(np.sum(time[mask] >= point))),
                ha="center",
                va="center",
                fontsize=7.8,
                color=color,
                fontweight="bold",
            )
    fig.suptitle(
        "B. Observed state is associated with survival",
        x=0.008,
        y=0.995,
        ha="left",
        fontsize=15.5,
        fontweight="bold",
    )
    fig.text(
        0.50,
        0.91,
        "Observed Tc.ae + Ki67+ + <=30 um / all cells | n=30; 20 deaths",
        ha="center",
        va="top",
        fontsize=8.2,
        color="#4D5962",
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.075, right=0.995, top=0.80, bottom=0.10)
    path = PANELS / "panel_b_observed_survival.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def rounded_box(ax, xy, width, height, text, face="#FFFFFF", edge="#173F5F", size=13):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=face,
        edgecolor=edge,
        linewidth=2.0,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        fontweight="bold",
        color="#17212A" if face != "#173F5F" else "white",
        linespacing=1.10,
    )


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=17,
            linewidth=2.1,
            color="#173F5F",
            shrinkA=8,
            shrinkB=8,
        )
    )


def build_workflow_panel() -> Path:
    plt.rcParams.update({"font.family": "Arial"})
    fig, ax = plt.subplots(figsize=(COL_W / 300, ROW_H / 300), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.015,
        0.985,
        "D. Patient-held-out state model",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15.5,
        fontweight="bold",
    )
    rounded_box(ax, (0.12, 0.76), 0.76, 0.12, "Spatial training patients", face="#EAF5F7")
    arrow(ax, (0.50, 0.75), (0.50, 0.66))
    rounded_box(
        ax,
        (0.04, 0.48),
        0.43,
        0.16,
        "15 RAMP-aligned\nproteins",
        face="#F2F7FA",
        size=12,
    )
    rounded_box(
        ax,
        (0.53, 0.48),
        0.43,
        0.16,
        "Fixed spatial\ntarget label",
        face="#FFF6D9",
        edge="#C92035",
        size=12,
    )
    ax.text(0.50, 0.555, "+", ha="center", va="center", fontsize=22, fontweight="bold")
    arrow(ax, (0.50, 0.47), (0.50, 0.39))
    rounded_box(
        ax,
        (0.16, 0.26),
        0.68,
        0.105,
        "Class-balanced logistic model",
        face="#173F5F",
        edge="#173F5F",
        size=12,
    )
    arrow(ax, (0.50, 0.25), (0.50, 0.17))
    rounded_box(
        ax,
        (0.08, 0.035),
        0.84,
        0.11,
        "Score every CD8 T cell\nin the held-out patient",
        face="#F4EAFE",
        edge="#7A4FA3",
        size=12,
    )
    path = PANELS / "panel_d_workflow.png"
    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.015)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def prepare_alignment_display() -> tuple[np.ndarray, pd.DataFrame]:
    coords_path = CACHE / "ramp230_clean15_display_umap.npy"
    meta_path = CACHE / "ramp230_clean15_display_metadata.csv.gz"
    cache_manifest_path = CACHE / "ramp230_clean15_display_manifest.json"
    cache_inputs = {
        "alignment_values_sha256": sha256(ALIGNMENT_VALUES),
        "alignment_metadata_sha256": sha256(ALIGNMENT_META),
        "display_labels_sha256": sha256(DISPLAY_LABELS),
        "seed": SEED,
        "n_neighbors": 30,
        "min_dist": 0.20,
        "metric": "euclidean",
    }
    if coords_path.exists() and meta_path.exists() and cache_manifest_path.exists():
        coords = np.load(coords_path)
        metadata = pd.read_csv(meta_path, low_memory=False)
        cached = json.loads(cache_manifest_path.read_text())
        if cached == cache_inputs and coords.shape == (len(metadata), 2):
            return coords, metadata

    identity = pd.read_csv(ALIGNMENT_META, low_memory=False)
    labels = pd.read_csv(DISPLAY_LABELS, low_memory=False)
    labels = labels.copy()
    labels["display_order"] = np.arange(len(labels))
    indexed = identity.reset_index(names="alignment_row")
    display = labels.merge(
        indexed[["cell_uid", "dataset", "donor", "alignment_row"]],
        on=["cell_uid", "dataset", "donor"],
        how="left",
        validate="one_to_one",
        sort=False,
    ).sort_values("display_order")
    if display["alignment_row"].isna().any() or len(display) != 40_000:
        raise AssertionError("The 40,000 display cells do not map to the formal clean15 alignment")
    values = np.load(ALIGNMENT_VALUES, mmap_mode="r")[
        display["alignment_row"].astype(int).to_numpy()
    ]
    standardized = StandardScaler().fit_transform(values).astype(np.float32)
    import umap

    coords = umap.UMAP(
        n_neighbors=30,
        min_dist=0.20,
        n_components=2,
        metric="euclidean",
        random_state=SEED,
        transform_seed=SEED,
        n_jobs=1,
        low_memory=True,
    ).fit_transform(standardized).astype(np.float32)
    display.to_csv(meta_path, index=False, compression="gzip")
    np.save(coords_path, coords)
    cache_manifest_path.write_text(json.dumps(cache_inputs, indent=2) + "\n")
    return coords, display


def clean_axis(ax):
    ax.set(xticks=[], yticks=[])
    ax.margins(0.005)
    for spine in ax.spines.values():
        spine.set_visible(False)


def handles(colors: dict[str, str], levels: list[str], size: float = 5.5):
    return [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=colors[level],
            markeredgecolor="none",
            label=level,
            markersize=size,
        )
        for level in levels
    ]


def scatter_categories(ax, xy, values, colors, order, size=2.0):
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(len(values))
    for level in order:
        rows = shuffled[values[shuffled] == level]
        if len(rows):
            ax.scatter(
                xy[rows, 0],
                xy[rows, 1],
                s=size,
                c=colors[level],
                alpha=0.72,
                linewidths=0,
                rasterized=True,
            )
    clean_axis(ax)


def build_alignment_panel() -> Path:
    xy, metadata = prepare_alignment_display()
    dataset = metadata["dataset"].astype(str).to_numpy()
    celltype = metadata["harmonized_coarse_label"].astype(str).to_numpy()
    if set(celltype) - set(CELLTYPE_COLORS):
        raise ValueError("Missing cell-type color")
    fig = plt.figure(figsize=(PAIR_W / 300, ROW_H / 300), dpi=300, facecolor="white")
    axes = [
        fig.add_axes([0.002, 0.16, 0.508, 0.72]),
        fig.add_axes([0.490, 0.16, 0.508, 0.72]),
    ]
    scatter_categories(axes[0], xy, dataset, DATASET_COLORS, list(DATASET_COLORS), size=2.1)
    scatter_categories(axes[1], xy, celltype, CELLTYPE_COLORS, CELLTYPE_DRAW_ORDER, size=2.1)
    axes[0].set_title("Dataset", fontsize=11.2, fontweight="bold", pad=1)
    axes[1].set_title("Cell type", fontsize=11.2, fontweight="bold", pad=1)
    fig.suptitle(
        "E. RAMP alignment",
        x=0.008,
        y=0.995,
        ha="left",
        fontsize=15.5,
        fontweight="bold",
    )
    fig.legend(
        handles=handles(DATASET_COLORS, list(DATASET_COLORS), 5.4),
        loc="lower left",
        bbox_to_anchor=(0.055, 0.012),
        ncol=2,
        frameon=False,
        fontsize=7.8,
        columnspacing=0.8,
        handletextpad=0.25,
    )
    present = [level for level in CELLTYPE_COLORS if np.any(celltype == level)]
    fig.legend(
        handles=handles(CELLTYPE_COLORS, present, 5.2),
        loc="lower right",
        bbox_to_anchor=(0.985, 0.006),
        ncol=3,
        frameon=False,
        fontsize=6.9,
        columnspacing=0.75,
        handletextpad=0.22,
        labelspacing=0.30,
    )
    path = PANELS / "panel_e_alignment.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def build_metric_panel() -> Path:
    data = pd.read_csv(PATIENT_METRICS)
    data = data.loc[data["evaluable_discrimination"].astype(bool)].copy()
    if len(data) != 14:
        raise AssertionError(f"Expected 14 evaluable patients, observed {len(data)}")
    auroc = data["AUROC"].to_numpy(float)
    enrichment = data["AUPRC_over_prevalence"].to_numpy(float)
    med_auroc = float(np.median(auroc))
    med_enrichment = float(np.median(enrichment))

    fig, axes = plt.subplots(2, 1, figsize=(COL_W / 300, ROW_H / 300), dpi=300)
    rng = np.random.default_rng(20260903)
    for ax, values, color, baseline, label, median, xlim in [
        (axes[0], auroc, "#3F78B5", 0.5, "AUROC", med_auroc, (0.38, 0.92)),
        (
            axes[1],
            enrichment,
            "#E68A2E",
            1.0,
            "AUPRC / prevalence",
            med_enrichment,
            (0.8, 11.5),
        ),
    ]:
        y = rng.uniform(-0.09, 0.09, len(values))
        ax.scatter(values, y, s=34, c=color, edgecolor="white", linewidth=0.7, alpha=0.9, zorder=3)
        ax.axvline(baseline, color="#8A939B", linestyle=(0, (3, 2)), linewidth=1.2)
        ax.scatter([median], [0.20], marker="D", s=58, c="#111111", edgecolor="white", linewidth=0.8, zorder=5)
        ax.text(
            median,
            0.31,
            f"median {median:.3f}" if label == "AUROC" else f"median {median:.2f}x",
            ha="center",
            va="bottom",
            fontsize=9.3,
            fontweight="bold",
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.20, 0.50)
        ax.set_yticks([])
        ax.set_xlabel(label, fontsize=9.8, fontweight="bold", labelpad=1)
        ax.grid(axis="x", color="#DDE2E6", linewidth=0.55)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8.2)
    axes[1].set_xscale("log")
    axes[1].set_xticks([1, 2, 5, 10], ["1", "2", "5", "10"])
    axes[0].text(0.505, -0.15, "random", fontsize=7.4, color="#6A737B", ha="left")
    axes[1].text(1.03, -0.15, "random", fontsize=7.4, color="#6A737B", ha="left")
    fig.suptitle(
        "F. Patient-held-out validation",
        x=0.025,
        y=0.99,
        ha="left",
        fontsize=15.5,
        fontweight="bold",
    )
    fig.text(0.03, 0.925, "14 evaluable patients", ha="left", va="top", fontsize=10.0, color="#4D5962")
    fig.subplots_adjust(left=0.14, right=0.97, top=0.82, bottom=0.13, hspace=0.58)
    path = PANELS / "panel_f_patient_heldout_metrics.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def display_cd8_label(values: pd.Series) -> pd.Series:
    kept = {"CD8ct", "CD8ct.CCL4", "CD8ct.MT1E+", "CD8nv"}
    return values.astype(str).where(values.astype(str).isin(kept), "Other CD8")


def build_projection_panel() -> Path:
    data = pd.read_csv(PROJECTION_TABLE, low_memory=False)
    if len(data) != 5_768:
        raise AssertionError(f"Expected 5,768 CITE-seq CD8 cells, observed {len(data)}")
    xy = data[["umap_1", "umap_2"]].to_numpy(float)
    score = data["all_cd8_spatial_state_score"].to_numpy(float)
    labels = display_cd8_label(data["clust.name"]).to_numpy(str)
    colors = {
        "CD8ct": "#D73027",
        "CD8ct.CCL4": "#F46D43",
        "CD8ct.MT1E+": "#E78AC3",
        "CD8nv": "#2C7BB6",
        "Other CD8": "#8E8E8E",
    }
    label_order = ["Other CD8", "CD8nv", "CD8ct", "CD8ct.CCL4", "CD8ct.MT1E+"]

    fig = plt.figure(figsize=(PAIR_W / 300, ROW_H / 300), dpi=300, facecolor="white")
    axes = [
        fig.add_axes([0.002, 0.16, 0.508, 0.72]),
        fig.add_axes([0.490, 0.16, 0.508, 0.72]),
    ]
    scatter_categories(axes[0], xy, labels, colors, label_order, size=5.0)
    axes[0].set_title("Provider CD8 states", fontsize=11.2, fontweight="bold", pad=1)
    present = [level for level in label_order if np.any(labels == level)]
    fig.legend(
        handles=handles(colors, present, 5.5),
        loc="lower left",
        bbox_to_anchor=(0.055, 0.012),
        ncol=2,
        frameon=False,
        fontsize=7.2,
        columnspacing=0.7,
        handletextpad=0.25,
        labelspacing=0.3,
    )
    order = np.argsort(score, kind="mergesort")
    dots = axes[1].scatter(
        xy[order, 0],
        xy[order, 1],
        c=score[order],
        cmap="magma",
        vmin=0,
        vmax=0.9,
        s=5.2,
        alpha=0.93,
        linewidths=0,
        rasterized=True,
    )
    clean_axis(axes[1])
    axes[1].set_title("Transferred spatial-state score", fontsize=11.2, fontweight="bold", pad=1)
    cbar_ax = fig.add_axes([0.59, 0.075, 0.34, 0.018])
    cbar = fig.colorbar(dots, cax=cbar_ax, orientation="horizontal", extend="max")
    cbar.set_ticks([0, 0.3, 0.6, 0.9])
    cbar.ax.tick_params(labelsize=7.5, length=2, pad=1)
    cbar.set_label("Predicted tumor-proximal Ki67+ CD8 state", fontsize=8.2, labelpad=1)
    cbar.outline.set_linewidth(0.45)
    fig.suptitle(
        "G. Transfer to CITE-seq CD8 T cells",
        x=0.008,
        y=0.995,
        ha="left",
        fontsize=15.5,
        fontweight="bold",
    )
    path = PANELS / "panel_g_citeseq_transfer.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def build_biomarker_panel() -> Path:
    effects = pd.read_csv(HELDOUT_EFFECTS)
    marker_map = {
        "CD11a-AB": "CD11a",
        "CD18-AB": "CD18",
        "CD2-AB": "CD2",
        "CD279-AB": "PD-1",
    }
    effects = effects.loc[
        effects["normalization"].eq("classical_clr")
        & effects["adt_feature"].isin(marker_map)
    ].copy()
    effects["marker"] = effects["adt_feature"].map(marker_map)
    order = ["CD11a", "CD18", "CD2", "PD-1"]
    palette = {
        "CD11a": "#0072B2",
        "CD18": "#009E73",
        "CD2": "#CC79A7",
        "PD-1": "#D55E00",
    }
    modules = pd.read_csv(MODULES)
    module = modules.loc[
        modules["module"].eq("core_4_LFA1_CD2_PD1")
        & modules["population"].eq("all_CD8")
    ].iloc[0]

    fig, ax = plt.subplots(figsize=(RIGHT_W / 300, ROW_H / 300), dpi=300)
    rng = np.random.default_rng(20260831)
    for x, marker in enumerate(order):
        values = effects.loc[effects["marker"].eq(marker), "top25_minus_rest75_rank"].to_numpy(float)
        jitter = rng.uniform(-0.10, 0.10, len(values))
        ax.scatter(
            np.full(len(values), x) + jitter,
            values,
            s=35,
            c=palette[marker],
            edgecolors="white",
            linewidths=0.6,
            alpha=0.92,
            zorder=3,
        )
        mean = float(np.mean(values))
        ax.plot([x - 0.22, x + 0.22], [mean, mean], color="#111111", lw=2.2)
        positive = int((values > 0).sum())
        ax.text(
            x,
            max(0.30, float(np.max(values)) + 0.025),
            f"{positive}/{len(values)} up",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color=palette[marker],
        )
    ax.axhline(0, color="#777777", linestyle=(0, (3, 2)), linewidth=1.0)
    ax.set_ylim(-0.035, 0.45)
    ax.set_xticks(range(4), order, fontsize=10.0, fontweight="bold")
    ax.set_ylabel("High-state - other CD8 (rank)", fontsize=8.4)
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.6, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.text(
        0.98,
        0.05,
        f"4-marker module\nmean patient rho = {float(module['mean_patient_spearman']):.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        color="#252525",
        bbox={"facecolor": "white", "edgecolor": "#AAB2BA", "pad": 3.0},
    )
    fig.suptitle(
        "H. Held-out protein program",
        x=0.025,
        y=0.99,
        ha="left",
        fontsize=15.5,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.925,
        "LFA-1/CD2 adhesion with PD-1 checkpoint",
        ha="left",
        va="top",
        fontsize=9.0,
        color="#4D5962",
    )
    fig.subplots_adjust(left=0.17, right=0.98, top=0.80, bottom=0.15)
    path = PANELS / "panel_h_heldout_biomarkers.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


def assemble(paths: dict[str, Path]) -> tuple[Path, Path]:
    canvas_h = ROW_H * 3 + GAP_Y * 2
    canvas = Image.new("RGB", (TOTAL_W, canvas_h), "white")
    x_left = 0
    x_middle = COL_W + GAP_X
    x_right = x_middle + PAIR_W + GAP_X
    y = [0, ROW_H + GAP_Y, 2 * (ROW_H + GAP_Y)]

    placements = [
        ("A", x_left, y[0]),
        ("B", x_middle, y[0]),
        ("C", x_right, y[0]),
        ("D", x_left, y[1]),
        ("E", x_middle, y[1]),
        ("F", x_left, y[2]),
        ("G", x_middle, y[2]),
        ("H", x_right, y[2]),
    ]
    for key, left, top in placements:
        image = Image.open(paths[key]).convert("RGB")
        if key in {"B", "E", "G"}:
            expected_w = PAIR_W
        elif key in {"C", "H"}:
            expected_w = RIGHT_W
        else:
            expected_w = COL_W
        expected_h = ROW_H * 2 + GAP_Y if key == "C" else ROW_H
        image = cover(image, expected_w, expected_h)
        canvas.paste(image, (left, top))

    png = OUTPUT / "melanoma_cd8_three_row_figure.png"
    pdf = PDF_OUTPUT / "melanoma_cd8_three_row_figure.pdf"
    canvas.save(png, dpi=(300, 300), optimize=True)
    canvas.save(pdf, resolution=300.0)
    return png, pdf


def main() -> None:
    required = [
        ALIGNMENT_VALUES,
        ALIGNMENT_META,
        ALIGNMENT_MANIFEST,
        DISPLAY_LABELS,
        PANEL_A_SOURCE,
        SURVIVAL_FEATURES,
        SURVIVAL_RESULTS,
        SPATIAL_PAIR_SOURCE,
        PROJECTION_TABLE,
        PATIENT_METRICS,
        HELDOUT_EFFECTS,
        MODULES,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing}")
    manifest = json.loads(ALIGNMENT_MANIFEST.read_text())
    if manifest.get("algorithm_version") != "2.3.0":
        raise AssertionError("Expected formal clean15 RAMP 2.3.0 alignment")

    panel_a, panel_b, panel_c = build_observed_panels()
    panel_d = build_workflow_panel()
    panel_e = build_alignment_panel()
    panel_f = build_metric_panel()
    panel_g = build_projection_panel()
    panel_h = build_biomarker_panel()
    panels = {
        "A": panel_a,
        "B": panel_b,
        "C": panel_c,
        "D": panel_d,
        "E": panel_e,
        "F": panel_f,
        "G": panel_g,
        "H": panel_h,
    }
    png, pdf = assemble(panels)

    metrics = pd.read_csv(PATIENT_METRICS)
    metrics = metrics.loc[metrics["evaluable_discrimination"].astype(bool)]
    output_manifest = {
        "schema_version": "final_xu.figure_melanoma.three_row.v1",
        "algorithm": "RAMP",
        "algorithm_version": "2.3.0",
        "ramp_source": manifest.get("canonical_source"),
        "ramp_source_sha256": manifest.get("canonical_source_sha256"),
        "layout": {
            "rows": 3,
            "columns": 4,
            "survival_spans_columns": 2,
            "ground_truth_and_prediction_vertical_pair": True,
        },
        "validation": {
            "evaluable_patients": int(len(metrics)),
            "median_AUROC": float(metrics["AUROC"].median()),
            "median_AUPRC_over_prevalence": float(metrics["AUPRC_over_prevalence"].median()),
        },
        "analysis_root": str(ANALYSIS_ROOT.resolve()),
        "inputs_root": str(INPUTS.resolve()),
        "no_test_directory_dependency": True,
        "no_raw_data_copied": True,
        "panels": {key: str(path.resolve()) for key, path in panels.items()},
        "sources": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in required
        ],
        "outputs": {
            "png": str(png.resolve()),
            "pdf": str(pdf.resolve()),
        },
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n")
    print(json.dumps(output_manifest, indent=2))


if __name__ == "__main__":
    main()
