#!/usr/bin/env python3
"""Assemble the current CD19 story panels into one review figure."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops


HERE = Path(__file__).resolve().parent
FIGURE = HERE.parent
SOURCES = {
    "A": FIGURE / "panel_A" / "panel_A_failed_codex_cd19.png",
    "B": FIGURE / "panel_B" / "panel_B_qc_and_cd19_specificity.png",
    "C": FIGURE / "panel_C" / "panel_C_cd19_ridges_before_after.png",
    "D": FIGURE / "panel_D" / "panel_D_prediction_workflow.png",
    "E": FIGURE / "panel_E" / "panel_E_heldout_marker_validation.png",
    "F": FIGURE / "panel_F" / "panel_F_cell_coordinate_validation.png",
    "G": FIGURE / "panel_G" / "panel_G_cd19_umap_comparison.png",
}

MARGIN = 80
GAP = 70
BOTTOM_GAP = 28
WHITE = (255, 255, 255)


def crop_white(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGB")
    white = Image.new("RGB", image.size, WHITE)
    difference = ImageChops.difference(image, white).convert("L")
    thresholded = difference.point(lambda value: 255 if value > 8 else 0)
    box = thresholded.getbbox()
    if box is None:
        raise RuntimeError(f"No visible content in {path}")
    left, top, right, bottom = box
    padding = 20
    return image.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(image.width, right + padding),
            min(image.height, bottom + padding),
        )
    )


def resize_height(image: Image.Image, height: int) -> Image.Image:
    width = round(image.width * height / image.height)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def resize_width(image: Image.Image, width: int) -> Image.Image:
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def main() -> None:
    images = {label: crop_white(path) for label, path in SOURCES.items()}

    # Top-left row: microscopy failure and quantitative QC.
    top_height = 1600
    top_a = resize_height(images["A"], top_height)
    top_b = resize_height(images["B"], top_height)
    left_width = top_a.width + GAP + top_b.width

    # Second-left row: workflow and held-out validation, fitted to the same width.
    d_ratio = images["D"].width / images["D"].height
    e_ratio = images["E"].width / images["E"].height
    second_height = round((left_width - GAP) / (d_ratio + e_ratio))
    second_d = resize_height(images["D"], second_height)
    second_e = resize_height(images["E"], second_height)
    second_total = second_d.width + GAP + second_e.width

    # C is the final column and spans both upper rows.
    upper_height = top_height + GAP + second_height
    upper_c = resize_height(images["C"], upper_height)
    content_width = left_width + GAP + upper_c.width

    # Final row: spatial validation and complete UMAP comparison at equal widths.
    bottom_slot_width = (content_width - BOTTOM_GAP) // 2
    bottom_f = resize_width(images["F"], bottom_slot_width)
    bottom_g = resize_width(images["G"], bottom_slot_width)
    bottom_height = max(bottom_f.height, bottom_g.height)

    canvas_height = (
        MARGIN + upper_height + GAP + bottom_height + MARGIN
    )
    canvas_width = content_width + 2 * MARGIN
    canvas = Image.new("RGB", (canvas_width, canvas_height), WHITE)

    y = MARGIN
    canvas.paste(top_a, (MARGIN, y))
    canvas.paste(top_b, (MARGIN + top_a.width + GAP, y))
    canvas.paste(upper_c, (MARGIN + left_width + GAP, y))

    y += top_height + GAP
    second_x = MARGIN + max(0, (left_width - second_total) // 2)
    canvas.paste(second_d, (second_x, y))
    canvas.paste(second_e, (second_x + second_d.width + GAP, y))

    y = MARGIN + upper_height + GAP
    canvas.paste(bottom_f, (MARGIN, y))
    canvas.paste(bottom_g, (MARGIN + bottom_slot_width + BOTTOM_GAP, y))

    png = HERE / "cd19_story_panels_A_G.png"
    pdf = HERE / "cd19_story_panels_A_G.pdf"
    canvas.save(png, optimize=True)
    canvas.save(pdf, "PDF", resolution=300.0)
    print(f"Saved {png} ({canvas.width} x {canvas.height})")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
