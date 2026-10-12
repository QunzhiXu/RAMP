#!/usr/bin/env python3
"""Assemble current HNSCC Panels A--F at native 320-dpi scale."""

from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
BIG_FIGURE = HERE.parent
PANEL_PATHS = {
    label: BIG_FIGURE / f"panel_{label}" / f"panel_{label}.png"
    for label in "ABCDEF"
}
DPI = 320
ROW_GAPS = {"AB": 28, "CD": 0, "EF": 0}
VERTICAL_GUTTER = 28


def paste_row(
    canvas: Image.Image,
    panels: dict[str, Image.Image],
    labels: str,
    y: int,
    gap: int,
) -> int:
    row_width = sum(panels[label].width for label in labels) + gap * (len(labels) - 1)
    x = (canvas.width - row_width) // 2
    for label in labels:
        canvas.paste(panels[label], (x, y))
        x += panels[label].width + gap
    return row_width


def main() -> None:
    panels: dict[str, Image.Image] = {}
    for label, path in PANEL_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        panels[label] = Image.open(path).convert("RGB")

    if panels["A"].height != panels["B"].height:
        raise AssertionError("Panels A and B must have equal native heights")
    if panels["C"].height != panels["D"].height:
        raise AssertionError("Panels C and D must have equal native heights")
    if panels["E"].height != panels["F"].height:
        raise AssertionError("Panels E and F must have equal native heights")

    row_specs = [("AB", ROW_GAPS["AB"]), ("CD", ROW_GAPS["CD"]), ("EF", ROW_GAPS["EF"])]
    row_widths = {
        labels: sum(panels[label].width for label in labels) + gap
        for labels, gap in row_specs
    }
    canvas_width = max(row_widths.values())
    canvas_height = (
        panels["A"].height + panels["C"].height + panels["E"].height
        + 2 * VERTICAL_GUTTER
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    y = 0
    for labels, gap in row_specs:
        paste_row(canvas, panels, labels, y, gap)
        y += panels[labels[0]].height + VERTICAL_GUTTER

    png = HERE / "panels_ABCDEF_native_scale.png"
    pdf = HERE / "panels_ABCDEF_native_scale.pdf"
    canvas.save(png, dpi=(DPI, DPI))
    canvas.save(pdf, "PDF", resolution=float(DPI), quality=100)
    print(png)
    print(pdf)
    for labels in ("AB", "CD", "EF"):
        print(f"{labels} row width: {row_widths[labels]}px")


if __name__ == "__main__":
    main()
