#!/usr/bin/env python3
"""Assemble the current A--D panels at their native 320-dpi scale."""

from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
BIG_FIGURE = HERE.parent
PANEL_PATHS = {
    "A": BIG_FIGURE / "panel_A" / "panel_A.png",
    "B": BIG_FIGURE / "panel_B" / "panel_B.png",
    "C": BIG_FIGURE / "panel_C" / "panel_C.png",
    "D": BIG_FIGURE / "panel_D" / "panel_D.png",
}
DPI = 320
TOP_GUTTER = 28
BOTTOM_GUTTER = 0
VERTICAL_GUTTER = 28


def main() -> None:
    panels: dict[str, Image.Image] = {}
    for label, path in PANEL_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(path)
        panels[label] = Image.open(path).convert("RGB")

    heights = {panel.height for panel in panels.values()}
    if len(heights) != 1:
        raise AssertionError(f"Panel heights differ: {sorted(heights)}")
    row_height = heights.pop()

    top_width = panels["A"].width + TOP_GUTTER + panels["B"].width
    bottom_width = panels["C"].width + BOTTOM_GUTTER + panels["D"].width
    canvas_width = max(top_width, bottom_width)
    canvas_height = row_height * 2 + VERTICAL_GUTTER
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    # Center each row while preserving every panel's native size.  This makes
    # any row-width imbalance visible without changing relative font sizes.
    top_x = (canvas_width - top_width) // 2
    bottom_x = (canvas_width - bottom_width) // 2
    canvas.paste(panels["A"], (top_x, 0))
    canvas.paste(panels["B"], (top_x + panels["A"].width + TOP_GUTTER, 0))
    canvas.paste(panels["C"], (bottom_x, row_height + VERTICAL_GUTTER))
    canvas.paste(
        panels["D"],
        (bottom_x + panels["C"].width + BOTTOM_GUTTER, row_height + VERTICAL_GUTTER),
    )

    png = HERE / "panels_ABCD_native_scale.png"
    pdf = HERE / "panels_ABCD_native_scale.pdf"
    canvas.save(png, dpi=(DPI, DPI))
    canvas.save(pdf, "PDF", resolution=float(DPI), quality=100)
    print(png)
    print(pdf)
    print(f"top row width: {top_width}px")
    print(f"bottom row width: {bottom_width}px")


if __name__ == "__main__":
    main()
