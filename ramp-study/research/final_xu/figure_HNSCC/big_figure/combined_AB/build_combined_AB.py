#!/usr/bin/env python3
"""Combine the selected Panel A and Panel B without rescaling either panel."""

from pathlib import Path

from PIL import Image


HERE = Path(__file__).resolve().parent
BIG_FIGURE = HERE.parent
PANEL_A = BIG_FIGURE / "panel_A" / "panel_A.png"
PANEL_B = BIG_FIGURE / "panel_B" / "panel_B.png"
DPI = 320
GUTTER = 28


def main() -> None:
    with Image.open(PANEL_A) as source_a, Image.open(PANEL_B) as source_b:
        panel_a = source_a.convert("RGB")
        panel_b = source_b.convert("RGB")

    # Preserve each panel's native 320-dpi scale so their font sizes remain
    # comparable.  Top alignment keeps the panel letters and titles level.
    width = panel_a.width + GUTTER + panel_b.width
    height = max(panel_a.height, panel_b.height)
    combined = Image.new("RGB", (width, height), "white")
    combined.paste(panel_a, (0, 0))
    combined.paste(panel_b, (panel_a.width + GUTTER, 0))

    png = HERE / "panels_AB.png"
    pdf = HERE / "panels_AB.pdf"
    combined.save(png, dpi=(DPI, DPI))
    combined.save(pdf, "PDF", resolution=float(DPI), quality=100)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
