#!/usr/bin/env python3
"""Build the 2 x 4 Harmony UMAP overview from the eight formal PNGs.

The overview preserves every source PNG at its native pixel dimensions.  No
resizing, resampling, cropping, or cached intermediate image is used.  Rows are
Harmony input variants and columns are benchmark datasets, matching the
historical overview layout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "harmony_umap_overview.png"
MANIFEST = ROOT / "harmony_umap_overview_manifest.json"

BENCHMARKS = ("bone_marrow", "tnbc", "melanoma", "lung_cancer")
VARIANTS = ("source_scale", "adtnorm_fixed_arcsinh")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_path(benchmark: str, variant: str) -> Path:
    return (
        ROOT
        / benchmark
        / "harmony"
        / variant
        / "umap"
        / "figures"
        / f"{benchmark}_harmony_{variant}_umap.png"
    )


def main() -> None:
    panels: dict[tuple[str, str], Image.Image] = {}
    sources: list[dict[str, object]] = []

    for variant in VARIANTS:
        for benchmark in BENCHMARKS:
            path = source_path(benchmark, variant)
            if not path.is_file():
                raise FileNotFoundError(f"Missing formal Harmony UMAP: {path}")
            with Image.open(path) as loaded:
                panel = loaded.copy()
            panels[(variant, benchmark)] = panel
            sources.append(
                {
                    "benchmark": benchmark,
                    "variant": variant,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                    "width": panel.width,
                    "height": panel.height,
                    "mode": panel.mode,
                }
            )

    column_widths = [
        max(panels[(variant, benchmark)].width for variant in VARIANTS)
        for benchmark in BENCHMARKS
    ]
    row_heights = [
        max(panels[(variant, benchmark)].height for benchmark in BENCHMARKS)
        for variant in VARIANTS
    ]

    canvas = Image.new(
        "RGBA", (sum(column_widths), sum(row_heights)), (255, 255, 255, 255)
    )
    placements: list[dict[str, object]] = []

    y_base = 0
    for row, variant in enumerate(VARIANTS):
        x_base = 0
        for column, benchmark in enumerate(BENCHMARKS):
            panel = panels[(variant, benchmark)]
            x = x_base + (column_widths[column] - panel.width) // 2
            y = y_base + (row_heights[row] - panel.height) // 2
            canvas.paste(panel, (x, y))

            # Fail closed if the overview did not retain the exact source pixels.
            if canvas.crop((x, y, x + panel.width, y + panel.height)).tobytes() != panel.tobytes():
                raise RuntimeError(
                    f"Pixel-identity check failed for {benchmark}/{variant}"
                )
            placements.append(
                {
                    "benchmark": benchmark,
                    "variant": variant,
                    "row": row,
                    "column": column,
                    "x": x,
                    "y": y,
                    "width": panel.width,
                    "height": panel.height,
                    "resized": False,
                }
            )
            x_base += column_widths[column]
        y_base += row_heights[row]

    canvas.save(OUTPUT)

    manifest = {
        "schema_version": 1,
        "description": "Native-resolution 2 x 4 overview of formal Harmony UMAP PNGs",
        "layout": {
            "rows": list(VARIANTS),
            "columns": list(BENCHMARKS),
            "column_widths": column_widths,
            "row_heights": row_heights,
            "resizing_or_resampling": False,
        },
        "sources": sources,
        "placements": placements,
        "output": {
            "path": OUTPUT.name,
            "sha256": sha256(OUTPUT),
            "width": canvas.width,
            "height": canvas.height,
            "mode": canvas.mode,
        },
        "script": {
            "path": Path(__file__).name,
            "sha256": sha256(Path(__file__).resolve()),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for panel in panels.values():
        panel.close()
    canvas.close()

    print(
        f"PASS: wrote {OUTPUT.name} from {len(sources)} current formal PNGs "
        f"at native resolution ({manifest['output']['width']} x "
        f"{manifest['output']['height']})"
    )


if __name__ == "__main__":
    main()
