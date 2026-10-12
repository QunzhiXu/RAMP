#!/usr/bin/env python3
"""Audit and summarize all four canonical fastMNN protein benchmarks."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "fastmnn_benchmark_runner.py"
METHOD_LABEL = "fastMNN (batchelor 1.18.1; corrected embedding)"
CONFIGS = [
    HERE / "bone_marrow" / "fastmnn" / "benchmark_config.json",
    HERE / "tnbc" / "fastmnn" / "benchmark_config.json",
    HERE / "melanoma" / "fastmnn" / "config.json",
    HERE / "lung_cancer" / "fastmnn" / "benchmark_config.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(value: object, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def build_overview(images: list[Path], output: Path) -> None:
    cell_width, cell_height, margin = 2400, 1320, 24
    canvas = Image.new(
        "RGB",
        (2 * cell_width + 3 * margin, 2 * cell_height + 3 * margin),
        "white",
    )
    for index, path in enumerate(images):
        image = Image.open(path).convert("RGB")
        image.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
        row, column = divmod(index, 2)
        x0 = margin + column * (cell_width + margin) + (cell_width - image.width) // 2
        y0 = margin + row * (cell_height + margin) + (cell_height - image.height) // 2
        canvas.paste(image, (x0, y0))
    canvas.save(output, format="PNG", optimize=True)


def main() -> None:
    rows: list[dict[str, object]] = []
    overview_images: list[Path] = []
    records: list[dict[str, object]] = []
    for config_path in CONFIGS:
        subprocess.run(
            [sys.executable, str(RUNNER), "audit", "--config", str(config_path)],
            check=True,
            cwd=HERE.parent,
        )
        config = load_json(config_path)
        method_dir = config_path.parent
        paths = {
            key: (method_dir / value).resolve()
            for key, value in config["paths"].items()
        }
        alignment_manifest_path = paths["alignment_output"] / "manifest.json"
        umap_manifest_path = paths["umap_output"] / "umap_manifest.json"
        preprocessing_manifest_path = paths["preprocessing_manifest"]
        alignment = load_json(alignment_manifest_path)
        umap = load_json(umap_manifest_path)
        preprocessing = load_json(preprocessing_manifest_path)
        merge = pd.read_csv(paths["alignment_output"] / "fastmnn_merge_diagnostics.csv")
        lost = pd.read_csv(paths["alignment_output"] / "fastmnn_lost_variance.csv")
        l2 = pd.read_csv(paths["alignment_output"] / "input_cell_l2_statistics.csv")
        png = paths["umap_output"] / "figures" / f"{config['benchmark_slug']}_fastmnn_umap.png"
        pdf = paths["umap_output"] / "figures" / f"{config['benchmark_slug']}_fastmnn_umap.pdf"
        if not png.is_file() or not pdf.is_file():
            raise FileNotFoundError(f"UMAP figure pair missing for {config['benchmark_slug']}")
        overview_images.append(png)

        flags = int(lost["review_flag_over_10_percent"].astype(bool).sum())
        rows.append(
            {
                "benchmark": config["benchmark_slug"],
                "cells": int(config["expected_cells"]),
                "markers": len(config["markers"]),
                "corrected_dimensions": int(alignment["algorithm"]["actual_d"]),
                "datasets": len(config["datasets"]),
                "merge_steps": len(merge),
                "all_merges_completed": bool((~merge["skipped"].astype(bool)).all()),
                "total_mnn_pairs": int(merge["mnn_pairs"].sum()),
                "maximum_lost_variance_percent": float(lost["lost_variance_percent"].max()),
                "lost_variance_rows_over_10_percent": flags,
                "zero_l2_cells": int(l2["zero_l2_n"].sum()),
                "umap_initialization": str(config["umap"]["initialization"]),
                "direct_corrected_embedding_umap": umap["pre_umap_processing"]
                == "direct corrected fastMNN embedding; no z-score; no PCA",
                "labels_used_for_alignment_or_umap_fit": bool(
                    umap["labels_used_for_alignment"] or umap["labels_used_for_umap_fitting"]
                ),
                "status": "complete",
            }
        )
        records.append(
            {
                "benchmark": config["benchmark_slug"],
                "config": str(config_path),
                "config_sha256": sha256(config_path),
                "preprocessing_manifest": str(preprocessing_manifest_path),
                "preprocessing_manifest_sha256": sha256(preprocessing_manifest_path),
                "declared_zero_l2_cells": preprocessing["zero_l2_cell_count"],
                "alignment_manifest": str(alignment_manifest_path),
                "alignment_manifest_sha256": sha256(alignment_manifest_path),
                "umap_manifest": str(umap_manifest_path),
                "umap_manifest_sha256": sha256(umap_manifest_path),
                "umap_png": str(png),
                "umap_png_sha256": sha256(png),
                "umap_pdf": str(pdf),
                "umap_pdf_sha256": sha256(pdf),
            }
        )

    summary_path = HERE / "fastmnn_benchmark_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False, float_format="%.12g")
    overview_path = HERE / "fastmnn_umap_overview.png"
    build_overview(overview_images, overview_path)
    audit = {
        "schema_version": "final_xu.fastmnn_global_audit.v1",
        "status": "complete",
        "method_label": METHOD_LABEL,
        "shared_runner": str(RUNNER),
        "shared_runner_sha256": sha256(RUNNER),
        "benchmarks": records,
        "summary_csv": str(summary_path),
        "summary_csv_sha256": sha256(summary_path),
        "overview_png": str(overview_path),
        "overview_png_sha256": sha256(overview_path),
        "all_four_complete": len(rows) == 4 and all(row["status"] == "complete" for row in rows),
        "all_umaps_direct_corrected_embedding": all(
            bool(row["direct_corrected_embedding_umap"]) for row in rows
        ),
        "any_labels_used_for_alignment_or_umap_fit": any(
            bool(row["labels_used_for_alignment_or_umap_fit"]) for row in rows
        ),
    }
    audit_path = HERE / "fastmnn_reproducibility_audit.json"
    write_json(audit, audit_path)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f"PASS: global fastMNN audit -> {audit_path}", flush=True)


if __name__ == "__main__":
    main()
