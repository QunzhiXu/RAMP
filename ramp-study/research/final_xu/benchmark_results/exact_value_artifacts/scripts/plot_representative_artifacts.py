#!/usr/bin/env python3
"""Render representative exact-value artifacts in the Tietscher-CD68 style."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
from pathlib import Path

CACHE = Path(
    os.environ.get(
        "RAMP_ARTIFACT_CACHE",
        str(Path(tempfile.gettempdir()) / "ramp-exact-value-artifact-cache"),
    )
)
for variable, subdirectory in (("MPLCONFIGDIR", "matplotlib"), ("XDG_CACHE_HOME", "xdg")):
    destination = CACHE / subdirectory
    destination.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(variable, str(destination))
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
SUMMARY = ROOT / "summary"
OUTPUT = SUMMARY / "artifact_examples"
AUDIT_SCRIPT = HERE.with_name("run_exact_value_audit.py")
HITS_PATH = SUMMARY / "canonical_artifact_hits.csv"


REPRESENTATIVES = (
    {
        "case": "tnbc",
        "dataset": "Tietscher IMC",
        "marker": "CD68",
        "expected_target": "Myeloid",
        "non_target_celltype": "Endothelial",
        "driver_method": "Harmony-direct[source_scale]",
    },
    {
        "case": "tnbc",
        "dataset": "Meyer IMC",
        "marker": "CD31",
        "expected_target": "Endothelial",
        "non_target_celltype": "Stroma",
        "driver_method": "Harmony-direct[source_scale]",
    },
    {
        "case": "bone_marrow",
        "dataset": "BM CyTOF",
        "marker": "CD11B",
        "expected_target": "Monocyte;Macrophage;Other myeloid",
        "non_target_celltype": "Pre / immature B",
        "driver_method": "Harmony-direct[source_scale]",
    },
    {
        "case": "lung_cancer",
        "dataset": "Cords IMC",
        "marker": "CD31",
        "expected_target": "Endothelial",
        "non_target_celltype": "CD4 T",
        "driver_method": "Harmony-direct[source_scale]",
    },
    {
        "case": "tnbc",
        "dataset": "Meyer IMC",
        "marker": "PANCK",
        "expected_target": "Tumor / epithelial",
        "non_target_celltype": "B lineage",
        "driver_method": "fastMNN-marker[d=NA]",
    },
    {
        "case": "lung_cancer",
        "dataset": "Cords IMC",
        "marker": "CD3",
        "expected_target": "CD4 T;CD8 T",
        "non_target_celltype": "Endothelial",
        "driver_method": "Harmony-direct[fixed_arcsinh]",
    },
    {
        "case": "tnbc",
        "dataset": "Tietscher IMC",
        "marker": "PANCK",
        "expected_target": "Tumor / epithelial",
        "non_target_celltype": "Endothelial",
        "driver_method": "fastMNN-marker[d=NA]",
    },
    {
        "case": "bone_marrow",
        "dataset": "Triana AbSeq",
        "marker": "CD33",
        "expected_target": "Monocyte;Macrophage;Other myeloid;Dendritic",
        "non_target_celltype": "CD4 T",
        "driver_method": "fastMNN-marker[d=NA]",
    },
)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_audit_module():
    spec = importlib.util.spec_from_file_location("exact_value_audit", AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {AUDIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    hits = pd.read_csv(HITS_PATH)
    audit = load_audit_module()
    cache: dict[str, tuple] = {}
    selected_rows = []
    statistics = []
    pdf_path = OUTPUT / "representative_artifact_percentile_boxplots.pdf"
    with PdfPages(pdf_path) as pdf:
        for number, selection in enumerate(REPRESENTATIVES, start=1):
            mask = pd.Series(True, index=hits.index)
            for column in (
                "case",
                "dataset",
                "marker",
                "expected_target",
                "non_target_celltype",
                "method",
            ):
                key = "driver_method" if column == "method" else column
                mask &= hits[column].astype(str).eq(str(selection[key]))
            match = hits[mask]
            if len(match) != 1:
                raise ValueError(f"representative selection did not resolve uniquely: {selection}")
            hit = match.iloc[0]
            selected_rows.append(hit.to_dict())
            case = str(selection["case"])
            if case not in cache:
                cache[case] = audit.load_case(case)
            metadata, markers, raw, methods, _ = cache[case]
            filename = (
                f"{number:02d}_{slug(case)}_{slug(str(selection['dataset']))}_"
                f"{slug(str(selection['marker']))}_{slug(str(selection['non_target_celltype']))}.png"
            )
            rows = audit.plot_box_grid(
                case,
                str(selection["dataset"]),
                str(selection["marker"]),
                audit.split_labels(selection["expected_target"]),
                str(selection["non_target_celltype"]),
                metadata,
                raw,
                methods,
                markers,
                display_scale="percentile",
                output_png=OUTPUT / filename,
                pdf=pdf,
            )
            for row in rows:
                row["example_number"] = number
                row["driver_method"] = selection["driver_method"]
                row["highlighted_non_target"] = selection["non_target_celltype"]
            statistics.extend(rows)

    selection_frame = pd.DataFrame(selected_rows)
    selection_frame.insert(0, "example_number", range(1, len(selection_frame) + 1))
    selection_frame.to_csv(OUTPUT / "representative_artifact_selection.csv", index=False)
    pd.DataFrame(statistics).to_csv(
        OUTPUT / "representative_artifact_boxplot_statistics.csv", index=False
    )

    output_paths = sorted(
        [path for path in OUTPUT.iterdir() if path.is_file() and path.name != "manifest.json"],
        key=lambda path: path.name,
    )
    manifest = {
        "schema_version": "final_xu.representative_artifact_plots.v1",
        "plot_script_sha256": sha256_file(HERE),
        "audit_script_sha256": sha256_file(AUDIT_SCRIPT),
        "canonical_hits_sha256": sha256_file(HITS_PATH),
        "display_scale": "within-dataset midrank percentile",
        "examples": len(REPRESENTATIVES),
        "selection_policy": "fixed representative canonical hits spanning formal Harmony source-scale, Harmony fixed-arcsinh, and the explicitly non-default fastMNN d=NA stress test; after the TNBC Meyer seed-20260960 refresh removed every cyCombine canonical hit, the obsolete Meyer CD8-in-CD4-T example was replaced by the refreshed Meyer's highest-severity canonical hit (fastMNN d=NA, PANCK-in-B-lineage)",
        "output_files": [
            {
                "file": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in output_paths
        ],
    }
    temporary = OUTPUT / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(OUTPUT / "manifest.json")
    print(f"rendered {len(REPRESENTATIVES)} representative artifacts", flush=True)


if __name__ == "__main__":
    main()
