#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIGURE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python3 "$FIGURE_ROOT/analysis/alignment/run_alignment.py"
python3 "$FIGURE_ROOT/analysis/all_cd8_prediction/run_all_cd8_patient_heldout.py"
python3 "$FIGURE_ROOT/analysis/all_cd8_prediction/scan_visual_rois.py"
python3 "$FIGURE_ROOT/analysis/all_cd8_prediction/make_all_cd8_spatial_figure.py"
python3 "$FIGURE_ROOT/analysis/all_cd8_prediction/build_all_cd8_citeseq_projection.py"
python3 "$FIGURE_ROOT/analysis/all_cd8_prediction/plot_all_cd8_citeseq_umap.py"
python3 "$FIGURE_ROOT/analysis/all_cd8_prediction/analyze_all_cd8_biology.py"
python3 "$FIGURE_ROOT/01_build_three_row_figure.py"
python3 "$FIGURE_ROOT/02_validate_formal_story.py"
