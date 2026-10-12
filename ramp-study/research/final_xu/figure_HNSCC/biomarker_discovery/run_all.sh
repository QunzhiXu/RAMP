#!/bin/sh
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

python3 "$HERE/01_project_spatial_state.py"
python3 "$HERE/02_screen_biomarkers.py"
python3 "$HERE/03_validate_discovery.py"
python3 "$HERE/04_plot_cd8_transfer_program_umap.py"
