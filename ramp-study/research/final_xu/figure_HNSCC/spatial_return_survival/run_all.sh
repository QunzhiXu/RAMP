#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

python3 01_return_fixed_program.py
python3 02_build_top40_features.py
Rscript 03_run_survival.R
python3 04_plot_spatial_validation.py
python3 05_validate_outputs.py
