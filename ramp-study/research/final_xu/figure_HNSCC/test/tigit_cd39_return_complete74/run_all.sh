#!/bin/sh
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

python3 "$HERE/01_prepare_complete74.py"
python3 "$HERE/02_run_ramp_ver2.py"
python3 "$HERE/03_project_fixed_pair_to_codex.py"
Rscript "$HERE/04_run_survival.R"
python3 "$HERE/05_validate_outputs.py"
python3 "$HERE/06_plot_complete74_umap.py"
