#!/bin/sh
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

python3 "$HERE/01_prepare_alignment_input.py"
python3 "$HERE/02_run_ramp_ver2.py"
python3 "$HERE/03_plot_umap.py"
python3 "$HERE/04_validate_alignment.py"
