#!/bin/zsh
set -euo pipefail

here="${0:A:h}"
python3 "$here/01_prepare_pt70.py"
python3 "$here/02_run_ramp_ver2.py"
python3 "$here/03_plot_full_umap.py"
