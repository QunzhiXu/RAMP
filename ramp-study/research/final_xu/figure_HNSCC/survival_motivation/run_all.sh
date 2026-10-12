#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"

python3 "$script_dir/01_build_spatial_niche_scores.py"
Rscript "$script_dir/02_run_survival_and_plot.R"
python3 "$script_dir/03_validate_outputs.py"
