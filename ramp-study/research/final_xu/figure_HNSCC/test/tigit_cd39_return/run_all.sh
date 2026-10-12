#!/bin/sh
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 "$HERE/01_project_fixed_pair_to_codex.py"
Rscript "$HERE/02_run_survival.R"
python3 "$HERE/03_validate_outputs.py"
