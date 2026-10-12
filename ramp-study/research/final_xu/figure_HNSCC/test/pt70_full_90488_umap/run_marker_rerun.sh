#!/bin/zsh
set -euo pipefail

here="${0:A:h}"
python3 "$here/04_project_niche_score_to_rahim.py"
python3 "$here/05_screen_projected_biomarkers.py"
