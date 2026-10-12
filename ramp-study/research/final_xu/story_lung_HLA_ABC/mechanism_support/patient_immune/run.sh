#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"
py=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
"$py" -B scripts/prepare.py > logs/prepare.log 2>&1
if [[ ! -f data/MEASURED_MARKERS.csv.gz ]]; then
    TMPDIR=tmp TMP=tmp TEMP=tmp /usr/local/bin/Rscript --vanilla scripts/export.R . > logs/export.log 2>&1
fi
"$py" -B scripts/analyze.py > logs/analyze.log 2>&1
"$py" -B scripts/report.py > logs/report.log 2>&1
