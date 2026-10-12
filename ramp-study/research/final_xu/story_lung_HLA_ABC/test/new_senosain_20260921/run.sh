#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/tmp"
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" -u scripts/align.py
"$python_bin" -u scripts/project.py validation
"$python_bin" -u scripts/project.py cords
"$python_bin" -u scripts/analyze.py validation
"$python_bin" -u scripts/analyze.py scores
TMPDIR=tmp TMP=tmp TEMP=tmp Rscript --vanilla scripts/survival.R "$PWD"
"$python_bin" -u scripts/mechanism.py
"$python_bin" -u scripts/umaps.py
"$python_bin" -u scripts/verify.py
"$python_bin" -u scripts/report.py
