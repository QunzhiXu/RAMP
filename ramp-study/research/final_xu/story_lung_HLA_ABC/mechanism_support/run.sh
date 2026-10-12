#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-/Library/Frameworks/Python.framework/Versions/3.14/bin/python3}"
MODE="${1:-all}"
mkdir -p tmp logs
if [[ "$MODE" == "all" ]]; then
  if [[ ! -f results/DESIGN.json ]]; then "$PYTHON_BIN" -B scripts/prepare.py; fi
  if [[ ! -f data/MEASURED_MARKERS.csv.gz ]]; then TMPDIR=tmp TMP=tmp TEMP=tmp Rscript --vanilla scripts/export_measurements.R .; fi
  if [[ ! -f results/ANALYSIS_STATUS.json ]]; then "$PYTHON_BIN" -B scripts/analyze.py; fi
  if [[ ! -f results/MEASURED_MARKER_DFS_COX.csv ]]; then TMPDIR=tmp TMP=tmp TEMP=tmp Rscript --vanilla scripts/survival.R .; fi
  "$PYTHON_BIN" -B scripts/report.py
  "$PYTHON_BIN" -B scripts/verify.py
elif [[ "$MODE" == "report" ]]; then
  "$PYTHON_BIN" -B scripts/report.py
elif [[ "$MODE" == "verify" ]]; then
  "$PYTHON_BIN" -B scripts/verify.py
else
  echo 'Usage: bash run.sh [all|report|verify]' >&2
  exit 2
fi
