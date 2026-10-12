#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-/Library/Frameworks/Python.framework/Versions/3.14/bin/python3}"
case "${1:-run}" in
  run) "$PYTHON_BIN" -B scripts/analyze.py; "$PYTHON_BIN" -B scripts/report.py; "$PYTHON_BIN" -B scripts/verify.py ;;
  plot) "$PYTHON_BIN" -B scripts/report.py ;;
  verify) "$PYTHON_BIN" -B scripts/verify.py ;;
  composition) "$PYTHON_BIN" -B scripts/analyze_composition.py; "$PYTHON_BIN" -B scripts/report_composition.py ;;
  composition-plot) "$PYTHON_BIN" -B scripts/report_composition.py ;;
  architecture) "$PYTHON_BIN" -B scripts/explore_architecture.py; "$PYTHON_BIN" -B scripts/report_architecture.py ;;
  *) echo 'Usage: bash run.sh [run|plot|verify|composition|composition-plot|architecture]'; exit 2 ;;
esac
