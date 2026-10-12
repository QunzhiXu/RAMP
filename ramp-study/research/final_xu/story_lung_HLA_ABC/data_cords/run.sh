#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PACKAGE_DIR"
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_EXECUTABLE="$PYTHON_BIN"
elif [[ -x /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 ]]; then
    PYTHON_EXECUTABLE=/Library/Frameworks/Python.framework/Versions/3.14/bin/python3
else
    PYTHON_EXECUTABLE=python3
fi
exec "$PYTHON_EXECUTABLE" -B -u scripts/run_data_cords.py "$@"
