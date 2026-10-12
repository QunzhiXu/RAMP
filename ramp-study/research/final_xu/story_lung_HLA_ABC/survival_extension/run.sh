#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
"${PYTHON_BIN:-/Library/Frameworks/Python.framework/Versions/3.14/bin/python3}" -B scripts/run.py
