#!/bin/zsh
set -eu
cd "$(dirname "$0")"
exec /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 -B scripts/run.py "${1:-all}"
