#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 /absolute/path/to/python3.12" >&2
    exit 2
fi

python_bin=$1
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
venv_dir="$script_dir/.venv"

"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
if [ -e "$venv_dir" ]; then
    echo "refusing to overwrite existing environment: $venv_dir" >&2
    exit 1
fi

"$python_bin" -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install \
    -r "$script_dir/maxfuse_repro/environment-lock.txt"
"$venv_dir/bin/python" -m pip install --no-deps --no-build-isolation "$script_dir"
"$venv_dir/bin/python" -m pip check
echo "dedicated MaxFuse environment created at $venv_dir"
