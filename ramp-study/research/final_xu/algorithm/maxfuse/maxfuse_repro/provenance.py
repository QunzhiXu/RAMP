"""Deterministic provenance helpers shared by the generic file runners."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import os
import sys
from typing import Iterable

import numpy as np


CORE_RELATIVE_FILES = (
    "run_maxfuse_pairwise.py",
    "run_maxfuse_threeway.py",
    "run_maxfuse_three_view_extension.py",
    "maxfuse_repro/__init__.py",
    "maxfuse_repro/gcca.py",
    "maxfuse_repro/pairwise.py",
    "maxfuse_repro/projection.py",
    "maxfuse_repro/provenance.py",
    "maxfuse_repro/threeway.py",
    "maxfuse_repro/types.py",
    "maxfuse_repro/environment-lock.txt",
    "maxfuse_repro/requirements-lock.txt",
    "maxfuse_repro/upstream_lock.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def algorithm_source_hashes(extra_files: Iterable[Path] = ()) -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    result = {
        relative: sha256_file(root / relative) for relative in CORE_RELATIVE_FILES
    }
    for path in extra_files:
        resolved = Path(path).resolve()
        try:
            key = str(resolved.relative_to(root))
        except ValueError:
            key = str(resolved)
        result[key] = sha256_file(resolved)
    return result


def output_hashes(root: Path) -> dict[str, str]:
    root = Path(root)
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def runtime_manifest() -> dict[str, object]:
    from threadpoolctl import threadpool_info

    packages = (
        "maxfuse", "numpy", "scipy", "scikit-learn", "pandas", "anndata",
        "scanpy", "zarr", "igraph", "leidenalg", "matplotlib", "requests",
        "networkx", "numba", "llvmlite", "pynndescent", "joblib", "threadpoolctl",
    )
    blas = np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {})
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: version(name) for name in packages},
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "threadpools": threadpool_info(),
        "blas": {
            key: value
            for key, value in blas.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
    }


def strict_string_vector(value: np.ndarray, name: str) -> tuple[str, ...]:
    """Require a nonempty one-dimensional NumPy unicode/bytes vector."""

    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.dtype.kind not in {"U", "S"}:
        raise TypeError(f"{name} must contain string scalars")
    if raw.dtype.kind == "S":
        try:
            raw = np.char.decode(raw, "utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{name} contains invalid UTF-8 bytes") from error
    result = tuple(map(str, raw.tolist()))
    if not result or any(not item for item in result):
        raise ValueError(f"{name} must not be empty or contain empty strings")
    return result
