#!/usr/bin/env python3
"""Direct numerical parity against a provenance-locked MaxFuse Archive notebook.

The upstream notebook is supplied at test time and is never copied into this
project.  Only the two GCCA function definitions are extracted through Python's
AST; no other notebook code is executed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.cross_decomposition import CCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maxfuse_repro.gcca import fit_archive_gcca


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_upstream_functions(path: Path):
    lock = json.loads((ROOT / "upstream_lock.json").read_text())
    allowed = {
        row["sha256"] for row in lock["archive_reference_notebooks"]
    }
    observed = sha256(path)
    if observed not in allowed:
        raise RuntimeError(f"Archive notebook SHA-256 is not locked: {observed}")
    notebook = json.loads(path.read_text())
    source = None
    for cell in notebook.get("cells", []):
        candidate = "".join(cell.get("source", []))
        if "def gcca_init" in candidate and "def gcca_refine" in candidate:
            source = candidate
            break
    if source is None:
        raise RuntimeError("locked notebook lacks the expected GCCA helper cell")
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"gcca_init", "gcca_refine"}
    ]
    if {node.name for node in selected} != {"gcca_init", "gcca_refine"}:
        raise RuntimeError("failed to isolate both Archive GCCA functions")
    namespace = {"np": np, "CCA": CCA}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["gcca_init"], namespace["gcca_refine"]


def population_scale(matrix: np.ndarray) -> np.ndarray:
    return (matrix - matrix.mean(axis=0)) / matrix.std(axis=0, ddof=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    gcca_init, gcca_refine = load_upstream_functions(args.notebook.resolve())

    rng = np.random.default_rng(20260830)
    latent = rng.normal(size=(96, 4))
    views = [
        population_scale(
            latent @ rng.normal(size=(4, width))
            + 0.15 * rng.normal(size=(len(latent), width))
        )
        for width in (7, 8, 9)
    ]
    indices = [
        np.arange(0, 72, dtype=np.int64),
        np.roll(np.arange(0, 72, dtype=np.int64), 5),
        np.roll(np.arange(0, 72, dtype=np.int64), 11),
    ]

    reference_initial = gcca_init(
        data_list=views,
        indices_list=indices,
        n_components=3,
        max_iter=2000,
    )
    scaled_initial = [population_scale(value) for value in reference_initial]
    reference_raw = gcca_refine(
        data_list=views,
        indices_list=indices,
        init_embeddings_list=scaled_initial,
        max_iter=7,
        tol=0.0,
        verbose=False,
    )
    reference = [population_scale(value) for value in reference_raw]

    observed = fit_archive_gcca(
        views,
        indices,
        requested_components=3,
        initialization_max_iter=2000,
        refinement_max_iter=7,
        tolerance=0.0,
    )
    if observed.iteration_counts != (7, 7, 7):
        raise AssertionError(f"unexpected forced iteration counts: {observed.iteration_counts}")
    errors = [
        float(np.max(np.abs(expected - actual)))
        for expected, actual in zip(reference, observed.embeddings)
    ]
    if max(errors) > 1e-8:
        raise AssertionError(f"Archive GCCA parity failed: max errors {errors}")
    print(
        "MAXFUSE_REPRO ARCHIVE_GCCA_DIRECT_PARITY_OK "
        f"max_abs_error={max(errors):.3e}"
    )


if __name__ == "__main__":
    main()
