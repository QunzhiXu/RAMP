#!/usr/bin/env python3
"""Small end-to-end test for Archive preprocessing, GCCA, and projection v2."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maxfuse_repro.gcca import archive_svd_scores, fit_archive_gcca
from maxfuse_repro.projection import (
    SCHEMA_VERSION,
    ThreeViewExtensionProjectionModel,
    build_projection_model,
)


def main() -> None:
    rng = np.random.default_rng(20260830)
    latent = rng.normal(size=(140, 4))
    active = [
        latent @ rng.normal(size=(4, width))
        + 0.12 * rng.normal(size=(len(latent), width))
        for width in (8, 9, 10)
    ]
    preprocessing = [
        archive_svd_scores(active[0], 6),
        archive_svd_scores(active[1], 6),
        archive_svd_scores(active[2], None),
    ]
    for result in preprocessing:
        if not np.allclose(result.scores.mean(axis=0), 0.0, atol=1e-12):
            raise AssertionError("Archive preprocessing is not full-view centered")
        if not np.allclose(result.scores.std(axis=0, ddof=0), 1.0, atol=1e-12):
            raise AssertionError("Archive preprocessing is not population-SD scaled")
    indices = [
        np.arange(110, dtype=np.int64),
        np.roll(np.arange(110, dtype=np.int64), 3),
        np.roll(np.arange(110, dtype=np.int64), 7),
    ]
    gcca = fit_archive_gcca(
        [result.scores for result in preprocessing],
        indices,
        requested_components=3,
        initialization_max_iter=2000,
        refinement_max_iter=60,
        tolerance=1e-3,
    )
    if gcca.method != "maxfuse_archive_iterative_ols":
        raise AssertionError("wrong GCCA method audit")
    if not all(np.isfinite(value).all() for value in gcca.embeddings):
        raise AssertionError("Archive GCCA produced non-finite embeddings")
    names = ("hub", "outer_a", "outer_b")
    feature_names = {
        name: tuple(f"marker_{column}" for column in range(matrix.shape[1]))
        for name, matrix in zip(names, active)
    }
    model = build_projection_model(
        names,
        {name: result for name, result in zip(names, preprocessing)},
        gcca,
        feature_names,
    )
    if model.schema_version != SCHEMA_VERSION:
        raise AssertionError("Archive projection did not use the current schema")
    for name, matrix, expected in zip(names, active, gcca.embeddings):
        observed = model.transform(name, matrix, feature_names[name])
        if not np.allclose(observed, expected, atol=1e-9, rtol=1e-9):
            raise AssertionError(f"{name}: Archive projection roundtrip failed")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "archive_projection.npz"
        model.save(path)
        loaded = ThreeViewExtensionProjectionModel.load(path)
        for name, matrix, expected in zip(names, active, gcca.embeddings):
            observed = loaded.transform(name, matrix, feature_names[name])
            if not np.allclose(observed, expected, atol=1e-9, rtol=1e-9):
                raise AssertionError(f"{name}: Archive projection save/load failed")
    print("MAXFUSE_REPRO ARCHIVE_GCCA_SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
