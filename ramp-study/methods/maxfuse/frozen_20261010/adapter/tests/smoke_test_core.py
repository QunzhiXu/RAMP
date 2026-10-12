#!/usr/bin/env python3
"""Deterministic numerical and serialization smoke tests for the local extension."""

from __future__ import annotations

import itertools
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maxfuse_repro.gcca import fit_maxvar_gcca, matched_component_correlations, pca_scores
from maxfuse_repro.projection import (
    ThreeViewExtensionProjectionModel,
    ThreeWayProjectionModel,
    build_projection_model,
)
from maxfuse_repro.provenance import strict_string_vector


def make_views(seed: int = 20260830) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(240, 4))
    return [
        latent @ rng.normal(size=(4, features))
        + 0.08 * rng.normal(size=(len(latent), features))
        for features in (9, 12, 7)
    ]


def normalized_gram(embeddings: list[np.ndarray]) -> np.ndarray:
    combined = np.concatenate(embeddings, axis=0)
    gram = combined @ combined.T
    return gram / np.linalg.norm(gram)


def main() -> None:
    if ThreeWayProjectionModel is not ThreeViewExtensionProjectionModel:
        raise AssertionError("legacy projection-model alias no longer targets canonical type")
    active = make_views()
    pca = [pca_scores(view, max_components=6) for view in active]
    indices = [np.arange(180, dtype=np.int64) for _ in active]
    baseline = fit_maxvar_gcca(
        [item.scores for item in pca], indices, requested_components=4
    )
    correlations = matched_component_correlations(baseline.embeddings, indices)
    if not np.isfinite(correlations).all():
        raise AssertionError("matched correlations contain non-finite values")
    if float(np.min(np.mean(correlations, axis=0))) < 0.80:
        raise AssertionError(f"unexpectedly weak matched correlation: {correlations}")
    base_gram = normalized_gram(list(baseline.embeddings))
    if not np.isfinite(base_gram).all():
        raise AssertionError("baseline Gram matrix contains non-finite values")
    for permutation in itertools.permutations(range(3)):
        result = fit_maxvar_gcca(
            [pca[index].scores for index in permutation],
            [indices[index] for index in permutation],
            requested_components=4,
        )
        restored: list[np.ndarray | None] = [None, None, None]
        for position, original in enumerate(permutation):
            restored[original] = result.embeddings[position]
        error = np.linalg.norm(
            normalized_gram([value for value in restored if value is not None]) - base_gram
        )
        if not np.isfinite(error) or error > 1e-6:
            raise AssertionError(f"view-order invariance failed: {permutation}, {error}")

    try:
        fit_maxvar_gcca(
            [item.scores for item in pca],
            [np.arange(180, dtype=float) for _ in active],
            requested_components=4,
        )
    except TypeError:
        pass
    else:
        raise AssertionError("floating-point matched indices were silently truncated")

    invalid_calls = (
        lambda: pca_scores(active[0], max_components=2.9),
        lambda: fit_maxvar_gcca(
            [item.scores for item in pca], indices, requested_components=2.9
        ),
        lambda: fit_maxvar_gcca(
            [item.scores for item in pca], indices, requested_components=4,
            ridge=float("inf"),
        ),
        lambda: fit_maxvar_gcca(
            [item.scores for item in pca], indices, requested_components=4,
            tolerance=float("nan"),
        ),
        lambda: fit_maxvar_gcca(
            [item.scores for item in pca], indices, requested_components=4,
            ridge=True,
        ),
    )
    for invalid_call in invalid_calls:
        try:
            invalid_call()
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid PCA/GCCA parameter was silently accepted")

    if strict_string_vector(np.asarray(["a", "b"]), "names") != ("a", "b"):
        raise AssertionError("valid string vector was not preserved")
    for invalid_names in (
        np.asarray([["a"], ["b"]]),
        np.asarray([1, 2]),
        np.asarray(["a", ""]),
    ):
        try:
            strict_string_vector(invalid_names, "names")
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid string vector was silently accepted")

    names = ("view_a", "view_b", "view_c")
    model = build_projection_model(
        names,
        {name: item for name, item in zip(names, pca)},
        baseline,
        {
            name: tuple(f"marker_{column}" for column in range(view.shape[1]))
            for name, view in zip(names, active)
        },
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "projection.npz"
        model.save(path)
        reloaded = ThreeViewExtensionProjectionModel.load(path)
        for name, matrix, expected in zip(names, active, baseline.embeddings):
            features = tuple(f"marker_{column}" for column in range(matrix.shape[1]))
            observed = reloaded.transform(name, matrix, features)
            if not np.isfinite(observed).all():
                raise AssertionError(f"{name}: reloaded projection is non-finite")
            if not np.allclose(observed, expected, atol=1e-12, rtol=1e-12):
                raise AssertionError(f"{name}: save/load projection roundtrip failed")
        try:
            reloaded.transform(
                names[0], active[0], tuple(reversed(model.views[names[0]].feature_names))
            )
        except ValueError:
            pass
        else:
            raise AssertionError("feature-order mismatch was not rejected")

    print("MAXFUSE_REPRO CORE_SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
