"""GCCA helpers for three-view MaxFuse workflows.

The primary implementation follows the numerical procedure in the official
MaxFuse HuBMAP Archive notebooks: CCA/OLS initialization followed by the
notebook's component-wise alternating OLS refinement.  The older symmetric
ridge MAXVAR implementation remains available only so historical local result
objects can still be reproduced and inspected.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
from typing import Sequence

import numpy as np
from sklearn.cross_decomposition import CCA


EPS = 1e-12


@dataclass(frozen=True)
class PCAResult:
    scores: np.ndarray
    input_center: np.ndarray
    loadings: np.ndarray
    score_center: np.ndarray
    score_scale: np.ndarray
    eigenvalues: np.ndarray

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        array = np.asarray(matrix, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.input_center):
            raise ValueError("PCA input has the wrong shape")
        if not np.isfinite(array).all():
            raise ValueError("PCA input contains non-finite values")
        raw_scores = (array - self.input_center) @ self.loadings
        return (raw_scores - self.score_center) / self.score_scale


@dataclass(frozen=True)
class GCCAResult:
    embeddings: tuple[np.ndarray, ...]
    weights: tuple[np.ndarray, ...]
    pivot_centers: tuple[np.ndarray, ...]
    pivot_scales: tuple[np.ndarray, ...]
    eigenvalues: np.ndarray
    n_components: int
    effective_ranks: tuple[int, ...]
    ridge_values: tuple[float, ...]
    method: str = "ridge_maxvar"
    iteration_counts: tuple[int, ...] = ()
    final_objectives: tuple[float, ...] = ()
    converged: tuple[bool, ...] = ()
    output_centers: tuple[np.ndarray, ...] = ()
    output_scales: tuple[np.ndarray, ...] = ()


def _strict_indices(value: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype")
    return raw.astype(np.int64, copy=False)


def pca_scores(matrix: np.ndarray, max_components: int = 50) -> PCAResult:
    """Fit deterministic PCA through the feature covariance matrix."""

    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError("matrix must be two-dimensional with at least two rows")
    if not np.isfinite(array).all():
        raise ValueError("matrix contains non-finite values")
    if isinstance(max_components, bool) or not isinstance(max_components, numbers.Integral):
        raise TypeError("max_components must be an integer")
    if max_components < 1:
        raise ValueError("max_components must be positive")

    input_center = array.mean(axis=0)
    centered = array - input_center
    variance = centered.var(axis=0, ddof=1)
    keep = np.isfinite(variance) & (variance > EPS)
    if int(np.sum(keep)) < 2:
        raise ValueError("fewer than two non-constant features remain for PCA")
    reduced = centered[:, keep]
    covariance = reduced.T @ reduced / max(1, reduced.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > max(EPS, eigenvalues[0] * 1e-10)
    n_components = min(int(max_components), int(np.sum(positive)))
    if n_components < 2:
        raise ValueError("PCA effective rank is smaller than two")

    reduced_loadings = eigenvectors[:, :n_components]
    raw_scores = reduced @ reduced_loadings
    score_center = raw_scores.mean(axis=0)
    score_scale = raw_scores.std(axis=0, ddof=1)
    score_scale[~np.isfinite(score_scale) | (score_scale < EPS)] = 1.0
    scores = (raw_scores - score_center) / score_scale
    loadings = np.zeros((array.shape[1], n_components), dtype=np.float64)
    loadings[keep] = reduced_loadings
    return PCAResult(
        scores=scores,
        input_center=input_center,
        loadings=loadings,
        score_center=score_center,
        score_scale=score_scale,
        eigenvalues=eigenvalues[:n_components],
    )


def archive_svd_scores(
    matrix: np.ndarray,
    n_components: int | None,
) -> PCAResult:
    """Apply the SVD and full-view scaling used by the MaxFuse Archive.

    ``n_components=None`` follows the protein branch in the official HuBMAP
    notebooks and skips SVD.  A non-null value calls the installed MaxFuse
    package's deterministic ``robust_svd`` helper, retains ``U @ diag(s)``,
    and deliberately does not reorder the returned singular triplets.  Scores
    are centered and scaled over the complete view with NumPy's population
    standard deviation (``ddof=0``), exactly as in ``utils.center_scale``.
    """

    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("matrix must be two-dimensional with at least two rows")
    if not np.isfinite(array).all():
        raise ValueError("matrix contains non-finite values")
    if n_components is not None:
        if isinstance(n_components, bool) or not isinstance(
            n_components, numbers.Integral
        ):
            raise TypeError("n_components must be an integer or None")
        if n_components < 1:
            raise ValueError("n_components must be positive")
        if n_components >= min(array.shape):
            raise ValueError(
                "Archive deterministic SVD requires n_components < min(matrix.shape)"
            )
        # Import lazily so the numerical helper is taken from the separately
        # installed, provenance-locked MaxFuse package rather than copied here.
        from maxfuse import utils as maxfuse_utils

        left, singular_values, right_transpose = maxfuse_utils.robust_svd(
            array,
            n_components=int(n_components),
            randomized=False,
            n_runs=1,
        )
        raw_scores = left @ np.diag(singular_values)
        loadings = right_transpose.T
    else:
        raw_scores = array.copy()
        loadings = np.eye(array.shape[1], dtype=np.float64)
        singular_values = np.empty(0, dtype=np.float64)

    score_center = raw_scores.mean(axis=0)
    score_scale = raw_scores.std(axis=0, ddof=0)
    if np.any(~np.isfinite(score_scale)) or np.any(score_scale <= 0):
        raise ValueError("Archive full-view scaling encountered a zero-variance score")
    scores = (raw_scores - score_center) / score_scale
    return PCAResult(
        scores=scores,
        input_center=np.zeros(array.shape[1], dtype=np.float64),
        loadings=loadings,
        score_center=score_center,
        score_scale=score_scale,
        eigenvalues=np.asarray(singular_values, dtype=np.float64) ** 2,
    )


def _validated_archive_inputs(
    views: Sequence[np.ndarray],
    matched_indices: Sequence[np.ndarray],
    requested_components: int,
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    if len(views) < 3 or len(views) != len(matched_indices):
        raise ValueError("Archive GCCA requires at least three matched views")
    if isinstance(requested_components, bool) or not isinstance(
        requested_components, numbers.Integral
    ):
        raise TypeError("requested_components must be an integer")
    if requested_components < 1:
        raise ValueError("requested_components must be positive")
    arrays = [np.asarray(view, dtype=np.float64) for view in views]
    indices = [
        _strict_indices(index, f"matched_indices[{position}]")
        for position, index in enumerate(matched_indices)
    ]
    tuple_counts = {len(index) for index in indices}
    if len(tuple_counts) != 1:
        raise ValueError("all matched index vectors must have equal length")
    n_tuples = tuple_counts.pop()
    if n_tuples < 3:
        raise ValueError("at least three matched tuples are required")
    for position, (array, index) in enumerate(zip(arrays, indices)):
        if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] < 1:
            raise ValueError(f"view {position} must be a non-empty two-dimensional matrix")
        if not np.isfinite(array).all():
            raise ValueError(f"view {position} contains non-finite values")
        if np.any(index < 0) or np.any(index >= len(array)):
            raise IndexError(f"matched indices for view {position} are out of bounds")
    maximum = min(n_tuples, arrays[0].shape[1], arrays[1].shape[1])
    if requested_components > maximum:
        raise ValueError(
            "requested_components exceeds the Archive CCA feasibility limit; "
            "the algorithm does not silently clip it"
        )
    return arrays, indices, n_tuples


def _population_center_scale(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Archive center/scale encountered a zero-variance component")
    return (matrix - center) / scale, center, scale


def _lstsq_operator(matrix: np.ndarray) -> np.ndarray:
    """Cache the Moore-Penrose operator equivalent to repeated ``lstsq``.

    The Archive notebook repeatedly solves least squares with an unchanged
    design matrix inside each component.  Caching its SVD changes runtime, not
    the stated OLS objective or solution.
    """

    left, singular_values, right_transpose = np.linalg.svd(
        np.asarray(matrix, dtype=np.float64), full_matrices=False
    )
    if singular_values.size == 0 or singular_values[0] <= 0:
        raise ValueError("least-squares design has zero numerical rank")
    cutoff = (
        np.finfo(np.float64).eps
        * max(matrix.shape)
        * float(singular_values[0])
    )
    keep = singular_values > cutoff
    if not np.any(keep):
        raise ValueError("least-squares design has zero numerical rank")
    return (right_transpose[keep].T / singular_values[keep]) @ left[:, keep].T


def fit_archive_gcca(
    views: Sequence[np.ndarray],
    matched_indices: Sequence[np.ndarray],
    *,
    requested_components: int = 15,
    initialization_max_iter: int = 2000,
    refinement_max_iter: int = 2000,
    tolerance: float = 1e-3,
) -> GCCAResult:
    """Reproduce the official Archive CCA/iterative-OLS GCCA procedure.

    View order is intentionally meaningful: CCA initializes the first two
    views and every later view is initialized by OLS against their mean score.
    Refinement preserves the Archive notebook's fixed-initial-score subtraction
    exactly, including on iterations after the first.
    """

    arrays, indices, _ = _validated_archive_inputs(
        views, matched_indices, requested_components
    )
    for name, value in (
        ("initialization_max_iter", initialization_max_iter),
        ("refinement_max_iter", refinement_max_iter),
    ):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if isinstance(tolerance, bool) or not isinstance(tolerance, numbers.Real):
        raise TypeError("tolerance must be a real number")
    if not math.isfinite(float(tolerance)) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")

    cca = CCA(
        n_components=int(requested_components),
        max_iter=int(initialization_max_iter),
    )
    cca.fit(arrays[0][indices[0]], arrays[1][indices[1]])
    first, second = cca.transform(arrays[0], arrays[1])
    initial_embeddings: list[np.ndarray] = [first, second]
    average_score = (first[indices[0]] + second[indices[1]]) / 2.0
    for array, index in zip(arrays[2:], indices[2:]):
        coefficient = np.linalg.lstsq(array[index], average_score, rcond=None)[0]
        initial_embeddings.append(array @ coefficient)
    scaled_initial = [
        _population_center_scale(embedding)[0]
        for embedding in initial_embeddings
    ]

    n_views = len(arrays)
    n_components = int(requested_components)
    raw_embeddings = [
        np.zeros((len(array), n_components), dtype=np.float64)
        for array in arrays
    ]
    raw_weights = [
        np.zeros((array.shape[1], n_components), dtype=np.float64)
        for array in arrays
    ]
    iteration_counts: list[int] = []
    final_objectives: list[float] = []
    converged: list[bool] = []

    for component in range(n_components):
        if component > 0:
            current_data: list[np.ndarray] = []
            for position in range(n_views):
                previous = raw_embeddings[position][indices[position], :component]
                coefficient = np.linalg.lstsq(
                    previous,
                    arrays[position][indices[position]],
                    rcond=None,
                )[0]
                current_data.append(
                    arrays[position][indices[position]] - previous @ coefficient
                )
        else:
            current_data = [
                array[index] for array, index in zip(arrays, indices)
            ]

        current_scores = [
            embedding[index, component].copy()
            for embedding, index in zip(scaled_initial, indices)
        ]
        fixed_initial_scores = [score.copy() for score in current_scores]
        score_sum = sum(current_scores)
        previous_objective = np.inf
        operators = [_lstsq_operator(matrix) for matrix in current_data]
        objective = np.inf
        did_converge = False

        for iteration in range(int(refinement_max_iter)):
            for position in range(n_views):
                target = (score_sum - current_scores[position]) / (n_views - 1)
                coefficient = operators[position] @ target
                norm = float(np.sqrt(np.sum(coefficient**2)))
                if not np.isfinite(norm) or norm <= 0:
                    raise RuntimeError("Archive GCCA produced a zero OLS coefficient")
                coefficient = coefficient / norm
                score = current_data[position] @ coefficient
                # This fixed-initial-score subtraction is the behavior in both
                # official HuBMAP Archive notebooks; it is not silently fixed.
                score_sum = score_sum - fixed_initial_scores[position] + score
                current_scores[position] = score

            objective = float(
                sum(
                    np.sqrt(np.sum((current_scores[position] - current_scores[position + 1]) ** 2))
                    for position in range(n_views - 1)
                )
            )
            if abs(objective - previous_objective) < float(tolerance):
                did_converge = True
                break
            previous_objective = objective

        iteration_counts.append(iteration + 1)
        final_objectives.append(objective)
        converged.append(did_converge)
        for position in range(n_views):
            coefficient = np.linalg.lstsq(
                arrays[position][indices[position]],
                current_scores[position],
                rcond=None,
            )[0]
            raw_weights[position][:, component] = coefficient
            raw_embeddings[position][:, component] = arrays[position] @ coefficient

    scaled_embeddings: list[np.ndarray] = []
    output_centers: list[np.ndarray] = []
    output_scales: list[np.ndarray] = []
    for embedding in raw_embeddings:
        scaled, center, scale = _population_center_scale(embedding)
        scaled_embeddings.append(scaled)
        output_centers.append(center)
        output_scales.append(scale)

    effective_ranks = tuple(
        int(np.linalg.matrix_rank(array[index]))
        for array, index in zip(arrays, indices)
    )
    return GCCAResult(
        embeddings=tuple(scaled_embeddings),
        weights=tuple(raw_weights),
        pivot_centers=tuple(
            np.zeros(array.shape[1], dtype=np.float64) for array in arrays
        ),
        pivot_scales=tuple(
            np.ones(array.shape[1], dtype=np.float64) for array in arrays
        ),
        eigenvalues=np.empty(0, dtype=np.float64),
        n_components=n_components,
        effective_ranks=effective_ranks,
        ridge_values=tuple(0.0 for _ in arrays),
        method="maxfuse_archive_iterative_ols",
        iteration_counts=tuple(iteration_counts),
        final_objectives=tuple(final_objectives),
        converged=tuple(converged),
        output_centers=tuple(output_centers),
        output_scales=tuple(output_scales),
    )


def fit_maxvar_gcca(
    views: Sequence[np.ndarray],
    matched_indices: Sequence[np.ndarray],
    *,
    requested_components: int = 15,
    ridge: float = 1e-3,
    tolerance: float = 1e-10,
) -> GCCAResult:
    """Fit symmetric ridge MAXVAR-GCCA on row-matched tuples.

    Repeated indices are permitted because official MaxFuse hub chaining may
    map multiple hub cells to the same outer cell.  Such repetition is an
    explicit tuple-weighting choice and is reported by the three-view audit.
    """

    if len(views) < 3 or len(views) != len(matched_indices):
        raise ValueError("GCCA requires at least three views and one index vector per view")
    if isinstance(requested_components, bool) or not isinstance(
        requested_components, numbers.Integral
    ):
        raise TypeError("requested_components must be an integer")
    if requested_components < 1:
        raise ValueError("requested_components must be positive")
    for name, value in (("ridge", ridge), ("tolerance", tolerance)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    arrays = [np.asarray(view, dtype=np.float64) for view in views]
    indices = [
        _strict_indices(index, f"matched_indices[{position}]")
        for position, index in enumerate(matched_indices)
    ]
    tuple_counts = {len(index) for index in indices}
    if len(tuple_counts) != 1:
        raise ValueError("all matched index vectors must have equal length")
    n_tuples = tuple_counts.pop()
    if n_tuples < 3:
        raise ValueError("at least three matched tuples are required")

    full_blocks: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    singular_systems: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    effective_ranks: list[int] = []
    ridge_values: list[float] = []
    for position, (array, index) in enumerate(zip(arrays, indices)):
        if array.ndim != 2 or array.shape[0] < 3 or not np.isfinite(array).all():
            raise ValueError(f"view {position} must be a finite two-dimensional matrix")
        if np.any(index < 0) or np.any(index >= len(array)):
            raise IndexError(f"matched indices for view {position} are out of bounds")
        pivot = array[index]
        center = pivot.mean(axis=0)
        scale = pivot.std(axis=0, ddof=1)
        scale[~np.isfinite(scale) | (scale < EPS)] = 1.0
        pivot_standardized = (pivot - center) / scale
        full_standardized = (array - center) / scale
        left, singular_values, right_transpose = np.linalg.svd(
            pivot_standardized, full_matrices=False
        )
        if not singular_values.size or singular_values[0] <= 0:
            raise ValueError(f"view {position} has zero numerical rank")
        rank_tolerance = max(
            float(tolerance),
            np.finfo(np.float64).eps * max(pivot_standardized.shape),
        ) * float(singular_values[0])
        effective_rank = int(np.sum(singular_values > rank_tolerance))
        if effective_rank < 1:
            raise ValueError(f"view {position} has zero numerical rank")
        left = left[:, :effective_rank]
        singular_values = singular_values[:effective_rank]
        right_transpose = right_transpose[:effective_rank]
        ridge_value = float((n_tuples - 1) * ridge)
        full_blocks.append(full_standardized)
        centers.append(center)
        scales.append(scale)
        singular_systems.append((left, singular_values, right_transpose))
        effective_ranks.append(effective_rank)
        ridge_values.append(ridge_value)

    n_components = min(
        int(requested_components), n_tuples - 1, min(effective_ranks)
    )
    if n_components < 2:
        raise ValueError("fewer than two GCCA components are identifiable")

    factors: list[np.ndarray] = []
    for (left, singular_values, _), ridge_value in zip(
        singular_systems, ridge_values
    ):
        shrinkage = singular_values / np.sqrt(singular_values**2 + ridge_value)
        factors.append(left * shrinkage[None, :])
    concatenated = np.concatenate(factors, axis=1)
    shared_scores, singular_values, _ = np.linalg.svd(concatenated, full_matrices=False)
    shared_scores = shared_scores[:, :n_components]
    for component in range(n_components):
        anchor = int(np.argmax(np.abs(shared_scores[:, component])))
        if shared_scores[anchor, component] < 0:
            shared_scores[:, component] *= -1.0

    weights: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    for full, system, ridge_value in zip(full_blocks, singular_systems, ridge_values):
        left, view_singular_values, right_transpose = system
        coefficients = view_singular_values / (view_singular_values**2 + ridge_value)
        weight = right_transpose.T @ (
            coefficients[:, None] * (left.T @ shared_scores)
        )
        weights.append(weight)
        embeddings.append(full @ weight)

    return GCCAResult(
        embeddings=tuple(embeddings),
        weights=tuple(weights),
        pivot_centers=tuple(centers),
        pivot_scales=tuple(scales),
        eigenvalues=singular_values[:n_components] ** 2,
        n_components=n_components,
        effective_ranks=tuple(effective_ranks),
        ridge_values=tuple(ridge_values),
    )


def matched_component_correlations(
    embeddings: Sequence[np.ndarray], matched_indices: Sequence[np.ndarray]
) -> np.ndarray:
    """Return component-wise Pearson correlations for every pair of views."""

    rows: list[np.ndarray] = []
    for first in range(len(embeddings)):
        for second in range(first + 1, len(embeddings)):
            left = np.asarray(embeddings[first])[
                _strict_indices(matched_indices[first], "matched_indices")
            ]
            right = np.asarray(embeddings[second])[
                _strict_indices(matched_indices[second], "matched_indices")
            ]
            rows.append(
                np.asarray(
                    [
                        np.corrcoef(left[:, component], right[:, component])[0, 1]
                        for component in range(left.shape[1])
                    ],
                    dtype=np.float64,
                )
            )
    return np.vstack(rows)
