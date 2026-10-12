"""Small immutable data containers for the MaxFuse reproduction layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _finite_matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.shape[0] < 3 or array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least three rows and one feature")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _strict_integer_vector(value: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype; implicit truncation is forbidden")
    return raw.astype(np.int64, copy=False)


def _strict_names(value: tuple[str, ...], name: str) -> tuple[str, ...]:
    try:
        result = tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence of strings") from error
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise TypeError(f"{name} must contain only non-empty string scalars")
    return result


@dataclass(frozen=True)
class ViewData:
    """One already-prepared dataset supplied to MaxFuse.

    Assay-specific normalization is deliberately outside the algorithm layer.
    ``shared`` contains homologous features in the same order in every view;
    ``active`` contains the view's full feature panel.
    """

    shared: np.ndarray
    active: np.ndarray
    cell_ids: tuple[str, ...] | None = None
    active_feature_names: tuple[str, ...] | None = None
    shared_feature_names: tuple[str, ...] | None = None

    def validated(self, name: str) -> "ViewData":
        shared = _finite_matrix(self.shared, f"{name}.shared")
        active = _finite_matrix(self.active, f"{name}.active")
        if shared.shape[0] != active.shape[0]:
            raise ValueError(f"{name}: shared and active arrays have different row counts")
        cell_ids = self.cell_ids
        if cell_ids is not None:
            cell_ids = _strict_names(cell_ids, f"{name}.cell_ids")
            if len(cell_ids) != shared.shape[0] or len(set(cell_ids)) != len(cell_ids):
                raise ValueError(f"{name}: cell_ids must be unique and row-aligned")
        features = self.active_feature_names
        if features is not None:
            features = _strict_names(features, f"{name}.active_feature_names")
            if len(features) != active.shape[1] or len(set(features)) != len(features):
                raise ValueError(
                    f"{name}: active_feature_names must be unique and column-aligned"
                )
        shared_features = self.shared_feature_names
        if shared_features is not None:
            shared_features = _strict_names(
                shared_features, f"{name}.shared_feature_names"
            )
            if (
                len(shared_features) != shared.shape[1]
                or len(set(shared_features)) != len(shared_features)
            ):
                raise ValueError(
                    f"{name}: shared_feature_names must be unique and column-aligned"
                )
        return ViewData(shared, active, cell_ids, features, shared_features)


@dataclass(frozen=True)
class Matching:
    """A MaxFuse matching: first-view indices, second-view indices, scores."""

    first: np.ndarray
    second: np.ndarray
    score: np.ndarray

    def validated(
        self,
        *,
        n_first: int | None = None,
        n_second: int | None = None,
        require_unique_first: bool = False,
    ) -> "Matching":
        first = _strict_integer_vector(self.first, "matching.first")
        second = _strict_integer_vector(self.second, "matching.second")
        score = np.asarray(self.score, dtype=np.float64)
        if score.ndim != 1:
            raise ValueError("matching.score must be one-dimensional")
        if not (len(first) == len(second) == len(score)):
            raise ValueError("matching vectors have different lengths")
        if len(first) == 0:
            raise ValueError("matching is empty")
        if np.any(first < 0) or (n_first is not None and np.any(first >= n_first)):
            raise IndexError("matching.first contains out-of-bounds indices")
        if np.any(second < 0) or (n_second is not None and np.any(second >= n_second)):
            raise IndexError("matching.second contains out-of-bounds indices")
        if not np.isfinite(score).all():
            raise ValueError("matching.score contains non-finite values")
        if require_unique_first and len(np.unique(first)) != len(first):
            raise ValueError("matching.first is not unique")
        return Matching(first, second, score)

    @classmethod
    def from_upstream(cls, value: Sequence[Sequence[float]]) -> "Matching":
        if len(value) != 3:
            raise ValueError("upstream matching must contain exactly three vectors")
        return cls(
            np.asarray(value[0]),
            np.asarray(value[1]),
            np.asarray(value[2], dtype=np.float64),
        ).validated()

    def as_upstream(self) -> list[list[float] | list[int]]:
        return [self.first.tolist(), self.second.tolist(), self.score.tolist()]


@dataclass(frozen=True)
class TripletMatching:
    """Three matched views ordered as hub, outer_left, outer_right."""

    hub: np.ndarray
    outer_left: np.ndarray
    outer_right: np.ndarray
    left_score: np.ndarray
    right_score: np.ndarray

    def validated(self, view_sizes: tuple[int, int, int]) -> "TripletMatching":
        hub = _strict_integer_vector(self.hub, "triplets.hub")
        left = _strict_integer_vector(self.outer_left, "triplets.outer_left")
        right = _strict_integer_vector(self.outer_right, "triplets.outer_right")
        left_score = np.asarray(self.left_score, dtype=np.float64)
        right_score = np.asarray(self.right_score, dtype=np.float64)
        lengths = {len(hub), len(left), len(right), len(left_score), len(right_score)}
        if len(lengths) != 1 or not lengths or len(hub) < 3:
            raise ValueError("triplets must contain at least three equally sized vectors")
        if len(np.unique(hub)) != len(hub):
            raise ValueError("triplet hub indices must be unique")
        for vector, size, label in zip(
            (hub, left, right), view_sizes, ("hub", "outer_left", "outer_right")
        ):
            if np.any(vector < 0) or np.any(vector >= size):
                raise IndexError(f"triplets.{label} contains out-of-bounds indices")
        if not np.isfinite(left_score).all() or not np.isfinite(right_score).all():
            raise ValueError("triplet scores contain non-finite values")
        return TripletMatching(hub, left, right, left_score, right_score)

    @property
    def outer_left_reused(self) -> int:
        return int(len(self.outer_left) - len(np.unique(self.outer_left)))

    @property
    def outer_right_reused(self) -> int:
        return int(len(self.outer_right) - len(np.unique(self.outer_right)))
