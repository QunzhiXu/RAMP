"""Official-Archive three-view workflow built on pairwise MaxFuse matches.

The installed MaxFuse package is pairwise.  The MaxFuse paper's released
HuBMAP Archive connects two pairwise fits through a shared hub and applies a
notebook-local iterative GCCA procedure.  This module reproduces that released
three-view workflow while keeping view order and preprocessing explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .gcca import (
    GCCAResult,
    archive_svd_scores,
    fit_archive_gcca,
    matched_component_correlations,
)
from .pairwise import PairwiseConfig, PairwiseResult, best_per_first, fit_pairwise
from .projection import ThreeWayProjectionModel, build_projection_model
from .types import Matching, TripletMatching, ViewData


THREE_VIEW_EXTENSION_METHOD = (
    "MaxFuse Archive three-view workflow (pairwise hub-pivot chaining + iterative GCCA)"
)


@dataclass(frozen=True)
class ThreeViewExtensionAudit:
    hub: str
    view_order: tuple[str, str, str]
    pairwise_pivot_records: dict[str, int]
    pairwise_unique_hub_pivots: dict[str, int]
    triplets: int
    outer_reuse_policy: str
    outer_reused_records: dict[str, int]
    outer_maximum_multiplicity: dict[str, int]
    archive_svd_components: tuple[int | None, ...]
    gcca_components: int
    gcca_effective_ranks: tuple[int, ...]
    gcca_iteration_counts: tuple[int, ...]
    gcca_final_objectives: tuple[float, ...]
    gcca_all_components_converged: bool
    projection_roundtrip_max_abs_error: dict[str, float]
    mean_matched_component_correlation: float


@dataclass(frozen=True)
class ThreeViewExtensionResult:
    embeddings: Mapping[str, np.ndarray]
    triplets: TripletMatching
    pairwise: Mapping[str, PairwiseResult]
    projection_model: ThreeWayProjectionModel
    gcca: GCCAResult
    audit: ThreeViewExtensionAudit


def _maximum_multiplicity(indices: np.ndarray) -> int:
    if len(indices) == 0:
        return 0
    return int(np.max(np.unique(indices, return_counts=True)[1]))


def _drop_reused_outer_triplets(triplets: TripletMatching) -> TripletMatching:
    """Greedy one-to-one sensitivity rule, not the official notebook rule."""

    combined = triplets.left_score + triplets.right_score
    order = np.lexsort((triplets.hub, -combined))
    used_left: set[int] = set()
    used_right: set[int] = set()
    keep: list[int] = []
    for row in order:
        left = int(triplets.outer_left[row])
        right = int(triplets.outer_right[row])
        if left in used_left or right in used_right:
            continue
        keep.append(int(row))
        used_left.add(left)
        used_right.add(right)
    keep_array = np.asarray(sorted(keep, key=lambda row: int(triplets.hub[row])), dtype=int)
    return TripletMatching(
        triplets.hub[keep_array],
        triplets.outer_left[keep_array],
        triplets.outer_right[keep_array],
        triplets.left_score[keep_array],
        triplets.right_score[keep_array],
    )


def chain_hub_matches(
    left: Matching,
    right: Matching,
    *,
    view_sizes: tuple[int, int, int],
    outer_reuse_policy: str,
) -> TripletMatching:
    """Join two pairwise pivot matchings on the first-view (hub) index.

    ``allow`` follows the hub-intersection endpoint-reuse convention used by
    the archived three-modality example: each hub keeps its best match in each
    outer view, while an outer cell may appear in several tuples.
    ``drop_reused`` is an explicitly non-official sensitivity rule that
    greedily removes repeated outer endpoints.
    """

    if outer_reuse_policy not in {"allow", "drop_reused"}:
        raise ValueError("outer_reuse_policy must be 'allow' or 'drop_reused'")
    left = best_per_first(left).validated(
        n_first=view_sizes[0], n_second=view_sizes[1], require_unique_first=True
    )
    right = best_per_first(right).validated(
        n_first=view_sizes[0], n_second=view_sizes[2], require_unique_first=True
    )
    left_lookup = {
        int(hub): (int(outer), float(score))
        for hub, outer, score in zip(left.first, left.second, left.score)
    }
    right_lookup = {
        int(hub): (int(outer), float(score))
        for hub, outer, score in zip(right.first, right.second, right.score)
    }
    common = np.asarray(sorted(set(left_lookup) & set(right_lookup)), dtype=np.int64)
    if len(common) < 3:
        raise RuntimeError("fewer than three hub pivots are shared by both pairwise fits")
    triplets = TripletMatching(
        hub=common,
        outer_left=np.asarray([left_lookup[int(row)][0] for row in common], dtype=np.int64),
        outer_right=np.asarray([right_lookup[int(row)][0] for row in common], dtype=np.int64),
        left_score=np.asarray([left_lookup[int(row)][1] for row in common], dtype=np.float64),
        right_score=np.asarray([right_lookup[int(row)][1] for row in common], dtype=np.float64),
    ).validated(view_sizes)
    if outer_reuse_policy == "drop_reused":
        triplets = _drop_reused_outer_triplets(triplets).validated(view_sizes)
    return triplets


def fit_three_view_extension(
    views: Mapping[str, ViewData],
    *,
    hub: str,
    view_order: Sequence[str],
    pairwise_configs: Mapping[str, PairwiseConfig],
    svd_components: Mapping[str, int | None],
    gcca_components: int = 15,
    gcca_initialization_max_iter: int = 2000,
    gcca_refinement_max_iter: int = 2000,
    gcca_tolerance: float = 1e-3,
    outer_reuse_policy: str,
) -> ThreeViewExtensionResult:
    """Fit the three-view workflow released in the MaxFuse HuBMAP Archive.

    The first element of ``view_order`` must be ``hub``.  Order is an explicit
    algorithm parameter because Archive initialization applies CCA to the
    first two views, initializes later views by OLS, and refines sequentially.
    """

    if len(views) != 3 or len(set(views)) != 3:
        raise ValueError("exactly three uniquely named views are required")
    if hub not in views:
        raise KeyError(f"unknown hub {hub!r}")
    view_order = tuple(map(str, view_order))
    if len(view_order) != 3 or set(view_order) != set(views):
        raise ValueError("view_order must contain each of the three views exactly once")
    if view_order[0] != hub:
        raise ValueError("the first view_order entry must be the pairwise hub")
    outer_names = list(view_order[1:])
    if set(pairwise_configs) != set(outer_names):
        raise ValueError("pairwise_configs must contain exactly one entry per outer view")
    validated = {name: views[name].validated(name) for name in view_order}
    if any(validated[name].active_feature_names is None for name in view_order):
        raise ValueError("active_feature_names are required for every three-way view")
    if any(validated[name].shared_feature_names is None for name in view_order):
        raise ValueError("shared_feature_names are required for every three-way view")
    shared_orders = {validated[name].shared_feature_names for name in view_order}
    if len(shared_orders) != 1:
        raise ValueError("shared feature names or column order differ across views")
    shared_widths = {validated[name].shared.shape[1] for name in view_order}
    if len(shared_widths) != 1:
        raise ValueError("all three shared arrays must have the same column count")

    pairwise_results: dict[str, PairwiseResult] = {}
    for outer in outer_names:
        pairwise_results[outer] = fit_pairwise(
            validated[hub], validated[outer], pairwise_configs[outer]
        )
    triplets = chain_hub_matches(
        pairwise_results[outer_names[0]].pivot_matching,
        pairwise_results[outer_names[1]].pivot_matching,
        view_sizes=tuple(len(validated[name].active) for name in view_order),
        outer_reuse_policy=outer_reuse_policy,
    )
    if not isinstance(svd_components, Mapping) or set(svd_components) != set(view_order):
        raise ValueError("svd_components must explicitly name all three ordered views")
    component_map = {name: svd_components[name] for name in view_order}
    preprocessing = {
        name: archive_svd_scores(
            validated[name].active,
            n_components=component_map[name],
        )
        for name in view_order
    }
    matched = {
        hub: triplets.hub,
        outer_names[0]: triplets.outer_left,
        outer_names[1]: triplets.outer_right,
    }
    gcca = fit_archive_gcca(
        [preprocessing[name].scores for name in view_order],
        [matched[name] for name in view_order],
        requested_components=gcca_components,
        initialization_max_iter=gcca_initialization_max_iter,
        refinement_max_iter=gcca_refinement_max_iter,
        tolerance=gcca_tolerance,
    )
    embeddings = {
        name: embedding for name, embedding in zip(view_order, gcca.embeddings)
    }
    projection_model = build_projection_model(
        view_order,
        preprocessing,
        gcca,
        {
            name: validated[name].active_feature_names
            for name in view_order
            if validated[name].active_feature_names is not None
        },
    )
    roundtrip: dict[str, float] = {}
    for name in view_order:
        projected = projection_model.transform(
            name,
            validated[name].active,
            validated[name].active_feature_names,
        )
        error = float(np.max(np.abs(projected - embeddings[name])))
        if error > 1e-9:
            raise AssertionError(f"{name}: projection roundtrip failed ({error})")
        roundtrip[name] = error
    correlations = matched_component_correlations(
        [embeddings[name] for name in view_order],
        [matched[name] for name in view_order],
    )
    if not np.isfinite(correlations).all():
        raise RuntimeError("matched GCCA component correlations are non-finite")
    audit = ThreeViewExtensionAudit(
        hub=hub,
        view_order=view_order,
        pairwise_pivot_records={
            outer: len(pairwise_results[outer].pivot_matching.first)
            for outer in outer_names
        },
        pairwise_unique_hub_pivots={
            outer: len(np.unique(pairwise_results[outer].pivot_matching.first))
            for outer in outer_names
        },
        triplets=len(triplets.hub),
        outer_reuse_policy=outer_reuse_policy,
        outer_reused_records={
            outer_names[0]: triplets.outer_left_reused,
            outer_names[1]: triplets.outer_right_reused,
        },
        outer_maximum_multiplicity={
            outer_names[0]: _maximum_multiplicity(triplets.outer_left),
            outer_names[1]: _maximum_multiplicity(triplets.outer_right),
        },
        archive_svd_components=tuple(component_map[name] for name in view_order),
        gcca_components=gcca.n_components,
        gcca_effective_ranks=gcca.effective_ranks,
        gcca_iteration_counts=gcca.iteration_counts,
        gcca_final_objectives=gcca.final_objectives,
        gcca_all_components_converged=all(gcca.converged),
        projection_roundtrip_max_abs_error=roundtrip,
        mean_matched_component_correlation=float(np.mean(correlations)),
    )
    return ThreeViewExtensionResult(
        embeddings=embeddings,
        triplets=triplets,
        pairwise=pairwise_results,
        projection_model=projection_model,
        gcca=gcca,
        audit=audit,
    )


# Backward-compatible public aliases.  New code and manifests should use the
# explicit ``ThreeViewExtension*`` / ``fit_three_view_extension`` names above.
ThreeWayAudit = ThreeViewExtensionAudit
ThreeWayResult = ThreeViewExtensionResult
fit_threeway = fit_three_view_extension
