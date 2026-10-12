#!/usr/bin/env python3
"""RAMP ver1: audited corrections to the canonical RAMP implementation.

This module intentionally leaves :mod:`ramp.ramp` unchanged.  It reuses only
the stable, low-level helpers from that module and owns every call path touched
by the seven corrections documented in the algorithm audit:

1. calibration and final application share one displacement-limited mapper;
2. one mid-rank within-unit percentile convention is used everywhere;
3. curve fitting uses adaptive equal-count bins;
4. compatibility and its PCA are fit on equal-sized dataset samples;
5. provider evidence treats units, rather than cells, as replicates;
6. pair fitting has a real computational maximum; and
7. ``GRAPH_K=15`` always means 15 non-self neighbours.

The public ``fit_transform`` signature and the original five result tables are
kept compatible.  Two additional audit tables expose the new behaviour.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm, rankdata
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from . import ramp as _legacy


SEED = 42
PAIR_FIT_MINIMUM = 4_000
PAIR_FIT_PER_UNIT = 100
# A genuine upper bound.  Eight thousand retains the historical 7,000-cell
# lung run while preventing unbounded quadratic similarity matrices and cubic
# Hungarian assignments when a dataset contains hundreds of units.
PAIR_FIT_MAXIMUM = 8_000
GRAPH_K = 15
GRID = np.linspace(0.0, 1.0, 257)
MAX_DISPLACEMENT = 0.75
STRENGTH_RIDGE = 0.25
TARGET_EFFECTIVE_NEIGHBORS = 15
CURVE_MAX_BINS = 20
CURVE_MIN_BIN_CELLS = 5
COMPATIBILITY_MAXIMUM = 3_500
PERCENTILE_DEFINITION = "within-unit midrank: (average_rank - 0.5) / n"
__version__ = "2.2.0-ver1"


stable_seed = _legacy.stable_seed
slug = _legacy.slug
kmeanslow_global = _legacy.kmeanslow_global
rank_inversion_count = _legacy.rank_inversion_count
row_normalize = _legacy.row_normalize
correlation_signature = _legacy.correlation_signature
cycle_quality = _legacy.cycle_quality
calibrated_temperature = _legacy.calibrated_temperature
reciprocal_pivot = _legacy.reciprocal_pivot
score_coupling = _legacy.score_coupling
weighted_median = _legacy.weighted_median
score_weights = _legacy.score_weights
optimal_alpha = _legacy.optimal_alpha
weighted_z = _legacy.weighted_z
pair_compatibility = _legacy.pair_compatibility
_validated_metadata = _legacy._validated_metadata
Benchmark = _legacy.Benchmark
DirectionalTarget = _legacy.DirectionalTarget


@dataclass
class RampResult:
    baseline: np.ndarray
    aligned: np.ndarray
    coupling_audit: pd.DataFrame
    strength_audit: pd.DataFrame
    evidence_audit: pd.DataFrame
    reference_weight_audit: pd.DataFrame
    strict_rank_inversions: int
    profile_audit: pd.DataFrame
    algorithm_audit: pd.DataFrame


def _nonself_knn(
    values: np.ndarray,
    k: int = GRAPH_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Return up to ``k`` nearest *non-self* neighbours for every row.

    Self is removed by row identity rather than by blindly dropping the first
    neighbour, so duplicate observations do not create an off-by-one error.
    """
    values = np.asarray(values, dtype=np.float64)
    n_rows = len(values)
    k_use = min(int(k), max(0, n_rows - 1))
    if k_use == 0:
        return (
            np.empty((n_rows, 0), dtype=np.float64),
            np.empty((n_rows, 0), dtype=np.int64),
        )
    distance, index = NearestNeighbors(
        n_neighbors=k_use + 1, n_jobs=1
    ).fit(values).kneighbors(values, return_distance=True)
    out_distance = np.empty((n_rows, k_use), dtype=np.float64)
    out_index = np.empty((n_rows, k_use), dtype=np.int64)
    for row in range(n_rows):
        keep = index[row] != row
        candidates = index[row, keep]
        candidate_distance = distance[row, keep]
        if len(candidates) < k_use:
            raise RuntimeError("nearest-neighbour query did not return enough non-self rows")
        out_index[row] = candidates[:k_use]
        out_distance[row] = candidate_distance[:k_use]
    return out_distance, out_index


def graph_smooth(values: np.ndarray, k: int = GRAPH_K) -> np.ndarray:
    _, neighbors = _nonself_knn(values, k)
    if neighbors.shape[1] == 0:
        return np.asarray(values, dtype=np.float64).copy()
    values = np.asarray(values, dtype=np.float64)
    return 0.5 * values + 0.5 * values[neighbors].mean(axis=1)


def graph_smooth_and_normalize(values: np.ndarray) -> np.ndarray:
    return row_normalize(graph_smooth(values))


def transition(values: np.ndarray, k: int = GRAPH_K) -> sparse.csr_matrix:
    distance, index = _nonself_knn(values, k)
    n_neighbors = index.shape[1]
    if n_neighbors == 0:
        return sparse.eye(len(values), format="csr", dtype=np.float64)
    sigma = np.maximum(distance[:, -1:], 1e-8)
    weight = np.exp(-0.5 * (distance / sigma) ** 2)
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
    row = np.repeat(np.arange(len(values)), n_neighbors)
    return sparse.csr_matrix(
        (weight.ravel(), (row, index.ravel())), shape=(len(values), len(values))
    )


def local_radius(values: np.ndarray, k: int = GRAPH_K) -> np.ndarray:
    distance, _ = _nonself_knn(values, k)
    if distance.shape[1] == 0:
        return np.full(len(values), 1e-8, dtype=np.float64)
    return np.maximum(distance[:, -1], 1e-8)


def within_unit_percentiles(values: np.ndarray) -> np.ndarray:
    """Canonical percentile used by fit, identity, copula, and final mapping."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("percentile input must be a non-empty one-dimensional array")
    return (rankdata(values, method="average") - 0.5) / len(values)


def empirical_profile(values: np.ndarray) -> np.ndarray:
    """Monotone inverse curve exactly paired with ``within_unit_percentiles``."""
    values = np.asarray(values, dtype=np.float64)
    u = within_unit_percentiles(values)
    order = np.argsort(values, kind="stable")
    x = values[order]
    p = u[order]
    unique_x, first = np.unique(x, return_index=True)
    unique_p = p[first]
    return np.interp(GRID, unique_p, unique_x, left=unique_x[0], right=unique_x[-1])


def bounded_profile_mapping(
    values: np.ndarray,
    percentiles: np.ndarray,
    profile: np.ndarray,
) -> np.ndarray:
    """The single mapper used by both calibration and final application."""
    values = np.asarray(values, dtype=np.float64)
    desired = np.interp(np.asarray(percentiles, dtype=np.float64), GRID, profile)
    return values + np.clip(
        desired - values, -MAX_DISPLACEMENT, MAX_DISPLACEMENT
    )


def donor_copula(raw: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
    answer = np.empty_like(raw, dtype=np.float64)
    for _, group in metadata.groupby(metadata.unit_key.astype(str), sort=True):
        rows = group.index.to_numpy(int)
        for marker in range(raw.shape[1]):
            u = within_unit_percentiles(raw[rows, marker])
            n = len(rows)
            answer[rows, marker] = norm.ppf(
                np.clip(u, 1 / (2 * n), 1 - 1 / (2 * n))
            )
    return answer


def unit_balanced_fit_indices(
    metadata: pd.DataFrame,
    dataset: str,
    maximum: int,
) -> np.ndarray:
    rows = np.flatnonzero(metadata.dataset.astype(str).eq(dataset))
    frame = metadata.iloc[rows]
    if len(rows) <= maximum:
        return rows
    groups = list(frame.groupby("unit_key", sort=True))
    if len(groups) >= maximum:
        rng = np.random.default_rng(stable_seed(dataset, "pair-fit-unit-subset"))
        chosen_groups = np.sort(rng.choice(len(groups), maximum, replace=False))
        selected = []
        for group_index in chosen_groups:
            values = groups[group_index][1].index.to_numpy(int)
            cell_rng = np.random.default_rng(
                stable_seed(dataset, group_index, "pair-fit-single-cell")
            )
            selected.append(int(cell_rng.choice(values)))
        return np.sort(np.asarray(selected, dtype=int))
    quota = max(1, maximum // len(groups))
    selected, leftovers = [], []
    for index, (_, group) in enumerate(groups):
        values = group.index.to_numpy(int)
        rng = np.random.default_rng(stable_seed(dataset, index, "pair-fit"))
        permutation = rng.permutation(values)
        take = min(quota, len(permutation))
        selected.extend(permutation[:take].tolist())
        leftovers.extend(permutation[take:].tolist())
    need = maximum - len(selected)
    if need > 0:
        rng = np.random.default_rng(stable_seed(dataset, "pair-fit-left"))
        selected.extend(
            rng.choice(np.asarray(leftovers), need, replace=False).tolist()
        )
    return np.sort(np.asarray(selected, dtype=int))


def adaptive_pair_fit_size(
    metadata: pd.DataFrame,
    datasets: list[str],
) -> tuple[int, dict[str, object]]:
    dataset_column = metadata.dataset.astype(str)
    unit_column = metadata.unit_key.astype(str)
    dataset_sizes = {
        dataset: int(dataset_column.eq(dataset).sum()) for dataset in datasets
    }
    unit_counts = {
        dataset: int(unit_column[dataset_column.eq(dataset)].nunique())
        for dataset in datasets
    }
    largest_unit_count = max(unit_counts.values())
    unbounded = max(PAIR_FIT_MINIMUM, PAIR_FIT_PER_UNIT * largest_unit_count)
    requested = min(unbounded, PAIR_FIT_MAXIMUM)
    smallest_dataset = min(dataset_sizes.values())
    effective = min(requested, smallest_dataset)
    audit: dict[str, object] = {
        "pair_fit_minimum": PAIR_FIT_MINIMUM,
        "pair_fit_per_unit": PAIR_FIT_PER_UNIT,
        "pair_fit_maximum": PAIR_FIT_MAXIMUM,
        "pair_fit_requested_before_maximum": unbounded,
        "pair_fit_requested": requested,
        "pair_fit_effective": effective,
        "pair_fit_limited_by_maximum": requested < unbounded,
        "pair_fit_limited_by_smallest_dataset": effective < requested,
        "largest_dataset_unit_count": largest_unit_count,
        "dataset_sizes": dataset_sizes,
        "dataset_unit_counts": unit_counts,
    }
    return effective, audit


def build_dual_view_bundle(
    family: str,
    raw: np.ndarray,
    shared: np.ndarray,
    metadata: pd.DataFrame,
    datasets: list[str],
):
    common_fit_size, fit_size_audit = adaptive_pair_fit_size(metadata, datasets)
    samples = {
        dataset: unit_balanced_fit_indices(metadata, dataset, common_fit_size)
        for dataset in datasets
    }
    source_spaces = {
        dataset: graph_smooth_and_normalize(correlation_signature(raw[rows]))
        for dataset, rows in samples.items()
    }
    baseline_spaces = {
        dataset: graph_smooth_and_normalize(correlation_signature(shared[rows]))
        for dataset, rows in samples.items()
    }
    spaces = {
        dataset: row_normalize(
            np.hstack([source_spaces[dataset], baseline_spaces[dataset]])
        )
        for dataset in datasets
    }
    copula_all = donor_copula(raw, metadata)
    copula_spaces = {
        dataset: graph_smooth(copula_all[rows]) for dataset, rows in samples.items()
    }
    maps, scores, similarities = {}, {}, {}
    for left_index, left in enumerate(datasets):
        for right in datasets[left_index + 1 :]:
            similarity = np.asarray(
                spaces[left] @ spaces[right].T, dtype=np.float32
            )
            similarities[(left, right)] = similarity
            similarities[(right, left)] = similarity.T
            row, column = linear_sum_assignment(similarity, maximize=True)
            mapping = np.empty(len(row), dtype=int)
            mapping[row] = column
            reverse = np.empty(len(column), dtype=int)
            reverse[column] = row
            assigned = np.clip(
                (similarity[row, column].astype(float) + 1.0) / 2.0, 1e-6, 1.0
            )
            ordered = np.empty(len(row), dtype=float)
            ordered[row] = assigned
            reversed_score = np.empty(len(column), dtype=float)
            reversed_score[mapping] = ordered
            maps[(left, right)] = mapping
            maps[(right, left)] = reverse
            scores[(left, right)] = ordered
            scores[(right, left)] = reversed_score
    radius = {dataset: local_radius(spaces[dataset]) for dataset in datasets}
    transitions = {dataset: transition(spaces[dataset]) for dataset in datasets}
    return (
        samples,
        spaces,
        copula_spaces,
        maps,
        scores,
        similarities,
        radius,
        transitions,
        fit_size_audit,
    )


def deterministic_roles(
    family: str,
    query: str,
    reference: str,
    indices: np.ndarray,
) -> np.ndarray:
    # Role allocation is deliberately unchanged in ver1; the seven fixes are
    # isolated so their benchmark effect can be interpreted separately.
    return _legacy.deterministic_roles(family, query, reference, indices)


def barycentric_direction(
    family: str,
    query: str,
    reference: str,
    score_matrix: sparse.csr_matrix,
    query_global: np.ndarray,
    reference_global: np.ndarray,
    shared: np.ndarray,
) -> DirectionalTarget:
    score_matrix = score_matrix.tocsr()
    row_sum = np.asarray(score_matrix.sum(axis=1)).ravel()
    if np.any(row_sum <= 0):
        raise RuntimeError(f"Zero coupling mass in {query} -> {reference}")
    normalized = sparse.diags(1.0 / row_sum) @ score_matrix
    target = np.asarray(normalized @ shared[reference_global], dtype=np.float64)
    concentration = np.asarray(
        normalized.multiply(normalized).sum(axis=1)
    ).ravel()
    confidence = np.asarray(
        score_matrix.multiply(score_matrix).sum(axis=1)
    ).ravel() / np.maximum(row_sum, 1e-12)
    return DirectionalTarget(
        query,
        reference,
        np.asarray(query_global, int),
        target,
        np.asarray(confidence, float),
        1.0 / np.maximum(concentration, 1e-12),
        deterministic_roles(family, query, reference, query_global),
    )


def build_v40_targets(
    family: str,
    raw: np.ndarray,
    shared: np.ndarray,
    metadata: pd.DataFrame,
    datasets: list[str],
):
    if len(datasets) not in (2, 3):
        raise ValueError("RAMP currently requires exactly two or three datasets")
    bundle = build_dual_view_bundle(family, raw, shared, metadata, datasets)
    (
        samples,
        spaces,
        copula_spaces,
        maps,
        scores,
        similarities,
        radius,
        transitions,
        fit_size_audit,
    ) = bundle
    directions, audit_rows = [], []
    for left_index, left in enumerate(datasets):
        for right in datasets[left_index + 1 :]:
            if len(datasets) == 3:
                third = next(x for x in datasets if x not in (left, right))
                cycle_weight = cycle_quality(
                    left, right, third, spaces, maps, radius
                )
            else:
                third = ""
                cycle_weight = np.ones_like(scores[(left, right)])
            hard_weight = scores[(left, right)] * cycle_weight
            hard = sparse.csr_matrix(
                (
                    hard_weight,
                    (np.arange(len(hard_weight)), maps[(left, right)]),
                ),
                shape=(len(spaces[left]), len(spaces[right])),
            )
            tau, effective = calibrated_temperature(similarities[(left, right)])
            soft, soft_audit = reciprocal_pivot(similarities[(left, right)], tau)
            pivot = (hard + soft).tocsr()
            pivot.eliminate_zeros()
            pivot_row, pivot_column = pivot.nonzero()
            delta = (
                copula_spaces[right][pivot_column]
                - copula_spaces[left][pivot_row]
            )
            copula_scale = max(
                float(np.median(np.sum(delta * delta, axis=1))), 1e-8
            )
            coupling = score_coupling(
                left,
                right,
                pivot,
                spaces,
                copula_spaces,
                transitions,
                copula_scale,
            )
            pair = (
                barycentric_direction(
                    family,
                    left,
                    right,
                    coupling,
                    samples[left],
                    samples[right],
                    shared,
                ),
                barycentric_direction(
                    family,
                    right,
                    left,
                    coupling.T.tocsr(),
                    samples[right],
                    samples[left],
                    shared,
                ),
            )
            directions.extend(pair)
            hard_mass = np.asarray(hard.sum(axis=1)).ravel()
            hard_mass = hard_mass[hard_mass > 0]
            soft_mass = np.asarray(soft.sum(axis=1)).ravel()
            soft_mass = soft_mass[soft_mass > 0]
            audit_rows.append(
                {
                    "family": family,
                    "left": left,
                    "right": right,
                    "third": third,
                    **{
                        key: fit_size_audit[key]
                        for key in (
                            "pair_fit_minimum",
                            "pair_fit_per_unit",
                            "pair_fit_maximum",
                            "pair_fit_requested_before_maximum",
                            "pair_fit_requested",
                            "pair_fit_effective",
                            "pair_fit_limited_by_maximum",
                            "pair_fit_limited_by_smallest_dataset",
                            "largest_dataset_unit_count",
                        )
                    },
                    "graph_k_nonself": GRAPH_K,
                    "transition_includes_self": False,
                    "hard_median_positive_row_mass": float(np.median(hard_mass)),
                    "soft_median_positive_row_mass": float(np.median(soft_mass)),
                    "implicit_soft_mass_fraction": float(np.median(soft_mass))
                    / (
                        float(np.median(hard_mass))
                        + float(np.median(soft_mass))
                    ),
                    "calibrated_median_effective_neighbors": effective,
                    "copula_scale": copula_scale,
                    "median_effective_references_left": float(
                        np.median(pair[0].effective_references)
                    ),
                    "median_effective_references_right": float(
                        np.median(pair[1].effective_references)
                    ),
                    **soft_audit,
                }
            )
            del hard, soft, pivot, coupling
    return directions, pd.DataFrame(audit_rows)


def unit_lookup(metadata: pd.DataFrame):
    return _legacy.unit_lookup(metadata)


def query_quantiles(
    shared: np.ndarray,
    unit_cells: np.ndarray,
    query_indices: np.ndarray,
    marker: int,
) -> np.ndarray:
    unit_cells = np.asarray(unit_cells, dtype=int)
    query_indices = np.asarray(query_indices, dtype=int)
    positions = np.searchsorted(unit_cells, query_indices)
    if (
        np.any(positions >= len(unit_cells))
        or np.any(unit_cells[np.minimum(positions, len(unit_cells) - 1)] != query_indices)
    ):
        raise ValueError("query indices must be a subset of unit_cells")
    return within_unit_percentiles(shared[unit_cells, marker])[positions]


def fit_profile(
    u: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, object]]:
    n_cells = len(u)
    n_bins = min(CURVE_MAX_BINS, n_cells // CURVE_MIN_BIN_CELLS)
    audit: dict[str, object] = {
        "curve_cells": int(n_cells),
        "curve_bins_requested": int(n_bins),
        "curve_equal_count_bins": True,
        "curve_minimum_bin_cells": CURVE_MIN_BIN_CELLS,
        "curve_profile_fitted": False,
    }
    if n_cells < 40 or n_bins < 4:
        return None, audit
    weights = score_weights(scores)
    order = np.argsort(u, kind="stable")
    groups = np.array_split(order, n_bins)
    ku, ky, kw, counts = [], [], [], []
    for group in groups:
        if len(group) == 0:
            continue
        ku.append(float(np.average(u[group], weights=weights[group])))
        ky.append(weighted_median(y[group], weights[group]))
        kw.append(float(weights[group].sum()))
        counts.append(int(len(group)))
    ku_array = np.asarray(ku)
    ky_array = np.asarray(ky)
    kw_array = np.asarray(kw)
    bin_order = np.argsort(ku_array, kind="stable")
    ku_array = ku_array[bin_order]
    ky_array = ky_array[bin_order]
    kw_array = kw_array[bin_order]
    fitted = IsotonicRegression(
        increasing=True, out_of_bounds="clip"
    ).fit_transform(ku_array, ky_array, sample_weight=kw_array)
    # Equal percentiles caused by ties may give duplicate bin centres.  Merge
    # those centres before interpolation while preserving their total weight.
    unique_u, inverse = np.unique(ku_array, return_inverse=True)
    merged = np.empty(len(unique_u), dtype=np.float64)
    for index in range(len(unique_u)):
        select = inverse == index
        merged[index] = np.average(fitted[select], weights=kw_array[select])
    audit.update(
        {
            "curve_bins_used": int(len(groups)),
            "curve_unique_bin_centres": int(len(unique_u)),
            "curve_smallest_bin": int(min(counts)),
            "curve_largest_bin": int(max(counts)),
        }
    )
    if len(unique_u) < 4:
        return None, audit
    audit["curve_profile_fitted"] = True
    return (
        np.interp(
            GRID,
            unique_u,
            merged,
            left=float(merged[0]),
            right=float(merged[-1]),
        ),
        audit,
    )


def fit_profiles(shared, metadata, directions):
    profiles, audit_rows = {}, []
    _, cells_by_unit, dataset_by_unit = unit_lookup(metadata)
    for direction in directions:
        role = direction.role == "curve_fit"
        for unit, unit_cells in cells_by_unit.items():
            if dataset_by_unit[unit] != direction.query_dataset:
                continue
            select = role & np.isin(direction.query_global, unit_cells)
            if select.sum() < 40:
                continue
            query = direction.query_global[select]
            confidence = direction.confidence[select]
            for marker in range(shared.shape[1]):
                u = query_quantiles(shared, unit_cells, query, marker)
                profile, profile_audit = fit_profile(
                    u, direction.target[select, marker], confidence
                )
                audit_rows.append(
                    {
                        "query_dataset": direction.query_dataset,
                        "reference_dataset": direction.reference_dataset,
                        "unit_key": unit,
                        "marker_index": marker,
                        **profile_audit,
                    }
                )
                if profile is not None:
                    profiles[(unit, marker, direction.reference_dataset)] = profile
    return profiles, pd.DataFrame(audit_rows)


def proposed_target(shared, query_indices, unit_cells, marker, profile):
    u = query_quantiles(shared, unit_cells, query_indices, marker)
    return bounded_profile_mapping(shared[query_indices, marker], u, profile)


def calibrate_strengths(shared, metadata, directions, datasets, profiles):
    # Logic is retained, but every calibration proposal now passes through the
    # same bounded_profile_mapping used by final application.
    strengths, audit_rows = {}, []
    _, cells_by_unit, dataset_by_unit = unit_lookup(metadata)
    for query in datasets:
        query_units = [u for u in cells_by_unit if dataset_by_unit[u] == query]
        for reference in datasets:
            if query == reference:
                continue
            direction = next(
                d
                for d in directions
                if d.query_dataset == query and d.reference_dataset == reference
            )
            role_mask = direction.role == "strength_calibration"
            for marker in range(shared.shape[1]):
                pieces = []
                for unit in query_units:
                    profile = profiles.get((unit, marker, reference))
                    if profile is None:
                        continue
                    unit_cells = cells_by_unit[unit]
                    select = role_mask & np.isin(direction.query_global, unit_cells)
                    if not np.any(select):
                        continue
                    q = direction.query_global[select]
                    pieces.append(
                        (
                            unit,
                            shared[q, marker],
                            direction.target[select, marker],
                            proposed_target(
                                shared, q, unit_cells, marker, profile
                            ),
                            direction.confidence[select],
                        )
                    )
                if not pieces:
                    continue
                x = np.concatenate([piece[1] for piece in pieces])
                y = np.concatenate([piece[2] for piece in pieces])
                target = np.concatenate([piece[3] for piece in pieces])
                confidence = np.concatenate([piece[4] for piece in pieces])
                weights = score_weights(confidence)
                dataset_alpha, dataset_gain = optimal_alpha(
                    x, y, target, weights
                )
                by_unit = {piece[0]: piece for piece in pieces}
                for unit in query_units:
                    piece = by_unit.get(unit)
                    if piece is not None:
                        n = len(piece[1])
                        local_weights = score_weights(piece[4])
                        donor_alpha, donor_gain = (
                            optimal_alpha(
                                piece[1], piece[2], piece[3], local_weights
                            )
                            if n >= 6
                            else (dataset_alpha, dataset_gain)
                        )
                        donor_weight = n / (n + 30.0) if n >= 6 else 0.0
                    else:
                        n = 0
                        donor_alpha, donor_gain, donor_weight = (
                            dataset_alpha,
                            dataset_gain,
                            0.0,
                        )
                    alpha = float(
                        donor_weight * donor_alpha
                        + (1 - donor_weight) * dataset_alpha
                    )
                    strengths[(unit, marker, reference)] = alpha
                    audit_rows.append(
                        {
                            "query_dataset": query,
                            "reference_dataset": reference,
                            "unit_key": unit,
                            "marker_index": marker,
                            "calibration_queries": n,
                            "dataset_alpha": dataset_alpha,
                            "donor_alpha": donor_alpha,
                            "adaptive_alpha": alpha,
                            "dataset_gain": dataset_gain,
                            "donor_gain": donor_gain,
                            "mapping_operator": "bounded_profile_mapping",
                            "maximum_displacement": MAX_DISPLACEMENT,
                        }
                    )
    return strengths, pd.DataFrame(audit_rows)


def _provider_unit_z(unit_means: np.ndarray) -> tuple[float, float, float]:
    """Provider evidence with units as the independent observations."""
    unit_means = np.asarray(unit_means, dtype=np.float64)
    if len(unit_means) < 2:
        return 0.0, float(unit_means.mean()) if len(unit_means) else 0.0, 0.0
    mean = float(unit_means.mean())
    se = float(unit_means.std(ddof=1) / np.sqrt(len(unit_means)))
    z = float(np.clip(mean / max(se, 1e-12), -6, 6))
    return z, mean, se


def evidence_scores(shared, metadata, directions, datasets, profiles, strengths):
    evidence, audit_rows = {}, []
    _, cells_by_unit, dataset_by_unit = unit_lookup(metadata)
    for query in datasets:
        query_units = [u for u in cells_by_unit if dataset_by_unit[u] == query]
        for reference in datasets:
            if query == reference:
                continue
            direction = next(
                d
                for d in directions
                if d.query_dataset == query and d.reference_dataset == reference
            )
            role_mask = direction.role == "weight_calibration"
            for marker in range(shared.shape[1]):
                pieces = {}
                for unit in query_units:
                    profile = profiles.get((unit, marker, reference))
                    if profile is None:
                        continue
                    unit_cells = cells_by_unit[unit]
                    select = role_mask & np.isin(direction.query_global, unit_cells)
                    if not np.any(select):
                        continue
                    q = direction.query_global[select]
                    x = shared[q, marker]
                    y = direction.target[select, marker]
                    target = proposed_target(
                        shared, q, unit_cells, marker, profile
                    )
                    alpha = strengths.get((unit, marker, reference), 0.0)
                    delta = (x - y) ** 2 - (
                        x + alpha * (target - x) - y
                    ) ** 2
                    confidence = direction.confidence[select]
                    weights = score_weights(confidence)
                    unit_mean = float(np.average(delta, weights=weights))
                    unit_neff = float(
                        weights.sum() ** 2 / max(np.sum(weights**2), 1e-12)
                    )
                    pieces[unit] = (delta, weights, unit_mean, unit_neff)
                if not pieces:
                    for unit in query_units:
                        evidence[(unit, marker, reference)] = -3.0
                    continue
                unit_means = np.asarray(
                    [pieces[unit][2] for unit in query_units if unit in pieces]
                )
                dataset_z, dataset_mean, dataset_se = _provider_unit_z(
                    unit_means
                )
                for unit in query_units:
                    piece = pieces.get(unit)
                    if piece is not None:
                        delta, weights, _, unit_neff = piece
                        n = len(delta)
                        donor_z, _, _, _ = (
                            weighted_z(delta, weights)
                            if n >= 6
                            else (dataset_z, 0.0, 0.0, 0.0)
                        )
                        shrink = n / (n + 30.0) if n >= 6 else 0.0
                    else:
                        n = 0
                        donor_z, shrink, unit_neff = dataset_z, 0.0, 0.0
                    z = float(shrink * donor_z + (1 - shrink) * dataset_z)
                    evidence[(unit, marker, reference)] = z / 2.0
                    audit_rows.append(
                        {
                            "query_dataset": query,
                            "reference_dataset": reference,
                            "unit_key": unit,
                            "marker_index": marker,
                            "dataset_z": dataset_z,
                            "donor_z": donor_z,
                            "hierarchical_z": z,
                            "reference_log_evidence": z / 2.0,
                            "dataset_mean_error_reduction": dataset_mean,
                            "dataset_unit_standard_error": dataset_se,
                            "provider_units": int(len(unit_means)),
                            "provider_evidence_unit_aggregated": True,
                            "donor_queries": n,
                            "donor_effective_queries": unit_neff,
                            "donor_shrinkage": shrink,
                        }
                    )
    return evidence, pd.DataFrame(audit_rows)


def compatibility_dict(
    family: str,
    baseline: np.ndarray,
    metadata: pd.DataFrame,
    datasets: list[str],
):
    del family  # Sampling is determined by dataset/unit identity and the seed.
    dataset_sizes = {
        dataset: int(metadata.dataset.astype(str).eq(dataset).sum())
        for dataset in datasets
    }
    common_size = min(COMPATIBILITY_MAXIMUM, min(dataset_sizes.values()))
    indices = {
        dataset: unit_balanced_fit_indices(metadata, dataset, common_size)
        for dataset in datasets
    }
    balanced = np.vstack([baseline[indices[dataset]] for dataset in datasets])
    keep = np.nanstd(balanced, axis=0) > 1e-10
    scaler = StandardScaler().fit(balanced[:, keep])
    standardized = scaler.transform(balanced[:, keep])
    components = min(20, standardized.shape[1])
    pca = None
    if standardized.shape[1] > components:
        pca = PCA(n_components=components, random_state=SEED).fit(standardized)
    spaces, offset = {}, 0
    for dataset in datasets:
        block = standardized[offset : offset + common_size]
        spaces[dataset] = (
            pca.transform(block).astype(np.float32)
            if pca is not None
            else block.astype(np.float32)
        )
        offset += common_size
    answer = {}
    rows = []
    for left, right in itertools.combinations(datasets, 2):
        score = pair_compatibility(spaces[left], spaces[right])
        answer[(left, right)] = answer[(right, left)] = score
        rows.append(
            {
                "left": left,
                "right": right,
                "compatibility": score,
                "compatibility_common_sample_size": common_size,
                "compatibility_dataset_balanced": True,
                "compatibility_preprocessing_fit_balanced": True,
                "compatibility_pca_applied": pca is not None,
                "compatibility_pca_fit_balanced": True,
                "compatibility_pca_fit_rows": common_size * len(datasets),
                "compatibility_pca_components": spaces[left].shape[1],
            }
        )
    return answer, pd.DataFrame(rows)


def automatic_profiles(
    baseline,
    metadata,
    datasets,
    profiles,
    strengths,
    evidence,
    compatibility,
    compatibility_audit,
):
    final, audit = {}, []
    unit_column = metadata.unit_key.astype(str)
    compatibility_lookup = {
        (row.left, row.right): row
        for row in compatibility_audit.itertuples(index=False)
    }
    compatibility_lookup.update(
        {
            (right, left): row
            for (left, right), row in list(compatibility_lookup.items())
        }
    )
    for unit in sorted(pd.unique(unit_column)):
        cells = np.flatnonzero(unit_column.eq(unit))
        dataset = str(metadata.iloc[cells[0]].dataset)
        for marker in range(baseline.shape[1]):
            identity = empirical_profile(baseline[cells, marker])
            candidates, scores, refs, alphas, compatibilities = [], [], [], [], []
            prelimit_maxima = []
            for reference in datasets:
                if reference == dataset:
                    continue
                profile = profiles.get((unit, marker, reference))
                if profile is None:
                    continue
                alpha = strengths.get((unit, marker, reference), 0.0)
                difference = profile - identity
                # This is the grid form of the exact same bounded mapping used
                # during calibration, followed by the calibrated alpha.
                candidate = identity + alpha * np.clip(
                    difference, -MAX_DISPLACEMENT, MAX_DISPLACEMENT
                )
                candidates.append(candidate)
                prelimit_maxima.append(float(np.max(np.abs(difference))))
                compatibility_score = compatibility[(dataset, reference)]
                scores.append(
                    evidence.get((unit, marker, reference), -3.0)
                    + np.log(max(compatibility_score, 1e-6))
                )
                refs.append(reference)
                alphas.append(alpha)
                compatibilities.append(compatibility_score)
            if not candidates:
                continue
            scores_array = np.asarray(scores)
            relative = np.exp(scores_array - scores_array.max())
            relative /= relative.sum()
            alignment_mass = float(
                np.sum(relative * np.asarray(compatibilities) ** 2)
            )
            reference_weights = alignment_mass * relative
            target = (1 - alignment_mass) * identity
            for weight, candidate in zip(reference_weights, candidates):
                target += weight * candidate
            monotone = np.asarray(
                IsotonicRegression(increasing=True).fit_transform(GRID, target)
            )
            final[(unit, marker)] = monotone
            row = {
                "dataset": dataset,
                "unit_key": unit,
                "marker_index": marker,
                "identity_weight": 1 - alignment_mass,
                "alignment_mass": alignment_mass,
                "maximum_displacement": MAX_DISPLACEMENT,
                "final_mapping_operator": "bounded_profile_mapping",
                "max_reference_profile_displacement_before_limit": max(
                    prelimit_maxima
                ),
                "max_final_profile_displacement_before_application": float(
                    np.max(np.abs(monotone - identity))
                ),
            }
            for ref, weight, alpha, score in zip(
                refs, reference_weights, alphas, compatibilities
            ):
                row[f"weight__{slug(ref)}"] = weight
                row[f"alpha__{slug(ref)}"] = alpha
                row[f"compatibility__{slug(ref)}"] = score
                compatibility_row = compatibility_lookup[(dataset, ref)]
                row["compatibility_common_sample_size"] = int(
                    compatibility_row.compatibility_common_sample_size
                )
                row["compatibility_dataset_balanced"] = bool(
                    compatibility_row.compatibility_dataset_balanced
                )
                row["compatibility_pca_fit_balanced"] = bool(
                    compatibility_row.compatibility_pca_fit_balanced
                )
                row["compatibility_pca_applied"] = bool(
                    compatibility_row.compatibility_pca_applied
                )
                row["compatibility_pca_fit_rows"] = int(
                    compatibility_row.compatibility_pca_fit_rows
                )
                row["compatibility_pca_components"] = int(
                    compatibility_row.compatibility_pca_components
                )
            audit.append(row)
    return final, pd.DataFrame(audit)


def apply_monotone_profiles(
    baseline: np.ndarray,
    metadata: pd.DataFrame,
    profiles: dict,
) -> np.ndarray:
    answer = baseline.copy()
    unit_column = metadata.unit_key.astype(str)
    for (unit, marker), profile in profiles.items():
        cells = np.flatnonzero(unit_column.eq(unit))
        u = within_unit_percentiles(baseline[cells, marker])
        answer[cells, marker] = bounded_profile_mapping(
            baseline[cells, marker], u, profile
        )
    return answer


def fit_v40_v21(benchmark: Benchmark, baseline: np.ndarray):
    print(f"{benchmark.family}: build corrected coupling", flush=True)
    directions, coupling_audit = build_v40_targets(
        benchmark.family,
        benchmark.raw,
        baseline,
        benchmark.metadata,
        benchmark.datasets,
    )
    print(f"{benchmark.family}: fit corrected monotone backend", flush=True)
    profiles, profile_audit = fit_profiles(
        baseline, benchmark.metadata, directions
    )
    strengths, strength_audit = calibrate_strengths(
        baseline,
        benchmark.metadata,
        directions,
        benchmark.datasets,
        profiles,
    )
    evidence, evidence_audit = evidence_scores(
        baseline,
        benchmark.metadata,
        directions,
        benchmark.datasets,
        profiles,
        strengths,
    )
    compatibility, compatibility_audit = compatibility_dict(
        benchmark.family,
        baseline,
        benchmark.metadata,
        benchmark.datasets,
    )
    final_profiles, weight_audit = automatic_profiles(
        baseline,
        benchmark.metadata,
        benchmark.datasets,
        profiles,
        strengths,
        evidence,
        compatibility,
        compatibility_audit,
    )
    candidate = apply_monotone_profiles(
        baseline, benchmark.metadata, final_profiles
    )
    return (
        candidate,
        coupling_audit,
        strength_audit,
        evidence_audit,
        weight_audit,
        profile_audit,
        compatibility_audit,
    )


def _algorithm_audit(
    baseline: np.ndarray,
    aligned: np.ndarray,
    compatibility_audit: pd.DataFrame,
) -> pd.DataFrame:
    displacement = np.abs(aligned - baseline)
    values: list[tuple[str, object]] = [
        ("algorithm_version", __version__),
        ("maximum_displacement", MAX_DISPLACEMENT),
        ("mapping_operator_calibration", "bounded_profile_mapping"),
        ("mapping_operator_final", "bounded_profile_mapping"),
        ("observed_maximum_displacement", float(displacement.max(initial=0.0))),
        (
            "cell_marker_values_over_maximum",
            int(np.sum(displacement > MAX_DISPLACEMENT + 1e-10)),
        ),
        (
            "cell_marker_fraction_at_maximum",
            float(np.mean(displacement >= MAX_DISPLACEMENT - 1e-8)),
        ),
        ("percentile_definition", PERCENTILE_DEFINITION),
        ("curve_binning", "adaptive equal-count"),
        ("curve_maximum_bins", CURVE_MAX_BINS),
        ("curve_minimum_cells_per_bin", CURVE_MIN_BIN_CELLS),
        ("compatibility_dataset_balanced", True),
        ("compatibility_preprocessing_fit_balanced", True),
        (
            "compatibility_pca_applied",
            bool(compatibility_audit.compatibility_pca_applied.iloc[0]),
        ),
        (
            "compatibility_pca_fit_source",
            "equal-sized dataset samples (when PCA is required)",
        ),
        (
            "compatibility_common_sample_size",
            int(compatibility_audit.compatibility_common_sample_size.iloc[0]),
        ),
        ("provider_evidence_unit_aggregated", True),
        ("pair_fit_maximum", PAIR_FIT_MAXIMUM),
        ("graph_k_nonself", GRAPH_K),
        ("transition_includes_self", False),
    ]
    return pd.DataFrame(values, columns=["audit_key", "audit_value"])


def fit_transform(
    raw: np.ndarray,
    metadata: pd.DataFrame,
    markers: list[str],
    *,
    datasets: list[str] | None = None,
    family: str = "ramp_ver1_run",
) -> RampResult:
    """Run RAMP ver1 on raw shared-marker matrices without using labels."""
    raw, fit_metadata, markers, datasets = _validated_metadata(
        raw, metadata, markers, datasets
    )
    baseline = kmeanslow_global(
        raw,
        fit_metadata["unit_key"].to_numpy(),
        fit_metadata["dataset"].to_numpy(),
        markers,
    )
    benchmark = Benchmark(
        family=str(family),
        raw=raw,
        metadata=fit_metadata,
        markers=markers,
        datasets=datasets,
        reference=datasets[0],
    )
    (
        aligned,
        coupling,
        strength,
        evidence,
        weights,
        profile_audit,
        compatibility_audit,
    ) = fit_v40_v21(benchmark, baseline)
    displacement = np.abs(aligned - baseline)
    if np.any(displacement > MAX_DISPLACEMENT + 1e-10):
        raise AssertionError("RAMP ver1 exceeded MAX_DISPLACEMENT")
    inversions = rank_inversion_count(raw, aligned, fit_metadata)
    if inversions:
        raise AssertionError(
            f"RAMP ver1 produced {inversions} strict within-unit rank inversions"
        )
    return RampResult(
        baseline=baseline,
        aligned=aligned,
        coupling_audit=coupling,
        strength_audit=strength,
        evidence_audit=evidence,
        reference_weight_audit=weights,
        strict_rank_inversions=inversions,
        profile_audit=profile_audit,
        algorithm_audit=_algorithm_audit(
            baseline, aligned, compatibility_audit
        ),
    )


__all__ = [
    "Benchmark",
    "RampResult",
    "fit_transform",
    "fit_v40_v21",
    "kmeanslow_global",
    "rank_inversion_count",
    "within_unit_percentiles",
    "empirical_profile",
    "bounded_profile_mapping",
    "graph_smooth",
    "transition",
    "local_radius",
    "unit_balanced_fit_indices",
    "adaptive_pair_fit_size",
    "SEED",
    "PAIR_FIT_MINIMUM",
    "PAIR_FIT_PER_UNIT",
    "PAIR_FIT_MAXIMUM",
    "GRAPH_K",
    "MAX_DISPLACEMENT",
    "__version__",
]
