#!/usr/bin/env python3
"""Unified label-free rank-preserving RAMP algorithm for 2 or 3 datasets.

This module contains algorithm definitions only.  It has no project-external
file paths and never reads precomputed matrices, pairs, embeddings, or results.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from scipy.stats import norm, rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


SEED = 42
# The matching fit uses at least 4,000 cells per dataset. Above 40 units in
# any dataset it scales at 100 fitting cells per unit, which gives a 95.6%
# binomial probability of retaining >=40 curve-fit cells under the fixed
# 12/25 curve-fit role allocation. PAIR_FIT_CAP is retained as a deprecated
# compatibility alias for scripts that previously reported the minimum.
PAIR_FIT_MINIMUM = 4_000
PAIR_FIT_PER_UNIT = 100
PAIR_FIT_CAP = PAIR_FIT_MINIMUM
GRAPH_K = 15
GRID = np.linspace(0.0, 1.0, 257)
MAX_DISPLACEMENT = 0.75
STRENGTH_RIDGE = 0.25
TARGET_EFFECTIVE_NEIGHBORS = 15
__version__ = "2.1.0"


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, (SEED, *parts))).encode()
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "little"
    ) % (2**32 - 1)


def slug(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(value)).strip("_")


@dataclass
class Benchmark:
    family: str
    raw: np.ndarray
    metadata: pd.DataFrame
    markers: list[str]
    datasets: list[str]
    reference: str


@dataclass
class RampResult:
    baseline: np.ndarray
    aligned: np.ndarray
    coupling_audit: pd.DataFrame
    strength_audit: pd.DataFrame
    evidence_audit: pd.DataFrame
    reference_weight_audit: pd.DataFrame
    strict_rank_inversions: int


def kmeanslow_global(
    raw: np.ndarray,
    units: np.ndarray,
    dataset_ids: np.ndarray,
    markers: list[str],
) -> np.ndarray:
    answer = np.empty_like(raw, dtype=np.float64)
    units = np.asarray(units).astype(str)
    dataset_ids = np.asarray(dataset_ids).astype(str)
    for unit in pd.unique(units):
        rows = np.flatnonzero(units == unit)
        dataset = dataset_ids[rows[0]]
        for j, marker in enumerate(markers):
            values = raw[rows, j].astype(float)
            shift = -float(values.min()) + 0.01 if values.min() < 0 else 0.0
            shifted = values + shift
            median, mean = float(np.median(shifted)), float(np.mean(shifted))
            cofactor = max(mean if median == 0 else median, 1e-6)
            transformed = np.arcsinh(shifted / cofactor)
            scale = max(float(np.std(transformed, ddof=0)), 1e-9)
            fit = KMeans(
                n_clusters=2, init="k-means++", n_init=20,
                max_iter=500, tol=1e-6, algorithm="lloyd",
                random_state=stable_seed("KMEANSLOW", dataset, unit, marker),
            ).fit(transformed[:, None])
            centers = fit.cluster_centers_.ravel()
            lower = int(np.argmin(centers))
            support = np.bincount(fit.labels_, minlength=2)
            minimum = max(20, int(np.ceil(0.005 * len(rows))))
            valid = (
                np.isfinite(centers).all()
                and support[lower] >= minimum
                and support[1-lower] >= minimum
                and fit.n_iter_ < 500
                and abs(np.diff(np.sort(centers))[0]) > 1e-10
            )
            low_mean = (
                float(transformed[fit.labels_ == lower].mean())
                if valid else float(transformed.mean())
            )
            answer[rows, j] = (transformed - low_mean) / scale
    return answer


def rank_inversion_count(raw: np.ndarray, transformed: np.ndarray, metadata: pd.DataFrame) -> int:
    total = 0
    for _, group in metadata.groupby("unit_key", sort=True):
        rows = group.index.to_numpy(int)
        for j in range(raw.shape[1]):
            order = np.argsort(raw[rows, j], kind="stable")
            dx = np.diff(raw[rows, j][order])
            dy = np.diff(transformed[rows, j][order])
            total += int(np.sum((dx > 0) & (dy < -1e-10)))
    return total

def row_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def correlation_signature(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return row_normalize(values - values.mean(axis=1, keepdims=True))


def graph_smooth(values: np.ndarray, k: int = GRAPH_K) -> np.ndarray:
    k_use = min(k + 1, len(values))
    neighbors = NearestNeighbors(n_neighbors=k_use, n_jobs=1).fit(values).kneighbors(
        return_distance=False
    )[:, 1:]
    if neighbors.shape[1] == 0:
        return values.copy()
    return 0.5 * values + 0.5 * values[neighbors].mean(axis=1)


def graph_smooth_and_normalize(values: np.ndarray) -> np.ndarray:
    return row_normalize(graph_smooth(values))


def donor_copula(raw: np.ndarray, metadata: pd.DataFrame) -> np.ndarray:
    answer = np.empty_like(raw, dtype=np.float64)
    for _, group in metadata.groupby(metadata.unit_key.astype(str), sort=True):
        rows = group.index.to_numpy(int)
        n = len(rows)
        for marker in range(raw.shape[1]):
            u = (rankdata(raw[rows, marker], method="average") - 0.5) / n
            answer[rows, marker] = norm.ppf(np.clip(u, 1/(2*n), 1-1/(2*n)))
    return answer


def unit_balanced_fit_indices(metadata: pd.DataFrame, dataset: str, cap: int) -> np.ndarray:
    rows = np.flatnonzero(metadata.dataset.astype(str).eq(dataset))
    frame = metadata.iloc[rows]
    if len(rows) <= cap:
        return rows
    groups = list(frame.groupby("unit_key", sort=True))
    if len(groups) >= cap:
        # It is impossible to retain every unit when there are more units than
        # fitting slots.  Select units and then one cell per selected unit
        # deterministically, while keeping the advertised hard cap.
        rng = np.random.default_rng(stable_seed(dataset, "pair-fit-unit-subset"))
        chosen_groups = np.sort(rng.choice(len(groups), cap, replace=False))
        selected = []
        for group_index in chosen_groups:
            values = groups[group_index][1].index.to_numpy(int)
            cell_rng = np.random.default_rng(
                stable_seed(dataset, group_index, "pair-fit-single-cell")
            )
            selected.append(int(cell_rng.choice(values)))
        return np.sort(np.asarray(selected, dtype=int))
    quota = max(1, cap // len(groups))
    selected, leftovers = [], []
    for i, (_, group) in enumerate(groups):
        values = group.index.to_numpy(int)
        rng = np.random.default_rng(stable_seed(dataset, i, "pair-fit"))
        perm = rng.permutation(values)
        take = min(quota, len(perm))
        selected.extend(perm[:take].tolist())
        leftovers.extend(perm[take:].tolist())
    need = cap - len(selected)
    if need > 0:
        rng = np.random.default_rng(stable_seed(dataset, "pair-fit-left"))
        selected.extend(rng.choice(np.asarray(leftovers), need, replace=False).tolist())
    return np.sort(np.asarray(selected, dtype=int))


def adaptive_pair_fit_size(
    metadata: pd.DataFrame,
    datasets: list[str],
) -> tuple[int, dict[str, int | bool | dict[str, int]]]:
    """Return a common fitting size that scales with the largest unit count.

    Hungarian matching requires the same number of fitting cells on every
    side. The requested size is therefore driven by the dataset with the most
    units, while the achievable size is bounded by the smallest dataset.
    """
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
    requested = max(PAIR_FIT_MINIMUM, PAIR_FIT_PER_UNIT * largest_unit_count)
    smallest_dataset = min(dataset_sizes.values())
    effective = min(requested, smallest_dataset)
    audit = {
        "pair_fit_minimum": PAIR_FIT_MINIMUM,
        "pair_fit_per_unit": PAIR_FIT_PER_UNIT,
        "largest_dataset_unit_count": largest_unit_count,
        "pair_fit_requested": requested,
        "pair_fit_effective": effective,
        "pair_fit_limited_by_smallest_dataset": effective < requested,
        "dataset_sizes": dataset_sizes,
        "dataset_unit_counts": unit_counts,
    }
    return effective, audit


def transition(values: np.ndarray, k: int = GRAPH_K) -> sparse.csr_matrix:
    n = min(k, len(values))
    distance, index = NearestNeighbors(n_neighbors=n, n_jobs=1).fit(values).kneighbors(values)
    sigma = np.maximum(distance[:, -1:], 1e-8)
    weight = np.exp(-0.5 * (distance / sigma) ** 2)
    weight /= np.maximum(weight.sum(axis=1, keepdims=True), 1e-12)
    row = np.repeat(np.arange(len(values)), n)
    return sparse.csr_matrix((weight.ravel(), (row, index.ravel())), shape=(len(values), len(values)))


def local_radius(values: np.ndarray, k: int = GRAPH_K) -> np.ndarray:
    n = min(k + 1, len(values))
    distance = NearestNeighbors(n_neighbors=n, n_jobs=1).fit(values).kneighbors(
        values, return_distance=True
    )[0]
    return np.maximum(distance[:, -1], 1e-8)


def build_dual_view_bundle(
    family: str,
    raw: np.ndarray,
    shared: np.ndarray,
    metadata: pd.DataFrame,
    datasets: list[str],
):
    # Hungarian hard matching needs the same number of fitting cells on every
    # side. The common size has a 4,000-cell floor and scales automatically
    # above 40 units, while remaining bounded by the smallest dataset.
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
        dataset: row_normalize(np.hstack([source_spaces[dataset], baseline_spaces[dataset]]))
        for dataset in datasets
    }
    copula_all = donor_copula(raw, metadata)
    copula_spaces = {
        dataset: graph_smooth(copula_all[rows]) for dataset, rows in samples.items()
    }
    maps, scores, similarities = {}, {}, {}
    for i, left in enumerate(datasets):
        for right in datasets[i+1:]:
            similarity = np.asarray(spaces[left] @ spaces[right].T, dtype=np.float32)
            similarities[(left, right)] = similarity
            similarities[(right, left)] = similarity.T
            row, col = linear_sum_assignment(similarity, maximize=True)
            mapping = np.empty(len(row), dtype=int); mapping[row] = col
            reverse = np.empty(len(col), dtype=int); reverse[col] = row
            assigned = np.clip((similarity[row, col].astype(float) + 1.0) / 2.0, 1e-6, 1.0)
            ordered = np.empty(len(row), dtype=float); ordered[row] = assigned
            reversed_score = np.empty(len(col), dtype=float); reversed_score[mapping] = ordered
            maps[(left, right)] = mapping; maps[(right, left)] = reverse
            scores[(left, right)] = ordered; scores[(right, left)] = reversed_score
    radius = {dataset: local_radius(spaces[dataset]) for dataset in datasets}
    transitions = {dataset: transition(spaces[dataset]) for dataset in datasets}
    return (
        samples, spaces, copula_spaces, maps, scores, similarities, radius,
        transitions, fit_size_audit,
    )


def cycle_quality(left: str, right: str, third: str, spaces, maps, radius) -> np.ndarray:
    start = np.arange(len(spaces[left]))
    returned_left = maps[(third, left)][maps[(right, third)][maps[(left, right)][start]]]
    q_left = np.linalg.norm(spaces[left][start] - spaces[left][returned_left], axis=1) / radius[left]
    start_right = maps[(left, right)][start]
    returned_right = maps[(third, right)][maps[(left, third)][maps[(right, left)][start_right]]]
    q_right = (
        np.linalg.norm(spaces[right][start_right] - spaces[right][returned_right], axis=1)
        / radius[right][start_right]
    )
    return np.exp(-0.25 * (q_left**2 + q_right**2))


def effective_count(log_probability: np.ndarray, axis: int) -> np.ndarray:
    probability = np.exp(log_probability)
    return np.exp(-np.sum(probability * log_probability, axis=axis))


def calibrated_temperature(similarity: np.ndarray, target: int = TARGET_EFFECTIVE_NEIGHBORS):
    n_left, n_right = similarity.shape
    left_ids = np.unique(np.linspace(0, n_left - 1, min(256, n_left)).astype(int))
    right_ids = np.unique(np.linspace(0, n_right - 1, min(256, n_right)).astype(int))
    row_sample = np.asarray(similarity[left_ids], dtype=np.float64)
    col_sample = np.asarray(similarity[:, right_ids], dtype=np.float64)

    def median_effective(tau: float) -> float:
        row_logits = row_sample / tau
        row_logp = row_logits - logsumexp(row_logits, axis=1, keepdims=True)
        col_logits = col_sample / tau
        col_logp = col_logits - logsumexp(col_logits, axis=0, keepdims=True)
        values = np.concatenate([effective_count(row_logp, 1), effective_count(col_logp, 0)])
        return float(np.median(values))

    low, high = 1e-5, 1.0
    while median_effective(high) < target and high < 64:
        high *= 2
    for _ in range(40):
        middle = np.sqrt(low * high)
        if median_effective(middle) < target:
            low = middle
        else:
            high = middle
    return high, median_effective(high)


def reciprocal_pivot(similarity: np.ndarray, tau: float, k: int = TARGET_EFFECTIVE_NEIGHBORS):
    logits = np.asarray(similarity, dtype=np.float64) / tau
    log_row = logits - logsumexp(logits, axis=1, keepdims=True)
    log_col = logits - logsumexp(logits, axis=0, keepdims=True)
    mutual = np.exp(0.5 * (log_row + log_col)).astype(np.float32)
    del logits, log_row, log_col
    n_left, n_right = mutual.shape
    keep = np.zeros((n_left, n_right), dtype=bool)
    row_k = min(k, n_right)
    row_top = np.argpartition(mutual, -row_k, axis=1)[:, -row_k:]
    keep[np.arange(n_left)[:, None], row_top] = True
    col_k = min(k, n_left)
    col_top = np.argpartition(mutual, -col_k, axis=0)[-col_k:, :]
    keep[col_top, np.arange(n_right)[None, :]] = True
    row, col = np.nonzero(keep)
    pivot = sparse.csr_matrix((mutual[row, col].astype(float), (row, col)), shape=mutual.shape)
    audit = {
        "temperature": float(tau),
        "soft_edges": int(pivot.nnz),
        "median_soft_row_degree": float(np.median(np.diff(pivot.indptr))),
        "median_soft_row_mass": float(np.median(np.asarray(pivot.sum(axis=1)).ravel())),
    }
    del mutual, keep
    return pivot, audit


def score_coupling(
    left: str,
    right: str,
    pivot: sparse.csr_matrix,
    spaces,
    copula_spaces,
    transitions,
    copula_scale: float,
) -> sparse.csr_matrix:
    coupling = (transitions[left] @ pivot @ transitions[right].T).tocsr()
    for row in range(coupling.shape[0]):
        lo, hi = coupling.indptr[row], coupling.indptr[row+1]
        candidates = coupling.indices[lo:hi]
        diffusion = coupling.data[lo:hi].copy()
        consensus = np.clip((spaces[right][candidates] @ spaces[left][row] + 1.0) / 2.0, 1e-6, 1.0)
        delta = copula_spaces[right][candidates] - copula_spaces[left][row]
        copula_support = np.exp(-0.5 * np.sum(delta * delta, axis=1) / max(copula_scale, 1e-8))
        coupling.data[lo:hi] = np.cbrt(np.maximum(diffusion * consensus * copula_support, 1e-18))
    coupling.eliminate_zeros()
    return coupling


@dataclass
class DirectionalTarget:
    query_dataset: str
    reference_dataset: str
    query_global: np.ndarray
    target: np.ndarray
    confidence: np.ndarray
    effective_references: np.ndarray
    role: np.ndarray


def deterministic_roles(family: str, query: str, reference: str, indices: np.ndarray) -> np.ndarray:
    buckets = np.asarray([
        stable_seed(family, query, reference, int(index), "full-coupling-four-way-role") % 25
        for index in indices
    ])
    roles = np.full(len(buckets), "curve_fit", dtype=object)
    roles[buckets < 5] = "test"
    roles[(buckets >= 5) & (buckets < 9)] = "strength_calibration"
    roles[(buckets >= 9) & (buckets < 13)] = "weight_calibration"
    return roles.astype(str)


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
    concentration = np.asarray(normalized.multiply(normalized).sum(axis=1)).ravel()
    confidence = np.asarray(score_matrix.multiply(score_matrix).sum(axis=1)).ravel() / np.maximum(row_sum, 1e-12)
    return DirectionalTarget(
        query, reference, np.asarray(query_global, int), target,
        np.asarray(confidence, float), 1.0 / np.maximum(concentration, 1e-12),
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
    samples, spaces, copula_spaces, maps, scores, similarities, radius, transitions, fit_size_audit = build_dual_view_bundle(
        family, raw, shared, metadata, datasets
    )
    directions, audit_rows = [], []
    for i, left in enumerate(datasets):
        for right in datasets[i+1:]:
            if len(datasets) == 3:
                # Preserve the original three-platform path exactly.
                third = next(x for x in datasets if x not in (left, right))
                cycle_weight = cycle_quality(left, right, third, spaces, maps, radius)
            else:
                # Natural two-platform reduction: there is no alternative path
                # through a third dataset, so the empty cycle product is the
                # multiplicative identity. Every other matching step is shared.
                third = ""
                cycle_weight = np.ones_like(scores[(left, right)])
            hard_weight = scores[(left, right)] * cycle_weight
            hard = sparse.csr_matrix(
                (hard_weight, (np.arange(len(hard_weight)), maps[(left, right)])),
                shape=(len(spaces[left]), len(spaces[right])),
            )
            tau, effective = calibrated_temperature(similarities[(left, right)])
            soft, soft_audit = reciprocal_pivot(similarities[(left, right)], tau)
            # V4.0: native evidence masses; deliberately no branch-normalization
            # and no fitted or hand-set hard/soft interpolation coefficient.
            pivot = (hard + soft).tocsr(); pivot.eliminate_zeros()
            prow, pcol = pivot.nonzero()
            delta = copula_spaces[right][pcol] - copula_spaces[left][prow]
            copula_scale = max(float(np.median(np.sum(delta * delta, axis=1))), 1e-8)
            coupling = score_coupling(left, right, pivot, spaces, copula_spaces, transitions, copula_scale)
            pair = (
                barycentric_direction(family, left, right, coupling, samples[left], samples[right], shared),
                barycentric_direction(family, right, left, coupling.T.tocsr(), samples[right], samples[left], shared),
            )
            directions.extend(pair)
            hard_mass = np.asarray(hard.sum(axis=1)).ravel(); hard_mass = hard_mass[hard_mass > 0]
            soft_mass = np.asarray(soft.sum(axis=1)).ravel(); soft_mass = soft_mass[soft_mass > 0]
            audit_rows.append({
                "family": family, "left": left, "right": right, "third": third,
                "pair_fit_minimum": fit_size_audit["pair_fit_minimum"],
                "pair_fit_per_unit": fit_size_audit["pair_fit_per_unit"],
                "largest_dataset_unit_count": fit_size_audit["largest_dataset_unit_count"],
                "pair_fit_requested": fit_size_audit["pair_fit_requested"],
                "pair_fit_effective": fit_size_audit["pair_fit_effective"],
                "pair_fit_limited_by_smallest_dataset": fit_size_audit["pair_fit_limited_by_smallest_dataset"],
                "hard_median_positive_row_mass": float(np.median(hard_mass)),
                "soft_median_positive_row_mass": float(np.median(soft_mass)),
                "implicit_soft_mass_fraction": float(np.median(soft_mass)) / (float(np.median(hard_mass)) + float(np.median(soft_mass))),
                "calibrated_median_effective_neighbors": effective,
                "copula_scale": copula_scale,
                "median_effective_references_left": float(np.median(pair[0].effective_references)),
                "median_effective_references_right": float(np.median(pair[1].effective_references)),
                **soft_audit,
            })
            del hard, soft, pivot, coupling
    return directions, pd.DataFrame(audit_rows)

def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    position = min(np.searchsorted(np.cumsum(weights), weights.sum()/2), len(values)-1)
    return float(values[position])


def score_weights(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    lo, hi = np.quantile(scores, [0.05, 0.95])
    return np.clip((scores-lo)/max(hi-lo, 1e-8), 0.05, 1.0)**2


def fit_profile(u: np.ndarray, y: np.ndarray, scores: np.ndarray):
    if len(u) < 40:
        return None
    weights = score_weights(scores)
    bins = np.minimum((u*20).astype(int), 19)
    ku, ky, kw = [], [], []
    for b in range(20):
        mask = bins == b
        if mask.sum() < 5:
            continue
        ku.append(float(np.average(u[mask], weights=weights[mask])))
        ky.append(weighted_median(y[mask], weights[mask]))
        kw.append(float(weights[mask].sum()))
    if len(ku) < 4:
        return None
    order = np.argsort(ku)
    fitted = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(
        np.asarray(ku)[order], np.asarray(ky)[order], sample_weight=np.asarray(kw)[order]
    )
    return np.interp(GRID, np.asarray(ku)[order], fitted)


def unit_lookup(metadata: pd.DataFrame):
    unit_col = metadata.unit_key.astype(str).to_numpy()
    cells = {unit: np.flatnonzero(unit_col == unit) for unit in pd.unique(unit_col)}
    datasets = {unit: str(metadata.iloc[index[0]].dataset) for unit, index in cells.items()}
    return unit_col, cells, datasets


def query_quantiles(shared, unit_cells, query_indices, marker):
    ordered = np.sort(shared[unit_cells, marker])
    return np.searchsorted(ordered, shared[query_indices, marker], side="right") / max(1, len(unit_cells))


def fit_profiles(shared, metadata, directions):
    profiles = {}
    _, cells_by_unit, dataset_by_unit = unit_lookup(metadata)
    for direction in directions:
        role = direction.role == "curve_fit"
        for unit, unit_cells in cells_by_unit.items():
            if dataset_by_unit[unit] != direction.query_dataset:
                continue
            select = role & np.isin(direction.query_global, unit_cells)
            if select.sum() < 40:
                continue
            q = direction.query_global[select]
            confidence = direction.confidence[select]
            for marker in range(shared.shape[1]):
                u = query_quantiles(shared, unit_cells, q, marker)
                profile = fit_profile(u, direction.target[select, marker], confidence)
                if profile is not None:
                    profiles[(unit, marker, direction.reference_dataset)] = profile
    return profiles


def proposed_target(shared, query_indices, unit_cells, marker, profile):
    u = query_quantiles(shared, unit_cells, query_indices, marker)
    raw_target = np.interp(u, GRID, profile)
    x = shared[query_indices, marker]
    return x + np.clip(raw_target - x, -MAX_DISPLACEMENT, MAX_DISPLACEMENT)


def optimal_alpha(x: np.ndarray, y: np.ndarray, target: np.ndarray, weights: np.ndarray):
    d = target - x
    denominator = np.sum(weights*d*d)
    if denominator <= 1e-10:
        return 0.0, 0.0
    alpha = float(np.clip(np.sum(weights*d*(y-x))/((1+STRENGTH_RIDGE)*denominator), 0, 1))
    base = np.average((x-y)**2, weights=weights)
    fitted = np.average((x+alpha*d-y)**2, weights=weights)
    gain = float(max(0, (base-fitted)/max(base, 1e-10)))
    if gain < 0.002:
        alpha *= gain/0.002
    return alpha, gain


def calibrate_strengths(shared, metadata, directions, datasets, profiles):
    strengths, audit_rows = {}, []
    _, cells_by_unit, dataset_by_unit = unit_lookup(metadata)
    for query in datasets:
        query_units = [u for u in cells_by_unit if dataset_by_unit[u] == query]
        for reference in datasets:
            if query == reference:
                continue
            direction = next(d for d in directions if d.query_dataset == query and d.reference_dataset == reference)
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
                    pieces.append((unit, shared[q, marker], direction.target[select, marker],
                                   proposed_target(shared, q, unit_cells, marker, profile),
                                   direction.confidence[select]))
                if not pieces:
                    continue
                x = np.concatenate([p[1] for p in pieces]); y = np.concatenate([p[2] for p in pieces])
                target = np.concatenate([p[3] for p in pieces]); confidence = np.concatenate([p[4] for p in pieces])
                weights = score_weights(confidence)
                dataset_alpha, dataset_gain = optimal_alpha(x, y, target, weights)
                offset = 0
                for unit in query_units:
                    available = [p for p in pieces if p[0] == unit]
                    if available:
                        piece = available[0]; n = len(piece[1]); sl = slice(offset, offset+n)
                        donor_alpha, donor_gain = optimal_alpha(x[sl], y[sl], target[sl], weights[sl]) if n >= 6 else (dataset_alpha, dataset_gain)
                        donor_weight = n/(n+30.0) if n >= 6 else 0.0
                        offset += n
                    else:
                        n = 0; donor_alpha, donor_gain, donor_weight = dataset_alpha, dataset_gain, 0.0
                    alpha = float(donor_weight*donor_alpha + (1-donor_weight)*dataset_alpha)
                    strengths[(unit, marker, reference)] = alpha
                    audit_rows.append({"query_dataset": query, "reference_dataset": reference, "unit_key": unit,
                                       "marker_index": marker, "calibration_queries": n, "dataset_alpha": dataset_alpha,
                                       "donor_alpha": donor_alpha, "adaptive_alpha": alpha,
                                       "dataset_gain": dataset_gain, "donor_gain": donor_gain})
    return strengths, pd.DataFrame(audit_rows)


def weighted_z(delta: np.ndarray, weights: np.ndarray):
    mean = float(np.average(delta, weights=weights))
    variance = float(np.average((delta-mean)**2, weights=weights))
    n_eff = float(weights.sum()**2/max(np.sum(weights**2), 1e-12))
    se = np.sqrt(variance/max(n_eff, 1.0) + 1e-12)
    return float(np.clip(mean/se, -6, 6)), mean, se, n_eff


def evidence_scores(shared, metadata, directions, datasets, profiles, strengths):
    evidence, audit_rows = {}, []
    _, cells_by_unit, dataset_by_unit = unit_lookup(metadata)
    for query in datasets:
        query_units = [u for u in cells_by_unit if dataset_by_unit[u] == query]
        for reference in datasets:
            if query == reference:
                continue
            direction = next(d for d in directions if d.query_dataset == query and d.reference_dataset == reference)
            role_mask = direction.role == "weight_calibration"
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
                    x = shared[q, marker]; y = direction.target[select, marker]
                    target = proposed_target(shared, q, unit_cells, marker, profile)
                    alpha = strengths.get((unit, marker, reference), 0.0)
                    delta = (x-y)**2 - (x + alpha*(target-x) - y)**2
                    pieces.append((unit, delta, direction.confidence[select]))
                if not pieces:
                    for unit in query_units:
                        evidence[(unit, marker, reference)] = -3.0
                    continue
                all_delta = np.concatenate([p[1] for p in pieces]); all_conf = np.concatenate([p[2] for p in pieces])
                all_weights = score_weights(all_conf)
                dataset_z, dataset_mean, _, dataset_neff = weighted_z(all_delta, all_weights)
                offset = 0
                for unit in query_units:
                    available = [p for p in pieces if p[0] == unit]
                    if available:
                        n = len(available[0][1]); sl = slice(offset, offset+n)
                        donor_z, _, _, _ = weighted_z(all_delta[sl], all_weights[sl]) if n >= 6 else (dataset_z, 0, 0, 0)
                        shrink = n/(n+30.0) if n >= 6 else 0.0
                        offset += n
                    else:
                        n = 0; donor_z, shrink = dataset_z, 0.0
                    z = float(shrink*donor_z + (1-shrink)*dataset_z)
                    evidence[(unit, marker, reference)] = z/2.0
                    audit_rows.append({"query_dataset": query, "reference_dataset": reference, "unit_key": unit,
                                       "marker_index": marker, "dataset_z": dataset_z, "donor_z": donor_z,
                                       "hierarchical_z": z, "reference_log_evidence": z/2.0,
                                       "dataset_mean_error_reduction": dataset_mean,
                                       "dataset_effective_queries": dataset_neff, "donor_queries": n,
                                       "donor_shrinkage": shrink})
    return evidence, pd.DataFrame(audit_rows)


def canonical_space(values: np.ndarray) -> np.ndarray:
    keep = np.nanstd(values, axis=0) > 1e-10
    z = StandardScaler().fit_transform(np.asarray(values[:, keep], dtype=np.float64))
    if z.shape[1] > 20:
        z = PCA(n_components=20, random_state=SEED).fit_transform(z)
    return np.asarray(z, dtype=np.float32)


def reciprocal_coverage(a_to_b: np.ndarray, b_to_a: np.ndarray):
    a = np.asarray([any(i in b_to_a[j] for j in row) for i, row in enumerate(a_to_b)])
    b = np.asarray([any(j in a_to_b[i] for i in row) for j, row in enumerate(b_to_a)])
    return float(a.mean()), float(b.mean())


def pair_compatibility(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    da = NearestNeighbors(n_neighbors=k+1).fit(a).kneighbors(a, return_distance=True)[0]
    db = NearestNeighbors(n_neighbors=k+1).fit(b).kneighbors(b, return_distance=True)[0]
    dab, iab = NearestNeighbors(n_neighbors=k).fit(b).kneighbors(a)
    dba, iba = NearestNeighbors(n_neighbors=k).fit(a).kneighbors(b)
    ra, rb = reciprocal_coverage(iab, iba)
    reciprocal = np.sqrt(ra*rb)
    ca = np.mean(dab[:, 0]/np.maximum(da[:, -1], 1e-8) <= 1.5)
    cb = np.mean(dba[:, 0]/np.maximum(db[:, -1], 1e-8) <= 1.5)
    return float(np.sqrt(reciprocal*np.sqrt(ca*cb)))


def compatibility_dict(family: str, baseline: np.ndarray, metadata: pd.DataFrame, datasets: list[str]):
    z = canonical_space(baseline)
    spaces = {dataset: z[unit_balanced_fit_indices(metadata, dataset, min(3500, len(metadata)))] for dataset in datasets}
    answer = {}
    for left, right in itertools.combinations(datasets, 2):
        score = pair_compatibility(spaces[left], spaces[right])
        answer[(left, right)] = answer[(right, left)] = score
    return answer


def automatic_profiles(baseline, metadata, datasets, profiles, strengths, evidence, compatibility):
    final, audit = {}, []
    unit_column = metadata.unit_key.astype(str)
    for unit in sorted(pd.unique(unit_column)):
        cells = np.flatnonzero(unit_column.eq(unit))
        dataset = str(metadata.iloc[cells[0]].dataset)
        for marker in range(baseline.shape[1]):
            identity = np.quantile(baseline[cells, marker], GRID)
            candidates, scores, refs, alphas, compatibilities = [], [], [], [], []
            for reference in datasets:
                if reference == dataset:
                    continue
                profile = profiles.get((unit, marker, reference))
                if profile is None:
                    continue
                alpha = strengths.get((unit, marker, reference), 0.0)
                candidates.append((1-alpha)*identity + alpha*profile)
                c = compatibility[(dataset, reference)]
                scores.append(evidence.get((unit, marker, reference), -3.0) + np.log(max(c, 1e-6)))
                refs.append(reference); alphas.append(alpha); compatibilities.append(c)
            if not candidates:
                continue
            scores = np.asarray(scores)
            relative = np.exp(scores-scores.max()); relative /= relative.sum()
            alignment_mass = float(np.sum(relative*np.asarray(compatibilities)**2))
            reference_weights = alignment_mass*relative
            target = (1-alignment_mass)*identity
            for weight, candidate in zip(reference_weights, candidates):
                target += weight*candidate
            final[(unit, marker)] = np.asarray(IsotonicRegression(increasing=True).fit_transform(GRID, target))
            row = {"dataset": dataset, "unit_key": unit, "marker_index": marker,
                   "identity_weight": 1-alignment_mass, "alignment_mass": alignment_mass}
            for ref, weight, alpha, c in zip(refs, reference_weights, alphas, compatibilities):
                row[f"weight__{slug(ref)}"] = weight; row[f"alpha__{slug(ref)}"] = alpha
                row[f"compatibility__{slug(ref)}"] = c
            audit.append(row)
    return final, pd.DataFrame(audit)


def apply_monotone_profiles(baseline: np.ndarray, metadata: pd.DataFrame, profiles: dict) -> np.ndarray:
    answer = baseline.copy()
    unit_column = metadata.unit_key.astype(str)
    for (unit, marker), profile in profiles.items():
        cells = np.flatnonzero(unit_column.eq(unit))
        u = rankdata(baseline[cells, marker], method="average") / (len(cells)+1)
        answer[cells, marker] = np.interp(u, GRID, profile)
    return answer


def fit_v40_v21(benchmark: Benchmark, baseline: np.ndarray):
    print(f"{benchmark.family}: build V4.0 coupling", flush=True)
    directions, coupling_audit = build_v40_targets(
        benchmark.family, benchmark.raw, baseline, benchmark.metadata, benchmark.datasets
    )
    print(f"{benchmark.family}: fit V2.1 monotone backend", flush=True)
    profiles = fit_profiles(baseline, benchmark.metadata, directions)
    strengths, strength_audit = calibrate_strengths(
        baseline, benchmark.metadata, directions, benchmark.datasets, profiles
    )
    evidence, evidence_audit = evidence_scores(
        baseline, benchmark.metadata, directions, benchmark.datasets, profiles, strengths
    )
    compatibility = compatibility_dict(
        benchmark.family, baseline, benchmark.metadata, benchmark.datasets
    )
    final_profiles, weight_audit = automatic_profiles(
        baseline, benchmark.metadata, benchmark.datasets, profiles, strengths, evidence, compatibility
    )
    candidate = apply_monotone_profiles(baseline, benchmark.metadata, final_profiles)
    return candidate, coupling_audit, strength_audit, evidence_audit, weight_audit

def _validated_metadata(
    raw: np.ndarray,
    metadata: pd.DataFrame,
    markers: list[str],
    datasets: list[str] | None,
) -> tuple[np.ndarray, pd.DataFrame, list[str], list[str]]:
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError("raw must be a finite two-dimensional matrix")
    markers = list(map(str, markers))
    if (
        raw.shape[1] != len(markers)
        or len(markers) != len(set(markers))
        or len(markers) < 2
    ):
        raise ValueError(
            "at least two marker names are required; names must be unique and match the raw columns"
        )
    required = {"dataset", "unit_key"}
    if not required.issubset(metadata.columns) or len(metadata) != len(raw):
        raise ValueError("metadata must align row-wise with raw and contain dataset and unit_key")
    if metadata[["dataset", "unit_key"]].isna().any().any():
        raise ValueError("dataset and unit_key cannot contain missing values")
    fit_metadata = metadata.loc[:, ["dataset", "unit_key"]].copy().reset_index(drop=True)
    fit_metadata["dataset"] = fit_metadata["dataset"].astype(str)
    fit_metadata["unit_key"] = fit_metadata["unit_key"].astype(str)
    fit_metadata["cell_order"] = fit_metadata.groupby("unit_key", sort=False).cumcount()
    fit_metadata["dataset_id"] = fit_metadata["dataset"]
    if int(fit_metadata.groupby("unit_key")["dataset"].nunique().max()) != 1:
        raise ValueError("each unit_key must belong to one and only one dataset")
    observed = list(pd.unique(fit_metadata["dataset"]))
    datasets = observed if datasets is None else list(map(str, datasets))
    if len(datasets) not in (2, 3) or len(set(datasets)) != len(datasets):
        raise ValueError("datasets must contain exactly two or three unique names")
    if set(datasets) != set(observed):
        raise ValueError("datasets must exactly match the dataset values in metadata")
    dataset_sizes = fit_metadata["dataset"].value_counts()
    if int(dataset_sizes.min()) < 11:
        raise ValueError("each dataset must contain at least 11 cells")
    unit_sizes = fit_metadata["unit_key"].value_counts()
    if int(unit_sizes.min()) < 2:
        raise ValueError("each unit_key must contain at least two cells")
    return raw, fit_metadata, markers, datasets


def fit_transform(
    raw: np.ndarray,
    metadata: pd.DataFrame,
    markers: list[str],
    *,
    datasets: list[str] | None = None,
    family: str = "ramp_run",
) -> RampResult:
    """Run the unified two- or three-dataset RAMP pipeline without labels."""
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
        family=str(family), raw=raw, metadata=fit_metadata, markers=markers,
        datasets=datasets, reference=datasets[0],
    )
    aligned, coupling, strength, evidence, weights = fit_v40_v21(
        benchmark, baseline
    )
    inversions = rank_inversion_count(raw, aligned, fit_metadata)
    if inversions:
        raise AssertionError(f"RAMP produced {inversions} strict within-unit rank inversions")
    return RampResult(
        baseline=baseline,
        aligned=aligned,
        coupling_audit=coupling,
        strength_audit=strength,
        evidence_audit=evidence,
        reference_weight_audit=weights,
        strict_rank_inversions=inversions,
    )


__all__ = [
    "Benchmark", "RampResult", "fit_transform", "kmeanslow_global",
    "fit_v40_v21", "rank_inversion_count", "SEED", "PAIR_FIT_CAP",
    "PAIR_FIT_MINIMUM", "PAIR_FIT_PER_UNIT", "adaptive_pair_fit_size",
    "GRAPH_K", "__version__",
]
