#!/usr/bin/env python3
"""Call-sequence, label-isolation, lock, and RNG smoke test using a fake Fusor."""

from __future__ import annotations

from pathlib import Path
import copy
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maxfuse_repro.pairwise import (
    BatchingConfig,
    GraphConfig,
    InitialPivotConfig,
    PairwiseConfig,
    PropagationConfig,
    RefineConfig,
    UpstreamAudit,
    fit_pairwise,
)
from maxfuse_repro.types import ViewData
from maxfuse_repro.types import Matching


CALLS: list[tuple[str, dict[str, object]]] = []


class FakeFusor:
    def __init__(self, **kwargs: object) -> None:
        CALLS.append(("constructor", dict(kwargs)))
        self.active1 = np.asarray(kwargs["active_arr1"])
        self.active2 = np.asarray(kwargs["active_arr2"])

    def split_into_batches(self, **kwargs: object) -> None:
        CALLS.append(("split_into_batches", dict(kwargs)))
        np.random.random()

    def construct_graphs(self, **kwargs: object) -> None:
        CALLS.append(("construct_graphs", dict(kwargs)))

    def find_initial_pivots(self, **kwargs: object) -> None:
        CALLS.append(("find_initial_pivots", dict(kwargs)))

    def refine_pivots(self, **kwargs: object) -> None:
        CALLS.append(("refine_pivots", dict(kwargs)))

    def filter_bad_matches(self, **kwargs: object) -> None:
        CALLS.append(("filter_bad_matches", dict(kwargs)))

    def get_matching(self, **kwargs: object) -> list[list[float] | list[int]]:
        CALLS.append(("get_matching", dict(kwargs)))
        return [[0, 1, 2], [2, 1, 0], [0.9, 0.8, 0.7]]

    def get_embedding(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        CALLS.append(("get_embedding", dict(kwargs)))
        return self.active1[:, :2], self.active2[:, :2]

    def propagate(self, **kwargs: object) -> None:
        CALLS.append(("propagate", dict(kwargs)))


class FakeMatchUtils:
    @staticmethod
    def address_matching_redundancy(
        matching: list[list[float] | list[int]], order: tuple[int, int]
    ) -> list[list[float] | list[int]]:
        return matching


def state_equal(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def main() -> None:
    rng = np.random.default_rng(91)
    shared_names = tuple(f"shared_{index}" for index in range(4))
    first = ViewData(
        rng.normal(size=(12, 4)), rng.normal(size=(12, 6)),
        shared_feature_names=shared_names,
    )
    second = ViewData(
        rng.normal(size=(13, 4)), rng.normal(size=(13, 5)),
        shared_feature_names=shared_names,
    )
    config = PairwiseConfig(
        batching=BatchingConfig(max_outward_size=100, metacell_size=1, seed=7),
        graph=GraphConfig(n_neighbors1=3, n_neighbors2=3, leiden_seed=7),
        initial=InitialPivotConfig(wt1=0.7, wt2=0.7),
        refine=RefineConfig(wt1=0.7, wt2=0.7, cca_components=2),
        pivot_filter_prop=0.3,
        propagation=PropagationConfig(enabled=True),
        numpy_seed=7,
    )
    np.random.seed(12345)
    before = np.random.get_state()
    with patch(
        "maxfuse_repro.pairwise.verify_upstream_lock",
        return_value=UpstreamAudit(
            "0.0.2", {"model.py": "fake"}, "fake-commit", "fake-lock", True
        ),
    ), patch(
        "maxfuse_repro.pairwise._load_upstream",
        return_value=(FakeFusor, FakeMatchUtils),
    ):
        result = fit_pairwise(first, second, config)
    after = np.random.get_state()
    if not state_equal(before, after):
        raise AssertionError("wrapper changed the caller's global NumPy RNG state")
    names = [name for name, _ in CALLS]
    expected = [
        "constructor",
        "split_into_batches",
        "construct_graphs",
        "find_initial_pivots",
        "refine_pivots",
        "filter_bad_matches",
        "get_matching",
        "get_embedding",
        "propagate",
        "filter_bad_matches",
        "get_matching",
    ]
    if names != expected:
        raise AssertionError(f"unexpected public API call sequence: {names}")
    constructor = CALLS[0][1]
    if constructor["labels1"] is not None or constructor["labels2"] is not None:
        raise AssertionError("cell labels leaked into MaxFuse fitting")
    split = CALLS[1][1]
    if set(split) != {
        "max_outward_size", "matching_ratio", "metacell_size", "method",
        "batching_scheme", "prebatching_smoothing", "shared_wt1", "shared_wt2",
        "active_wt1", "active_wt2", "shared_svd_components1",
        "shared_svd_components2", "active_svd_components1",
        "active_svd_components2", "randomized_svd", "svd_runs", "seed", "verbose",
    }:
        raise AssertionError("split_into_batches arguments are not fully explicit")
    official = PairwiseConfig.citeseq_pbmc_tutorial(deterministic_seed=19)
    if not (
        official.batching.method == "random"
        and official.batching.max_outward_size == 5000
        and official.graph.svd_components1 == 30
        and official.graph.resolution1 == 2
        and official.initial.svd_components1 == 25
        and official.refine.cca_components == 20
        and official.refine.n_iters == 3
        and official.pivot_filter_prop == 0.3
        and official.propagation.full_matching_order == (2, 1)
    ):
        raise AssertionError("official tutorial profile drifted")
    if PairwiseConfig.from_dict(official.to_dict()) != official:
        raise AssertionError("complete PairwiseConfig serialization did not roundtrip")
    if result.full_matching is None:
        raise AssertionError("enabled propagation did not return a full matching")
    try:
        Matching.from_upstream([[0.9, 1.1, 2.8], [0, 1, 2], [0.1, 0.2, 0.3]])
    except TypeError:
        pass
    else:
        raise AssertionError("floating-point upstream matching indices were truncated")
    invalid_configs = []
    invalid = copy.deepcopy(official.to_dict())
    invalid["numppy_seed"] = 7
    invalid_configs.append(invalid)
    invalid = copy.deepcopy(official.to_dict())
    invalid["batching"]["matching_ratio"] = float("nan")
    invalid_configs.append(invalid)
    invalid = copy.deepcopy(official.to_dict())
    invalid["propagation"]["enabled"] = "false"
    invalid_configs.append(invalid)
    invalid = copy.deepcopy(official.to_dict())
    del invalid["graph"]["metric"]
    invalid_configs.append(invalid)
    invalid = copy.deepcopy(official.to_dict())
    invalid["schema_version"] = "unsupported"
    invalid_configs.append(invalid)
    invalid_configs.append({})
    for invalid in invalid_configs:
        try:
            PairwiseConfig.from_dict(invalid)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"invalid configuration was accepted: {invalid}")
    mismatched_second = ViewData(
        second.shared,
        second.active,
        shared_feature_names=tuple(reversed(shared_names)),
    )
    try:
        fit_pairwise(first, mismatched_second, config)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched shared feature order was accepted")
    print("MAXFUSE_REPRO PAIRWISE_ADAPTER_SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
