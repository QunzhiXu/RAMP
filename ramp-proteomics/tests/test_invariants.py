from __future__ import annotations

import numpy as np

from ramp_proteomics import fit_transform
from ramp_proteomics import _algorithm

from _synthetic import synthetic_input


def test_three_dataset_rank_and_displacement_invariants() -> None:
    raw, metadata, markers = synthetic_input(datasets=3, seed=201)
    datasets = list(dict.fromkeys(metadata["dataset"]))
    result = fit_transform(
        raw,
        metadata,
        markers,
        datasets=datasets,
        family="invariant-three-dataset",
    )
    assert result.strict_rank_inversions == 0
    displacement = np.abs(result.aligned - result.baseline)
    assert np.any(displacement > 1e-12)
    assert float(displacement.max()) <= _algorithm.MAX_DISPLACEMENT + 1e-10
    audit = result.algorithm_audit.set_index("audit_key")["audit_value"]
    assert int(audit["cell_marker_values_over_maximum"]) == 0
    assert audit["percentile_definition"] == _algorithm.PERCENTILE_DEFINITION
    assert audit["algorithm_version"] == "2.3.0"


def test_graph_k_means_nonself_neighbors() -> None:
    values = np.arange(40, dtype=float)[:, None]
    _, neighbors = _algorithm._nonself_knn(values, 15)
    assert neighbors.shape == (40, 15)
    assert not np.any(neighbors == np.arange(40)[:, None])


def test_bounded_mapping_enforces_maximum_displacement() -> None:
    values = np.asarray([1.0, 1.0, 2.0, 4.0, 9.0])
    percentiles = _algorithm.within_unit_percentiles(values)
    identity = _algorithm.empirical_profile(values)
    mapped = _algorithm.bounded_profile_mapping(
        values, percentiles, identity + 100.0
    )
    assert np.max(np.abs(mapped - values)) <= _algorithm.MAX_DISPLACEMENT + 1e-12
