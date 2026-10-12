from __future__ import annotations

import importlib.metadata
import os
import sys

import numpy as np
import pytest

from ramp_proteomics import fit_transform

from _synthetic import synthetic_input


FROZEN_NUMERICAL_VERSIONS = {
    "numpy": "2.4.0",
    "pandas": "2.3.3",
    "scipy": "1.16.3",
    "scikit-learn": "1.8.0",
    "joblib": "1.5.3",
    "threadpoolctl": "3.6.0",
}
RUNNING_FROZEN_LOCK = all(
    importlib.metadata.version(package) == version
    for package, version in FROZEN_NUMERICAL_VERSIONS.items()
)

if os.environ.get("RAMP_REQUIRE_FROZEN_LOCK") == "1" and (
    not RUNNING_FROZEN_LOCK or sys.version_info[:3] != (3, 14, 1)
):
    raise RuntimeError(
        "frozen regression requires Python 3.14.1 and "
        "reproducibility/requirements-paper.txt; refusing to skip golden tests"
    )


pytestmark = pytest.mark.skipif(
    not RUNNING_FROZEN_LOCK,
    reason="exact numerical fingerprint requires the audited dependency lock",
)


@pytest.mark.parametrize(
    ("dataset_count", "seed", "family", "expected"),
    [
        (
            2,
            909,
            "golden-active-two",
            {
                "baseline_sum": 2534.4267294258843,
                "aligned_sum": 2525.8455454566524,
                "aligned_std": 0.776823210939398,
                "absolute_delta_sum": 773.6405536310847,
                "maximum_delta": 0.7360625731737009,
                "corrected_values": 2424,
                "selected": [
                    0.6666085636088435,
                    1.6720907860118788,
                    1.1721995790634139,
                    1.1144246166426803,
                    0.9558396987631782,
                    0.16617239228701725,
                ],
                "audit_shapes": [(1, 26), (16, 12), (16, 15), (16, 21), (16, 13)],
            },
        ),
        (
            3,
            910,
            "golden-active-three",
            {
                "baseline_sum": 3750.3344598685635,
                "aligned_sum": 3755.0307041051306,
                "aligned_std": 0.763055612969259,
                "absolute_delta_sum": 1037.067509427173,
                "maximum_delta": 0.7102992605754244,
                "corrected_values": 3636,
                "selected": [
                    0.07854055062576337,
                    0.5426345446468905,
                    1.466959196623154,
                    0.5215144968788294,
                    0.8076140027156324,
                    1.3523301053025771,
                ],
                "audit_shapes": [(3, 26), (48, 12), (48, 15), (24, 24), (48, 13)],
            },
        ),
    ],
)
def test_active_alignment_matches_frozen_numerical_fingerprint(
    dataset_count: int,
    seed: int,
    family: str,
    expected: dict[str, object],
) -> None:
    raw, metadata, markers = synthetic_input(
        datasets=dataset_count, cells_per_unit=150, seed=seed
    )
    datasets = [f"D{index + 1}" for index in range(dataset_count)]
    result = fit_transform(
        raw, metadata, markers, datasets=datasets, family=family
    )
    delta = result.aligned - result.baseline
    observed = np.asarray(
        [
            result.baseline.sum(),
            result.aligned.sum(),
            result.aligned.std(),
            np.abs(delta).sum(),
            np.abs(delta).max(),
        ]
    )
    target = np.asarray(
        [
            expected["baseline_sum"],
            expected["aligned_sum"],
            expected["aligned_std"],
            expected["absolute_delta_sum"],
            expected["maximum_delta"],
        ]
    )
    np.testing.assert_allclose(observed, target, rtol=1e-7, atol=1e-8)
    assert int(np.sum(np.abs(delta) > 1e-12)) == expected["corrected_values"]
    selected = result.aligned[
        [0, 17, 149, 150, 301, -1], [0, 1, 2, 3, 0, 3]
    ]
    np.testing.assert_allclose(
        selected, np.asarray(expected["selected"]), rtol=1e-7, atol=1e-8
    )
    audit_shapes = [
        result.coupling_audit.shape,
        result.strength_audit.shape,
        result.evidence_audit.shape,
        result.reference_weight_audit.shape,
        result.profile_audit.shape,
    ]
    assert audit_shapes == expected["audit_shapes"]
    assert result.strict_rank_inversions == 0
