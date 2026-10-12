from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ramp_proteomics import fit_transform as packaged_fit_transform

from _synthetic import synthetic_input


def test_packaged_algorithm_matches_manuscript_source_exactly() -> None:
    project_root = Path(__file__).resolve().parents[2]
    algorithm_root = project_root / "final_xu" / "algorithm"
    if not (algorithm_root / "ramp" / "ramp_ver2.py").exists():
        pytest.skip("manuscript source tree is not present in this checkout")

    sys.path.insert(0, str(algorithm_root))
    try:
        manuscript = importlib.import_module("ramp")
        raw, metadata, markers = synthetic_input(datasets=3, seed=501)
        datasets = list(dict.fromkeys(metadata["dataset"]))
        expected = manuscript.fit_transform(
            raw,
            metadata,
            markers,
            datasets=datasets,
            family="source-equivalence",
        )
        observed = packaged_fit_transform(
            raw,
            metadata,
            markers,
            datasets=datasets,
            family="source-equivalence",
        )
    finally:
        sys.path.remove(str(algorithm_root))

    np.testing.assert_array_equal(observed.baseline, expected.baseline)
    np.testing.assert_array_equal(observed.aligned, expected.aligned)
    assert observed.strict_rank_inversions == expected.strict_rank_inversions == 0
    pd.testing.assert_frame_equal(observed.coupling_audit, expected.coupling_audit)
    pd.testing.assert_frame_equal(observed.strength_audit, expected.strength_audit)
    pd.testing.assert_frame_equal(observed.evidence_audit, expected.evidence_audit)
    pd.testing.assert_frame_equal(
        observed.reference_weight_audit, expected.reference_weight_audit
    )
    pd.testing.assert_frame_equal(observed.profile_audit, expected.profile_audit)
    pd.testing.assert_frame_equal(observed.algorithm_audit, expected.algorithm_audit)

