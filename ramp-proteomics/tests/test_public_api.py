from __future__ import annotations

import numpy as np

from ramp_proteomics import ALGORITHM_VERSION, __version__, fit_transform

from _synthetic import synthetic_input


def test_version_and_two_dataset_result() -> None:
    raw, metadata, markers = synthetic_input(datasets=2, seed=101)
    datasets = list(dict.fromkeys(metadata["dataset"]))
    result = fit_transform(
        raw,
        metadata,
        markers,
        datasets=datasets,
        family="public-api-two-dataset",
    )
    assert __version__ == ALGORITHM_VERSION == "2.3.0"
    assert result.baseline.shape == raw.shape
    assert result.aligned.shape == raw.shape
    assert np.isfinite(result.baseline).all()
    assert np.isfinite(result.aligned).all()
    assert result.strict_rank_inversions == 0
    assert np.any(np.abs(result.aligned - result.baseline) > 1e-12)


def test_extra_donor_and_annotation_columns_do_not_change_fit() -> None:
    raw, metadata, markers = synthetic_input(datasets=2, seed=102)
    datasets = list(dict.fromkeys(metadata["dataset"]))
    complete = fit_transform(
        raw,
        metadata,
        markers,
        datasets=datasets,
        family="metadata-isolation",
    )
    fit_only = fit_transform(
        raw,
        metadata[["dataset", "unit_key"]],
        markers,
        datasets=datasets,
        family="metadata-isolation",
    )
    np.testing.assert_array_equal(complete.baseline, fit_only.baseline)
    np.testing.assert_array_equal(complete.aligned, fit_only.aligned)


def test_repeat_run_is_deterministic() -> None:
    raw, metadata, markers = synthetic_input(datasets=2, seed=103)
    datasets = list(dict.fromkeys(metadata["dataset"]))
    first = fit_transform(
        raw, metadata, markers, datasets=datasets, family="deterministic-run"
    )
    second = fit_transform(
        raw, metadata, markers, datasets=datasets, family="deterministic-run"
    )
    np.testing.assert_array_equal(first.baseline, second.baseline)
    np.testing.assert_array_equal(first.aligned, second.aligned)
