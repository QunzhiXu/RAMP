"""Public interface for RAMP 2.3.0."""

from __future__ import annotations

import warnings

import numpy as np

from . import _algorithm
from ._algorithm import RampResult, SEED
from ._algorithm import fit_transform as _fit_transform
from ._validation import validate_identity_values, validate_raw_matrix

__version__ = "2.3.0"
ALGORITHM_VERSION = "2.3.0"


def fit_transform(
    raw,
    metadata,
    markers,
    *,
    datasets=None,
    family="ramp_run",
) -> RampResult:
    """Run RAMP with a clear error for a non-informative shared panel.

    This wrapper does not alter valid numerical inputs or outputs; fitting
    remains in the frozen RAMP 2.3.0 implementation.
    """
    values = validate_raw_matrix(raw)
    if hasattr(metadata, "columns"):
        for column in ("dataset", "unit_key"):
            if column in metadata.columns:
                validate_identity_values(metadata[column], column)
    if datasets is not None:
        dataset_names = list(map(str, datasets))
    elif hasattr(metadata, "columns") and "dataset" in metadata.columns:
        dataset_names = list(dict.fromkeys(metadata["dataset"].astype(str)))
    else:
        dataset_names = []
    dataset_slugs = [_algorithm.slug(name) for name in dataset_names]
    if len(set(dataset_names)) == len(dataset_names) and (
        any(not value for value in dataset_slugs)
        or len(set(dataset_slugs)) != len(dataset_slugs)
    ):
        raise ValueError(
            "dataset names must remain unique after audit-column normalization; "
            "avoid names that differ only by punctuation or spaces"
        )
    if not np.any(values != values[0]):
        raise ValueError(
            "RAMP requires at least one shared marker with non-zero variation"
        )
    try:
        result = _fit_transform(
            values,
            metadata,
            markers,
            datasets=datasets,
            family=family,
        )
        profile_audit = result.profile_audit
        fitted_profile = (
            not profile_audit.empty
            and "curve_profile_fitted" in profile_audit.columns
            and bool(
                profile_audit["curve_profile_fitted"]
                .fillna(False)
                .astype(bool)
                .any()
            )
        )
        if not fitted_profile:
            warnings.warn(
                "RAMP learned no correction profiles; the aligned matrix is the "
                "KMEANSLOW baseline. Increase cells per fitting unit or inspect "
                "shared-marker information.",
                RuntimeWarning,
                stacklevel=2,
            )
        return result
    except ValueError as error:
        if "0 feature(s)" in str(error):
            raise ValueError(
                "RAMP requires at least one shared marker that remains variable "
                "after within-unit KMEANSLOW preprocessing"
            ) from error
        raise

__all__ = [
    "ALGORITHM_VERSION",
    "RampResult",
    "SEED",
    "__version__",
    "fit_transform",
]
