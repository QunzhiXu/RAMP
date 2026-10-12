"""Shared boundary validation for the Python and file interfaces."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_raw_matrix(raw) -> np.ndarray:
    """Use the canonical float64 input conversion before checking variation."""
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("raw must be a finite two-dimensional matrix")
    if not values.shape[0] or not values.shape[1]:
        raise ValueError("raw must be a non-empty two-dimensional matrix")
    return values


def validate_identity_values(series: pd.Series, column: str) -> pd.Series:
    """Reject missing, blank, or whitespace-padded identity values."""
    if series.isna().any():
        raise ValueError(f"{column} cannot contain missing values")
    values = series.astype(str)
    if values.str.strip().eq("").any():
        raise ValueError(f"{column} cannot contain blank values")
    if values.ne(values.str.strip()).any():
        raise ValueError(f"{column} cannot contain leading or trailing whitespace")
    return values
