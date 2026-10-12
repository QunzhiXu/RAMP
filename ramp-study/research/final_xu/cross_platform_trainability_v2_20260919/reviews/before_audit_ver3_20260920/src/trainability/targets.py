"""Source-only empirical calibration. Target ranking lives in scoring.py."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .core import require, unique_ids


@dataclass(frozen=True)
class SourceCalibration:
    values_by_unit: dict[str, np.ndarray]
    calibration_ids: tuple[str, ...]

    @classmethod
    def fit(cls, ids, values, units, allowed_calibration_ids):
        ids = unique_ids(ids)
        require(set(ids) == set(allowed_calibration_ids), "Calibration differs from supervision ledger")
        values = np.asarray(values, dtype=float)
        require(values.shape == (len(ids),) and len(units) == len(ids), "Calibration shape mismatch")
        require(np.isfinite(values).all() and len(ids) > 0, "Invalid calibration answers")
        units = np.asarray(units, dtype=str)
        by_unit = {u: np.sort(values[units == u]) for u in np.unique(units)}
        return cls(by_unit, ids)

    def transform(self, ids, values, units) -> np.ndarray:
        ids = unique_ids(ids)
        require(not set(ids).intersection(self.calibration_ids), "Calibration reused as fit/validation")
        values = np.asarray(values, dtype=float)
        units = np.asarray(units, dtype=str)
        require(values.shape == units.shape == (len(ids),), "Answer shape mismatch")
        require(np.isfinite(values).all(), "Nonfinite source answer")
        require(set(units) <= self.values_by_unit.keys(), "Missing calibration: cannot borrow another unit")
        out = np.empty(len(ids))
        for u in np.unique(units):
            mask = units == u
            cal = self.values_by_unit[u]
            left = np.searchsorted(cal, values[mask], side="left")
            right = np.searchsorted(cal, values[mask], side="right")
            out[mask] = (left+right)/(2*len(cal))
        return out


def constant_mae_baseline(source_fit_percentiles) -> float:
    values = np.asarray(source_fit_percentiles, dtype=float)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), "Invalid source targets")
    require(((values >= 0) & (values <= 1)).all(), "Not source percentiles")
    return float(np.median(values))
