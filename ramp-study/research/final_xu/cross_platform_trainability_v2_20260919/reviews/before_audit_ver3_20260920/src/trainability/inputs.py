"""Label-free representation boundary and mask-before-processing controls.

Historical provider-normalized matrices are not implicitly certified as native.
The full raw matrix may enter only the reviewed builder; adapters see masked X.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.stats import rankdata

from .core import ProtocolError, require, unique_ids

METHODS = {
    "unaligned": "implemented",
    "ramp": "adapter_pending",
    "harmony_direct": "adapter_pending",
    "cycombine": "adapter_pending",
    "cytonorm": "adapter_pending",
    "fastmnn": "adapter_pending",
    "maxfuse": "adapter_pending",
    "clipped_zscore": "implemented",
    "minmax": "implemented",
    "percentile": "implemented",
}


@dataclass(frozen=True)
class LabelFreeInput:
    x: np.ndarray
    cell_ids: tuple[str, ...]
    datasets: tuple[str, ...]
    units: tuple[str, ...]
    markers: tuple[str, ...]
    masked_markers: tuple[str, ...] = ()

    def __post_init__(self):
        ids = unique_ids(self.cell_ids)
        require(self.x.ndim == 2 and self.x.shape == (len(ids), len(self.markers)),
                "Input row/column identity mismatch")
        require(len(self.datasets) == len(ids) == len(self.units), "Metadata length mismatch")
        require(len(set(self.markers)) == len(self.markers), "Duplicate physical marker columns")
        require(bool(ids) and bool(self.markers), "Empty input")
        require(np.isfinite(self.x).all(), "Nonfinite representation input")
        unit_dataset = {}
        for unit, ds in zip(self.units, self.datasets):
            require(bool(unit) and bool(ds), "Missing dataset/unit")
            require(unit_dataset.setdefault(unit, ds) == ds,
                    "unit IDs collide across datasets; namespace them first")
        x = np.array(self.x, dtype=float, copy=True)
        x.flags.writeable = False
        object.__setattr__(self, "x", x)


def build_assay_input(raw: np.ndarray, *, cell_ids, datasets, units, markers,
                      canonical_marker: Mapping[str, str], masked_marker: str | None,
                      contracts: Mapping[str, dict]) -> LabelFreeInput:
    markers, datasets = tuple(markers), tuple(datasets)
    require(raw.ndim == 2 and raw.shape == (len(cell_ids), len(markers)), "Native shape mismatch")
    require(set(markers) <= canonical_marker.keys(), "Unreviewed marker aliases")
    require(set(datasets) <= contracts.keys(), "Missing dataset input contracts")
    require(all(canonical_marker[m] for m in markers), "Empty canonical marker name")
    if masked_marker is not None:
        require(masked_marker in {canonical_marker[m] for m in markers}, "Unknown masked marker")
    keep = [i for i, m in enumerate(markers) if canonical_marker[m] != masked_marker]
    require(bool(keep), "No features after masking")
    # Critically, the removed values are never used for finite checks, CLR or scaling.
    x = np.array(raw[:, keep], dtype=float, copy=True)
    require(np.isfinite(x).all(), "Nonfinite allowed native values")
    for ds in sorted(set(datasets)):
        c = contracts[ds]
        require(c.get("review_status") == "verified_no_target_dependency",
                f"Native input dependencies unresolved for {ds}")
        require(bool(c.get("evidence_path")) and bool(c.get("input_contract_version")),
                f"Contract lacks provenance for {ds}")
        rows = np.asarray(datasets) == ds
        op = c.get("operator")
        if op == "identity":
            pass  # Signed provider values are allowed; never silently zero-clamp.
        elif op == "asinh":
            cf = float(c.get("cofactor", 0))
            require(np.isfinite(cf) and cf > 0, "Invalid asinh cofactor")
            x[rows] = np.arcsinh(x[rows] / cf)
        elif op == "log1p":
            require((x[rows] >= 0).all(), "log1p recipe requires nonnegative counts")
            x[rows] = np.log1p(x[rows])
        elif op == "clr_shared":
            require((x[rows] >= 0).all(), "CLR requires native nonnegative counts")
            denom = np.exp(np.log1p(x[rows]).mean(axis=1, keepdims=True))
            x[rows] = np.log1p(x[rows] / denom)
        else:
            raise ProtocolError(f"Unimplemented assay operator: {op}")
    return LabelFreeInput(x, tuple(cell_ids), datasets, tuple(units),
                          tuple(markers[i] for i in keep),
                          () if masked_marker is None else (masked_marker,))


def simple_representation(data: LabelFreeInput, method: str) -> np.ndarray:
    require(method in METHODS, f"Unknown method {method}")
    require(METHODS[method] == "implemented", f"{method}: reviewed adapter not implemented")
    x = data.x.copy()
    datasets, units = np.asarray(data.datasets), np.asarray(data.units)
    if method == "clipped_zscore":
        for unit in np.unique(units):
            rows = units == unit
            x[rows] = np.minimum(x[rows], np.quantile(x[rows], .99, axis=0))
        for ds in np.unique(datasets):
            rows = datasets == ds
            mean, sd = x[rows].mean(axis=0), x[rows].std(axis=0, ddof=0)
            x[rows] = (x[rows] - mean) / np.where(sd > 0, sd, 1)
    elif method == "minmax":
        for ds in np.unique(datasets):
            rows = datasets == ds
            lo, hi = x[rows].min(axis=0), x[rows].max(axis=0)
            x[rows] = 10*(x[rows] - lo) / np.where(hi > lo, hi-lo, 1)
    elif method == "percentile":
        for unit in np.unique(units):
            rows = units == unit
            x[rows] = (rankdata(x[rows], axis=0, method="average")-.5)/rows.sum()
    require(np.isfinite(x).all(), "Method produced nonfinite values")
    return x


def assert_target_invariance(builder, raw: np.ndarray, target_columns,
                              *, seed: int = 0, atol: float = 0) -> None:
    """Builder must encompass all preprocessing AND adapter fitting under a fixed seed."""
    require(bool(target_columns), "No target columns provided to perturbation test")
    original = np.asarray(builder(raw))
    changed = raw.copy()
    rng = np.random.default_rng(seed)
    changed[:, target_columns] = rng.normal(10000, 1000, (len(raw), len(target_columns)))
    other = np.asarray(builder(changed))
    require(original.shape == other.shape and np.isfinite(original).all()
            and np.isfinite(other).all() and np.allclose(original, other, rtol=0, atol=atol),
            "Hidden target affects allowed inputs/representation")

