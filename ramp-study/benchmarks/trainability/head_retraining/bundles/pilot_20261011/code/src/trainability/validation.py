"""ID-bound SOURCE validation; no arbitrary scoring closures in head selection."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np

from .core import digest, require, unique_ids
from .scoring import classification_score, regression_score


def _ordered(mapping: Mapping, ids, name):
    require(isinstance(mapping, Mapping) and set(mapping) == set(ids),
            f"{name} keys differ from the declared validation cells")
    return [mapping[cell] for cell in ids]


@dataclass(frozen=True)
class SourceValidation:
    cell_ids: tuple[str, ...]
    x: np.ndarray
    targets: np.ndarray
    patients: tuple[str, ...]
    task: str
    classes: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    measured: np.ndarray | None = None
    identity_level: str = "patient"
    identity_reviewed: bool = False
    minimum_cells: int = 30

    @classmethod
    def from_cell_maps(cls, *, cell_ids, x, targets_by_cell, patients_by_cell, task,
                       identity_level, identity_reviewed, classes=(), units_by_cell=None,
                       measured_by_cell=None, minimum_cells=30):
        ids = unique_ids(cell_ids)
        return cls(ids, np.asarray(x), np.asarray(_ordered(targets_by_cell, ids, "Target")),
                   tuple(_ordered(patients_by_cell, ids, "Patient")), task, tuple(classes),
                   () if units_by_cell is None else tuple(_ordered(units_by_cell, ids, "Unit")),
                   None if measured_by_cell is None else np.asarray(_ordered(measured_by_cell, ids, "Measured marker")),
                   identity_level, identity_reviewed, minimum_cells)

    def __post_init__(self):
        ids = unique_ids(self.cell_ids)
        require(bool(ids), "Empty source validation")
        require(self.task in {"classification", "regression"}, "Unknown validation task")
        require(self.identity_level in {"patient", "unit_descriptive"}, "Unknown identity level")
        require(self.identity_level != "patient" or self.identity_reviewed,
                "Patient-level validation requires reviewed identity mapping")
        x = np.array(self.x, dtype=float, copy=True)
        y = np.array(self.targets, copy=True)
        require(x.ndim == 2 and len(x) == len(ids) and x.shape[1] > 0 and np.isfinite(x).all(),
                "Invalid source validation features")
        require(y.shape == (len(ids),) and len(self.patients) == len(ids), "Validation identity/answer mismatch")
        require(not self.units or len(self.units) == len(ids), "Validation unit count differs from cell IDs")
        require(all(p and str(p) not in {"None", "nan"} for p in self.patients), "Missing validation identity")
        classes = unique_ids(self.classes, "validation ontology")
        if self.task == "classification":
            require(bool(classes) and set(y) <= set(classes), "Validation labels outside ontology")
        else:
            y = y.astype(float)
            require(np.isfinite(y).all(), "Nonfinite source regression targets")
            require(self.measured is not None and len(self.units) == len(ids),
                    "Regression validation needs source measured A and unit IDs")
            measured = np.array(self.measured, dtype=float, copy=True)
            require(measured.shape == y.shape and np.isfinite(measured).all(), "Invalid measured source A")
            require(all(u and str(u) not in {"None", "nan"} for u in self.units), "Missing validation unit")
            require(self.minimum_cells >= 2, "Invalid validation support rule")
            unit_patient = {}
            for unit, patient in zip(self.units, self.patients):
                require(unit_patient.setdefault(unit, patient) == patient, "Validation unit spans patients")
            measured.flags.writeable = False
            object.__setattr__(self, "measured", measured)
        if self.identity_level == "unit_descriptive":
            require(len(self.units) == len(ids) and tuple(self.patients) == tuple(self.units),
                    "Unit-descriptive weighting must use units, not inferred patients")
        x.flags.writeable = y.flags.writeable = False
        object.__setattr__(self, "cell_ids", ids)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "targets", y)
        object.__setattr__(self, "patients", tuple(self.patients))
        object.__setattr__(self, "units", tuple(self.units))
        object.__setattr__(self, "classes", classes)

    @property
    def identity_sha256(self):
        return digest(dict(cell_ids=self.cell_ids, patients=self.patients, units=self.units,
                           identity_level=self.identity_level, task=self.task, classes=self.classes))

    def score(self, prediction, *, cell_ids) -> float:
        require(tuple(cell_ids) == self.cell_ids, "Validation prediction order differs from identities")
        if self.task == "classification":
            return classification_score(self.targets, prediction, self.patients, self.classes)["macro_f1"]
        result = regression_score(self.measured, prediction, self.units, self.patients,
                                  minimum_cells=self.minimum_cells)
        score = result.get("effective_spearman")
        require(result["status"] == "ok" and score is not None and np.isfinite(score),
                "Source validation not evaluable; do not substitute target answers or another metric")
        return score

    def assert_scoreable(self):
        dummy = (np.repeat(self.classes[0], len(self.cell_ids)) if self.task == "classification"
                 else np.zeros(len(self.cell_ids)))
        self.score(dummy, cell_ids=self.cell_ids)
