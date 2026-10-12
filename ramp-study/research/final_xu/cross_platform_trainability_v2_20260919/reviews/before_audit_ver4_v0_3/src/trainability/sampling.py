"""Sampling functions never silently clip a requested budget."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .core import require, seed_for, unique_ids


def nested_draws(cell_ids, budgets, seed: int) -> dict[int, tuple[str, ...]]:
    """Label-blind, row-order-invariant prefixes of one random permutation."""
    ids = sorted(unique_ids(cell_ids))
    budgets = tuple(budgets)
    require(len(budgets) == len(set(budgets)), "Duplicate budgets")
    require(all(isinstance(n, int) and 0 < n <= len(ids) for n in budgets),
            "Requested budget unsupported; do not silently truncate")
    order = np.random.default_rng(seed).permutation(len(ids))
    shuffled = tuple(ids[i] for i in order)
    return {n: shuffled[:n] for n in budgets}


@dataclass(frozen=True)
class Supervision:
    fit: tuple[str, ...]
    validation: tuple[str, ...]
    calibration: tuple[str, ...] = ()

    def __post_init__(self):
        groups = [unique_ids(x) for x in (self.fit, self.validation, self.calibration)]
        require(len(set().union(*map(set, groups))) == sum(map(len, groups)),
                "fit/validation/calibration overlap")
        require(bool(self.fit) and bool(self.validation), "Empty supervised fit or validation")
        for name, ids in zip(("fit", "validation", "calibration"), groups):
            object.__setattr__(self, name, ids)

    @property
    def total(self) -> int:
        return len(set(self.fit) | set(self.validation) | set(self.calibration))

    def assert_allowed(self, allowed_source_ids, target_ids) -> None:
        used = set(self.fit) | set(self.validation) | set(self.calibration)
        require(used <= set(allowed_source_ids), "Supervision outside declared source")
        require(not used.intersection(target_ids), "Target cells used for supervision")


def source_split(cell_ids, patient_by_cell: Mapping[str, str], validation_fraction: float,
                 seed: int, *, patient_mapping_verified: bool) -> dict:
    ids = sorted(unique_ids(cell_ids))
    require(patient_mapping_verified, "Patient identity requires a reviewed mapping")
    require(0 < validation_fraction < 1, "Invalid validation fraction")
    require(set(ids) <= patient_by_cell.keys(), "Incomplete patient mapping")
    require(all(patient_by_cell[i] not in {"", None, "nan"} for i in ids), "Missing patient ID")
    patients = sorted({patient_by_cell[i] for i in ids})
    if len(patients) == 1:
        require(len(ids) >= 2, "Cannot split a single cell")
        n = min(len(ids) - 1, max(1, round(len(ids) * validation_fraction)))
        val = set(nested_draws(ids, [n], seed)[n])
        mode = "within_patient_cell_validation"
    else:
        n = min(len(patients) - 1, max(1, round(len(patients) * validation_fraction)))
        vp = set(nested_draws(patients, [n], seed)[n])
        val = {i for i in ids if patient_by_cell[i] in vp}
        mode = "patient_disjoint_validation"
    return dict(fit_pool=tuple(i for i in ids if i not in val),
                validation_pool=tuple(i for i in ids if i in val), mode=mode)


def reserve_calibration(cell_ids, unit_by_cell: Mapping[str, str], per_unit: int,
                        seed: int, *, minimum_remaining: int = 1) -> dict:
    ids = unique_ids(cell_ids)
    require(per_unit > 0 and minimum_remaining > 0, "Invalid calibration budget")
    require(set(ids) <= unit_by_cell.keys(), "Incomplete unit mapping")
    calibration, remaining = [], []
    for unit in sorted({unit_by_cell[i] for i in ids}):
        pool = [i for i in ids if unit_by_cell[i] == unit]
        require(len(pool) >= per_unit + minimum_remaining,
                f"Insufficient calibration/remaining support in unit {unit}")
        cal = nested_draws(pool, [per_unit], seed_for(seed, "calibration", unit))[per_unit]
        calibration.extend(cal)
        remaining.extend(sorted(set(pool) - set(cal)))
    return dict(calibration=tuple(calibration), remaining=tuple(remaining))


def matched_j3(a_ids, b_ids, labels: Mapping[str, str], quotas: Mapping[str, int],
               seed: int) -> dict[str, tuple[str, ...]]:
    """Apply independently to fit pools and disjoint validation pools.

    Quotas are q_c; each single source has 2*q_c and the mixture q_c from each.
    This deliberately label-aware experiment is NOT the J1 labelling curve.
    """
    a, b = unique_ids(a_ids), unique_ids(b_ids)
    require(not set(a).intersection(b), "Source A/B cell overlap")
    require(set(a + b) <= labels.keys(), "Missing source labels")
    require(bool(quotas), "No J3 quotas")
    out = {"A-only": [], "B-only": [], "A+B": []}
    for label, q in sorted(quotas.items()):
        require(isinstance(q, int) and q > 0, "Invalid per-class quota")
        sa = nested_draws([i for i in a if labels[i] == label], [2*q],
                          seed_for(seed, "A", label))[2*q]
        sb = nested_draws([i for i in b if labels[i] == label], [2*q],
                          seed_for(seed, "B", label))[2*q]
        out["A-only"].extend(sa)
        out["B-only"].extend(sb)
        out["A+B"].extend(sa[:q] + sb[:q])
    for label, q in quotas.items():
        require(all(sum(labels[i] == label for i in v) == 2*q for v in out.values()),
                "J3 class budgets differ")
    require(set(out["A+B"]) & set(a) <= set(out["A-only"]), "Non-nested source A")
    require(set(out["A+B"]) & set(b) <= set(out["B-only"]), "Non-nested source B")
    return {k: tuple(v) for k, v in out.items()}


def validate_j3_supervision(plans: Mapping[str, Supervision], labels: Mapping[str, str],
                            source_a, source_b, target_ids) -> None:
    require(set(plans) == {"A-only", "B-only", "A+B"}, "Incomplete J3 comparison")
    require(not set(source_a).intersection(source_b), "Sources overlap")
    allowed = {"A-only": set(source_a), "B-only": set(source_b),
               "A+B": set(source_a) | set(source_b)}
    for name, plan in plans.items():
        require(not plan.calibration, "J3 has no marker calibration budget")
        plan.assert_allowed(allowed[name], target_ids)
    # Across conditions no cell may change from fit to validation.
    fits = set().union(*(set(p.fit) for p in plans.values()))
    vals = set().union(*(set(p.validation) for p in plans.values()))
    require(not fits.intersection(vals), "Cross-condition fit/validation contamination")
    require(fits | vals <= labels.keys(), "Missing J3 labels")
    for role in ("fit", "validation"):
        by_arm = {n: getattr(p, role) for n, p in plans.items()}
        hist = [{c: sum(labels[i] == c for i in ids) for c in set(labels.values())}
                for ids in by_arm.values()]
        require(all(x == hist[0] for x in hist), f"Unequal J3 {role} class budgets")
        for source, single in ((set(source_a), "A-only"), (set(source_b), "B-only")):
            mixed = set(by_arm["A+B"]) & source
            require(mixed <= set(by_arm[single]), f"Non-nested J3 {role}")
            require(all(2*sum(labels[i] == c for i in mixed) == n for c, n in hist[0].items()),
                    "J3 mixture does not contain half of each source's class quota")
    require(len({p.total for p in plans.values()}) == 1, "J3 total supervision differs")


def validate_pretraining_sets(corpus, pretrain_validation, probe: Supervision,
                              allowed_source_ids, target_ids) -> None:
    require(not probe.calibration, "Classification probe has no marker calibration")
    groups = [unique_ids(x) for x in (corpus, pretrain_validation, probe.fit, probe.validation)]
    require(all(groups), "Empty pretraining/probe partition")
    all_ids = set().union(*map(set, groups))
    require(len(all_ids) == sum(map(len, groups)), "Pretraining/probe cell overlap")
    require(all_ids <= set(allowed_source_ids), "Non-source in pretraining/probe")
    require(not all_ids.intersection(target_ids), "Target entered encoder corpus/probe")
