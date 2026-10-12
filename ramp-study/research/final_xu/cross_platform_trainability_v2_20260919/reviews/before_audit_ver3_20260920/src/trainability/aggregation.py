"""Strict paired support, explicit means and resource-boundary summaries."""
from __future__ import annotations

from collections import Counter
import math

import numpy as np
import pandas as pd

from .core import Q_GRID, require


def validate_records(frame: pd.DataFrame, keys, *, expected_keys=None) -> None:
    require(len(frame) > 0 and set(keys) <= set(frame.columns), "Missing record keys")
    require(not frame[list(keys)].isna().any().any(), "Null record key")
    require(not frame.duplicated(list(keys)).any(), "Duplicate metric/prediction key")
    if expected_keys is not None:
        actual = set(frame[list(keys)].itertuples(index=False, name=None))
        require(actual == set(expected_keys), "Record coverage differs from frozen manifest")


def hierarchical_mean(frame: pd.DataFrame, value: str, levels, *, keys) -> pd.DataFrame:
    """Each level is the grouping retained AFTER one explicitly equal-weighted mean.

    Example: repeat -> marker -> direction -> task, by providing successively
    coarser grouping columns. Caller must validate expected coverage first.
    """
    validate_records(frame, keys)
    require(np.isfinite(frame[value].to_numpy(dtype=float)).all(), "Cannot silently skip failed scores")
    require(bool(levels) and bool(levels[0]), "No aggregation levels")
    prev = set(keys)
    out = frame.copy()
    for group in levels:
        require(bool(group) and set(group) < prev, "Aggregation must strictly remove grouping dimensions")
        out = out.groupby(list(group), sort=True, dropna=False)[value].mean().reset_index()
        prev = set(group)
    return out


def paired_difference(left, right, *, keys, value="score") -> pd.DataFrame:
    validate_records(left, keys)
    validate_records(right, keys)
    require(set(left[list(keys)].itertuples(index=False, name=None)) ==
            set(right[list(keys)].itertuples(index=False, name=None)),
            "Paired comparison support differs; publish coverage before intersecting")
    joined = left[list(keys)+[value]].merge(right[list(keys)+[value]], on=list(keys),
                                          validate="one_to_one", suffixes=("_left", "_right"))
    require(np.isfinite(joined[[value+"_left", value+"_right"]].to_numpy(dtype=float)).all(),
            "Missing/failed paired score")
    joined["difference"] = joined[value+"_left"]-joined[value+"_right"]
    return joined


def resource_crossings(resources, scores, *, q_grid=Q_GRID) -> list[dict]:
    """Descriptive crossings of already repeat-averaged complete curves; no smoothing.

    Missing points make the curve unevaluable: a later hit cannot establish the
    smallest tested successful budget when earlier points are unknown.
    """
    require(len(resources) == len(scores) and len(resources) > 0, "Invalid curve")
    r = np.asarray(resources, dtype=float)
    require(np.isfinite(r).all() and (r > 0).all() and len(set(r)) == len(r), "Invalid resources")
    order = np.argsort(r)
    r = r[order]
    s = np.array([np.nan if scores[i] is None else scores[i] for i in order], dtype=float)
    require(all(math.isfinite(q) and 0 < q < 1 for q in q_grid), "Invalid Q grid")
    rows = []
    for q in q_grid:
        row = dict(q=float(q), status="not_evaluable", minimum_tested_resource=None,
                   resource_upper_bound=None, observed_score=None)
        if not np.isfinite(s).all():
            row["reason"] = "incomplete_or_failed_curve"
        else:
            hits = np.flatnonzero(s >= q)
            if not len(hits):
                row.update(status="not_reached", largest_tested_resource=float(r[-1]))
            elif hits[0] == 0:
                row.update(status="met_at_minimum_budget", resource_upper_bound=float(r[0]),
                           observed_score=float(s[0]))
            else:
                i = hits[0]
                row.update(status="met", minimum_tested_resource=float(r[i]), observed_score=float(s[i]))
        rows.append(row)
    return rows


def count_crossing_states(rows: list[dict]) -> list[dict]:
    """No conditional-success efficiency ratio is silently advertised as all-task."""
    result = []
    states = {"met", "met_at_minimum_budget", "not_reached", "not_evaluable"}
    for q in sorted({r["q"] for r in rows}):
        counts = Counter(r["status"] for r in rows if r["q"] == q)
        require(set(counts) <= states, "Unknown crossing state")
        result.append(dict(q=q, **{k: counts[k] for k in sorted(states)},
                           evaluable=counts["met"]+counts["met_at_minimum_budget"]+counts["not_reached"],
                           achieved=counts["met"]+counts["met_at_minimum_budget"]))
    return result
