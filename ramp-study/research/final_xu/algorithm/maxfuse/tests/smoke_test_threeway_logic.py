#!/usr/bin/env python3
"""Deterministic tests for hub chaining and explicit outer-reuse semantics."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maxfuse_repro import (
    THREE_VIEW_EXTENSION_METHOD,
    Matching,
    ThreeViewExtensionAudit,
    ThreeWayAudit,
    chain_hub_matches,
    fit_three_view_extension,
    fit_threeway,
)


def main() -> None:
    if not THREE_VIEW_EXTENSION_METHOD.startswith("MaxFuse Archive three-view workflow"):
        raise AssertionError("three-view extension method name drifted")
    if fit_threeway is not fit_three_view_extension:
        raise AssertionError("legacy fit_threeway alias no longer targets the canonical API")
    if ThreeWayAudit is not ThreeViewExtensionAudit:
        raise AssertionError("legacy ThreeWayAudit alias no longer targets the canonical type")
    left = Matching(
        first=np.asarray([0, 1, 2, 3], dtype=np.int64),
        second=np.asarray([0, 0, 1, 2], dtype=np.int64),
        score=np.asarray([0.90, 0.80, 0.70, 0.60]),
    )
    right = Matching(
        first=np.asarray([0, 1, 2, 3], dtype=np.int64),
        second=np.asarray([0, 1, 2, 3], dtype=np.int64),
        score=np.asarray([0.85, 0.75, 0.65, 0.55]),
    )
    allowed = chain_hub_matches(
        left, right, view_sizes=(4, 3, 4), outer_reuse_policy="allow"
    )
    if len(allowed.hub) != 4 or allowed.outer_left_reused != 1:
        raise AssertionError("allow policy did not preserve explicit outer reuse")
    dropped = chain_hub_matches(
        left, right, view_sizes=(4, 3, 4), outer_reuse_policy="drop_reused"
    )
    if len(dropped.hub) != 3 or dropped.outer_left_reused != 0:
        raise AssertionError("drop_reused policy did not remove repeated outer endpoints")
    if set(dropped.hub) != {0, 2, 3}:
        raise AssertionError("drop_reused did not retain the highest-score conflicting tuple")
    print("MAXFUSE_REPRO THREE_VIEW_EXTENSION_LOGIC_SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
