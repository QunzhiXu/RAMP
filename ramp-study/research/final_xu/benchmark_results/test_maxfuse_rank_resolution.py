#!/usr/bin/env python3
"""Fast numerical-contract test for benchmark-side MaxFuse rank clipping."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "_maxfuse_benchmark.py"
FINAL_XU = HERE.parent
sys.path.insert(0, str(FINAL_XU / "algorithm" / "maxfuse"))
from maxfuse_repro import PairwiseConfig
spec = importlib.util.spec_from_file_location("maxfuse_benchmark_runner", RUNNER)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import benchmark runner")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def main() -> None:
    config_path = HERE / "lung_cancer" / "maxfuse_v2" / "pairwise_config.json"
    declared = json.loads(config_path.read_text())
    frozen = copy.deepcopy(declared)
    resolved, audit = runner.resolve_pairwise_dimensions(
        declared,
        policy=runner.PAIRWISE_RANK_ADAPTATION,
        shared_width_first=8,
        shared_width_second=8,
        active_width_first=34,
        active_width_second=26,
    )
    assert declared == frozen, "resolver mutated the declared config"
    assert resolved["graph"]["svd_components1"] == 30
    assert resolved["graph"]["svd_components2"] == 25
    assert resolved["initial"]["svd_components1"] == 7
    assert resolved["initial"]["svd_components2"] == 7
    assert resolved["refine"]["svd_components1"] == 30
    assert resolved["refine"]["svd_components2"] == 25
    assert resolved["refine"]["cca_components"] == 20
    assert resolved["propagation"]["svd_components1"] == 30
    assert resolved["propagation"]["svd_components2"] == 25
    assert audit["policy"] == runner.PAIRWISE_RANK_ADAPTATION
    assert audit["feature_widths"] == {
        "shared1": 8, "shared2": 8, "active1": 34, "active2": 26
    }

    unchanged, no_adaptation = runner.resolve_pairwise_dimensions(
        declared,
        policy=None,
        shared_width_first=8,
        shared_width_second=8,
        active_width_first=34,
        active_width_second=26,
    )
    assert unchanged == declared
    assert no_adaptation["changes"] == []

    for benchmark in ("lung_cancer", "melanoma", "tnbc", "bone_marrow"):
        root = HERE / benchmark / "maxfuse_v2"
        benchmark_config = json.loads((root / "config.json").read_text())
        pair_template = json.loads((root / "pairwise_config.json").read_text())
        assert pair_template["batching"]["max_outward_size"] == 5000
        assert pair_template["graph"]["svd_components1"] == 30
        assert pair_template["initial"]["svd_components1"] == 25
        assert pair_template["initial"]["svd_components2"] == 20
        assert pair_template["refine"]["cca_components"] == 20
        datasets = benchmark_config["datasets"]
        for first in datasets:
            for second in datasets:
                if first["slug"] == second["slug"]:
                    continue
                candidate, _ = runner.resolve_pairwise_dimensions(
                    pair_template,
                    policy=benchmark_config["pairwise_rank_adaptation"],
                    shared_width_first=len(benchmark_config["shared_markers"]),
                    shared_width_second=len(benchmark_config["shared_markers"]),
                    active_width_first=len(first["active_markers"]),
                    active_width_second=len(second["active_markers"]),
                )
                PairwiseConfig.from_dict(candidate)
    print("MAXFUSE_BENCHMARK_RANK_RESOLUTION_OK")


if __name__ == "__main__":
    main()
