"""Thin, label-free adapter around the official MaxFuse 0.0.2 public API.

This module does not contain or modify MaxFuse source code.  It freezes the
public call sequence used by the authors' current CITE-seq tutorial and makes
every argument explicit for reproducibility.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import math
import numbers
from typing import Any, Iterator, Mapping

import numpy as np

from .types import Matching, ViewData


PAIRWISE_CONFIG_SCHEMA = "maxfuse_repro.PairwiseConfig.v1"


@dataclass(frozen=True)
class BatchingConfig:
    max_outward_size: int = 5000
    matching_ratio: float = 3.0
    metacell_size: int = 2
    method: str = "random"
    batching_scheme: str = "pairwise"
    prebatching_smoothing: bool = False
    shared_wt1: float = 1.0
    shared_wt2: float = 1.0
    active_wt1: float = 1.0
    active_wt2: float = 1.0
    shared_svd_components1: int | None = None
    shared_svd_components2: int | None = None
    active_svd_components1: int | None = None
    active_svd_components2: int | None = None
    randomized_svd: bool = False
    svd_runs: int = 1
    seed: int | None = None


@dataclass(frozen=True)
class GraphConfig:
    n_neighbors1: int = 15
    n_neighbors2: int = 15
    svd_components1: int | None = None
    svd_components2: int | None = None
    resolution1: float = 1.0
    resolution2: float = 1.0
    randomized_svd: bool = False
    svd_runs: int = 1
    resolution_tol: float = 0.1
    leiden_runs: int = 1
    metric: str = "correlation"
    leiden_seed: int | None = None


@dataclass(frozen=True)
class InitialPivotConfig:
    wt1: float = 0.3
    wt2: float = 0.3
    svd_components1: int | None = None
    svd_components2: int | None = None
    randomized_svd: bool = False
    svd_runs: int = 1


@dataclass(frozen=True)
class RefineConfig:
    wt1: float = 0.5
    wt2: float = 0.5
    svd_components1: int | None = None
    svd_components2: int | None = None
    cca_components: int | None = None
    filter_prop: float = 0.0
    n_iters: int = 1
    randomized_svd: bool = False
    svd_runs: int = 1
    cca_max_iter: int = 2000


@dataclass(frozen=True)
class PropagationConfig:
    enabled: bool = False
    wt1: float = 0.7
    wt2: float = 0.7
    svd_components1: int | None = None
    svd_components2: int | None = None
    metric: str = "euclidean"
    randomized_svd: bool = False
    svd_runs: int = 1
    filter_prop: float = 0.0
    full_matching_order: tuple[int, int] | None = (2, 1)


@dataclass(frozen=True)
class PairwiseConfig:
    """All numerical choices required by a pairwise MaxFuse run."""

    method: str = "centroid_shrinkage"
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    initial: InitialPivotConfig = field(default_factory=InitialPivotConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)
    pivot_filter_prop: float = 0.0
    propagation: PropagationConfig = field(default_factory=PropagationConfig)
    numpy_seed: int | None = None

    @classmethod
    def citeseq_pbmc_tutorial(
        cls,
        *,
        deterministic_seed: int | None = None,
        include_propagation: bool = True,
    ) -> "PairwiseConfig":
        """Exact published tutorial hyperparameters plus optional fixed seeds.

        The authors' notebook leaves the NumPy, batching, and Leiden seeds
        unspecified.  Supplying ``deterministic_seed`` is our sole addition to
        make repeated benchmarking reproducible; all numerical hyperparameters
        remain those in ``docs/citeseq_pbmc_evaluate.ipynb``.
        """

        return cls(
            method="centroid_shrinkage",
            batching=BatchingConfig(
                max_outward_size=5000,
                matching_ratio=3.0,
                metacell_size=2,
                method="random",
                batching_scheme="pairwise",
                prebatching_smoothing=False,
                shared_wt1=1.0,
                shared_wt2=1.0,
                active_wt1=1.0,
                active_wt2=1.0,
                shared_svd_components1=None,
                shared_svd_components2=None,
                active_svd_components1=None,
                active_svd_components2=None,
                randomized_svd=False,
                svd_runs=1,
                seed=deterministic_seed,
            ),
            graph=GraphConfig(
                n_neighbors1=15,
                n_neighbors2=15,
                svd_components1=30,
                svd_components2=30,
                resolution1=2.0,
                resolution2=2.0,
                randomized_svd=False,
                svd_runs=1,
                resolution_tol=0.1,
                leiden_runs=1,
                metric="correlation",
                leiden_seed=deterministic_seed,
            ),
            initial=InitialPivotConfig(
                wt1=0.7,
                wt2=0.7,
                svd_components1=25,
                svd_components2=20,
                randomized_svd=False,
                svd_runs=1,
            ),
            refine=RefineConfig(
                wt1=0.7,
                wt2=0.7,
                svd_components1=30,
                svd_components2=30,
                cca_components=20,
                filter_prop=0.0,
                n_iters=3,
                randomized_svd=False,
                svd_runs=1,
                cca_max_iter=2000,
            ),
            pivot_filter_prop=0.3,
            propagation=PropagationConfig(
                enabled=include_propagation,
                wt1=0.7,
                wt2=0.7,
                svd_components1=30,
                svd_components2=30,
                metric="euclidean",
                randomized_svd=False,
                svd_runs=1,
                filter_prop=0.0,
                full_matching_order=(2, 1),
            ),
            numpy_seed=deterministic_seed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": PAIRWISE_CONFIG_SCHEMA, **asdict(self)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PairwiseConfig":
        if not isinstance(value, Mapping):
            raise TypeError("PairwiseConfig must be a JSON object")
        payload = dict(value)
        required = {
            "schema_version",
            "method", "batching", "graph", "initial", "refine",
            "pivot_filter_prop", "propagation", "numpy_seed",
        }
        missing = required - set(payload)
        unknown = set(payload) - required
        if missing or unknown:
            raise ValueError(
                "PairwiseConfig keys differ from the complete schema; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if payload["schema_version"] != PAIRWISE_CONFIG_SCHEMA:
            raise ValueError(
                "unsupported PairwiseConfig schema_version: "
                f"{payload['schema_version']!r}"
            )

        def load_block(name: str, block_type: type[Any]) -> Any:
            raw = payload[name]
            if not isinstance(raw, Mapping):
                raise TypeError(f"PairwiseConfig.{name} must be a JSON object")
            block_payload = dict(raw)
            expected = {item.name for item in fields(block_type)}
            block_missing = expected - set(block_payload)
            block_unknown = set(block_payload) - expected
            if block_missing or block_unknown:
                raise ValueError(
                    f"PairwiseConfig.{name} keys differ from the complete schema; "
                    f"missing={sorted(block_missing)}, unknown={sorted(block_unknown)}"
                )
            return block_payload

        batching = load_block("batching", BatchingConfig)
        graph = load_block("graph", GraphConfig)
        initial = load_block("initial", InitialPivotConfig)
        refine = load_block("refine", RefineConfig)
        propagation = load_block("propagation", PropagationConfig)
        order = propagation["full_matching_order"]
        if order is not None:
            if not isinstance(order, (list, tuple)):
                raise TypeError("propagation.full_matching_order must be a list or null")
            propagation["full_matching_order"] = tuple(order)
        result = cls(
            method=payload["method"],
            batching=BatchingConfig(**batching),
            graph=GraphConfig(**graph),
            initial=InitialPivotConfig(**initial),
            refine=RefineConfig(**refine),
            pivot_filter_prop=payload["pivot_filter_prop"],
            propagation=PropagationConfig(**propagation),
            numpy_seed=payload["numpy_seed"],
        )
        _validate_config(result)
        return result


@dataclass(frozen=True)
class UpstreamAudit:
    package_version: str
    source_files: dict[str, str]
    reviewed_commit: str
    lock_file_sha256: str
    lock_verified: bool


@dataclass(frozen=True)
class PairwiseResult:
    pivot_matching: Matching
    embedding1: np.ndarray
    embedding2: np.ndarray
    full_matching: Matching | None
    config: PairwiseConfig
    upstream: UpstreamAudit


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_upstream_lock() -> UpstreamAudit:
    """Verify the installed external MaxFuse package against the frozen lock."""

    import maxfuse

    lock_path = Path(__file__).resolve().with_name("upstream_lock.json")
    lock = json.loads(lock_path.read_text())
    observed_version = version("maxfuse")
    package_root = Path(maxfuse.__file__).resolve().parent
    observed: dict[str, str] = {}
    mismatches: list[str] = []
    if observed_version != lock["package_version"]:
        mismatches.append(
            f"package version {observed_version!r} != {lock['package_version']!r}"
        )
    for relative, expected in lock["source_sha256"].items():
        path = package_root / relative
        if not path.is_file():
            mismatches.append(f"missing installed source file: {relative}")
            continue
        observed[relative] = _sha256(path)
        if observed[relative] != expected:
            mismatches.append(
                f"{relative} sha256 {observed[relative]} != frozen {expected}"
            )
    if mismatches:
        raise RuntimeError("MaxFuse upstream lock mismatch: " + "; ".join(mismatches))
    return UpstreamAudit(
        package_version=observed_version,
        source_files=observed,
        reviewed_commit=lock["reviewed_repository_commit"],
        lock_file_sha256=_sha256(lock_path),
        lock_verified=True,
    )


def _validate_config(config: PairwiseConfig) -> None:
    def require_bool(name: str, value: object) -> None:
        if type(value) is not bool:
            raise TypeError(f"{name} must be a boolean")

    def require_int(name: str, value: object, minimum: int = 0) -> None:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(f"{name} must be an integer")
        if int(value) < minimum:
            raise ValueError(f"{name} must be at least {minimum}")

    def require_optional_int(name: str, value: object, minimum: int = 1) -> None:
        if value is not None:
            require_int(name, value, minimum)

    def require_number(name: str, value: object, minimum: float | None = None) -> None:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        if minimum is not None and float(value) < minimum:
            raise ValueError(f"{name} must be at least {minimum}")

    if config.method not in {"centroid_shrinkage", "graph_smoothing"}:
        raise ValueError("unsupported MaxFuse smoothing method")
    if config.batching.method not in {"random", "binning"}:
        raise ValueError("batching.method must be 'random' or 'binning'")
    if config.batching.batching_scheme not in {"pairwise", "cyclic"}:
        raise ValueError("unsupported batching scheme")
    require_int("batching.max_outward_size", config.batching.max_outward_size, 2)
    require_number("batching.matching_ratio", config.batching.matching_ratio, 1e-12)
    require_int("batching.metacell_size", config.batching.metacell_size, 1)
    require_bool("batching.prebatching_smoothing", config.batching.prebatching_smoothing)
    require_bool("batching.randomized_svd", config.batching.randomized_svd)
    require_int("batching.svd_runs", config.batching.svd_runs, 1)
    require_optional_int("batching.seed", config.batching.seed, 0)
    for field_name in ("shared_wt1", "shared_wt2", "active_wt1", "active_wt2"):
        value = getattr(config.batching, field_name)
        require_number(f"batching.{field_name}", value)
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"batching.{field_name} must lie in [0, 1]")
    for field_name in (
        "shared_svd_components1", "shared_svd_components2",
        "active_svd_components1", "active_svd_components2",
    ):
        require_optional_int(
            f"batching.{field_name}", getattr(config.batching, field_name), 1
        )
    require_int("graph.n_neighbors1", config.graph.n_neighbors1, 1)
    require_int("graph.n_neighbors2", config.graph.n_neighbors2, 1)
    require_optional_int("graph.svd_components1", config.graph.svd_components1, 1)
    require_optional_int("graph.svd_components2", config.graph.svd_components2, 1)
    require_number("graph.resolution1", config.graph.resolution1, 1e-12)
    require_number("graph.resolution2", config.graph.resolution2, 1e-12)
    require_number("graph.resolution_tol", config.graph.resolution_tol, 0.0)
    require_bool("graph.randomized_svd", config.graph.randomized_svd)
    require_int("graph.svd_runs", config.graph.svd_runs, 1)
    require_int("graph.leiden_runs", config.graph.leiden_runs, 1)
    require_optional_int("graph.leiden_seed", config.graph.leiden_seed, 0)
    if not isinstance(config.graph.metric, str) or not config.graph.metric:
        raise TypeError("graph.metric must be a non-empty string")
    require_optional_int("numpy_seed", config.numpy_seed, 0)
    for name, value in (
        ("pivot_filter_prop", config.pivot_filter_prop),
        ("refine.filter_prop", config.refine.filter_prop),
        ("propagation.filter_prop", config.propagation.filter_prop),
    ):
        require_number(name, value)
        if not 0.0 <= float(value) < 1.0:
            raise ValueError(f"{name} must lie in [0, 1)")
    for name, value in (
        ("initial.wt1", config.initial.wt1),
        ("initial.wt2", config.initial.wt2),
        ("refine.wt1", config.refine.wt1),
        ("refine.wt2", config.refine.wt2),
        ("propagation.wt1", config.propagation.wt1),
        ("propagation.wt2", config.propagation.wt2),
    ):
        require_number(name, value)
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    for prefix, block in (
        ("initial", config.initial),
        ("refine", config.refine),
    ):
        require_bool(f"{prefix}.randomized_svd", block.randomized_svd)
        require_int(f"{prefix}.svd_runs", block.svd_runs, 1)
        require_optional_int(f"{prefix}.svd_components1", block.svd_components1, 1)
        require_optional_int(f"{prefix}.svd_components2", block.svd_components2, 1)
    require_optional_int("refine.cca_components", config.refine.cca_components, 1)
    require_int("refine.n_iters", config.refine.n_iters, 1)
    require_int("refine.cca_max_iter", config.refine.cca_max_iter, 1)
    require_bool("propagation.enabled", config.propagation.enabled)
    require_bool("propagation.randomized_svd", config.propagation.randomized_svd)
    require_int("propagation.svd_runs", config.propagation.svd_runs, 1)
    require_optional_int(
        "propagation.svd_components1", config.propagation.svd_components1, 1
    )
    require_optional_int(
        "propagation.svd_components2", config.propagation.svd_components2, 1
    )
    if not isinstance(config.propagation.metric, str) or not config.propagation.metric:
        raise TypeError("propagation.metric must be a non-empty string")
    order = config.propagation.full_matching_order
    if order is not None and (
        not isinstance(order, tuple) or order not in {(1, 2), (2, 1)}
    ):
        raise ValueError("propagation.full_matching_order must be null, [1,2], or [2,1]")


@contextmanager
def _preserved_numpy_rng(seed: int | None) -> Iterator[None]:
    state = np.random.get_state()
    try:
        if seed is not None:
            np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)


def _load_upstream() -> tuple[Any, Any]:
    from maxfuse import match_utils
    from maxfuse.model import Fusor

    return Fusor, match_utils


def best_per_first(matching: Matching) -> Matching:
    """Apply MaxFuse's official first-to-second redundancy rule."""

    _, match_utils = _load_upstream()
    resolved = match_utils.address_matching_redundancy(
        matching=matching.as_upstream(), order=(1, 2)
    )
    return Matching.from_upstream(resolved).validated(require_unique_first=True)


def fit_pairwise(
    first: ViewData,
    second: ViewData,
    config: PairwiseConfig,
) -> PairwiseResult:
    """Run the official label-free pairwise workflow with explicit arguments."""

    first = first.validated("first")
    second = second.validated("second")
    if first.shared.shape[1] != second.shared.shape[1]:
        raise ValueError("the two shared arrays must have the same number of columns")
    if first.shared_feature_names is None or second.shared_feature_names is None:
        raise ValueError("shared_feature_names are required for both views")
    if first.shared_feature_names != second.shared_feature_names:
        raise ValueError("shared feature names or column order differ between the two views")
    _validate_config(config)
    upstream = verify_upstream_lock()
    Fusor, _ = _load_upstream()
    batch = config.batching
    graph = config.graph
    initial = config.initial
    refine = config.refine
    propagation = config.propagation

    with _preserved_numpy_rng(config.numpy_seed):
        model = Fusor(
            shared_arr1=first.shared,
            shared_arr2=second.shared,
            active_arr1=first.active,
            active_arr2=second.active,
            method=config.method,
            labels1=None,
            labels2=None,
        )
        model.split_into_batches(
            max_outward_size=batch.max_outward_size,
            matching_ratio=batch.matching_ratio,
            metacell_size=batch.metacell_size,
            method=batch.method,
            batching_scheme=batch.batching_scheme,
            prebatching_smoothing=batch.prebatching_smoothing,
            shared_wt1=batch.shared_wt1,
            shared_wt2=batch.shared_wt2,
            active_wt1=batch.active_wt1,
            active_wt2=batch.active_wt2,
            shared_svd_components1=batch.shared_svd_components1,
            shared_svd_components2=batch.shared_svd_components2,
            active_svd_components1=batch.active_svd_components1,
            active_svd_components2=batch.active_svd_components2,
            randomized_svd=batch.randomized_svd,
            svd_runs=batch.svd_runs,
            seed=batch.seed,
            verbose=False,
        )
        model.construct_graphs(
            n_neighbors1=graph.n_neighbors1,
            n_neighbors2=graph.n_neighbors2,
            svd_components1=graph.svd_components1,
            svd_components2=graph.svd_components2,
            resolution1=graph.resolution1,
            resolution2=graph.resolution2,
            randomized_svd=graph.randomized_svd,
            svd_runs=graph.svd_runs,
            resolution_tol=graph.resolution_tol,
            leiden_runs=graph.leiden_runs,
            metric=graph.metric,
            leiden_seed=graph.leiden_seed,
            verbose=False,
        )
        model.find_initial_pivots(
            wt1=initial.wt1,
            wt2=initial.wt2,
            svd_components1=initial.svd_components1,
            svd_components2=initial.svd_components2,
            randomized_svd=initial.randomized_svd,
            svd_runs=initial.svd_runs,
            verbose=False,
        )
        model.refine_pivots(
            wt1=refine.wt1,
            wt2=refine.wt2,
            svd_components1=refine.svd_components1,
            svd_components2=refine.svd_components2,
            cca_components=refine.cca_components,
            filter_prop=refine.filter_prop,
            n_iters=refine.n_iters,
            randomized_svd=refine.randomized_svd,
            svd_runs=refine.svd_runs,
            cca_max_iter=refine.cca_max_iter,
            verbose=False,
        )
        model.filter_bad_matches(
            target="pivot", filter_prop=config.pivot_filter_prop, verbose=False
        )
        pivot = Matching.from_upstream(
            model.get_matching(order=None, target="pivot")
        ).validated(n_first=len(first.shared), n_second=len(second.shared))
        embedding1, embedding2 = model.get_embedding(
            active_arr1=first.active,
            active_arr2=second.active,
            refit=False,
            matching=None,
            order=None,
            cca_components=None,
            cca_max_iter=None,
        )
        full: Matching | None = None
        if propagation.enabled:
            model.propagate(
                wt1=propagation.wt1,
                wt2=propagation.wt2,
                svd_components1=propagation.svd_components1,
                svd_components2=propagation.svd_components2,
                metric=propagation.metric,
                randomized_svd=propagation.randomized_svd,
                svd_runs=propagation.svd_runs,
                verbose=False,
            )
            model.filter_bad_matches(
                target="propagated",
                filter_prop=propagation.filter_prop,
                verbose=False,
            )
            full = Matching.from_upstream(
                model.get_matching(
                    order=propagation.full_matching_order, target="full_data"
                )
            ).validated(n_first=len(first.shared), n_second=len(second.shared))

    embedding1 = np.asarray(embedding1, dtype=np.float64)
    embedding2 = np.asarray(embedding2, dtype=np.float64)
    if embedding1.shape[0] != len(first.active) or embedding2.shape[0] != len(second.active):
        raise RuntimeError("MaxFuse returned row-misaligned CCA embeddings")
    if not np.isfinite(embedding1).all() or not np.isfinite(embedding2).all():
        raise RuntimeError("MaxFuse returned non-finite CCA embeddings")
    return PairwiseResult(pivot, embedding1, embedding2, full, config, upstream)
