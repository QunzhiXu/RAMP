"""Public API for the internal MaxFuse reproducibility layer."""

from .gcca import (
    GCCAResult,
    PCAResult,
    archive_svd_scores,
    fit_archive_gcca,
    fit_maxvar_gcca,
    pca_scores,
)
from .pairwise import (
    BatchingConfig,
    GraphConfig,
    InitialPivotConfig,
    PairwiseConfig,
    PAIRWISE_CONFIG_SCHEMA,
    PairwiseResult,
    PropagationConfig,
    RefineConfig,
    best_per_first,
    fit_pairwise,
    verify_upstream_lock,
)
from .projection import (
    ThreeViewExtensionProjectionModel,
    ThreeWayProjectionModel,
    ViewProjection,
)
from .provenance import (
    algorithm_source_hashes,
    output_hashes,
    runtime_manifest,
    sha256_file,
    strict_string_vector,
)
from .threeway import (
    THREE_VIEW_EXTENSION_METHOD,
    ThreeViewExtensionAudit,
    ThreeViewExtensionResult,
    ThreeWayAudit,
    ThreeWayResult,
    chain_hub_matches,
    fit_three_view_extension,
    fit_threeway,
)
from .types import Matching, TripletMatching, ViewData

__version__ = "1.2.0"

__all__ = [
    "BatchingConfig",
    "GCCAResult",
    "GraphConfig",
    "InitialPivotConfig",
    "Matching",
    "PCAResult",
    "PairwiseConfig",
    "PAIRWISE_CONFIG_SCHEMA",
    "PairwiseResult",
    "PropagationConfig",
    "RefineConfig",
    "THREE_VIEW_EXTENSION_METHOD",
    "ThreeViewExtensionAudit",
    "ThreeViewExtensionProjectionModel",
    "ThreeViewExtensionResult",
    "ThreeWayAudit",
    "ThreeWayProjectionModel",
    "ThreeWayResult",
    "TripletMatching",
    "ViewData",
    "ViewProjection",
    "best_per_first",
    "algorithm_source_hashes",
    "archive_svd_scores",
    "chain_hub_matches",
    "fit_archive_gcca",
    "fit_maxvar_gcca",
    "fit_pairwise",
    "fit_three_view_extension",
    "fit_threeway",
    "pca_scores",
    "output_hashes",
    "runtime_manifest",
    "sha256_file",
    "strict_string_vector",
    "verify_upstream_lock",
]
