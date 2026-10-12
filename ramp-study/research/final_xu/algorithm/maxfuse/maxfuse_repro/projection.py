"""Serializable projection state for already-prepared active feature panels."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .gcca import GCCAResult, PCAResult


SCHEMA_VERSION = "maxfuse_repro.projection.v2"
LEGACY_SCHEMA_VERSION = "maxfuse_repro.projection.v1"


@dataclass(frozen=True)
class ViewProjection:
    pca_input_center: np.ndarray
    pca_loadings: np.ndarray
    pca_score_center: np.ndarray
    pca_score_scale: np.ndarray
    gcca_pivot_center: np.ndarray
    gcca_pivot_scale: np.ndarray
    gcca_weight: np.ndarray
    output_center: np.ndarray
    output_scale: np.ndarray
    feature_names: tuple[str, ...]

    def transform(
        self,
        active: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        array = np.asarray(active, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.pca_input_center):
            raise ValueError("active feature matrix has the wrong shape")
        if not np.isfinite(array).all():
            raise ValueError("active feature matrix contains non-finite values")
        if feature_names is None:
            raise ValueError("feature_names are required for projection")
        if tuple(map(str, feature_names)) != self.feature_names:
            raise ValueError("active feature names or order do not match the fitted model")
        raw_scores = (array - self.pca_input_center) @ self.pca_loadings
        pca_scores = (raw_scores - self.pca_score_center) / self.pca_score_scale
        standardized = (pca_scores - self.gcca_pivot_center) / self.gcca_pivot_scale
        raw_result = standardized @ self.gcca_weight
        result = (raw_result - self.output_center) / self.output_scale
        if not np.isfinite(result).all():
            raise RuntimeError("projection produced non-finite values")
        return result


@dataclass(frozen=True)
class ThreeWayProjectionModel:
    view_order: tuple[str, str, str]
    views: Mapping[str, ViewProjection]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if len(self.view_order) != 3 or len(set(self.view_order)) != 3:
            raise ValueError("view_order must contain three unique names")
        if set(self.view_order) != set(self.views):
            raise ValueError("view_order and projection-state names differ")
        if self.schema_version not in {SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}:
            raise ValueError("unsupported projection schema version")

    def transform(
        self,
        view_name: str,
        active: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        if view_name not in self.views:
            raise KeyError(f"unknown view {view_name!r}")
        return self.views[view_name].transform(active, feature_names)

    def save(self, path: Path) -> None:
        path = Path(path)
        arrays: dict[str, np.ndarray] = {}
        metadata = {
            "schema_version": self.schema_version,
            "view_order": list(self.view_order),
            "feature_names": {
                name: list(self.views[name].feature_names)
                for name in self.view_order
            },
        }
        for index, name in enumerate(self.view_order):
            state = self.views[name]
            prefix = f"view{index}"
            arrays[f"{prefix}__pca_input_center"] = state.pca_input_center
            arrays[f"{prefix}__pca_loadings"] = state.pca_loadings
            arrays[f"{prefix}__pca_score_center"] = state.pca_score_center
            arrays[f"{prefix}__pca_score_scale"] = state.pca_score_scale
            arrays[f"{prefix}__gcca_pivot_center"] = state.gcca_pivot_center
            arrays[f"{prefix}__gcca_pivot_scale"] = state.gcca_pivot_scale
            arrays[f"{prefix}__gcca_weight"] = state.gcca_weight
            arrays[f"{prefix}__output_center"] = state.output_center
            arrays[f"{prefix}__output_scale"] = state.output_scale
        arrays["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        )
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "ThreeWayProjectionModel":
        path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            stored_schema = metadata.get("schema_version")
            if stored_schema not in {SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}:
                raise ValueError("unsupported or missing projection schema version")
            order = tuple(map(str, metadata["view_order"]))
            if len(order) != 3:
                raise ValueError("corrupt projection model view order")
            states: dict[str, ViewProjection] = {}
            for index, name in enumerate(order):
                prefix = f"view{index}"
                features = metadata["feature_names"].get(name)
                if not isinstance(features, list) or not features:
                    raise ValueError("projection model lacks required feature names")
                weight = archive[f"{prefix}__gcca_weight"].copy()
                if stored_schema == SCHEMA_VERSION:
                    output_center = archive[f"{prefix}__output_center"].copy()
                    output_scale = archive[f"{prefix}__output_scale"].copy()
                else:
                    output_center = np.zeros(weight.shape[1], dtype=np.float64)
                    output_scale = np.ones(weight.shape[1], dtype=np.float64)
                states[name] = ViewProjection(
                    pca_input_center=archive[f"{prefix}__pca_input_center"].copy(),
                    pca_loadings=archive[f"{prefix}__pca_loadings"].copy(),
                    pca_score_center=archive[f"{prefix}__pca_score_center"].copy(),
                    pca_score_scale=archive[f"{prefix}__pca_score_scale"].copy(),
                    gcca_pivot_center=archive[f"{prefix}__gcca_pivot_center"].copy(),
                    gcca_pivot_scale=archive[f"{prefix}__gcca_pivot_scale"].copy(),
                    gcca_weight=weight,
                    output_center=output_center,
                    output_scale=output_scale,
                    feature_names=tuple(map(str, features)),
                )
        # Loaded v1 models retain their exact old transformation but are held
        # in the current in-memory schema so a subsequent save writes v2.
        return cls(order, states, schema_version=SCHEMA_VERSION)


def build_projection_model(
    view_order: Sequence[str],
    pca_results: Mapping[str, PCAResult],
    gcca_result: GCCAResult,
    feature_names: Mapping[str, tuple[str, ...]],
) -> ThreeWayProjectionModel:
    order = tuple(map(str, view_order))
    if len(order) != 3 or len(gcca_result.weights) != 3:
        raise ValueError("exactly three views are required")
    states: dict[str, ViewProjection] = {}
    if gcca_result.output_centers:
        output_centers = gcca_result.output_centers
        output_scales = gcca_result.output_scales
    else:
        output_centers = tuple(
            np.zeros(weight.shape[1], dtype=np.float64)
            for weight in gcca_result.weights
        )
        output_scales = tuple(
            np.ones(weight.shape[1], dtype=np.float64)
            for weight in gcca_result.weights
        )
    for name, weight, center, scale, output_center, output_scale in zip(
        order,
        gcca_result.weights,
        gcca_result.pivot_centers,
        gcca_result.pivot_scales,
        output_centers,
        output_scales,
    ):
        pca = pca_results[name]
        states[name] = ViewProjection(
            pca_input_center=pca.input_center,
            pca_loadings=pca.loadings,
            pca_score_center=pca.score_center,
            pca_score_scale=pca.score_scale,
            gcca_pivot_center=center,
            gcca_pivot_scale=scale,
            gcca_weight=weight,
            output_center=output_center,
            output_scale=output_scale,
            feature_names=feature_names[name],
        )
    return ThreeWayProjectionModel(order, states)


# Canonical name for projection state produced by the explicitly named local
# three-view extension.  The legacy class name and NPZ schema remain supported
# so previously fitted models can still be loaded exactly.
ThreeViewExtensionProjectionModel = ThreeWayProjectionModel
