#!/usr/bin/env python3
"""Deterministic label-free multitask model used for CODEX CD19 completion.

Cell labels, spatial coordinates, and the failed CODEX CD19 measurement are
never supplied to this model.  The 14 healthy shared markers are masked
reconstruction targets in all platforms; CD19 is a target only in Triana AbSeq
and BM CyTOF.
"""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ALGORITHM_ROOT = HERE.parents[1] / "algorithm"
sys.path.insert(0, str(ALGORITHM_ROOT))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import torch
from torch import nn

from ramp import ramp_ver2 as ramp


DATASETS = ("BM CODEX", "Triana AbSeq", "BM CyTOF")
NORMAL_MARKERS = (
    "CD117", "CD11B", "CD11C", "CD123", "CD14", "CD3", "CD33",
    "CD34", "CD38", "CD4", "CD45", "CD45RA", "CD61", "CD8",
)
TARGETS = NORMAL_MARKERS + ("CD19",)
CD19_INDEX = len(NORMAL_MARKERS)
PLATFORM_TO_ID = {name: index for index, name in enumerate(DATASETS)}
CODEX_ID = PLATFORM_TO_ID["BM CODEX"]
TRIANA_ID = PLATFORM_TO_ID["Triana AbSeq"]
CYTOF_ID = PLATFORM_TO_ID["BM CyTOF"]

FIT_CAP_PER_PLATFORM = 4_000
BATCH_SIZE = 512
EPOCHS_FINAL = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 2e-4
ADAPTER_SCALE = 0.20
SOURCE_RESIDUAL_SCALE = 0.20


def stable_seed(*parts: object) -> int:
    # Frozen for exact reproducibility with the selected development model.
    return ramp.stable_seed("HIERARCHICAL_PROXY", *parts)


def donor_balanced_fit_indices(metadata: pd.DataFrame) -> np.ndarray:
    """Select a fixed donor-balanced fitting subset from each platform."""
    selected: list[np.ndarray] = []
    for dataset in DATASETS:
        rows = np.flatnonzero(metadata.dataset.astype(str).eq(dataset))
        local = metadata.iloc[rows]
        groups = list(local.groupby("unit_key", sort=True))
        quota = FIT_CAP_PER_PLATFORM // len(groups)
        pieces: list[np.ndarray] = []
        leftovers: list[np.ndarray] = []
        for group_index, (_, frame) in enumerate(groups):
            values = frame.index.to_numpy(int)
            rng = np.random.default_rng(stable_seed("FIT", dataset, group_index))
            values = rng.permutation(values)
            pieces.append(values[: min(quota, len(values))])
            leftovers.append(values[min(quota, len(values)):])
        chosen = np.concatenate(pieces)
        remaining = FIT_CAP_PER_PLATFORM - len(chosen)
        if remaining > 0:
            pool = np.concatenate(leftovers)
            rng = np.random.default_rng(stable_seed("FIT_REMAINDER", dataset))
            chosen = np.concatenate([chosen, rng.permutation(pool)[:remaining]])
        selected.append(np.sort(chosen))
    answer = np.concatenate(selected)
    expected = FIT_CAP_PER_PLATFORM * len(DATASETS)
    if len(answer) != expected:
        raise RuntimeError(f"Expected {expected:,} fitting cells; found {len(answer):,}")
    return answer


class MaskedMarkerModel(nn.Module):
    """Shared trunk with small input- and output-platform residuals."""

    def __init__(self, n_features: int, n_targets: int, n_platforms: int) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_targets = n_targets
        self.shared = nn.Sequential(
            nn.Linear(2 * n_features, 64), nn.SiLU(),
            nn.Linear(64, 40), nn.SiLU(),
        )
        self.input_adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(40, 8), nn.Tanh(), nn.Linear(8, 40))
            for _ in range(n_platforms)
        ])
        self.marker_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(40, 16), nn.SiLU(), nn.Linear(16, 1))
            for _ in range(n_targets)
        ])
        self.output_residuals = nn.ModuleList([
            nn.Sequential(nn.Linear(40, 8), nn.Tanh(), nn.Linear(8, n_targets))
            for _ in range(n_platforms)
        ])

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        input_platform: torch.Tensor,
        output_platform: torch.Tensor | None,
    ) -> torch.Tensor:
        masked = x.clone()
        mask = torch.zeros_like(masked)
        normal = target < self.n_features
        if torch.any(normal):
            rows = torch.nonzero(normal, as_tuple=False).squeeze(1)
            columns = target[normal]
            masked[rows, columns] = 0.0
            mask[rows, columns] = 1.0

        shared = self.shared(torch.cat([masked, mask], dim=1))
        hidden = shared.clone()
        for platform, adapter in enumerate(self.input_adapters):
            rows = input_platform == platform
            if torch.any(rows):
                hidden[rows] = shared[rows] + ADAPTER_SCALE * adapter(shared[rows])

        prediction = torch.empty(len(hidden), dtype=hidden.dtype, device=hidden.device)
        for marker, head in enumerate(self.marker_heads):
            rows = target == marker
            if torch.any(rows):
                prediction[rows] = head(hidden[rows]).squeeze(1)

        if output_platform is not None:
            for platform, residual in enumerate(self.output_residuals):
                rows = output_platform == platform
                if torch.any(rows):
                    values = residual(hidden[rows])
                    prediction[rows] += (
                        SOURCE_RESIDUAL_SCALE
                        * values.gather(1, target[rows, None]).squeeze(1)
                    )
        return prediction


def target_balance_weights(
    target_values: np.ndarray,
    metadata: pd.DataFrame,
) -> np.ndarray:
    """Balance donor-specific unsupervised K=2 low/high states per target."""
    weights = np.ones_like(target_values, dtype=np.float32)
    unit_values = metadata.unit_key.astype(str)
    for target, marker in enumerate(TARGETS):
        valid = np.isfinite(target_values[:, target])
        for unit in sorted(pd.unique(metadata.loc[valid, "unit_key"].astype(str))):
            rows = np.flatnonzero(valid & unit_values.eq(unit).to_numpy())
            values = target_values[rows, target]
            fit = KMeans(
                n_clusters=2,
                n_init=20,
                max_iter=500,
                random_state=stable_seed("TRAIN_BALANCE", marker, unit),
            ).fit(values[:, None])
            for state in (0, 1):
                local = fit.labels_ == state
                if np.any(local):
                    weights[rows[local], target] = len(rows) / (2.0 * local.sum())
    return weights


def allowed_targets(platform: int) -> np.ndarray:
    targets = list(range(len(NORMAL_MARKERS)))
    if platform in (TRIANA_ID, CYTOF_ID):
        targets.append(CD19_INDEX)
    return np.asarray(targets, dtype=np.int64)


def train_model(
    features: np.ndarray,
    target_values: np.ndarray,
    balance_weights: np.ndarray,
    platform_ids: np.ndarray,
    fit_indices: np.ndarray,
) -> MaskedMarkerModel:
    seed = stable_seed("DIRECT_MULTITASK_SHARED_SEED")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MaskedMarkerModel(
        n_features=features.shape[1],
        n_targets=target_values.shape[1],
        n_platforms=len(DATASETS),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    fit = np.asarray(fit_indices, dtype=int)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(EPOCHS_FINAL):
        order = rng.permutation(fit)
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start:start + BATCH_SIZE]
            platforms = platform_ids[rows]
            target = np.empty(len(rows), dtype=np.int64)
            for platform in range(len(DATASETS)):
                local = np.flatnonzero(platforms == platform)
                if len(local):
                    target[local] = rng.choice(
                        allowed_targets(platform), size=len(local), replace=True
                    )

            xb = torch.from_numpy(features[rows]).float()
            tb = torch.from_numpy(target).long()
            pb = torch.from_numpy(platforms).long()
            yb = torch.from_numpy(target_values[rows, target]).float()
            wb = torch.from_numpy(balance_weights[rows, target]).float()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb, tb, pb, pb)
            loss = (
                nn.functional.smooth_l1_loss(
                    prediction, yb, beta=0.5, reduction="none"
                ) * wb
            ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
    return model


@torch.no_grad()
def predict_cd19(
    model: MaskedMarkerModel,
    features: np.ndarray,
    codex_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict Triana-like and CyTOF-like CD19 and return their fixed mean."""

    def one_source(output_platform: int) -> np.ndarray:
        model.eval()
        answer = np.empty(len(codex_rows), dtype=np.float64)
        for start in range(0, len(codex_rows), 4096):
            stop = min(start + 4096, len(codex_rows))
            rows = codex_rows[start:stop]
            x = torch.from_numpy(features[rows]).float()
            target = torch.full((len(rows),), CD19_INDEX, dtype=torch.long)
            input_platform = torch.full((len(rows),), CODEX_ID, dtype=torch.long)
            output = torch.full((len(rows),), output_platform, dtype=torch.long)
            answer[start:stop] = model(
                x, target, input_platform, output
            ).cpu().numpy()
        return answer

    triana = one_source(TRIANA_ID)
    cytof = one_source(CYTOF_ID)
    return 0.5 * (triana + cytof), triana, cytof


@torch.no_grad()
def predict_normal_marker(
    model: MaskedMarkerModel,
    features: np.ndarray,
    codex_rows: np.ndarray,
    marker_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict a masked normal marker using the same two-source rule as CD19."""
    if marker_index < 0 or marker_index >= len(NORMAL_MARKERS):
        raise ValueError(marker_index)

    def one_source(output_platform: int) -> np.ndarray:
        model.eval()
        answer = np.empty(len(codex_rows), dtype=np.float64)
        for start in range(0, len(codex_rows), 4096):
            stop = min(start + 4096, len(codex_rows))
            rows = codex_rows[start:stop]
            x = torch.from_numpy(features[rows]).float()
            target = torch.full((len(rows),), marker_index, dtype=torch.long)
            input_platform = torch.full((len(rows),), CODEX_ID, dtype=torch.long)
            output = torch.full((len(rows),), output_platform, dtype=torch.long)
            answer[start:stop] = model(
                x, target, input_platform, output
            ).cpu().numpy()
        return answer

    triana = one_source(TRIANA_ID)
    cytof = one_source(CYTOF_ID)
    return 0.5 * (triana + cytof), triana, cytof
