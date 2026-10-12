"""J4 architectural/masking primitives, not a formal pretraining runner."""
from __future__ import annotations

import numpy as np
from .core import require


def make_mask(shape, fraction: float, seed: int) -> np.ndarray:
    require(len(shape) == 2 and shape[0] > 0 and shape[1] >= 2, "At least two protein features required")
    require(0 < fraction < 1, "Invalid mask fraction")
    k = min(shape[1]-1, max(1, round(shape[1]*fraction)))
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, dtype=bool)
    for row in mask:
        row[rng.choice(shape[1], k, replace=False)] = True
    return mask


def make_encoder(n_markers: int, width: int):
    import torch
    from torch import nn

    require(n_markers >= 2 and width > 0, "Invalid encoder architecture")

    class MaskedProteinModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(2*n_markers, width), nn.ReLU(),
                                         nn.Linear(width, width), nn.ReLU())
            self.decoder = nn.Linear(width, n_markers)

        def encode(self, values, mask):
            require(values.shape == mask.shape and values.ndim == 2
                    and values.shape[1] == n_markers and mask.dtype == torch.bool,
                    "Mask/encoder shape or dtype mismatch")
            require(bool(torch.isfinite(values).all()), "Nonfinite encoder input")
            visible = values.masked_fill(mask, 0)
            return self.encoder(torch.cat([visible, mask.to(values.dtype)], dim=1))

        def forward(self, values, mask):
            return self.decoder(self.encode(values, mask))

    return MaskedProteinModel()


def masked_mse(prediction, target, mask):
    import torch
    require(prediction.shape == target.shape == mask.shape and mask.dtype == torch.bool
            and bool(mask.any()), "Invalid reconstruction mask")
    require(bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(target).all()),
            "Nonfinite reconstruction")
    return ((prediction[mask]-target[mask])**2).mean()
