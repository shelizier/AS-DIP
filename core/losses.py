"""Loss functions for self-supervised seismic denoising."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn


def total_variation_loss(image: torch.Tensor, mode: str = "l1") -> torch.Tensor:
    """Compute isotropic total variation regularization."""
    diff_t = image[:, :, 1:, :] - image[:, :, :-1, :]
    diff_x = image[:, :, :, 1:] - image[:, :, :, :-1]
    if mode == "l2":
        value = diff_t.pow(2).mean() + diff_x.pow(2).mean()
    elif mode == "l1":
        value = diff_t.abs().mean() + diff_x.abs().mean()
    else:
        raise ValueError(f"Unsupported TV mode: {mode}")
    return value


@dataclass
class LossBreakdown:
    """Structured loss values for logging."""

    total: float
    mse: float
    tv: float


class CombinedLoss(nn.Module):
    """MSE plus total variation regularization."""

    def __init__(self, tv_weight: float = 0.05, tv_mode: str = "l1") -> None:
        super().__init__()
        self.reconstruction = nn.MSELoss()
        self.tv_weight = tv_weight
        self.tv_mode = tv_mode

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        mse = self.reconstruction(prediction, target)
        tv = total_variation_loss(prediction, mode=self.tv_mode)
        total = mse + self.tv_weight * tv
        metrics = {"total": float(total.item()), "mse": float(mse.item()), "tv": float((self.tv_weight * tv).item())}
        return total, metrics
