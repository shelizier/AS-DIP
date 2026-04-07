"""Loss functions for self-supervised seismic denoising."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def _reduce_difference(difference: torch.Tensor, mode: str = "l1") -> torch.Tensor:
    if mode == "l2":
        return difference.pow(2).mean()
    if mode == "l1":
        return difference.abs().mean()
    raise ValueError(f"Unsupported reduction mode: {mode}")


def total_variation_loss(
    image: torch.Tensor,
    mode: str = "l1",
    weight_t: float = 1.0,
    weight_x: float = 1.0,
) -> torch.Tensor:
    """Compute anisotropic total variation regularization."""
    diff_t = image[:, :, 1:, :] - image[:, :, :-1, :]
    diff_x = image[:, :, :, 1:] - image[:, :, :, :-1]
    return weight_t * _reduce_difference(diff_t, mode=mode) + weight_x * _reduce_difference(diff_x, mode=mode)


def gradient_consistency_loss(prediction: torch.Tensor, target: torch.Tensor, mode: str = "l1") -> torch.Tensor:
    pred_t = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_t = target[:, :, 1:, :] - target[:, :, :-1, :]
    pred_x = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    return _reduce_difference(pred_t - target_t, mode=mode) + _reduce_difference(pred_x - target_x, mode=mode)


class CombinedLoss(nn.Module):
    """Configurable reconstruction loss with TV and gradient regularization."""

    def __init__(
        self,
        tv_weight: float = 0.05,
        tv_mode: str = "l1",
        mse_weight: float = 1.0,
        l1_weight: float = 0.0,
        tv_weight_t: float = 1.0,
        tv_weight_x: float = 1.0,
        gradient_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.tv_weight = tv_weight
        self.tv_mode = tv_mode
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.tv_weight_t = tv_weight_t
        self.tv_weight_x = tv_weight_x
        self.gradient_weight = gradient_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
        mse = self.mse(prediction, target)
        l1 = self.l1(prediction, target)
        reconstruction = self.mse_weight * mse + self.l1_weight * l1
        tv = total_variation_loss(
            prediction,
            mode=self.tv_mode,
            weight_t=self.tv_weight_t,
            weight_x=self.tv_weight_x,
        )
        gradient = gradient_consistency_loss(prediction, target, mode=self.tv_mode)
        total = reconstruction + self.tv_weight * tv + self.gradient_weight * gradient
        metrics = {
            "total": float(total.item()),
            "mse": float((self.mse_weight * mse).item()),
            "l1": float((self.l1_weight * l1).item()),
            "tv": float((self.tv_weight * tv).item()),
            "gradient": float((self.gradient_weight * gradient).item()),
        }
        return total, metrics
