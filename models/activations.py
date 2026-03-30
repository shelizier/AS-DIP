"""Activation utilities used by AS-DIP models."""

from __future__ import annotations

import torch
from torch import nn


class Mish(nn.Module):
    """Mish activation."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * torch.tanh(nn.functional.softplus(inputs))


def get_activation(name: str, negative_slope: float = 0.2) -> nn.Module:
    """Return an activation module by name."""
    normalized = name.lower()
    if normalized == "mish":
        return Mish()
    if normalized == "leaky_relu":
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
    raise ValueError(f"Unsupported activation: {name}")
