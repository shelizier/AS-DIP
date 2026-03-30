"""Lightweight generator variants for accelerated seismic DIP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .activations import get_activation


@dataclass(frozen=True)
class LightweightGeneratorSpec:
    """Configuration for a compact decoder generator."""

    in_channels: int = 32
    out_channels: int = 1
    hidden_channels: Sequence[int] = (128, 128, 64, 64)
    activation: str = "leaky_relu"
    norm: str = "batch"


def _make_norm(norm: str, channels: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    raise ValueError(f"Unsupported normalization: {norm}")


class LightweightGenerator(nn.Module):
    """Compact fully convolutional generator for faster DRP experiments."""

    def __init__(self, spec: LightweightGeneratorSpec) -> None:
        super().__init__()
        layers = []
        in_channels = spec.in_channels
        for hidden in spec.hidden_channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
                    _make_norm(spec.norm, hidden),
                    get_activation(spec.activation),
                ]
            )
            in_channels = hidden
        layers.append(nn.Conv2d(in_channels, spec.out_channels, kernel_size=1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
