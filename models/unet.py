"""Reusable U-Net generator for DIP-based seismic denoising."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .activations import get_activation


@dataclass(frozen=True)
class UNetSpec:
    """Configuration for the U-Net backbone."""

    in_channels: int = 32
    out_channels: int = 1
    features: Sequence[int] = (64, 128, 256, 512)
    activation: str = "mish"
    norm: str = "batch"
    bilinear: bool = True
    pad_border: int = 0


def _make_norm(norm: str, channels: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    raise ValueError(f"Unsupported normalization: {norm}")


class DoubleConv(nn.Module):
    """Two-layer convolutional block."""

    def __init__(self, in_channels: int, out_channels: int, activation: str, norm: str) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(norm, out_channels),
            get_activation(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(norm, out_channels),
            get_activation(activation),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class Down(nn.Module):
    """Downsampling block."""

    def __init__(self, in_channels: int, out_channels: int, activation: str, norm: str) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels, activation=activation, norm=norm),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class Up(nn.Module):
    """Upsampling block with skip connection fusion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str,
        norm: str,
        bilinear: bool,
    ) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            mid_channels = in_channels // 2
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            mid_channels = in_channels
        self.conv = DoubleConv(in_channels, out_channels, activation=activation, norm=norm)
        self.bilinear = bilinear
        self.mid_channels = mid_channels

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(inputs)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output projection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class UNet(nn.Module):
    """U-Net backbone adapted for seismic DIP."""

    def __init__(self, spec: UNetSpec) -> None:
        super().__init__()
        self.spec = spec
        features: List[int] = list(spec.features)
        factor = 2 if spec.bilinear else 1

        self.input_block = DoubleConv(spec.in_channels, features[0], spec.activation, spec.norm)
        self.down_blocks = nn.ModuleList(
            [
                Down(features[idx], features[idx + 1], spec.activation, spec.norm)
                for idx in range(len(features) - 1)
            ]
        )
        bottleneck_channels = features[-1]
        self.bottom = Down(features[-1], (2 * features[-1]) // factor, spec.activation, spec.norm)

        reversed_features = list(reversed(features))
        current_channels = (2 * bottleneck_channels) // factor
        up_blocks = []
        for feature in reversed_features:
            up_blocks.append(
                Up(current_channels + feature, feature // factor if feature == bottleneck_channels else feature,
                   activation=spec.activation, norm=spec.norm, bilinear=spec.bilinear)
            )
            current_channels = feature // factor if feature == bottleneck_channels else feature
        self.up_blocks = nn.ModuleList(up_blocks)
        self.output_block = OutConv(features[0], spec.out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        border = self.spec.pad_border
        x = inputs
        if border > 0:
            x = F.pad(x, (border, border, border, border), mode="reflect")

        skip_features = [self.input_block(x)]
        current = skip_features[0]
        for block in self.down_blocks:
            current = block(current)
            skip_features.append(current)

        current = self.bottom(current)
        for block, skip in zip(self.up_blocks, reversed(skip_features)):
            current = block(current, skip)

        outputs = self.output_block(current)
        if border > 0:
            outputs = outputs[:, :, border:-border, border:-border]
        return outputs
