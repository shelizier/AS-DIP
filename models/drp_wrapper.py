"""Deep Random Projector wrapper utilities."""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class DRPWrapper(nn.Module):
    """Wrap a generator and expose lightweight trainable parameters for DRP."""

    def __init__(
        self,
        backbone: nn.Module,
        train_norm_layers: bool = True,
        train_output_adapter: bool = True,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.output_adapter = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
        if not train_output_adapter:
            for parameter in self.output_adapter.parameters():
                parameter.requires_grad = False

        self._freeze_backbone(train_norm_layers=train_norm_layers)

    def _freeze_backbone(self, train_norm_layers: bool) -> None:
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                requires_grad = train_norm_layers and getattr(module, "affine", False)
                if module.weight is not None:
                    module.weight.requires_grad = requires_grad
                if module.bias is not None:
                    module.bias.requires_grad = requires_grad

        for name, parameter in self.backbone.named_parameters():
            if "weight" in name or "bias" in name:
                parameter.requires_grad = False

        if train_norm_layers:
            for module in self.backbone.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)) and getattr(module, "affine", False):
                    if module.weight is not None:
                        module.weight.requires_grad = True
                    if module.bias is not None:
                        module.bias.requires_grad = True

    def trainable_backbone_parameters(self) -> List[nn.Parameter]:
        """Return the backbone parameters that remain trainable."""
        return [parameter for parameter in self.backbone.parameters() if parameter.requires_grad]

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        """Return all trainable wrapper parameters."""
        parameters: List[nn.Parameter] = list(self.trainable_backbone_parameters())
        parameters.extend(parameter for parameter in self.output_adapter.parameters() if parameter.requires_grad)
        return parameters

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(inputs)
        return self.output_adapter(outputs)
