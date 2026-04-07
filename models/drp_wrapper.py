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
        self.train_norm_layers = train_norm_layers
        self.train_output_adapter = train_output_adapter
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

    def backbone_stage_names(self) -> List[str]:
        """Return ordered backbone stage names for progressive unfreezing."""
        names: List[str] = []
        for name, module in self.backbone.named_children():
            if isinstance(module, nn.ModuleList):
                for index, _ in enumerate(module):
                    names.append(f"{name}.{index}")
            else:
                names.append(name)
        return names

    def _resolve_module(self, module_path: str) -> nn.Module:
        module: nn.Module = self.backbone
        for token in module_path.split("."):
            if token.isdigit():
                module = module[int(token)]  # type: ignore[index]
            else:
                module = getattr(module, token)
        return module

    def _set_module_requires_grad(self, module: nn.Module, requires_grad: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad

    def freeze_all(self) -> None:
        """Freeze the entire wrapper so training can be re-enabled stage by stage."""
        self._set_module_requires_grad(self.backbone, False)
        self._set_module_requires_grad(self.output_adapter, False)

    def unfreeze_layer(self, layer_type: str) -> None:
        """Unfreeze a logical layer group or specific backbone stage."""
        if layer_type == "adapter":
            if self.train_output_adapter:
                self._set_module_requires_grad(self.output_adapter, True)
            return

        if layer_type == "norm":
            if not self.train_norm_layers:
                return
            for module in self.backbone.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)) and getattr(module, "affine", False):
                    if module.weight is not None:
                        module.weight.requires_grad = True
                    if module.bias is not None:
                        module.bias.requires_grad = True
            return

        if layer_type == "backbone":
            self._set_module_requires_grad(self.backbone, True)
            return

        if layer_type.startswith("backbone:"):
            module_path = layer_type.split(":", maxsplit=1)[1]
            self._set_module_requires_grad(self._resolve_module(module_path), True)
            return

        if layer_type == "all":
            self._set_module_requires_grad(self.backbone, True)
            self.unfreeze_layer("adapter")
            return

        raise ValueError(f"Unsupported layer_type: {layer_type}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(inputs)
        return self.output_adapter(outputs)
