"""Unified training loop for standard DIP and DRP-accelerated DIP."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn

from models import DRPWrapper, LightweightGenerator, UNet
from models.generators import LightweightGeneratorSpec
from models.unet import UNetSpec
from utils.metrics import MetricsTracker, compute_metrics

from .losses import CombinedLoss

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for a single AS-DIP run."""

    mode: str = "as_dip"
    backbone: str = "unet"
    input_channels: int = 32
    output_channels: int = 1
    activation: str = "mish"
    norm: str = "batch"
    learning_rate: float = 1e-2
    latent_learning_rate: float = 1e-2
    adapter_learning_rate: float = 1e-3
    iterations: int = 1500
    tv_weight: float = 0.05
    tv_mode: str = "l1"
    log_interval: int = 100
    exp_smoothing: float = 0.99
    reg_noise_std: float = 0.0
    train_norm_layers: bool = True
    train_output_adapter: bool = True
    unet_features: tuple[int, ...] = (64, 128, 256, 512)
    lightweight_channels: tuple[int, ...] = (128, 128, 64, 64)
    pad_border: int = 0


@dataclass
class ExperimentArtifacts:
    """Outputs from a denoising experiment."""

    denoised: np.ndarray
    residual: np.ndarray
    best_iteration: int
    best_metrics: Dict[str, float]
    history: Dict[str, Any]
    elapsed_seconds: float
    config: Dict[str, Any] = field(default_factory=dict)


class ASDIPTrainer:
    """Unified trainer for DIP and DRP seismic denoising."""

    def __init__(self, config: TrainerConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.loss_fn = CombinedLoss(tv_weight=config.tv_weight, tv_mode=config.tv_mode)

    def _build_backbone(self) -> nn.Module:
        if self.config.backbone == "unet":
            spec = UNetSpec(
                in_channels=self.config.input_channels,
                out_channels=self.config.output_channels,
                features=self.config.unet_features,
                activation=self.config.activation,
                norm=self.config.norm,
                pad_border=self.config.pad_border,
            )
            return UNet(spec)
        if self.config.backbone == "lightweight":
            spec = LightweightGeneratorSpec(
                in_channels=self.config.input_channels,
                out_channels=self.config.output_channels,
                hidden_channels=self.config.lightweight_channels,
                activation=self.config.activation,
                norm=self.config.norm,
            )
            return LightweightGenerator(spec)
        raise ValueError(f"Unsupported backbone: {self.config.backbone}")

    def _build_model(self) -> nn.Module:
        backbone = self._build_backbone()
        if self.config.mode == "standard_dip":
            return backbone.to(self.device)
        if self.config.mode in {"drp_dip", "as_dip"}:
            return DRPWrapper(
                backbone=backbone,
                train_norm_layers=self.config.train_norm_layers,
                train_output_adapter=self.config.train_output_adapter,
                out_channels=self.config.output_channels,
            ).to(self.device)
        raise ValueError(f"Unsupported mode: {self.config.mode}")

    def _prepare_latent(self, shape: tuple[int, int]) -> torch.Tensor:
        height, width = shape
        latent = torch.randn(
            1,
            self.config.input_channels,
            height,
            width,
            device=self.device,
            requires_grad=(self.config.mode in {"drp_dip", "as_dip"}),
        )
        return latent

    def _build_optimizer(self, model: nn.Module, latent: torch.Tensor) -> torch.optim.Optimizer:
        if self.config.mode == "standard_dip":
            return torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        parameter_groups = [{"params": [latent], "lr": self.config.latent_learning_rate}]
        if isinstance(model, DRPWrapper):
            trainable = list(model.trainable_parameters())
            if trainable:
                parameter_groups.append({"params": trainable, "lr": self.config.adapter_learning_rate})
        return torch.optim.Adam(parameter_groups, lr=self.config.latent_learning_rate)

    def run(
        self,
        noisy: np.ndarray,
        clean: Optional[np.ndarray] = None,
        output_dir: Optional[Path] = None,
    ) -> ExperimentArtifacts:
        """Run denoising on a single seismic section."""
        noisy_tensor = torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(0).to(self.device)
        clean_tensor = None if clean is None else torch.from_numpy(clean).float().unsqueeze(0).unsqueeze(0).to(self.device)

        model = self._build_model()
        latent = self._prepare_latent(noisy.shape)
        optimizer = self._build_optimizer(model, latent)
        tracker = MetricsTracker()

        best_output = None
        best_score = -float("inf")
        smoothed_output = None
        saved_latent = latent.detach().clone()
        start_time = time.time()

        for iteration in range(1, self.config.iterations + 1):
            optimizer.zero_grad(set_to_none=True)

            current_latent = latent
            if self.config.mode == "standard_dip" and self.config.reg_noise_std > 0.0:
                current_latent = saved_latent + torch.randn_like(saved_latent) * self.config.reg_noise_std

            prediction = model(current_latent)
            if prediction.shape[-2:] != noisy_tensor.shape[-2:]:
                prediction = torch.nn.functional.interpolate(
                    prediction,
                    size=noisy_tensor.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )

            if smoothed_output is None:
                smoothed_output = prediction.detach()
            else:
                smoothed_output = (
                    self.config.exp_smoothing * smoothed_output
                    + (1.0 - self.config.exp_smoothing) * prediction.detach()
                )

            loss, loss_terms = self.loss_fn(prediction, noisy_tensor)
            loss.backward()
            optimizer.step()

            elapsed = time.time() - start_time
            output_np = prediction.detach().squeeze().cpu().numpy()
            smooth_np = smoothed_output.squeeze().cpu().numpy()
            metrics = compute_metrics(output_np=smooth_np, noisy_np=noisy, clean_np=clean)
            tracker.update(iteration=iteration, elapsed_seconds=elapsed, loss_terms=loss_terms, metrics=metrics)

            model_score = metrics.get("psnr", -loss_terms["total"])
            if model_score > best_score:
                best_score = model_score
                best_output = smooth_np.copy()
                tracker.best_iteration = iteration
                tracker.best_metrics = metrics.copy()

            if iteration % self.config.log_interval == 0 or iteration == 1 or iteration == self.config.iterations:
                LOGGER.info(
                    "Iter %d/%d | loss %.6f | mse %.6f | tv %.6f | snr %.2f | psnr %s | time %.2fs",
                    iteration,
                    self.config.iterations,
                    loss_terms["total"],
                    loss_terms["mse"],
                    loss_terms["tv"],
                    metrics.get("snr", float("nan")),
                    f"{metrics['psnr']:.2f}" if "psnr" in metrics else "N/A",
                    elapsed,
                )

        elapsed = time.time() - start_time
        if best_output is None:
            best_output = prediction.detach().squeeze().cpu().numpy()

        residual = noisy - best_output
        history = tracker.to_dict()
        history["config"] = asdict(self.config)

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_dir / "denoised.npy", best_output.astype(np.float32))
            np.save(output_dir / "residual.npy", residual.astype(np.float32))
            with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)

        return ExperimentArtifacts(
            denoised=best_output.astype(np.float32),
            residual=residual.astype(np.float32),
            best_iteration=tracker.best_iteration,
            best_metrics=tracker.best_metrics,
            history=history,
            elapsed_seconds=elapsed,
            config=asdict(self.config),
        )
