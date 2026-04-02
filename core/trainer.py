"""Unified training loop for standard DIP and DRP-accelerated DIP."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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
    learning_rate: float = 1e-3
    seed_learning_rate: float = 1e-2
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
    warmup_ratio: float = 0.2
    adapter_ratio: float = 0.3
    local_similarity_threshold: float = 0.95
    local_similarity_window: int = 9
    as_phase3_tv_scale: float = 0.2
    as_phase3_backbone_lr_scale: float = 0.3


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

        parameter_groups = [{"params": [latent], "lr": self.config.seed_learning_rate}]
        if isinstance(model, DRPWrapper):
            trainable = list(model.trainable_parameters())
            if trainable:
                parameter_groups.append({"params": trainable, "lr": self.config.adapter_learning_rate})
        return torch.optim.Adam(parameter_groups, lr=self.config.seed_learning_rate)

    def _phase_boundaries(self) -> tuple[int, int]:
        warmup_end = max(1, int(self.config.iterations * self.config.warmup_ratio))
        adapter_end = max(warmup_end + 1, int(self.config.iterations * (self.config.warmup_ratio + self.config.adapter_ratio)))
        adapter_end = min(adapter_end, self.config.iterations)
        return warmup_end, adapter_end

    def _configure_wrapper_training(
        self,
        model: DRPWrapper,
        latent: torch.Tensor,
        phase: int,
        backbone_stage_count: int = 0,
    ) -> torch.optim.Optimizer:
        model.freeze_all()
        seed_learning_rate = self.config.seed_learning_rate
        backbone_learning_rate = self.config.learning_rate

        if phase >= 2:
            model.unfreeze_layer("adapter")
            model.unfreeze_layer("norm")

        if phase >= 3:
            seed_learning_rate *= 0.1
            if self.config.mode == "as_dip":
                backbone_learning_rate *= self.config.as_phase3_backbone_lr_scale
            stage_names = model.backbone_stage_names()
            stage_count = min(backbone_stage_count, len(stage_names))
            for stage_name in stage_names[:stage_count]:
                model.unfreeze_layer(f"backbone:{stage_name}")

        parameter_groups = [{"params": [latent], "lr": seed_learning_rate}]

        adapter_parameters = [parameter for parameter in model.output_adapter.parameters() if parameter.requires_grad]
        norm_parameter_ids = {
            id(parameter)
            for module in model.backbone.modules()
            if isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)) and getattr(module, "affine", False)
            for parameter in (module.weight, module.bias)
            if parameter is not None and parameter.requires_grad
        }
        backbone_trainable = list(model.trainable_backbone_parameters())
        norm_parameters = [parameter for parameter in backbone_trainable if id(parameter) in norm_parameter_ids]
        adapter_and_norm = adapter_parameters + norm_parameters
        if adapter_and_norm:
            parameter_groups.append({"params": adapter_and_norm, "lr": self.config.adapter_learning_rate})

        backbone_parameters = [parameter for parameter in backbone_trainable if id(parameter) not in norm_parameter_ids]
        if backbone_parameters:
            parameter_groups.append({"params": backbone_parameters, "lr": backbone_learning_rate})

        return torch.optim.Adam(parameter_groups)

    def _effective_tv_weight(self, phase: int) -> float:
        if self.config.mode == "as_dip" and phase >= 3:
            return self.config.tv_weight * self.config.as_phase3_tv_scale
        return self.config.tv_weight

    def _phase_for_iteration(self, iteration: int, warmup_end: int, adapter_end: int) -> int:
        if iteration <= warmup_end:
            return 1
        if iteration <= adapter_end:
            return 2
        return 3

    def _target_backbone_stage_count(self, iteration: int, adapter_end: int, stage_total: int) -> int:
        if stage_total == 0:
            return 0
        remaining = max(self.config.iterations - adapter_end, 1)
        phase_progress = min(max(iteration - adapter_end, 0), remaining)
        unlocked = math.ceil((phase_progress / remaining) * stage_total)
        return max(1, unlocked)

    def _compute_local_similarity(self, source: torch.Tensor, residual: torch.Tensor) -> float:
        kernel = max(3, self.config.local_similarity_window)
        if kernel % 2 == 0:
            kernel += 1
        padding = kernel // 2

        def pooled(value: torch.Tensor) -> torch.Tensor:
            return F.avg_pool2d(value, kernel_size=kernel, stride=1, padding=padding)

        mean_source = pooled(source)
        mean_residual = pooled(residual)
        centered_source = source - mean_source
        centered_residual = residual - mean_residual
        covariance = pooled(centered_source * centered_residual)
        variance_source = pooled(centered_source.pow(2))
        variance_residual = pooled(centered_residual.pow(2))
        similarity = covariance / torch.sqrt(variance_source * variance_residual + 1e-8)
        return float(similarity.abs().mean().item())

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
        warmup_end, adapter_end = self._phase_boundaries()
        current_phase = 1
        backbone_stage_count = 0
        phase_locked = False
        leakage_flagged = False
        backbone_stage_names: list[str] = []

        if isinstance(model, DRPWrapper):
            backbone_stage_names = model.backbone_stage_names()
            optimizer = self._configure_wrapper_training(model, latent, phase=1, backbone_stage_count=0)
            LOGGER.info(
                "Phased training enabled | warmup=%d iter | adapter=%d iter | finetune=%d iter",
                warmup_end,
                max(adapter_end - warmup_end, 0),
                max(self.config.iterations - adapter_end, 0),
            )

        best_output = None
        best_score = -float("inf")
        smoothed_output = None
        saved_latent = latent.detach().clone()
        start_time = time.time()

        for iteration in range(1, self.config.iterations + 1):
            if isinstance(model, DRPWrapper):
                scheduled_phase = self._phase_for_iteration(iteration, warmup_end, adapter_end)
                target_phase = current_phase if phase_locked else scheduled_phase
                if target_phase != current_phase:
                    current_phase = target_phase
                    if current_phase == 3:
                        backbone_stage_count = self._target_backbone_stage_count(
                            iteration, adapter_end, len(backbone_stage_names)
                        )
                    optimizer = self._configure_wrapper_training(
                        model,
                        latent,
                        phase=current_phase,
                        backbone_stage_count=backbone_stage_count,
                    )
                    LOGGER.info("Entering Phase %d at iter %d", current_phase, iteration)

                if current_phase == 3 and not phase_locked:
                    target_stage_count = self._target_backbone_stage_count(
                        iteration, adapter_end, len(backbone_stage_names)
                    )
                    if target_stage_count > backbone_stage_count:
                        backbone_stage_count = target_stage_count
                        optimizer = self._configure_wrapper_training(
                            model,
                            latent,
                            phase=current_phase,
                            backbone_stage_count=backbone_stage_count,
                        )
                        LOGGER.info(
                            "Progressively unfroze backbone stages: %d/%d at iter %d",
                            backbone_stage_count,
                            len(backbone_stage_names),
                            iteration,
                        )

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

            self.loss_fn.tv_weight = self._effective_tv_weight(current_phase)
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
                if isinstance(model, DRPWrapper) and smoothed_output is not None and current_phase >= 2:
                    residual_tensor = noisy_tensor - smoothed_output
                    local_similarity = self._compute_local_similarity(noisy_tensor, residual_tensor)
                    if (not phase_locked) and local_similarity > self.config.local_similarity_threshold:
                        phase_locked = True
                        leakage_flagged = True
                        LOGGER.warning(
                            "Signal leakage risk detected at iter %d | local similarity %.4f exceeds %.4f | locking unfreeze progress",
                            iteration,
                            local_similarity,
                            self.config.local_similarity_threshold,
                        )
                LOGGER.info(
                    "Iter %d/%d | phase %d | loss %.6f | mse %.6f | tv %.6f | snr %.2f | psnr %s | time %.2fs",
                    iteration,
                    self.config.iterations,
                    current_phase,
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
        history["phase_locked"] = phase_locked
        history["signal_leakage_risk"] = leakage_flagged

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
