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
    learning_rate: float = 1e-2
    latent_learning_rate: float = 1e-2
    adapter_learning_rate: float = 1e-3
    iterations: int = 1000
    tv_weight: float = 0.05
    tv_mode: str = "l1"
    log_interval: int = 100
    exp_smoothing: float = 0.99
    reg_noise_std: float = 0.015
    train_norm_layers: bool = True
    train_output_adapter: bool = True
    unet_features: tuple[int, ...] = (64, 128, 256, 512)
    lightweight_channels: tuple[int, ...] = (128, 128, 64, 64)
    pad_border: int = 0
    phased_optimization: bool = False
    phase1_fraction: float = 0.2
    phase2_fraction: float = 0.3
    phase3_latent_lr_scale: float = 0.2
    backbone_learning_rate: float = 5e-4
    residual_similarity_window: int = 9
    residual_similarity_threshold: float = 0.95
    residual_backbone_lr_scale: float = 0.5
    use_structured_latent: bool = True
    latent_coord_channels: int = 2
    mse_weight: float = 1.0
    l1_weight: float = 0.1
    tv_weight_t: float = 0.1
    tv_weight_x: float = 1.5
    gradient_weight: float = 0.0
    reg_noise_std_phase1: Optional[float] = None
    reg_noise_std_phase2: Optional[float] = None
    reg_noise_std_phase3: Optional[float] = None
    lr_decay_end_factor: float = 0.25
    ssim_weight: float = 0.2
    ortho_weight: float = 0.25
    ortho_window: int = 9
    max_allowed_snr: float = 13

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
        self.loss_fn = CombinedLoss(
            tv_weight=config.tv_weight,
            tv_mode=config.tv_mode,
            mse_weight=config.mse_weight,
            l1_weight=config.l1_weight,
            tv_weight_t=config.tv_weight_t,
            tv_weight_x=config.tv_weight_x,
            gradient_weight=config.gradient_weight,
            ssim_weight=config.ssim_weight,
            ortho_weight=config.ortho_weight,
            ortho_window=config.ortho_window
        )

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

    def _prepare_latent(self, shape: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor | None]:
        height, width = shape
        trainable_channels = self.config.input_channels
        fixed_latent = None

        if self.config.mode == "as_dip" and self.config.use_structured_latent and self.config.input_channels > 2:
            coord_channels = min(self.config.latent_coord_channels, self.config.input_channels - 1)
            trainable_channels = self.config.input_channels - coord_channels
            y_coords = torch.linspace(-1.0, 1.0, steps=height, device=self.device).view(1, 1, height, 1)
            x_coords = torch.linspace(-1.0, 1.0, steps=width, device=self.device).view(1, 1, 1, width)
            coord_tensors = [y_coords.expand(1, 1, height, width), x_coords.expand(1, 1, height, width)]
            fixed_latent = torch.cat(coord_tensors[:coord_channels], dim=1)

        latent = torch.randn(
            1,
            trainable_channels,
            height,
            width,
            device=self.device,
            requires_grad=(self.config.mode in {"drp_dip", "as_dip"}),
        )
        return latent, fixed_latent

    @staticmethod
    def _compose_latent(trainable_latent: torch.Tensor, fixed_latent: torch.Tensor | None) -> torch.Tensor:
        if fixed_latent is None:
            return trainable_latent
        return torch.cat([trainable_latent, fixed_latent], dim=1)

    def _build_optimizer(
        self,
        model: nn.Module,
        latent: torch.Tensor,
        *,
        latent_lr: float,
        adapter_lr: Optional[float] = None,
        backbone_lr: Optional[float] = None,
    ) -> torch.optim.Optimizer:
        if self.config.mode == "standard_dip":
            return torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        parameter_groups = [{"params": [latent], "lr": latent_lr}]
        if isinstance(model, DRPWrapper):
            norm_parameters = list(model.trainable_norm_parameters())
            adapter_parameters = list(model.trainable_adapter_parameters())
            backbone_parameters = list(model.trainable_non_norm_backbone_parameters())
            if norm_parameters or adapter_parameters:
                parameter_groups.append(
                    {
                        "params": [*norm_parameters, *adapter_parameters],
                        "lr": adapter_lr if adapter_lr is not None else self.config.adapter_learning_rate,
                    }
                )
            if backbone_parameters:
                parameter_groups.append(
                    {
                        "params": backbone_parameters,
                        "lr": backbone_lr if backbone_lr is not None else self.config.backbone_learning_rate,
                    }
                )
        return torch.optim.Adam(parameter_groups, lr=latent_lr)

    def _reg_noise_std_for_phase(self, phase_name: str) -> float:
        if self.config.mode != "as_dip" or not self.config.phased_optimization:
            return self.config.reg_noise_std
        if phase_name == "phase1_warmup":
            return (
                self.config.reg_noise_std_phase1
                if self.config.reg_noise_std_phase1 is not None
                else self.config.reg_noise_std
            )
        if phase_name == "phase2_adapter":
            return (
                self.config.reg_noise_std_phase2
                if self.config.reg_noise_std_phase2 is not None
                else self.config.reg_noise_std
            )
        if phase_name == "phase3_backbone":
            return (
                self.config.reg_noise_std_phase3
                if self.config.reg_noise_std_phase3 is not None
                else self.config.reg_noise_std
            )
        return self.config.reg_noise_std

    def _phase_boundaries(self) -> tuple[int, int]:
        iterations = self.config.iterations
        phase1_end = max(1, int(iterations * self.config.phase1_fraction))
        phase2_end = max(phase1_end + 1, int(iterations * (self.config.phase1_fraction + self.config.phase2_fraction)))
        phase2_end = min(max(phase2_end, phase1_end + 1), iterations)
        return phase1_end, phase2_end

    def _local_similarity(self, first: torch.Tensor, second: torch.Tensor) -> float:
        window = self.config.residual_similarity_window
        if window % 2 == 0:
            window += 1
        mean_first = F.avg_pool2d(first, kernel_size=window, stride=1, padding=window // 2)
        mean_second = F.avg_pool2d(second, kernel_size=window, stride=1, padding=window // 2)
        centered_first = first - mean_first
        centered_second = second - mean_second
        covariance = F.avg_pool2d(centered_first * centered_second, kernel_size=window, stride=1, padding=window // 2)
        variance_first = F.avg_pool2d(centered_first.pow(2), kernel_size=window, stride=1, padding=window // 2)
        variance_second = F.avg_pool2d(centered_second.pow(2), kernel_size=window, stride=1, padding=window // 2)
        similarity = covariance / torch.sqrt(variance_first * variance_second + 1e-8)
        return float(similarity.abs().mean().item())

    def _resolve_asdip_phase(
        self,
        iteration: int,
        model: DRPWrapper,
        max_backbone_groups: int,
        backbone_lr_scale: float,
    ) -> Dict[str, Any]:
        phase1_end, phase2_end = self._phase_boundaries()
        total_groups = len(model.backbone_progression_groups())

        if iteration <= phase1_end:
            return {
                "name": "phase1_warmup",
                "train_norm_layers": False,
                "train_output_adapter": False,
                "backbone_groups": 0,
                "latent_lr": self.config.latent_learning_rate,
                "adapter_lr": self.config.adapter_learning_rate,
                "backbone_lr": self.config.backbone_learning_rate * backbone_lr_scale,
            }

        if iteration <= phase2_end:
            return {
                "name": "phase2_adapter",
                "train_norm_layers": True,
                "train_output_adapter": self.config.train_output_adapter,
                "backbone_groups": 0,
                "latent_lr": self.config.latent_learning_rate,
                "adapter_lr": self.config.adapter_learning_rate,
                "backbone_lr": self.config.backbone_learning_rate * backbone_lr_scale,
            }

        phase3_total = max(self.config.iterations - phase2_end, 1)
        phase3_progress = max(iteration - phase2_end - 1, 0)
        requested_groups = 0
        if total_groups > 0:
            if phase3_total == 1:
                requested_groups = total_groups
            else:
                requested_groups = 1 + math.floor(
                    phase3_progress * max(total_groups - 1, 0) / max(phase3_total - 1, 1)
                )
        backbone_groups = min(requested_groups, max_backbone_groups)
        return {
            "name": "phase3_backbone",
            "train_norm_layers": True,
            "train_output_adapter": self.config.train_output_adapter,
            "backbone_groups": backbone_groups,
            "latent_lr": self.config.latent_learning_rate * self.config.phase3_latent_lr_scale,
            "adapter_lr": self.config.adapter_learning_rate,
            "backbone_lr": self.config.backbone_learning_rate * backbone_lr_scale,
        }

    def _resolve_training_state(
        self,
        model: nn.Module,
        *,
        iteration: int,
        max_backbone_groups: int,
        backbone_lr_scale: float,
    ) -> Dict[str, Any]:
        if self.config.mode == "standard_dip" or not isinstance(model, DRPWrapper):
            return {"name": "standard_dip", "backbone_groups": 0}

        if self.config.phased_optimization and self.config.mode == "as_dip":
            return self._resolve_asdip_phase(
                iteration=iteration,
                model=model,
                max_backbone_groups=max_backbone_groups,
                backbone_lr_scale=backbone_lr_scale,
            )

        return {
            "name": self.config.mode,
            "train_norm_layers": self.config.train_norm_layers,
            "train_output_adapter": self.config.train_output_adapter,
            "backbone_groups": 0,
            "latent_lr": self.config.latent_learning_rate,
            "adapter_lr": self.config.adapter_learning_rate,
            "backbone_lr": self.config.backbone_learning_rate * backbone_lr_scale,
        }

    def _apply_training_state(
        self,
        model: nn.Module,
        latent: torch.Tensor,
        phase_state: Dict[str, Any],
    ) -> torch.optim.Optimizer:
        if self.config.mode == "standard_dip" or not isinstance(model, DRPWrapper):
            return self._build_optimizer(model, latent, latent_lr=self.config.learning_rate)

        model.configure_trainable_state(
            train_norm_layers=phase_state["train_norm_layers"],
            train_output_adapter=phase_state["train_output_adapter"],
            backbone_groups_to_unfreeze=phase_state["backbone_groups"],
        )
        return self._build_optimizer(
            model,
            latent,
            latent_lr=phase_state["latent_lr"],
            adapter_lr=phase_state["adapter_lr"],
            backbone_lr=phase_state["backbone_lr"],
        )

    def _lr_decay_iteration_bounds(self) -> tuple[int, int]:
        total = self.config.iterations
        if self.config.mode != "as_dip" or not self.config.phased_optimization:
            return total + 1, total
        _, phase2_end = self._phase_boundaries()
        start = phase2_end + 1
        end = total
        return start, end

    @staticmethod
    def _cosine_lr_multiplier(progress: float, end_factor: float) -> float:
        progress = min(max(progress, 0.0), 1.0)
        return end_factor + (1.0 - end_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _nominal_param_group_lrs(self, model: nn.Module, phase_state: Dict[str, Any]) -> list[float]:
        if self.config.mode == "standard_dip" or not isinstance(model, DRPWrapper):
            return [self.config.learning_rate]
        latent_lr = float(phase_state["latent_lr"])
        adapter_lr = float(phase_state["adapter_lr"])
        backbone_lr = float(phase_state["backbone_lr"])
        norm_parameters = list(model.trainable_norm_parameters())
        adapter_parameters = list(model.trainable_adapter_parameters())
        backbone_parameters = list(model.trainable_non_norm_backbone_parameters())
        lrs = [latent_lr]
        if norm_parameters or adapter_parameters:
            lrs.append(adapter_lr)
        if backbone_parameters:
            lrs.append(backbone_lr)
        return lrs

    def _apply_optimizer_lr_decay(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        phase_state: Dict[str, Any],
        iteration: int,
    ) -> None:
        if self.config.mode != "as_dip" or not self.config.phased_optimization:
            return
        decay_start, decay_end = self._lr_decay_iteration_bounds()
        if decay_start > decay_end or iteration < decay_start or iteration > decay_end:
            return
        denom = max(decay_end - decay_start, 1)
        progress = (iteration - decay_start) / denom
        mult = self._cosine_lr_multiplier(progress, self.config.lr_decay_end_factor)
        nominal = self._nominal_param_group_lrs(model, phase_state)
        if len(nominal) != len(optimizer.param_groups):
            LOGGER.warning(
                "LR decay skipped | param group count mismatch | groups=%d nominal=%d",
                len(optimizer.param_groups),
                len(nominal),
            )
            return
        for group, lr in zip(optimizer.param_groups, nominal):
            group["lr"] = lr * mult

    def run(
        self,
        noisy: np.ndarray,
        clean: Optional[np.ndarray] = None,
        output_dir: Optional[Path] = None,
    ) -> ExperimentArtifacts:
        noisy_tensor = torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(0).to(self.device)
        model = self._build_model()
        latent, fixed_latent = self._prepare_latent(noisy.shape)
        total_backbone_groups = len(model.backbone_progression_groups()) if isinstance(model, DRPWrapper) else 0
        backbone_lr_scale = 1.0
        unfreeze_hold_until = 0
        held_backbone_groups = total_backbone_groups
        allowed_backbone_groups = total_backbone_groups
        phase_state = self._resolve_training_state(
            model,
            iteration=1,
            max_backbone_groups=allowed_backbone_groups,
            backbone_lr_scale=backbone_lr_scale,
        )
        optimizer = self._apply_training_state(model, latent, phase_state)
        tracker = MetricsTracker()
        phase_history: list[dict[str, Any]] = []

        best_output = None
        best_score = -float("inf")
        smoothed_output = None
        start_time = time.time()
        previous_phase_name = None
        previous_backbone_groups = -1
        optimizer_signature = None

        for iteration in range(1, self.config.iterations + 1):
            if isinstance(model, DRPWrapper):
                allowed_backbone_groups = (
                    held_backbone_groups if iteration <= unfreeze_hold_until else total_backbone_groups
                )
                phase_state = self._resolve_training_state(
                    model,
                    iteration=iteration,
                    max_backbone_groups=allowed_backbone_groups,
                    backbone_lr_scale=backbone_lr_scale,
                )
                next_signature = (
                    phase_state["name"],
                    phase_state.get("backbone_groups"),
                    phase_state.get("latent_lr"),
                    phase_state.get("adapter_lr"),
                    phase_state.get("backbone_lr"),
                    phase_state.get("train_norm_layers"),
                    phase_state.get("train_output_adapter"),
                )
                if next_signature != optimizer_signature:
                    optimizer = self._apply_training_state(model, latent, phase_state)
                    optimizer_signature = next_signature

            if phase_state["name"] != previous_phase_name or phase_state["backbone_groups"] != previous_backbone_groups:
                LOGGER.info(
                    "Training state -> %s | backbone groups=%d",
                    phase_state["name"],
                    phase_state["backbone_groups"],
                )
                phase_history.append(
                    {
                        "iteration": iteration,
                        "phase": phase_state["name"],
                        "backbone_groups": phase_state["backbone_groups"],
                        "latent_lr": phase_state.get("latent_lr"),
                        "adapter_lr": phase_state.get("adapter_lr"),
                        "backbone_lr": phase_state.get("backbone_lr"),
                    }
                )
                previous_phase_name = phase_state["name"]
                previous_backbone_groups = phase_state["backbone_groups"]

            self._apply_optimizer_lr_decay(optimizer, model, phase_state, iteration)

            optimizer.zero_grad(set_to_none=True)

            reg_noise_std = self._reg_noise_std_for_phase(phase_state["name"])
            current_latent = latent
            if reg_noise_std > 0.0:
                current_latent = current_latent + torch.randn_like(current_latent) * reg_noise_std
            current_inputs = self._compose_latent(current_latent, fixed_latent)

            prediction = model(current_inputs)
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
            # 修复：移除完全没有被使用的幽灵变量 output_np
            smooth_np = smoothed_output.squeeze().cpu().numpy()
            metrics = compute_metrics(output_np=smooth_np, noisy_np=noisy, clean_np=clean)
            tracker.update(iteration=iteration, elapsed_seconds=elapsed, loss_terms=loss_terms, metrics=metrics)

            model_score = metrics.get("snr", -loss_terms["total"])
            import math
            has_snr = "snr" in metrics and not math.isnan(metrics["snr"])

            # 核心拦截逻辑：
            # 1. 如果当前 SNR 突破了允许的安全上限，说明极大概率正在发生严重的信号泄露
            # 我们拒绝更新 best_iteration，即使当前的得分在数值上更高
            if has_snr and metrics["snr"] > self.config.max_allowed_snr:
                pass

                # 2. 在安全范围内，只要当前成绩优于历史最高分，就正常更新
            elif model_score > best_score:
                best_score = model_score
                best_output = smooth_np.copy()
                tracker.best_iteration = iteration
                tracker.best_metrics = metrics.copy()

            if iteration % self.config.log_interval == 0 or iteration == 1 or iteration == self.config.iterations:
                residual_similarity = None
                if self.config.mode == "as_dip":
                    residual_tensor = noisy_tensor - prediction.detach()
                    residual_similarity = self._local_similarity(residual_tensor, noisy_tensor)
                    metrics["residual_local_similarity"] = residual_similarity
                    if (
                        phase_state["name"] == "phase3_backbone"
                        and residual_similarity >= self.config.residual_similarity_threshold
                    ):
                        LOGGER.warning(
                            "Signal leakage risk | iter=%d | local similarity=%.4f | reducing backbone lr scale to %.2f and pausing deeper unfreezing",
                            iteration,
                            residual_similarity,
                            self.config.residual_backbone_lr_scale,
                        )
                        backbone_lr_scale = min(backbone_lr_scale, self.config.residual_backbone_lr_scale)
                        held_backbone_groups = phase_state["backbone_groups"]
                        unfreeze_hold_until = max(unfreeze_hold_until, iteration + self.config.log_interval)
                        optimizer_signature = None

                LOGGER.info(
                    "Iter %d/%d | phase %s | loss %.6f | mse %.6f | tv %.6f | snr %.2f | time %.2fs%s",
                    iteration,
                    self.config.iterations,
                    phase_state["name"],
                    loss_terms["total"],
                    loss_terms["mse"],
                    loss_terms["tv"],
                    metrics.get("snr", float("nan")),
                    elapsed,
                    f" | local-sim {residual_similarity:.4f}" if residual_similarity is not None else "",
                )

        elapsed = time.time() - start_time
        if best_output is None:
            best_output = prediction.detach().squeeze().cpu().numpy()

        residual = noisy - best_output
        history = tracker.to_dict()
        history["config"] = asdict(self.config)
        history["phase_history"] = phase_history

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