"""Evaluation metrics for seismic denoising experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


def _to_float64(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float64)


def compute_snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Compute SNR of an estimate relative to a clean reference."""
    reference = _to_float64(reference)
    estimate = _to_float64(estimate)
    noise = estimate - reference
    numerator = np.sum(reference ** 2)
    denominator = np.sum(noise ** 2) + 1e-12
    return float(10.0 * np.log10(numerator / denominator))


def compute_psnr(reference: np.ndarray, estimate: np.ndarray, data_range: Optional[float] = None) -> float:
    """Compute PSNR."""
    reference = _to_float64(reference)
    estimate = _to_float64(estimate)
    mse = np.mean((estimate - reference) ** 2)
    if mse <= 1e-12:
        return float("inf")
    if data_range is None:
        data_range = float(np.max(reference) - np.min(reference))
        data_range = data_range if data_range > 0 else 1.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def compute_ssim(reference: np.ndarray, estimate: np.ndarray, data_range: Optional[float] = None) -> float:
    """Compute a global SSIM approximation for 2D seismic sections."""
    reference = _to_float64(reference)
    estimate = _to_float64(estimate)
    if data_range is None:
        data_range = float(max(reference.max(), estimate.max()) - min(reference.min(), estimate.min()))
        data_range = data_range if data_range > 0 else 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mean_ref = reference.mean()
    mean_est = estimate.mean()
    var_ref = reference.var()
    var_est = estimate.var()
    covariance = np.mean((reference - mean_ref) * (estimate - mean_est))
    numerator = (2 * mean_ref * mean_est + c1) * (2 * covariance + c2)
    denominator = (mean_ref ** 2 + mean_est ** 2 + c1) * (var_ref + var_est + c2)
    return float(numerator / (denominator + 1e-12))


def compute_metrics(output_np: np.ndarray, noisy_np: np.ndarray, clean_np: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute available denoising metrics."""
    metrics: Dict[str, float] = {
        "residual_energy": float(np.mean((noisy_np - output_np) ** 2)),
    }
    if clean_np is not None:
        metrics["snr"] = compute_snr(clean_np, output_np)
        metrics["psnr"] = compute_psnr(clean_np, output_np)
        metrics["ssim"] = compute_ssim(clean_np, output_np)
        metrics["snr_input"] = compute_snr(clean_np, noisy_np)
        metrics["snr_gain"] = metrics["snr"] - metrics["snr_input"]
    return metrics


@dataclass
class MetricsTracker:
    """Track optimization history for one experiment."""

    iterations: list[int] = field(default_factory=list)
    elapsed_seconds: list[float] = field(default_factory=list)
    total_loss: list[float] = field(default_factory=list)
    mse_loss: list[float] = field(default_factory=list)
    tv_loss: list[float] = field(default_factory=list)
    snr: list[float] = field(default_factory=list)
    psnr: list[float] = field(default_factory=list)
    ssim: list[float] = field(default_factory=list)
    best_iteration: int = 0
    best_metrics: Dict[str, float] = field(default_factory=dict)

    def update(
        self,
        iteration: int,
        elapsed_seconds: float,
        loss_terms: Dict[str, float],
        metrics: Dict[str, float],
    ) -> None:
        self.iterations.append(iteration)
        self.elapsed_seconds.append(float(elapsed_seconds))
        self.total_loss.append(float(loss_terms["total"]))
        self.mse_loss.append(float(loss_terms["mse"]))
        self.tv_loss.append(float(loss_terms["tv"]))
        if "snr" in metrics:
            self.snr.append(float(metrics["snr"]))
        if "psnr" in metrics:
            self.psnr.append(float(metrics["psnr"]))
        if "ssim" in metrics:
            self.ssim.append(float(metrics["ssim"]))

    def to_dict(self) -> Dict[str, Any]:
        """Convert tracked metrics to a serializable dictionary."""
        return {
            "iterations": self.iterations,
            "elapsed_seconds": self.elapsed_seconds,
            "total_loss": self.total_loss,
            "mse_loss": self.mse_loss,
            "tv_loss": self.tv_loss,
            "snr": self.snr,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "best_iteration": self.best_iteration,
            "best_metrics": self.best_metrics,
        }
