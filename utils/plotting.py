"""Publication-style plotting helpers for seismic denoising."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .naming import method_display_name

matplotlib.use("Agg")


def set_publication_style() -> None:
    """Apply publication-oriented Matplotlib defaults."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 150,
        }
    )


def _plot_panel(ax: plt.Axes, data: np.ndarray, title: str, cmap: str, units: str = "Amplitude") -> None:
    vmax = float(np.max(np.abs(data))) if np.max(np.abs(data)) > 0 else 1.0
    image = ax.imshow(data, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Trace")
    ax.set_ylabel("Time")
    colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(units)


def plot_seismic_panels(
    original: np.ndarray,
    noisy: np.ndarray,
    denoised: np.ndarray,
    residual: np.ndarray,
    save_path: str | Path,
    clean: Optional[np.ndarray] = None,
) -> None:
    """Plot original, noisy, denoised, and residual seismic sections."""
    set_publication_style()
    figure, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    reference = clean if clean is not None else original
    _plot_panel(axes[0], reference, "Original Data", cmap="seismic")
    _plot_panel(axes[1], noisy, "Noisy Data", cmap="seismic")
    _plot_panel(axes[2], denoised, "Denoised Result", cmap="seismic")
    _plot_panel(axes[3], residual, "Residual Difference", cmap="seismic")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


def plot_benchmark_curves(results: Dict[str, Dict[str, list[float]]], save_path: str | Path) -> None:
    """Plot PSNR and loss benchmark curves for multiple methods."""
    set_publication_style()
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for method, history in results.items():
        label = method_display_name(method)
        if history.get("elapsed_seconds") and history.get("psnr"):
            axes[0].plot(history["elapsed_seconds"][: len(history["psnr"])], history["psnr"], label=label, linewidth=1.8)
        axes[1].plot(history["iterations"], history["total_loss"], label=label, linewidth=1.6)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("Convergence Speed")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Optimization History")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
