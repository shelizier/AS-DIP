"""Benchmark reporting and experiment aggregation utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .naming import method_display_name
from .plotting import set_publication_style

def _safe_metric(metrics: Dict[str, Any], key: str) -> Optional[float]:
    value = metrics.get(key)
    return None if value is None else float(value)


def export_benchmark_summary(
    output_root: str | Path,
    experiment_name: str,
    dataset_type: str,
    results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Write benchmark summary tables in CSV and JSON."""
    output_root = Path(output_root)
    rows: List[Dict[str, Any]] = []
    for method, result in results.items():
        metrics = result.best_metrics or {}
        rows.append(
            {
                "experiment_name": experiment_name,
                "dataset_type": dataset_type,
                "method": method_display_name(method),
                "method_id": method,
                "best_iteration": int(result.best_iteration),
                "elapsed_seconds": float(result.elapsed_seconds),
                "snr": _safe_metric(metrics, "snr"),
                "snr_input": _safe_metric(metrics, "snr_input"),
                "snr_gain": _safe_metric(metrics, "snr_gain"),
                "residual_energy": _safe_metric(metrics, "residual_energy"),
            }
        )

    if rows:
        with (output_root / "benchmark_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with (output_root / "benchmark_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
    return rows


def plot_method_overview(
    noisy: np.ndarray,
    clean: Optional[np.ndarray],
    results: Dict[str, Any],
    save_path: str | Path,
) -> None:
    """Create an overview figure containing the input and all method outputs."""
    set_publication_style()
    method_names = list(results.keys())
    ncols = 2 + len(method_names)
    figure, axes = plt.subplots(2, ncols, figsize=(4.2 * ncols, 8.0), constrained_layout=True)

    def plot_panel(ax: plt.Axes, data: np.ndarray, title: str) -> None:
        vmax = float(np.max(np.abs(data))) if np.max(np.abs(data)) > 0 else 1.0
        image = ax.imshow(data, cmap="seismic", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time")
        colorbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Amplitude")

    reference = clean if clean is not None else noisy
    plot_panel(axes[0, 0], reference, "Original Data")
    plot_panel(axes[1, 0], np.zeros_like(reference), "Reference Residual")
    plot_panel(axes[0, 1], noisy, "Noisy Data")
    plot_panel(axes[1, 1], noisy - reference, "Input Residual")

    for column, method in enumerate(method_names, start=2):
        result = results[method]
        method_title = method_display_name(method)
        if result.best_metrics and "snr_gain" in result.best_metrics:
            method_title += f"\nSNR gain {result.best_metrics['snr_gain']:.2f} dB"
        plot_panel(axes[0, column], result.denoised, method_title)
        residual = (clean - result.denoised) if clean is not None else result.residual
        plot_panel(axes[1, column], residual, f"{method_title}\nResidual")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


def collect_experiment_rows(outputs_root: str | Path) -> List[Dict[str, Any]]:
    """Scan outputs and collect one summary row per method run."""
    outputs_root = Path(outputs_root)
    rows: List[Dict[str, Any]] = []
    for history_path in sorted(outputs_root.glob("*/*/history.json")):
        with history_path.open("r", encoding="utf-8") as handle:
            history = json.load(handle)
        method_dir = history_path.parent
        experiment_dir = method_dir.parent
        metrics = history.get("best_metrics", {})
        elapsed = history.get("elapsed_seconds", [])
        rows.append(
            {
                "experiment_name": experiment_dir.name,
                "method": method_display_name(method_dir.name),
                "method_id": method_dir.name,
                "best_iteration": history.get("best_iteration"),
                "elapsed_seconds": elapsed[-1] if elapsed else None,
                "snr": metrics.get("snr"),
                "snr_input": metrics.get("snr_input"),
                "snr_gain": metrics.get("snr_gain"),
                "residual_energy": metrics.get("residual_energy"),
            }
        )
    return rows


def export_aggregate_table(rows: Iterable[Dict[str, Any]], save_path: str | Path) -> None:
    """Write an aggregated summary CSV."""
    rows = list(rows)
    if not rows:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_aggregate_summary(rows: Iterable[Dict[str, Any]], save_path: str | Path) -> None:
    """Plot aggregate runtime and SNR gain comparisons."""
    rows = list(rows)
    if not rows:
        return
    set_publication_style()
    methods = sorted({row["method"] for row in rows})
    experiments = sorted({row["experiment_name"] for row in rows})
    x_positions = np.arange(len(experiments))
    width = 0.35

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for index, method in enumerate(methods):
        method_rows = {row["experiment_name"]: row for row in rows if row["method"] == method}
        elapsed_values = [method_rows.get(name, {}).get("elapsed_seconds", np.nan) for name in experiments]
        snr_gain_values = [method_rows.get(name, {}).get("snr_gain", np.nan) for name in experiments]
        offset = (index - (len(methods) - 1) / 2.0) * width
        axes[0].bar(x_positions + offset, elapsed_values, width=width, label=method)
        axes[1].bar(x_positions + offset, snr_gain_values, width=width, label=method)

    axes[0].set_title("Runtime Comparison")
    axes[0].set_ylabel("Time (s)")
    axes[1].set_title("SNR Gain Comparison")
    axes[1].set_ylabel("SNR Gain (dB)")
    for axis in axes:
        axis.set_xticks(x_positions)
        axis.set_xticklabels(experiments, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.3)
        axis.legend()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
