"""Publication-style plotting helpers for seismic denoising."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch  # 引入 torch 用于处理深度学习输出格式

from .naming import method_display_name

matplotlib.use("Agg")


def set_publication_style():
    """统一的 Matplotlib 论文风格设定"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })


def _to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """内部辅助函数：自动处理 Tensor 转换并去掉多余的 channel 维度"""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    return data.squeeze()


def plot_seismic_panels(
        original: Union[np.ndarray, torch.Tensor],
        noisy: Union[np.ndarray, torch.Tensor],
        denoised: Union[np.ndarray, torch.Tensor],
        residual: Union[np.ndarray, torch.Tensor],
        save_path: str | Path,
        clean: Optional[Union[np.ndarray, torch.Tensor]] = None,
        snr: Optional[float] = None
) -> None:
    """绘制地震剖面全流程对比图"""
    set_publication_style()

    original = _to_numpy(original)
    noisy = _to_numpy(noisy)
    denoised = _to_numpy(denoised)
    residual = _to_numpy(residual)
    if clean is not None:
        clean = _to_numpy(clean)

    reference = clean if clean is not None else original

    # 【回退】：重新使用动态 99.5% 分位数计算色标范围
    vmax = np.max([
        np.percentile(np.abs(noisy), 99.5),
        np.percentile(np.abs(reference), 99.5),
        np.percentile(np.abs(denoised), 99.5)
    ])
    vmin = -vmax

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

    # 【保留】：所有 imshow 继续保留 interpolation='bicubic'
    im0 = axes[0].imshow(reference, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax, interpolation='bicubic')
    axes[0].set_title('Original Data' if clean is None else 'Clean Data')
    axes[0].set_xlabel('Trace')
    axes[0].set_ylabel('Time Sample')
    fig.colorbar(im0, ax=axes[0], label='Amplitude')

    im1 = axes[1].imshow(noisy, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax, interpolation='bicubic')
    axes[1].set_title('Noisy Data')
    axes[1].set_xlabel('Trace')
    fig.colorbar(im1, ax=axes[1], label='Amplitude')

    title_denoised = 'Denoised Result'
    if snr is not None:
        title_denoised += f'\n(SNR: {snr:.2f} dB)'
    im2 = axes[2].imshow(denoised, aspect='auto', cmap='seismic', vmin=vmin, vmax=vmax, interpolation='bicubic')
    axes[2].set_title(title_denoised)
    axes[2].set_xlabel('Trace')
    fig.colorbar(im2, ax=axes[2], label='Amplitude')

    # 残差同样使用动态计算
    res_max = float(np.percentile(np.abs(residual), 99.9))
    im3 = axes[3].imshow(residual, aspect='auto', cmap='seismic', vmin=-res_max, vmax=res_max, interpolation='bicubic')
    axes[3].set_title('Residual (Clean - Denoised)' if clean is not None else 'Residual Difference')
    axes[3].set_xlabel('Trace')
    fig.colorbar(im3, ax=axes[3], label='Amplitude')

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_benchmark_curves(results: Dict[str, Dict[str, list[float]]], save_path: str | Path) -> None:
    """Plot SNR and loss benchmark curves for multiple methods (Publication Optimized)."""
    set_publication_style()

    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    line_styles = ['-', '--', '-.', ':']

    for i, ((method, history), color) in enumerate(zip(results.items(), colors)):
        label = method_display_name(method)
        ls = line_styles[i % len(line_styles)]

        # --- 左图：SNR 曲线 ---
        if history.get("elapsed_seconds") and history.get("snr"):
            times = history["elapsed_seconds"][: len(history["snr"])]
            snrs = history["snr"]
            axes[0].plot(times, snrs, label=label, color=color, linestyle=ls, linewidth=2.0)

            # 【修改】：去掉了黑边，尺寸稍微调精巧一些，使用纯同色五角星
            if len(snrs) > 0:
                best_idx = np.argmax(snrs)
                axes[0].scatter(times[best_idx], snrs[best_idx],
                                color=color, marker='*', s=200, zorder=5)

        # --- 右图：Loss 曲线 ---
        if history.get("iterations") and history.get("total_loss"):
            axes[1].plot(history["iterations"], history["total_loss"],
                         label=label, color=color, linestyle=ls, linewidth=2.0, alpha=0.85)

    # --- 左图格式优化 (Convergence Speed) ---
    axes[0].set_xlabel("Time (s)", fontweight='bold')
    axes[0].set_ylabel("SNR (dB)", fontweight='bold')
    axes[0].set_title("Convergence Speed", pad=15)
    axes[0].set_xlim(left=0)  # 强制时间从 0 开始
    axes[0].grid(True, linestyle='--', alpha=0.4)
    axes[0].legend(loc='lower right', framealpha=0.95, edgecolor='black')

    # --- 右图格式优化 (Optimization History) ---
    axes[1].set_xlabel("Iteration", fontweight='bold')
    axes[1].set_ylabel("Loss", fontweight='bold')
    axes[1].set_title("Optimization History", pad=15)
    axes[1].set_xlim(left=0)  # 强制迭代次数从 0 开始
    axes[1].set_yscale("log")

    # 次级网格保留，提升专业感
    axes[1].grid(True, which="major", linestyle='--', alpha=0.4)
    axes[1].grid(True, which="minor", linestyle=':', alpha=0.2)

    axes[1].legend(loc='upper right', framealpha=0.95, edgecolor='black')

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)