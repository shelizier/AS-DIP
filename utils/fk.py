"""Basic f-k transform helpers for seismic analysis."""

from __future__ import annotations

import numpy as np


def fk_transform(section: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the complex f-k spectrum and its amplitude."""
    spectrum = np.fft.fftshift(np.fft.fft2(section))
    amplitude = np.abs(spectrum)
    return spectrum, amplitude


def fk_mask_filter(section: np.ndarray, keep_fraction: float = 0.6) -> np.ndarray:
    """Apply a simple centered f-k mask filter."""
    spectrum, _ = fk_transform(section)
    rows, cols = section.shape
    row_center, col_center = rows // 2, cols // 2
    row_radius = int(rows * keep_fraction * 0.5)
    col_radius = int(cols * keep_fraction * 0.5)
    mask = np.zeros_like(section, dtype=np.float32)
    mask[row_center - row_radius:row_center + row_radius, col_center - col_radius:col_center + col_radius] = 1.0
    filtered = spectrum * mask
    return np.real(np.fft.ifft2(np.fft.ifftshift(filtered))).astype(np.float32)
